# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PPU fused MoE kernels from the 0.27 fork; selected only by PPU launchers."""

import torch
from vllm.model_executor.layers.fused_moe.fused_moe import write_zeros_to_output
from vllm.triton_utils import tl, triton


@triton.jit
def _moe_sum_reduce_kernel(
    input_ptr,
    output_ptr,
    M,
    T: tl.constexpr,
    K: tl.constexpr,
    stride_im,
    stride_it,
    stride_ik,
    stride_om,
    stride_ok,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    num_stages: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    rm = rm.to(tl.int64)
    rk = rk.to(tl.int64)
    mask = (rm[:, None] < M) & (rk[None, :] < K)
    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    for t in tl.range(T, num_stages=num_stages):
        curr_input_ptr = (
            input_ptr
            + rm[:, None] * stride_im
            + t * stride_it
            + rk[None, :] * stride_ik
        )
        tile = tl.load(curr_input_ptr, mask=mask, other=0.0)
        acc += tile
    output_tile_ptr = output_ptr + rm[:, None] * stride_om + rk[None, :] * stride_ok

    tl.store(output_tile_ptr, acc, mask=mask)


def moe_sum_reduce_triton(x: torch.Tensor, output: torch.Tensor):
    M, T, K = x.shape
    assert x.is_contiguous()
    assert output.is_contiguous()
    assert output.shape[0] == M and output.shape[1] == K

    BLOCK_M = 32
    BLOCK_K = 64

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_K))
    # BLOCK_M: 32, BLOCK_K: 32, num_warps: 8, num_ctas: 1, num_stages: 4, maxnreg: None;
    _moe_sum_reduce_kernel[grid](
        x,
        output,
        M,
        T,
        K,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        output.stride(0),
        output.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
        num_warps=8,
        num_stages=1,
    )


@triton.jit
def valu_dot(a, b, Tile_M: tl.constexpr, Tile_N: tl.constexpr, Tile_K: tl.constexpr):
    b = b.trans()
    a = a.reshape(Tile_M, 1, Tile_K)
    b = b.reshape(1, Tile_N, Tile_K)
    c = a * b
    return tl.sum(c, 2)


@triton.jit
def fused_moe_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    b_bias_ptr,
    a_scale_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bse,
    stride_bsk,
    stride_bsn,
    stride_bbe,  # bias expert stride
    stride_bbn,  # bias N stride
    # Block size for block-wise quantization
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    naive_block_assignment: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    per_channel_quant: tl.constexpr,
    per_out_ch_quant: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SWAP_AB: tl.constexpr,
    use_valu: tl.constexpr,
    even_Ks: tl.constexpr,
    # Tensor-descriptor path for the A gather and B load in the K-loop.
    USE_TD: tl.constexpr = False,
    num_stages: tl.constexpr = 3,
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.
    - naive_block_assignment: A boolean flag indicating whether to use naive
        token wise block assignment. If True, each block corresponds to a
        single token.
    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    if GROUP_SIZE_M > 1:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    offs = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    if not naive_block_assignment:
        offs_token_id = pid_m * BLOCK_SIZE_M + offs
        offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    else:
        offs_token = tl.where(
            offs == 0,
            pid_m,  # first element = pid_m
            num_valid_tokens,  # remaining elements = constant
        )
    # Cast to int64 to prevent overflow in stride*offset products
    # (e.g. stride_cm * offs_token can exceed int32 for large token counts)
    offs_token = offs_token.to(tl.int64)

    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        # -----------------------------------------------------------
        # Write back zeros to the output when the expert is not
        # in the current expert parallel rank.
        write_zeros_to_output(
            c_ptr,
            stride_cm,
            stride_cn,
            pid_n,
            N,
            offs_token,
            token_mask,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            compute_type,
        )
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    # TD gather and the SWAP_AB accumulator layout are mutually exclusive.
    tl.static_assert(not (USE_TD and SWAP_AB))
    if USE_TD:
        # ``tt.descriptor_gather`` requires block_shape[0] == 1 and i32 idx.
        m_td = num_valid_tokens // top_k
        a_desc = tl.make_tensor_descriptor(
            base=a_ptr,
            shape=(m_td, K),
            strides=(stride_am, stride_ak),
            block_shape=(1, BLOCK_SIZE_K),
        )
        b_desc = tl.make_tensor_descriptor(
            base=b_ptr + off_experts * stride_be,
            shape=(N, K),
            strides=(stride_bn, stride_bk),
            block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_K),
        )
        gather_idx = (offs_token // top_k).to(tl.int32)
    elif SWAP_AB:
        a_ptrs = a_ptr + (
            offs_k[:, None] * stride_ak + offs_token[None, :] // top_k * stride_am
        )
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + (offs_bn[:, None] * stride_bn + offs_k[None, :] * stride_bk)
        )
    else:
        a_ptrs = a_ptr + (
            offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
        )
        b_block_ptr = tl.make_block_ptr(
            base=b_ptr + off_experts * stride_be,
            shape=(K, N),
            strides=(stride_bk, stride_bn),
            offsets=(0, (pid_n * BLOCK_SIZE_N).to(tl.int32)),
            block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
            order=(0, 1),
        )
    if use_int8_w8a16:
        b_scale_ptrs = (
            b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
        )
        b_scale = tl.load(b_scale_ptrs)

    if use_fp8_w8a8 or use_int8_w8a8:
        # block-wise
        if group_k > 0 and group_n > 0:
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            offs_bsn = offs_bn // group_n
            b_scale_ptrs = (
                b_scale_ptr + off_experts * stride_bse + offs_bsn * stride_bsn
            )
        else:
            if per_out_ch_quant:
                b_scale_ptrs = (
                    b_scale_ptr
                    + off_experts * stride_bse
                    + offs_bn[None, :] * stride_bsn
                )
                b_scale = tl.load(b_scale_ptrs)
            else:
                b_scale = tl.load(b_scale_ptr + off_experts)
            if per_channel_quant:
                a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
                a_scale = tl.load(a_scale_ptrs, mask=token_mask, other=0.0)[:, None]
            else:
                a_scale = tl.load(a_scale_ptr)
    if HAS_BIAS:
        # bias shape: [num_experts, N]
        bias_ptrs = b_bias_ptr + off_experts * stride_bbe + offs_bn * stride_bbn
        bias = tl.load(bias_ptrs, mask=(offs_bn < N), other=0.0)
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 or int32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    if SWAP_AB:
        accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    else:
        if use_int8_w8a8:
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
        else:
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K), num_stages=num_stages):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.
        if USE_TD:
            a = a_desc.gather(gather_idx, k * BLOCK_SIZE_K)
            b = b_desc.load([pid_n * BLOCK_SIZE_N, k * BLOCK_SIZE_K]).T
        elif SWAP_AB:
            a_mask = (offs_k[:, None] < K - k * BLOCK_SIZE_K) & token_mask[None, :]
            b_mask = offs_k[None, :] < K - k * BLOCK_SIZE_K
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        else:
            a_mask = token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            if even_Ks:
                b = tl.aiu_load(
                    b_block_ptr,
                )
            else:
                b = tl.load(b_block_ptr, boundary_check=(0, 1), padding_option="zero")
        # We accumulate along the K dimension.
        if use_int8_w8a16:
            accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
        elif use_fp8_w8a8 or use_int8_w8a8:
            if group_k > 0 and group_n > 0:
                k_start = k * BLOCK_SIZE_K
                offs_ks = k_start // group_k
                a_scale = tl.load(
                    a_scale_ptrs + offs_ks * stride_ask, mask=token_mask, other=0.0
                )
                b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)
                if SWAP_AB:
                    accumulator += tl.dot(b, a) * b_scale[:, None] * a_scale[None, :]
                else:
                    accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
            else:
                if use_fp8_w8a8:
                    # acc used to enable fp8_fast_accum
                    if SWAP_AB:
                        accumulator = tl.dot(b, a, acc=accumulator)
                    else:
                        accumulator = tl.dot(a, b, acc=accumulator)
                else:
                    accumulator += tl.dot(a, b)
        elif use_valu:
            accumulator += valu_dot(a, b, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)
        else:
            accumulator += tl.dot(a, b)
        if not USE_TD:
            # Advance the ptrs to the next K block.
            a_ptrs += BLOCK_SIZE_K * stride_ak
            if SWAP_AB:
                b_ptrs += BLOCK_SIZE_K * stride_bk
            else:
                b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K, 0))

    if SWAP_AB:
        accumulator = tl.trans(accumulator, (1, 0))

    # Dequantization for supported quantization schemes:
    #   - int8_w8a16
    #   - fp8_w8a8
    #   - int8_w8a8
    # Accumulator and scalings are in float32 to preserve numerical accuracy.
    if use_int8_w8a16:
        accumulator = accumulator * b_scale
    elif (use_fp8_w8a8 or use_int8_w8a8) and not (group_k > 0 and group_n > 0):
        accumulator = accumulator * a_scale * b_scale

    # Bias addition:
    # Bias must be applied after dequantization:
    #   - Since bias is typically not quantized
    #   - Bias should not be scaled by quantization factors
    if HAS_BIAS:
        accumulator += bias[None, :]

    # Router (MoE) weight multiplication:
    # This multiplication MUST be performed in float32 before any precision
    # conversion to ensure numerical stability, which is especially critical
    # on ROCm platforms.
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(
            topk_weights_ptr + offs_token,
            mask=token_mask,
            other=0,
        )
        accumulator *= moe_weight[:, None]

    # Final precision conversion:
    # Cast once at the end to the desired compute/output dtype.
    accumulator = accumulator.to(compute_type)

    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)
