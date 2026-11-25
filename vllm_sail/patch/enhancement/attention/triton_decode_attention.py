# SPDX-License-Identifier: Apache-2.0
"""PPU patch for ``vllm.v1.attention.ops.triton_decode_attention``.

The fork changes ``_decode_grouped_att_m_fwd``:

* Adds ``is_ppu_`` / ``is_btv_1_1`` module-level platform flags and a
  ``BLOCK = 16`` override for ``is_btv_1_1 and Lk >= 576`` (plus a duplicated
  hip variant of an existing workaround). NOTE: ``BLOCK`` is reassigned to 32
  further down in the same function, so the fork's override has no observable
  effect there — reproduced as-is and flagged in the Phase-4 report.
* Removes the ``not is_hip_`` guards from the MLA tile-size branches, so ROCm
  now takes the ``Lk == 576`` / ``Lk == 288`` tile layouts too. This is an
  UNCONDITIONAL upstream behaviour change (it alters stock ROCm), reproduced
  as in the fork.

Deviation from the fork: the module-level ``is_ppu_`` / ``is_btv_1_1`` flags
are resolved inside the patched function body instead, per the plugin's lazy
platform-detection rule; ``is_hip_`` keeps its upstream module-level meaning
(re-resolved here the same way).
"""

from __future__ import annotations

from vllm.platforms import current_platform
from vllm.triton_utils import triton
from vllm.v1.attention.ops.triton_decode_attention import (
    _fwd_grouped_kernel_stage1,
    _page_stride,
)

from vllm_sail.patch.utils import patch

_AFFECTED = ">=0.32.0,<0.33.0"
_MODULE = "vllm.v1.attention.ops.triton_decode_attention"


@patch(
    _MODULE,
    "_decode_grouped_att_m_fwd",
    reason=(
        "PPU 1.0 (capability 8.0) needs BLOCK=16 for Lk>=576 grouped "
        "decode attention on that architecture, and the fork removes the "
        "'not is_hip_' guards from the MLA tile branches — the latter is an "
        "unconditional upstream behaviour change (also affects ROCm), kept "
        "as-in-fork. Module-level platform flags from the fork are resolved "
        "lazily inside the body per plugin rules."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "upstream adopts the PPU block-size workaround and the ROCm tile-layout "
        "change, or the fork drops its architecture-specific tile edits."
    ),
)
def _decode_grouped_att_m_fwd(
    q,
    k_buffer,
    v_buffer,
    att_out,
    Req_to_tokens,
    B_Seqlen,
    num_kv_splits,
    sm_scale,
    page_size,
    logit_cap,
    k_scale,
    v_scale,
    is_mla=False,
    sliding_window=0,
):
    # with is_mla there is only a single c_kv in smem.
    # could increase BLOCK or num_stages.
    Lk = k_buffer.shape[-1]
    Lv = v_buffer.shape[-1]

    # PPU MODIFICATION: begin
    # Fork's module-level flags, resolved lazily here.
    is_hip_ = current_platform.is_rocm()
    is_btv_1_1 = current_platform.is_ppu() and current_platform.is_device_capability(
        (8, 0)
    )

    # [TODO] work around shmem limit on MI3xx
    if is_hip_ and Lk >= 576:
        BLOCK = 16

    if is_btv_1_1 and Lk >= 576:
        BLOCK = 16
    # PPU MODIFICATION: end

    # Align tile dimensions with latent rank for MLA to avoid shape mismatch.
    if is_mla:
        # PPU MODIFICATION: begin
        # Fork drops the `not is_hip_` guards (unconditional upstream change).
        if Lk == 576:
            # PPU MODIFICATION: end
            BLOCK_DMODEL = 512
            BLOCK_DPE = 64
        # PPU MODIFICATION: begin
        elif Lk == 288:
            # PPU MODIFICATION: end
            BLOCK_DMODEL = 256
            BLOCK_DPE = 32
        else:
            BLOCK_DMODEL = triton.next_power_of_2(Lv)
            BLOCK_DPE = triton.next_power_of_2(Lk - Lv) if Lk > Lv else 0
    else:
        BLOCK_DMODEL = triton.next_power_of_2(Lk)
        BLOCK_DPE = 0
    BLOCK_DV = triton.next_power_of_2(Lv)

    BLOCK = 32
    if is_hip_:
        BLOCK = 16

    batch, head_num = q.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k_buffer.shape[-2]

    BLOCK_H = 16
    NUM_KV_SPLITS = num_kv_splits
    grid = (
        batch,
        triton.cdiv(head_num, min(BLOCK_H, kv_group_num)),
        NUM_KV_SPLITS,
    )

    extra_kargs = {}
    num_stages = 2
    if is_hip_:
        # https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html#mi300x-triton-kernel-performance-optimization
        # https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
        extra_kargs = {"waves_per_eu": 1, "matrix_instr_nonkdim": 16, "kpack": 2}
        num_stages = 1
    elif not is_hip_ and BLOCK_DMODEL >= 1024:
        # Avoid shared memory overflow on NVIDIA when BLOCK_DMODEL is large
        # like non-MLA D_QK=576, BLOCK_DMODEL=1024, BLOCK_H=16
        # exceeds 101376 bytes limit
        num_stages = 1

    _fwd_grouped_kernel_stage1[grid](
        q,
        k_buffer,
        v_buffer,
        sm_scale,
        Req_to_tokens,
        B_Seqlen,
        att_out,
        Req_to_tokens.stride(0),
        q.stride(0),
        q.stride(1),
        _page_stride(k_buffer, page_size),
        k_buffer.stride(-3),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        k_buffer.stride(-2),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        _page_stride(v_buffer, page_size),
        v_buffer.stride(-3),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        v_buffer.stride(-2),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        k_scale,
        v_scale,
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK,
        BLOCK_H=BLOCK_H,
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        PAGE_SIZE=page_size,
        logit_cap=logit_cap,
        num_warps=4,
        num_stages=num_stages,
        Lk=Lk,
        Lv=Lv,
        IS_MLA=is_mla,
        SLIDING_WINDOW=sliding_window,
        **extra_kargs,
    )
