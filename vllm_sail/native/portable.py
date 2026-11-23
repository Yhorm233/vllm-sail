# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load-time weight packing using Torch operations, with no CUDA/CUTLASS code.

The permutation follows vLLM 0.30 marlin_utils_test.py's reference layout.
It is used once per expert during loading; inference remains in PPU DeepGEMM.
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=4)
def _weight_permutation(num_bits: int, is_a_8bit: bool) -> tuple[int, ...]:
    if num_bits not in (4, 8):
        raise ValueError("Marlin repacking supports 4 or 8 bits")
    permutation = []
    for thread in range(32):
        col = thread // 4
        rows = (
            [4 * (thread % 4) + i for i in range(4)]
            + [4 * (thread % 4 + 4) + i for i in range(4)]
            if is_a_8bit
            else [
                2 * (thread % 4),
                2 * (thread % 4) + 1,
                2 * (thread % 4 + 4),
                2 * (thread % 4 + 4) + 1,
            ]
        )
        tile = [16 * row + col + 8 * block for block in (0, 1) for row in rows]
        for offset in range(2 if is_a_8bit else 4):
            permutation.extend(p + (512 if is_a_8bit else 256) * offset for p in tile)
    if num_bits == 4:
        interleave = (0, 4, 1, 5, 2, 6, 3, 7) if is_a_8bit else (0, 2, 4, 6, 1, 3, 5, 7)
    else:
        interleave = (0, 1, 2, 3) if is_a_8bit else (0, 2, 1, 3)
    width = len(interleave)
    return tuple(
        permutation[start + i]
        for start in range(0, len(permutation), width)
        for i in interleave
    )


def gptq_marlin_repack(b_q_weight, size_k: int, size_n: int, num_bits: int):
    import torch

    if num_bits not in (4, 8):
        raise ValueError("Marlin repacking supports 4 or 8 bits")
    pack = 32 // num_bits
    tile_k = 16
    if size_k % tile_k or size_n % 64:
        raise ValueError(
            "Marlin repacking requires aligned K tiles and N divisible by 64"
        )
    if b_q_weight.dtype != torch.int32 or tuple(b_q_weight.shape) != (
        size_k // pack,
        size_n,
    ):
        raise ValueError("Expected GPTQ int32 weights of shape [K / pack_factor, N]")
    shifts = torch.arange(pack, device=b_q_weight.device, dtype=torch.int32) * num_bits
    unpacked = (
        (b_q_weight[:, None, :] >> shifts[None, :, None]) & ((1 << num_bits) - 1)
    ).reshape(size_k, size_n)
    tiled = unpacked.reshape(size_k // tile_k, tile_k, size_n // 16, 16)
    tiled = tiled.permute(0, 2, 1, 3).reshape(size_k // 16, size_n * 16)
    order = torch.tensor(_weight_permutation(num_bits, False), device=b_q_weight.device)
    tiled = (
        tiled.reshape(-1, order.numel())
        .index_select(1, order)
        .reshape(size_k // 16, size_n * 16)
    )
    packed = torch.zeros(
        (size_k // 16, size_n * 16 // pack), device=b_q_weight.device, dtype=torch.int32
    )
    for i in range(pack):
        packed.bitwise_or_(tiled[:, i::pack] << (num_bits * i))
    return packed.contiguous()


def per_token_group_fp8_quant_ppu_opt(*args):
    from vllm_sail.model_executor.layers.quantization.utils.group_quant_kernel import (
        per_token_group_fp8_quant_ppu_opt as quantize,
    )

    return quantize(*args)


_libraries = []
_installed = False


def install() -> None:
    """Claim only operations still absent after loading native extensions."""
    global _installed
    if _installed:
        return
    import torch

    if not hasattr(torch.ops._C, "gptq_marlin_repack"):
        library = torch.library.Library("_C", "FRAGMENT")
        library.define(
            "gptq_marlin_repack(Tensor b_q_weight, SymInt size_k, SymInt size_n, int num_bits) -> Tensor"
        )
        library.impl("gptq_marlin_repack", gptq_marlin_repack, "CUDA")
        _libraries.append(library)
    if not hasattr(torch.ops._C, "per_token_group_fp8_quant_ppu_opt"):
        library = torch.library.Library("_C", "FRAGMENT")
        library.define(
            "per_token_group_fp8_quant_ppu_opt(Tensor input, Tensor(a!) output_q, Tensor(b!) output_s, int group_size, float eps, float min_8bit, float max_8bit, bool scale_ue8m0, bool column_major_scales, bool tma_aligned_scales) -> ()"
        )
        library.impl(
            "per_token_group_fp8_quant_ppu_opt",
            per_token_group_fp8_quant_ppu_opt,
            "CUDA",
        )
        library.impl("per_token_group_fp8_quant_ppu_opt", lambda *args: None, "Meta")
        _libraries.append(library)
    _installed = True
