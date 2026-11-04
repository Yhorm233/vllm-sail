# SPDX-License-Identifier: Apache-2.0
"""Real PPU lookup checks for per-row FP32 Engram scales, including UVA."""

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.ppu


@pytest.mark.parametrize("cpu_offload", [False, True])
def test_channelwise_engram_lookup_preserves_fp32_scales(cpu_offload):
    import torch
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    import vllm_sail

    vllm_sail.register_out_of_tree()
    from vllm.models.deepseek_v41.common.engram import ParallelEngramEmbedding

    placement = (
        {"device": "cpu", "pin_memory": True} if cpu_offload else {"device": "cuda"}
    )
    weight = torch.empty(8, 256, dtype=torch.float8_e4m3fn, **placement)
    weight.fill_(2.0)
    scales = torch.empty(8, 1, dtype=torch.float32, **placement)
    scales.copy_(torch.arange(1, 9, dtype=torch.float32).view(8, 1) * 0.3)
    # Keep host allocations alive while the kernel reads their UVA aliases.
    views = (weight, scales)
    if cpu_offload:
        views = tuple(get_accelerator_view_from_cpu_tensor(t) for t in views)
    ids = torch.tensor(
        [[0, 4, 8, 999], [0, 4, 11, -1]], dtype=torch.int64, device="cuda"
    )
    out = torch.empty(2, 3, 256, dtype=torch.bfloat16, device="cuda")
    # Use the installed model entry point to select 810E software or native FP8 decoding.
    owner = SimpleNamespace(
        weight=weight,
        _storage=lambda: views,
        vocab_start_idx=8,
        vocab_end_idx=16,
        head_start=2,
        part_n_hash_cols=3,
        n_hash_cols=4,
        dim=256,
        block_size=256,
        _num_sms=torch.cuda.get_device_properties(0).multi_processor_count,
    )
    ParallelEngramEmbedding.lookup(owner, ids, out)
    torch.cuda.synchronize()
    expected = torch.zeros(2, 3, 256, dtype=torch.bfloat16)
    expected[0, 0] = (2.0 * scales[0, 0].cpu()).to(torch.bfloat16)
    expected[1, 0] = (2.0 * scales[3, 0].cpu()).to(torch.bfloat16)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
