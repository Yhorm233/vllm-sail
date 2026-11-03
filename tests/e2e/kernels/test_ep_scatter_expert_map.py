# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for the ``expert_map`` lane bug in the DeepEP scatter kernel.

``_fwd_kernel_ep_scatter_2_optimal`` maps global expert ids to local ones with
a vector masked load::

    expert_loc = tl.load(expert_map + expert_loc, mask=(expert_loc != -1))

Masked lanes are left undefined unless ``other`` is given, and the very next
statement reads *every* lane::

    tt_mask = expert_mask & (expert_loc >= 0)

A lane that was skipped because ``recv_topk[token, k] == -1`` (a padding slot)
therefore reaches the ``>= 0`` test holding an arbitrary value.  If it comes
back non-negative the token is treated as locally routed, so
``tl.atomic_add(expert_start_loc + expert_loc, ...)`` indexes a tensor with only
``local_num_experts`` entries out of bounds and the token is scattered to a
garbage row of the permuted activation buffer -- overwriting a real token's
data.  Only ``--enable-expert-parallel`` reaches this path, since
``HAS_EXPERT_MAP`` is false otherwise.

Lanes beyond ``topk_num`` are *not* a trigger: the first load already passes
``other=-1`` and ``expert_mask`` is false for them, so ``tt_mask`` excludes them
regardless of what the second load left behind.  A non-local expert is not a
trigger either -- it has a valid global id, so the mask is true and the load
returns ``-1`` correctly.  The padding slot is the one gap, which is why every
case below carries ``-1`` entries inside ``topk_num``.

Why the assertions are structural rather than a value golden: destination rows
come from ``tl.atomic_add`` across one Triton program per token, so which row a
token lands in is not reproducible between runs.  What *is* reproducible is the
partition -- each expert's region must receive exactly its own tokens, once
each, with the right payload, and nothing outside those regions may be touched.
Those invariants are asserted here, and they are precisely what the bug breaks.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("requires a real PPU/CUDA device", allow_module_level=True)
pytest.importorskip("vllm")

import vllm_sail  # noqa: E402

vllm_sail.register_out_of_tree()

from vllm.platforms import current_platform  # noqa: E402

if not current_platform.is_ppu():
    pytest.skip("requires a PPU platform (is_ppu() is False)", allow_module_level=True)

from vllm_sail.model_executor.layers.fused_moe.deep_gemm_utils import (  # noqa: E402
    deepgemm_moe_permute,
)

pytestmark = [pytest.mark.ppu]

NUM_GLOBAL_EXPERTS = 16
LOCAL_NUM_EXPERTS = 8
TOP_K = 8
HIDDEN = 512
# Re-run each case with fresh buffers. A single run only samples the undefined
# lanes once; repeating raises the chance of observing a non-negative value in
# a masked lane on a build that still has the bug.
REPEATS = 8

# Even globals are local, odd globals are not, so every token mixes mapped and
# non-local slots and the -1 return path of expert_map is exercised throughout.
EXPERT_MAP = [i // 2 if i % 2 == 0 else -1 for i in range(NUM_GLOBAL_EXPERTS)]

# (quant, is_block_wise, is_channel_wise, sentinel)
QUANT_CASES = [
    pytest.param("bf16", False, False, -7.5, id="bf16-noscale"),
    pytest.param("int8", False, True, -128, id="int8-channelwise"),
]

# (num_real_tokens, num_full_pad_rows, num_partial_pad_tokens) - a control case
# with no padding at all, the cudagraph shape where whole trailing rows are -1,
# and the DeepEP shape where individual slots within a real row are -1.
PAD_CASES = [
    pytest.param(64, 0, 0, id="pad-none"),
    pytest.param(64, 6, 0, id="pad-6-full-rows"),
    pytest.param(64, 0, 16, id="pad-16-partial"),
    pytest.param(256, 24, 32, id="pad-mixed-large"),
]


def _make_routing(
    num_real: int,
    num_full_pad: int,
    num_partial_pad: int,
    seed: int,
) -> torch.Tensor:
    """Build ``topk_ids`` of shape ``[num_real + num_full_pad, TOP_K]`` on CPU.

    Real rows carry ``TOP_K`` distinct global expert ids. The first
    ``num_partial_pad`` real rows have their trailing slots overwritten with
    ``-1``, and ``num_full_pad`` all-``-1`` rows are appended last, matching how
    cudagraph capture pads a batch up to a captured size.
    """
    gen = torch.Generator().manual_seed(seed)
    rows = []
    for i in range(num_real):
        ids = torch.randperm(NUM_GLOBAL_EXPERTS, generator=gen)[:TOP_K].to(torch.int32)
        if i < num_partial_pad:
            # Leave at least one routable slot so the row is not degenerate.
            blank = int(torch.randint(1, TOP_K, (1,), generator=gen).item())
            ids[TOP_K - blank :] = -1
        rows.append(ids)
    for _ in range(num_full_pad):
        rows.append(torch.full((TOP_K,), -1, dtype=torch.int32))
    return torch.stack(rows)


def _oracle(topk_ids: torch.Tensor) -> tuple[list[int], list[tuple[int, int, int]]]:
    """Independent expected routing: per-local-expert counts and (t, k, local).

    Deliberately a plain Python loop over the int32 CPU tensor rather than a
    vectorised expression, so it shares no code with either the kernel or
    ``count_expert_num_tokens`` and can disagree with both.
    """
    counts = [0] * LOCAL_NUM_EXPERTS
    routed: list[tuple[int, int, int]] = []
    for t in range(topk_ids.shape[0]):
        for k in range(topk_ids.shape[1]):
            global_id = int(topk_ids[t, k])
            if global_id < 0:
                continue
            local_id = EXPERT_MAP[global_id]
            if local_id < 0:
                continue
            counts[local_id] += 1
            routed.append((t, k, local_id))
    return counts, routed


@pytest.mark.parametrize("num_real,num_full_pad,num_partial_pad", PAD_CASES)
@pytest.mark.parametrize("quant,is_block_wise,is_channel_wise,sentinel", QUANT_CASES)
def test_ep_scatter_expert_map_padding(
    quant: str,
    is_block_wise: bool,
    is_channel_wise: bool,
    sentinel: float,
    num_real: int,
    num_full_pad: int,
    num_partial_pad: int,
) -> None:
    """Padding slots must not be scattered, whatever the masked lanes hold."""
    device = torch.device("cuda")
    topk_ids = _make_routing(num_real, num_full_pad, num_partial_pad, seed=1234)
    num_tokens = topk_ids.shape[0]
    expected_counts, routed = _oracle(topk_ids)
    total_routed = sum(expected_counts)
    assert total_routed > 0, "case must route at least one token"
    assert total_routed < num_tokens * TOP_K, "case must leave unrouted slots"

    topk_ids = topk_ids.to(device)
    expert_map = torch.tensor(EXPERT_MAP, dtype=torch.int32, device=device)
    if quant == "bf16":
        aq = torch.randn(num_tokens, HIDDEN, device=device, dtype=torch.float32)
        aq = aq.to(torch.bfloat16)
        aq_scale = None
    else:
        aq = torch.randint(-8, 8, (num_tokens, HIDDEN), device=device, dtype=torch.int8)
        aq_scale = torch.rand(num_tokens, 1, device=device, dtype=torch.float32)

    common = {
        "local_num_experts": LOCAL_NUM_EXPERTS,
        "expert_map": expert_map,
        "expert_tokens_meta": None,
        "is_block_wise_quant": is_block_wise,
        "is_channel_wise_quant": is_channel_wise,
    }

    # Discover the production row budget instead of recomputing it, so the test
    # tracks any change to compute_aligned_M rather than silently drifting.
    probe = deepgemm_moe_permute(aq=aq, aq_scale=aq_scale, topk_ids=topk_ids, **common)
    m_sum = probe[0].shape[0]
    assert m_sum > total_routed, "case needs untouched rows to check for corruption"
    del probe

    routed_t = torch.tensor([r[0] for r in routed], dtype=torch.long, device=device)
    routed_k = torch.tensor([r[1] for r in routed], dtype=torch.long, device=device)
    routed_local = torch.tensor(
        [r[2] for r in routed], dtype=torch.int32, device=device
    )
    expected_m_indices = torch.repeat_interleave(
        torch.arange(LOCAL_NUM_EXPERTS, dtype=torch.int32, device=device),
        torch.tensor(expected_counts, dtype=torch.long, device=device),
    )

    for _ in range(REPEATS):
        aq_out = torch.full((m_sum, HIDDEN), sentinel, device=device, dtype=aq.dtype)
        aq_out_ref, aq_scale_out, expert_ids, inv_perm, expert_num_tokens = (
            deepgemm_moe_permute(
                aq=aq, aq_scale=aq_scale, topk_ids=topk_ids, aq_out=aq_out, **common
            )
        )
        assert aq_out_ref is aq_out, "aq_out= must be honoured, not reallocated"

        # The upstream counter and the kernel must agree with the oracle about
        # how many tokens each local expert owns.
        assert expert_num_tokens.tolist() == expected_counts

        # Kernel 1 laid out the per-expert regions and left the aligned tail
        # marked invalid for DeepGEMM to skip.
        assert torch.equal(expert_ids[:total_routed], expected_m_indices)
        assert torch.all(expert_ids[total_routed:] == -1)

        # Every routed slot claims a distinct row, and the claims cover the
        # routed region exactly: no duplicates, no gaps, nothing out of range.
        # A garbage destination from an undefined lane breaks this immediately.
        dest = inv_perm[routed_t, routed_k].to(torch.long)
        assert torch.equal(
            dest.sort().values,
            torch.arange(total_routed, dtype=torch.long, device=device),
        )

        # Each token landed in the region of the expert it actually routed to.
        assert torch.equal(expert_ids[dest], routed_local)

        # The payload is a pure copy, so it must match bit for bit.
        assert torch.equal(aq_out[dest], aq[routed_t])
        if aq_scale is not None:
            assert torch.equal(aq_scale_out[dest], aq_scale[routed_t])

        # Nothing was scattered past the claimed rows. This is the assertion
        # that catches the reported symptom directly: a real token's row being
        # overwritten by a token that should have been skipped.
        assert torch.all(aq_out[total_routed:] == sentinel)
