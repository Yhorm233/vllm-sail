# SPDX-License-Identifier: Apache-2.0
"""Keep MORI buffer geometry and dispatch flags consistent with quantization."""

from types import SimpleNamespace as NS

import pytest

from tests.support.source import function

PATCH = "vllm_sail/patch/enhancement/deepep.py"
CASES = [
    (True, True, True, True, "fp4", 128, 1, False),
    (False, True, False, False, "fp8", 1, 4, True),
    (False, False, True, False, "fp8", 32, 4, True),
    (False, False, False, True, "fp8", 1, 4, True),
    (False, False, False, False, "bf16", 0, 0, False),
]


def check_dispatch(factory, case):
    fp4, token, block, tensor, dtype, scale_dim, scale_bytes, fp8 = case
    buffers, constructors = [], []
    handle, expected = object(), object()
    moe = NS(
        moe_parallel_config=NS(use_passthrough_all2all=False, use_all2all_kernels=True),
        use_deepep_ht_kernels=False,
        use_deepep_ll_kernels=False,
        use_deepep_v2_kernels=False,
        use_moonep_kernels=False,
        use_mori_epv2_kernels=False,
        use_mori_epv1_kernels=True,
        hidden_dim=4096,
        in_dtype="bf16",
        max_num_tokens=256,
        num_experts=128,
        experts_per_token=8,
    )
    quant = NS(
        use_mxfp4_w4a4=fp4,
        is_per_act_token=token,
        is_block_quantized=block,
        is_per_tensor=tensor,
        quant_dtype="fp8",
    )

    def get_handle(arguments):
        buffers.append(arguments)
        return handle

    def prepare(*args, **kwargs):
        constructors.append((args, kwargs))
        return expected

    namespace = {
        "torch": NS(
            float4_e2m1fn_x2="fp4",
            float8_e8m0fnu=NS(itemsize=1),
            float32=NS(itemsize=4),
        ),
        "MoriPrepareAndFinalize": prepare,
    }
    manager = NS(rank=3, world_size=8, get_handle=get_handle)
    assert factory(namespace)(moe, quant, all2all_manager=manager) is expected
    assert buffers == [
        dict(
            rank=3,
            num_ep_ranks=8,
            quant_dtype=dtype,
            token_hidden_size=4096,
            scale_dim=scale_dim,
            scale_type_size=scale_bytes,
            max_num_tokens_per_dp_rank=256,
            input_dtype="bf16",
            num_local_experts=16,
            num_experts_per_token=8,
        )
    ]
    assert constructors == [
        (
            (handle,),
            dict(
                max_tokens_per_rank=256,
                num_dispatchers=8,
                use_fp8_dispatch=fp8,
                use_fp4_dispatch=fp4,
            ),
        )
    ]


@pytest.mark.parametrize("case", CASES)
def test_copied_factory_preserves_dispatch_buffer_contract(case):
    check_dispatch(
        lambda ns: function(PATCH, "_maybe_make_prepare_finalize_body", ns), case
    )


@pytest.mark.upstream_source
@pytest.mark.parametrize("case", CASES)
def test_actual_upstream_factory_has_same_dispatch_buffer_contract(
    upstream_source_root, case
):
    path = (
        upstream_source_root / "vllm/model_executor/layers/fused_moe/all2all_utils.py"
    )
    check_dispatch(lambda ns: function(path, "maybe_make_prepare_finalize", ns), case)
