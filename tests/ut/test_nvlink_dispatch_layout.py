# SPDX-License-Identifier: Apache-2.0
"""The one-sided factory forwards the layout's format and extra payload."""

import inspect
from types import SimpleNamespace as NS

import pytest

from tests.support.source import function

PATCH = "vllm_sail/patch/enhancement/deepep.py"
UPSTREAM = "vllm/model_executor/layers/fused_moe/all2all_utils.py"
CONSTRUCTOR = "vllm/model_executor/layers/fused_moe/prepare_finalize/flashinfer_nvlink_one_sided.py"


def check_factory(path, name, payload, source_root=None):
    calls, layouts = [], []
    dispatch_format = object() if payload else None
    layout = NS(dispatch_format=dispatch_format, extra_payload_bytes_per_token=payload)
    quant, dtype, result = object(), object(), object()

    def dispatch_layout(*args, **kwargs):
        layouts.append((args, kwargs))
        return layout

    def prepare_finalize(**kwargs):
        if source_root is not None:
            constructor = function(
                source_root / CONSTRUCTOR,
                "FlashInferNVLinkOneSidedPrepareAndFinalize.__init__",
                {},
            )
            inspect.signature(constructor).bind(object(), **kwargs)
        calls.append(kwargs)
        return result

    parallel = NS(use_passthrough_all2all=False, use_all2all_kernels=True)
    moe = NS(
        moe_parallel_config=parallel,
        use_deepep_ht_kernels=False,
        use_deepep_ll_kernels=False,
        use_deepep_v2_kernels=False,
        use_moonep_kernels=False,
        use_mori_epv2_kernels=False,
        use_mori_epv1_kernels=False,
        use_fi_nvl_two_sided_kernels=False,
        use_fi_nvl_one_sided_kernels=True,
        hidden_dim=4096,
        experts_per_token=8,
        num_experts=256,
    )
    namespace = {
        "get_current_vllm_config": lambda: NS(
            scheduler_config=NS(max_num_batched_tokens=512)
        ),
        "flashinfer_one_sided_dispatch_layout": dispatch_layout,
        "FlashInferNVLinkOneSidedPrepareAndFinalize": prepare_finalize,
    }
    actual = function(path, name, namespace)(
        moe, quant, all2all_manager=NS(world_size=8), input_dtype=dtype
    )
    assert actual is result
    assert layouts == [((4096, quant), {"input_dtype": dtype})]
    assert calls == [
        {
            "max_num_tokens": 512,
            "top_k": 8,
            "num_experts": 256,
            "hidden_size": 4096,
            "num_dispatchers": 8,
            "dispatch_format": dispatch_format,
            "extra_payload_bytes_per_token": payload,
        }
    ]


@pytest.mark.parametrize("payload", [0, 64])
def test_one_sided_factory_forwards_dispatch_layout(payload):
    check_factory(PATCH, "_maybe_make_prepare_finalize_body", payload)


@pytest.mark.upstream_source
@pytest.mark.parametrize("copied", [False, True])
@pytest.mark.parametrize("payload", [0, 64])
def test_actual_one_sided_layout_constructor(upstream_source_root, copied, payload):
    path, name = (
        (PATCH, "_maybe_make_prepare_finalize_body")
        if copied
        else (upstream_source_root / UPSTREAM, "maybe_make_prepare_finalize")
    )
    check_factory(path, name, payload, upstream_source_root)
