# SPDX-License-Identifier: Apache-2.0
"""GLM preserves the INT8 quantizer; only FP8 and Quark use BF16 staging."""

from types import SimpleNamespace as NS

import pytest

from tests.support.source import function


@pytest.mark.upstream_source
def test_glm_mla_keeps_the_mixed_precision_quantizer(upstream_source_root, modules):
    class Fp8Config:
        def __init__(self, serialized):
            self.is_checkpoint_fp8_serialized = serialized

    class QuarkConfig:
        pass

    modules("vllm.model_executor.layers.quantization.fp8", Fp8Config=Fp8Config)
    modules(
        "vllm.model_executor.layers.quantization.quark.quark", QuarkConfig=QuarkConfig
    )
    use_bf16 = function(
        upstream_source_root / "vllm/models/glm5next/common/model.py",
        "_mla_dequantized_at_load",
        {},
    )
    assert not use_bf16(NS(get_name=lambda: "mixed_precision_w4"))
    assert not use_bf16(Fp8Config(False))
    assert use_bf16(Fp8Config(True)) and use_bf16(QuarkConfig())


@pytest.mark.upstream_source
@pytest.mark.requires_runtime
@pytest.mark.parametrize("shape", [(8,), (8, 1)])
def test_per_channel_fp8_mla_uses_one_scale_per_row(upstream_source_root, shape):
    torch = pytest.importorskip("torch")

    dequant = function(
        upstream_source_root
        / "vllm/model_executor/layers/quantization/dequant_linear.py",
        "_dequantize",
        {"torch": torch},
    )
    scales = torch.arange(1, 9, dtype=torch.float32).reshape(shape)
    weight = torch.ones(8, 16).to(torch.float8_e4m3fn)
    actual = dequant(
        NS(weight=weight, weight_scale=scales, params_dtype=torch.bfloat16)
    )
    torch.testing.assert_close(actual.float(), scales.reshape(8, 1).expand(8, 16))
