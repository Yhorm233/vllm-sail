# SPDX-License-Identifier: Apache-2.0
"""Preserve checkpoint-selected Engram WKV precision through the real TP loader."""

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.ppu


@pytest.mark.parametrize("layer_id", [1, 14])
@pytest.mark.parametrize("tp_rank", [0, 7])
@pytest.mark.parametrize("bf16_wkv", [False, True])
def test_engram_wkv_checkpoint_dtype_and_tp_loading(
    monkeypatch, layer_id, tp_rank, bf16_wkv
):
    import torch
    from vllm.config import VllmConfig, set_current_vllm_config

    import vllm_sail

    vllm_sail.register_out_of_tree()
    from vllm.model_executor.layers.linear import (
        ColumnParallelLinear,
        UnquantizedLinearMethod,
    )
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
        CompressedTensorsLinearMethod,
    )
    from vllm.model_executor.model_loader.utils import configure_quant_config

    from vllm_sail.models.deepseek_v41.model import DeepseekV41ForCausalLM

    int8 = {"num_bits": 8, "type": "int", "symmetric": True}
    checkpoint_name = f"layers.{layer_id}.engram.wkv"
    quant = CompressedTensorsConfig.from_config(
        {
            "quant_method": "compressed-tensors",
            "format": "int-quantized",
            "config_groups": {
                "W8A8": {
                    "targets": ["Linear"],
                    "weights": {**int8, "strategy": "channel", "dynamic": False},
                    "input_activations": {
                        **int8,
                        "strategy": "token",
                        "dynamic": True,
                    },
                }
            },
            "ignore": [checkpoint_name] if bf16_wkv else [],
        }
    )
    configure_quant_config(quant, DeepseekV41ForCausalLM)
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: tp_rank
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 8
    )
    config = VllmConfig()
    config.model_config = SimpleNamespace(dtype=torch.bfloat16)
    # A small CPU fixture exercises the same parameter creation and weight loader
    # as Engram's TP=8 WKV without allocating the full model or a process group.
    with set_current_vllm_config(config), torch.device("cpu"):
        layer = ColumnParallelLinear(
            8,
            16,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=quant,
            prefix=f"language_model.model.{checkpoint_name}",
            tp_rank=tp_rank,
            tp_size=8,
        )
    if bf16_wkv:
        # Every nonzero value is smaller than one: an erroneous INT8 allocation
        # would silently truncate these BF16 weights to zero on copy_.
        weight = (torch.arange(128).reshape(16, 8) / 256).to(torch.bfloat16)
        assert isinstance(layer.quant_method, UnquantizedLinearMethod)
        assert layer.weight.dtype == torch.bfloat16
        assert not hasattr(layer, "weight_scale")
    else:
        weight = (torch.arange(128).reshape(16, 8) - 64).to(torch.int8)
        assert isinstance(layer.quant_method, CompressedTensorsLinearMethod)
        assert layer.weight.dtype == torch.int8
        assert hasattr(layer, "weight_scale")
    layer.weight.weight_loader(layer.weight, weight)
    expected = weight.narrow(0, tp_rank * 2, 2)
    assert torch.equal(layer.weight, expected)
    assert torch.count_nonzero(layer.weight) > 0
