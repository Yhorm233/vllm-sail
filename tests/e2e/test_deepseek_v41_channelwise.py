# SPDX-License-Identifier: Apache-2.0
"""Exercise channelwise loading through the PPU FP8 kernel and W8A8 INT8 numerics."""

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.ppu


@pytest.mark.parametrize("input_size", [256, 576])
def test_channelwise_dense_loading_and_per_token_activation(monkeypatch, input_size):
    import torch
    from compressed_tensors.quantization import QuantizationArgs
    from vllm.config import VllmConfig, set_current_vllm_config

    import vllm_sail

    monkeypatch.setenv("VLLM_USE_DEEP_GEMM", "1")
    vllm_sail.register_out_of_tree()
    # This is a single-device linear oracle; no distributed engine is started.
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 1
    )
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 import (
        CompressedTensorsW8A8Fp8,
    )

    from vllm_sail.model_executor.kernels.linear.scaled_mm.ppu import (
        PPUDeepGemmFP8ScaledMMLinearKernel,
    )

    # SM80 PPUs such as 810E have no FP8 linear kernel; they run the INT8 cases below.
    supported, reason = PPUDeepGemmFP8ScaledMMLinearKernel.is_supported()
    if not supported:
        pytest.skip(reason)
    config = VllmConfig()
    config.model_config = SimpleNamespace(dtype=torch.bfloat16)
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with set_current_vllm_config(config), torch.device("cuda"):
            scheme = CompressedTensorsW8A8Fp8(
                QuantizationArgs(num_bits=8, type="float", strategy="channel"),
                is_static_input_scheme=False,
            )
            layer = torch.nn.Module()
            scheme.create_weights(
                layer,
                input_size,
                [128],
                input_size,
                128,
                torch.bfloat16,
                weight_loader=lambda param, value: param.copy_(value),
            )
            scales = torch.linspace(0.125, 0.5, 128, dtype=torch.float32)[:, None]
            layer.weight.data.fill_(2.0)
            layer.weight_scale.data.copy_(scales)
            scheme.process_weights_after_loading(layer)
            assert isinstance(scheme.fp8_linear, PPUDeepGemmFP8ScaledMMLinearKernel)
            assert layer.weight.dtype == torch.float8_e4m3fn
            assert layer.weight.shape == (128, input_size)
            x = torch.tensor([0.5, -1.0, 2.0], dtype=torch.bfloat16)
            x = x[:, None].expand(3, input_size).contiguous()
            actual = scheme.apply_weights(layer, x)
            expected = (x[:, :1].float() * (2 * input_size) * scales.T).to(
                torch.bfloat16
            )
            torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.01)
    finally:
        torch.set_default_dtype(previous_dtype)


@pytest.mark.parametrize("per_token", [False, True])
@pytest.mark.parametrize("per_channel", [False, True])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("output_dtype", ["float16", "bfloat16"])
def test_int8_linear_scales_bias_and_graph_updates(
    monkeypatch, per_token, per_channel, with_bias, output_dtype
):
    import torch

    import vllm_sail

    vllm_sail.register_out_of_tree()
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import (
        CompressedTensorsW8A8Int8,
    )

    from vllm_sail.model_executor.kernels.linear.scaled_mm.ppu import (
        PPUInt8ScaledMMLinearKernel,
    )

    monkeypatch.setenv("VLLM_SAIL_DENSE_BACKEND", "triton")
    monkeypatch.setenv("VLLM_SAIL_USE_TRITON_INT8_QUANT", "1")
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 1
    )
    dtype = getattr(torch, output_dtype)
    generator = torch.Generator().manual_seed(14280)
    a_cpu = torch.randint(-8, 9, (5, 64), generator=generator, dtype=torch.int8)
    # Fix each row's amax so dynamic quantization scales match the independent oracle.
    a_cpu[:, 0] = 127
    b_cpu = torch.randint(-8, 9, (17, 64), generator=generator, dtype=torch.int8).T
    as_cpu = torch.linspace(0.125, 0.5, 5 if per_token else 1)[:, None]
    bs_cpu = torch.linspace(0.25, 0.75, 17 if per_channel else 1)[:, None]
    bias_cpu = torch.linspace(-0.25, 0.25, 17).to(dtype) if with_bias else None
    x_cpu = (a_cpu.float() * as_cpu).to(dtype)
    x = x_cpu.cuda()[None, :, :]
    bias = bias_cpu.cuda() if bias_cpu is not None else None
    config = VllmConfig()
    config.model_config = SimpleNamespace(dtype=dtype)
    with set_current_vllm_config(config), torch.device("cuda"):
        scheme = CompressedTensorsW8A8Int8(
            "channel" if per_channel else "tensor",
            is_static_input_scheme=not per_token,
            input_symmetric=True,
        )
        layer = torch.nn.Module()
        scheme.create_weights(
            layer,
            output_partition_sizes=[17],
            input_size_per_partition=64,
            params_dtype=dtype,
            weight_loader=lambda param, value: param.copy_(value),
        )
        layer.weight.data.copy_(b_cpu.T)
        layer.weight_scale.data.copy_(bs_cpu.reshape_as(layer.weight_scale))
        if not per_token:
            layer.input_scale.data.fill_(as_cpu.item())
        scheme.process_weights_after_loading(layer)
    assert type(scheme.kernel) is PPUInt8ScaledMMLinearKernel
    assert layer.weight.dtype == torch.int8
    assert not scheme.kernel.weight_RowMajor

    def step():
        return scheme.apply_weights(layer, x, bias)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = step()
    for factor in (1.0, 2.0):
        x.copy_((x_cpu * factor)[None, :, :])
        if not per_token:
            layer.input_scale.data.fill_(as_cpu.item() * factor)
        layer.weight_scale.data.copy_((bs_cpu * factor).reshape_as(layer.weight_scale))
        graph.replay()
        expected = (
            (a_cpu.float() @ b_cpu.float()) * (as_cpu * factor) * (bs_cpu.T * factor)
        )
        if bias_cpu is not None:
            expected = expected + bias_cpu.float()
        torch.testing.assert_close(
            actual.cpu(), expected.to(dtype)[None, :, :], rtol=0.01, atol=0.01
        )
