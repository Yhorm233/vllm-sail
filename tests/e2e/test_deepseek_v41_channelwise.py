# SPDX-License-Identifier: Apache-2.0
"""Exercise channelwise checkpoint loading through the actual PPU FP8 kernel."""

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
    from vllm.platforms import current_platform

    from vllm_sail.model_executor.kernels.linear.scaled_mm.ppu import (
        PPUCutlassFP8ScaledMMLinearKernel,
        PPUDeepGemmFP8ScaledMMLinearKernel,
    )

    use_cutlass = current_platform.is_device_capability((8, 0))
    expected_kernel = (
        PPUCutlassFP8ScaledMMLinearKernel
        if use_cutlass
        else PPUDeepGemmFP8ScaledMMLinearKernel
    )

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
            assert type(scheme.fp8_linear) is expected_kernel
            assert layer.weight.dtype == torch.float8_e4m3fn
            assert layer.weight.shape == (
                (input_size, 128) if use_cutlass else (128, input_size)
            )
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
def test_fp8_linear_scales_bias_and_graph_updates(
    per_token, per_channel, with_bias, output_dtype
):
    import torch

    import vllm_sail

    vllm_sail.register_out_of_tree()
    from vllm_sail.model_executor.kernels.linear.scaled_mm.ppu import (
        PPUCutlassFP8ScaledMMLinearKernel,
    )

    dtype = getattr(torch, output_dtype)
    generator = torch.Generator().manual_seed(14280)
    a_cpu = (torch.randn(5, 64, generator=generator) / 8).to(torch.float8_e4m3fn)
    b_cpu = (torch.randn(17, 64, generator=generator) / 8).to(torch.float8_e4m3fn).T
    as_cpu = torch.linspace(0.125, 0.5, 5 if per_token else 1)[:, None]
    bs_cpu = torch.linspace(0.25, 0.75, 17 if per_channel else 1)[:, None]
    bias_cpu = torch.linspace(-0.25, 0.25, 17).to(dtype) if with_bias else None
    a, b, as_, bs = a_cpu.cuda(), b_cpu.cuda(), as_cpu.cuda(), bs_cpu.cuda()
    bias = bias_cpu.cuda() if bias_cpu is not None else None
    kernel = object.__new__(PPUCutlassFP8ScaledMMLinearKernel)

    def step():
        return kernel.apply_scaled_mm(
            A=a,
            B=b,
            As=as_,
            Bs=bs,
            bias=bias,
            out_dtype=dtype,
            output_shape=[1, 5, 17],
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = step()
    for factor in (1.0, 1.25):
        as_.copy_(as_cpu * factor)
        graph.replay()
        expected = (a_cpu.float() @ b_cpu.float()) * (as_cpu * factor) * bs_cpu.T
        if bias_cpu is not None:
            expected = expected + bias_cpu.float()
        torch.testing.assert_close(
            actual.cpu(), expected.to(dtype)[None, :, :], rtol=0.01, atol=0.01
        )
