# SPDX-License-Identifier: Apache-2.0
"""Retain ordinary MiniMax MoE finalize on PPU."""

from vllm.models.minimax_m3.nvidia import model as _model
from vllm.models.minimax_m3.nvidia.model import MiniMaxM3Model
from vllm.platforms import current_platform

from vllm_sail.models.minimax_m3_mlp import init_mlp as _mlp_body
from vllm_sail.patch.bodies import bind_body
from vllm_sail.patch.utils import patch

_original_init = MiniMaxM3Model._init_fused_moe_finalize
_original_mlp_init = _model.MiniMaxM3MLP.__init__
_original_maybe_init_gemm_rs = _model._maybe_init_gemm_rs
_original_bind_gemm_rs = _model._bind_gemm_rs
_original_norm_forward = _model.MiniMAXGemmaRMSNorm.forward
_ppu_mlp_init = bind_body(_mlp_body, _model)


@patch(
    "vllm.models.minimax_m3.nvidia.model",
    "MiniMaxM3MLP.__init__",
    reason="The dense MLP eagerly imports an SM100-only MXFP8 optimization before checking its capability. PPU keeps the original linear and activation path without importing NVIDIA kernels.",
    affected_versions=">=0.32.0,<0.33.0",
    remove_when="MiniMax dense MLP gates the NVIDIA-only optimization import through a platform capability, or PPU implements that optimization.",
)
def init_mlp(
    self,
    config,
    intermediate_size,
    quant_config=None,
    reduce_results=True,
    disable_tp=False,
    use_sequence_parallel=False,
    run_gemm_rs=False,
    prefix="",
):
    function = _ppu_mlp_init if current_platform.is_ppu() else _original_mlp_init
    return function(
        self,
        config,
        intermediate_size,
        quant_config=quant_config,
        reduce_results=reduce_results,
        disable_tp=disable_tp,
        use_sequence_parallel=use_sequence_parallel,
        run_gemm_rs=run_gemm_rs,
        prefix=prefix,
    )


@patch(
    "vllm.models.minimax_m3.nvidia.model",
    "_maybe_init_gemm_rs",
    reason="MiniMax GEMM-RS imports SM100 CuTeDSL before its capability check. PPU retains sequence parallel with ordinary GEMM and reduce-scatter.",
    affected_versions=">=0.32.0,<0.33.0",
    remove_when="MiniMax resolves GEMM-RS through a platform capability before importing CuTeDSL.",
)
def maybe_init_gemm_rs(vllm_config, use_sequence_parallel):
    if current_platform.is_ppu():
        return False
    return _original_maybe_init_gemm_rs(vllm_config, use_sequence_parallel)


@patch(
    "vllm.models.minimax_m3.nvidia.model",
    "_bind_gemm_rs",
    reason="Explicit GEMM-RS constructor flags must not import NVIDIA CuTeDSL on PPU; a None binding preserves the real reduce-scatter fallback.",
    affected_versions=">=0.32.0,<0.33.0",
    remove_when="The GEMM-RS binding is platform-aware before importing NVIDIA dependencies.",
)
def bind_gemm_rs(linear):
    if current_platform.is_ppu():
        return None
    return _original_bind_gemm_rs(linear)


@patch(
    "vllm.models.minimax_m3.nvidia.model",
    "MiniMAXGemmaRMSNorm.forward",
    reason="The new 2-D norm imports the NVIDIA ops package, whose initializer eagerly imports CuTeDSL. PPU retains the existing unquantized Gemma norm contract.",
    affected_versions=">=0.32.0,<0.33.0",
    remove_when="The Triton norm can be imported without NVIDIA-only package initialization and is verified on PPU.",
)
def norm_forward(self, x, residual=None, mxfp8=False):
    if not current_platform.is_ppu():
        return _original_norm_forward(self, x, residual=residual, mxfp8=mxfp8)
    if mxfp8:
        raise ValueError("PPU Gemma norm does not emit FlashInfer 128x4 MXFP8 scales")
    from flashinfer.norm import gemma_fused_add_rmsnorm, gemma_rmsnorm

    if residual is None:
        return gemma_rmsnorm(x, self.weight, self.variance_epsilon)
    gemma_fused_add_rmsnorm(x, residual, self.weight, self.variance_epsilon)
    return x, residual


@patch(
    "vllm.models.minimax_m3.nvidia.model",
    "MiniMaxM3Model._init_fused_moe_finalize",
    reason="PPU inherits CUDA identity but does not implement NVIDIA's collective MoE finalize kernel or its dependencies.",
    affected_versions=">=0.32.0,<0.33.0",
    remove_when="MiniMax gates fused finalize through a platform implementation capability or PPU implements and verifies that collective kernel.",
)
def init_fused_moe_finalize(self, vllm_config):
    if current_platform.is_ppu():
        return None
    return _original_init(self, vllm_config)
