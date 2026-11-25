# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: I001  (V4 cache ownership and delayed providers precede V4.1 consumers)
"""Model-layer PPU patches (Phase 5).

Ports the fork's seven modified ``vllm/model_executor/models/`` files
(+155/-44) plus the Marlin MoE gate from decision D2 (phase-1 spec 1.8(c)).
Importing this package applies every patch in it, exactly like the
``bugfix`` / ``enhancement`` / ``performance`` category packages do.

Wired into ``vllm_sail/patch/enhancement/__init__.py`` (import applies every
patch); ``tests/ut/test_model_patches.py::test_subpackage_wiring_is_present``
guards the connection.

Modules, and why each is import-on-install like the rest of the framework:

* ``deepseek_v2`` — separate PPU indexer wk projection with shared upstream model and MTP loaders.
* ``deepseek_v4`` — selects the plugin FlashMLA subclass only on PPU; the
  quantization-specific weights mapper remains plugin-local so CUDA behavior
  is unchanged.
* ``deepseek_v41_mega_mhc`` — routes the V4.1 Mega mHC kernel to the PPU
  DeepGEMM build, which supports ZW-M890P instead of the upstream SM 10.x
  family.
* ``glm5_next_mtp`` — quantization support for the GLM-5.3-Flash MTP draft:
  ``SupportsQuant`` in ``Glm5NextMTP``'s bases, a ``hf_to_vllm_mapper`` that
  strips the checkpoint's tower prefix (``model.language_model.layers.N.`` ->
  ``model.layers.N.``) and the ``packed_modules_mapping`` the fused
  shared-expert projection needs, so the channelwise dense layers survive the
  draft's layer naming. The pinned upstream passes the quantization config
  directly to the MLA projections.
* ``kimi_k3`` — quantized routed up-projection support for the latent-MoE
  tail's fused tiers.
* ``llama4`` — torch.compile on the custom router (+1).
* ``minimax_m3`` — retain ordinary MoE finalize on PPU before NVIDIA imports.
* ``minimax_m3_msa`` — route MiniMax M3 indexer + sparse attend to the SAIL
  (``msa_for_sail``) kernels on PPU, with 890P fp8 / 810E bf16 defaults.
* ``qwen3_moe`` — per-layer mix_layer quant overrides (+36).
* ``qwen3_next`` — NVTX profiling scaffold (+19).
* ``qwen4_exp_qsa`` — route Qwen4Exp QSA through PPU FA3's CUDA kernel where it
  can run the shape; upstream hard-wires the Triton one.
* ``qwen4_exp_qsa_fp4`` — QSA indexer compressed key cache as MXFP4, scored by
  DeepGEMM's FP4 average MQA logits kernels.
* ``step5_dsa_backend`` — admits SM 8.x PPU to Step5 DSA and runs upstream's
  Triton indexer scorer there, since PPU FlashInfer has no ``attn_scores``;
  resolves the region top-k to PPU DeepSelect or ``per_row``.
* ``step5_soft_fp8`` — byte-based Step5 indexer kernels with software E4M3
  for PPU 1.0, whose Triton has no ``fp8e4nv``.
* ``step5_indexer_fp16`` — ``VLLM_STEP5_INDEXER_FP8_AS_FP16``: stores the
  Step5 indexer summaries and query as exact FP16 values of their E4M3 codes.
* ``step5_mtp`` — maps the checkpoint ignore names of BF16 Step5 MTP layers
  to the draft model's ``mtp_block`` / ``shared_head.head`` module paths.
* ``step5_norm`` — keeps Step5 RMSNorm on upstream's PyTorch fallback on PPU
  instead of its ``is_cuda``-selected CuTeDSL kernel.
* ``moe_marlin_gate`` — VLLM_PPU_ENABLE_MOE_MARLIN default-off gate.
"""

# Capture the shared Q alias for V4.1 before V4 patches it to avoid duplicate registration.
from vllm_sail.patch.enhancement.models import deepseek_v41_int8  # noqa: F401, I001
from vllm_sail.patch.enhancement.models import (
    deepseek_v2,  # noqa: F401
    deepseek_v4,  # noqa: F401
    deepseek_v4_cache,  # noqa: F401
    deepseek_v4_compressor,  # noqa: F401
    deepseek_v4_graph_warmup,  # noqa: F401
    deepseek_v4_metadata,  # noqa: F401
    deepseek_v41,  # noqa: F401
    deepseek_v41_device,  # noqa: F401
    deepseek_v41_engram,  # noqa: F401
    deepseek_v41_indexer,  # noqa: F401
    deepseek_v41_mega_mhc,  # noqa: F401
    glm5_next,  # noqa: F401
    glm5_next_indexer,  # noqa: F401
    glm5_next_mtp,  # noqa: F401
    kimi_k3,  # noqa: F401
    llama4,  # noqa: F401
    minimax_m3,  # noqa: F401
    minimax_m3_msa,  # noqa: F401
    moe_marlin_gate,  # noqa: F401
    qwen3_moe,  # noqa: F401
    qwen3_next,  # noqa: F401
    qwen4_exp_qsa,  # noqa: F401
    qwen4_exp_qsa_fp4,  # noqa: F401
    step5_dsa_backend,  # noqa: F401
    step5_indexer_fp16,  # noqa: F401
    step5_mtp,  # noqa: F401
    step5_norm,  # noqa: F401
    step5_soft_fp8,  # noqa: F401
)
