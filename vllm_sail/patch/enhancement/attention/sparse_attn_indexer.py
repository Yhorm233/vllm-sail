# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F821
# Copied bodies resolve globals in their upstream module through bind_body.
"""PPU patches for ``vllm.model_executor.layers.sparse_attn_indexer``.

The fork makes three changes to ``SparseAttnIndexer``:

* ``__init__`` warns instead of raising when DeepGEMM is missing on PPU;
  logits execution still requires SAIL DeepGEMM. Upstream raises on CUDA,
  which PPU inherits via ``PlatformEnum.CUDA``. Verbatim body with markers —
  delegation cannot intercept the upstream raise. It also freezes the V4.1
  W8A8 indexer contract and registers INT8 pack warmup for that path.
* ``forward_native`` redirects to ``forward_ppu`` on PPU before the CUDA
  branch. Verbatim body with markers (the fork's exact insertion point).
* Adds ``forward_ppu``, which calls the PPU ``ppu_sparse_attn_indexer`` custom
  op (registered by ``vllm_sail.ops``). Additive, installed with
  ``allow_missing=True``; the enabled-custom-op path reaches it through the
  ``custom_op_dispatch`` patch.

The fork's module-level ``from vllm._ppu_ops import ppu_ops`` is what
``vllm_sail.ops`` replaces; importing it here ensures the custom op is
registered before any forward call.
"""

from __future__ import annotations

import torch
from vllm.config import get_current_vllm_config
from vllm.distributed import get_dcp_group
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers import sparse_attn_indexer as _indexer
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import has_deep_gemm
from vllm.utils.torch_utils import _encode_layer_name

import vllm_sail.ops  # noqa: F401  (registers torch.ops.vllm.ppu_sparse_attn_indexer)
from vllm_sail.patch.bodies import bind_body
from vllm_sail.patch.utils import patch
from vllm_sail.utils.deep_gemm import is_deep_gemm_supported

_AFFECTED = ">=0.31.0,<0.32.0"
_MODULE = "vllm.model_executor.layers.sparse_attn_indexer"

logger = init_logger(__name__)


def _with_target_globals(fn):
    return bind_body(fn, _indexer)


@patch(
    _MODULE,
    "SparseAttnIndexer.__init__",
    reason=(
        "PPU warns about a missing DeepGEMM in SparseAttnIndexer, while upstream "
        "raises on CUDA. PPU inherits the CUDA raise via PlatformEnum.CUDA, so "
        "the fork's PPU branch must run first. Delegation cannot intercept the "
        "upstream raise, hence the verbatim body. PPU logits still require SAIL "
        "DeepGEMM. Freeze the V4.1 W8A8 contract and warm up packing with INT8 "
        "rather than FP8 for that path."
    ),
    affected_versions=_AFFECTED,
    remove_when="upstream downgrades the missing-DeepGEMM error to a warning, or PPU always ships DeepGEMM.",
)
@_with_target_globals
def sparse_attn_indexer_init(
    self,
    k_cache,
    quant_block_size: int,
    scale_fmt: str,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    max_total_seq_len: int,
    topk_indices_buffer: torch.Tensor,
    skip_k_cache_insert: bool = False,
    use_fp4_cache: bool = False,
    compress_ratio: int = 1,
    candidate_blocks: torch.Tensor | None = None,
    candidate_block_size: int = 0,
    candidate_write: bool = False,
):
    # PPU MODIFICATION: begin
    # An out-of-class copied method has no implicit __class__ cell.
    super(SparseAttnIndexer, self).__init__()
    # PPU MODIFICATION: end
    self.k_cache = k_cache
    self.quant_block_size = quant_block_size
    self.scale_fmt = scale_fmt
    self.topk_tokens = topk_tokens
    self.head_dim = head_dim
    self.max_model_len = max_model_len
    self.max_total_seq_len = max_total_seq_len
    self.topk_indices_buffer = topk_indices_buffer
    self.skip_k_cache_insert = skip_k_cache_insert
    self.use_fp4_cache = use_fp4_cache
    self.compress_ratio = compress_ratio
    # v4.1 two-level selection: the candidate source indexer writes the
    # top candidate blocks here; later indexers mask their scores with it.
    self.candidate_blocks = candidate_blocks
    self.candidate_block_size = candidate_block_size
    self.candidate_write = candidate_write
    self.dense_mha_metadata_layer_name = ""
    # DCP scalars are constant for the run; resolve them here (config is set
    # during model construction) and pass them into the custom op, rather
    # than threading them through per-step metadata.
    vllm_config = get_current_vllm_config()
    # PPU MODIFICATION: begin
    from vllm_sail.models.deepseek_v41.config import uses_int8_w8a8

    self._v41_int8_indexer = current_platform.is_ppu() and uses_int8_w8a8(vllm_config)
    # PPU MODIFICATION: end
    parallel_config = vllm_config.parallel_config
    self.topk_backend = vllm_config.kernel_config.sparse_indexer_topk_backend
    self._parallel_config = parallel_config
    self.dcp_world_size = parallel_config.decode_context_parallel_size
    self.dcp_rank = get_dcp_group().rank_in_group if self.dcp_world_size > 1 else 0
    self.use_pcp = parallel_config.prefill_context_parallel_size > 1
    self._cp_kv_cache_interleave_size: int | None = None
    # PPU MODIFICATION: begin
    from vllm_sail.utils.deep_gemm import is_deep_gemm_supported

    if current_platform.is_ppu() and not is_deep_gemm_supported():
        logger.warning_once("SparseAttnIndexer requires SAIL DeepGEMM for PPU logits.")
    elif (
        not current_platform.is_ppu()
        and current_platform.is_cuda()
        and not has_deep_gemm()
    ):
        # PPU MODIFICATION: end
        raise RuntimeError(
            "Sparse Attention Indexer CUDA op requires DeepGEMM support in "
            "the current vLLM environment."
        )

    if vllm_config.kernel_config.enable_jit_warmup:
        from vllm.v1.attention.ops.common import (
            _PACK_SEQ_TRITON_KERNEL,
            _UNPACK_SEQ_TRITON_KERNEL,
        )

        pack_dtype = torch.uint8 if use_fp4_cache else current_platform.fp8_dtype()
        # PPU MODIFICATION: begin
        if self._v41_int8_indexer:
            pack_dtype = torch.int8
        # PPU MODIFICATION: end
        _PACK_SEQ_TRITON_KERNEL.register_warmup(
            dtype=pack_dtype,
            pad_value=0 if use_fp4_cache else -float("inf"),
        )
        _UNPACK_SEQ_TRITON_KERNEL.register_warmup()

        # PPU MODIFICATION: begin
        if (
            self.dcp_world_size > 1
            and not current_platform.is_ppu()
            and current_platform.is_cuda()
            and has_cutedsl()
        ):
            # PPU MODIFICATION: end
            from vllm.model_executor.kernels.attention.dsa.dcp_indexer_cutedsl import (  # noqa: E501
                _PACK_DCP_TOPK_CANDIDATES_KERNEL,
                _STABLE_TOPK_FROM_GATHERED_CANDIDATES_KERNEL,
            )

            _PACK_DCP_TOPK_CANDIDATES_KERNEL.register_warmup()
            _STABLE_TOPK_FROM_GATHERED_CANDIDATES_KERNEL.register_warmup()


_upstream_forward_native = _indexer.SparseAttnIndexer.forward_native


@patch(
    _MODULE,
    "SparseAttnIndexer.forward_native",
    reason=(
        "Fork redirects SparseAttnIndexer.forward_native to forward_ppu on PPU "
        "before the CUDA branch, so the disabled-custom-op path also uses the "
        "PPU kernel; other platforms delegate to the upstream implementation."
    ),
    affected_versions=_AFFECTED,
    remove_when="upstream's forward_native grows a platform hook, or PPU stops overriding this op.",
)
def sparse_attn_indexer_forward_native(
    self,
    hidden_states: torch.Tensor,
    q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    k: torch.Tensor | None,
    weights: torch.Tensor,
):
    if current_platform.is_ppu():
        return self.forward_ppu(hidden_states, q_quant, k, weights)
    return _upstream_forward_native(self, hidden_states, q_quant, k, weights)


@patch(
    _MODULE,
    "SparseAttnIndexer.forward_ppu",
    allow_missing=True,
    reason=(
        "PPU implementation of the sparse attention indexer, calling the PPU "
        "ppu_sparse_attn_indexer custom op registered by vllm_sail.ops. Upstream "
        "has no forward_ppu; installed additively and reached through the "
        "custom_op_dispatch patch (enabled path) or the forward_native redirect "
        "(disabled path). V4.1 W8A8 requires INT8 Q with its scale folded into "
        "weights; reject FP8/FP4 inputs before invoking the op."
    ),
    affected_versions=_AFFECTED,
    remove_when="upstream adds a forward_ppu hook and the PPU op moves into the dispatch ladder.",
)
def sparse_attn_indexer_forward_ppu(
    self,
    hidden_states: torch.Tensor,
    q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    k: torch.Tensor | None,
    weights: torch.Tensor,
):
    if isinstance(q_quant, tuple):
        q_values, q_scale = q_quant
    else:
        q_values, q_scale = q_quant, None
    # PPU MODIFICATION: begin
    if self._v41_int8_indexer and (
        q_values.dtype != torch.int8 or q_scale is not None or self.use_fp4_cache
    ):
        raise ValueError(
            "PPU V4.1 W8A8 indexer requires torch.int8 Q, no FP4 cache, "
            "and Q scales folded into weights; FP8/FP4 fallback is not supported."
        )
    # PPU MODIFICATION: end
    return torch.ops.vllm.ppu_sparse_attn_indexer(
        hidden_states,
        _encode_layer_name(self.k_cache.prefix),
        self.k_cache.kv_cache,
        q_values,
        q_scale,
        k,
        weights,
        self.quant_block_size,
        self.scale_fmt,
        self.topk_tokens,
        self.head_dim,
        self.max_model_len,
        self.max_total_seq_len,
        self.topk_indices_buffer,
        self.skip_k_cache_insert,
        self.use_fp4_cache,
        self.candidate_blocks,
        self.candidate_block_size,
        self.candidate_write,
    )
