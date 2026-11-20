# SPDX-License-Identifier: Apache-2.0
"""Honor ``ple_quant_scheme`` for the Qwen4Exp PLE n-gram table on MXFP4.

The internal Qwen4-Max MXFP4 checkpoint keeps its routed experts in MXFP4 but
stores the PLE n-gram embedding table as FP8 per-tensor shards, and declares
that with ``quantization_config.ple_quant_scheme = "fp8_pertensor"``.
Upstream drops the declaration twice:

* ``Mxfp4Config.from_config`` is ``return cls()`` -- every checkpoint key
  beyond ``__init__``'s arguments is discarded, so ``ple_quant_scheme`` never
  reaches the config instance;
* ``Qwen4ExpPLEEmbeddingMethod.from_quant_config`` only honors an explicit
  ``embedding_dtype`` (fed from ``text_config.ple_embedding_dtype``, a key no
  official checkpoint writes) and global quant-method inference.  An
  ``Mxfp4Config`` matches none of the ModelOpt / CompressedTensors / INC /
  Quark / Fp8Config branches, so the table falls through to the unquantized
  warning path and the FP8 shards are misloaded as raw codes.

This module bridges the dispatch half: ``from_quant_config`` dispatches
``"fp8_pertensor"`` to ``Qwen4ExpPLEFp8EmbeddingMethod`` and rejects any other
declared scheme.  An explicit ``embedding_dtype`` still wins, and without a
declared scheme upstream's inference runs unchanged -- so every checkpoint
that loads today (BF16 PLE tables, blockwise-FP8 global inference) keeps its
current method.

The config half lives in ``enhancement/mxfp4.py``: that module already
replaces ``Mxfp4Config.from_config`` for the PPU ``ignore`` /
``fp8_channelwise_layers`` plumbing, and the patch framework forbids two
patches on one target, so its replacement attaches ``ple_quant_scheme`` from
the checkpoint dict as well.

Leaf-style module: importable on a bare CPU runner, patches install when
``install()`` runs.  All call sites reach the helper through the class
attribute at runtime (``common`` / ``nvidia`` / ``amd`` PLE layers), so no
import-time consumer alias needs rebinding.
"""

from __future__ import annotations

import importlib

from vllm_sail.patch.utils import patch

_NGRAM_MODULE = "vllm.models.qwen4_exp.common.ngram_embedding"
_NGRAM_TARGET = "Qwen4ExpPLEEmbeddingMethod.from_quant_config"
_AFFECTED = ">=0.32.0,<0.33.0"

REASON_DISPATCH = (
    "Qwen4ExpPLEEmbeddingMethod.from_quant_config only honors an explicit "
    "embedding_dtype (text_config.ple_embedding_dtype, written by no official "
    "checkpoint) and global quant-method inference; an Mxfp4Config matches "
    "none of its branches, so an FP8 PLE table declared via "
    "quantization_config.ple_quant_scheme falls through to the unquantized "
    "warning path. Dispatch 'fp8_pertensor' to the FP8 PLE method and reject "
    "unknown schemes; an explicit embedding_dtype still takes priority. The "
    "scheme reaches the config instance through enhancement/mxfp4.py's "
    "from_config replacement."
)
REMOVE_WHEN = (
    "upstream Mxfp4Config.from_config preserves ple_quant_scheme and "
    "Qwen4ExpPLEEmbeddingMethod.from_quant_config dispatches on it (no "
    "upstream PR as of 2026-11; verify by reading both functions), or the "
    "PLE n-gram table stops shipping quantized shards."
)

METADATA = (
    (f"{_NGRAM_MODULE}.{_NGRAM_TARGET}", REASON_DISPATCH, _AFFECTED, REMOVE_WHEN),
)

_installed = False
#: True when the upstream modules cannot be imported (e.g. a vLLM without the
#: qwen4_exp model family); the fix has no consumer there.
_unavailable = False


def install() -> None:
    """Register the ple_quant_scheme dispatch bridge once."""
    global _installed, _unavailable
    if _installed or _unavailable:
        return

    from vllm.logger import init_logger

    logger = init_logger(__name__)
    try:
        # Probe exactly how patch() imports the target: a from-import can
        # resolve through an already-bound parent attribute and succeed even
        # when importlib.import_module would fail.
        ngram_embedding = importlib.import_module(_NGRAM_MODULE)
    except ImportError as exc:
        _unavailable = True
        logger.warning(
            "Skipping the Qwen4Exp PLE ple_quant_scheme patch: the upstream "
            "qwen4_exp modules cannot be imported (%s).",
            exc,
        )
        return

    _upstream_from_quant_config = (
        ngram_embedding.Qwen4ExpPLEEmbeddingMethod.from_quant_config
    )

    @patch(
        _NGRAM_MODULE,
        _NGRAM_TARGET,
        reason=REASON_DISPATCH,
        affected_versions=_AFFECTED,
        remove_when=REMOVE_WHEN,
    )
    def _from_quant_config(quant_config, prefix, embedding_dtype=None):
        if embedding_dtype is None:
            scheme = getattr(quant_config, "ple_quant_scheme", None)
            if scheme is not None:
                if scheme != "fp8_pertensor":
                    raise ValueError(
                        f"Unsupported ple_quant_scheme {scheme!r} for the PLE "
                        f"embedding {prefix!r}; expected 'fp8_pertensor'"
                    )
                from vllm.models.qwen4_exp.common.ngram_embedding import (
                    Qwen4ExpPLEFp8EmbeddingMethod,
                )

                return Qwen4ExpPLEFp8EmbeddingMethod()
        return _upstream_from_quant_config(quant_config, prefix, embedding_dtype)

    _installed = True
