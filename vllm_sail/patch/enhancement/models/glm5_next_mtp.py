# SPDX-License-Identifier: Apache-2.0
"""Quantization support for the GLM-5.3-Flash MTP draft (``Glm5NextMTP``).

The checkpoint names its FP8-channelwise dense layers in the text-tower space:
``model.language_model.layers.45.mlp.shared_experts.down_proj``. The target
model builds its layers under exactly that prefix, so the list matches as-is.
The MTP draft builds ``model.layers.45.mlp...`` instead -- the *parameter* tree
carries an extra ``mtp_block`` attribute, the quantization prefixes do not -- so
``Mxfp4Config.get_quant_method`` never selects the channelwise scheme for its
dense projections, and loading their ``weight_scale`` raises ``KeyError`` in
``Glm5NextMTP.load_weights``.

The ``SupportsQuant`` protocol that ``NemotronHMTP`` already implements covers
all of it with three class attributes:

* ``SupportsQuant`` in the bases gives the draft the ``__new__`` hook that
  applies the model mapping to the quantization config before any layer is
  built;
* ``hf_to_vllm_mapper`` maps the checkpoint's layer prefix onto the draft's
  (``model.language_model.layers.N.<rest>`` -> ``model.layers.N.<rest>``);
* ``packed_modules_mapping`` mirrors ``Glm5NextForConditionalGeneration`` so
  ``is_layer_skipped`` expands a fused ``gate_up_proj`` into the checkpoint's
  ``gate_proj`` / ``up_proj`` names.

The mapping is applied in place to the config carried by ``vllm_config``, which
the target and the draft share. The target has finished building its layers by
the time the draft is constructed, so only the draft observes the mapped names.
"""

from __future__ import annotations

import re

from vllm_sail.patch.utils import PATCH_MARKER, PATCH_REGISTRY, PatchRecord, patch

_MTP_MODULE = "vllm.models.glm5next.common.mtp"
_AFFECTED = ">=0.32.0,<0.33.0; upstream main e83d82a"

#: Fused -> checkpoint projection names, as on ``Glm5NextForConditionalGeneration``.
_PACKED_MODULES_MAPPING = {"gate_up_proj": ["gate_proj", "up_proj"]}

#: Text-tower prefixes that put ``layers.N.`` at the root of the draft's
#: quantization namespace. The plain ``model.layers.`` alternative makes the
#: mapping idempotent if it runs twice.
_LAYER_PREFIX_RE = re.compile(
    r"^(?:model\.language_model|language_model\.model|language_model|model)\."
    r"layers\.(\d+)\."
)

_BASES_REASON = (
    "Glm5NextMTP is a plain nn.Module, so it misses the SupportsQuant.__new__ "
    "hook that applies hf_to_vllm_mapper to the quantization config before any "
    "layer is built (NemotronHMTP does inherit SupportsQuant). Without it the "
    "MTP dense shared-expert projections are built unquantised and their "
    "weight_scale tensors KeyError on load."
)
_BASES_REMOVE_WHEN = (
    "upstream declares Glm5NextMTP as SupportsQuant, or builds the MTP draft "
    "under the checkpoint's layer naming."
)
_MAPPER_REASON = (
    "Glm5NextMTP defines no hf_to_vllm_mapper, so the checkpoint's layer lists "
    "are never mapped onto the draft's prefixes: the draft builds "
    "model.layers.N.<rest> while the checkpoint names "
    "model.language_model.layers.N.<rest>. SupportsQuant.__new__ forwards this "
    "mapper to quant_config.apply_vllm_mapper."
)
_MAPPER_REMOVE_WHEN = (
    "upstream defines hf_to_vllm_mapper on Glm5NextMTP, or builds the MTP draft "
    "under the checkpoint's layer naming."
)
_PACKED_REASON = (
    "Glm5NextMTP declares no packed_modules_mapping, so the Mxfp4Config it hands "
    "to its layers cannot expand a fused gate_up_proj into the checkpoint's "
    "gate_proj/up_proj names. is_layer_skipped then misses "
    "fp8_channelwise_layers entries for the MTP shared experts, those layers are "
    "built unquantised and loading their weight_scale raises KeyError. Mirrors "
    "Glm5NextForConditionalGeneration.packed_modules_mapping."
)
_PACKED_REMOVE_WHEN = (
    "upstream declares packed_modules_mapping on Glm5NextMTP (or on "
    "Glm5NextMultiTokenPredictorLayer) with the gate_up_proj entry."
)


def _enable_supports_quant() -> None:
    """Add ``SupportsQuant`` to ``Glm5NextMTP``'s bases (bookkept)."""
    from vllm.logger import init_logger
    from vllm.model_executor.models.interfaces import SupportsQuant
    from vllm.models.glm5next.common import mtp as mtp_module

    model_class = mtp_module.Glm5NextMTP
    if issubclass(model_class, SupportsQuant):
        return

    try:
        model_class.__bases__ = (*model_class.__bases__, SupportsQuant)
    except TypeError as exc:  # pragma: no cover - layout conflict
        init_logger("vllm.models.glm5next.common.mtp").warning(
            "vllm-sail glm5_next_mtp: cannot add SupportsQuant to %s (%s); the "
            "MTP draft stays unquantised and will fail to load weight_scale",
            model_class.__name__,
            exc,
        )
        return

    # Recorded as a value install against the class rather than
    # ``Glm5NextMTP.__bases__``: the patch inventory gate only resolves a
    # hand-written ``PatchRecord`` whose ``kind`` is "value" and whose target is
    # declared by the pinned source, and ``__bases__`` is a runtime attribute of
    # every class that no source declares. Naming the class keeps the record
    # checkable, the shape ``residual/kernel_moe_backend_literal`` already uses
    # for a non-callable target.
    PATCH_REGISTRY.append(
        PatchRecord(
            target=f"{_MTP_MODULE}.Glm5NextMTP",
            reason=_BASES_REASON,
            affected_versions=_AFFECTED,
            remove_when=_BASES_REMOVE_WHEN,
            kind="value",
            was_missing=False,
        )
    )


def _enable_packed_modules_mapping() -> None:
    """Declare the fused->checkpoint projection table on ``Glm5NextMTP``.

    ``patch``/``patch_value`` cannot install this: both record the replaced
    object on the replacement itself, and a dict rejects attribute writes.
    """
    from vllm.models.glm5next.common import mtp as mtp_module

    model_class = mtp_module.Glm5NextMTP
    if getattr(model_class, "packed_modules_mapping", None):
        return
    model_class.packed_modules_mapping = {
        fused: list(shards) for fused, shards in _PACKED_MODULES_MAPPING.items()
    }
    PATCH_REGISTRY.append(
        PatchRecord(
            target=f"{_MTP_MODULE}.Glm5NextMTP.packed_modules_mapping",
            reason=_PACKED_REASON,
            affected_versions=_AFFECTED,
            remove_when=_PACKED_REMOVE_WHEN,
            kind="value",
            was_missing=True,
        )
    )


try:
    from vllm.model_executor.models.utils import WeightsMapper

    class _DraftWeightsMapper(WeightsMapper):
        # The upstream mapper is slotted. Keep patch provenance on our own
        # instance without modifying the upstream type or other mappers.
        __slots__ = (PATCH_MARKER,)

    _enable_supports_quant()
    _enable_packed_modules_mapping()

    # A class attribute, so the imperative form: ``patch`` is a decorator and
    # only works on callables/classes.
    patch(
        _MTP_MODULE,
        "Glm5NextMTP.hf_to_vllm_mapper",
        reason=_MAPPER_REASON,
        affected_versions=_AFFECTED,
        remove_when=_MAPPER_REMOVE_WHEN,
        allow_missing=True,
    )(
        _DraftWeightsMapper(
            orig_to_new_regex={_LAYER_PREFIX_RE: r"model.layers.\1."},
        )
    )
except ImportError:  # pragma: no cover - older transformers without the family
    # glm5next model family unavailable; nothing to patch.
    pass
