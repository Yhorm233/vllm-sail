# SPDX-License-Identifier: Apache-2.0
"""PPU patches for ``vllm.v1.attention.backends.mla.flashmla`` (dense FlashMLA).

The fork makes three changes, all PPU-gated; each is applied here by delegation
rather than a body copy:

* ``FlashMLABackend.supports_compute_capability`` accepts capability major 8 on
  PPU (upstream: 9 and 10).
* ``FlashMLAMetadataBuilder.__init__`` sizes the CUDA-graph tile-scheduler
  buffer from a hard-coded ``num_sms = 320`` on PPU ("PPU FIXME (kai):
  temporarily use hard-coded ppu magic number, this is align with ppu mla
  algo"). The upstream ``__init__`` is run first, then the buffer is
  re-allocated with the PPU size.

The fork's third change — the ``[FM_FMHA]`` NVTX range around the decode kernel
in ``FlashMLAImpl.forward_mqa`` — is not patched here: the plugin emits it from
``vllm.v1.attention.ops.flashmla.flash_mla_with_kvcache`` (see
``flashmla_ops``), i.e. around the very call the fork brackets, without wrapping
``forward_mqa`` or handling its untuple ``q``.
"""

from __future__ import annotations

import torch
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.mla import flashmla as _flashmla_backend

from vllm_sail.patch.utils import patch

_AFFECTED = ">=0.31.0,<0.32.0"
_MODULE = "vllm.v1.attention.backends.mla.flashmla"

#: Hard-coded SM count the fork uses for the PPU MLA scheduler buffers.
_PPU_NUM_SMS = 320

_upstream_supports_compute_capability = (
    _flashmla_backend.FlashMLABackend.supports_compute_capability
)
_upstream_builder_init = _flashmla_backend.FlashMLAMetadataBuilder.__init__


@patch(
    _MODULE,
    "FlashMLABackend.supports_compute_capability",
    reason=(
        "PPU runs FlashMLA on capability major 8. Upstream accepts only majors "
        "9 and 10, so without this PPU could never select the FlashMLA backend."
    ),
    affected_versions=_AFFECTED,
    remove_when="upstream accepts PPU's capability for FlashMLA natively.",
)
def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
    if current_platform.is_ppu():
        return capability.major in [8]
    return _upstream_supports_compute_capability(capability)


@patch(
    _MODULE,
    "FlashMLAMetadataBuilder.__init__",
    reason=(
        "PPU FIXME (kai): the fork hard-codes num_sms = 320 for the PPU MLA "
        "tile-scheduler buffers ('temporarily use hard-coded ppu magic number, "
        "this is align with ppu mla algo') instead of querying the device."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "PPU reports a compute-unit count the FlashMLA scheduler can use "
        "directly, and the fork's hard-coded 320 is removed."
    ),
)
def builder_init(
    self,
    kv_cache_spec,
    layer_names,
    vllm_config,
    device: torch.device,
) -> None:
    _upstream_builder_init(self, kv_cache_spec, layer_names, vllm_config, device)
    if current_platform.is_ppu() and self.cg_buf_tile_scheduler_metadata is not None:
        self.cg_buf_tile_scheduler_metadata = torch.zeros(
            (_PPU_NUM_SMS, 8),
            device=self.device,
            dtype=torch.int32,
        )
