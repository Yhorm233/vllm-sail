# SPDX-License-Identifier: Apache-2.0
"""Disable automatic NUMA binding unless explicitly requested."""

from __future__ import annotations

from vllm_sail.patch.utils import patch

_MODULE = "vllm.utils.numa_utils"
_TARGET = "_can_auto_bind"
_installed = False


def install() -> None:
    """Make automatic NUMA binding opt-in for vllm-sail, once per process."""
    global _installed
    if _installed:
        return

    @patch(
        _MODULE,
        _TARGET,
        reason=(
            "Upstream automatic NUMA binding concurrently mutates the global "
            "multiprocessing executable and can fail worker startup with "
            "vllm-project/vllm#67851. Default to the same unbound behavior as "
            "--no-numa-bind while preserving explicit --numa-bind opt-in."
        ),
        affected_versions=">=0.34.0,<0.35.0",
        remove_when=(
            "upstream changes numa_bind to default to False, making automatic "
            "NUMA binding opt-in."
        ),
    )
    def _can_auto_bind(
        parallel_config, local_rank: int, dp_local_rank: int | None = None
    ) -> bool:
        return False

    _installed = True
