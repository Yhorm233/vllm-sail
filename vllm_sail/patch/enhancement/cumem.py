# SPDX-License-Identifier: Apache-2.0
"""Select native HGGC bindings at the vLLM allocator entry points."""

from vllm_sail.patch.utils import patch

_MODULE = "vllm.device_allocator.cumem"
_PATCH = {
    "reason": "vLLM's empty wheel needs native HGGC pluggable allocation on PPU.",
    "affected_versions": ">=0.34.0,<0.35.0",
    "remove_when": "vLLM provides a platform hook for tagged memory allocators.",
}


def install():
    from vllm.platforms import current_platform

    if not current_platform.is_ppu():
        return

    from vllm.v1.executor.abstract import SLEEP_TAGS, Executor

    from vllm_sail.native import cumem as hggc

    original_sleep = Executor.sleep

    @patch(_MODULE, "CuMemAllocator.get_instance", **_PATCH)
    def get_instance():
        return hggc.get_instance()

    @patch(_MODULE, "get_pluggable_allocator", **_PATCH)
    def get_pluggable_allocator(python_malloc_fn, python_free_func):
        return hggc.get_pluggable_allocator(python_malloc_fn, python_free_func)

    @patch(
        "vllm.v1.executor.abstract",
        "Executor.sleep",
        reason="A partially suspended PPU must wake its pools before serving again.",
        affected_versions=_PATCH["affected_versions"],
        remove_when="Executor.sleep records sleeping tags when the worker RPC fails.",
    )
    def sleep(self, level=1):
        try:
            return original_sleep(self, level)
        except Exception:
            # A worker can fail after dropping some backing pages. Keep the
            # scheduler paused until a real worker wake RPC succeeds.
            self.sleeping_tags |= SLEEP_TAGS
            raise
