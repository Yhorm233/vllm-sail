# SPDX-License-Identifier: Apache-2.0
"""Bind vLLM 0.33's tagged pools to native HGGC virtual memory."""

from __future__ import annotations

import atexit
import gc
import importlib
import logging

logger = logging.getLogger(__name__)


def load_extension():
    return importlib.import_module("vllm_sail._cumem_allocator")


def is_available() -> bool:
    try:
        load_extension()
        from torch.cuda import memory
    except (ImportError, OSError) as error:
        logger.warning("Native HGGC allocator unavailable: %s", error)
        return False
    return all(
        hasattr(memory, name)
        for name in ("CUDAPluggableAllocator", "MemPool", "use_mem_pool")
    )


def get_pluggable_allocator(allocate, free):
    from torch.cuda.memory import CUDAPluggableAllocator

    native = load_extension()
    native.init_module(allocate, free)
    return CUDAPluggableAllocator(native.__file__, "my_malloc", "my_free")


def get_instance():
    from vllm.device_allocator.cumem import CuMemAllocator

    if CuMemAllocator.instance is None:
        native = load_extension()
        CuMemAllocator.instance = _create_allocator(CuMemAllocator, native)
        atexit.register(CuMemAllocator._shutdown_singleton)
    return CuMemAllocator.instance


def _create_allocator(base, native):
    import torch
    from vllm.device_allocator import DEFERRABLE_TAGS
    from vllm.device_allocator.cumem import PIN_MEMORY

    class HggcAllocator(base):
        def _python_free_callback(self, ptr):
            # PyTorch also frees segments whose Python records were trimmed.
            self.pointer_to_data.pop(ptr, None)

        def _suspend(self, ptr, data, offload):
            if data.is_asleep:
                # Retry release of any handle left after a successful unmap.
                native.suspend(ptr, 0)
                return

            backup = None
            if offload:
                backup = torch.empty(
                    data.handle[1],
                    dtype=torch.uint8,
                    device="cpu",
                    pin_memory=PIN_MEMORY,
                )
            try:
                native.suspend(ptr, backup.data_ptr() if backup is not None else 0)
            finally:
                if not native.is_mapped(ptr):
                    data.cpu_backup_tensor = backup
                    data.is_asleep = True

        def sleep(self, offload_tags=None):
            if offload_tags is None:
                offload_tags = (self.default_tag,)
            elif isinstance(offload_tags, str):
                offload_tags = (offload_tags,)
            for ptr in tuple(self.pointer_to_data):
                if (data := self.pointer_to_data.get(ptr)) is not None:
                    self._suspend(ptr, data, data.tag in offload_tags)
            gc.collect()
            torch.accelerator.empty_cache()

        def discard(self, tags):
            if isinstance(tags, str):
                tags = (tags,)
            for ptr, data in self.pointer_to_data.items():
                if data.tag in tags:
                    self._suspend(ptr, data, False)

        def wake_up(self, tags=None):
            gc.collect()
            torch.accelerator.empty_cache()
            for ptr, data in self.pointer_to_data.items():
                if not data.is_asleep:
                    continue
                if (
                    tags is not None
                    and data.tag in DEFERRABLE_TAGS
                    and data.tag not in tags
                ):
                    continue
                backup = data.cpu_backup_tensor
                native.resume(ptr, backup.data_ptr() if backup is not None else 0)
                data.cpu_backup_tensor = None
                data.is_asleep = False

        def _trim(self, pool):
            for allocation in pool.snapshot():
                ptr = allocation["address"]
                if allocation["allocated_size"] == 0 and ptr in self.pointer_to_data:
                    native.suspend(ptr, 0)
                    del self.pointer_to_data[ptr]

    return HggcAllocator()
