# SPDX-License-Identifier: Apache-2.0
"""Single-device checks for the installed HGGC allocator wheel."""

import gc
import unittest
from importlib import metadata

MIB = 1 << 20
SEGMENT = 16 * MIB


class TestHggcAllocator(unittest.TestCase):
    def setUp(self):
        import torch

        if torch.cuda.device_count() != 1:
            self.skipTest("requires exactly one visible accelerator")
        self.torch = torch
        torch.set_num_threads(1)
        torch.cuda.set_device(0)
        torch.cuda.synchronize()

    def test_native_segments(self):
        from vllm_sail.native import cumem

        torch = self.torch
        native = cumem.load_extension()
        live = {}

        def allocate(handle):
            live[handle[2]] = handle

        allocator = cumem.get_pluggable_allocator(allocate, live.pop)
        pool = torch.cuda.memory.MemPool(allocator.allocator())
        with torch.cuda.memory.use_mem_pool(pool):
            first = torch.full((SEGMENT,), 0x39, dtype=torch.uint8, device="cuda")
            second = torch.full((SEGMENT,), 0xA7, dtype=torch.uint8, device="cuda")
        first_ptr = first.data_ptr()
        second_ptr = second.data_ptr()
        self.assertEqual(set(live), {first_ptr, second_ptr})
        saved = first.cpu()
        backup = torch.empty(live[first_ptr][1], dtype=torch.uint8, device="cpu")
        before = torch.cuda.mem_get_info()[0]
        native.suspend(first_ptr, backup.data_ptr())
        native.suspend(second_ptr, 0)
        asleep = torch.cuda.mem_get_info()[0]
        print(f"HGGC segments={len(live)} recovered_bytes={asleep - before}")
        self.assertGreaterEqual(
            asleep - before, sum(h[1] for h in live.values()) - 2 * MIB
        )
        self.assertFalse(native.is_mapped(first_ptr))
        self.assertFalse(native.is_mapped(second_ptr))

        native.resume(first_ptr, backup.data_ptr())
        self.assertEqual(first.data_ptr(), first_ptr)
        self.assertTrue(torch.equal(first.cpu(), saved))
        self.assertFalse(native.is_mapped(second_ptr))

        del second, first, pool
        gc.collect()
        self.assertEqual(live, {})

    @unittest.skipUnless(
        any(dist.metadata["Name"] == "vllm" for dist in metadata.distributions()),
        "requires the vLLM 0.34 wheel",
    )
    def test_plugin_backend_suspend_and_selective_resume(self):
        from vllm.plugins import load_general_plugins

        load_general_plugins()

        from vllm.device_allocator import get_mem_allocator_instance
        from vllm.device_allocator.sleep_mode_backend import CuMemBackend
        from vllm.platforms import current_platform

        self.assertTrue(current_platform.is_ppu())
        self.assertTrue(current_platform.is_cumem_allocator_available())
        torch = self.torch
        allocator = get_mem_allocator_instance()
        backend = CuMemBackend()
        with allocator.use_memory_pool("workspace"):
            transient = torch.empty((SEGMENT,), dtype=torch.uint8, device="cuda")
            self.assertIn(transient.data_ptr(), allocator.pointer_to_data)
            del transient
        self.assertEqual(allocator.get_current_usage(), 0)
        with allocator.use_memory_pool("weights"):
            weights = torch.full((SEGMENT,), 0x39, dtype=torch.uint8, device="cuda")
        with allocator.use_memory_pool("kv_cache"):
            cache = torch.empty((SEGMENT,), dtype=torch.uint8, device="cuda")
        weight_ptr = weights.data_ptr()
        cache_ptr = cache.data_ptr()
        expected = weights.cpu()
        before = torch.cuda.mem_get_info()[0]
        backend.suspend()
        asleep = torch.cuda.mem_get_info()[0]
        print(f"vLLM Level 1 recovered_bytes={asleep - before}")
        self.assertGreaterEqual(asleep - before, 2 * SEGMENT - 2 * MIB)
        backend.resume(tags=["weights"])
        self.assertEqual(weights.data_ptr(), weight_ptr)
        self.assertTrue(torch.equal(weights.cpu(), expected))
        self.assertTrue(allocator.pointer_to_data[cache_ptr].is_asleep)
        backend.resume(tags=["kv_cache"])
        self.assertEqual(cache.data_ptr(), cache_ptr)
        cache.fill_(1)
        self.assertEqual(int(cache[0]), 1)
        del weights, cache
        allocator.release_pools()
        self.assertEqual(allocator.get_current_usage(), 0)


if __name__ == "__main__":
    unittest.main()
