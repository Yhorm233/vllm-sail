# SPDX-License-Identifier: Apache-2.0
"""Check Python tag and backup bookkeeping without torch or vLLM installed."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_sail.native import cumem


@pytest.fixture
def allocator(monkeypatch):
    backup = SimpleNamespace(data_ptr=lambda: 11)
    torch = SimpleNamespace(
        uint8=object(),
        empty=Mock(return_value=backup),
        accelerator=SimpleNamespace(empty_cache=lambda: None),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(
        sys.modules,
        "vllm.device_allocator",
        SimpleNamespace(
            DEFERRABLE_TAGS=("weights", "kv_cache"),
            discardable_allocations=Mock(return_value=set()),
        ),
    )
    monkeypatch.setitem(
        sys.modules, "vllm.device_allocator.cumem", SimpleNamespace(PIN_MEMORY=False)
    )

    class Base:
        default_tag = "default"

        def __init__(self):
            self.allocator_and_pools = {}
            self.pointer_to_data = {
                tag: self._data(tag) for tag in ("weights", "kv_cache")
            }

        @staticmethod
        def _data(tag):
            return SimpleNamespace(
                tag=tag, handle=(0, 64, 0, 0), is_asleep=False, cpu_backup_tensor=None
            )

    native = Mock()
    native.is_mapped.return_value = False
    return cumem._create_allocator(Base, native), native, backup


def test_partial_sleep_failure_preserves_backup(allocator):
    pool, native, backup = allocator
    native.suspend.side_effect = RuntimeError("release failed after unmap")
    with pytest.raises(RuntimeError, match="release failed"):
        pool.sleep("weights")
    weights = pool.pointer_to_data["weights"]
    assert weights.is_asleep
    assert weights.cpu_backup_tensor is backup
    assert not pool.pointer_to_data["kv_cache"].is_asleep


def test_failed_wake_retries_with_same_backup(allocator):
    pool, native, backup = allocator
    pool.sleep("weights")
    native.resume.side_effect = [RuntimeError("copy failed"), None]
    with pytest.raises(RuntimeError, match="copy failed"):
        pool.wake_up(["weights"])
    weights = pool.pointer_to_data["weights"]
    assert weights.is_asleep
    assert weights.cpu_backup_tensor is backup
    pool.wake_up(["weights"])
    assert not weights.is_asleep
    assert weights.cpu_backup_tensor is None
    assert pool.pointer_to_data["kv_cache"].is_asleep
    assert [call.args for call in native.resume.call_args_list] == [
        ("weights", 11),
        ("weights", 11),
    ]


def test_sleep_skips_backup_for_discardable_weight_segments(allocator):
    pool, native, backup = allocator
    pool.pointer_to_data[100] = pool._data("weights")
    discardable = {200}
    find_discardable = sys.modules["vllm.device_allocator"].discardable_allocations
    find_discardable.return_value = {100}

    pool.sleep(("weights",), discardable)

    find_discardable.assert_called_once_with(pool.allocator_and_pools, discardable)
    native.suspend.assert_any_call(100, 0)
    assert pool.pointer_to_data[100].is_asleep
    assert pool.pointer_to_data[100].cpu_backup_tensor is None
    assert pool.pointer_to_data["weights"].cpu_backup_tensor is backup


def test_pluggable_allocator_hook_uses_hggc_callbacks(allocator, monkeypatch):
    pool, _, _ = allocator
    pool.python_malloc_callback = Mock()
    pool.python_free_callback = Mock()
    allocator_factory = Mock()
    monkeypatch.setattr(cumem, "get_pluggable_allocator", allocator_factory)

    assert pool._get_pluggable_allocator() is allocator_factory.return_value
    allocator_factory.assert_called_once_with(
        pool.python_malloc_callback, pool.python_free_callback
    )
