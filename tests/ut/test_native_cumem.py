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
        SimpleNamespace(DEFERRABLE_TAGS=("weights", "kv_cache")),
    )
    monkeypatch.setitem(
        sys.modules, "vllm.device_allocator.cumem", SimpleNamespace(PIN_MEMORY=False)
    )

    class Base:
        default_tag = "default"

        def __init__(self):
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
