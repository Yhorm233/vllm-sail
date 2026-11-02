# SPDX-License-Identifier: Apache-2.0
"""Engram construction and loading tests using CPU tensors."""

import types

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture
def modules(monkeypatch):
    # Shared conftest may have probed devices; this test must add no device operations.
    initialized_before = torch.cuda.is_initialized()

    def forbidden(*args, **kwargs):
        raise RuntimeError(
            "Device initialization and probing are forbidden in this CPU test"
        )

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(torch._C, "_cuda_getDeviceCount", forbidden)
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: types.SimpleNamespace(multi_processor_count=24),
    )
    from vllm.models.deepseek_v41.common import engram as common
    from vllm.models.deepseek_v41.nvidia import engram as upstream

    from vllm_sail.models.deepseek_v41 import engram as plugin

    yield common, upstream, plugin
    assert torch.cuda.is_initialized() == initialized_before


def test_packed_storage_and_exact_head_loader(modules):
    plugin = modules[2]

    class Storage(plugin.Int8DPSharedEngramStorage):
        def _allocate(self, num_bytes):
            self.requested_bytes = num_bytes
            return torch.full((num_bytes,), 193, dtype=torch.uint8)

    storage = Storage(3, 256, 32, types.SimpleNamespace())
    assert storage.requested_bytes == 3 * 260
    assert storage.weight.dtype == torch.int8
    assert storage.weight_scale_inv.dtype == torch.float32
    assert storage.weight_scale_inv.data_ptr() - storage.weight.data_ptr() == 3 * 256
    for target in (storage.weight, storage.weight_scale_inv):
        target.engram_vocab_start = 2
        loaded = (
            torch.arange(7 * target.shape[1])
            .reshape(7, target.shape[1])
            .to(target.dtype)
        )
        plugin.load_weight(target, loaded)
        assert torch.equal(target, loaded[2:5])
        before = target.clone()
        for invalid in (loaded.double(), loaded[:, :0], loaded[:4]):
            with pytest.raises(ValueError):
                plugin.load_weight(target, invalid)
            assert torch.equal(target, before)
    with pytest.raises(ValueError, match="four-byte aligned"):
        Storage(1, 3, 32, types.SimpleNamespace())


@pytest.mark.parametrize("rank", [0, 1])
def test_shared_loader_single_writer_barrier(modules, monkeypatch, rank):
    plugin = modules[2]
    storage = object.__new__(plugin.Int8DPSharedEngramStorage)
    storage.group = types.SimpleNamespace(rank_in_group=rank, cpu_group=object())
    barriers = []
    monkeypatch.setattr(torch.distributed, "barrier", lambda **kw: barriers.append(kw))
    target = torch.full((2, 1), -99.0)
    target.engram_vocab_start = 1
    loaded = torch.arange(4).float().view(4, 1)
    storage.load_weight(target, loaded)
    expected = loaded[1:3] if rank == 0 else torch.full_like(target, -99.0)
    assert torch.equal(target, expected)
    assert barriers == [{"group": storage.group.cpu_group}]


@pytest.mark.parametrize("shared", [False, True])
def test_real_constructor_attributes_and_tp_ownership(modules, monkeypatch, shared):
    common, upstream, plugin = modules
    monkeypatch.setattr(common, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(common, "get_tensor_model_parallel_rank", lambda: 1)
    # The upstream constructor retains ownership of DP sharding and UVA prerequisites.
    monkeypatch.setattr(upstream, "get_engram_dp_size", lambda: 16 if shared else 1)
    monkeypatch.setattr(upstream, "is_uva_available", lambda: True)
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: types.SimpleNamespace(multi_processor_count=24),
    )
    group = types.SimpleNamespace(rank_in_group=0, cpu_group=object())
    monkeypatch.setattr(plugin, "get_engram_dp_group", lambda: group)
    # Mock only allocation; execute the parent/child constructors and set_weight_attrs.
    monkeypatch.setattr(
        plugin.Int8DPSharedEngramStorage,
        "_allocate",
        lambda _, size: torch.empty(size, dtype=torch.uint8),
    )
    monkeypatch.setattr(
        plugin,
        "_allocate_huge_page_storage",
        lambda size: torch.empty(size, dtype=torch.uint8),
    )
    obj = plugin.Int8ParallelEngramEmbedding(
        17, 256, (2, 3, 5, 7), cpu_offload=True, dp_shared_memory=shared, use_thp=True
    )
    assert (obj.head_start, obj.part_n_hash_cols) == (2, 2)
    assert (obj.vocab_start_idx, obj.vocab_end_idx) == (5, 17)
    assert obj.dp_size == 1
    assert obj.weight.shape == (12, 256)
    assert obj.weight_scale_inv.shape == (12, 1)
    assert obj.weight.nbytes + obj.weight_scale_inv.nbytes == 12 * 260
    for target in (obj.weight, obj.weight_scale_inv):
        assert target.dummy_weight_value == 1.0
        assert target.engram_vocab_start == 5
        assert callable(target.weight_loader)
    if shared:
        assert obj._shared_memory is not None
        assert obj.weight.data_ptr() == obj._shared_memory.weight.data_ptr()
        assert obj.weight_scale_inv.weight_loader.__self__ is obj._shared_memory
    else:
        assert obj._packed is not None
        assert obj.weight_scale_inv.weight_loader is plugin.load_weight
    assert obj.weight.device.type == obj.weight_scale_inv.device.type == "cpu"


def test_constructor_rejects_invalid_shared_and_uva_settings(modules, monkeypatch):
    _, upstream, plugin = modules
    monkeypatch.setattr(upstream, "get_engram_dp_size", lambda: 1)
    with pytest.raises(ValueError, match="cpu_offload"):
        plugin.Int8ParallelEngramEmbedding(2, 256, (2,), dp_shared_memory=True)
    with pytest.raises(ValueError, match="size > 1"):
        plugin.Int8ParallelEngramEmbedding(
            2, 256, (2,), cpu_offload=True, dp_shared_memory=True
        )
    monkeypatch.setattr(upstream, "is_uva_available", lambda: False)
    with pytest.raises(RuntimeError, match="UVA"):
        plugin.Int8ParallelEngramEmbedding(2, 256, (2,), cpu_offload=True)
    monkeypatch.setattr(upstream, "is_uva_available", lambda: True)
    monkeypatch.setattr(upstream, "get_engram_dp_size", lambda: 2)
    monkeypatch.setattr(plugin, "get_engram_dp_group", lambda: None)
    common = modules[0]
    monkeypatch.setattr(common, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(common, "get_tensor_model_parallel_rank", lambda: 0)
    with pytest.raises(RuntimeError, match="node-local shared group"):
        plugin.Int8ParallelEngramEmbedding(
            2, 256, (2,), cpu_offload=True, dp_shared_memory=True
        )
