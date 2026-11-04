# SPDX-License-Identifier: Apache-2.0
"""Channelwise Engram storage with upstream sharding, UVA and prefetch."""

import os
import shutil

import torch
from vllm.distributed import get_engram_dp_group, get_engram_dp_size
from vllm.distributed.device_communicators.shm_broadcast import SHM_PATH
from vllm.models.deepseek_v41.common.engram import _engram_head_shard_weight_loader
from vllm.models.deepseek_v41.nvidia.engram import (
    DPSharedEngramStorage,
    ParallelEngramEmbedding,
    _allocate_huge_page_storage,
)


def validate_weight(param, loaded_weight):
    """Reject implicit casts or incomplete shards of INT8 tables and FP32 scales."""
    if (
        param.dtype not in (torch.int8, torch.float32)
        or loaded_weight.dtype != param.dtype
    ):
        raise ValueError(
            "1017 Engram requires INT8 tables or FP32 row scales; implicit casting is forbidden"
        )
    if (
        param.ndim != 2
        or loaded_weight.ndim != 2
        or loaded_weight.shape[1:] != param.shape[1:]
    ):
        raise ValueError("1017 Engram weight or scale shape does not match the shard")
    start = param.engram_vocab_start
    if start < 0 or start + param.shape[0] > loaded_weight.shape[0]:
        raise ValueError(
            "1017 Engram weights do not contain all rows of the head shard"
        )


def load_weight(param, loaded_weight):
    validate_weight(param, loaded_weight)
    _engram_head_shard_weight_loader(param, loaded_weight)


def can_share_engram_tables(layout, block_size=32):
    if not os.path.isdir(SHM_PATH) or get_engram_dp_size() <= 1:
        return False
    num_bytes = sum(layout.num_embeddings) * (layout.head_dim + 4)
    margin = max(8 * 1024**3, (num_bytes + 9) // 10)
    if shutil.disk_usage(SHM_PATH).free < num_bytes + margin:
        return False
    return True


class Int8DPSharedEngramStorage(DPSharedEngramStorage):
    def __init__(self, num_rows, dim, block_size, group):
        self.group = group
        weight_bytes = num_rows * dim
        if weight_bytes % 4:
            raise ValueError(
                "The INT8 Engram FP32 scale base address must be four-byte aligned"
            )
        storage = self._allocate(weight_bytes + num_rows * 4)
        self.weight = storage[:weight_bytes].view(torch.int8).view(num_rows, dim)
        self.weight_scale_inv = (
            storage[weight_bytes:].view(torch.float32).view(num_rows, 1)
        )
        self._views = None

    def load_weight(self, param, loaded_weight):
        validate_weight(param, loaded_weight)
        super().load_weight(param, loaded_weight)


class Int8ParallelEngramEmbedding(ParallelEngramEmbedding):
    """Replace only INT8/FP32 storage; retain upstream setup, sharding, UVA and prefetch."""

    _weight_loader = staticmethod(load_weight)

    def __init__(
        self,
        num_embeddings,
        dim,
        head_sizes,
        block_size=32,
        cpu_offload=False,
        dp_shared_memory=False,
        use_thp=False,
    ):
        super().__init__(
            num_embeddings,
            dim,
            head_sizes,
            block_size=dim,
            cpu_offload=cpu_offload,
            dp_shared_memory=dp_shared_memory,
            use_thp=use_thp,
        )
        self.weight_scale_inv.dummy_weight_value = 1.0

    def _allocate_weights(self):
        rows, dim = self.part_num_embeddings, self.dim
        if self.dp_shared_memory:
            group = get_engram_dp_group()
            if group is None:
                raise RuntimeError("Engram requires a node-local shared group")
            storage = Int8DPSharedEngramStorage(rows, dim, self.block_size, group)
            self._shared_memory = storage
            self._weight_loader = storage.load_weight
            return storage.weight, storage.weight_scale_inv
        if self.cpu_offload and self.use_thp:
            weight_bytes = rows * dim
            if weight_bytes % 4:
                raise ValueError("INT8 Engram scale storage must be four-byte aligned")
            packed = _allocate_huge_page_storage(weight_bytes + rows * 4)
            if packed is not None:
                self._packed = packed
                return (
                    packed[:weight_bytes].view(torch.int8).view(rows, dim),
                    packed[weight_bytes:].view(torch.float32).view(rows, 1),
                )
        placement = {"device": "cpu", "pin_memory": True} if self.cpu_offload else {}
        return (
            torch.empty(rows, dim, dtype=torch.int8, **placement),
            torch.empty(rows, 1, dtype=torch.float32, **placement),
        )


class ChannelwiseEngramEmbedding(ParallelEngramEmbedding):
    """FP8 embedding values and one FP32 scale per row, with no requantization."""

    def __init__(
        self,
        num_embeddings,
        dim,
        head_sizes,
        *,
        cpu_offload=False,
        dp_shared_memory=False,
    ):
        if dp_shared_memory:
            raise ValueError(
                "Channelwise Engram does not support dp_shared_memory; "
                "use cpu_offload without shared DP storage."
            )
        super().__init__(
            num_embeddings,
            dim,
            head_sizes,
            block_size=dim,
            cpu_offload=cpu_offload,
            dp_shared_memory=False,
        )
        # Keep the upstream parameter name and head-shard loader. The PPU
        # lookup distinguishes FP32 scales from exponent bytes by tensor dtype.
        self.weight_scale_inv.dummy_weight_value = 1.0

    def _allocate_weights(self):
        placement = {"device": "cpu", "pin_memory": True} if self.cpu_offload else {}
        return (
            torch.empty(
                self.part_num_embeddings,
                self.dim,
                dtype=torch.float8_e4m3fn,
                **placement,
            ),
            torch.empty(
                self.part_num_embeddings,
                1,
                dtype=torch.float32,
                **placement,
            ),
        )
