# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""INT8/FP32 Engram for 1017 W8A8, reusing upstream head sharding and shared mmap/UVA."""

import os
import shutil

import torch
from vllm.distributed import get_engram_dp_group, get_engram_dp_size
from vllm.distributed.device_communicators.shm_broadcast import SHM_PATH
from vllm.logger import init_logger
from vllm.model_executor.utils import set_weight_attrs
from vllm.models.deepseek_v41.common.engram import (
    ParallelEngramEmbedding as BaseParallelEngramEmbedding,
)
from vllm.models.deepseek_v41.common.engram import (
    _engram_head_shard_weight_loader,
)
from vllm.models.deepseek_v41.nvidia.engram import (
    DPSharedEngramStorage,
    ParallelEngramEmbedding,
    _allocate_huge_page_storage,
)
from vllm.utils.platform_utils import is_uva_available

logger = init_logger(__name__)


def validate_weight(param, loaded_weight):
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
        raise ValueError("1017 Engram weights do not contain all rows of the head shard")


def load_weight(param, loaded_weight):
    validate_weight(param, loaded_weight)
    _engram_head_shard_weight_loader(param, loaded_weight)


def can_share_engram_tables(layout, block_size=32):
    if not os.path.isdir(SHM_PATH) or get_engram_dp_size() <= 1:
        raise RuntimeError("810E Engram requires node-local DP sharing; fallback to per-rank full tables is forbidden")
    num_bytes = sum(layout.num_embeddings) * (layout.head_dim + 4)
    margin = max(8 * 1024**3, (num_bytes + 9) // 10)
    if shutil.disk_usage(SHM_PATH).free < num_bytes + margin:
        raise RuntimeError("Engram shared storage has insufficient capacity (including a margin of at least 10% or 8 GiB, whichever is larger)")
    return True


class Int8DPSharedEngramStorage(DPSharedEngramStorage):
    def __init__(self, num_rows, dim, block_size, group):
        self.group = group
        weight_bytes = num_rows * dim
        if weight_bytes % 4:
            raise ValueError("The INT8 Engram FP32 scale base address must be four-byte aligned")
        storage = self._allocate(weight_bytes + num_rows * 4)
        self.weight = storage[:weight_bytes].view(torch.int8).view(num_rows, dim)
        # Keep the upstream parameter name; it stores FP32 row scales, not UE8M0.
        self.weight_scale_inv = (
            storage[weight_bytes:].view(torch.float32).view(num_rows, 1)
        )
        self._views = None

    def load_weight(self, param, loaded_weight):
        validate_weight(param, loaded_weight)
        super().load_weight(param, loaded_weight)


class Int8ParallelEngramEmbedding(ParallelEngramEmbedding):
    _weight_loader = staticmethod(load_weight)

    # Constructor adapted from vLLM db9527a46873454610df6dbedf79a36d6bf1a7f6
    # models/deepseek_v41/nvidia/engram.py; preserve UVA/DP precondition checks.
    def __init__(
        self,
        num_embeddings: int,
        dim: int,
        head_sizes: tuple[int, ...],
        block_size: int = 32,
        cpu_offload: bool = False,
        dp_shared_memory: bool = False,
        use_thp: bool = False,
    ) -> None:
        self.cpu_offload = cpu_offload
        self.dp_shared_memory = dp_shared_memory
        self.use_thp = use_thp
        self._packed: torch.Tensor | None = None
        self.dp_size = get_engram_dp_size()
        if dp_shared_memory:
            if not cpu_offload:
                raise ValueError("dp_shared_memory requires cpu_offload=True")
            if self.dp_size <= 1:
                raise ValueError(
                    "dp_shared_memory requires a node-local Engram DP "
                    f"group with size > 1; effective Engram DP size is {self.dp_size}. "
                    "Check that the node layout and rank placement allow complete "
                    "DP replicas to be co-located."
                )
            self.dp_size = 1
        if cpu_offload and not is_uva_available():
            raise RuntimeError("Engram CPU offload requires UVA support")
        self._views: tuple[torch.Tensor, torch.Tensor] | None = None
        self._view_src: tuple[int, int] | None = None
        BaseParallelEngramEmbedding.__init__(
            self, num_embeddings, dim, head_sizes, block_size
        )
        if cpu_offload:
            # Constant dummy values avoid randomizing huge CPU lookup tables.
            set_weight_attrs(self.weight, {"dummy_weight_value": 1.0})
            set_weight_attrs(self.weight_scale_inv, {"dummy_weight_value": 1.0})
            logger.info(
                "INT8/FP32 Engram offloaded to pinned host memory: %d rows x %d, "
                "%d bytes (%.2f GiB), %s",
                self.part_num_embeddings,
                dim,
                self.weight.nbytes + self.weight_scale_inv.nbytes,
                (self.weight.nbytes + self.weight_scale_inv.nbytes) / 1024**3,
                "DP shared" if dp_shared_memory else "per-rank head shard",
            )

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
            packed = _allocate_huge_page_storage(rows * (dim + 4))
            if packed is not None:
                self._packed = packed
                return (
                    packed[: rows * dim].view(torch.int8).view(rows, dim),
                    packed[rows * dim :].view(torch.float32).view(rows, 1),
                )
        options = dict(device="cpu", pin_memory=True) if self.cpu_offload else {}
        return (
            torch.empty(rows, dim, dtype=torch.int8, **options),
            torch.empty(rows, 1, dtype=torch.float32, **options),
        )
