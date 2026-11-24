# SPDX-License-Identifier: Apache-2.0
"""Lazy access to the external ``msa_for_sail`` JIT kernels.

``msa_for_sail`` compiles its hgcc variants on first use and is not installed
in every PPU environment, so nothing here imports it at module scope. A missing
package disables the SAIL MSA path (the upstream Triton implementation keeps
working) instead of failing plugin import.

The JIT cache (``~/.cache/minfer/msa_for_sail``, or ``MINFER_FMHA_CACHE_DIR``)
is shared by every TP worker. ``msa_for_sail`` skips a target once its ``.so``
exists but does not guard the *writing*: when all workers reach a target at the
same time (first warmup batch of a tensor-parallel run), one worker can load a
half-written ``.so`` and fail with ``Module has no function ...``.
Guard the JIT module getters used by msa_for_sail.api instead of the
operators that call them. The lock covers building and loading a complete
artifact, never planning or executing a kernel on a TP worker.
"""

from __future__ import annotations

import fcntl
import functools
import os
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

_LOCK_NAME = ".vllm_sail_msa_compile.lock"
# msa_for_sail.api imports these symbols from jit by value. They return modules
# or callable handles, before api.py launches any work on the device.
_MODULE_GETTERS = (
    "get_plan_fn",
    "get_prepare_metadata_fn",
    "get_fmha_variant",
    "get_reduction_module",
    "get_sparse_topk_module",
)


@dataclass(frozen=True)
class SailMSAOps:
    fmha_sail_plan: Callable[..., Any]
    fmha_sail: Callable[..., Any]
    sparse_topk_select: Callable[..., Any]


def _cache_base() -> Path:
    """Directory ``msa_for_sail`` builds into (``jit.CACHE_BASE``)."""
    try:
        from msa_for_sail.jit import CACHE_BASE

        return Path(CACHE_BASE)
    except Exception:
        explicit = os.environ.get("MINFER_FMHA_CACHE_DIR")
        if explicit:
            return Path(explicit)
        return Path(os.path.expanduser("~/.cache/minfer/msa_for_sail"))


@contextmanager
def _compile_lock():
    path = _cache_base() / _LOCK_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            logger.info(
                "Waiting for another worker to finish an msa_for_sail JIT "
                "build (lock: %s)",
                path,
            )
            fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _guard_module_getter(getter):
    # Only the first lookup per process and actual JIT variant needs flock.
    # A cache hit on disk still needs the lock: the linker may be writing it.
    @functools.cache
    @functools.wraps(getter)
    def load(*args, **kwargs):
        with _compile_lock():
            return getter(*args, **kwargs)

    return load


@functools.cache
def sail_msa_ops() -> SailMSAOps | None:
    """The ``msa_for_sail`` entry points, or ``None`` when unusable."""
    try:
        from msa_for_sail import api, fmha_sail, fmha_sail_plan, sparse_topk_select
    except ImportError:
        logger.warning_once(
            "MiniMax M3 SAIL MSA is available on this PPU but the "
            "`msa_for_sail` package is not importable; sparse attention stays "
            "on the Triton implementation. Install MSA-for-sail "
            "(`pip install -e python/`) to enable it."
        )
        return None
    if not all(callable(getattr(api, name, None)) for name in _MODULE_GETTERS):
        logger.warning_once(
            "Unsupported msa_for_sail JIT interface; using Triton sparse attention."
        )
        return None
    for name in _MODULE_GETTERS:
        getter = getattr(api, name)
        if not getattr(getter, "_sail_compile_guard", False):
            guarded = _guard_module_getter(getter)
            guarded._sail_compile_guard = True
            setattr(api, name, guarded)
    return SailMSAOps(fmha_sail_plan, fmha_sail, sparse_topk_select)


def sail_msa_available() -> bool:
    return sail_msa_ops() is not None


def require_sail_msa_ops() -> SailMSAOps:
    """The entry points for a backend the gating policy already selected."""
    ops = sail_msa_ops()
    if ops is None:
        raise RuntimeError(
            "MiniMax M3 SAIL MSA was selected but `msa_for_sail` is not importable"
        )
    return ops
