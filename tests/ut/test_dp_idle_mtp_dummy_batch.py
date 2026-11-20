# SPDX-License-Identifier: Apache-2.0
"""DP idle-replica dummy batches keep the agreed MTP/EP query length."""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from tests.support.upstream import SUPPORTED_RANGE
from vllm_sail.patch.utils import PATCH_REGISTRY, patch

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

MODULE_PATH = (
    Path(__file__).parents[2] / "vllm_sail/patch/bugfix/dp_idle_mtp_dummy_batch.py"
)

INPUT_BATCH_TARGET = "vllm.v1.worker.gpu.input_batch.InputBatch.make_dummy"
SET_DUMMY_CONTEXT_TARGET = "vllm.v1.worker.gpu.input_batch.set_dummy_context"
PREPARE_DUMMY_ATTN_TARGET = (
    "vllm.v1.worker.gpu.model_runner.GPUModelRunner.prepare_dummy_attn"
)
DISPATCH_CG_TARGET = "vllm.v1.worker.gpu.model_runner.dispatch_cg_and_sync_dp"
EXECUTE_MODEL_TARGET = "vllm.v1.worker.gpu.model_runner.GPUModelRunner.execute_model"


@pytest.fixture
def dummy_batch_module():
    spec = importlib.util.spec_from_file_location("_dp_dummy_batch_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    count = len(PATCH_REGISTRY)
    try:
        yield module
    finally:
        del PATCH_REGISTRY[count:]


def modules(monkeypatch, *, input_batch=None, model_runner=None, random_uuid=None):
    leaves = {
        "vllm.utils": NS(random_uuid=random_uuid),
        "vllm.utils.math_utils": NS(
            cdiv=lambda a, b: 0
        ),  # placeholder, overridden below
        "vllm.v1.worker.gpu.input_batch": input_batch or NS(),
        "vllm.v1.worker.gpu.model_runner": model_runner or NS(),
    }
    names = set(leaves)
    for name in leaves:
        parts = name.split(".")
        names.update(".".join(parts[:i]) for i in range(1, len(parts)))
    for name in sorted(names, key=lambda value: value.count(".")):
        module = types.ModuleType(name)
        module.__path__ = []
        if name in leaves:
            module.__dict__.update(vars(leaves[name]))
        monkeypatch.setitem(sys.modules, name, module)


def test_bugfix_and_leaf_import_without_installing(monkeypatch):
    import vllm_sail.patch.bugfix as bugfix

    def unexpected():
        pytest.fail("importing a cached category must not install patches")

    monkeypatch.setattr(bugfix, "install", unexpected)
    assert importlib.import_module("vllm_sail.patch.bugfix") is bugfix
    assert callable(bugfix.install)


def test_cached_category_is_explicitly_installed_once(monkeypatch):
    import vllm_sail.patch as package
    import vllm_sail.patch.bugfix as bugfix

    calls = []
    monkeypatch.setattr(bugfix, "install", lambda: calls.append("bugfix"))
    monkeypatch.setattr(package, "_CATEGORIES", ("bugfix",))
    monkeypatch.setattr(package, "_installed", False)
    package.install()
    package.install()
    assert calls == ["bugfix"]


def _fake_input_batch():
    class InputBatch:
        @classmethod
        def make_dummy(cls, *args, **kwargs):
            raise AssertionError("the original make_dummy must be replaced")

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    return InputBatch


def _fake_input_buffers(device, max_num_reqs, max_num_tokens):
    buffers = NS(
        device=device,
        seq_lens=torch.zeros(max_num_reqs, dtype=torch.int32, device=device),
        query_start_loc=torch.zeros(max_num_reqs + 1, dtype=torch.int32, device=device),
        input_ids=torch.zeros(max_num_tokens, dtype=torch.int64, device=device),
        positions=torch.zeros(max_num_tokens, dtype=torch.int64, device=device),
        is_padding=torch.zeros(max_num_tokens, dtype=torch.bool, device=device),
    )
    return buffers


def test_make_dummy_preserves_uniform_query_length(monkeypatch, dummy_batch_module):
    """A varlen graph's request capacity must not repartition the agreed batch."""
    InputBatch = _fake_input_batch()
    buffers = _fake_input_buffers(torch.device("cpu"), 8, 8)

    modules(
        monkeypatch,
        input_batch=NS(InputBatch=InputBatch),
        random_uuid=lambda: "deadbeef",
    )

    # Patch make_dummy onto the fake class so the replacement body can call cls(...).
    dummy_batch_module._install_make_dummy()

    batch = InputBatch.make_dummy(8, 8, buffers, max_query_len=4, uniform_token_count=3)
    assert batch.num_reqs == 2
    assert batch.num_tokens == 6
    assert batch.num_reqs_after_padding == 8
    assert batch.num_tokens_after_padding == 8
    assert batch.seq_lens.tolist() == [3, 3, 0, 0, 0, 0, 0, 0]
    assert batch.query_start_loc.tolist() == [0, 3, 6, 6, 6, 6, 6, 6, 6]
    assert batch.logits_indices.tolist() == [2, 5]


def test_make_dummy_without_uniform_token_count_falls_back_to_even_split(
    monkeypatch, dummy_batch_module
):
    InputBatch = _fake_input_batch()
    buffers = _fake_input_buffers(torch.device("cpu"), 4, 4)

    modules(
        monkeypatch,
        input_batch=NS(InputBatch=InputBatch),
        random_uuid=lambda: "deadbeef",
    )
    dummy_batch_module._install_make_dummy()

    batch = InputBatch.make_dummy(4, 4, buffers, max_query_len=4)
    assert batch.num_reqs == 4
    assert batch.num_tokens == 4
    assert batch.num_reqs_after_padding == 4
    assert batch.num_tokens_after_padding == 4
    assert batch.seq_lens.tolist() == [1, 1, 1, 1]


def test_set_dummy_context_only_touches_active_rows(monkeypatch, dummy_batch_module):
    class BlockTables:
        def __init__(self):
            self.input_block_tables = [torch.zeros(8, 4, dtype=torch.int32)]
            self.block_sizes = [16]

    batch = NS(
        num_reqs=2,
        num_tokens=6,
        max_query_len=3,
        num_scheduled_tokens=np.array([3, 3], dtype=np.int32),
        query_start_loc_np=np.array([0, 3, 6, 8, 8, 8, 8, 8, 8], dtype=np.int32),
        seq_lens=torch.tensor([3, 3, 0, 0, 0, 0, 0, 0], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor(
            [3, 3, 0, 0, 0, 0, 0, 0], dtype=torch.int32
        ),
        num_computed_tokens_np=np.zeros(8, dtype=np.int32),
        num_computed_prefill_tokens_np=np.zeros(8, dtype=np.int32),
        positions=torch.zeros(8, dtype=torch.int64),
    )

    def cdiv(a, b):
        return -(-a // b)

    modules(
        monkeypatch,
        input_batch=NS(set_dummy_context=lambda *args, **kwargs: None),
    )
    monkeypatch.setattr(sys.modules["vllm.utils.math_utils"], "cdiv", cdiv)
    dummy_batch_module._install_set_dummy_context()

    installed = sys.modules["vllm.v1.worker.gpu.input_batch"].set_dummy_context
    installed(batch, BlockTables(), 16, 64, 4096)
    assert batch.seq_lens.tolist()[:2] == [19, 19]
    assert batch.seq_lens.tolist()[2:] == [0] * 6
    assert batch.seq_lens_cpu_upper_bound.tolist()[:2] == [19, 19]
    assert batch.seq_lens_cpu_upper_bound.tolist()[2:] == [0] * 6
    assert batch.num_computed_tokens_np.tolist()[:2] == [16, 16]
    assert batch.positions.tolist()[:6] == [16, 17, 18, 16, 17, 18]


def test_prepare_dummy_attn_uses_padded_capacity_for_metadata(
    monkeypatch, dummy_batch_module
):
    class Provider:
        def __init__(self):
            self.calls = []

        def get_dummy_block_tables(self, n):
            self.calls.append(("block_tables", n))
            return [torch.zeros(n, 4, dtype=torch.int32)]

        def get_dummy_slot_mappings(self, n):
            self.calls.append(("slot_mappings", n))
            return torch.zeros(n, dtype=torch.int64)

    class GPUModelRunner:
        def prepare_dummy_attn(self, input_batch, valid_state_slots=False):
            raise AssertionError("the original prepare_dummy_attn must be replaced")

    modules(
        monkeypatch,
        model_runner=NS(GPUModelRunner=GPUModelRunner),
    )
    dummy_batch_module._install_prepare_dummy_attn()

    provider = Provider()
    batch = NS(num_reqs=2, num_reqs_after_padding=8, num_tokens_after_padding=8)
    runner = GPUModelRunner()
    runner.device = torch.device("cpu")
    runner.pcp_manager = None
    runner.block_tables = provider

    block_tables, slot_mappings = runner.prepare_dummy_attn(batch)
    assert provider.calls == [("block_tables", 8), ("slot_mappings", 8)]
    assert block_tables[0].shape == (8, 4)
    assert slot_mappings.shape == (8,)


def test_dispatch_wrapper_records_uniform_token_count_for_dp(
    monkeypatch, dummy_batch_module
):
    calls = []

    class Sync:
        def __init__(self, value):
            self.uniform_token_count = value

    def original(*args, **kwargs):
        calls.append(("original", args, kwargs))
        return (NS(num_tokens=4), Sync(3))

    modules(
        monkeypatch,
        model_runner=NS(dispatch_cg_and_sync_dp=original),
    )
    dummy_batch_module._install_dispatch_wrapper()
    from vllm.v1.worker.gpu import model_runner as mr

    result = mr.dispatch_cg_and_sync_dp(NS(), 1, 4, 3, 4, dp_size=2, dp_rank=1)
    assert result[1].uniform_token_count == 3
    assert dummy_batch_module._get_uniform_token_count() == 3


def test_dispatch_wrapper_clears_uniform_token_count_for_single_rank(
    monkeypatch, dummy_batch_module
):
    """dp_size == 1 returns sync=None; the local vote must not be recorded."""

    def original(*args, **kwargs):
        return (NS(num_tokens=4), None)

    modules(
        monkeypatch,
        model_runner=NS(dispatch_cg_and_sync_dp=original),
    )
    dummy_batch_module._set_uniform_token_count(7)
    dummy_batch_module._install_dispatch_wrapper()
    from vllm.v1.worker.gpu import model_runner as mr

    result = mr.dispatch_cg_and_sync_dp(NS(), 1, 4, 3, 4, dp_size=1, dp_rank=0)
    assert result[1] is None
    assert dummy_batch_module._get_uniform_token_count() is None


def test_uniform_token_count_does_not_leak_out_of_execute_model(
    monkeypatch, dummy_batch_module
):
    """The profile sampler builds its own dummy batch; it must not inherit
    the execute_model dummy batch's agreed uniform_token_count."""
    InputBatch = _fake_input_batch()

    class Sync:
        uniform_token_count = 4

    def dispatch_cg_and_sync_dp(*args, **kwargs):
        return NS(num_tokens=8), Sync()

    class GPUModelRunner:
        def execute_model(self):
            model_runner = sys.modules["vllm.v1.worker.gpu.model_runner"]
            model_runner.dispatch_cg_and_sync_dp(self, 2, 8, 4, 1, 0)
            return InputBatch.make_dummy(
                8, 8, _fake_input_buffers(torch.device("cpu"), 8, 8)
            )

    modules(
        monkeypatch,
        input_batch=NS(InputBatch=InputBatch),
        model_runner=NS(
            GPUModelRunner=GPUModelRunner,
            dispatch_cg_and_sync_dp=dispatch_cg_and_sync_dp,
        ),
        random_uuid=lambda: "deadbeef",
    )
    dummy_batch_module._install_make_dummy()
    dummy_batch_module._install_dispatch_wrapper()
    dummy_batch_module._install_execute_model_scope()

    inside = GPUModelRunner().execute_model()
    assert (inside.num_reqs, inside.num_tokens) == (2, 8)

    outside = InputBatch.make_dummy(
        4, 4, _fake_input_buffers(torch.device("cpu"), 4, 4)
    )
    assert (outside.num_reqs, outside.num_tokens) == (4, 4)
    assert outside.seq_lens.tolist() == [1, 1, 1, 1]


def test_uniform_token_count_is_ignored_for_single_rank(
    monkeypatch, dummy_batch_module
):
    """Without DP there is no agreed value; make_dummy must keep num_reqs."""
    InputBatch = _fake_input_batch()

    def dispatch_cg_and_sync_dp(*args, **kwargs):
        return NS(num_tokens=8), None

    class GPUModelRunner:
        def execute_model(self):
            model_runner = sys.modules["vllm.v1.worker.gpu.model_runner"]
            model_runner.dispatch_cg_and_sync_dp(self, 2, 8, 4, 1, 0)
            return InputBatch.make_dummy(
                8, 8, _fake_input_buffers(torch.device("cpu"), 8, 8)
            )

    modules(
        monkeypatch,
        input_batch=NS(InputBatch=InputBatch),
        model_runner=NS(
            GPUModelRunner=GPUModelRunner,
            dispatch_cg_and_sync_dp=dispatch_cg_and_sync_dp,
        ),
        random_uuid=lambda: "deadbeef",
    )
    dummy_batch_module._install_make_dummy()
    dummy_batch_module._install_dispatch_wrapper()
    dummy_batch_module._install_execute_model_scope()

    inside = GPUModelRunner().execute_model()
    assert (inside.num_reqs, inside.num_tokens) == (8, 8)
    assert inside.seq_lens.tolist() == [1, 1, 1, 1, 1, 1, 1, 1]


def test_install_records_real_targets_and_reapplication_guard(
    monkeypatch, dummy_batch_module
):
    class InputBatch:
        @classmethod
        def make_dummy(cls, *args, **kwargs):
            return None

    def set_dummy_context(*args, **kwargs):
        return None

    class GPUModelRunner:
        def prepare_dummy_attn(self, *args, **kwargs):
            return None

        def execute_model(self, *args, **kwargs):
            return None

    def dispatch_cg_and_sync_dp(*args, **kwargs):
        return None, None

    modules(
        monkeypatch,
        input_batch=NS(InputBatch=InputBatch, set_dummy_context=set_dummy_context),
        model_runner=NS(
            GPUModelRunner=GPUModelRunner,
            dispatch_cg_and_sync_dp=dispatch_cg_and_sync_dp,
        ),
        random_uuid=lambda: "deadbeef",
    )

    dummy_batch_module.install()

    # Classmethod patches store the marker on the underlying function.
    assert hasattr(InputBatch.make_dummy.__func__, "__vllm_sail_patch__")
    assert (
        InputBatch.make_dummy.__func__.__vllm_sail_patch__[INPUT_BATCH_TARGET]
        is not None
    )

    records = {r.target: r for r in PATCH_REGISTRY[-5:]}
    assert INPUT_BATCH_TARGET in records
    assert SET_DUMMY_CONTEXT_TARGET in records
    assert PREPARE_DUMMY_ATTN_TARGET in records
    assert DISPATCH_CG_TARGET in records
    assert EXECUTE_MODEL_TARGET in records
    for record in records.values():
        assert record.reason and record.remove_when
        assert record.affected_versions == SUPPORTED_RANGE

    # The framework rejects a second application of the same patch; idempotency
    # is enforced one level up, by vllm_sail.patch.install().
    with pytest.raises(RuntimeError, match="already patched"):
        dummy_batch_module.install()

    with pytest.raises(RuntimeError, match="already patched"):
        patch(
            "vllm.v1.worker.gpu.input_batch",
            "InputBatch.make_dummy",
            reason="test",
            affected_versions="test",
            remove_when="test",
        )(lambda *args, **kwargs: None)
