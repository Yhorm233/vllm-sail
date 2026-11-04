# SPDX-License-Identifier: Apache-2.0
"""Device-independent unit tests for runtime audit infrastructure."""

import ast
import importlib.util
import json
import os
import sys
import types
from contextlib import contextmanager, nullcontext
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
SOURCE = ROOT / "vllm_sail/models/deepseek_v41/runtime_audit.py"


@contextmanager
def _isolated_audit_hooks():
    """Isolate this thread's audit hooks and restore outer tools such as coverage."""
    profile = sys.getprofile()
    trace = sys.gettrace()
    try:
        sys.setprofile(None)
        sys.settrace(None)
        yield
    finally:
        sys.setprofile(profile)
        sys.settrace(trace)


@pytest.fixture
def audit():
    spec = importlib.util.spec_from_file_location("v41_audit_unit", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with _isolated_audit_hooks():
        yield module


@pytest.mark.parametrize("hooks", ["none", "profile", "trace", "both"])
@pytest.mark.parametrize("failed", [False, True])
def test_isolated_audit_hooks_restore_outer_state(hooks, failed):
    outer_profile = sys.getprofile()
    outer_trace = sys.gettrace()

    def existing_profile(*args):
        return None

    def existing_trace(*args):
        return existing_trace

    profile = existing_profile if hooks in ("profile", "both") else None
    trace = existing_trace if hooks in ("trace", "both") else None
    try:
        sys.setprofile(profile)
        sys.settrace(trace)
        expected = (
            pytest.raises(RuntimeError, match="fixture failure")
            if failed
            else nullcontext()
        )
        with expected:
            with _isolated_audit_hooks():
                assert sys.getprofile() is None
                assert sys.gettrace() is None
                if failed:
                    raise RuntimeError("fixture failure")
        assert sys.getprofile() is profile
        assert sys.gettrace() is trace
    finally:
        sys.setprofile(outer_profile)
        sys.settrace(outer_trace)


def test_import_has_no_runtime_dependency(audit):
    assert "torch" not in audit.__dict__
    assert "vllm" not in audit.__dict__


def test_alias_observes_executed_frame(audit, monkeypatch):
    def target(x):
        return x + 1

    alias = target
    monkeypatch.setitem(audit.TARGETS, (__name__, "target"), "q")
    synced = []
    output, events = audit.execute_traced(alias, (3,), {}, lambda: synced.append(True))
    assert output == 4 and synced == [True]
    assert events["q"]["count"] == 1
    assert events["q"]["samples"][0]["source"]["sha256"]
    assert sys.getprofile() is None


def test_non_target_does_not_claim_kernel(audit):
    output, events = audit.execute_traced(lambda: 7, (), {}, lambda: None)
    assert output == 7 and events == {}


@pytest.mark.parametrize("where", ["forward", "synchronize"])
def test_failure_removes_profiler_and_is_not_success(audit, where):
    def fail():
        raise RuntimeError("fixture failure")

    fn = fail if where == "forward" else lambda: 1
    sync = fail if where == "synchronize" else lambda: None
    with pytest.raises(RuntimeError, match="fixture failure"):
        audit.execute_traced(fn, (), {}, sync)
    assert sys.getprofile() is None


@pytest.mark.parametrize("kind", ["profile", "trace"])
def test_existing_profiler_or_trace_not_overwritten(audit, kind):
    def existing(*args):
        return existing

    getter = getattr(sys, "get" + kind)
    setter = getattr(sys, "set" + kind)
    previous = getter()
    calls = []
    setter(existing)
    try:
        with pytest.raises(RuntimeError, match="profiler/trace"):
            audit.execute_traced(
                lambda: calls.append("forward"),
                (),
                {},
                lambda: calls.append("synchronize"),
            )
        assert getter() is existing
        assert calls == []
    finally:
        setter(previous)


def test_first_last_samples_bounded_counts_complete(audit, monkeypatch):
    def target(compress_ratio):
        return compress_ratio

    monkeypatch.setitem(audit.TARGETS, (__name__, "target"), "k")
    _, events = audit.execute_traced(
        lambda: [target(r) for r in [1, 2] * 10], (), {}, lambda: None
    )
    assert set(events) == {"k1", "k2"}
    assert all(
        item["count"] == 10 and len(item["samples"]) == 2 for item in events.values()
    )


def test_candidate_before_after(audit, monkeypatch):
    def target():
        marker = 1
        return marker

    monkeypatch.setitem(audit.TARGETS, (__name__, "target"), "candidate_mask")
    calls = []

    def sample(values):
        calls.append(1)
        return dict(marker=values.get("marker", 0))

    monkeypatch.setattr(audit, "candidate_sample", sample)
    _, events = audit.execute_traced(target, (), {}, lambda: None)
    details = events["candidate_mask"]["samples"][0]["details"]
    assert details == {"before": {"marker": 0}, "after": {"marker": 1}}
    assert len(calls) == 2


@pytest.mark.parametrize("overload", [False, True])
def test_native_packet_records_actual_dispatch(audit, overload):
    namespace = dict(__name__="torch._ops")
    exec(compile("def __call__(self):\n    return 9\n", __file__, "exec"), namespace)
    packet = (
        types.SimpleNamespace(_schema=types.SimpleNamespace(name="_C::persistent_topk"))
        if overload
        else types.SimpleNamespace(
            _qualified_op_name="_C::persistent_topk", _schemas={"": "schema"}
        )
    )
    result, events = audit.execute_traced(
        namespace["__call__"], (packet,), {}, lambda: None
    )
    assert result == 9
    assert events["topk_decode"]["samples"][0]["details"]["op"] == "_C::persistent_topk"


def make_runtime(audit, tmp_path, monkeypatch):
    monkeypatch.setattr(
        audit, "worker_identity", lambda *args: dict(groups={"dp": {"rank": 2}})
    )
    monkeypatch.setattr(
        audit, "loaded_runtime", lambda: dict(files={"fixture": "not-device-evidence"})
    )
    runtime = audit.RuntimeAudit(str(tmp_path), object(), object())
    return runtime


class PaddingMask:
    """Pure-Python mask stub for testing runtime audit classification."""

    ndim = 1
    dtype = "torch.bool"

    def __init__(self, values):
        self.values = values

    def numel(self):
        return len(self.values)

    def sum(self):
        return types.SimpleNamespace(item=lambda: sum(self.values))

    def all(self):
        return types.SimpleNamespace(item=lambda: all(self.values))

    def __getitem__(self, index):
        return PaddingMask(self.values[index])


def install_runtime_stubs(
    monkeypatch, metadata, synchronize=lambda: None, padding=None
):
    if padding is None:
        count = max(
            (
                getattr(value, "num_actual_tokens", 0)
                for value in (metadata or {}).values()
            ),
            default=0,
        )
        padding = PaddingMask([False] * count)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(cuda=types.SimpleNamespace(synchronize=synchronize)),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.forward_context",
        types.SimpleNamespace(
            get_forward_context=lambda: types.SimpleNamespace(
                attn_metadata=metadata, is_padding=padding
            )
        ),
    )


def test_unarmed_and_stopped_do_not_import_runtime(audit, tmp_path, monkeypatch):
    runtime = make_runtime(audit, tmp_path, monkeypatch)
    monkeypatch.setitem(sys.modules, "torch", None)
    assert runtime.run(lambda: 4) == 4
    (tmp_path / "capture.json").write_text("invalid but stopped")
    (tmp_path / "stop").touch()
    assert runtime.run(lambda: 5) == 5
    assert len(runtime.log.read_text().splitlines()) == 1


@pytest.mark.parametrize(
    "metadata", [None, {}, {"layer": types.SimpleNamespace(num_actual_tokens=0)}]
)
def test_dummy_or_idle_rank_not_credited(audit, tmp_path, monkeypatch, metadata):
    runtime = make_runtime(audit, tmp_path, monkeypatch)
    (tmp_path / "capture.json").write_text(
        json.dumps(dict(schema=1, session_id="sample"))
    )
    install_runtime_stubs(monkeypatch, metadata)
    assert runtime.run(lambda: 3) == 3
    assert len(runtime.log.read_text().splitlines()) == 1


def test_positive_metadata_dummy_not_credited_and_later_real_is_recorded(
    audit, tmp_path, monkeypatch
):
    runtime = make_runtime(audit, tmp_path, monkeypatch)
    (tmp_path / "capture.json").write_text(
        json.dumps(dict(schema=1, session_id="sample"))
    )
    metadata = {"layer": types.SimpleNamespace(num_actual_tokens=1)}
    mask = PaddingMask([True])
    install_runtime_stubs(monkeypatch, metadata, padding=mask)
    calls = []
    for _ in range(2):
        runtime.run(lambda: calls.append("idle"))
    mask.values = [False]
    runtime.run(lambda: calls.append("real"))
    assert calls == ["idle", "idle", "real"]
    rows = [json.loads(line) for line in runtime.log.read_text().splitlines()]
    forwards = [row for row in rows if row["event"] in ("forward", "idle_forward")]
    assert [row["event"] for row in forwards] == ["idle_forward", "forward"]
    assert [row["actual_tokens"] for row in forwards] == [0, 1]
    assert forwards[0]["token_counts"]["metadata_tokens"] == 1
    assert forwards[0]["events"] == {}
    assert any(row["event"] == "runtime_loaded" for row in rows)


@pytest.mark.parametrize(
    "metadata_tokens,values,actual",
    [(1, [True], 0), (1, [False], 1), (3, [False, False, True, True], 2)],
)
def test_padding_counts(audit, metadata_tokens, values, actual):
    result = audit.forward_token_counts(
        {"layer": types.SimpleNamespace(num_actual_tokens=metadata_tokens)},
        PaddingMask(values),
    )
    assert result["actual_tokens"] == actual
    assert result["padded_tokens"] == len(values)
    assert result["padding_tokens"] == sum(values)


@pytest.mark.parametrize("values", [None, [], [False, False], [True, False]])
def test_invalid_padding_evidence_rejected(audit, values):
    with pytest.raises(RuntimeError):
        audit.forward_token_counts(
            {"layer": types.SimpleNamespace(num_actual_tokens=1)},
            PaddingMask(values) if values is not None else None,
        )


@pytest.mark.parametrize("failed", [False, True])
def test_success_and_failure_are_distinct(audit, tmp_path, monkeypatch, failed):
    runtime = make_runtime(audit, tmp_path, monkeypatch)
    (tmp_path / "capture.json").write_text(
        json.dumps(dict(schema=1, session_id="sample"))
    )
    install_runtime_stubs(
        monkeypatch, {"layer": types.SimpleNamespace(num_actual_tokens=17)}
    )

    def forward():
        if failed:
            raise RuntimeError("bad")
        return 42

    if failed:
        with pytest.raises(RuntimeError, match="bad"):
            runtime.run(forward)
    else:
        assert runtime.run(forward) == 42
    row = json.loads(runtime.log.read_text().splitlines()[-1])
    assert row["event"] == ("forward_failed" if failed else "forward")
    assert row["session_id"] == "sample"
    assert sys.getprofile() is None


def test_loaded_snapshot_only_after_success_and_once(audit, tmp_path, monkeypatch):
    runtime = make_runtime(audit, tmp_path, monkeypatch)
    (tmp_path / "capture.json").write_text(
        json.dumps(dict(schema=1, session_id="sample"))
    )
    install_runtime_stubs(
        monkeypatch, {"layer": types.SimpleNamespace(num_actual_tokens=1)}
    )
    identities = []

    def identity(*args):
        identities.append(True)
        return dict(groups=runtime.identity["groups"], after_loading=True)

    monkeypatch.setattr(audit, "worker_identity", identity)
    runtime.run(lambda: 1)
    runtime.run(lambda: 2)
    records = [json.loads(line) for line in runtime.log.read_text().splitlines()]
    assert [record["event"] for record in records] == [
        "constructed",
        "runtime_loaded",
        "forward",
        "forward",
    ]
    assert identities == [True]
    assert records[1]["identity"]["after_loading"] is True


def test_late_decode_path_refreshes_loaded_snapshot(audit, tmp_path, monkeypatch):
    runtime = make_runtime(audit, tmp_path, monkeypatch)
    (tmp_path / "capture.json").write_text(
        json.dumps(dict(schema=1, session_id="sample"))
    )
    install_runtime_stubs(
        monkeypatch, {"layer": types.SimpleNamespace(num_actual_tokens=1)}
    )
    runtime.run(lambda: 0)

    def later():
        return 1

    monkeypatch.setitem(audit.TARGETS, (__name__, "later"), "logits_decode")
    runtime.run(later)
    runtime.run(later)
    rows = [json.loads(line) for line in runtime.log.read_text().splitlines()]
    assert [row["sequence"] for row in rows if row["event"] == "runtime_loaded"] == [
        1,
        2,
    ]


def test_loaded_runtime_no_import_and_native_identity(audit, tmp_path, monkeypatch):
    plugin = tmp_path / "vllm_sail"
    plugin.mkdir()
    native = plugin / "fixture.so"
    native.write_bytes(b"not a real device binary")
    source = plugin / "fixture.py"
    source.write_text("raise RuntimeError('must not import')\n")
    record = tmp_path / "fixture.dist-info/RECORD"
    record.parent.mkdir()
    record.write_text("fixture\n")
    distribution = types.SimpleNamespace(
        version="fixture",
        files=[Path("fixture.dist-info/RECORD")],
        locate_file=lambda path: tmp_path / path,
    )
    monkeypatch.setattr(
        audit.importlib.metadata, "distribution", lambda name: distribution
    )
    monkeypatch.setitem(
        sys.modules, "vllm_sail.fixture", types.SimpleNamespace(__file__=str(source))
    )
    info = native.stat()
    line = f"100-200 r-xp 0000 {os.major(info.st_dev):x}:{os.minor(info.st_dev):x} {info.st_ino} {native}\n"
    original = Path.read_text
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path, *args, **kwargs: (
            line if str(path) == "/proc/self/maps" else original(path, *args, **kwargs)
        ),
    )
    snapshot = audit.loaded_runtime()
    assert snapshot["modules"]["vllm_sail.fixture"] == str(source)
    assert snapshot["files"][str(native)] == audit.file_identity(native)["sha256"]
    assert snapshot["native_mappings"][0]["inode"] == info.st_ino
    assert snapshot["uid"] == os.getuid()
    line = line.rstrip() + " (deleted)\n"
    with pytest.raises(RuntimeError, match="replaced or deleted"):
        audit.loaded_runtime()


@pytest.mark.parametrize("mode", ["same", "other_inode", "other_hash", "denied"])
def test_overlay_mapping_requires_real_object_proof(audit, tmp_path, monkeypatch, mode):
    native = tmp_path / "native.so"
    native.write_bytes(b"original binary")
    other = tmp_path / "other.so"
    other.write_bytes(native.read_bytes())
    info = native.stat()
    parts = ["100-200", "r-xp", "0", "ff:ff", str(info.st_ino), str(native)]
    original = Path.open

    def mapped_open(path, *args, **kwargs):
        if str(path) == "/proc/self/map_files/100-200":
            if mode == "denied":
                raise PermissionError("map_files denied")
            return original(other if mode == "other_inode" else native, *args, **kwargs)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", mapped_open)
    if mode == "other_hash":
        real_identity = audit.opened_file_identity
        calls = []

        def changed_hash(handle):
            identity = real_identity(handle)
            calls.append(True)
            if len(calls) == 2:
                identity["sha256"] = "0" * 64
            return identity

        monkeypatch.setattr(audit, "opened_file_identity", changed_hash)
    if mode == "same":
        record = audit.native_mapping_identity(parts)
        assert record["identity_method"] == "proc_map_files"
        assert record["device"] == "ff:ff"
        assert record["file"] == record["mapped_file"]
        assert record["file"]["sha256"] == audit.file_identity(native)["sha256"]
    else:
        with pytest.raises((RuntimeError, PermissionError)):
            audit.native_mapping_identity(parts)


def test_opened_binary_mutation_rejected(audit, tmp_path):
    path = tmp_path / "native.so"
    path.write_bytes(b"original")
    with path.open("rb") as handle:

        class MutatingReader:
            def fileno(self):
                return handle.fileno()

            def read(self, size):
                value = handle.read(size)
                if value:
                    path.write_bytes(b"changed size")
                return value

        with pytest.raises(RuntimeError, match="changed"):
            audit.opened_file_identity(MutatingReader())


def test_binary_path_replacement_during_sampling_rejected(audit, tmp_path, monkeypatch):
    native = tmp_path / "native.so"
    native.write_bytes(b"same bytes")
    other = tmp_path / "other.so"
    other.write_bytes(native.read_bytes())
    info = native.stat()
    parts = [
        "100-200",
        "r-xp",
        "0",
        f"{os.major(info.st_dev):x}:{os.minor(info.st_dev):x}",
        str(info.st_ino),
        str(native),
    ]
    original = audit.opened_file_identity

    def replace(handle):
        result = original(handle)
        other.replace(native)
        return result

    monkeypatch.setattr(audit, "opened_file_identity", replace)
    with pytest.raises(RuntimeError, match="path was replaced"):
        audit.native_mapping_identity(parts)


def test_real_readonly_mapping_identity_or_permission_denial(audit, tmp_path):
    import mmap

    native = tmp_path / "native.so"
    native.write_bytes(b"real host mapping only")
    with (
        native.open("rb") as handle,
        mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ),
    ):
        parts = next(
            line.split(None, 5)
            for line in Path("/proc/self/maps").read_text().splitlines()
            if line.endswith(str(native))
        )
        try:
            record = audit.native_mapping_identity(parts)
        except PermissionError as exc:
            # CPU tests require no container privileges; reject mappings that cannot be verified.
            assert exc.filename == "/proc/self/map_files/" + parts[0]
            info = native.stat()
            device = tuple(int(value, 16) for value in parts[3].split(":"))
            assert (
                os.major(info.st_dev),
                os.minor(info.st_dev),
            ) != device or info.st_ino != int(parts[4])
            return
    assert record["file"]["sha256"] == audit.file_identity(native)["sha256"]


def test_code_digest_stable_but_sensitive(audit):
    text = "def target(value):\n    return value + 1\n"
    first = compile(text, "/fixture.py", "exec", dont_inherit=True)
    second = compile(text, "/fixture.py", "exec", dont_inherit=True)
    assert audit.code_fingerprint(first) == audit.code_fingerprint(second)
    changed = compile(
        text.replace("+ 1", "+ 2"), "/fixture.py", "exec", dont_inherit=True
    )
    assert audit.code_fingerprint(first) != audit.code_fingerprint(changed)


def test_existing_worker_file_rejected(audit, tmp_path, monkeypatch):
    make_runtime(audit, tmp_path, monkeypatch)
    with pytest.raises(FileExistsError):
        make_runtime(audit, tmp_path, monkeypatch)


@pytest.mark.parametrize(
    "enabled,ppu,cap,eager",
    [
        (False, False, (8, 9), False),
        (False, True, (8, 0), True),
        (False, True, (8, 9), True),
        (True, True, (8, 0), True),
        (True, False, (8, 0), True),
        (True, True, (8, 9), True),
        (True, True, (8, 0), False),
    ],
)
@pytest.mark.parametrize(
    "int8,fail_forward", [(False, False), (True, False), (True, True)]
)
def test_wrapper_audit_guard_and_forward_delegation(
    monkeypatch, enabled, ppu, cap, eager, int8, fail_forward
):
    calls = []
    outer_config = object()
    current = [outer_config]
    selected = ppu and cap == (8, 0) and int8

    @contextmanager
    def configuration(value):
        previous = current[0]
        current[0] = value
        calls.append(("configure", value))
        try:
            yield
        finally:
            current[0] = previous

    class Base:
        def __init__(self, **kwargs):
            self.hf_to_vllm_mapper = object()
            calls.append(("construct", kwargs))

        def forward(self, *args, **kwargs):
            assert current[0] is (config if selected else outer_config)
            calls.append(("forward", args, kwargs))
            if fail_forward:
                raise RuntimeError("forward failed")
            return 17

    class Audit:
        def __init__(self, *args):
            calls.append(("audit", args))

        def run(self, fn, *args, **kwargs):
            calls.append(("trace",))
            return fn(*args, **kwargs)

    module = types.ModuleType("v41_wrapper_fixture.runtime_audit")
    module.RuntimeAudit = Audit
    monkeypatch.setitem(sys.modules, module.__name__, module if enabled else None)
    tree = ast.parse(SOURCE.with_name("ppu_model.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    namespace = dict(
        __name__="v41_wrapper_fixture.ppu_model",
        __package__="v41_wrapper_fixture",
        UpstreamDeepseekV41ForCausalLM=Base,
        uses_int8_w8a8=lambda config: int8,
        int8_weights_mapper=lambda mapper: mapper,
        set_current_vllm_config=configuration,
        nullcontext=nullcontext,
        envs=types.SimpleNamespace(VLLM_SAIL_V41_AUDIT_DIR="/audit" if enabled else ""),
        current_platform=types.SimpleNamespace(
            is_ppu=lambda: ppu, is_device_capability=lambda value: value == cap
        ),
    )
    exec(
        compile(
            ast.Module(body=[node], type_ignores=[]),
            str(SOURCE.with_name("ppu_model.py")),
            "exec",
        ),
        namespace,
    )
    config = types.SimpleNamespace(
        quant_config=None, model_config=types.SimpleNamespace(enforce_eager=eager)
    )
    cls = namespace["DeepseekV41ForCausalLM"]
    if enabled and not (ppu and cap == (8, 0) and eager):
        with pytest.raises(ValueError, match="810E PPU eager"):
            cls(vllm_config=config)
        assert calls == []
        return
    model = cls(vllm_config=config, prefix="model")
    assert model._v41_int8_config is (config if selected else None)
    if fail_forward:
        with pytest.raises(RuntimeError, match="forward failed"):
            model.forward(1, 2, 3, 4, lookback_token_ids=5, extra=6)
    else:
        assert model.forward(1, 2, 3, 4, lookback_token_ids=5, extra=6) == 17
    assert calls[-1] == ("forward", (1, 2, 3, 4), dict(lookback_token_ids=5, extra=6))
    assert current[0] is outer_config
    assert any(call[0] == "configure" for call in calls) == selected
    assert any(call[0] == "trace" for call in calls) == enabled


def test_worker_identity_uses_actual_head_and_vocab_bounds(audit, monkeypatch):
    class Model:
        def named_modules(self):
            yield "language_model.model.layers.1.engram.embed_tokens", table

    value = types.SimpleNamespace(
        device=types.SimpleNamespace(type="cpu"), is_pinned=lambda: True
    )
    table = types.SimpleNamespace(
        dp_shared_memory=True,
        cpu_offload=True,
        weight=value,
        weight_scale_inv=value,
        head_start=7,
        part_n_hash_cols=3,
        vocab_start_idx=123,
        vocab_end_idx=456,
    )
    model = Model()
    model.language_model = Model()
    group = types.SimpleNamespace(world_size=1, rank_in_group=0, ranks=[0])
    parallel = types.SimpleNamespace(
        get_tp_group=lambda: group, get_dp_group=lambda: group
    )
    monkeypatch.setitem(
        sys.modules, "vllm.distributed", types.SimpleNamespace(parallel_state=parallel)
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(
            cuda=types.SimpleNamespace(
                current_device=lambda: 0, get_device_name=lambda device: "PPU-ZW810E"
            )
        ),
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "8,9,10,11,12,13,14,15")
    monkeypatch.setattr(audit, "tensor_layout", lambda tensor: {})
    monkeypatch.setattr(audit, "mapped_region", lambda tensor: dict(inode="123"))
    config = types.SimpleNamespace(
        parallel_config=types.SimpleNamespace(enable_expert_parallel=False),
        model_config=types.SimpleNamespace(model="1017"),
        cache_config=None,
    )
    identity = audit.worker_identity(model, config)
    actual = next(iter(identity["engram"].values()))
    assert (
        actual["head_start"],
        actual["head_end"],
        actual["vocab_start"],
        actual["vocab_end"],
    ) == (7, 10, 123, 456)
    assert identity["physical_device"] == 8
    assert identity["groups"]["ep"] == dict(enabled=False, size=1, rank=0)
