# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for model-checkpoint resolution, availability and selection.

Covers the behaviours ``tests/e2e/models/_offline_runner`` grew so that a real
machine and a machine without the model NAS mount both do the right thing:

1. **Mirrored catalogue roots** -- ``/nas_aisw/datasets`` and ``/ppusw/datasets``
   are two mount points of one NFS export, so a recorded path is resolved across
   both before it is declared missing.
2. **Availability as a skip, not a failure** -- ``checkpoint_skip_reason`` turns
   an unusable weight tree into ``weights unavailable: <path>``, and a weight
   shard too large for the chip's tensor-parallel degree into
   ``weights do not fit: ...``; the shared ``offline_llm`` fixture feeds either
   to ``pytest.skip``.
3. **Single-valued override vs multi-variant tests** -- ``variant_checkpoint``
   keeps ``VLLM_SAIL_MODEL_ROOT`` dispatch honest for a test file that
   parametrizes over two checkpoints (the DeepSeek V4 quantizations).
4. **Engine teardown is best effort** -- ``release_llm`` never raises, so a
   cleanup failure cannot turn a passing test into an error.

The module under test imports neither ``torch`` nor ``vllm``, so everything here
runs on a plain CPU interpreter against ``tmp_path`` fixtures.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.e2e.models import _offline_runner as runner


@pytest.fixture
def mirrored_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    """Point ``CHECKPOINT_ROOT_MIRRORS`` at two ``tmp_path`` directories.

    Mirrors the production pair (``/nas_aisw/datasets`` <-> ``/ppusw/datasets``)
    without needing either mount, so the fallback logic is exercised verbatim.
    """
    primary = tmp_path / "primary" / "datasets"
    mirror = tmp_path / "mirror" / "datasets"
    primary.mkdir(parents=True)
    mirror.mkdir(parents=True)
    monkeypatch.setattr(
        runner,
        "CHECKPOINT_ROOT_MIRRORS",
        ((str(primary), str(mirror)), (str(mirror), str(primary))),
    )
    monkeypatch.delenv(runner.MODEL_ROOT_ENV, raising=False)
    return primary, mirror


def _make_checkpoint(root: Path, *segments: str) -> Path:
    """Create ``root/segments/config.json`` and return the checkpoint directory."""
    path = root.joinpath(*segments)
    path.mkdir(parents=True, exist_ok=True)
    (path / runner.CHECKPOINT_MARKER).write_text("{}", encoding="utf-8")
    return path


GIB = 1024**3


def _write_sparse_weights(path: Path, count: int, size_bytes: int) -> None:
    """Stage ``count`` weight shards of ``size_bytes`` each, sparsely.

    Sparse so a test can stand up a 1.5 TiB checkpoint in milliseconds and in no
    disk space at all: ``os.path.getsize`` reports the logical size, which is
    exactly what the capacity guard measures.
    """
    for index in range(count):
        with open(path / f"model-{index:05d}.safetensors", "wb") as handle:
            handle.truncate(size_bytes)


# ---------------------------------------------------------------------------
# Mirror roots.
# ---------------------------------------------------------------------------


def test_mirror_roots_lists_the_path_first_then_equivalents(
    mirrored_roots: tuple[Path, Path],
) -> None:
    primary, mirror = mirrored_roots
    recorded = str(
        primary / "checkpoints" / "LLM" / "Qwen" / "v3.8" / "Qwen3.8-Flash-Next"
    )
    assert runner._mirror_roots(recorded) == [
        recorded,
        str(mirror / "checkpoints" / "LLM" / "Qwen" / "v3.8" / "Qwen3.8-Flash-Next"),
    ]
    # A path under neither root has no equivalent and is returned alone.
    assert runner._mirror_roots("/somewhere/else/ckpt") == ["/somewhere/else/ckpt"]


def test_resolve_prefers_the_recorded_path_when_loadable(
    mirrored_roots: tuple[Path, Path],
) -> None:
    primary, mirror = mirrored_roots
    recorded = _make_checkpoint(primary, "checkpoints", "model")
    _make_checkpoint(mirror, "checkpoints", "model")
    assert runner.resolve_checkpoint(str(recorded)) == str(recorded)


def test_resolve_falls_back_to_the_mirror_root(
    mirrored_roots: tuple[Path, Path],
) -> None:
    """The catalogue root is not mounted; the same export is, elsewhere."""
    primary, mirror = mirrored_roots
    only_mirror = _make_checkpoint(mirror, "checkpoints", "model")
    recorded = str(primary / "checkpoints" / "model")
    assert runner.resolve_checkpoint(recorded) == str(only_mirror)


def test_resolve_returns_the_recorded_path_when_loadable_nowhere(
    mirrored_roots: tuple[Path, Path],
) -> None:
    """The skip reason must quote what the code asked for, not a guess."""
    primary, _mirror = mirrored_roots
    recorded = str(primary / "checkpoints" / "absent")
    assert runner.resolve_checkpoint(recorded) == recorded


def test_resolve_env_override_wins_outright(
    mirrored_roots: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit operator instruction is never second-guessed."""
    primary, _mirror = mirrored_roots
    recorded = _make_checkpoint(primary, "checkpoints", "model")
    monkeypatch.setenv(runner.MODEL_ROOT_ENV, "/operator/said/this")
    assert runner.resolve_checkpoint(str(recorded)) == "/operator/said/this"


# ---------------------------------------------------------------------------
# Availability guard.
# ---------------------------------------------------------------------------


def test_checkpoint_skip_reason_accepts_a_loadable_tree(tmp_path: Path) -> None:
    checkpoint = _make_checkpoint(tmp_path, "checkpoints", "model")
    assert runner.checkpoint_skip_reason(str(checkpoint)) is None


def test_checkpoint_skip_reason_rejects_missing_and_incomplete(tmp_path: Path) -> None:
    absent = tmp_path / "checkpoints" / "absent"
    assert runner.checkpoint_skip_reason(str(absent)) == (
        f"weights unavailable: {absent}"
    )
    # A directory with no config.json is not a checkpoint either.
    empty = tmp_path / "checkpoints" / "empty"
    empty.mkdir(parents=True)
    assert runner.checkpoint_skip_reason(str(empty)) == f"weights unavailable: {empty}"
    # Neither is a stray config.json that is not a file.
    marker_dir = tmp_path / "checkpoints" / "marker-is-a-dir"
    (marker_dir / runner.CHECKPOINT_MARKER).mkdir(parents=True)
    assert runner.checkpoint_skip_reason(str(marker_dir)) is not None


# ---------------------------------------------------------------------------
# Capacity guard: a weight shard bigger than one device cannot load at any
# ``gpu_memory_utilization``, and saying so beats an opaque engine OOM.
# ---------------------------------------------------------------------------


def test_checkpoint_weight_bytes_sums_only_weight_files(tmp_path: Path) -> None:
    checkpoint = _make_checkpoint(tmp_path, "checkpoints", "model")
    _write_sparse_weights(checkpoint, 2, 1024)
    for name in ("README.md", "generation_config.json", "tokenizer.model"):
        (checkpoint / name).write_text("not a weight", encoding="utf-8")
    assert runner.checkpoint_weight_bytes(str(checkpoint)) == 2048


def test_checkpoint_weight_bytes_counts_every_weight_suffix(tmp_path: Path) -> None:
    """Not every checkpoint is safetensors; the older formats are weights too."""
    checkpoint = _make_checkpoint(tmp_path, "checkpoints", "legacy")
    for suffix in runner.WEIGHT_FILE_SUFFIXES:
        with open(checkpoint / f"shard{suffix}", "wb") as handle:
            handle.truncate(256)
    assert runner.checkpoint_weight_bytes(str(checkpoint)) == 256 * len(
        runner.WEIGHT_FILE_SUFFIXES
    )


def test_checkpoint_weight_bytes_is_none_when_unmeasurable(tmp_path: Path) -> None:
    """Measured-zero and could-not-measure must stay distinguishable."""
    assert runner.checkpoint_weight_bytes(str(tmp_path / "absent")) is None
    no_weights = _make_checkpoint(tmp_path, "checkpoints", "no-weights")
    assert runner.checkpoint_weight_bytes(str(no_weights)) is None


def test_checkpoint_skip_reason_reports_a_shard_that_cannot_fit(tmp_path: Path) -> None:
    """Large MoE on ZW-890P, reproduced without the checkpoint.

    1.5 TiB of bf16 weights over 96 shards at TP=8 is 192 GiB per rank against
    144 GiB per device; on real hardware all eight workers died in
    ``initialize_model`` holding 142.87 GiB and asking for 1.15 GiB more.  The
    skip reason has to carry those numbers, because "weights do not fit" alone
    does not tell an operator whether to add cards or to file a bug.
    """
    checkpoint = _make_checkpoint(
        tmp_path,
        "checkpoints",
        "ppu_quantization",
        "qlean-test",
        "Large-MoE-Quant-W-INT4-PerChannel-A-INT8-PerToken",
    )
    _write_sparse_weights(checkpoint, 96, 16 * GIB)
    reason = runner.checkpoint_skip_reason(
        str(checkpoint), tensor_parallel_size=8, device_total_bytes=144 * GIB
    )
    assert reason is not None
    assert reason.startswith(f"weights do not fit: {checkpoint}")
    assert "1536.0 GiB of weight files" in reason
    assert "192.0 GiB per rank" in reason
    assert "tensor_parallel_size=8" in reason
    assert "144.0 GiB in total" in reason


def test_checkpoint_skip_reason_accepts_the_same_model_at_a_wider_tp(
    tmp_path: Path,
) -> None:
    """At the ZW-810E degree of 16 the MoE shard is 96 GiB and does fit."""
    checkpoint = _make_checkpoint(
        tmp_path,
        "checkpoints",
        "ppu_quantization",
        "qlean-test",
        "Large-MoE-Quant-W-INT4-PerChannel-A-INT8-PerToken",
    )
    _write_sparse_weights(checkpoint, 96, 16 * GIB)
    assert (
        runner.checkpoint_skip_reason(
            str(checkpoint), tensor_parallel_size=16, device_total_bytes=144 * GIB
        )
        is None
    )


def test_checkpoint_skip_reason_accepts_a_shard_that_exactly_fits(
    tmp_path: Path,
) -> None:
    """The comparison is inclusive: equal to the device is not over it."""
    checkpoint = _make_checkpoint(tmp_path, "checkpoints", "exact")
    _write_sparse_weights(checkpoint, 8, 16 * GIB)
    assert (
        runner.checkpoint_skip_reason(
            str(checkpoint), tensor_parallel_size=8, device_total_bytes=16 * GIB
        )
        is None
    )


def test_capacity_guard_needs_both_inputs(tmp_path: Path) -> None:
    """Without a TP degree *and* a device size there is nothing to compare."""
    checkpoint = _make_checkpoint(tmp_path, "checkpoints", "huge")
    _write_sparse_weights(checkpoint, 8, 16 * GIB)
    assert runner.checkpoint_skip_reason(str(checkpoint)) is None
    assert (
        runner.checkpoint_skip_reason(str(checkpoint), tensor_parallel_size=8) is None
    )
    assert (
        runner.checkpoint_skip_reason(str(checkpoint), device_total_bytes=GIB) is None
    )


def test_capacity_guard_never_masks_the_availability_guard(tmp_path: Path) -> None:
    """An unmounted tree is ``weights unavailable`` whatever the device is."""
    absent = tmp_path / "checkpoints" / "absent"
    assert (
        runner.checkpoint_skip_reason(
            str(absent), tensor_parallel_size=8, device_total_bytes=GIB
        )
        == f"weights unavailable: {absent}"
    )


def test_capacity_guard_does_not_guess_from_an_unmeasurable_tree(
    tmp_path: Path,
) -> None:
    """A loadable tree with no weight file is not declared too big on a guess."""
    checkpoint = _make_checkpoint(tmp_path, "checkpoints", "no-weights")
    assert (
        runner.checkpoint_skip_reason(
            str(checkpoint), tensor_parallel_size=8, device_total_bytes=GIB
        )
        is None
    )


def test_catalogue_and_prod_mirrors_are_symmetric() -> None:
    """Both directions must be listed, or one mount would resolve and not the other."""
    pairs = dict(runner.CHECKPOINT_ROOT_MIRRORS)
    for source, target in list(pairs.items()):
        assert pairs.get(target) == source, (source, target)
    assert "/nas_aisw/datasets" in pairs and "/ppusw/datasets" in pairs


# ---------------------------------------------------------------------------
# Variant selection under a single-valued override.
# ---------------------------------------------------------------------------


def test_variant_checkpoint_resolves_normally_without_override(
    mirrored_roots: tuple[Path, Path],
) -> None:
    primary, mirror = mirrored_roots
    only_mirror = _make_checkpoint(mirror, "checkpoints", "model")
    recorded = str(primary / "checkpoints" / "model")
    assert runner.variant_checkpoint(recorded) == str(only_mirror)


def test_variant_checkpoint_selects_the_named_variant(
    mirrored_roots: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    primary, mirror = mirrored_roots
    int8 = _make_checkpoint(primary, "checkpoints", "v4", "INT8")
    _make_checkpoint(primary, "checkpoints", "qlean", "MXFP4")

    # The override may name either mount point of the same tree.
    monkeypatch.setenv(runner.MODEL_ROOT_ENV, str(int8))
    assert runner.variant_checkpoint(str(int8)) == str(int8)
    via_mirror = str(mirror / "checkpoints" / "v4" / "INT8")
    monkeypatch.setenv(runner.MODEL_ROOT_ENV, via_mirror)
    assert runner.variant_checkpoint(str(int8)) == via_mirror

    # The *other* variant is deselected rather than silently re-pointed.
    monkeypatch.setenv(runner.MODEL_ROOT_ENV, str(int8))
    other = str(primary / "checkpoints" / "qlean" / "MXFP4")
    assert runner.variant_checkpoint(other) is None


def test_same_checkpoint_ignores_the_mount_point(
    mirrored_roots: tuple[Path, Path],
) -> None:
    primary, mirror = mirrored_roots
    left = str(primary / "checkpoints" / "v4" / "INT8")
    right = str(mirror / "checkpoints" / "v4" / "INT8")
    assert runner._same_checkpoint(left, right)
    assert runner._same_checkpoint(left, left)
    assert not runner._same_checkpoint(
        left, str(primary / "checkpoints" / "qlean" / "X")
    )
    assert not runner._same_checkpoint("/a/b", "/c/d")


def test_prod_variant_tails_are_distinct() -> None:
    """The two DeepSeek V4 quantizations must never share a golden identity."""
    int8 = (
        "/nas_aisw/datasets/checkpoints/LLM/T-HEAD/v4/DeepSeek-V4-Flash-0731-W8A8-INT8"
    )
    mxfp4 = (
        "/nas_aisw/datasets/checkpoints/ppu_quantization/qlean-test/"
        "DeepSeek-V4-Flash-0731-mxfp4-fp8channel"
    )
    assert runner._checkpoint_tail(int8) != runner._checkpoint_tail(mxfp4)
    assert not runner._same_checkpoint(int8, mxfp4)


# ---------------------------------------------------------------------------
# Engine teardown.
# ---------------------------------------------------------------------------


class _RaisingCore:
    def shutdown(self) -> None:
        raise RuntimeError("engine core refused to shut down")


class _RaisingLLM:
    llm_engine = type("_E", (), {"engine_core": _RaisingCore()})()


def test_release_llm_calls_shutdown_when_present() -> None:
    calls: list[str] = []

    class _Core:
        def shutdown(self) -> None:
            calls.append("shutdown")

    llm = type("_LLM", (), {"llm_engine": type("_E", (), {"engine_core": _Core()})()})()
    runner.release_llm(llm)
    assert calls == ["shutdown"]


def test_release_llm_survives_junk_and_a_raising_core() -> None:
    """Teardown must never turn a passing test into an error."""
    runner.release_llm(None)
    runner.release_llm(object())
    runner.release_llm(type("_NoEngine", (), {})())
    with pytest.warns(UserWarning, match="engine core shutdown failed"):
        runner.release_llm(_RaisingLLM())


def test_release_llm_is_idempotent() -> None:
    calls: list[str] = []

    class _Core:
        def shutdown(self) -> None:
            calls.append("shutdown")

    llm = type("_LLM", (), {"llm_engine": type("_E", (), {"engine_core": _Core()})()})()
    runner.release_llm(llm)
    runner.release_llm(llm)
    assert calls == ["shutdown", "shutdown"]
