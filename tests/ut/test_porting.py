# SPDX-License-Identifier: Apache-2.0
"""Tests for ``vllm_sail.native.porting``, the upstream-kernel vendoring pipeline.

The port tool is the only thing standing between a vLLM bump and a corpus that
either misses a header, exports upstream's ``PyInit_`` symbol, or registers ops
whose implementation was never compiled. None of those fail at port time -- they
fail at link time on a PPU host, or worse, at runtime -- so the pipeline's
decisions are pinned here instead.

Everything below runs on a synthetic upstream checkout built under ``tmp_path``
and a synthetic manifest, with :func:`porting.recording_translator` in place of
sailify. That is deliberate: the merge gate has no vLLM checkout, no sailify, no
torch and no device, and ``port()`` takes an injected translator precisely so the
copy/closure/overlay/pruning logic stays testable without them.
"""

from __future__ import annotations

import sys
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest

from vllm_sail.native import manifest as m
from vllm_sail.native import porting, stubs


@pytest.mark.parametrize(
    "suffix",
    [
        "float_to_fp8",
        "float2_to_fp8x2",
        "fp8_to_halfraw",
        "fp8x2_to_halfraw2",
        "halfraw_to_fp8",
        "halfraw2_to_fp8x2",
    ],
)
def test_sailify_completes_fp8_intrinsics(tmp_path, monkeypatch, suffix):
    # Sailify 1.0.0 leaves these intrinsics unchanged. Exercise the production
    # adapter with that behaviour, without importing sailify in the merge gate.
    module = ModuleType("sailify.sailify_python")
    module.sailify_extra_files_recursive = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "sailify", ModuleType("sailify"))
    monkeypatch.setitem(sys.modules, "sailify.sailify_python", module)
    old, new = f"__nv_cvt_{suffix}", f"__hg_cvt_{suffix}"
    untouched = (
        f'// {old}\n/* {old} */\nconst char* s = "{old}";\n'
        f'const char* r = R"note("{old}")note";\n'
        f"int prefix{old}, {old}_suffix;\n"
        "void __nv_cvt_unknown();\n"
    )
    unit = tmp_path / "fp8.cuh"
    unit.write_text(untouched + f"auto value = {old}(x);\n", encoding="utf-8")
    porting.sailify_translator(tmp_path, [unit])
    expected = untouched + f"auto value = {new}(x);\n"
    assert unit.read_text(encoding="utf-8") == expected
    porting.sailify_translator(tmp_path, [unit])
    assert unit.read_text(encoding="utf-8") == expected


# ---------------------------------------------------------------------------
# a synthetic upstream checkout
# ---------------------------------------------------------------------------

# The include collision that makes own-directory-first resolution load-bearing:
# upstream ships BOTH csrc/ops.h and csrc/libtorch_stable/ops.h, and a
# libtorch_stable TU writing `#include "ops.h"` means its own neighbour. If the
# wrong one were picked, csrc/legacy_only.h would appear in the closure -- which
# is exactly what the test asserts never happens.
_WRONG_OPS_H = """\
#pragma once
// csrc/ops.h -- the non-stable API. Never reachable from libtorch_stable/.
#include "legacy_only.h"
"""

_MAIN_BINDINGS = """\
#include "core/registration.h"
#include "ops.h"
#include "torch_utils.h"

#include <torch/csrc/stable/library.h>

STABLE_TORCH_LIBRARY_FRAGMENT(_C, ops) {
  // Compiled tier: kept verbatim, comment and all.
  ops.def("copy_blocks(Tensor! key_cache, Tensor! value_cache) -> ()");

  // Excluded tier: this comment goes with the statement below.
  ops.def(
      "cutlass_scaled_mm(Tensor! out, Tensor a, Tensor b, "
      "Tensor a_scales, Tensor b_scales, Tensor? bias) -> ()");

  ops.def("cutlass_scaled_mm_supports_fp8(int cuda_device_capability) -> bool");

  // Registered upstream, described by no manifest entry.
  ops.def("brand_new_kernel(Tensor a, Tensor! out) -> ()");

  ops.def("reshape_and_cache.out(Tensor a, Tensor! out) -> ()");

  ops.set_python_module("vllm._C");

#ifndef USE_ROCM
  ops.def("cutlass_scaled_mm_supports_fp4(int cuda_device_capability) -> bool");
#endif
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, ops) {
  ops.impl("copy_blocks", TORCH_BOX(&copy_blocks));
  ops.impl("cutlass_scaled_mm", TORCH_BOX(&cutlass_scaled_mm));
}

REGISTER_EXTENSION(_C_stable_libtorch)
"""

_MOE_BINDINGS = """\
#include "core/registration.h"
#include "moe_ops.h"

#include <torch/csrc/stable/library.h>

STABLE_TORCH_LIBRARY_FRAGMENT(_moe_C, m) {
  m.def(
      "moe_align_block_size(Tensor topk_ids, int num_experts, "
      "int block_size, Tensor! sorted_token_ids) -> ()");

  m.def("moe_permute_unpermute_supported() -> bool");

  m.def("moe_permute_sort_workspace_size(int num_tokens, int topk) -> int");

  m.def("moe_wna16_gemm(Tensor input, Tensor! output) -> bool");
}

STABLE_TORCH_LIBRARY_IMPL(_moe_C, CUDA, m) {
  m.impl("moe_align_block_size", TORCH_BOX(&moe_align_block_size));
#ifndef USE_ROCM
  m.impl("moe_permute", TORCH_BOX(&moe_permute));
#endif
}

REGISTER_EXTENSION(_moe_C_stable_libtorch)
"""

_TREE: dict[str, str] = {
    "csrc/ops.h": _WRONG_OPS_H,
    "csrc/legacy_only.h": "#pragma once\n",
    "csrc/cuda_compat.h": "#pragma once\n#include <cuda_runtime.h>\n",
    "csrc/core/registration.h": "#pragma once\n",
    "csrc/libtorch_stable/ops.h": '#pragma once\n#include "cuda_compat.h"\n',
    # torch_utils.h is in nearly every TU's closure and drags in cuBLAS even
    # where nothing calls it; sailify renames the header, the overlay drops it.
    "csrc/libtorch_stable/torch_utils.h": "#pragma once\n#include <cublas_v2.h>\n",
    "csrc/libtorch_stable/cache_kernels.cu": (
        "#include <torch/all.h>\n"
        '#include "ops.h"\n'
        '#include "torch_utils.h"\n'
        '#include "ppu_sdk/absent.h"\n'
        "\nvoid copy_blocks() {}\n"
    ),
    "csrc/libtorch_stable/torch_bindings.cpp": _MAIN_BINDINGS,
    "csrc/libtorch_stable/moe/moe_ops.h": "#pragma once\n",
    "csrc/libtorch_stable/moe/moe_align.cu": (
        '#include "moe_ops.h"\n\nvoid moe_align_block_size() {}\n'
    ),
    "csrc/libtorch_stable/moe/torch_bindings.cpp": _MOE_BINDINGS,
    # Declared in uncompiled tiers, so present upstream but never vendored.
    "csrc/moe/moe_permute.cu": "void moe_permute() {}\n",
    "csrc/quantization/cutlass_mm.cu": "void cutlass_scaled_mm() {}\n",
}

_MANIFEST = """\
schema = 1
upstream_ref = "v0.27.1"
compiled_tiers = ["a"]

[tiers]
a = "ported now"
b = "ported next"
x = "excluded, stubbed"

[hazards]
cub = "CUB / Thrust / libcu++ (cuda/std/...)"
cutlass = "CUTLASS / CuTe in the include closure"
ptx = "inline asm volatile PTX"

[overlay]
drop_includes = ["cublas_v2.h", "acblas_v2.h"]
prune_bindings = [
    "csrc/libtorch_stable/torch_bindings.cpp",
    "csrc/libtorch_stable/moe/torch_bindings.cpp",
]

[overlay.rename_extension]
_C_stable_libtorch = "_upstream_C"
_moe_C_stable_libtorch = "_upstream_moe_C"

[[entry]]
path = "csrc/libtorch_stable/torch_bindings.cpp"
tier = "a"
ops = ["weak_ref_tensor"]

[[entry]]
path = "csrc/libtorch_stable/cache_kernels.cu"
tier = "a"
hazards = ["cub"]
ops = ["copy_blocks", "reshape_and_cache.out"]

[[entry]]
path = "csrc/libtorch_stable/moe/torch_bindings.cpp"
tier = "a"

[[entry]]
path = "csrc/libtorch_stable/moe/moe_align.cu"
tier = "a"
ops = ["moe_align_block_size"]

[[entry]]
path = "csrc/moe/moe_permute.cu"
tier = "b"
ops = [
    "moe_permute",
    "moe_permute_unpermute_supported",
    "moe_permute_sort_workspace_size",
    "moe_wna16_gemm",
]

[[entry]]
path = "csrc/quantization/cutlass_mm.cu"
tier = "x"
hazards = ["cutlass", "ptx"]
ops = [
    "cutlass_scaled_mm",
    "cutlass_scaled_mm_supports_fp8",
    "cutlass_scaled_mm_supports_fp4",
]
"""

_CUTLASS_REASON = "CUTLASS / CuTe in the include closure; inline asm volatile PTX"
_UNCLAIMED_REASON = "not declared in csrc/upstream/manifest.toml"


@pytest.fixture
def upstream(tmp_path: Path) -> Path:
    """A vLLM-shaped source checkout: the directory holding ``csrc/``."""
    src = tmp_path / "vllm"
    for rel, body in _TREE.items():
        path = src / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    return src


@pytest.fixture
def kernel_manifest(tmp_path: Path) -> m.Manifest:
    """Loaded through ``manifest.load`` so the fixture itself stays valid."""
    path = tmp_path / "manifest.toml"
    path.write_text(_MANIFEST, encoding="utf-8")
    return m.load(path)


@pytest.fixture
def dest(tmp_path: Path) -> Path:
    return tmp_path / "vendored"


@dataclass(frozen=True)
class Ported:
    result: porting.PortResult
    offered: tuple[Path, ...]
    dest: Path

    def read(self, rel: str) -> str:
        return (self.dest / rel).read_text(encoding="utf-8")


@pytest.fixture
def ported(upstream: Path, dest: Path, kernel_manifest: m.Manifest) -> Ported:
    seen: list[Path] = []
    result = porting.port(
        upstream, dest, kernel_manifest, porting.recording_translator(seen)
    )
    return Ported(result=result, offered=tuple(seen), dest=dest.resolve())


def _balanced(text: str) -> bool:
    """The cheap proxy for "still valid C++" after line-oriented surgery."""
    return text.count("{") == text.count("}") and text.count("(") == text.count(")")


def _pruner(manifest: m.Manifest) -> porting._BindingPruner:
    return porting._BindingPruner(
        owner={op: entry for entry in manifest for op in entry.ops},
        compiled=frozenset(manifest.compiled_tiers),
        hazards=manifest.hazards,
    )


# ---------------------------------------------------------------------------
# include closure
# ---------------------------------------------------------------------------


def test_a_libtorch_stable_tu_includes_its_own_ops_h_not_the_top_level_one(
    ported: Ported,
) -> None:
    """Own-directory-first resolution, on upstream's real header collision.

    ``csrc/ops.h`` and ``csrc/libtorch_stable/ops.h`` are different APIs. Picking
    the top-level one would compile the wrong declarations into the stable-ABI
    TUs, so its private include is the canary.
    """
    assert "csrc/upstream/libtorch_stable/ops.h" in ported.result.headers
    assert "csrc/upstream/ops.h" not in ported.result.headers
    assert "csrc/upstream/legacy_only.h" not in ported.result.headers
    assert not (ported.dest / "ops.h").exists()


def test_the_closure_is_transitive_and_falls_back_to_csrc(ported: Ported) -> None:
    """``cuda_compat.h`` is reached only via ``libtorch_stable/ops.h``."""
    assert ported.result.headers == (
        "csrc/upstream/core/registration.h",
        "csrc/upstream/cuda_compat.h",
        "csrc/upstream/libtorch_stable/moe/moe_ops.h",
        "csrc/upstream/libtorch_stable/ops.h",
        "csrc/upstream/libtorch_stable/torch_utils.h",
    )


def test_angle_bracket_includes_are_never_followed(ported: Ported) -> None:
    """``<torch/all.h>``, ``<cuda_runtime.h>`` and friends belong to the toolchain."""
    assert ported.result.unresolved_includes == ("ppu_sdk/absent.h",)
    assert not any("torch" in name for name in ported.result.unresolved_includes)


def test_an_include_that_resolves_nowhere_is_reported_not_fatal(
    ported: Ported,
) -> None:
    """SDK-provided headers legitimately resolve to nothing; that is data, not death."""
    assert "ppu_sdk/absent.h" in ported.result.unresolved_includes
    assert ported.result.sources


def test_a_manifest_entry_whose_file_is_missing_is_fatal(upstream: Path) -> None:
    """A wrong ``--vllm-src`` or an unbumped ref must not silently port less."""
    entry = m.Entry(path="csrc/libtorch_stable/vanished.cu", tier="a")
    with pytest.raises(porting.PortError, match="vanished.cu, which does not exist"):
        porting._closure(upstream.resolve(), (entry,))


def test_an_include_escaping_csrc_is_fatal(tmp_path: Path) -> None:
    """The vendored layout only mirrors ``csrc/``; anything above it has no home."""
    src = tmp_path / "vllm"
    (src / "csrc").mkdir(parents=True)
    (src / "outside.h").write_text("#pragma once\n", encoding="utf-8")
    (src / "csrc" / "a.cu").write_text('#include "../outside.h"\n', encoding="utf-8")

    entry = m.Entry(path="csrc/a.cu", tier="a")
    with pytest.raises(porting.PortError, match="which is outside csrc/"):
        porting._closure(src.resolve(), (entry,))


# ---------------------------------------------------------------------------
# vendored layout and the translator handoff
# ---------------------------------------------------------------------------


def test_vendored_paths_mirror_upstream_below_csrc(ported: Ported) -> None:
    """``csrc/a/b.cu`` -> ``csrc/upstream/a/b.hg``, so a vLLM bump stays a diff."""
    assert ported.result.sources == (
        "csrc/upstream/libtorch_stable/torch_bindings.cpp",
        "csrc/upstream/libtorch_stable/cache_kernels.hg",
        "csrc/upstream/libtorch_stable/moe/torch_bindings.cpp",
        "csrc/upstream/libtorch_stable/moe/moe_align.hg",
    )
    assert (ported.dest / "libtorch_stable/moe/moe_align.hg").is_file()
    assert (ported.dest / "core/registration.h").is_file()
    # Compiled TUs are renamed, not copied alongside their upstream names.
    assert not list(ported.dest.rglob("*.cu"))


def test_only_compiled_tiers_are_vendored(ported: Ported) -> None:
    """A tier-b or tier-x TU exists upstream but must never reach the corpus."""
    assert not (ported.dest / "moe/moe_permute.cu").exists()
    assert not (ported.dest / "quantization/cutlass_mm.cu").exists()
    assert not any("cutlass_mm" in name for name in ported.result.files)


def test_the_translator_is_handed_every_copied_file_after_the_copy(
    upstream: Path, dest: Path, kernel_manifest: m.Manifest
) -> None:
    """sailify rewrites in place under upstream names; the rename comes after.

    The translator must see the copied tree exactly as upstream named it, with
    every file already on disk; the vendor tree applies ``vendored_name`` once
    translation is done.
    """
    offered: list[Path] = []
    existing: list[bool] = []

    def translate(dest_root: Path, files: Sequence[Path]) -> None:
        offered.extend(files)
        existing.extend(path.is_file() for path in files)

    result = porting.port(upstream, dest, kernel_manifest, translate)

    def vendored(path: Path) -> str:
        rel = path.relative_to(dest.resolve()).as_posix()
        return f"csrc/upstream/{m.vendored_name(f'csrc/{rel}')}"

    assert sorted(vendored(path) for path in offered) == sorted(result.files)
    assert all(existing)


# ---------------------------------------------------------------------------
# rebuild semantics
# ---------------------------------------------------------------------------


def test_a_rebuild_wipes_the_previous_run_but_keeps_the_manifest(
    upstream: Path, dest: Path, kernel_manifest: m.Manifest
) -> None:
    """Otherwise a tier demotion leaves an orphan TU that CMake still compiles.

    ``manifest.toml`` is the hand-written *input* that happens to live in the
    output directory, so it is the one thing the rebuild must not eat.
    """
    dest.mkdir(parents=True)
    (dest / "manifest.toml").write_text("# hand written\n", encoding="utf-8")
    stale = dest / "quantization" / "demoted.cu"
    stale.parent.mkdir(parents=True)
    stale.write_text("void gone() {}\n", encoding="utf-8")
    (dest / "excluded_ops.toml").write_text("# from the last run\n", encoding="utf-8")

    porting.port(upstream, dest, kernel_manifest, porting.recording_translator([]))

    assert not stale.exists()
    assert not stale.parent.exists()
    assert (dest / "manifest.toml").read_text(encoding="utf-8") == "# hand written\n"
    assert "GENERATED" in (dest / "excluded_ops.toml").read_text(encoding="utf-8")
    assert porting.PRESERVED == frozenset({"manifest.toml"})


# ---------------------------------------------------------------------------
# overlay
# ---------------------------------------------------------------------------


def test_drop_includes_removes_the_header_the_overlay_refuses(ported: Ported) -> None:
    """Left alone, ``torch_utils.h`` would drag acblas into the whole corpus."""
    text = ported.read("libtorch_stable/torch_utils.h")
    assert "cublas_v2.h" not in text
    assert text.startswith("#pragma once")


def test_drop_includes_runs_after_the_translator_renamed_the_header(
    upstream: Path, dest: Path, kernel_manifest: m.Manifest
) -> None:
    """sailify turns ``<cublas_v2.h>`` into ``<acblas_v2.h>``; both must go.

    The order matters: the overlay is applied to the *translated* tree, so a
    rename the translator performs cannot smuggle the include past the drop list.
    """

    def translate(dest_root: Path, files: Sequence[Path]) -> None:
        for path in files:
            body = path.read_text(encoding="utf-8")
            if "cublas_v2.h" in body:
                path.write_text(
                    body.replace("cublas_v2.h", "acblas_v2.h"), encoding="utf-8"
                )

    porting.port(upstream, dest, kernel_manifest, translate)
    text = (dest / "libtorch_stable/torch_utils.h").read_text(encoding="utf-8")
    assert "blas_v2.h" not in text


@pytest.mark.parametrize("occurrences", [0, 1, 2])
def test_text_overlay_requires_one_translated_match(
    upstream: Path, dest: Path, tmp_path: Path, occurrences: int
) -> None:
    manifest_path = tmp_path / "overlay.toml"
    manifest_path.write_text(
        _MANIFEST
        + '\n[overlay.replace_text."csrc/libtorch_stable/cache_kernels.cu"]\n'
        + '"void translated() {}" = "void adapted() {}"\n',
        encoding="utf-8",
    )
    manifest = m.load(manifest_path)
    source = upstream / "csrc/libtorch_stable/cache_kernels.cu"
    original = "void original() {}\n" * occurrences
    source.write_text(original, encoding="utf-8")

    def translate(dest_root, files):
        unit = dest_root / "libtorch_stable/cache_kernels.cu"
        unit.write_text(
            unit.read_text(encoding="utf-8").replace("original", "translated"),
            encoding="utf-8",
        )

    if occurrences == 1:
        porting.port(upstream, dest, manifest, translate)
        # The manifest keys the upstream `.cu` name; the overlay lands in the
        # renamed vendored file.
        assert (dest / "libtorch_stable/cache_kernels.hg").read_text() == (
            "void adapted() {}\n"
        )
    else:
        with pytest.raises(porting.PortError, match=f"found {occurrences}"):
            porting.port(upstream, dest, manifest, translate)
    assert source.read_text(encoding="utf-8") == original


def test_text_overlay_refuses_a_file_outside_the_corpus(tmp_path: Path) -> None:
    outside = tmp_path / "outside.cu"
    outside.write_text("original", encoding="utf-8")
    with pytest.raises(porting.PortError, match="was not copied"):
        porting._replace_text([], tmp_path, {"../outside.cu": {"original": "adapted"}})
    assert outside.read_text(encoding="utf-8") == "original"


@pytest.mark.parametrize(
    "line",
    [
        '#include "cublas_v2.h"',
        "#include <cublas_v2.h>",
        '  #  include   "third_party/cublas_v2.h"',
    ],
)
def test_drop_includes_matches_on_basename_in_either_form(line: str) -> None:
    assert porting._includes_any(line + "\n", frozenset({"cublas_v2.h"}))


@pytest.mark.parametrize(
    "line",
    ['#include "cublas_v2.hpp"', "// #include <cublas_v2.h>", "int cublas_v2_h = 0;"],
)
def test_drop_includes_leaves_everything_else_alone(line: str) -> None:
    assert not porting._includes_any(line + "\n", frozenset({"cublas_v2.h"}))


def test_rename_extension_retargets_our_pyinit_symbol_only(ported: Ported) -> None:
    """``REGISTER_EXTENSION(NAME)`` defines ``PyInit_<NAME>``.

    A verbatim copy would export upstream's module-init symbol, and the extension
    would fail to import as ``vllm_sail._upstream_C``.
    """
    assert "REGISTER_EXTENSION(_upstream_C)" in ported.read(
        "libtorch_stable/torch_bindings.cpp"
    )
    assert "REGISTER_EXTENSION(_upstream_moe_C)" in ported.read(
        "libtorch_stable/moe/torch_bindings.cpp"
    )
    assert "_stable_libtorch" not in ported.read("libtorch_stable/torch_bindings.cpp")


def test_rename_extension_leaves_unlisted_names_untouched() -> None:
    text = "REGISTER_EXTENSION( _C_stable_libtorch )\nREGISTER_EXTENSION(_ppu_moe_C)\n"
    renamed = porting._rename_extension(text, {"_C_stable_libtorch": "_upstream_C"})

    assert "REGISTER_EXTENSION( _upstream_C )" in renamed
    assert "REGISTER_EXTENSION(_ppu_moe_C)" in renamed


def test_rename_extension_without_a_rename_map_is_the_identity() -> None:
    text = "REGISTER_EXTENSION(_C_stable_libtorch)\n"
    assert porting._rename_extension(text, {}) == text


# ---------------------------------------------------------------------------
# binding pruning
# ---------------------------------------------------------------------------


def test_pruned_bindings_keep_compiled_ops_and_drop_the_rest(ported: Ported) -> None:
    text = ported.read("libtorch_stable/torch_bindings.cpp")

    # Claimed by a compiled entry: an implementation exists, so the registration
    # must survive byte for byte.
    assert (
        'ops.def("copy_blocks(Tensor! key_cache, Tensor! value_cache) -> ()");' in text
    )
    assert 'ops.impl("copy_blocks", TORCH_BOX(&copy_blocks));' in text
    # Claimed by tier x / tier b: registering it would not link.
    assert "cutlass_scaled_mm" not in text
    # Claimed by nobody: no entry means no ported TU means no implementation.
    assert "brand_new_kernel" not in text


def test_an_overload_name_is_extracted_whole(ported: Ported) -> None:
    """``reshape_and_cache.out`` is a distinct op; truncating it would drop it."""
    assert "reshape_and_cache.out" in ported.read("libtorch_stable/torch_bindings.cpp")


@pytest.mark.parametrize(
    ("kind", "text", "expected"),
    [
        ("def", 'ops.def("foo.out(Tensor a, Tensor! out) -> ()");', "foo.out"),
        ("impl", 'ops.impl("foo.out", TORCH_BOX(&foo_out));', "foo.out"),
        # A def schema is a run of adjacent literals the C++ compiler joins, so
        # the name is the leading identifier of the *joined* text.
        ("def", 'ops.def(\n  "moe_"\n  "sum(Tensor a) -> ()");', "moe_sum"),
        ("def", "ops.def(TORCH_SELECTIVE_SCHEMA(kSchema));", None),
    ],
)
def test_op_name_comes_from_the_joined_string_literals(
    kind: str, text: str, expected: str | None
) -> None:
    assert porting._op_name(text, kind) == expected


def test_a_multi_line_statement_is_kept_or_dropped_as_a_whole(ported: Ported) -> None:
    """Half a statement is a compile error, so the unit is the statement."""
    text = ported.read("libtorch_stable/moe/torch_bindings.cpp")

    assert '"moe_align_block_size(Tensor topk_ids, int num_experts, "' in text
    assert '"int block_size, Tensor! sorted_token_ids) -> ()");' in text
    assert "moe_wna16_gemm" not in text


def test_a_comment_follows_the_statement_it_documents(ported: Ported) -> None:
    """A comment left behind by its dropped statement documents nothing."""
    text = ported.read("libtorch_stable/torch_bindings.cpp")

    assert "// Compiled tier: kept verbatim, comment and all." in text
    assert "// Excluded tier:" not in text
    assert "// Registered upstream, described by no manifest entry." not in text


def test_preprocessor_lines_survive_at_statement_boundaries(ported: Ported) -> None:
    """They are emitted verbatim, never swallowed into a neighbouring statement."""
    for rel in (
        "libtorch_stable/torch_bindings.cpp",
        "libtorch_stable/moe/torch_bindings.cpp",
    ):
        text = ported.read(rel)
        assert text.count("#ifndef USE_ROCM") == 1
        assert text.count("#endif") == 1


def test_an_ifdef_whose_every_statement_is_dropped_leaves_a_harmless_pair(
    kernel_manifest: m.Manifest,
) -> None:
    """An empty ``#ifndef``/``#endif`` compiles; a dangling one does not."""
    text = textwrap.dedent("""\
        STABLE_TORCH_LIBRARY_IMPL(_moe_C, CUDA, m) {
        #ifndef USE_ROCM
          m.impl("moe_permute", TORCH_BOX(&moe_permute));
        #endif
        }
        """)
    pruned = _pruner(kernel_manifest).prune(text)

    assert pruned == (
        "STABLE_TORCH_LIBRARY_IMPL(_moe_C, CUDA, m) {\n#ifndef USE_ROCM\n#endif\n}\n"
    )
    assert _balanced(pruned)


def test_a_statement_that_is_neither_def_nor_impl_is_preserved(ported: Ported) -> None:
    """Unrecognised is preserved rather than guessed at."""
    assert 'ops.set_python_module("vllm._C");' in ported.read(
        "libtorch_stable/torch_bindings.cpp"
    )


def test_pruned_output_stays_brace_and_paren_balanced(ported: Ported) -> None:
    for rel in (
        "libtorch_stable/torch_bindings.cpp",
        "libtorch_stable/moe/torch_bindings.cpp",
    ):
        assert _balanced(ported.read(rel)), rel


def test_the_number_of_dropped_registrations_is_reported_per_file(
    ported: Ported,
) -> None:
    """The count is what a reviewer scans after a vLLM bump."""
    assert ported.result.pruned == {
        "csrc/upstream/libtorch_stable/torch_bindings.cpp": 5,
        "csrc/upstream/libtorch_stable/moe/torch_bindings.cpp": 4,
    }


def test_an_unterminated_registration_block_is_fatal(
    kernel_manifest: m.Manifest,
) -> None:
    """Silently truncating the TU at EOF would emit code that cannot compile."""
    text = (
        'STABLE_TORCH_LIBRARY_FRAGMENT(_C, ops) {\n  ops.def("copy_blocks() -> ()");\n'
    )
    with pytest.raises(porting.PortError, match="unterminated _C registration block"):
        _pruner(kernel_manifest).prune(text)


def test_prune_bindings_naming_an_uncopied_file_is_fatal(
    upstream: Path, dest: Path, tmp_path: Path
) -> None:
    """A bindings TU parked in an uncompiled tier cannot be pruned into place."""
    body = _MANIFEST.replace(
        'path = "csrc/libtorch_stable/moe/torch_bindings.cpp"\ntier = "a"',
        'path = "csrc/libtorch_stable/moe/torch_bindings.cpp"\ntier = "b"',
    )
    path = tmp_path / "demoted.toml"
    path.write_text(body, encoding="utf-8")

    with pytest.raises(porting.PortError, match="which this port did not copy"):
        porting.port(upstream, dest, m.load(path), porting.recording_translator([]))


# ---------------------------------------------------------------------------
# excluded ops
# ---------------------------------------------------------------------------


def test_every_dropped_def_becomes_an_excluded_op(ported: Ported) -> None:
    """Sorted by op name, so the generated file's diff is readable."""
    assert [(op.namespace, op.name) for op in ported.result.excluded_ops] == [
        ("_C", "brand_new_kernel"),
        ("_C", "cutlass_scaled_mm"),
        ("_C", "cutlass_scaled_mm_supports_fp4"),
        ("_C", "cutlass_scaled_mm_supports_fp8"),
        ("_moe_C", "moe_permute_sort_workspace_size"),
        ("_moe_C", "moe_permute_unpermute_supported"),
        ("_moe_C", "moe_wna16_gemm"),
    ]


def test_an_excluded_op_carries_its_tier_and_its_hazard_descriptions(
    ported: Ported,
) -> None:
    """The reason is what a maintainer reads out of a NotImplementedError.

    It must be the hazard *descriptions* from the manifest's ``[hazards]`` table,
    not the terse keys, because the message has no room for a glossary.
    """
    by_name = {op.name: op for op in ported.result.excluded_ops}

    assert by_name["cutlass_scaled_mm"].tier == "x"
    assert by_name["cutlass_scaled_mm"].reason == _CUTLASS_REASON
    assert by_name["cutlass_scaled_mm"].schema == (
        "cutlass_scaled_mm(Tensor! out, Tensor a, Tensor b, Tensor a_scales, "
        "Tensor b_scales, Tensor? bias) -> ()"
    )


def test_an_op_in_a_hazard_free_uncompiled_tier_says_so_plainly(
    ported: Ported,
) -> None:
    op = {o.name: o for o in ported.result.excluded_ops}["moe_wna16_gemm"]
    assert op.tier == "b"
    assert op.reason == "no PPU implementation"


def test_an_unclaimed_op_is_stubbed_and_announced(
    ported: Ported, kernel_manifest: m.Manifest
) -> None:
    """This list is how a vLLM bump's new kernels ask to be triaged into a tier."""
    assert ported.result.unclaimed_ops == ("brand_new_kernel",)

    op = {o.name: o for o in ported.result.excluded_ops}["brand_new_kernel"]
    # Not a manifest tier, so the stub message says "undeclared" out loud rather
    # than blaming a tier the maintainer would then go looking for.
    assert op.tier == "unclaimed"
    assert op.tier not in kernel_manifest.tiers
    assert op.reason == _UNCLAIMED_REASON


def test_an_op_claimed_by_an_uncompiled_tier_is_not_called_unclaimed(
    ported: Ported,
) -> None:
    """Deliberate exclusion and undeclared novelty need different follow-ups."""
    assert "moe_permute" not in ported.result.unclaimed_ops
    assert "cutlass_scaled_mm" not in ported.result.unclaimed_ops


def test_a_dropped_impl_without_a_def_produces_no_schema(ported: Ported) -> None:
    """``moe_permute`` is only ``impl``-ed here; there is no signature to stub."""
    assert "moe_permute" not in {op.name for op in ported.result.excluded_ops}


# ---------------------------------------------------------------------------
# probe vs compute
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("schema", "kind"),
    [
        ("moe_permute_unpermute_supported() -> bool", "probe"),
        ("cutlass_scaled_mm_supports_fp8(int cuda_device_capability) -> bool", "probe"),
        # Bool-returning but tensor-taking: answering False would be a lie about
        # data, not about the device.
        ("moe_wna16_gemm(Tensor input, Tensor! output) -> bool", "compute"),
        # Primitive-only but not a capability question.
        ("moe_permute_sort_workspace_size(int num_tokens, int topk) -> int", "compute"),
        ("copy_blocks(Tensor! a, Tensor! b) -> ()", "compute"),
    ],
)
def test_only_a_tensor_free_bool_query_counts_as_a_probe(
    schema: str, kind: str
) -> None:
    """A probe that raises breaks vLLM's backend selection; a compute op that
    silently returns produces wrong numerics. The classification is the whole
    difference between the two stub behaviours."""
    assert porting._is_probe(schema) is (kind == "probe")


def test_the_ported_corpus_classifies_its_probes(ported: Ported) -> None:
    kinds = {op.name: op.kind for op in ported.result.excluded_ops}
    assert kinds["moe_permute_unpermute_supported"] == "probe"
    assert kinds["cutlass_scaled_mm_supports_fp8"] == "probe"
    assert kinds["cutlass_scaled_mm_supports_fp4"] == "probe"
    assert kinds["moe_permute_sort_workspace_size"] == "compute"
    assert kinds["moe_wna16_gemm"] == "compute"


# ---------------------------------------------------------------------------
# excluded_ops.toml: the contract with vllm_sail.native.stubs
# ---------------------------------------------------------------------------


def test_the_generated_schema_file_round_trips_through_the_stub_reader(
    ported: Ported,
) -> None:
    """The one test that pins the porting/stubs contract.

    ``stubs.load_schemas`` is the only consumer of this file, and it runs at
    import time inside every vLLM process. If the two modules disagree about a
    field, the corpus ships with no stubs at all.
    """
    generated = ported.dest / porting.EXCLUDED_OPS_NAME
    assert generated.is_file()
    assert stubs.load_schemas(generated) == ported.result.excluded_ops


def test_the_generated_schema_file_records_what_it_was_generated_from(
    ported: Ported, kernel_manifest: m.Manifest
) -> None:
    text = (ported.dest / porting.EXCLUDED_OPS_NAME).read_text(encoding="utf-8")
    assert "do not edit" in text
    assert f"Upstream vLLM {kernel_manifest.upstream_ref}" in text
    assert "compiled tiers a" in text


def test_quotes_and_backslashes_survive_the_round_trip(
    tmp_path: Path, kernel_manifest: m.Manifest
) -> None:
    """Schemas carry default values and sailify notes carry paths; both bite TOML."""
    op = stubs.ExcludedOp(
        namespace="_C",
        name="odd",
        schema='odd(str tag="a\\"b") -> ()',
        kind="compute",
        tier="x",
        reason='needs C:\\cutlass and a "quoted" note',
    )
    path = tmp_path / porting.EXCLUDED_OPS_NAME
    path.write_text(
        porting.render_excluded_ops(kernel_manifest, [op]), encoding="utf-8"
    )

    assert stubs.load_schemas(path) == (op,)


def test_an_empty_exclusion_set_still_renders_a_readable_file(
    kernel_manifest: m.Manifest, tmp_path: Path
) -> None:
    """A fully ported corpus is the goal state, not an error state."""
    path = tmp_path / porting.EXCLUDED_OPS_NAME
    path.write_text(porting.render_excluded_ops(kernel_manifest, []), encoding="utf-8")

    assert stubs.load_schemas(path) == ()
    assert path.read_text(encoding="utf-8").startswith("# SPDX-License-Identifier")


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def test_the_report_asks_for_triage_of_the_unclaimed_ops(
    ported: Ported, kernel_manifest: m.Manifest
) -> None:
    text = porting.report(kernel_manifest, ported.result)

    assert "brand_new_kernel" in text
    assert "manifest does not describe" in text
    assert "ported 4 translation unit(s) and 5 header(s)" in text
    assert "stubbed ops: 7" in text


def test_the_report_names_the_hazards_sailify_cannot_fix(
    ported: Ported, kernel_manifest: m.Manifest
) -> None:
    """A compiled TU with a hazard needs a human to read the translated output."""
    text = porting.report(kernel_manifest, ported.result)

    assert "csrc/libtorch_stable/cache_kernels.cu: cub" in text
    # Hazards of excluded entries are not this section's business.
    assert "cutlass_mm.cu" not in text


def test_the_report_surfaces_unresolved_includes(
    ported: Ported, kernel_manifest: m.Manifest
) -> None:
    text = porting.report(kernel_manifest, ported.result)
    assert "ppu_sdk/absent.h" in text


def test_translate_corpus_copies_the_local_include_closure_and_rebuilds(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    csrc = source / "csrc"
    (csrc / "detail").mkdir(parents=True)
    (csrc / "kernel.cu").write_text(
        '#include "detail/a.cuh"\nCUDA_TOKEN kernel;\n', encoding="utf-8"
    )
    (csrc / "detail" / "a.cuh").write_text(
        '#include "b.h"\nCUDA_TOKEN a;\n', encoding="utf-8"
    )
    (csrc / "detail" / "b.h").write_text("CUDA_TOKEN b;\n", encoding="utf-8")
    destination = tmp_path / "generated"
    destination.mkdir()
    (destination / "stale.cu").write_text("stale\n", encoding="utf-8")

    def translate(root: Path, files: tuple[Path, ...]) -> None:
        for path in files:
            path.write_text(
                path.read_text(encoding="utf-8").replace("CUDA_TOKEN", "HGGC_TOKEN"),
                encoding="utf-8",
            )

    result = porting.translate_corpus(
        source, destination, ("csrc/kernel.cu",), translate=translate
    )

    assert result.sources == ("kernel.cu",)
    assert result.headers == ("detail/a.cuh", "detail/b.h")
    assert result.unresolved_includes == ()
    assert not (destination / "stale.cu").exists()
    assert "HGGC_TOKEN" in (destination / "kernel.cu").read_text(encoding="utf-8")
    assert "HGGC_TOKEN" in (destination / "detail" / "b.h").read_text(encoding="utf-8")
