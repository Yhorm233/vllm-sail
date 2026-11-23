# SPDX-License-Identifier: Apache-2.0
"""Integrity of the shipped kernel corpus manifest, and the reader's refusals.

``csrc/upstream/manifest.toml`` is the single source of truth behind the port
tool, the CMake source lists, the runtime stubs and the docs table, so a
malformed or drifted manifest is a build-and-runtime bug rather than a docs bug.
These tests are stdlib-only: the reader never imports torch or vLLM, and neither
does the merge gate.
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import pytest

from vllm_sail.native import manifest as m

# The upstream ops the plugin's own Python already resolves through the shared
# `_C` / `_moe_C` namespaces, from docs/developer_guide/kernels.md
# section 2.3. `scaled_int8_quant` is deliberately absent: it is a Python wrapper
# in vllm/_custom_ops.py over the two real `*_scaled_int8_quant` ops, not an op.
BORROWED_OPS = (
    "cp_gather_and_upconvert_fp8_kv_cache",
    "cp_gather_cache",
    "cp_gather_indexer_k_quant_cache",
    "cutlass_scaled_mm",
    "cutlass_scaled_mm_azp",
    "dynamic_scaled_int8_quant",
    "gather_and_maybe_dequant_cache",
    "indexer_k_quant_and_cache",
    "per_token_group_fp8_quant",
    "per_token_group_quant_int8",
    "persistent_masked_m_silu_mul_quant",
    "persistent_topk",
    "scaled_fp4_quant",
    "static_scaled_int8_quant",
    "top_k_per_row_decode",
    "top_k_per_row_prefill",
)

# The accepted phase-7 regressions: sailify carries no CUTLASS mappings and this
# build wires no CUTLASS, so these lose their implementation and gain a stub
# (section 2.3, tier x). The two GEMM ops are CUTLASS to the core;
# `scaled_fp4_quant` is not -- only its 175-line entry TU is, so it returns once
# that entry is rewritten CUTLASS-free.
ACCEPTED_REGRESSIONS = frozenset(
    {"cutlass_scaled_mm", "cutlass_scaled_mm_azp", "scaled_fp4_quant"}
)

# Plugin-original, registered by vllm_sail._moe_C into `_ppu_moe_C`, therefore not
# part of the ported upstream corpus.
PLUGIN_OWNED_OPS = ("ep_scatter_2_cuda", "top_k_per_row_prefill_bf16")

_TOPK_DTYPES = (("f32", "float"), ("f16", "__half"), ("bf16", "__nv_bfloat16"))
_TOPK_INDICES = (("int32", "int"), ("uint32", "uint32_t"), ("int64", "int64_t"))
_TOPK_DISPATCH = "csrc/moe/topk_softplus_sqrt.cu"
_TOPK_HEADER = "csrc/moe/topk_softplus_sqrt_kernels.cuh"
_TOPK_INSTANCES = tuple(
    f"csrc/moe/topk_softplus_sqrt_{dtype}_{index}.cu"
    for dtype, _ in _TOPK_DTYPES
    for index, _ in _TOPK_INDICES
)


@pytest.fixture(scope="module")
def shipped() -> m.Manifest:
    return m.load()


def test_upstream_ref_matches_the_verified_vllm_commit(shipped: m.Manifest) -> None:
    """Runtime compatibility and CI must select the same upstream source."""
    pin = (
        (Path(__file__).parents[2] / ".github/vllm-main-verified.commit")
        .read_text()
        .strip()
    )
    assert re.fullmatch(r"[0-9a-f]{40}", shipped.upstream_ref)
    assert shipped.upstream_ref == pin


@pytest.mark.parametrize("operation,symbol", [("sum", "+="), ("or", "|=")])
def test_histogram_warp_reduction_uses_hggc_shuffle_fallback(
    shipped, operation, symbol
):
    """检查实际生成 header；CUDA 兼容架构标识不得绕过 HGGC 门控。"""
    root = Path(__file__).resolve().parents[2]
    source = (root / "csrc/upstream/topk_histogram_4096.cuh").read_text()
    body = source.split(f"warp_reduce_{operation}_full(uint32_t v) {{", 1)[1].split(
        "\n}", 1
    )[0]
    assert body.startswith(
        "\n#if !defined(__HGGCCC__) && defined(COMPATIBLE_ARCH) && "
        "(COMPATIBLE_ARCH >= 800)"
    )
    fallback = body.split("#else", 1)[1].split("#endif", 1)[0]
    assert "kWarpSize >> 1; mask > 0; mask >>= 1" in fallback
    assert f"v {symbol} __shfl_xor_sync(0xFFFFFFFF, v, mask);" in fallback
    assert "return v;" in fallback
    assert "constexpr uint32_t kWarpSize = 32;" in source
    assert len(shipped.replace_text["csrc/topk_histogram_4096.cuh"]) == 2


@pytest.mark.upstream_source
def test_histogram_overlay_changes_only_real_upstream_architecture_guards(
    shipped, upstream_source_root, tmp_path
):
    """对真实 header 执行生产 overlay；此处不模拟 SDK 或运行归约。"""
    from vllm_sail.native import porting

    path = "csrc/topk_histogram_4096.cuh"
    source = (upstream_source_root / path).read_text()
    # sailify 的架构标识映射先于 manifest overlay。
    translated = source.replace("__CUDA_ARCH__", "COMPATIBLE_ARCH")
    unit = tmp_path / "topk_histogram_4096.cuh"
    unit.write_text(translated)
    replacements = shipped.replace_text[path]
    expected = translated
    for original, adapted in replacements.items():
        assert translated.count(original) == 1
        assert adapted == original.replace("#if ", "#if !defined(__HGGCCC__) && ", 1)
        expected = expected.replace(original, adapted)
    porting._replace_text([unit], tmp_path, {path: replacements})
    assert unit.read_text() == expected
    assert (upstream_source_root / path).read_text() == source


def test_every_tier_and_hazard_is_declared(shipped: m.Manifest) -> None:
    assert shipped.tiers
    assert shipped.hazards
    for entry in shipped:
        assert entry.tier in shipped.tiers, entry.path
        for hazard in entry.hazards:
            assert hazard in shipped.hazards, entry.path


def test_paths_are_unique(shipped: m.Manifest) -> None:
    paths = [e.path for e in shipped]
    assert sorted(paths) == sorted(set(paths))


def test_no_op_is_claimed_twice(shipped: m.Manifest) -> None:
    claims = [op for e in shipped for op in e.ops]
    assert sorted(claims) == sorted(set(claims))


def test_no_cutlass_hazard_in_a_compiled_tier(shipped: m.Manifest) -> None:
    assert [e.path for e in shipped.compiled() if "cutlass" in e.hazards] == []


def test_compiled_tiers_are_known_and_never_include_x(shipped: m.Manifest) -> None:
    assert shipped.compiled_tiers
    assert set(shipped.compiled_tiers) <= set(shipped.tiers)
    assert "x" not in shipped.compiled_tiers


def test_prune_bindings_name_real_bindings_entries(shipped: m.Manifest) -> None:
    by_path = {e.path: e for e in shipped}
    assert shipped.prune_bindings
    for path in shipped.prune_bindings:
        assert path in by_path
        assert by_path[path].is_bindings


def test_rename_extension_retargets_upstreams_own_module_names(
    shipped: m.Manifest,
) -> None:
    """Sources must be upstream's names, targets must be ours.

    ``REGISTER_EXTENSION(NAME)`` defines ``PyInit_<NAME>``, so the rename is what
    stops a verbatim copy from exporting upstream's symbol.
    """
    assert shipped.rename_extension
    for upstream, ours in shipped.rename_extension.items():
        assert upstream.endswith("_stable_libtorch"), upstream
        assert ours.startswith("_upstream"), ours
        assert upstream != ours


def test_rename_extension_targets_match_the_cmake_and_python_module_names(
    shipped: m.Manifest,
) -> None:
    """The three-way agreement that only breaks on a PPU host if it drifts.

    A rename target names the ``PyInit_`` symbol the ported TU will export; CMake
    decides the ``.so`` filename; ``extensions.py`` decides what gets imported. If
    any pair disagrees the extension builds and then fails to import, which is
    unobservable without hardware -- so it is gated here.
    """
    # Inspect the checkout under test, even when Python resolves ``vllm_sail``
    # from an installed wheel that intentionally does not ship build sources.
    root = Path(__file__).resolve().parents[2]
    cmake = (root / "CMakeLists.txt").read_text(encoding="utf-8")
    loader = Path(m.__file__).with_name("extensions.py").read_text(encoding="utf-8")

    targets = set(shipped.rename_extension.values())
    cmake_targets = set(re.findall(r"^ppu_add_extension\((_\w+)", cmake, re.MULTILINE))

    # A subset, not equal: CMake also builds the plugin-original `_C` / `_moe_C`,
    # which are ours already and so have nothing to rename.
    assert targets <= cmake_targets
    assert cmake_targets - targets == {"_C", "_moe_C"}
    for target in targets:
        assert f'"vllm_sail.{target}"' in loader


@pytest.mark.parametrize(
    ("path", "vendored"),
    [
        (
            "csrc/libtorch_stable/cache_kernels.cu",
            "csrc/upstream/libtorch_stable/cache_kernels.hg",
        ),
        (
            "csrc/libtorch_stable/moe/moe_wna16.cu",
            "csrc/upstream/libtorch_stable/moe/moe_wna16.hg",
        ),
        (
            "csrc/libtorch_stable/torch_bindings.cpp",
            "csrc/upstream/libtorch_stable/torch_bindings.cpp",
        ),
    ],
)
def test_vendored_path_mirrors_upstream_layout(path: str, vendored: str) -> None:
    assert m.Entry(path=path, tier="a").vendored == vendored


def test_is_moe_is_exactly_the_moe_subtree(shipped: m.Manifest) -> None:
    for entry in shipped:
        assert entry.is_moe == entry.path.startswith("csrc/moe/")


def test_compiled_sources_partition_the_compiled_entries(shipped: m.Manifest) -> None:
    sources, moe_sources = shipped.compiled_sources()
    compiled = shipped.compiled()

    assert sources == tuple(e.vendored for e in compiled if not e.is_moe)
    assert moe_sources == tuple(e.vendored for e in compiled if e.is_moe)
    assert set(sources) | set(moe_sources) == {e.vendored for e in compiled}
    assert set(sources) & set(moe_sources) == set()
    assert sources and moe_sources


def test_every_vendored_hg_source_is_compiled(shipped: m.Manifest) -> None:
    """A normal native build must not silently omit a vendored device TU."""
    root = Path(__file__).parents[2]
    sources, moe_sources = shipped.compiled_sources()
    configured = {path for path in (*sources, *moe_sources) if path.endswith(".hg")}
    vendored = {
        path.relative_to(root).as_posix()
        for path in (root / "csrc" / "upstream").rglob("*.hg")
    }

    assert configured == vendored


def test_topk_split_corpus_compiles_all_instances_and_one_op_owner(shipped):
    """检查真实生成布局；公共 header 不作为编译 TU，旧 TU 不得残留。"""
    entries = [e for e in shipped if e.path.startswith("csrc/moe/topk_softplus_sqrt")]
    assert [e.path for e in entries] == [_TOPK_DISPATCH, *_TOPK_INSTANCES]
    assert entries[0].ops == ("topk_softplus_sqrt",)
    assert all(e.ops == () for e in entries[1:])
    assert all(e.tier == "a" and e in shipped.compiled() for e in entries)
    assert [e.path for e in shipped if "topk_softplus_sqrt" in e.ops] == [
        _TOPK_DISPATCH
    ]
    sources, moe_sources = shipped.compiled_sources()
    expected = {e.vendored for e in entries}
    assert expected <= set(moe_sources) and expected.isdisjoint(sources)
    root = Path(__file__).resolve().parents[2]
    generated = root / "csrc/upstream/moe"
    assert {
        p.relative_to(root).as_posix() for p in generated.glob("topk_softplus_sqrt*.hg")
    } == expected
    assert not (generated / "topk_softplus_sqrt_kernels.hg").exists()
    header = generated / "topk_softplus_sqrt_kernels.cuh"
    assert header.is_file() and str(header.relative_to(root)) not in moe_sources
    body = header.read_text()
    assert "VLLM_TOPK_SOFTPLUS_SQRT_DECLARE(extern," in body
    assert "#undef MAX\n#undef MIN" in body
    assert "topk_softplus_sqrt" not in shipped.excluded_ops()
    helper = body.split("void topkGatingSoftplusSqrtLauncherHelper(", 1)[1].split(
        "// Launches the kernel", 1
    )[0]
    guard = "#if defined(__HGGCCC__)\n  if (num_rows == 0) return;\n#endif"
    assert helper.count(guard) == 1
    assert helper.index(guard) < helper.index("const int num_warps")


@pytest.mark.upstream_source
@pytest.mark.parametrize("dtype,input_type", _TOPK_DTYPES)
@pytest.mark.parametrize("index,output_type", _TOPK_INDICES)
def test_topk_real_instance_types_and_shared_header_closure(
    shipped, upstream_source_root, tmp_path, dtype, input_type, index, output_type
):
    """真实 TU 经过生产闭包复制；recording translator 不证明 sailify 或设备计算。"""
    from vllm_sail.native import porting

    path = f"csrc/moe/topk_softplus_sqrt_{dtype}_{index}.cu"
    source = (upstream_source_root / path).read_text()
    instances = re.findall(
        r"^VLLM_TOPK_SOFTPLUS_SQRT_INSTANTIATE\(([^)]+)\)$", source, re.M
    )
    expected = [
        (output_type, input_type, hash_type) for hash_type in ("int64_t", "int")
    ]
    if output_type == "uint32_t":
        expected.append((output_type, input_type, output_type))
    actual = [
        tuple(part.strip() for part in instance.split(",")) for instance in instances
    ]
    assert actual == expected and len(actual) == len(set(actual))
    assert '#include "topk_softplus_sqrt_kernels.cuh"' in source
    cmake = (upstream_source_root / "CMakeLists.txt").read_text()
    cmake_instances = cmake.split("set(VLLM_TOPK_SOFTPLUS_SQRT_SRC", 1)[1].split(
        ")", 1
    )[0]
    assert re.findall(r'"([^"]+\.cu)"', cmake_instances) == list(_TOPK_INSTANCES)
    seen = []
    dest = (tmp_path / "closure").resolve()
    result = porting.translate_corpus(
        upstream_source_root,
        dest,
        (_TOPK_DISPATCH, path),
        porting.recording_translator(seen),
    )
    assert result.sources == tuple(
        p.removeprefix("csrc/") for p in (_TOPK_DISPATCH, path)
    )
    assert _TOPK_HEADER.removeprefix("csrc/") in result.headers
    assert "topk_softplus_sqrt_kernels.cuh" not in result.unresolved_includes
    assert len(seen) == len(set(seen))
    for rel in result.files:
        assert (dest / rel).read_bytes() == (
            upstream_source_root / "csrc" / rel
        ).read_bytes()
    header = (dest / _TOPK_HEADER.removeprefix("csrc/")).read_text()
    assert f"VLLM_TOPK_SOFTPLUS_SQRT_EXTERN_PAIR({output_type}, {input_type})" in header
    assert next(e for e in shipped if e.path == path).ops == ()


@pytest.mark.upstream_source
def test_topk_real_binding_pruner_preserves_complete_schema(
    shipped, upstream_source_root
):
    """对完整真实 binding 使用生产 pruner；这里不替代真实 Torch schema parsing。"""
    from vllm_sail.native import porting

    source = (upstream_source_root / "csrc/moe/torch_bindings.cpp").read_text()
    pruner = porting._BindingPruner(
        {op: entry for entry in shipped for op in entry.ops},
        frozenset(shipped.compiled_tiers),
        shipped.hazards,
    )
    pruned = pruner.prune(source)
    schemas = [
        porting._schema(match.group(0))
        for match in re.finditer(r"\bm\.def\s*\(.*?\);", pruned, re.S)
        if porting._op_name(match.group(0), "def") == "topk_softplus_sqrt"
    ]
    assert schemas == [
        "topk_softplus_sqrt(Tensor! topk_weights, Tensor! topk_indices, Tensor! "
        "token_expert_indices, Tensor gating_output, bool renormalize, float "
        "routed_scaling_factor, Tensor? bias, Tensor? input_ids, Tensor? tid2eid, "
        "Tensor? is_padding, Tensor? bias_vl=None, int image_sentinel_lo=0) -> ()"
    ]
    assert (
        pruned.count('m.impl("topk_softplus_sqrt", TORCH_BOX(&topk_softplus_sqrt));')
        == 1
    )
    assert "topk_softplus_sqrt" not in pruner.unclaimed()
    assert all(op.name != "topk_softplus_sqrt" for op in pruner.excluded())


@pytest.mark.upstream_source
def test_topk_real_header_overlay_preserves_launch_arguments_and_kernel_body(
    shipped, upstream_source_root, tmp_path
):
    """只替换已核实的名称以提供 post-sailify 边界；不证明 translator、SDK 或编译。"""
    from vllm_sail.native import porting

    source = (upstream_source_root / _TOPK_HEADER).read_text()
    names = {
        "__CUDA_ARCH__": "COMPATIBLE_ARCH",
        "cudaLaunchConfig_t": "hggcLaunchConfig_t",
        "cudaLaunchKernelEx": "hggcLaunchKernelEx",
        "cudaGridDependencySynchronize": "hggcGridDependencySynchronize",
        "cudaTriggerProgrammaticLaunchCompletion": "hggcTriggerProgrammaticLaunchCompletion",
    }
    translated = re.sub(
        r"\b(" + "|".join(names) + r")\b", lambda match: names[match[0]], source
    )
    unit = tmp_path / "moe/topk_softplus_sqrt_kernels.cuh"
    unit.parent.mkdir()
    unit.write_text(translated)
    replacements = shipped.replace_text[_TOPK_HEADER]
    assert len(replacements) == 9
    assert all(translated.count(before) == 1 for before in replacements)
    porting._replace_text([unit], tmp_path, {_TOPK_HEADER: replacements})
    adapted = unit.read_text()
    restored = adapted
    for before, after in reversed(list(replacements.items())):
        assert restored.count(after) == 1
        restored = restored.replace(after, before, 1)
    assert restored == translated
    guards = re.findall(
        r"(?m)^[ \t]*#if([^\n]*)\n[ \t]*hggc(?:GridDependencySynchronize|TriggerProgrammaticLaunchCompletion)\(\);",
        adapted,
    )
    assert len(guards) == 5 and all("!defined(__HGGCCC__)" in guard for guard in guards)
    ordinary = re.findall(r"kernel<<<(.*?)>>>\((.*?)\);", adapted, re.S)
    extended = re.findall(
        r"hggcLaunchKernelEx\(&config, kernel,(.*?)\);", adapted, re.S
    )
    assert len(ordinary) == len(extended) == 2

    def compact(text):
        return re.sub(r"\s+", "", text)

    assert [compact(args) for _, args in ordinary] == [
        compact(args) for args in extended
    ]
    assert compact(ordinary[0][0]) == "(num_rows+3)/4,128,0,stream"
    assert compact(ordinary[1][0]) == "num_blocks,block_dim,0,stream"
    assert (
        "#if !defined(USE_ROCM) && !defined(__HGGCCC__)\n    hggcLaunchConfig_t"
        in adapted
    )
    assert adapted.count("if (num_rows == 0) return;") == 2
    helper = adapted.split("void topkGatingSoftplusSqrtLauncherHelper(", 1)[1].split(
        "// Launches the kernel", 1
    )[0]
    guard = "#if defined(__HGGCCC__)\n  if (num_rows == 0) return;\n#endif"
    assert helper.count(guard) == 1
    assert helper.index(guard) < helper.index("const int num_warps")
    original_helper = translated.split("void topkGatingSoftplusSqrtLauncherHelper(", 1)[
        1
    ].split("// Launches the kernel", 1)[0]
    assert "if (num_rows == 0) return;" not in original_helper
    assert (
        adapted.count("const float scale = static_cast<float>(routed_scaling_factor);")
        == 2
    )
    assert (upstream_source_root / _TOPK_HEADER).read_text() == source


def test_spec_section_2_3_lists_sixteen_borrowed_ops() -> None:
    assert len(BORROWED_OPS) == 16
    assert sorted(BORROWED_OPS) == list(BORROWED_OPS)


def test_tier_a_registers_every_borrowed_op(shipped: m.Manifest) -> None:
    """Tier a must cover everything the plugin already calls.

    Anything here other than the two accepted CUTLASS regressions is a silent
    breakage of an existing PPU code path: the op resolves to nothing and the
    call site fails at runtime.
    """
    tier_a = set(shipped.ops(shipped.of_tier("a")))
    missing = sorted(set(BORROWED_OPS) - ACCEPTED_REGRESSIONS - tier_a)
    assert missing == []


def test_accepted_regressions_are_excluded_and_stubbable(shipped: m.Manifest) -> None:
    assert ACCEPTED_REGRESSIONS <= set(shipped.excluded_ops())


@pytest.mark.upstream_source
def test_lut_b_sm107_exclusion_follows_real_conditional_binding(
    shipped, upstream_source_root
):
    """生产 pruner 保留完整 compute schema；不执行 C++ 或 Torch schema parser。"""
    from vllm_sail.native import porting
    from vllm_sail.native.manifest import tomllib

    name = "lut_b_moe_gemm"
    entries = [e for e in shipped if name in e.ops]
    assert len(entries) == 1
    entry = entries[0]
    assert entry.path == "csrc/moe/lut_b_moe_sm107.cu"
    assert entry.tier == "x" and entry not in shipped.compiled()
    assert entry.hazards == ("tcgen05", "cp_async", "ptx")
    assert entry.vendored not in sum(shipped.compiled_sources(), ())
    source = (upstream_source_root / "csrc/moe/torch_bindings.cpp").read_text()
    assert source.count("#ifdef VLLM_ENABLE_LUT_B_MOE") == 2
    pruner = porting._BindingPruner(
        {op: e for e in shipped for op in e.ops},
        frozenset(shipped.compiled_tiers),
        shipped.hazards,
    )
    pruned = pruner.prune(source)
    assert f'"{name}(' not in pruned and f"TORCH_BOX(&{name})" not in pruned
    assert pruned.count("#ifdef VLLM_ENABLE_LUT_B_MOE") == 2
    assert name not in pruner.unclaimed()
    excluded = [op for op in pruner.excluded() if op.name == name]
    assert len(excluded) == 1
    op = excluded[0]
    assert (op.namespace, op.tier, op.kind) == ("_moe_C", "x", "compute")
    assert op.schema == (
        "lut_b_moe_gemm(Tensor a, Tensor a_scales, Tensor b, Tensor! c, "
        "Tensor? topk_weights, Tensor sorted_token_ids, Tensor expert_ids, "
        "Tensor num_tokens_post_padded, int top_k) -> ()"
    )
    assert op.reason == "; ".join(shipped.hazards[h] for h in entry.hazards)
    rendered = tomllib.loads(porting.render_excluded_ops(shipped, excluded))["op"]
    assert rendered == [
        dict(
            namespace=op.namespace,
            name=op.name,
            schema=op.schema,
            kind=op.kind,
            tier=op.tier,
            reason=op.reason,
        )
    ]


@pytest.mark.upstream_source
def test_sm100_hc_exclusion_and_marlin_schemas_follow_real_bindings(
    shipped,
    upstream_source_root,
):
    """对完整真实 binding 执行生产 pruner；不模拟 Torch schema parser 或设备。"""
    from vllm_sail.native import porting

    name = "qwen4_exp_hc_up_gate_mix"
    entry = next(e for e in shipped if name in e.ops)
    assert entry.path == "csrc/qwen4_exp/hc_up_gate_mix.cu"
    assert entry.tier == "x" and "tcgen05" in entry.hazards
    assert entry not in shipped.compiled()
    owner = {op: e for e in shipped for op in e.ops}
    pruner = porting._BindingPruner(
        owner, frozenset(shipped.compiled_tiers), shipped.hazards
    )
    for path in shipped.prune_bindings:
        source = (upstream_source_root / path).read_text()
        pruned = pruner.prune(source)
        assert f'"{name}(' not in pruned
        assert f"TORCH_BOX(&{name})" not in pruned
    excluded = {(op.namespace, op.name): op for op in pruner.excluded()}
    hc = excluded[("_C", name)]
    assert (
        hc.schema
        == "qwen4_exp_hc_up_gate_mix(Tensor! out, Tensor lowrank, Tensor xn, Tensor w_up) -> ()"
    )
    assert hc.tier == "x" and hc.kind == "compute"
    assert hc.reason == "; ".join(shipped.hazards[h] for h in entry.hazards)
    assert name not in pruner.unclaimed()
    for op in ("gptq_marlin_repack", "awq_marlin_repack"):
        assert excluded[("_C", op)].schema == (
            f"{op}(Tensor b_q_weight, SymInt size_k, SymInt size_n, int num_bits) -> Tensor"
        )
    for namespace, op in (("_C", "marlin_gemm"), ("_moe_C", "moe_wna16_marlin_gemm")):
        assert "a_scales" not in excluded[(namespace, op)].schema
    assert all(op.name != "marlin_int4_fp8_preprocess" for op in pruner.excluded())
    assert 'name = "marlin_int4_fp8_preprocess"' not in porting.render_excluded_ops(
        shipped, pruner.excluded()
    )


@pytest.mark.parametrize(
    "op",
    [
        "fused_deepseek_v4_kv_rope_insert",
        "fused_minimax_m3_qknorm_rope_kv_insert",
        "fused_kimi_k3_mla_decode_q_concat_ds_mla_insert",
        "fused_kimi_k3_mla_decode_q_concat_kv_cache_fp8_insert",
        "fused_kimi_k3_mla_decode_q_concat_kv_cache_insert",
        "fused_kimi_k3_mla_key_concat_ds_mla_insert",
        "fused_kimi_k3_mla_key_concat_kv_cache_insert",
        "fused_kimi_k3_mla_qkv_quant_kv_cache_fp8_insert",
        "concat_and_cache_mla_rope_fused",
        "selective_scan_fwd",
        "rms_norm_static_fp8_quant",
        "fused_add_rms_norm_static_fp8_quant",
        "rms_norm_dynamic_per_token_quant",
        "rms_norm_per_block_quant",
        "silu_and_mul_per_block_quant",
        "fused_qk_norm_rope",
        "fused_deepseek_v4_kv_rope_insert",
    ],
)
def test_model_kernel_dependencies_are_compiled_and_bound(shipped, op):
    """Check the generated registration as well as the requested build set."""
    assert op in shipped.ops(shipped.compiled())
    root = Path(__file__).parents[2]
    bindings = (root / "csrc/upstream/torch_bindings.cpp").read_text()
    assert f'"{op}(' in bindings or f'"{op}( ' in bindings
    namespace = "_C_cache_ops" if op == "concat_and_cache_mla_rope_fused" else "_C"
    cuda_block = bindings.split(
        f"STABLE_TORCH_LIBRARY_IMPL({namespace}, CUDA, ops) {{"
    )[1].split("\n}")[0]
    assert f"TORCH_BOX(&{op})" in cuda_block
    from vllm_sail.native.stubs import load_schemas

    assert op not in {entry.name for entry in load_schemas()}


def test_step5_qknorm_rope_cache_is_compiled_and_bound(shipped: m.Manifest) -> None:
    # Step5 attention calls this for every non-fused-prepare forward; upstream
    # binds it through the vllm:: namespace rather than a bare symbol.
    op = "fused_qknorm_rope_cache"
    assert op in shipped.ops(shipped.compiled())
    root = Path(__file__).parents[2]
    bindings = (root / "csrc/upstream/torch_bindings.cpp").read_text()
    assert f'"{op}("' in bindings
    cuda_block = bindings.split("STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, ops) {")[1]
    assert f"TORCH_BOX(&vllm::{op})" in cuda_block.split("\n}")[0]
    from vllm_sail.native.stubs import load_schemas

    assert op not in {entry.name for entry in load_schemas()}


@pytest.mark.parametrize(
    "op",
    [
        "topk_softplus_sqrt",
        "fused_deepseek_v4_kv_rope_insert",
        "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert",
        "fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert",
        "fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert",
    ],
)
def test_deepseek_v4_routing_and_cache_ops_are_compiled(shipped: m.Manifest, op: str):
    # These are reached by DeepSeek V4 profiling and the first real forward.
    assert op in shipped.ops(shipped.compiled())
    assert op not in shipped.excluded_ops()


@pytest.mark.parametrize("op", PLUGIN_OWNED_OPS)
def test_plugin_original_ops_are_not_in_the_upstream_corpus(
    shipped: m.Manifest, op: str
) -> None:
    assert op not in shipped.ops()


def test_python_wrapper_is_not_mistaken_for_an_op(shipped: m.Manifest) -> None:
    """`scaled_int8_quant` is vllm/_custom_ops.py glue, so no TU may claim it."""
    assert "scaled_int8_quant" not in shipped.ops()


_VALID = """\
schema = 1
upstream_ref = "v0.27.1"
compiled_tiers = ["a"]

[tiers]
a = "compiled"
x = "excluded"

[hazards]
cutlass = "CUTLASS / CuTe in the include closure"

[[entry]]
path = "csrc/libtorch_stable/layernorm_kernels.cu"
tier = "a"
ops = ["rms_norm"]
"""


def _write(tmp_path: Path, body: str) -> Path:
    src = tmp_path / "manifest.toml"
    src.write_text(textwrap.dedent(body), encoding="utf-8")
    return src


def test_the_reference_manifest_used_by_the_refusal_cases_is_valid(
    tmp_path: Path,
) -> None:
    """Each refusal case below mutates this one document, so it must load."""
    loaded = m.load(_write(tmp_path, _VALID))
    assert loaded.upstream_ref == "v0.27.1"
    assert loaded.compiled_tiers == ("a",)
    assert [e.path for e in loaded.compiled()] == [
        "csrc/libtorch_stable/layernorm_kernels.cu"
    ]


@pytest.mark.parametrize(
    ("body", "message"),
    [
        pytest.param(
            """\
            schema = 2
            upstream_ref = "v0.27.1"
            compiled_tiers = ["a"]
            [tiers]
            a = "compiled"
            [[entry]]
            path = "a.cu"
            tier = "a"
            """,
            "this reader speaks 1",
            id="wrong-schema",
        ),
        pytest.param(
            """\
            schema = 1
            compiled_tiers = ["a"]
            [tiers]
            a = "compiled"
            [[entry]]
            path = "a.cu"
            tier = "a"
            """,
            "upstream_ref must be a non-empty",
            id="missing-upstream-ref",
        ),
        pytest.param(
            """\
            schema = 1
            upstream_ref = "   "
            compiled_tiers = ["a"]
            [tiers]
            a = "compiled"
            [[entry]]
            path = "a.cu"
            tier = "a"
            """,
            "upstream_ref must be a non-empty",
            id="blank-upstream-ref",
        ),
        pytest.param(
            """\
            schema = 1
            upstream_ref = "v0.27.1"
            compiled_tiers = ["a"]
            [[entry]]
            path = "a.cu"
            tier = "a"
            """,
            "no tiers declared",
            id="no-tiers-table",
        ),
        pytest.param(
            """\
            schema = 1
            upstream_ref = "v0.27.1"
            [tiers]
            a = "compiled"
            [[entry]]
            path = "a.cu"
            tier = "a"
            """,
            "compiled_tiers is empty or missing",
            id="missing-compiled-tiers",
        ),
        pytest.param(
            """\
            schema = 1
            upstream_ref = "v0.27.1"
            compiled_tiers = []
            [tiers]
            a = "compiled"
            [[entry]]
            path = "a.cu"
            tier = "a"
            """,
            "compiled_tiers is empty or missing",
            id="empty-compiled-tiers",
        ),
        pytest.param(
            """\
            schema = 1
            upstream_ref = "v0.27.1"
            compiled_tiers = ["a", "z"]
            [tiers]
            a = "compiled"
            [[entry]]
            path = "a.cu"
            tier = "a"
            """,
            "compiled_tiers names unknown tier 'z'",
            id="compiled-tier-undeclared",
        ),
        pytest.param(
            """\
            schema = 1
            upstream_ref = "v0.27.1"
            compiled_tiers = ["a"]
            [tiers]
            a = "compiled"
            [[entry]]
            tier = "a"
            """,
            "an entry has no path",
            id="entry-without-path",
        ),
        pytest.param(
            """\
            schema = 1
            upstream_ref = "v0.27.1"
            compiled_tiers = ["a"]
            [tiers]
            a = "compiled"
            [[entry]]
            path = "a.cu"
            tier = "q"
            """,
            "a.cu: unknown tier 'q'",
            id="entry-tier-undeclared",
        ),
        pytest.param(
            """\
            schema = 1
            upstream_ref = "v0.27.1"
            compiled_tiers = ["a"]
            [tiers]
            a = "compiled"
            [hazards]
            cub = "CUB"
            [[entry]]
            path = "a.cu"
            tier = "a"
            hazards = ["cub", "tcgen05"]
            """,
            "a.cu: unknown hazard 'tcgen05'",
            id="entry-hazard-undeclared",
        ),
        pytest.param(
            """\
            schema = 1
            upstream_ref = "v0.27.1"
            compiled_tiers = ["a"]
            [tiers]
            a = "compiled"
            [[entry]]
            path = "a.cu"
            tier = "a"
            [[entry]]
            path = "a.cu"
            tier = "a"
            """,
            "duplicate entry 'a.cu'",
            id="duplicate-path",
        ),
        pytest.param(
            """\
            schema = 1
            upstream_ref = "v0.27.1"
            compiled_tiers = ["a"]
            [tiers]
            a = "compiled"
            [[entry]]
            path = "a.cu"
            tier = "a"
            ops = ["rms_norm"]
            [[entry]]
            path = "b.cu"
            tier = "a"
            ops = ["rms_norm"]
            """,
            "op 'rms_norm' is claimed by both a.cu and b.cu",
            id="op-claimed-twice",
        ),
        pytest.param(
            """\
            schema = 1
            upstream_ref = "v0.27.1"
            compiled_tiers = ["a"]
            [tiers]
            a = "compiled"
            [hazards]
            cutlass = "CUTLASS"
            [[entry]]
            path = "a.cu"
            tier = "a"
            hazards = ["cutlass"]
            """,
            "has a cutlass hazard but sits in compiled tier",
            id="cutlass-in-compiled-tier",
        ),
        pytest.param(
            """\
            schema = 1
            upstream_ref = "v0.27.1"
            compiled_tiers = ["a"]
            [tiers]
            a = "compiled"
            [overlay]
            prune_bindings = ["csrc/nope/torch_bindings.cpp"]
            [[entry]]
            path = "a.cu"
            tier = "a"
            """,
            "prune_bindings names unknown 'csrc/nope/torch_bindings.cpp'",
            id="prune-bindings-unknown",
        ),
        pytest.param(
            """\
            schema = 1
            upstream_ref = "v0.27.1"
            compiled_tiers = ["a"]
            [tiers]
            a = "compiled"
            """,
            "no entries",
            id="no-entries",
        ),
    ],
)
def test_load_refuses_malformed_manifest(
    tmp_path: Path, body: str, message: str
) -> None:
    with pytest.raises(m.ManifestError, match=re.escape(message)):
        m.load(_write(tmp_path, body))


def test_load_refuses_compiled_tiers_captured_by_the_tiers_table(
    tmp_path: Path,
) -> None:
    """The TOML footgun this manifest's own comment warns about.

    A bare key written *after* a table header belongs to that table, so
    ``compiled_tiers`` written below ``[tiers]`` becomes ``tiers.compiled_tiers``
    and the top-level key vanishes. That must fail loudly rather than quietly
    compile nothing.
    """
    body = """\
    schema = 1
    upstream_ref = "v0.27.1"

    [tiers]
    a = "compiled"
    compiled_tiers = ["a"]

    [[entry]]
    path = "a.cu"
    tier = "a"
    """
    with pytest.raises(m.ManifestError, match="compiled_tiers is empty or missing"):
        m.load(_write(tmp_path, body))


def test_load_refuses_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(m.ManifestError, match="manifest not found"):
        m.load(tmp_path / "absent" / "manifest.toml")
