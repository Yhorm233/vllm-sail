# SPDX-License-Identifier: Apache-2.0
"""Portable packing layout invariants and optional independent CPU reference."""

import ast
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from tests.support.source import function

ROOT = Path(__file__).parents[2]


def _portable():
    spec = importlib.util.spec_from_file_location(
        "portable", ROOT / "vllm_sail/native/portable.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("activation8", [False, True])
def test_weight_layout_permutation_is_bijective(bits, activation8):
    permutation = _portable()._weight_permutation(bits, activation8)
    assert len(permutation) == 1024
    assert sorted(permutation) == list(range(1024))


@pytest.mark.parametrize("bits", [4, 8])
def test_public_repack_rejects_removed_activation_argument_before_import(bits):
    portable = _portable()
    with pytest.raises(TypeError):
        portable.gptq_marlin_repack(None, 32, 64, bits, True)


@pytest.mark.upstream_source
def test_registered_portable_schema_matches_exact_upstream_binding(
    upstream_source_root, modules
):
    portable = _portable()
    definitions = []

    class Library:
        def __init__(self, namespace, kind):
            assert namespace == "_C" and kind == "FRAGMENT"

        def define(self, schema):
            definitions.append(schema)

        def impl(self, name, implementation, dispatch):
            assert name == "gptq_marlin_repack" and dispatch == "CUDA"
            assert implementation is portable.gptq_marlin_repack

    modules(
        "torch",
        ops=NS(_C=NS(per_token_group_fp8_quant_ppu_opt=object())),
        library=NS(Library=Library),
    )
    portable.install()
    portable.install()
    assert definitions == [
        "gptq_marlin_repack(Tensor b_q_weight, SymInt size_k, SymInt size_n, int num_bits) -> Tensor"
    ]
    binding = (upstream_source_root / "csrc/torch_bindings.cpp").read_text()
    assert '"gptq_marlin_repack(Tensor b_q_weight, "' in binding
    assert '"SymInt size_k, SymInt size_n, int num_bits) -> Tensor"' in binding


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("activation8", [False, True])
@pytest.mark.requires_runtime
def test_packed_weights_match_upstream_independent_reference(bits, activation8):
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    source = Path(os.environ.get("VLLM_SOURCE_ROOT", ROOT.parent / "vllm"))
    reference = (
        source / "vllm/model_executor/layers/quantization/utils/marlin_utils_test.py"
    )
    if not reference.exists():
        pytest.skip(
            "set VLLM_SOURCE_ROOT to the pinned upstream checkout for reference packing"
        )
    functions = [
        node
        for node in ast.parse(reference.read_text()).body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("get_weight_perm", "marlin_permute_weights", "marlin_weights")
    ]
    namespace = {
        "torch": torch,
        "np": np,
        "GPTQ_MARLIN_TILE": 16,
        "get_pack_factor": lambda bits: 32 // bits,
    }
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(reference), "exec"),
        namespace,
    )
    generator = torch.Generator().manual_seed(17)
    portable = _portable()
    if activation8:
        with pytest.raises(TypeError):
            portable.gptq_marlin_repack(None, 32, 64, bits, True)
        return
    caller = function(
        source / "vllm/_custom_ops.py",
        "gptq_marlin_repack",
        {"torch": NS(ops=NS(_C=NS(gptq_marlin_repack=portable.gptq_marlin_repack)))},
    )
    for k, n in ((32, 64), (64, 128), (256, 256)):
        weights = torch.randint(
            0, 1 << bits, (k, n), dtype=torch.int32, generator=generator
        )
        pack = 32 // bits
        gptq = torch.zeros((k // pack, n), dtype=torch.int32)
        for i in range(pack):
            gptq |= weights[i::pack] << (bits * i)
        actual = caller(gptq, k, n, bits)
        expected = namespace["marlin_weights"](
            weights,
            k,
            n,
            bits,
            namespace["get_weight_perm"](bits),
        )
        assert torch.equal(actual, expected)


@pytest.mark.upstream_source
@pytest.mark.requires_runtime
@pytest.mark.parametrize("frontend", ["gptq", "compressed"])
@pytest.mark.parametrize("backend", ["PPU_DEEPGEMM", "BATCHED_PPU_DEEPGEMM"])
@pytest.mark.parametrize("experts", [1, 3])
@pytest.mark.parametrize("intermediate", [64, 96])
def test_real_wna16_ppu_conversion_preserves_expert_weights_scales_and_padding(
    upstream_source_root, modules, frontend, backend, experts, intermediate
):
    """真实 adapter、转换、padding、scales 与 MoE wrapper；配置类型和 op 注册是边界，不证明 GEMM。"""
    import inspect
    import math
    from enum import Enum

    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    source = upstream_source_root / "vllm"
    oracle_path = source / "model_executor/layers/fused_moe/oracle/int_wna16.py"
    utils_path = source / "model_executor/layers/quantization/utils/marlin_utils.py"
    reference_path = (
        source / "model_executor/layers/quantization/utils/marlin_utils_test.py"
    )
    portable = _portable()
    calls, warnings = [], []

    def repack(weight, size_k, size_n, num_bits):
        assert num_bits == 4 and weight.is_contiguous()
        assert weight.shape == (size_k // 8, size_n)
        calls.append((size_k, size_n, num_bits))
        return portable.gptq_marlin_repack(weight, size_k, size_n, num_bits)

    moe_repack = function(
        source / "_custom_ops.py",
        "gptq_marlin_moe_repack",
        {"torch": NS(empty=torch.empty, ops=NS(_C=NS(gptq_marlin_repack=repack)))},
    )
    namespace = dict(
        torch=torch,
        math=math,
        Enum=Enum,
        ops=NS(gptq_marlin_moe_repack=moe_repack),
        logger=NS(warning_once=lambda *args: warnings.append(args)),
        current_platform=NS(is_rocm=lambda: False),
    )
    function(source / "utils/math_utils.py", "round_up", namespace)
    for name in (
        "marlin_moe_padded_intermediate",
        "check_moe_marlin_supports_config",
        "get_scale_perms",
        "marlin_permute_scales",
        "marlin_moe_permute_scales",
    ):
        function(utils_path, name, namespace)
    for name in (
        "WNA16MoEBackend",
        "_pad_w13_shard_cols",
        "_pad_rows",
        "_process_weights_marlin",
    ):
        function(oracle_path, name, namespace)

    class GPTQConfig:
        quant_type = NS(size_bits=4)
        pack_factor, group_size, desc_act = 8, 32, False

    class CompressedConfig:
        num_bits, group_size = 4, 32

    class AWQConfig:
        pass

    modules(
        "vllm.model_executor.layers.quantization.auto_gptq", AutoGPTQConfig=GPTQConfig
    )
    modules("vllm.model_executor.layers.quantization.auto_awq", AutoAWQConfig=AWQConfig)
    modules("compressed_tensors.quantization", QuantizationArgs=CompressedConfig)
    namespace["QuantizationArgs"] = CompressedConfig
    upstream_convert = function(
        oracle_path, "convert_to_wna16_moe_kernel_format", namespace
    )
    adapter_namespace = dict(
        inspect=inspect,
        _upstream_convert=upstream_convert,
        _PPU=("PPU_DEEPGEMM", "BATCHED_PPU_DEEPGEMM"),
        WNA16MoEBackend=namespace["WNA16MoEBackend"],
    )
    adapter_path = "vllm_sail/registry/moe_backends/int_wna16.py"
    convert = function(
        adapter_path, "convert_to_wna16_moe_kernel_format", adapter_namespace
    )
    modules(
        "vllm_sail.patch.enhancement.models.moe_marlin_gate",
        supports_marlin_layout=namespace["check_moe_marlin_supports_config"],
    )
    reason = function(
        adapter_path, "_backend_incompatibility_reason", adapter_namespace
    )
    quant = GPTQConfig() if frontend == "gptq" else CompressedConfig()
    hidden, group, pack = 128, 32, 8
    padded = ((intermediate + 63) // 64) * 64
    config = NS(
        has_bias=False,
        hidden_dim=hidden,
        intermediate_size_per_partition_unpadded=intermediate,
    )
    assert reason(backend, config, quant, False, False, True) is None
    generator = torch.Generator().manual_seed(41)
    raw13 = torch.randint(
        0,
        16,
        (experts, hidden, 2 * intermediate),
        dtype=torch.int32,
        generator=generator,
    )
    raw2 = torch.randint(
        0, 16, (experts, intermediate, hidden), dtype=torch.int32, generator=generator
    )

    def canonical(raw):
        e, k, n = raw.shape
        packed = torch.zeros((e, k // pack, n), dtype=torch.int32)
        for i in range(pack):
            packed |= raw[:, i::pack, :] << (4 * i)
        return packed.transpose(1, 2).contiguous()

    scales13 = (
        torch.arange(
            experts * 2 * intermediate * (hidden // group), dtype=torch.float32
        ).reshape(experts, 2 * intermediate, hidden // group)
        + 1
    )
    scales2 = (
        torch.arange(
            experts * hidden * (intermediate // group), dtype=torch.float32
        ).reshape(experts, hidden, intermediate // group)
        + 10001
    )
    inputs = (canonical(raw13), canonical(raw2), scales13, scales2)
    before = tuple(value.clone() for value in inputs)
    layer = NS(intermediate_size_per_partition=intermediate)
    actual = convert(
        backend=backend,
        layer=layer,
        quant_config=quant,
        w13=inputs[0],
        w2=inputs[1],
        w13_scale=scales13,
        w2_scale=scales2,
    )
    assert len(actual) == 8 and actual[4:] == (None, None, None, None)
    assert (
        calls == [(hidden, 2 * padded, 4)] * experts + [(padded, hidden, 4)] * experts
    )
    assert bool(warnings) is (padded != intermediate)
    assert vars(layer) == {"intermediate_size_per_partition": intermediate}
    assert all(
        torch.equal(value, snapshot)
        for value, snapshot in zip(inputs, before, strict=False)
    )

    # 独立构造每个 gate/up shard 的 padding；不以被测 padding/helper 自证。
    expected13 = torch.zeros((experts, hidden, 2 * padded), dtype=torch.int32)
    expected13[:, :, :intermediate] = raw13[:, :, :intermediate]
    expected13[:, :, padded : padded + intermediate] = raw13[:, :, intermediate:]
    expected2 = torch.zeros((experts, padded, hidden), dtype=torch.int32)
    expected2[:, :intermediate, :] = raw2
    reference = dict(
        torch=torch, np=np, GPTQ_MARLIN_TILE=16, get_pack_factor=lambda bits: 32 // bits
    )
    for name in ("get_weight_perm", "marlin_permute_weights", "marlin_weights"):
        function(reference_path, name, reference)
    for packed, raw in zip(actual[:2], (expected13, expected2), strict=False):
        k, n = raw.shape[1:]
        expected = torch.stack(
            [
                reference["marlin_weights"](
                    weight, k, n, 4, reference["get_weight_perm"](4)
                )
                for weight in raw
            ]
        )
        assert packed.shape == (experts, k // 16, n * 2)
        assert packed.dtype == torch.int32 and packed.is_contiguous()
        assert torch.equal(packed, expected)

    # group32 scales 的 8×8 转置顺序是独立 oracle，不调用真实 permutation helper。
    expected_s13 = torch.zeros((experts, hidden // group, 2 * padded))
    expected_s13[:, :, :intermediate] = scales13.transpose(1, 2)[:, :, :intermediate]
    expected_s13[:, :, padded : padded + intermediate] = scales13.transpose(1, 2)[
        :, :, intermediate:
    ]
    expected_s2 = torch.zeros((experts, padded // group, hidden))
    expected_s2[:, : intermediate // group, :] = scales2.transpose(1, 2)
    for scales, raw in zip(actual[2:4], (expected_s13, expected_s2), strict=False):
        expected = raw.reshape(experts, -1, 8, 8).transpose(-1, -2).reshape(raw.shape)
        assert scales.shape == raw.shape and scales.dtype == torch.float32
        assert scales.is_contiguous() and torch.equal(scales, expected)

    # 将真实转换结果交给实际标准／batched expert 的形状消费者；不执行构造或 GEMM。
    base_size = function(
        source / "model_executor/layers/fused_moe/modular_kernel.py",
        "FusedMoEExpertsModular.moe_problem_size",
        {},
    )
    base = type("ExpertBaseBoundary", (), {"moe_problem_size": base_size})
    batched = backend == "BATCHED_PPU_DEEPGEMM"
    class_name = "PPUBatchedDeepGemmExperts" if batched else "PPUDeepGemmExperts"
    filename = "batched_deep_gemm_moe.py" if batched else "deep_gemm_moe.py"
    cls = function(
        "vllm_sail/model_executor/layers/fused_moe/experts/" + filename,
        class_name,
        {"mk": NS(FusedMoEExpertsModular=base)},
    )
    owner = object.__new__(cls)
    owner.quant_config = NS(use_mxfp4_w4a16=False, use_int4_w4a16=True)
    activations = torch.empty((experts, 5, hidden) if batched else (5, hidden))
    topk_ids = torch.zeros((5, 2), dtype=torch.int32)
    assert owner.moe_problem_size(activations, actual[0], actual[1], topk_ids) == (
        experts,
        5,
        2 * padded,
        hidden,
        2,
    )


@pytest.mark.upstream_source
@pytest.mark.parametrize("wrapper", ["dense", "dispatcher", "moe"])
@pytest.mark.parametrize("explicit_options", [False, True])
def test_real_marlin_wrappers_preserve_native_arguments_after_scale_removal(
    upstream_source_root, wrapper, explicit_options
):
    """真实 Python wrapper 的完整参数行序；native 仅捕获调用，不证明 PPU GEMM。"""
    calls = []
    result = object()

    def capture(*args):
        calls.append(args)
        return result

    source = upstream_source_root / "vllm"
    namespace = dict(
        torch=NS(
            ops=NS(
                _C=NS(marlin_gemm=capture),
                _moe_C=NS(moe_wna16_marlin_gemm=capture),
            )
        )
    )
    scalar = NS(id=37)
    if wrapper == "moe":
        fn = function(source / "_custom_ops.py", "moe_wna16_marlin_gemm", namespace)
        inputs = tuple(object() for _ in range(12))
        options = (
            dict(thread_k=64, thread_n=128, blocks_per_sm=2) if explicit_options else {}
        )
        assert (
            fn(*inputs, 16, 2, True, scalar, 3, 128, 64, False, True, False, **options)
            is result
        )
        expected = (
            inputs
            + (16, 2, True, 37, 3, 128, 64, False, True, False)
            + ((64, 128, 2) if explicit_options else (-1, -1, -1))
        )
    else:
        path = (
            source / "_custom_ops.py"
            if wrapper == "dense"
            else (source / "model_executor/layers/quantization/utils/marlin_utils.py")
        )
        name = "marlin_gemm" if wrapper == "dense" else "_marlin_gemm"
        fn = function(path, name, namespace)
        inputs = tuple(object() for _ in range(8))
        options = (
            dict(use_atomic_add=True, use_fp32_reduce=True, is_zp_float=True)
            if explicit_options
            else {}
        )
        if wrapper == "dense":
            assert fn(*inputs, scalar, 3, 128, 64, **options) is result
        else:
            assert (
                fn(inputs[0], *inputs[2:], scalar.id, 3, 128, 64, **options) is result
            )
        expected = (
            inputs[0],
            inputs[1] if wrapper == "dense" else None,
            *inputs[2:],
        ) + (
            37,
            3,
            128,
            64,
            *((True, True, True) if explicit_options else (False, False, False)),
        )
    # 完整 tuple 同时约束 native 参数个数、global scale/zero/workspace 行序和所有选项。
    assert calls == [expected]


@pytest.mark.upstream_source
@pytest.mark.requires_runtime
@pytest.mark.parametrize("policy", ["gptq", "fp8", "mxfp8", "nvfp4", "mxfp4"])
@pytest.mark.parametrize("provided_workspace", [False, True])
@pytest.mark.parametrize("fp32_reduce", [False, True])
def test_real_dense_marlin_consumers_preserve_padding_and_new_gemm_contract(
    upstream_source_root, policy, provided_workspace, fp32_reduce
):
    """真实 consumer、padding 和 dispatcher；Torch 为真，GEMM 只返回预定行，不提供数值证据。"""
    torch = pytest.importorskip("torch")
    source = upstream_source_root / "vllm"
    utils = source / "model_executor/layers/quantization/utils/marlin_utils.py"
    bits = 8 if policy in ("fp8", "mxfp8") else 4
    quant = NS(id=11 if policy == "gptq" else (12 if bits == 8 else 13), size_bits=bits)
    inputs = torch.arange(2 * 3 * 96, dtype=torch.float32).reshape(2, 3, 96)
    before = inputs.clone()
    weight = torch.zeros((128 // 16, 128 * bits // 2), dtype=torch.int32)
    scales, bias = object(), object()
    zeros = object() if policy == "gptq" else None
    global_scale = object() if policy == "nvfp4" else None
    workspace = object()
    supplied = workspace if provided_workspace else None
    calls, workspace_reads = [], []
    native_output = torch.arange(6 * 128, dtype=inputs.dtype).reshape(6, 128)

    def native(*args):
        assert len(args) == 15
        a, c, w, b, s, global_s, zp, locks = args[:8]
        assert a.shape == (6, 128) and torch.equal(a[:, :96], inputs.reshape(6, 96))
        assert torch.count_nonzero(a[:, 96:]) == 0
        assert c is None and w is weight and b is bias and s is scales
        assert global_s is global_scale and zp is zeros and locks is workspace
        assert args[8:] == (quant.id, 6, 128, 128, False, fp32_reduce, False)
        calls.append("gemm")
        return native_output

    def resolve_workspace(device):
        assert device == inputs.device
        workspace_reads.append(device)
        return workspace

    class Torch:
        ops = NS(_C=NS(marlin_gemm=native), vllm=NS())

        def __getattr__(self, name):
            return getattr(torch, name)

    namespace = dict(
        torch=Torch(),
        GPTQ_MARLIN_TILE=16,
        USE_FP32_REDUCE_DEFAULT=True,
        scalar_types=NS(float8_e4m3fn=quant, float4_e2m1f=quant),
        get_marlin_workspace=resolve_workspace,
    )
    for name in (
        "marlin_repacked_nk",
        "marlin_pad_dim",
        "marlin_unpad_output",
        "should_use_atomic_add_reduce",
        "_marlin_gemm",
    ):
        function(utils, name, namespace)
    namespace["torch"].ops.vllm.marlin_gemm = namespace["_marlin_gemm"]
    if policy == "gptq":
        apply = function(utils, "apply_gptq_marlin_linear", namespace)
        output = apply(
            inputs,
            weight,
            scales,
            zeros,
            supplied,
            quant,
            96,
            96,
            bias=bias,
            use_fp32_reduce=fp32_reduce,
        )
    else:
        fp4 = policy in ("nvfp4", "mxfp4")
        path = utils.with_name("marlin_utils_fp4.py" if fp4 else "marlin_utils_fp8.py")
        name = (
            "apply_fp4_marlin_linear"
            if fp4
            else (
                "apply_mxfp8_marlin_linear"
                if policy == "mxfp8"
                else "apply_fp8_marlin_linear"
            )
        )
        apply = function(path, name, namespace)
        arguments = (inputs, weight, scales)
        if fp4:
            arguments += (global_scale,)
        output = apply(
            *arguments, supplied, 96, 96, bias=bias, use_fp32_reduce=fp32_reduce
        )
    assert calls == ["gemm"]
    assert workspace_reads == ([] if provided_workspace else [inputs.device])
    assert output.shape == (2, 3, 96) and output.is_contiguous()
    assert torch.equal(output, native_output[:, :96].reshape(2, 3, 96))
    assert torch.equal(inputs, before)
