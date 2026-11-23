# SPDX-License-Identifier: Apache-2.0
"""Exercise the complete upstream warmup and installed PPU import boundary."""

import ast
import contextlib
import importlib.abc
import importlib.util
import sys
from types import SimpleNamespace as NS

import pytest

from tests.support.modules import load_patch
from tests.support.source import ROOT, definition, function
from tests.ut.test_ppu_kernel_capabilities import complete_warmup_contract


@pytest.mark.upstream_source
def test_complete_warmup_body_matches_provider_outside_ppu_guards(upstream_source_root):
    copied = definition(
        ROOT / "vllm_sail/model_executor/warmup/ppu_kernel_warmup.py",
        "kernel_warmup",
    )
    original = definition(
        upstream_source_root / "vllm/model_executor/warmup/kernel_warmup.py",
        "kernel_warmup",
    )
    assert ast.unparse(copied.args.args[0].annotation) == "Worker"
    assert ast.literal_eval(original.args.args[0].annotation) == "Worker"
    copied.args.args[0].annotation = original.args.args[0].annotation
    for node in ast.walk(copied):
        if isinstance(node, ast.If):
            if (
                ast.unparse(node.test)
                == "current_platform.is_cuda() and (not current_platform.is_ppu())"
            ):
                node.test = ast.parse("current_platform.is_cuda()", mode="eval").body
            node.body = [
                child
                for statement in node.body
                for child in (
                    statement.body
                    if isinstance(statement, ast.If)
                    and ast.unparse(statement.test) == "not current_platform.is_ppu()"
                    else [statement]
                )
            ]
    assert ast.dump(copied, include_attributes=False) == ast.dump(
        original, include_attributes=False
    )


@pytest.mark.upstream_source
@pytest.mark.parametrize(
    "replay,backend", [(False, "flashinfer"), (True, "triton"), (True, "flashinfer")]
)
def test_real_replayssm_lifecycle_admission_uses_config_not_cuda_identity(
    upstream_source_root,
    replay,
    backend,
):
    prop = function(
        upstream_source_root / "vllm/config/vllm.py",
        "VllmConfig.use_flashinfer_replayssm",
        {"MambaBackendEnum": NS(FLASHINFER="flashinfer")},
    )
    config = NS(cache_config=NS(use_replayssm=replay), mamba_config=NS(backend=backend))
    assert prop(config) is (replay and backend == "flashinfer")
    if not prop(config):
        lifecycle = function(
            upstream_source_root / "vllm/model_executor/warmup/replayssm_warmup.py",
            "replayssm_lifecycle_warmup",
            {},
        )
        assert lifecycle(NS(vllm_config=NS(use_flashinfer_replayssm=False))) is None


@pytest.mark.upstream_source
@pytest.mark.requires_runtime
@pytest.mark.parametrize("v2", [False, True])
@pytest.mark.parametrize("align", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_real_replayssm_lifecycle_bounds_scratch_and_cleans_after_failure(
    modules,
    upstream_source_root,
    v2,
    align,
    fail,
):
    """完整生命周期和 temporary helper；仅替换外部 runner/model/native 边界。"""
    torch = pytest.importorskip("torch")
    calls = []
    error = RuntimeError("postprocess failed")
    path = upstream_source_root / "vllm/model_executor/warmup/replayssm_warmup.py"
    base = type("MambaBase", (), {})
    mixer_cls = type("MambaMixer2", (base,), {})
    spec_cls = type("MambaSpec", (), {})
    mixer = mixer_cls()
    mixer.use_replayssm = mixer.use_flashinfer_replayssm = True
    mixer.replayssm_buffer_len = 4
    mixer.kv_cache = [torch.full((5, 2), 7.0), torch.full((5, 3), 9.0)]
    mixer._replayssm_ring_start = torch.ones(5, dtype=torch.int32)
    mixer._replayssm_prev_num_accepted = torch.ones(5, dtype=torch.int32)
    spec = spec_cls()
    spec.mamba_cache_mode, spec.block_size = "align" if align else "all", 16
    table = torch.full((1, 4), 99, dtype=torch.int32)
    config = NS(
        use_flashinfer_replayssm=True,
        use_v2_model_runner=v2,
        compilation_config=NS(static_forward_context={"mixer": mixer}),
    )
    runner = NS(
        vllm_config=config,
        max_num_reqs=4,
        decode_query_len=2,
        uniform_decode_query_len=2,
        kv_cache_config=NS(
            kv_cache_groups=[NS(kv_cache_spec=spec, layer_names=["mixer"])]
        ),
        block_tables=NS(input_block_tables=[table]),
        input_batch=NS(block_table=[NS(get_device_tensor=lambda n: table)]),
        get_model=lambda: NS(modules=lambda: [mixer]),
    )
    modules("vllm.model_executor.layers.mamba.abstract", MambaBase=base)
    modules("vllm.model_executor.layers.mamba.mamba_mixer2", MambaMixer2=mixer_cls)
    modules("vllm.v1.kv_cache_interface", MambaSpec=spec_cls)
    modules(
        "vllm.v1.worker.extensible_kv_cache",
        num_committable_kv_blocks=lambda actual: 10,
        ensure_kv_cache_blocks=lambda actual, count: calls.append((actual, count)),
    )

    def reset(**kwargs):
        assert kwargs["src_cols"].tolist() == [-1]
        assert kwargs["dst_cols"].tolist() == [1]
        assert kwargs["num_reqs"] == 1
        calls.append("reset")

    def postprocess(**kwargs):
        assert kwargs["query_metadata"].tolist() == ([0, 2] if v2 else [2])
        assert kwargs["query_metadata_is_cumulative"] is v2
        assert kwargs["num_computed_tokens"].tolist() == ([16] if v2 else [15])
        assert kwargs["num_computed_is_post_step"] is v2
        assert kwargs["num_accepted_tokens"].tolist() == [1]
        assert kwargs["is_prefilling"].tolist() == [False]
        assert kwargs["num_reqs"] == 1
        assert (
            kwargs["idx_mapping"].tolist() == [0]
            if v2
            else kwargs["idx_mapping"] is None
        )
        assert (
            kwargs["live_cols"].tolist() == [1]
            if align
            else kwargs["live_cols"] is None
        )
        for state in mixer.kv_cache:
            assert torch.count_nonzero(state[1:3]).item() == 0
            state[1:3].fill_(42)
        mixer._replayssm_ring_start.fill_(42)
        calls.append("postprocess")
        if fail:
            raise error

    def create(grouped, max_reqs):
        assert max_reqs == 4 and len(grouped) == 1
        actual_mixers, actual_spec, scratch = grouped[0]
        assert actual_mixers == [mixer] and actual_spec is spec
        assert scratch is not table and scratch.stride() == table.stride()
        assert scratch.tolist() == [[1, 2]]
        return NS(reset_new_slots=reset, postprocess=postprocess)

    modules("vllm.v1.worker.replayssm_utils", ReplaySSMModelContext=NS(create=create))
    namespace = {"torch": torch}
    temporary = contextlib.contextmanager(
        function(path, "_temporary_replayssm_autotune_state", namespace)
    )
    namespace["_temporary_replayssm_autotune_state"] = temporary
    lifecycle = function(path, "replayssm_lifecycle_warmup", namespace)
    if fail:
        with pytest.raises(RuntimeError) as raised:
            lifecycle(runner)
        assert raised.value is error
    else:
        assert lifecycle(runner) is None
    assert calls == [(runner, 3), *(["reset"] if align else []), "postprocess"]
    assert table.tolist() == [[99, 99, 99, 99]]
    for state, value in zip(mixer.kv_cache, [7, 9], strict=False):
        assert torch.count_nonzero(state[1:3]).item() == 0
        assert torch.all(state[0] == value) and torch.all(state[3:] == value)
    assert torch.count_nonzero(mixer._replayssm_ring_start).item() == 0
    assert torch.count_nonzero(mixer._replayssm_prev_num_accepted).item() == 0


@pytest.mark.upstream_source
@pytest.mark.parametrize("ppu", [False, True])
@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("preloaded", [False, True])
def test_real_warmup_keeps_applicable_work_and_guards_msa(
    modules, monkeypatch, upstream_source_root, ppu, local, jit, preloaded
):
    calls = []

    def unavailable(*args, **kwargs):
        raise AssertionError("Unconfigured external warmup was called")

    imports = {
        "attention_warmup": ["mixed_batch_attention_warmup"],
        "b12x_warmup": ["b12x_warmup"],
        "cutedsl_warmup": ["cutedsl_warmup"],
        "deep_gemm_warmup": ["deep_gemm_warmup"],
        "flashinfer_autotune_cache": [
            "resolve_flashinfer_autotune_v2_root",
            "resolve_rank_flashinfer_autotune_file",
            "use_flashinfer_autotune_v2",
        ],
        "flashinfer_sparse_mla_warmup": [
            "autotune_hisparse_flashinfer_attention",
            "deepseek_v4_sparse_mla_attention_warmup",
            "flashinfer_sparse_mla_decode_autotune_warmup",
        ],
        "kimi_k3_triton_warmup": ["kimi_k3_triton_warmup"],
        "mamba_triton_warmup": ["mamba_triton_warmup"],
        "qwen4_exp_qsa_warmup": ["qwen4_exp_qsa_triton_warmup"],
        "qwen_triton_warmup": ["qwen_triton_warmup"],
        "qwen_vl_triton_warmup": ["qwen_vl_triton_warmup"],
        "replayssm_warmup": ["replayssm_autotune_warmup", "replayssm_lifecycle_warmup"],
        "watermark_sample_warmup": ["watermark_sample_warmup"],
    }
    for name, names in imports.items():
        modules(
            "vllm.model_executor.warmup." + name, **dict.fromkeys(names, unavailable)
        )
    logger = NS(
        info=lambda *a: None,
        info_once=lambda *a: None,
        warning=lambda *a: None,
        exception=lambda *a: None,
    )
    modules("vllm.logger", init_logger=lambda _: logger)
    modules("torch", nn=NS(Module=object), Tensor=object)
    envs = modules(
        "vllm.envs",
        VLLM_DEEP_GEMM_WARMUP="full",
        VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=False,
    )
    modules("vllm.utils.deep_gemm", is_deep_gemm_supported=lambda: False)
    modules("vllm.utils.flashinfer", has_flashinfer=lambda: False)
    platform = NS(is_ppu=lambda: ppu, has_device_capability=lambda cap: cap <= 89)
    modules("vllm.platforms", current_platform=platform)
    name = "vllm.model_executor.warmup.kernel_warmup"
    modules(name)
    spec = importlib.util.spec_from_file_location(
        name, upstream_source_root / "vllm/model_executor/warmup/kernel_warmup.py"
    )
    provider = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, provider)
    spec.loader.exec_module(provider)
    original = provider.kernel_warmup

    class Kernel:
        def compile_many(self, keys):
            assert keys == [((), {})]
            calls.append("triton")

    registry = NS(
        _registrations={Kernel(): [((), {})]}, activate=lambda: contextlib.nullcontext()
    )
    worker = NS(
        model_runner=NS(
            jit_warmup_registry=registry, is_last_pp_rank=True, is_pooling_model=False
        ),
        vllm_config=NS(
            kernel_config=NS(enable_jit_warmup=jit, enable_cutedsl_warmup=True)
        ),
        get_model=lambda: "model",
        scheduler_config=NS(max_num_batched_tokens=64),
    )
    complete_warmup_contract(modules, provider, worker, platform)
    provider.envs = envs

    def registry_warmup():
        calls.append("registry")
        for kernel, keys in registry._registrations.items():
            kernel.compile_many(keys)

    registry.warmup = registry_warmup
    for callback, label in [
        ("qwen_triton_warmup", "qwen"),
        ("qwen_vl_triton_warmup", "vl"),
        ("mamba_triton_warmup", "mamba"),
        ("replayssm_lifecycle_warmup", "replayssm_lifecycle"),
        ("kimi_k3_triton_warmup", "kimi"),
        ("watermark_sample_warmup", "watermark"),
        ("qwen4_exp_qsa_triton_warmup", "qsa"),
        ("_warmup_gemm_rs_ar", "gemm_rs_ar"),
        ("flashinfer_sparse_mla_decode_autotune_warmup", "sparse_decode"),
        ("deepseek_v4_sparse_mla_attention_warmup", "dsv4"),
        ("b12x_warmup", "b12x"),
    ]:

        def spy(*args, _label=label):
            calls.append(_label)

        setattr(provider, callback, spy)
    modules(
        "vllm.model_executor.warmup.step5_dsa_warmup",
        step5_dsa_warmup=lambda worker: calls.append("step5"),
    )
    provider._warmup_skinny_gemm = lambda model: calls.append("skinny")
    provider.cutedsl_warmup = lambda: calls.append("legacy")
    modules(
        "vllm.model_executor.warmup.cutedsl_warmup",
        cutedsl_warmup=provider.cutedsl_warmup,
    )
    modules(
        "vllm.model_executor.warmup.jit_warmup_cutedsl_helper",
        VllmCuTeDSLJitKernel=type("CuTeKernel", (), {}),
    )
    modules(
        "vllm.v1.sample.ops.topk_topp_sampler",
        register_top_k_top_p_warmups=lambda: calls.append("sampler"),
    )
    modules(
        "vllm_sail.envs",
        VLLM_SAIL_MOE_BACKEND="",
        VLLM_SAIL_DENSE_BACKEND="",
        VLLM_SAIL_DENSE_BF16_DEEPGEMM=False,
    )
    modules("vllm_sail.utils.deep_gemm", is_deep_gemm_supported=lambda: True)
    modules(
        "vllm_sail.model_executor.warmup.deep_gemm_warmup",
        deep_gemm_warmup=lambda model, tokens: calls.append(
            ("ppu_deepgemm", model, tokens)
        ),
    )
    msa = "vllm.model_executor.warmup.minimax_m3_msa_warmup"

    class ForbiddenMSA(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == msa:
                calls.append("msa-import")
                raise ModuleNotFoundError("NVIDIA-only warmup dependency", name="quack")
            return None

    if ppu:
        monkeypatch.delitem(sys.modules, msa, raising=False)
        monkeypatch.setattr(sys, "meta_path", [ForbiddenMSA(), *sys.meta_path])
        if not local:
            with pytest.raises(ModuleNotFoundError, match="NVIDIA-only"):
                original(worker, process_local_only=local)
            assert calls[-1] == "msa-import"
            calls.clear()
    else:
        modules(msa, minimax_m3_msa_warmup=lambda worker: calls.append("msa"))
    consumer_name = "vllm.v1.worker.gpu_worker"
    if preloaded:
        consumer = modules(consumer_name, kernel_warmup=original)
    load_patch("vllm_sail/patch/enhancement/kernel_warmup.py")
    if not preloaded:
        consumer = modules(consumer_name, kernel_warmup=provider.kernel_warmup)
    assert consumer.kernel_warmup is provider.kernel_warmup
    assert consumer.kernel_warmup(worker, process_local_only=local) is None
    expected = ["sampler"] if ppu else []
    if jit:
        expected += ["registry", "triton"]
    expected += ["qwen", "vl", "mamba", "replayssm_lifecycle"]
    if jit:
        expected += ["kimi", "watermark", "qsa"]
    expected += ["gemm_rs_ar"]
    if not ppu:
        expected += ["skinny", "legacy"]
    if not local:
        expected += ["sparse_decode", "dsv4", "step5", "b12x"]
        expected += [("ppu_deepgemm", "model", 64)] if ppu else ["msa"]
    assert calls == expected
    assert worker.vllm_config.kernel_config.enable_jit_warmup is jit
    assert worker.vllm_config.kernel_config.enable_cutedsl_warmup is True
    if ppu and not local:
        no_ppu_gemm = expected[:-1]
        envs.VLLM_DEEP_GEMM_WARMUP = "skip"
        calls.clear()
        consumer.kernel_warmup(worker)
        assert calls == no_ppu_gemm
        envs.VLLM_DEEP_GEMM_WARMUP = "full"
        support = sys.modules["vllm_sail.utils.deep_gemm"]
        support.is_deep_gemm_supported = lambda: False
        calls.clear()
        consumer.kernel_warmup(worker)
        assert calls == no_ppu_gemm
        support.is_deep_gemm_supported = lambda: True
        selectors = sys.modules["vllm_sail.envs"]
        selectors.VLLM_SAIL_MOE_BACKEND = "triton"
        selectors.VLLM_SAIL_DENSE_BACKEND = "triton"
        calls.clear()
        consumer.kernel_warmup(worker)
        assert calls == no_ppu_gemm
        selectors.VLLM_SAIL_DENSE_BF16_DEEPGEMM = True
        calls.clear()
        consumer.kernel_warmup(worker)
        assert calls == expected
    elif not ppu and not local:
        error = RuntimeError("MSA warmup failed")

        def failed_msa(worker):
            raise error

        sys.modules[msa].minimax_m3_msa_warmup = failed_msa
        with pytest.raises(RuntimeError) as raised:
            consumer.kernel_warmup(worker)
        assert raised.value is error


@pytest.mark.upstream_source
@pytest.mark.parametrize("v2", [False, True])
@pytest.mark.parametrize(
    "backends", [[], ["OTHER"], ["FLASHINFER", "OTHER"], ["FLASHINFER"]]
)
@pytest.mark.parametrize("remote_eligible", [0, 1])
@pytest.mark.parametrize("pooling", [False, True])
def test_flashinfer_warmup_coordinates_all_pp_ranks(
    modules, upstream_source_root, v2, backends, remote_eligible, pooling
):
    """使用真实 helper 验证 V2 的 any 和旧 runner 的 all 集体决策。"""
    calls = []
    collective_inputs = []

    class Eligibility:
        def __init__(self, value):
            self.value = value

        def item(self):
            return self.value

    def tensor(values, *, dtype, device):
        assert dtype == "int32" and device == "cpu"
        return Eligibility(values[0])

    def reduce(eligibility):
        collective_inputs.append(eligibility.item())
        eligibility.value += remote_eligible

    modules(
        "vllm.distributed.parallel_state",
        get_pp_group=lambda: NS(world_size=2, cpu_all_reduce_=reduce),
    )
    warmup = function(
        upstream_source_root / "vllm/model_executor/warmup/kernel_warmup.py",
        "_warmup_flashinfer_attention",
        dict(
            torch=NS(tensor=tensor, int32="int32"),
            logger=NS(info_once=lambda *args: None),
            mixed_batch_attention_warmup=lambda worker: calls.append(worker),
        ),
    )
    groups = [NS(backend=NS(get_name=lambda name=name: name)) for name in backends]
    worker = NS(
        use_v2_model_runner=v2,
        model_runner=NS(
            is_pooling_model=pooling, attn_groups=[groups] if groups else []
        ),
    )
    warmup(worker)
    eligible = (
        not pooling
        and bool(backends)
        and (any if v2 else all)(name == "FLASHINFER" for name in backends)
    )
    assert collective_inputs == [int(eligible)]
    count = int(eligible) + remote_eligible
    assert calls == ([worker] if (count > 0 if v2 else count == 2) else [])


@pytest.mark.upstream_source
@pytest.mark.parametrize("max_m", [0, 1, 3, 8, 17, 255, 256, 513, 1025, 4097])
@pytest.mark.parametrize("enabled,fail", [(False, False), (True, False), (True, True)])
def test_real_aiter_warmup_keeps_buckets_and_propagates_failures(
    upstream_source_root,
    max_m,
    enabled,
    fail,
):
    """运行真实桶生成与 warmup；仅替换 dummy runner 和计时/日志边界。"""
    path = upstream_source_root / "vllm/model_executor/warmup/kernel_warmup.py"
    calls, logs = [], []
    error = RuntimeError("AITER dummy run failed")

    def dummy(num_tokens, *, skip_eplb):
        assert skip_eplb is True
        calls.append(num_tokens)
        if fail:
            raise error

    namespace = {
        "time": NS(perf_counter=lambda: 1.0),
        "logger": NS(info=lambda *args: logs.append(args)),
    }
    buckets = function(path, "_aiter_gemm_m_buckets", namespace)
    namespace["_aiter_gemm_m_buckets"] = buckets
    warmup = function(path, "_rocm_aiter_gemm_warmup", namespace)
    worker = NS(
        vllm_config=NS(kernel_config=NS(enable_rocm_aiter_gemm_warmup=enabled)),
        scheduler_config=NS(max_num_batched_tokens=max_m),
        model_runner=NS(_dummy_run=dummy),
    )
    expected_buckets = set()
    for m in range(1, max_m + 1):
        if m <= 8:
            bucket = 1 << (m - 1).bit_length()
        else:
            step = 16 if m <= 256 else 32 if m <= 1024 else 64 if m <= 4096 else 128
            bucket = ((m + step - 1) // step) * step
        if bucket <= max_m:
            expected_buckets.add(bucket)
    if max_m > 0:
        expected_buckets.add(max_m)
    assert buckets(max_m) == sorted(expected_buckets)
    expected = (
        sorted(
            expected_buckets
            | set(range(8, min(512, max_m) + 1, 8))
            | ({max_m - 1} if max_m > 1 else set())
        )
        if enabled
        else []
    )
    if fail and expected:
        with pytest.raises(RuntimeError) as raised:
            warmup(worker)
        assert raised.value is error
        assert calls == expected[:1] and logs == []
    else:
        assert warmup(worker) is None
        assert calls == expected
        assert len(logs) == int(enabled)
        if enabled:
            assert logs[0][1:] == (len(expected), 0.0)


@pytest.mark.upstream_source
@pytest.mark.parametrize("v2", [False, True])
@pytest.mark.parametrize(
    "ineligible",
    ["padding-only", "non-mamba", "no-mixer", "replay-disabled", "wrong-mixer"],
)
def test_real_replayssm_lifecycle_skips_without_committable_eligible_state(
    modules,
    upstream_source_root,
    v2,
    ineligible,
):
    """不合格路径不得触碰 scratch、模型状态或 native context。"""
    calls = []

    def forbidden(*args, **kwargs):
        pytest.fail("Ineligible ReplaySSM warmup touched state")

    base, spec_cls = type("MambaBase", (), {}), type("MambaSpec", (), {})
    mixer = base() if ineligible != "wrong-mixer" else NS()
    mixer.use_flashinfer_replayssm = ineligible != "replay-disabled"
    spec = spec_cls() if ineligible != "non-mamba" else object()
    modules("vllm.model_executor.layers.mamba.abstract", MambaBase=base)
    modules("vllm.v1.kv_cache_interface", MambaSpec=spec_cls)
    modules(
        "vllm.v1.worker.replayssm_utils", ReplaySSMModelContext=NS(create=forbidden)
    )

    def committed(actual):
        calls.append(actual)
        return 1 if ineligible == "padding-only" else 4

    modules(
        "vllm.v1.worker.extensible_kv_cache",
        num_committable_kv_blocks=committed,
        ensure_kv_cache_blocks=forbidden,
    )
    runner = NS(
        vllm_config=NS(
            use_flashinfer_replayssm=True,
            use_v2_model_runner=v2,
            compilation_config=NS(static_forward_context={"mixer": mixer}),
        ),
        kv_cache_config=NS(
            kv_cache_groups=[
                NS(
                    kv_cache_spec=spec,
                    layer_names=[] if ineligible == "no-mixer" else ["mixer"],
                )
            ]
        ),
    )
    warmup = function(
        upstream_source_root / "vllm/model_executor/warmup/replayssm_warmup.py",
        "replayssm_lifecycle_warmup",
        {
            "torch": NS(empty_strided=forbidden),
            "_temporary_replayssm_autotune_state": forbidden,
        },
    )
    assert warmup(runner) is None
    assert calls == [runner]


@pytest.mark.upstream_source
@pytest.mark.requires_runtime
@pytest.mark.parametrize("fail", [False, True])
def test_real_replayssm_temporary_state_cleans_overlaid_views_not_padding(
    modules,
    upstream_source_root,
    fail,
):
    torch = pytest.importorskip("torch")
    mixer_cls = type("MambaMixer2", (), {})
    modules("vllm.model_executor.layers.mamba.mamba_mixer2", MambaMixer2=mixer_cls)
    storage = torch.full((5, 4), 7.0)
    narrow, wide = storage[:, :2], storage[:, :3]
    assert narrow.data_ptr() == wide.data_ptr()
    tracker = torch.ones(5, dtype=torch.int32)
    mixers = []
    for view in (narrow, wide):
        mixer = mixer_cls()
        mixer.use_replayssm, mixer.replayssm_buffer_len = True, 4
        mixer.kv_cache = [view]
        mixer._replayssm_ring_start = mixer._replayssm_prev_num_accepted = tracker
        mixers.append(mixer)
    runner = NS(
        vllm_config=NS(use_v2_model_runner=True),
        get_model=lambda: NS(modules=lambda: mixers),
    )
    temporary = contextlib.contextmanager(
        function(
            upstream_source_root / "vllm/model_executor/warmup/replayssm_warmup.py",
            "_temporary_replayssm_autotune_state",
            {"torch": torch},
        )
    )
    error = RuntimeError("overlaid state failed")

    def exercise():
        with temporary(runner, 2, prepare_dummy_block_tables=False):
            storage[1:3, :3].fill_(42)
            tracker.fill_(42)
            if fail:
                raise error

    if fail:
        with pytest.raises(RuntimeError) as raised:
            exercise()
        assert raised.value is error
    else:
        exercise()
    assert torch.count_nonzero(storage[1:3, :3]).item() == 0
    assert torch.all(storage[0] == 7) and torch.all(storage[3:] == 7)
    assert torch.all(storage[:, 3] == 7)
    assert torch.count_nonzero(tracker).item() == 0


def _glm_kda_warmup_boundary(modules, monkeypatch, root, torch, capability):
    """真实 registry/dispatcher/l2norm；Triton 编译、GDN 分配和线性层是边界。"""
    import inspect
    import textwrap

    platform = NS(
        is_cuda=lambda: True,
        get_device_capability=lambda: NS(major=capability // 10),
        has_device_capability=lambda required: capability >= required,
    )
    modules("vllm.platforms", current_platform=platform)
    modules("vllm.envs")
    triton = NS(
        cdiv=lambda n, d: (n + d - 1) // d,
        next_power_of_2=lambda n: 1 << (n - 1).bit_length(),
    )
    modules("vllm.triton_utils", HAS_TRITON=False, triton=triton)

    def source_module(name, path):
        modules(name)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    base_path = root / "vllm/model_executor/warmup"
    jit = source_module(
        "vllm.model_executor.warmup.jit_warmup", base_path / "jit_warmup.py"
    )
    helper = source_module(
        "vllm.model_executor.warmup.jit_warmup_triton_helper",
        base_path / "jit_warmup_triton_helper.py",
    )
    norm_path = root / "vllm/third_party/flash_linear_attention/ops/l2norm.py"
    namespace = dict(
        triton=triton,
        WarmupChoices=jit.WarmupChoices,
        TritonWarmupTensor=helper.TritonWarmupTensor,
    )
    function(norm_path, "_l2norm_block_dim", namespace)
    warmup_inputs = function(norm_path, "_l2norm_fwd_block_warmup_inputs", namespace)
    launches = []

    class Kernel:
        arg_names = ("X", "Y", "eps", "M", "N", "BD", "MBLOCK")
        fn = staticmethod(function(norm_path, "l2norm_fwd_kernel2", {}))

        def __getitem__(self, grid):
            def launch(**kwargs):
                launches.append((grid, kwargs))

            return launch

    dispatcher = helper.triton_kernel_dispatcher_with_warmup(
        kernel=Kernel(), warmup_inputs=warmup_inputs
    )(function(norm_path, "l2norm_fwd_block", {"triton": triton, "_L2NORM_MBLOCK": 32}))
    filters = dict(
        ast=ast,
        inspect=inspect,
        textwrap=textwrap,
        _UNSUPPORTED_FP8_DTYPE="float8e4nv",
        logger=NS(warning=lambda *a: None),
    )
    for name in (
        "_called_name",
        "_names_unsupported_fp8",
        "_kernel_needs_unsupported_fp8",
        "_exclude_sm80_fp8_registrations",
    ):
        function("vllm_sail/patch/enhancement/kernel_warmup.py", name, filters)
    return NS(
        jit=jit, helper=helper, norm=dispatcher, launches=launches, filters=filters
    )


@pytest.mark.upstream_source
@pytest.mark.requires_runtime
@pytest.mark.parametrize("capability", [80, 89])
@pytest.mark.parametrize("backend", ["auto", "triton"])
@pytest.mark.parametrize("failure", [None, "base", "duplicate", "registration"])
def test_glm_real_kda_constructor_registers_l2norm_and_restores_quant_config(
    modules,
    monkeypatch,
    upstream_source_root,
    capability,
    backend,
    failure,
):
    """完整真实 constructor/class 与真实注册；不验证 KDA 数值或真实编译 key。"""
    torch = pytest.importorskip("torch")
    state = _glm_kda_warmup_boundary(
        modules, monkeypatch, upstream_source_root, torch, capability
    )
    error = RuntimeError("KDA allocation/registration failed")
    projections = []
    static = {"layer": object()} if failure == "duplicate" else {}
    quant = object()
    config = NS(
        quant_config=quant,
        model_config=NS(dtype=torch.bfloat16),
        cache_config=NS(mamba_cache_dtype="auto", mamba_ssm_cache_dtype="float32"),
        scheduler_config=NS(max_num_seqs=8),
        additional_config={"kda_prefill_backend": backend},
    )
    registry = state.jit.JitWarmupRegistry(config)
    scatter = object()

    class Base(torch.nn.Module):
        def __init__(self, text, actual_config, prefix):
            super().__init__()
            assert actual_config is config and actual_config.quant_config is None
            if failure == "base":
                raise error
            self.quant_config = actual_config.quant_config
            self.model_config, self.cache_config = (
                actual_config.model_config,
                actual_config.cache_config,
            )
            self.tp_size, self.tp_rank, self.hidden_size, self.num_spec = 2, 1, 8, 2

    class Linear(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            projections.append(kwargs)

    def register_scatter(**kwargs):
        state.jit.JitWarmupRegistry.register(scatter, **kwargs)

    def set_attrs(param, attrs):
        for name, value in attrs.items():
            setattr(param, name, value)

    def fused_conv(*args, **kwargs):
        return NS(weight=torch.empty(3, 1, 4), bias=None)

    path = upstream_source_root / "vllm/models/glm5next/common/kda.py"
    namespace = dict(
        nn=torch.nn,
        torch=torch,
        GatedDeltaNetAttention=Base,
        divide=lambda a, b: a // b,
        _Glm5NextMergedColumnParallelLinear=Linear,
        ColumnParallelLinear=Linear,
        RowParallelLinear=Linear,
        FusedRMSNormGated=lambda *a, **kw: Linear(),
        _fused_qkv_conv1d=fused_conv,
        sharded_weight_loader=lambda axis: lambda *a: None,
        set_weight_attrs=set_attrs,
        get_current_vllm_config=lambda: NS(
            compilation_config=NS(static_forward_context=static)
        ),
        is_conv_state_dim_first=lambda: True,
        scatter_states=NS(register_warmup=register_scatter),
        l2norm_fwd_block=state.norm,
        eager_break_during_capture=lambda fn: fn,
        MambaStateShapeCalculator=NS(
            kda_state_shape=lambda *a, **kw: ((3, 4), (2, 128, 128))
        ),
        MambaStateDtypeCalculator=NS(
            kda_state_dtype=lambda *a: (torch.float32, torch.float32)
        ),
        current_platform=sys.modules["vllm.platforms"].current_platform,
    )
    function(path, "_resolve_kda_prefill_backend", namespace)
    layer_type = function(path, "Glm5NextLinearAttention", namespace)
    text = NS(
        linear_head_dim=128,
        linear_num_heads=4,
        linear_conv_kernel_dim=4,
        linear_lower_bound=-5.0,
    )
    original_add = registry._add

    def add(kernel, args, kwargs):
        if failure == "registration" and kernel is state.norm:
            raise error
        original_add(kernel, args, kwargs)

    monkeypatch.setattr(registry, "_add", add)
    with registry.activate():
        if failure:
            with pytest.raises((RuntimeError, ValueError)) as raised:
                layer_type(text, config, "layer")
            if failure == "duplicate":
                assert str(raised.value) == "Duplicate layer name: layer"
            else:
                assert raised.value is error
        else:
            first = layer_type(text, config, "layer")
            second = layer_type(text, config, "layer2")
            assert first.quant_config is second.quant_config is None
            assert first.kda_prefill_backend == second.kda_prefill_backend == "triton"
            assert first._flashkda_buffer_specs is first._checkpoint_exporter is None
            assert static == {"layer": first, "layer2": second}
    assert config.quant_config is quant
    assert all(kwargs.get("quant_config") is None for kwargs in projections)
    if failure:
        assert state.norm not in registry._registrations and state.launches == []
        assert len(registry) == (0 if failure == "base" else 1)
    else:
        assert registry._registrations[state.norm] == [
            ((), {"dtype": torch.bfloat16, "dim": 128})
        ]
        assert len(registry) == 2
        assert registry._registrations[scatter] == [
            (
                (),
                dict(
                    state_shape=(2, 128, 128),
                    state_dtype=torch.float32,
                    indices_dtype=torch.int32,
                    max_num_tokens=8,
                ),
            )
        ]
        worker = NS(model_runner=NS(jit_warmup_registry=registry))
        state.filters["_exclude_sm80_fp8_registrations"](worker)
        assert state.norm in registry._registrations
        cases = list(
            state.norm._provider_cases(
                state.norm._warmup_inputs_fn, dtype=torch.bfloat16, dim=128
            )
        )
        assert {(case["x"].shape, case["x"].aligned) for case in cases} == {
            ((rows, 128), aligned) for rows in (1, 16, 2) for aligned in (True, False)
        }
        assert len(cases) == 6
        for case in cases:
            assert case["x"].dtype == case["y"].dtype == torch.bfloat16
            assert case["eps"] == 1e-6 and case["block_dim"] == 128
            state.norm(**case)
        assert len(state.launches) == 6
        for grid, kwargs in state.launches:
            assert grid == (1,) and kwargs["N"] == kwargs["BD"] == 128
            assert kwargs["MBLOCK"] == 32 and kwargs["M"] in (1, 16, 2)
        # 自动/显式 Triton 路径不会碰 NVIDIA workspace；显式 FlashKDA 必须拒绝。
        with pytest.raises(RuntimeError, match="FlashKDA requires CUDA"):
            namespace["_resolve_kda_prefill_backend"](
                "flashkda", 128, torch.bfloat16, -5.0
            )
