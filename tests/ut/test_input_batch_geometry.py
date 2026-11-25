# SPDX-License-Identifier: Apache-2.0
"""Exercise real token ownership and request transport across the scheduler boundary."""

from types import SimpleNamespace as NS

import pytest

from tests.support.source import definition, function


def _source_classes(modules, name, path, names, **globals_):
    """保留真实 class body/decorators；模块导入和未触达的依赖不由此验证。"""
    import ast

    provider = modules(name, **globals_)
    tree = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *(definition(path, class_name) for class_name in names),
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), provider.__dict__)
    return provider


def _token_buffer_provider(modules, root):
    collection = modules("vllm.utils.collection_utils")
    function(root / "vllm/utils/collection_utils.py", "as_list", collection.__dict__)
    provider = modules("vllm.v1.token_buffer")
    path = root / "vllm/v1/token_buffer.py"
    exec(compile(path.read_text(), str(path), "exec"), provider.__dict__)
    return provider


def _token_request_boundary(modules, root):
    """真实存储/请求/输出/worker body；结构化输出、事件和 GC C API 是边界。"""
    import enum
    import time
    from collections.abc import Sequence
    from dataclasses import dataclass, replace
    from functools import cached_property
    from typing import Generic, TypeVar, overload

    np = pytest.importorskip("numpy")
    token = _token_buffer_provider(modules, root)
    views = _source_classes(
        modules,
        "vllm.v1.utils",
        root / "vllm/v1/utils.py",
        ["ConstantList"],
        Generic=Generic,
        Sequence=Sequence,
        T=TypeVar("T"),
        overload=overload,
    )
    length = function(
        root / "vllm/utils/__init__.py", "length_from_prompt_token_ids_or_embeds", {}
    )
    gc_calls = []

    def untrack(value):
        gc_calls.append(value)
        return value

    request = _source_classes(
        modules,
        "vllm.v1.request",
        root / "vllm/v1/request.py",
        ["RequestStatus", "StreamingUpdate", "Request"],
        enum=enum,
        dataclass=dataclass,
        time=time,
        TokenBuffer=token.TokenBuffer,
        ConstantList=views.ConstantList,
        length_from_prompt_token_ids_or_embeds=length,
        gc_untrack=untrack,
        PrefillStats=NS,
        StructuredOutputRequest=NS(from_sampling_params=lambda params: None),
        EngineCoreEvent=NS(
            new_event=lambda event_type, timestamp: NS(
                type=event_type, timestamp=timestamp
            )
        ),
    )
    output = modules("vllm.v1.core.sched.output", gc_untrack=untrack)
    function(root / "vllm/utils/gc_utils.py", "untrack_long_lists", output.__dict__)
    function(
        root / "vllm/multimodal/utils.py", "strip_covered_mm_data", output.__dict__
    )
    output = _source_classes(
        modules,
        output.__name__,
        root / "vllm/v1/core/sched/output.py",
        ["NewRequestData", "CachedRequestData"],
        dataclass=dataclass,
        cached_property=cached_property,
        replace=replace,
    )

    def sampling(**overrides):
        return NS(
            **{
                "max_tokens": 8,
                "extra_args": None,
                "skip_reading_prefix_cache": False,
                "prompt_logprobs": None,
                "prompt_logprob_token_ids": None,
                "prompt_logprob_start": None,
                **overrides,
            }
        )

    def make(tokens, **kwargs):
        params = kwargs.pop("sampling_params", sampling())
        return request.Request(
            "tokens", tokens, params, None, arrival_time=1.0, **kwargs
        )

    return NS(
        np=np,
        token=token,
        request=request,
        output=output,
        gc_calls=gc_calls,
        sampling=sampling,
        make=make,
    )


@pytest.mark.upstream_source
@pytest.mark.requires_runtime
@pytest.mark.parametrize("protocol", [4, 5])
def test_real_token_buffer_copies_remain_resizable_at_tensor_and_pickle_boundaries(
    modules,
    upstream_source_root,
    protocol,
):
    import pickle

    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")

    provider = _token_buffer_provider(modules, upstream_source_root)
    tokens = provider.TokenBuffer(np.array([1, 2, 3], dtype=np.int32))
    tensorize = function(
        upstream_source_root / "vllm/utils/torch_utils.py",
        "token_ids_to_tensor",
        {"np": np, "torch": torch},
    )
    tensor = tensorize(tokens)
    buffers = []
    kwargs = {"buffer_callback": buffers.append} if protocol == 5 else {}
    encoded = pickle.dumps(tokens, protocol=protocol, **kwargs)
    tokens.extend([4, 5])
    tokens[0] = 9
    assert tensor.tolist() == [1, 2, 3] and tensor.dtype == torch.int64
    restored = pickle.loads(encoded, buffers=buffers)
    assert restored == [1, 2, 3] and restored is not tokens
    restored.extend([6])
    assert tokens == [9, 2, 3, 4, 5]
    view = provider.TokenIdsView(tokens, [6, 7])
    view.pop()
    assert provider.rfind_tokens(view, {None, 6}) == 5
    assert list(view) == [9, 2, 3, 4, 5, 6]
    assert tokens == [9, 2, 3, 4, 5]
    extended = provider.TokenBuffer()
    provider.extend_from_slice(extended, tokens, 1, 4)
    assert extended == [2, 3, 4]
    assert list(provider.token_slice(tokens, 1, 3)) == [2, 3]


@pytest.mark.upstream_source
@pytest.mark.requires_runtime
@pytest.mark.parametrize("mode", ["list", "ndarray", "mixed", "embeds"])
def test_real_request_owns_int32_buffers_and_preserves_readonly_views(
    modules, upstream_source_root, mode
):
    state = _token_request_boundary(modules, upstream_source_root)
    source = [10, 20, 30]
    kwargs = {}
    if mode == "ndarray":
        source = state.np.array([10, 99, 20, 99, 30, 99], dtype=state.np.int32)[::2]
        assert not source.flags.c_contiguous
    elif mode == "mixed":
        source = [10, -999, 30]
        kwargs = {
            "prompt_embeds": state.np.zeros((3, 2)),
            "prompt_is_token_ids": [True, False, True],
        }
    elif mode == "embeds":
        source = None
        kwargs = {"prompt_embeds": state.np.zeros((3, 2))}
    request = state.make(source, **kwargs)
    expected = (
        [0, 0, 0]
        if mode == "embeds"
        else [10, 0, 30]
        if mode == "mixed"
        else [10, 20, 30]
    )
    prompt, all_tokens, outputs = (
        request.prompt_token_ids,
        request._all_token_ids,
        request._output_token_ids,
    )
    assert (
        isinstance(all_tokens, state.token.TokenBuffer) and all_tokens.typecode == "i"
    )
    assert all_tokens == expected and request.num_prompt_tokens == 3
    if source is not None:
        assert isinstance(prompt, state.token.TokenBuffer)
        assert prompt is not all_tokens and prompt is not source
        source[0] = 99
        assert prompt[0] == 10 and all_tokens[0] == 10
    assert request.prefill_token_ids is (
        all_tokens if mode in ("mixed", "embeds") else prompt
    )
    readonly = request.all_token_ids
    assert readonly._x is all_tokens and request.output_token_ids._x is outputs
    with pytest.raises(TypeError, match="Cannot append"):
        readonly.append(99)
    with pytest.raises(TypeError, match="Cannot set item"):
        readonly[0] = 99
    snapshot = readonly.copy()
    assert type(snapshot) is list
    request.append_output_token_ids(40)
    request.append_output_token_ids([50, 60])
    assert request._all_token_ids is all_tokens and request._output_token_ids is outputs
    assert list(readonly) == expected + [40, 50, 60]
    assert list(request.output_token_ids) == [40, 50, 60] and snapshot == expected
    assert request.prefill_token_ids is all_tokens and request.num_tokens == 6
    assert request.num_output_tokens == 3
    if prompt is not None:
        assert len(prompt) == 3
    with pytest.raises(ValueError, match="Neither prompt_token_ids"):
        state.make(None)
    with pytest.raises(ValueError, match="different lengths"):
        state.make([1], prompt_embeds=state.np.zeros((2, 2)))


@pytest.mark.upstream_source
@pytest.mark.requires_runtime
@pytest.mark.parametrize("protocol", [4, 5])
@pytest.mark.parametrize("mode", ["prompt", "resumed", "mixed", "embeds"])
def test_real_new_request_pickle_keeps_prefill_alias_and_worker_token_boundaries(
    modules,
    upstream_source_root,
    protocol,
    mode,
):
    import pickle

    state = _token_request_boundary(modules, upstream_source_root)
    kwargs = {}
    tokens = [10, 20, 30]
    if mode == "mixed":
        tokens = [10, -999, 30]
        kwargs = {
            "prompt_embeds": state.np.zeros((3, 2)),
            "prompt_is_token_ids": [True, False, True],
        }
    elif mode == "embeds":
        tokens = None
        kwargs = {"prompt_embeds": state.np.zeros((3, 2))}
    request = state.make(tokens, **kwargs)
    if mode == "resumed":
        request.append_output_token_ids([40, 50])
    data = state.output.NewRequestData.from_request(
        request,
        (list(range(300)),),
        prefill_token_ids=request.prefill_token_ids,
    )
    assert data.prompt_token_ids is request.prompt_token_ids
    assert data.prefill_token_ids is request.prefill_token_ids and data.prompt_len == 3
    prompt_before = (
        None if data.prompt_token_ids is None else list(data.prompt_token_ids)
    )
    prefill_before = list(data.prefill_token_ids)
    alias = mode == "prompt"
    assert (data.prompt_token_ids is data.prefill_token_ids) is alias
    buffers = []
    encoded = pickle.dumps(
        data,
        protocol=protocol,
        **({"buffer_callback": buffers.append} if protocol == 5 else {}),
    )
    if protocol == 5:
        # ndarray embedding 也可能出带外 buffer；这里只核查 token payload 及 prompt/prefill 去重。
        token_payloads = [data.prefill_token_ids.tobytes()]
        if data.prompt_token_ids is not None and not alias:
            token_payloads.append(data.prompt_token_ids.tobytes())
        for payload in token_payloads:
            assert sum(buffer.raw().tobytes() == payload for buffer in buffers) == 1
    request.append_output_token_ids(70)
    if request.prompt_token_ids is not None:
        request.prompt_token_ids.extend([80])
    received = pickle.loads(encoded, buffers=buffers)
    assert (
        received.prompt_token_ids == prompt_before
        and received.prefill_token_ids == prefill_before
    )
    assert (received.prompt_token_ids is received.prefill_token_ids) is alias
    assert received.prefill_token_ids is not data.prefill_token_ids
    assert received.prompt_len == 3
    assert any(value is received.block_ids[0] for value in state.gc_calls)
    assert not any(
        value is received.prefill_token_ids or value is received.prompt_token_ids
        for value in state.gc_calls
    )
    # MRv2 passes the deserialized prefill buffer directly to request state.
    recorded = []

    def noop(*a, **kw):
        return None

    owner = NS(
        req_states=NS(
            add_request=lambda **kw: recorded.append(kw),
            req_id_to_index={received.req_id: 0},
            apply_staged_writes=noop,
        ),
        _remove_request=noop,
        adaptive_verification=None,
        speculator=None,
        pooling_runner=None,
        encoder_cache=None,
        model_state=NS(
            add_request=noop, get_mrope_position_delta=noop, apply_staged_writes=noop
        ),
        block_tables=NS(append_block_ids=noop),
        lora_state=NS(add_request=noop),
        is_last_pp_rank=False,
        sampler=None,
    )
    add = function(
        upstream_source_root / "vllm/v1/worker/gpu/model_runner.py",
        "GPUModelRunner.add_requests",
        {},
    )
    add(owner, NS(scheduled_new_reqs=[received]))
    assert recorded[0]["prompt_len"] == 3
    assert recorded[0]["all_token_ids"] is received.prefill_token_ids
    assert recorded[0]["num_computed_tokens"] == received.num_computed_tokens


def _request_block_hasher(state, root, lookahead=0):
    """真实 hash/helper 链；摘要算法和初始 seed 固定为测试边界。"""
    import bisect
    import hashlib
    from typing import NewType

    torch = pytest.importorskip("torch")
    namespace = {
        "bisect": bisect,
        "hashlib": hashlib,
        "BlockHash": NewType("BlockHash", bytes),
        "NONE_HASH": b"token-test-seed",
        "token_slice": state.token.token_slice,
        "torch": torch,
        "TORCH_DTYPE_TO_NUMPY_DTYPE": {torch.float32: state.np.float32},
    }
    function(root / "vllm/v1/utils.py", "tensor_data", namespace)
    function(root / "vllm/multimodal/utils.py", "get_mm_features_in_window", namespace)
    path = root / "vllm/v1/core/kv_cache_utils.py"
    for name in (
        "_gen_mm_extra_hash_keys",
        "_gen_lora_extra_hash_keys",
        "_gen_prompt_embeds_extra_hash_keys",
        "generate_block_hash_extra_keys",
        "generate_lookahead_block_hash_extra_keys",
        "hash_block_tokens",
        "get_request_block_hasher",
    ):
        function(path, name, namespace)

    def digest(value):
        return hashlib.sha256(repr(value).encode()).digest()

    return namespace["get_request_block_hasher"](2, digest, lookahead), digest


@pytest.mark.upstream_source
@pytest.mark.requires_runtime
@pytest.mark.parametrize("lookahead", [0, 1, 2])
@pytest.mark.parametrize("salt", [None, "private-cache"])
def test_real_request_hash_consumer_matches_list_reference_without_holding_buffer_exports(
    modules,
    upstream_source_root,
    lookahead,
    salt,
):
    state = _token_request_boundary(modules, upstream_source_root)
    hasher, digest = _request_block_hasher(state, upstream_source_root, lookahead)
    request = state.make(
        state.np.array([1, 2, 3, 4, 5, 6], dtype=state.np.int32),
        cache_salt=salt,
        block_hasher=hasher,
    )
    request.append_output_token_ids([7, 8])
    expected = []
    tokens = list(request.all_token_ids)
    for start in range(0, len(tokens) - lookahead - 1, 2):
        extra = (("cache_salt", salt),) if start == 0 and salt else None
        if lookahead:
            extra = (extra, *tokens[start + 2 : start + 2 + lookahead])
        expected.append(
            digest(
                (
                    expected[-1] if expected else b"token-test-seed",
                    tuple(tokens[start : start + 2]),
                    extra,
                )
            )
        )
    assert request.block_hashes == expected
    saved_slice = state.token.token_slice(request._all_token_ids, 0, 2)
    request._all_token_ids.extend([9, 10])
    request._all_token_ids[0] = 99
    assert list(saved_slice) == [1, 2]


@pytest.mark.upstream_source
@pytest.mark.requires_runtime
@pytest.mark.parametrize("waiting", [False, True])
@pytest.mark.parametrize("update_tokens", [[70, 80], [], None])
def test_real_streaming_session_mutates_owned_buffers_and_rehashes_kept_output(
    modules,
    upstream_source_root,
    waiting,
    update_tokens,
):
    from dataclasses import dataclass, replace

    state = _token_request_boundary(modules, upstream_source_root)
    hasher, _ = _request_block_hasher(state, upstream_source_root)
    session = state.make([10, 20, 30], resumable=True, block_hasher=hasher)
    session.append_output_token_ids([40, 50, 60])
    before_hashes = session.block_hashes.copy()
    session.num_computed_tokens = 4
    status = state.request.RequestStatus
    session.status = status.WAITING_FOR_STREAMING_REQ if waiting else status.RUNNING
    prompt, all_tokens, outputs = (
        session.prompt_token_ids,
        session._all_token_ids,
        session._output_token_ids,
    )
    readonly_all, readonly_output = session.all_token_ids, session.output_token_ids
    next_request = state.make(
        update_tokens if update_tokens is not None else [],
        resumable=True,
        sampling_params=state.sampling(skip_reading_prefix_cache=True),
    )
    next_request.arrival_time = 2.0
    update = state.request.StreamingUpdate.from_request(next_request)
    assert update.prompt_token_ids is next_request.prompt_token_ids
    if update_tokens is None:
        update.prompt_token_ids = None
    next_request.resumable = False
    assert state.request.StreamingUpdate.from_request(next_request) is None

    @dataclass
    class Position:
        offset: int
        length: int

    feature = NS(mm_position=Position(0, 1), identifier="session-image")
    update.mm_features = [feature] if update_tokens else []
    queued = object()
    scheduler = NS(
        hash_block_size=2,
        num_waiting_for_streaming_input=3,
        log_stats=True,
        num_sampled_tokens_per_step=1,
        max_model_len=16,
    )
    apply = function(
        upstream_source_root / "vllm/v1/core/sched/scheduler.py",
        "Scheduler._update_request_as_session",
        {
            "extend_from_slice": state.token.extend_from_slice,
            "replace": replace,
            "RequestStatus": status,
            "EngineCoreEventType": NS(QUEUED=queued),
        },
    )
    assert apply(scheduler, session, update) is True
    expected = [10, 20, 30, 40, 50] + (update_tokens or [])
    assert session.prompt_token_ids is prompt and session._all_token_ids is all_tokens
    assert session._output_token_ids is outputs
    assert list(readonly_all) == expected and list(readonly_output) == []
    assert prompt == expected and session.num_prompt_tokens == len(expected)
    assert session.prefill_token_ids is prompt and session.num_output_tokens == 0
    assert (
        session.block_hashes[:2] == before_hashes[:2]
        and len(session.block_hashes) == len(expected) // 2
    )
    assert (
        session.block_hashes
        == state.make(
            expected,
            block_hasher=hasher,
            mm_features=session.mm_features,
        ).block_hashes
    )
    assert session.status is status.WAITING and session.events[-1].type is queued
    assert scheduler.num_waiting_for_streaming_input == (2 if waiting else 3)
    assert (
        session.arrival_time == 2.0
        and session.sampling_params is update.sampling_params
    )
    assert (
        session.prefix_cache_hit_limit == 0 and session.max_tokens == update.max_tokens
    )
    if update_tokens:
        assert session.mm_features == [feature] and feature.mm_position.offset == 5
        assert next_request.prompt_token_ids == update_tokens
    session.append_output_token_ids(90)
    assert all_tokens[-1] == 90 and list(readonly_output) == [90]


@pytest.mark.upstream_source
@pytest.mark.requires_runtime
@pytest.mark.parametrize("use_pp", [False, True])
@pytest.mark.parametrize("async_scheduling", [False, True])
def test_real_cached_scheduler_payloads_copy_int32_tokens_before_worker_transport(
    modules,
    upstream_source_root,
    use_pp,
    async_scheduling,
):
    import pickle

    state = _token_request_boundary(modules, upstream_source_root)
    running, resumed = state.make([10, 20, 30]), state.make([60, 70])
    running.request_id, resumed.request_id = "running", "resumed"
    running.append_output_token_ids([40, 50])
    resumed.append_output_token_ids(80)
    running.num_computed_tokens, resumed.num_computed_tokens = 3, 2
    running.num_output_placeholders = 2
    block_calls = []

    def blocks(allow_none):
        block_calls.append(allow_none)
        return None

    scheduler = NS(
        use_pp=use_pp,
        scheduler_config=NS(async_scheduling=async_scheduling),
    )
    make = function(
        upstream_source_root / "vllm/v1/core/sched/scheduler.py",
        "Scheduler._make_cached_request_data",
        {"CachedRequestData": state.output.CachedRequestData},
    )
    payload = make(
        scheduler,
        [running, resumed],
        {"running": 3, "resumed": 2},
        {"running": [1], "resumed": [2]},
        {"running": NS(get_block_ids=blocks), "resumed": NS(get_block_ids=blocks)},
    )
    expected_new = [[40, 50], [80]] if use_pp and not async_scheduling else []
    assert payload.req_ids == ["running", "resumed"]
    assert payload.new_token_ids == expected_new
    assert all(type(value) is list for value in payload.new_token_ids)
    assert payload.num_computed_tokens == [3, 2] and payload.num_output_tokens == [4, 1]
    assert payload.new_block_ids == [None, None] and block_calls == [True, True]
    assert payload.num_reqs == 2 and not payload.is_context_phase("running")
    transported = pickle.loads(pickle.dumps(payload, protocol=5))
    running.append_output_token_ids(90)
    resumed.append_output_token_ids(100)
    assert payload.new_token_ids == transported.new_token_ids == expected_new
