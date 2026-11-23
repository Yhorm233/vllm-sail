# SPDX-License-Identifier: Apache-2.0
"""Device-length flattening and paged-logits oracles for FP8 and W8A8 INT8."""

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.ppu


@pytest.mark.parametrize("tokens", [16, 4096])
@pytest.mark.parametrize("heads", [32, 64])
def test_mxfp4_builder_copies_grouped_schedule(monkeypatch, tokens, heads):
    """The builder must accommodate the real grouped DeepGEMM metadata ABI."""
    import torch

    import vllm_sail

    vllm_sail.register_out_of_tree()
    from vllm.platforms import current_platform

    if not current_platform.is_ppu() or torch.cuda.get_device_capability() != (8, 9):
        pytest.skip("Grouped MXFP4 metadata requires Zhenwu M890")

    from vllm_sail.patch.enhancement.attention import mla_indexer
    from vllm_sail.utils import deep_gemm

    # Isolate the upstream constructor's unrelated model/distributed setup.
    # The patched constructor and metadata producer are both real providers.
    def upstream_init(builder):
        builder.device = "cuda"
        builder.indexer_uses_fp4 = True
        builder.kv_cache_spec = SimpleNamespace(
            indexer_n_head=heads, indexer_q_head_dim=128
        )
        builder.vllm_config = SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_batched_tokens=tokens),
            model_config=SimpleNamespace(hf_text_config=SimpleNamespace()),
        )

    monkeypatch.setattr(mla_indexer, "_upstream_builder_init", upstream_init)
    builder = SimpleNamespace()
    mla_indexer._builder_init_wrapper(builder)
    contexts = torch.full((tokens, 1), 128, dtype=torch.int32, device="cuda")
    request_ids = torch.arange(tokens, dtype=torch.int32, device="cuda") // 2
    metadata = deep_gemm.get_paged_mqa_logits_metadata(
        contexts, 64, builder.num_sms, (1, heads, 128, torch.int8), indices=request_ids
    )
    assert metadata.shape[1] == 4
    assert metadata.shape[0] > tokens
    # Exact operation in the upstream builder; no padding/truncation adapter.
    schedule = builder.scheduler_metadata_buffer[: metadata.shape[0]]
    schedule[:] = metadata
    torch.cuda.synchronize()
    assert torch.equal(schedule, metadata)


def _allocate_v41_indexer_cache(compress_ratio, dim):
    import torch
    from vllm.models.deepseek_v41.attention import DeepseekV4IndexerCache
    from vllm.v1.kv_cache_interface import (
        KVCacheLayout,
        KVCacheTensor,
        create_kv_cache_views,
    )
    from vllm.v1.worker.utils import select_common_block_size

    # No weights/config construction is needed to exercise the model's actual
    # cache-spec producer, backend selector and padded BLHNC allocation.
    layer = SimpleNamespace(
        head_dim=dim + 4,
        dtype=torch.uint8,
        cache_config=SimpleNamespace(block_size=128),
        compress_ratio=compress_ratio,
        sparse_logits=False,
        rocm_mxfp4=False,
    )
    config = SimpleNamespace(cache_config=SimpleNamespace(cache_dtype="fp8_ds_mla"))
    spec = DeepseekV4IndexerCache.get_kv_cache_spec(layer, config)
    backend = DeepseekV4IndexerCache.get_attn_backend(layer)
    kernel_block = select_common_block_size(spec.block_size, [backend])
    assert spec.block_size == kernel_block == 64 * compress_ratio
    # Leave space for another layer's page between consecutive indexer pages.
    stride = spec.page_size_bytes + 512
    raw = torch.zeros(8 * stride, dtype=torch.uint8, device="cuda")
    allocation = KVCacheTensor(
        size=raw.numel(),
        layers=["indexer"],
        layer_stride=spec.page_size_bytes,
        block_stride=stride,
    )
    views = create_kv_cache_views(
        raw, spec, 8, KVCacheLayout.BLHNC, allocation, kernel_block
    )
    DeepseekV4IndexerCache.bind_kv_cache(layer, views[0])
    assert layer.kv_cache.shape == (8, 64, dim + 4)
    assert layer.kv_cache.stride(0) == stride
    return layer.kv_cache.unsqueeze(-2)


def test_adaptive_indexer_backend_gate():
    import vllm_sail

    vllm_sail.register_out_of_tree()
    from vllm.v1.attention.backends.mla.indexer import DeepseekV41IndexerBackend
    from vllm.v1.worker.gpu.attn_utils import (
        get_query_lens_mismatch_unsupported_backend,
    )

    group = SimpleNamespace(backend=DeepseekV41IndexerBackend, layer_names=["indexer"])
    assert get_query_lens_mismatch_unsupported_backend([[group]]) is None


def _replay_flattened_logits(compute, q, cache, weights, compress_ratio, table_cpu):
    """Rebuild step metadata before replay, as the V2 target runner does."""
    import torch
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadataBuilder

    from vllm_sail.utils.deep_gemm import get_num_sms

    capacity = q.shape[0]
    builder = object.__new__(DeepseekV32IndexerMetadataBuilder)
    builder.vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(enable_adaptive_verification=True)
    )
    builder.supports_varlen = builder.indexer_uses_fp4 = False
    builder.use_flattening = True
    builder.num_speculative_tokens = 5
    builder.decode_threshold = 6
    builder.use_pcp = False
    builder.pcp_world_size = builder.dcp_world_size = 1
    builder.cp_kv_cache_interleave_size = 1
    builder.compress_ratio = compress_ratio
    builder.indexer_n_head, builder.indexer_q_head_dim = q.shape[-2:]
    builder.kv_cache_spec = SimpleNamespace(num_states=cache.shape[1])
    builder.num_sms = get_num_sms()
    builder.scheduler_metadata_buffer = torch.empty(
        (builder.num_sms * 8 + 1, 2), dtype=torch.int32, device=q.device
    )
    builder.decode_seq_lens_buffer = torch.zeros(
        capacity, dtype=torch.int32, device=q.device
    )
    builder.decode_lens_buffer = torch.zeros_like(builder.decode_seq_lens_buffer)
    builder.per_req_decode_lens_buffer = torch.zeros_like(
        builder.decode_seq_lens_buffer
    )
    builder.expanded_block_table_buffer = table_cpu.new_zeros(
        (capacity, table_cpu.shape[1]), device=q.device
    )
    builder.arange_buffer = torch.arange(capacity, dtype=torch.int32, device=q.device)
    builder.compressed_slot_mapping_buffer = torch.empty(
        capacity, dtype=torch.int64, device=q.device
    )
    context = torch.tensor([127, 185], dtype=torch.int32, device=q.device)
    starts_cpu = torch.zeros(3, dtype=torch.int32)
    starts = starts_cpu.to(q.device)
    seq_lens = torch.empty_like(context)
    common = CommonAttentionMetadata(
        query_start_loc=starts,
        query_start_loc_cpu=starts_cpu,
        seq_lens=seq_lens,
        num_reqs=2,
        num_actual_tokens=capacity,
        max_query_len=6,
        max_seq_len=256,
        block_table_tensor=table_cpu.to(q.device),
        slot_mapping=torch.full((capacity,), -1, dtype=torch.int64, device=q.device),
    )

    def prepare(host_counts, device_counts):
        assert sum(host_counts) == sum(device_counts)
        starts_cpu[1:].copy_(torch.tensor(host_counts).cumsum(0))
        starts[1:].copy_(torch.tensor(device_counts).cumsum(0))
        seq_lens.copy_(context + torch.tensor(device_counts, device=q.device))
        decode = builder.build(0, common).decode
        assert decode is not None and not decode.requires_padding
        assert decode.indices is None
        return decode

    def inputs(decode):
        return decode.seq_lens, decode.block_table, decode.schedule_metadata

    def forward(decode):
        return compute(q, cache, weights, *inputs(decode), 256, False)

    steps = [((3, 3), (5, 1)), ((3, 3), (1, 5)), ((2, 1), (1, 2))]
    captured = prepare(*steps[0])
    forward(captured)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        logits = forward(captured)
    for host_counts, device_counts in steps:
        decode = prepare(host_counts, device_counts)
        assert [tensor.data_ptr() for tensor in inputs(decode)] == [
            tensor.data_ptr() for tensor in inputs(captured)
        ]
        graph.replay()
        yield (
            device_counts,
            decode.seq_lens.cpu(),
            decode.block_table.cpu(),
            decode.decode_lens.cpu(),
            logits.cpu(),
        )


@pytest.mark.parametrize("compress_ratio", [1, 2])
def test_device_lengths_flattened_logits_graph_replay(compress_ratio):
    import torch

    # SM80 PPUs such as 810E have no FP8 paged-logits kernel; see the INT8 case below.
    if torch.cuda.get_device_capability() == (8, 0):
        pytest.skip("FP8 paged MQA logits are not supported on SM80 devices.")

    import vllm_sail

    vllm_sail.register_out_of_tree()
    from vllm_sail.utils.deep_gemm import fp8_paged_mqa_logits

    # Use the real V4.1 cache spec/backend/allocation path. Hard-coding a
    # 64-row tensor here hid the model's unsupported 128-row C1 pages.
    capacity, heads, dim = 8, 64, 128
    cache = _allocate_v41_indexer_cache(compress_ratio, dim)
    block_size = cache.shape[1]
    assert block_size == 64
    table_cpu = torch.tensor([[4, 0, 6, 2], [5, 1, 7, 3]], dtype=torch.int32)
    generator = torch.Generator().manual_seed(83)
    q_cpu = (torch.randn(capacity, 1, heads, dim, generator=generator) / 8).to(
        torch.float8_e4m3fn
    )
    k_cpu = (torch.randn(8, block_size, dim, generator=generator) / 8).to(
        torch.float8_e4m3fn
    )
    scales_cpu = torch.rand(8, block_size, generator=generator) / 2 + 0.25
    weight_cpu = torch.rand(capacity, heads, generator=generator) / heads
    packed = torch.cat(
        (k_cpu.view(torch.uint8).flatten(1), scales_cpu.view(torch.uint8).flatten(1)),
        dim=1,
    )
    cache.copy_(packed.reshape(8, block_size, 1, dim + 4))
    q, weights = q_cpu.cuda(), weight_cpu.cuda()

    for device_lens, ends, blocks, lengths, logits in _replay_flattened_logits(
        fp8_paged_mqa_logits, q, cache, weights, compress_ratio, table_cpu
    ):
        tokens = sum(device_lens)
        expected_ends, request_ids = [], []
        for req, count in enumerate(device_lens):
            base = (127, 185)[req]
            expected_ends.extend((base + i + 1) // compress_ratio for i in range(count))
            request_ids.extend([req] * count)
        assert ends.flatten().tolist() == expected_ends + [0] * (capacity - tokens)
        assert lengths.tolist() == [1] * capacity
        torch.testing.assert_close(
            blocks[:tokens], table_cpu[request_ids], rtol=0, atol=0
        )
        for row, (req, end) in enumerate(zip(request_ids, expected_ends, strict=False)):
            positions = torch.arange(end)
            pages = table_cpu[req, positions // block_size].long()
            slots = positions % block_size
            keys = k_cpu.float()[pages, slots] * scales_cpu[pages, slots, None]
            reference = (
                (q_cpu[row, 0].float() @ keys.T).clamp_min(0) * weight_cpu[row, :, None]
            ).sum(0)
            torch.testing.assert_close(
                logits[row, :end], reference, rtol=0.01, atol=0.002
            )


@pytest.mark.parametrize("compress_ratio", [1, 2])
def test_int8_device_lengths_flattened_logits_graph_replay(compress_ratio):
    import torch

    import vllm_sail

    vllm_sail.register_out_of_tree()
    from vllm_sail.utils.deep_gemm import int8_paged_mqa_logits

    # Use the real V4.1 cache spec/backend/allocation path. Hard-coding a
    # 64-row tensor here hid the model's unsupported 128-row C1 pages.
    capacity, heads, dim = 8, 64, 128
    cache = _allocate_v41_indexer_cache(compress_ratio, dim)
    block_size = cache.shape[1]
    assert block_size == 64
    table_cpu = torch.tensor([[4, 0, 6, 2], [5, 1, 7, 3]], dtype=torch.int32)
    generator = torch.Generator().manual_seed(83)
    q_cpu = torch.randint(
        -16, 17, (capacity, 1, heads, dim), generator=generator, dtype=torch.int8
    )
    k_cpu = torch.randint(
        -16, 17, (8, block_size, dim), generator=generator, dtype=torch.int8
    )
    scales_cpu = torch.rand(8, block_size, generator=generator) / 128 + 0.002
    q_scales_cpu = torch.rand(capacity, heads, generator=generator) / 128 + 0.002
    weight_cpu = torch.rand(capacity, heads, generator=generator) / heads
    # Fold INT8 Q scales into head weights; keep K values and FP32 scales planar.
    weight_cpu = weight_cpu * q_scales_cpu
    packed = torch.cat(
        (k_cpu.view(torch.uint8).flatten(1), scales_cpu.view(torch.uint8).flatten(1)),
        dim=1,
    )
    cache.copy_(packed.reshape(8, block_size, 1, dim + 4))
    q, weights = q_cpu.cuda(), weight_cpu.cuda()

    for device_lens, ends, blocks, lengths, logits in _replay_flattened_logits(
        int8_paged_mqa_logits, q, cache, weights, compress_ratio, table_cpu
    ):
        tokens = sum(device_lens)
        expected_ends, request_ids = [], []
        for req, count in enumerate(device_lens):
            base = (127, 185)[req]
            expected_ends.extend((base + i + 1) // compress_ratio for i in range(count))
            request_ids.extend([req] * count)
        assert ends.flatten().tolist() == expected_ends + [0] * (capacity - tokens)
        assert lengths.tolist() == [1] * capacity
        torch.testing.assert_close(
            blocks[:tokens], table_cpu[request_ids], rtol=0, atol=0
        )
        for row, (req, end) in enumerate(zip(request_ids, expected_ends, strict=False)):
            positions = torch.arange(end)
            pages = table_cpu[req, positions // block_size].long()
            slots = positions % block_size
            keys = k_cpu.float()[pages, slots] * scales_cpu[pages, slots, None]
            reference = (
                (q_cpu[row, 0].float() @ keys.T).clamp_min(0) * weight_cpu[row, :, None]
            ).sum(0)
            torch.testing.assert_close(
                logits[row, :end], reference, rtol=0.01, atol=0.002
            )


@pytest.mark.parametrize("next_n", [1, 3])
@pytest.mark.parametrize("per_query_context", [False, True])
@pytest.mark.parametrize("clean", [False, True])
def test_int8_paged_causal_prefixes_and_padding_graph(next_n, per_query_context, clean):
    import torch

    import vllm_sail

    vllm_sail.register_out_of_tree()
    from vllm_sail.utils.deep_gemm import (
        get_num_sms,
        get_paged_mqa_logits_metadata,
        int8_paged_mqa_logits,
    )

    batch, heads, dim = 3, 32, 128
    cache = _allocate_v41_indexer_cache(1, dim)
    generator = torch.Generator().manual_seed(14281)
    q_cpu = torch.randint(
        -16, 17, (batch, next_n, heads, dim), generator=generator, dtype=torch.int8
    )
    k_cpu = torch.randint(-16, 17, (8, 64, dim), generator=generator, dtype=torch.int8)
    scale_cpu = torch.rand(8, 64, generator=generator) / 128 + 0.002
    q_scale_cpu = torch.rand(batch * next_n, heads, generator=generator) / 128 + 0.002
    weights_cpu = torch.randn(batch * next_n, heads, generator=generator) / heads
    weights_cpu = weights_cpu * q_scale_cpu
    packed = torch.cat(
        (k_cpu.view(torch.uint8).flatten(1), scale_cpu.view(torch.uint8).flatten(1)),
        dim=1,
    )
    cache.copy_(packed.reshape_as(cache))
    table_cpu = torch.tensor(
        [[4, 0, 6, 2], [5, 1, 7, 3], [-1, -1, -1, -1]], dtype=torch.int32
    )
    first = torch.tensor([70, 131, 0], dtype=torch.int32)
    second = torch.tensor([72, 134, 0], dtype=torch.int32)
    if per_query_context:
        offsets = torch.arange(next_n, dtype=torch.int32) - next_n + 1
        first = (first[:, None] + offsets[None, :]).clamp_min(0)
        second = (second[:, None] + offsets[None, :]).clamp_min(0)
    context = first.cuda()
    q, weights, table = q_cpu.cuda(), weights_cpu.cuda(), table_cpu.cuda()
    sms = get_num_sms()

    def step():
        # Match production: INT8 SDK metadata and logits use 2D visible prefixes.
        metadata_context = context
        if not per_query_context:
            offsets = torch.arange(next_n, dtype=context.dtype, device=context.device)
            metadata_context = (
                context[:, None] - next_n + offsets[None, :] + 1
            ).clamp_min(0)
        metadata = get_paged_mqa_logits_metadata(
            metadata_context, 64, sms, (next_n, heads, dim, 1)
        )
        logits = int8_paged_mqa_logits(
            q,
            cache,
            weights,
            metadata_context,
            table,
            metadata,
            256,
            False,
        )
        if clean:
            # The SDK guarantees only visible prefixes; validate external masking.
            columns = torch.arange(256, device=context.device)
            logits = logits.masked_fill(
                columns[None, :] >= metadata_context.reshape(-1, 1), float("-inf")
            )
        return logits

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = step()
    for values in (first, second):
        context.copy_(values)
        graph.replay()
        result = actual.cpu()
        for request in range(batch):
            for query in range(next_n):
                end = (
                    int(values[request, query])
                    if per_query_context
                    else int(values[request]) - next_n + query + 1
                )
                end = max(0, end)
                row = request * next_n + query
                positions = torch.arange(end)
                pages = table_cpu[request, positions // 64].long()
                slots = positions % 64
                keys = k_cpu.float()[pages, slots] * scale_cpu[pages, slots, None]
                expected = (
                    (q_cpu[request, query].float() @ keys.T).clamp_min(0)
                    * weights_cpu[row, :, None]
                ).sum(0)
                torch.testing.assert_close(
                    result[row, :end], expected, rtol=0.01, atol=0.002
                )
                if clean:
                    assert torch.isneginf(result[row, end:]).all()
