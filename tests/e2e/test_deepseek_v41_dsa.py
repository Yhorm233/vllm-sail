# SPDX-License-Identifier: Apache-2.0
"""810E V4.1 DSA: INT8 logits, candidate filtering, and native TopK.

The reference uses CPU FP32 tensors without invoking production kernels.
"""

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.ppu
BLOCK_SIZE = 8
TOPK_BLOCKS = 2048
TOPK_TOKENS = 512
# Use fixed tolerances and absolute weighted sums for signed-weight cancellation.
LOGITS_RTOL = 1e-3
LOGITS_ATOL = 1e-4


@pytest.fixture(scope="module")
def runtime():
    if not torch.cuda.is_available():
        pytest.skip("An 810E device is required")
    import vllm_sail

    vllm_sail.register_out_of_tree()
    from vllm.platforms import current_platform

    if not current_platform.is_ppu() or torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("This test matrix supports only 810E")
    current_platform.import_kernels()
    from vllm_sail.attention.ops import mla_sparse
    from vllm_sail.utils import deep_gemm

    assert deep_gemm.is_deep_gemm_supported(), "SAIL DeepGEMM is unavailable"
    return mla_sparse, deep_gemm


def block_scores_cpu(logits, starts, ends, repeat=1):
    rows, width = logits.shape
    scores = torch.full((rows, (width + BLOCK_SIZE - 1) // BLOCK_SIZE), -float("inf"))
    for row in range(rows):
        start = 0 if starts is None else int(starts[row // repeat])
        end = int(ends[row // repeat])
        for block in range((max(0, end - start) + BLOCK_SIZE - 1) // BLOCK_SIZE):
            lo = start + block * BLOCK_SIZE
            hi = min(lo + BLOCK_SIZE, end, width)
            if lo < hi:
                scores[row, block] = logits[row, lo:hi].max()
        if end > start:
            scores[row, (end - start - 1) // BLOCK_SIZE] = float("inf")
    return scores


def mask_cpu(logits, starts, ends, candidates, repeat=1):
    result = torch.full_like(logits, -float("inf"))
    width = logits.shape[1]
    for row in range(logits.shape[0]):
        start = 0 if starts is None else int(starts[row // repeat])
        end = int(ends[row // repeat])
        for block in candidates[row].tolist():
            if block < 0:
                continue
            lo = start + block * BLOCK_SIZE
            if lo >= width:
                # Upstream contract: candidates beyond the packed width retain only the last column.
                if start <= width - 1 < end:
                    result[row, -1] = logits[row, -1]
            else:
                hi = min(lo + BLOCK_SIZE, end, width)
                if lo < hi:
                    result[row, lo:hi] = logits[row, lo:hi]
    return result


def assert_topk(scores, indices, k):
    """Check the selected set; for ties, check the threshold and all higher scores, ignoring order."""
    assert indices.numel() == k
    assert torch.all(indices >= -1)
    chosen = indices[indices >= 0].long()
    valid = torch.nonzero(scores != -float("inf")).flatten()
    count = min(k, valid.numel())
    assert chosen.numel() == count
    assert chosen.unique().numel() == count
    assert torch.all(chosen < scores.numel())
    if not count:
        return
    threshold = scores[valid].topk(count).values[-1]
    assert torch.all(scores[chosen] >= threshold)
    mandatory = set(torch.nonzero(scores > threshold).flatten().tolist())
    assert mandatory <= set(chosen.tolist())
    if scores[valid].unique().numel() == valid.numel():
        assert set(chosen.tolist()) == set(
            valid[scores[valid].topk(count).indices].tolist()
        )


@pytest.mark.parametrize("width", [512, 1536, 16385, 32768, 102400])
@pytest.mark.parametrize("decode", [False, True])
@pytest.mark.parametrize("ties", [False, True])
def test_candidate_write_mask_and_request_isolation(runtime, width, decode, ties):
    sparse, _ = runtime
    repeat = 2 if decode else 1
    starts = None if decode else torch.tensor([0, 7, 33, 0], dtype=torch.int32)
    ends = torch.tensor([0, 8, width - 17, width], dtype=torch.int32)
    rows = ends.numel() * repeat
    generator = torch.Generator().manual_seed(1017)
    logits = torch.stack(
        [torch.randperm(width, generator=generator).float() for _ in range(rows)]
    )
    if ties:
        logits = (logits % 5) - 2
    # Lower the latest block's scores to verify forced retention.
    for row in range(rows):
        start = 0 if starts is None else int(starts[row // repeat])
        end = int(ends[row // repeat])
        if end > start:
            lo = start + (end - start - 1) // BLOCK_SIZE * BLOCK_SIZE
            logits[row, lo:end] = -1000000
    backing = torch.full((rows, width * 2 + 8), 193.0, device="cuda")
    device_logits = backing[:, 4 : 4 + width * 2 : 2]
    device_logits.copy_(logits)
    out_backing = torch.full(
        (rows, TOPK_BLOCKS * 2 + 8), 193, dtype=torch.int32, device="cuda"
    )
    candidates = out_backing[:, 4 : 4 + TOPK_BLOCKS * 2 : 2]
    start_gpu = None if starts is None else starts.cuda()
    end_gpu = ends.cuda()
    sparse._filter_candidates(
        device_logits, start_gpu, end_gpu, candidates, BLOCK_SIZE, True, repeat
    )
    scores = block_scores_cpu(logits, starts, ends, repeat)
    selected = candidates.cpu()
    for row in range(rows):
        assert_topk(scores[row], selected[row], TOPK_BLOCKS)
    torch.testing.assert_close(device_logits.cpu(), logits, rtol=0, atol=0)
    assert torch.all(out_backing[:, :4] == 193) and torch.all(
        out_backing[:, -4:] == 193
    )
    assert torch.all(out_backing[:, 5:-4:2] == 193)
    sparse._filter_candidates(
        device_logits, start_gpu, end_gpu, candidates, BLOCK_SIZE, False, repeat
    )
    expected = mask_cpu(logits, starts, ends, selected, repeat)
    torch.testing.assert_close(device_logits.cpu(), expected, rtol=0, atol=0)
    assert torch.all(backing[:, :4] == 193) and torch.all(backing[:, -4:] == 193)
    assert torch.all(backing[:, 5:-4:2] == 193)
    if width > TOPK_BLOCKS * BLOCK_SIZE:
        assert torch.isneginf(expected[-1]).any(), (
            "Long contexts must trigger candidate filtering"
        )
    # Repeated masking is idempotent and must not modify the shared candidate buffer.
    sparse._filter_candidates(
        device_logits, start_gpu, end_gpu, candidates, BLOCK_SIZE, False, repeat
    )
    torch.testing.assert_close(device_logits.cpu(), expected, rtol=0, atol=0)
    assert torch.equal(candidates.cpu(), selected)


def test_candidate_mask_invalid_ids_and_empty(runtime):
    sparse, _ = runtime
    logits = torch.arange(64).float().reshape(2, 32)
    starts, ends = (
        torch.tensor([0, 3], dtype=torch.int32),
        torch.tensor([32, 29], dtype=torch.int32),
    )
    candidates = torch.tensor([[-1, 0, 0, 99999], [-1, 0, 2, 99999]], dtype=torch.int32)
    actual = logits.cuda()
    sparse._filter_candidates(
        actual, starts.cuda(), ends.cuda(), candidates.cuda(), BLOCK_SIZE, False
    )
    torch.testing.assert_close(
        actual.cpu(), mask_cpu(logits, starts, ends, candidates), rtol=0, atol=0
    )
    for rows, width in ((0, 32), (2, 0)):
        values = torch.empty(rows, width, device="cuda")
        selected = torch.full(
            (rows, TOPK_BLOCKS), 193, dtype=torch.int32, device="cuda"
        )
        bounds = torch.zeros(rows, dtype=torch.int32, device="cuda")
        sparse._filter_candidates(values, None, bounds, selected, BLOCK_SIZE, True)
        assert torch.all(selected == -1)
        sparse._filter_candidates(values, None, bounds, selected, BLOCK_SIZE, False)


def logits_inputs(rows, length):
    generator = torch.Generator().manual_seed(41019)
    q = torch.randint(-127, 128, (rows, 32, 128), generator=generator, dtype=torch.int8)
    k = torch.randint(-127, 128, (length, 128), generator=generator, dtype=torch.int8)
    qs = torch.rand(rows, 32, generator=generator) * 0.01 + 0.001
    ks = torch.rand(length, generator=generator) * 0.01 + 0.001
    weights = torch.randn(rows, 32, generator=generator) / 32
    return q, k, qs, ks, weights


def logits_cpu(q, k, qs, ks, weights):
    dots = torch.einsum(
        "mhd,nd->mhn", q.float() * qs[..., None], k.float() * ks[:, None]
    )
    terms = dots.relu() * weights[..., None]
    return terms.sum(1), terms.abs().sum(1)


def assert_logits(actual, expected, magnitude):
    assert torch.isfinite(actual).all()
    assert torch.all((actual - expected).abs() <= LOGITS_ATOL + LOGITS_RTOL * magnitude)


@pytest.mark.parametrize("clean", [False, True])
def test_int8_prefill_logits_against_dequantized_fp32(runtime, clean):
    _, deep_gemm = runtime
    q, k, qs, ks, weights = logits_inputs(16, 4096)
    starts = torch.tensor([0] * 8 + [2048] * 8, dtype=torch.int32)
    ends = starts + torch.tensor(
        [1, 16, 511, 512, 513, 1023, 1536, 2048] * 2, dtype=torch.int32
    )
    args = (
        q.cuda(),
        (k.cuda(), ks.cuda()),
        (weights * qs).cuda(),
        starts.cuda(),
        ends.cuda(),
    )
    result = deep_gemm.int8_mqa_logits(*args, clean_logits=clean).cpu()
    again = deep_gemm.int8_mqa_logits(*args, clean_logits=clean).cpu()
    expected, magnitude = logits_cpu(q, k, qs, ks, weights)
    assert result.shape == expected.shape and result.dtype == torch.float32
    for row, (start, end) in enumerate(
        zip(starts.tolist(), ends.tolist(), strict=True)
    ):
        assert_logits(
            result[row, start:end], expected[row, start:end], magnitude[row, start:end]
        )
        assert torch.equal(result[row, start:end], again[row, start:end])
        if clean:
            assert torch.isneginf(result[row, :start]).all()
            assert torch.isneginf(result[row, end:]).all()


def allocate_model_indexer_cache(ratio, shared, blocks):
    """Allocate through the model spec, backend, and BLHNC layout to preserve layout differences."""
    from types import SimpleNamespace

    from vllm.models.deepseek_v41.attention import DeepseekV4IndexerCache
    from vllm.v1.kv_cache_interface import (
        KVCacheLayout,
        KVCacheTensor,
        create_kv_cache_views,
    )
    from vllm.v1.worker.utils import select_common_block_size

    layer = SimpleNamespace(
        head_dim=132,
        dtype=torch.uint8,
        compress_ratio=ratio,
        cache_config=SimpleNamespace(block_size=128),
        sparse_logits=False,
    )
    config = SimpleNamespace(
        cache_config=SimpleNamespace(cache_dtype="fp8_ds_mla"),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(index_n_heads=32, index_head_dim=128)
        ),
    )
    spec = DeepseekV4IndexerCache.get_kv_cache_spec(layer, config)
    backend = DeepseekV4IndexerCache.get_attn_backend(layer)
    kernel_block = select_common_block_size(spec.block_size, [backend])
    assert spec.block_size == kernel_block == 64 * ratio
    layers = ["before", "indexer", "after"] if shared else ["indexer"]
    stride = spec.page_size_bytes * len(layers)
    raw = torch.full((blocks * stride,), 193, dtype=torch.uint8, device="cuda")
    allocation = KVCacheTensor(
        size=raw.numel(),
        layers=layers,
        layer_stride=spec.page_size_bytes,
        block_stride=stride,
    )
    views = create_kv_cache_views(
        raw, spec, blocks, KVCacheLayout.BLHNC, allocation, kernel_block
    )
    DeepseekV4IndexerCache.bind_kv_cache(layer, views[1 if shared else 0])
    assert layer.kv_cache.shape == (blocks, 64, 132)
    assert layer.kv_cache.stride(0) == stride
    return layer.kv_cache, raw


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("ratio", [1, 2])
def test_int8_paged_logits_and_native_gather(runtime, shared, ratio):
    _, deep_gemm = runtime
    from deep_gemm import get_num_sms
    from vllm import _custom_ops as ops

    lengths = torch.tensor([511, 1024, 1536], dtype=torch.int32)
    block_size, blocks_per_req = 64, 24
    q, k, qs, ks, weights = logits_inputs(3, 3 * 1536)
    generator = torch.Generator().manual_seed(41020)
    table = torch.randperm(72, generator=generator).reshape(3, blocks_per_req).int()
    cache, backing = allocate_model_indexer_cache(ratio, shared, 72)
    host = backing.cpu()
    host_cache = torch.as_strided(
        host, cache.shape, cache.stride(), cache.storage_offset()
    )
    for request in range(3):
        for block in range(blocks_per_req):
            lo = request * 1536 + block * block_size
            page = host_cache[table[request, block]].flatten()
            page[: block_size * 128] = (
                k[lo : lo + block_size].view(torch.uint8).flatten()
            )
            page[block_size * 128 :] = ks[lo : lo + block_size].view(torch.uint8)
    backing.copy_(host)
    table_gpu, context = table.cuda(), lengths[:, None].cuda()
    schedule = deep_gemm.get_paged_mqa_logits_metadata(
        context, block_size, get_num_sms(), (1, 32, 128, 1)
    )
    args = (
        q[:, None].cuda(),
        cache.unsqueeze(-2),
        (weights * qs).cuda(),
        context,
        table_gpu,
        schedule,
    )
    # Match the model: the SDK writes the valid prefix for 2D context; TopK excludes the tail.
    device_result = deep_gemm.int8_paged_mqa_logits(
        *args, max_model_len=2048, clean_logits=False
    )
    result = device_result.cpu()
    again = deep_gemm.int8_paged_mqa_logits(
        *args, max_model_len=2048, clean_logits=False
    ).cpu()
    assert result.shape == (3, 2048) and result.dtype == torch.float32
    for row, length in enumerate(lengths.tolist()):
        lo = row * 1536
        expected, magnitude = logits_cpu(
            q[row : row + 1],
            k[lo : lo + length],
            qs[row : row + 1],
            ks[lo : lo + length],
            weights[row : row + 1],
        )
        assert_logits(result[row, :length], expected[0], magnitude[0])
        assert torch.equal(result[row, :length], again[row, :length])
        device_result[row, length:] = float("inf")
    selected = torch.full((3, TOPK_TOKENS), -2, dtype=torch.int32, device="cuda")
    workspace = torch.empty(1024 * 1024, dtype=torch.uint8, device="cuda")
    torch.ops._C.persistent_topk(
        device_result, context, selected, workspace, TOPK_TOKENS, 2048
    )
    for row, length in enumerate(lengths.tolist()):
        assert_topk(result[row, :length], selected[row].cpu(), TOPK_TOKENS)
    total = int(lengths.sum())
    gathered = torch.empty(total, 128, dtype=torch.int8, device="cuda")
    scale_bytes = torch.empty(total, 4, dtype=torch.uint8, device="cuda")
    cu = torch.cat((torch.zeros(1, dtype=torch.int32), lengths.cumsum(0).int())).cuda()
    ops.cp_gather_indexer_k_quant_cache(cache, gathered, scale_bytes, table_gpu, cu)
    expected_k = torch.cat(
        [
            k[row * 1536 : row * 1536 + length]
            for row, length in enumerate(lengths.tolist())
        ]
    )
    expected_s = torch.cat(
        [
            ks[row * 1536 : row * 1536 + length]
            for row, length in enumerate(lengths.tolist())
        ]
    )
    assert torch.equal(gathered.cpu(), expected_k)
    assert torch.equal(scale_bytes.cpu().view(torch.float32).flatten(), expected_s)
    assert torch.equal(backing.cpu(), host), (
        "Consumer operators must not modify the cache or adjacent shared layers"
    )


@pytest.mark.parametrize("decode", [False, True])
@pytest.mark.parametrize("ties", [False, True])
def test_native_topk_causal_bounds_and_ties(runtime, decode, ties):
    width = 2048
    generator = torch.Generator().manual_seed(41021)
    lengths = torch.tensor([0, 1, 511, 512, 513, 1024, 1536], dtype=torch.int32)
    starts = (
        torch.zeros_like(lengths)
        if decode
        else torch.tensor([0, 3, 17, 41, 63, 5, 129], dtype=torch.int32)
    )
    ends = starts + lengths
    logits = torch.stack(
        [torch.randperm(width, generator=generator).float() for _ in lengths]
    )
    if ties:
        logits = (logits % 7) - 3
    device_logits = logits.cuda()
    output = torch.full(
        (len(lengths), TOPK_TOKENS), -2, dtype=torch.int32, device="cuda"
    )
    if decode:
        workspace = torch.empty(1024 * 1024, dtype=torch.uint8, device="cuda")
        torch.ops._C.persistent_topk(
            device_logits,
            lengths[:, None].cuda(),
            output,
            workspace,
            TOPK_TOKENS,
            width,
        )
    else:
        torch.ops._C.top_k_per_row_prefill(
            device_logits,
            starts.cuda(),
            ends.cuda(),
            output,
            len(lengths),
            *device_logits.stride(),
            TOPK_TOKENS,
        )
    actual = output.cpu()
    for row, (start, end) in enumerate(
        zip(starts.tolist(), ends.tolist(), strict=True)
    ):
        assert_topk(logits[row, start:end], actual[row], TOPK_TOKENS)
    assert torch.equal(device_logits.cpu(), logits)


def test_reference_contract_cpu(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Device initialization is forbidden in the CPU oracle")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(torch._C, "_cuda_getDeviceCount", forbidden)
    logits = torch.arange(64).float().reshape(2, 32)
    starts = torch.tensor([0, 8])
    ends = torch.tensor([9, 25])
    scores = block_scores_cpu(logits, starts, ends)
    assert scores[0].tolist() == [7, float("inf"), -float("inf"), -float("inf")]
    assert scores[1].tolist() == [47, 55, float("inf"), -float("inf")]
    candidates = torch.tensor([[1, -1], [0, 2]])
    masked = mask_cpu(logits, starts, ends, candidates)
    assert torch.nonzero(torch.isfinite(masked[0])).flatten().tolist() == [8]
    assert torch.nonzero(torch.isfinite(masked[1])).flatten().tolist() == list(
        range(8, 16)
    ) + [24]
    assert_topk(torch.tensor([5.0, 5.0, 4.0]), torch.tensor([1, 0]), 2)
    with pytest.raises(AssertionError):
        assert_topk(torch.tensor([5.0, 5.0, 4.0]), torch.tensor([0, 2]), 2)
    with pytest.raises(AssertionError):
        assert_topk(torch.tensor([5.0, 5.0, 4.0]), torch.tensor([0, 0]), 2)
    q, k = (
        torch.ones(1, 32, 128, dtype=torch.int8),
        torch.ones(1, 128, dtype=torch.int8),
    )
    value, magnitude = logits_cpu(
        q, k, torch.ones(1, 32) * 0.5, torch.ones(1) * 0.25, torch.ones(1, 32)
    )
    assert value.item() == magnitude.item() == 512
