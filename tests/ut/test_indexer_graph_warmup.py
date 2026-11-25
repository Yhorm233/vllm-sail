# SPDX-License-Identifier: Apache-2.0
"""Short-context eager warmup must initialize the captured indexer path."""

from types import SimpleNamespace as NS

import pytest

from tests.support.modules import load_patch


@pytest.fixture
def harness(modules, patch_utils_module):
    state = NS(
        ppu=True, capturing=False, ready=False, calls=0, fail=False, max_seq_len=1
    )

    class Query:
        def view(self, *shape):
            return self

    class Indexer:
        k_cache = NS(prefix="cache")
        compress_ratio = 4
        topk_tokens = 512
        n_head = 64
        head_dim = 128

        def _wq_b_proj(self, qr, qr_scale):
            if state.capturing and not state.ready:
                raise RuntimeError("HGGC_ERROR_STREAM_CAPTURE_UNSUPPORTED")
            state.ready = True
            state.calls += 1
            if state.fail:
                raise ValueError("warmup failed")
            return Query()

        def _q_rope_quant(self, positions, q, rotary_emb, weights):
            return q, weights

        def forward(
            self,
            hidden,
            qr,
            kv,
            weights,
            positions,
            rope,
            qr_scale=None,
            skip_compressor=False,
        ):
            if state.max_seq_len // 4 <= 512 and not state.capturing:
                return None, None, None
            return self._wq_b_proj(qr, qr_scale)

    indexer = Indexer()

    def run():
        return indexer.forward(None, "qr", None, "weights", "pos", "rope")

    class Manager:
        def capture(self):
            run()
            state.capturing = True
            try:
                return run()
            finally:
                state.capturing = False

    modules("torch", cuda=NS(is_current_stream_capturing=lambda: state.capturing))
    modules("vllm.platforms", current_platform=NS(is_ppu=lambda: state.ppu))
    modules(
        "vllm.forward_context",
        get_forward_context=lambda: NS(
            attn_metadata={"cache": NS(max_seq_len=state.max_seq_len)}
        ),
    )
    modules("vllm.models.deepseek_v4.attention", DeepseekV4Indexer=Indexer)
    modules("vllm.v1.worker.gpu.cudagraph_utils", CudaGraphManager=Manager)
    relative = "vllm_sail/patch/enhancement/models/deepseek_v4_graph_warmup.py"
    load_patch(relative)
    return state, NS(capture=Manager().capture), run


def test_short_context_projection_is_ready_before_capture(harness):
    state, runner, run = harness
    assert run() == (None, None, None)
    assert state.calls == 0
    runner.capture()
    assert state.ready
    assert state.calls == 2
    run()
    assert state.calls == 2  # ordinary short-context inference stays cheap


def test_warmup_scope_restores_on_error(harness):
    state, runner, run = harness
    state.fail = True
    with pytest.raises(ValueError, match="warmup failed"):
        runner.capture()
    state.fail = False
    before = state.calls
    run()
    assert state.calls == before


def test_non_ppu_keeps_original_behavior(harness):
    state, runner, run = harness
    state.ppu = False
    with pytest.raises(RuntimeError, match="STREAM_CAPTURE_UNSUPPORTED"):
        runner.capture()
    assert state.calls == 0


def test_long_context_warmup_does_not_duplicate_projection(harness):
    state, runner, _ = harness
    state.max_seq_len = 8192
    runner.capture()
    assert state.calls == 2
