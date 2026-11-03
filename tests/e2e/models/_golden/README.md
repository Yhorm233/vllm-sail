# Golden baselines — accuracy model E2E (schema v2)

Per-chip evalscope GSM8K accuracy baselines for the model-precision suite
(`tests/e2e/models`). Baselines are **captured on real PPU hardware**, never
hand-written.

## What changed (schema v1 → v2)

Schema v1 stored per-token logprob goldens; schema v2 stores a single accuracy
score from evalscope GSM8K evaluation. The primary gate is now
**`score >= threshold`** (default threshold = **0.9**). The golden file refines
this with a regression tolerance once a baseline is established.

## File naming

```
accuracy_<model_key>_<chip>.json                   formal baseline
accuracy_<model_key>_<chip>.pending-review.json    unreviewed first-run output
```

`chip` is derived from the detected device capability (`(8,0) → 810e`,
`(8,9) → 890p`).

## Checkpoint resolution

The checkpoint paths in scripts and `scripts/ci/ppu_e2e_models.json` may be
recorded under either `/nas_aisw/datasets` or `/ppusw/datasets`. Both mount the
same NFS export. `resolve_checkpoint` tries the recorded path first and then
each mirrored root (`_offline_runner.CHECKPOINT_ROOT_MIRRORS`).

## Schema (`schema_version: 2`)

```jsonc
{
  "schema_version": 2,
  "model": "deepseek_v4_int8",        // model key
  "chip": "810e",                      // must equal the running device chip
  "checkpoint": "/ppusw/datasets/...", // resolved weight path
  "checkpoint_tail": "v4/DeepSeek-V4-Flash-0731-W8A8-INT8",  // identity
  "benchmark": "gsm8k",
  "evalscope_version": "0.7.2",
  "dataset": {
    "dataset_id": "/ppusw/datasets/vision/EvalScope/data/gsm8k/v2/gsm8k",
    "few_shot_num": 4,
    "few_shot_random": false,
    "subset_list": ["main"],
    "num_expected": 1319
  },
  "serving": {
    "tensor_parallel_size": 8,
    "max_model_len": 4096,
    "dtype": "bfloat16",
    "gpu_memory_utilization": 0.90,
    "enforce_eager": true,
    "seed": 0
  },
  "score": 0.9234,                     // evalscope accuracy (0.0–1.0)
  "score_num": 1319,                   // number of evaluated samples
  "tolerance": {
    "mode": "lower_bound",
    "rel": 0.99,
    "abs": 0.02,
    "hard_floor": 0.9                  // == threshold
  },
  "measured_at": "2025-07-15T06:00:00+00:00"
}
```

## Comparison rules

1. **Threshold gate** (primary): `score >= 0.9` passes. Configurable via
   `--threshold`.
2. **Identity gate**: `chip` must match; `checkpoint_tail` (when present) must
   match — prevents cross-chip or cross-checkpoint comparison.
3. **Golden regression gate** (when a formal golden exists):
   a. **Completeness**: `num == dataset.num_expected`.
   b. **Hard floor**: `score >= tolerance.hard_floor`.
   c. **Regression**: `score >= golden_score - max(abs, (1-rel)*golden_score)`.

## Nightly CI

`ppu-accuracy-nightly.yaml` runs every accuracy model once a night against the
wheels of the newest successful PPU CI build. The checkout inside the runtime
container is read-only, so `scripts/ci/ppu_e2e_accuracy.sh` copies every
`accuracy_*.json` from this directory into the writable results tree before
evaluating (`GOLDEN_SEED_DIR`, default `/source/tests/e2e/models/_golden`).
Consequences:

* A model with a committed golden here **is** gated on the regression bounds.
* A model without one reports `created`, uploads
  `accuracy_<model>_<chip>.pending-review.json` inside the run's
  `ppu-accuracy-*` artifact, and only enforces `score >= threshold`.

Promotion therefore stays a human step: review the pending artifact, commit it
here as `accuracy_<model>_<chip>.json`, and the next run gates against it.

`chip` is whatever device the nightly runner exposes, not the catalog `board`
field: the lane runs on the self-hosted `8-ppu-m890p` runner, so
`deepseek_v4_int8` is currently captured against the ZW-M890P int8 path.

## Capturing / refreshing

Run the eval script or test on the target PPU machine with `--update-golden`:

```bash
python scripts/eval/ppu_gsm8k_eval.py \
    --model-key deepseek_v4_int8 --update-golden
```

Or via pytest:

```bash
pytest tests/e2e/models/test_model_accuracy.py::test_deepseek_v4_int8_accuracy \
    --update-golden
```

## Pending-review → formal promotion

A `.pending-review.json` file never participates in comparison — an unreviewed
first run cannot silently become the baseline. Promote it by either:

1. Review the pending file, then copy it:
   `cp accuracy_<model>_<chip>.pending-review.json accuracy_<model>_<chip>.json`
2. Re-run with `--update-golden`, which writes the formal name directly and
   deletes the pending sibling.

## Model configurations

| Model key | Chip | Checkpoint |
|-----------|------|-----------|
| `deepseek_v4_int8` | 810e | `DeepSeek-V4-Flash-0731-W8A8-INT8` |
| `deepseek_v4_mxfp4` | 890p | `DeepSeek-V4-Flash-0731-mxfp4-fp8channel` |
| `minimax-m3-fp8` | 890p | `MiniMax-M3-Quant-W-FP8-PerChannel-A-FP8-PerToken` |
| `qwen3.8-flash-next` | 890p | `Qwen3.8-Flash-Next` |
