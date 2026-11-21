# Verification Guide

Validate the same package installation, SDK and hardware that will run the
workload. Keep the results of each check separate:

| Check | What it establishes |
| --- | --- |
| CPU unit tests | Python behavior, registration contracts and packaging invariants |
| Source regeneration checks | Generated upstream kernels match their declared inputs |
| HGGC build | Selected sources compile and link for the requested targets |
| Extension imports and strict preflight | Libraries load and representative operations are registered |
| Device correctness tests | Tested kernels match their numerical references on that device |
| Model run | The tested checkpoint and configuration complete inference |
| Benchmark | Performance under the recorded workload and environment |

Passing one check does not imply the later checks pass. A skipped test provides
no execution evidence for that feature.

## 1. Record the environment

After following [Installation](../getting_started/installation.md), run:

```bash
git rev-parse HEAD
python -c "from vllm_sail import collect_env; collect_env()"
python -c "import vllm_sail; print(vllm_sail.__file__)"
```

Record the PPU model and count, SDK version, SAIL PyTorch version, vLLM revision
and plugin revision. Confirm the printed package path points to the installation
being tested. A source checkout can shadow a separately installed native wheel.

Run the four-library import and strict preflight commands in
[Check the installation](../getting_started/installation.md#4-check-the-installation).
Resolve shared-library errors before running a model.

## 2. Run unit tests

From the plugin checkout, using its test environment:

```bash
python -m pytest tests/ut -q
```

The Python 3.10–3.13 CPU matrix uses only `requirements/dev.txt` and selects UT
that need no torch, vLLM, upstream checkout or device. The PPU pin-UT lane runs
the remaining source, installed-runtime and device UT with the exact upstream
checkout prepared. The selections are complementary: each UT belongs to one
runner tier, though CPU tests repeat across Python versions. Investigate failures
and report skips; skips are not passing runtime evidence.

Compatibility UT may intentionally import the plugin checkout beside installed
vLLM and torch. Record those import paths and set `VLLM_SOURCE_ROOT` to the same
exact upstream revision even when source-marked cases are deselected. This UT
mode does not verify the installed plugin wheel: separately compare its full
package contents with the wheel and run native imports, device tests and models
from that verified installation.

### Check upstream source compatibility

Tests live with the feature they cover, independently of the vLLM release that
introduced them. Tests marked `upstream_source` read a selected vLLM checkout
without importing vLLM, torch or the SDK:

```bash
VLLM_SOURCE_ROOT=/path/to/vllm python -m pytest tests/ut -q -rs \
  -m upstream_source --require-upstream-source
```

`VLLM_SOURCE_ROOT` names the checkout root containing `vllm/__init__.py`.
Without it, ordinary CPU runs skip source-dependent checks. The required-source
option makes a missing or invalid checkout an error, so CI cannot silently
skip these checks. The source tests cover selected upstream call sites, keyword
compatibility, quantization behavior and imported consumer inventories. They do
not establish that module imports, patch installation or device inference work.

### CI jobs

The [CPU matrix](../../.github/workflows/ut.yaml) runs for PRs through an
authorized `/ci run`, as well as pushes to `main` or `master` and manual dispatch.
Opening or updating a PR does not start it. CPU UT runs independently of the
PPU build, and both tiers contribute to the tested head's `PPU CI` status:

| Job | Environment and checks | Failure handling |
| --- | --- | --- |
| CPU UT | Python 3.10–3.13; only `requirements/dev.txt`; excludes `upstream_source`, `requires_runtime`, `ppu`, `cap80` and `cap89`; branch coverage | Test failure fails the workflow; coverage-upload failure does not |
| PPU pin UT | Installed vLLM, SAIL PyTorch, profiling tooling and PPU runtime; runs the complementary source/runtime/device UT selection | Missing source or test failure fails the PPU gate |

The selections are disjoint and cover all collected UT. Runtime requirements
must be marked on each mixed-file test, not on its fixture; fully runtime-only
modules can carry a module mark. No-vLLM-only contracts and pure Python logic
remain in the CPU lane. Dependency or platform skips are not passing evidence.

Pin-UT reads the full SHA in
[verified main commit](../../.github/vllm-main-verified.commit), verifies the
upstream checkout and mounts it read-only as `VLLM_SOURCE_ROOT`. It requires
that source path, so source contracts and independent packing references cannot
silently fall back to an absent checkout. Update the pin only after qualifying
that exact revision. Repository branch protection determines the merge gate.

The separate [upstream canary](../../.github/workflows/upstream-canary.yaml) runs
weekly on Tuesday at 03:17 UTC and on manual dispatch. It installs upstream
`main` and checks patch drift. A drift failure opens or updates the tracking
issue and fails the job. Installation failure stops before that report step.
This scheduled check does not replace PR checks, and none of these public CPU
jobs runs PPU inference or `tests/e2e`.

### Recipe sanity results

The PR recipe job runs every model policy with a `fast_sanity` tier by default.
PRs on pre-Recipe branches run the legacy Qwen smoke, which is also available
through the manual PPU debug workflow.
The registered accuracy nightly runs `model_sanity` (including GSM8K and
micro-E2E). When recipes are enabled, scheduled Nightly skips the legacy
accuracy cases covered by recipes. Other cases still run on the legacy lane.
With recipes disabled, the full legacy matrix runs; manual legacy case
selection is unchanged.
Each replacement is declared in the model policy's `replaces_legacy_cases`
list and requires a matching legacy accuracy case and a `model_sanity` tier.
See [PPU Recipe CI Matrix](../../tests/ppu/README.md) for the generated table
of active PR `fast_sanity`, Nightly `model_sanity`, and remaining legacy
Nightly configurations.

The runner administrator must
set `PPU_CI_PRIVATE_RESULTS_ROOT` in the runner service environment on every
eligible runner to retain private results after the runtime container exits.
Provision a persistent root outside the job workspace and temporary storage with
mode `0700` or `0755`. The `vllm-sail` subdirectory must be `0700` and writable
by the runner service identity. Pre-create it if the runner cannot write to the
root. Mode `0755` allows others to list the root, not to enter the private
subdirectory. Retained directories and files use modes `0700` and `0600`.
Set an operator-managed cleanup policy based on the baseline comparison window
and available space; remove expired run attempts and abandoned staging directories
only when no writer is active.

The console, job summary, and downloadable summary report the recipe outcome,
retention status (`retained`, `not_configured`, or `failed`), and an allowlisted
recipe failure reason. For example, `server_readiness_timeout` means the server
did not become ready; `gsm8k_gate_failed` means evaluation produced a result
that did not satisfy the golden gate. Unclassified failures report `unknown`.
The nightly job summary and downloadable
artifact also report aggregate GSM8K accuracy, sample counts, golden score,
gate bounds and verdict, micro-E2E verdict, and baseline completeness. They
never include raw results, per-question scores, provenance, or logs. Metrics
that were not produced are reported as unavailable. An unset or empty variable
warns in either tier without changing the recipe verdict; no private baseline
is retained for that run. A configured copy failure still fails either job.
Private retention includes the launcher, container and model-server logs that
were produced, even when a failed recipe did not write a result summary.
`retained` confirms a private copy, not a complete or comparable GSM8K baseline.

### Upstream test reuse

Upstream reuse runs on a single M890 card in the authorized PPU pipeline, not
automatically on PRs or a nightly schedule. It uses the runner-configured Docker
runtime and the validated CI wheels with the SAIL plugin enabled, preserves the
image's torch/torchvision stack and installs only the dedicated
`requirements/upstream-reuse.txt` tooling. `requirements/dev.txt` stays unchanged.

The container reads `HF_HOME=/nas_aisw/datasets/hf_cache` in offline mode. The
cache and `/nas_aisw/datasets/checkpoints/LLM` are mounted read-only at their host
paths, so snapshot symlinks into the checkpoint tree remain valid. Imported
model code uses a writable temporary `HF_MODULES_CACHE`; no checkpoints are
downloaded or modified. Missing model files still fail strict reuse, including
cache entries whose snapshot directory exists but is empty. Diagnostics record
the cache and checkpoint paths.

For a local diagnostic in a prepared CPU environment with the exact upstream
checkout, run the CPU entries with plugin loading disabled. This checks upstream
behavior only and is not PPU qualification:

```bash
VLLM_PLUGINS='' VLLM_TARGET_DEVICE=cpu VLLM_SOURCE_ROOT=/path/to/pinned/vllm \
  python -m pytest tests/upstream_reuse -q -rs \
  -m "upstream_source and not ppu" --require-upstream-source \
  --strict-upstream-reuse --upstream-reuse-target=cpu \
  --upstream-reuse-report-dir=/path/to/reuse-report
```

Strict reuse requires each active entry to collect its declared minimum and
actually pass at least one upstream case. Missing dependencies, import errors,
empty runs and undeclared skips fail CI. Parked entries do not count as coverage.
Reviewed skips identify the node, reason and maximum count; an all-skipped entry
still fails. Reports retain per-entry JUnit XML, logs and execution counts.
Pooling reuse excludes the Snowflake embedding-dimension case and requires the
remaining 40 pooling cases; it does not qualify Snowflake model support.

The PPU pipeline consumes validated wheels for single-card M890 reuse, records
source/wheel/upstream identities and verifies installed-package imports and the
SDK/Torch stack. 810E is explicitly deferred because no CI runner is available.
The authorized `/ci run` dispatcher uses default-branch workflow definitions;
new reusable-job wiring becomes active after merge. Before merge, the existing
`ci-preview` PR path can execute the changed pipeline for validation. Neither a
local CPU run nor a deferred device establishes M890 or 810E correctness.
Sources without `scripts/ci/ppu_upstream_reuse.sh`, such as release branches
cut before this lane, skip M890 reuse and the PPU CI gate accepts that skip.

## 3. Run device correctness tests

In the PPU environment, install `requirements/dev.txt`. Use an editable native
plugin build in the checkout, or ensure the test process imports the installed
wheel as described above. Start with activation and the affected feature:

```bash
pytest tests/e2e/test_native_activation.py -q -rs
pytest tests/e2e/test_deepseek_v4_native.py -q -rs
pytest tests/e2e/test_mxfp4_prepare.py -q -rs
pytest tests/e2e/tier_b/test_norm_quant.py -q -rs
```

The [`tests/e2e`](../../tests/e2e) directory also contains linear-attention,
indexer, cache and model-specific kernel tests. Select the relevant tests for
the change; run the full suite when qualifying an environment:

```bash
pytest tests/e2e -q -rs
```

These tests require a real PPU and the applicable SDK libraries. Verify that the
intended tests actually ran. Include dtype, shape, tolerance and device in a
correctness report, and test both architecture targets when claiming both.

Capability markers restrict tests to one chip: `cap80` runs only on compute
capability 8.0 (Zhenwu 810E) and `cap89` only on 8.9 (Zhenwu M890); other
devices skip them. `perf` marks performance benchmarks and `model_e2e` marks
offline model end-to-end tests. Select or exclude them with `-m`, for example
`pytest tests/e2e -q -rs -m "not perf"`.

### GDN fused MTP decode

Run the native GDN MTP accuracy matrix with a native plugin build:

```bash
python -m pytest tests/e2e/gdn/test_gdn_fused_mtp.py -q -rs \
  -o junit_family=xunit1 --junitxml=gdn-mtp-accuracy.xml
```

The tests call the native fused op directly and force the unfused reference onto
Triton, bypassing PLA. Selected cases also compare with an independent eager
PyTorch reference. On PPU, a missing native dispatch registration fails the
suite rather than skipping it.

Coverage includes all supported head ratios, BF16/FP32 recurrent state and norm
weights, FP16/BF16/FP32 gate bias, SiLU/sigmoid output gates, MTP lengths 1–8,
strided rows and state slots, padding and NULL slots, and repeated accepted-token
rewind with changing request lengths. Every round checks outputs, candidate
state writes, unused state storage and output-buffer guards. The suite also
tests numerical boundaries and the layer's post-convolution entry point.

The native-versus-Triton output keeps upstream's relative-L2 limit of `5e-4`;
pointwise output/state comparisons retain the upstream model-path tolerances
`atol=rtol=3e-2`. Each written slot additionally has a relative-L2 limit of `3e-2`.
Untouched slots and buffer guards must remain bitwise identical. The JUnit file
records relative-L2 and maximum absolute errors per round. These limits are
source-derived, not calibrated from a PPU run; investigate failures before
changing them.

The independent oracle has worked-example CPU tests that run when torch is
available:

```bash
python -m pytest tests/ut/test_gdn_mtp_reference.py -q
```

CPU oracle tests and device-test collection do not establish native accuracy.
Record actual execution counts and skips from the PPU command above.

## 4. Run a model

### Quick serving sanity

The standalone sanity runner launches one installed model, checks readiness
against a unique served-model identity, sends a deterministic arithmetic request,
checks its complete final answer and stops its owned server process group:

```bash
python scripts/eval/ppu_model_sanity.py --list
python scripts/eval/ppu_model_sanity.py \
  --profile deepseek-v4-community-tp8 \
  --checkpoint-root /path/to/datasets \
  --output /absolute/new-sanity-attempt \
  --with-stream
```

Profiles cover community DeepSeek V4 Flash 0731 TP8, its MXFP4/FP8-channel TP8
and DP8/EP8 low-latency variants, GLM 5.2 MXFP4/FP8 TP8, and MiniMax M3
FP8-channel TP8. Checkpoint suffixes and serving arguments are recorded in
[`model_sanity_profiles.json`](../../scripts/eval/model_sanity_profiles.json).
The dataset root defaults to `/ppusw/datasets`; set it to the actual mounted
root. Each attempt requires a new output directory. Run profiles sequentially
on free devices in the selected environment, preserving existing work.

The default sends one ordinary request; `--with-stream` adds a streamed request.
Both must return the final answer `42` and finish naturally. Truncated output,
reasoning without a final answer, a wrong answer, HTTP failure, serving or
shutdown ERROR logs, and failed cleanup fail the attempt. The process exits
nonzero on failure and retains its invocation, installed package metadata,
responses, metrics, serving log and result. Adjust readiness, request and token
budgets explicitly with `--ready-timeout`, `--request-timeout`, and `--max-tokens`.
Each request budget covers connection setup, response headers and the complete
body, including a streamed response that never finishes a line. Expiration
fails the attempt and enters service cleanup; cleanup has its own bounded wait.

This is a short inference check for the selected configuration. It does not
replace the accuracy evaluation below, kernel correctness checks, package-byte
verification or evidence that every configured optimization was active.

### Model verification

Follow the [quickstart](../getting_started/quickstart.md) with a checkpoint
appropriate for the available memory and supported precision. Confirm that:

- The PPU platform and intended compute backend are selected.
- Model loading and warm-up complete.
- The readiness endpoint succeeds and generation returns sensible output.
- Repeated generation and the intended context lengths complete successfully.

For MoE, quantized or multi-device workloads, repeat the check using the intended
expert backend, checkpoint layout and parallelism. A small dense-model smoke
test does not cover those paths.

### Zhenwu 810E serving sanity

The shared [PPU sanity runner](../../scripts/ci/ppu_sanity.py) selects presets
from [ppu_sanity_810e.json](../../scripts/ci/ppu_sanity_810e.json) by default.
Its ten presets cover DeepSeek V4 W4A8/INT8 (TP4, TP8 and DP16/EP16 low latency),
DeepSeek V4.1 Flash INT8 TP8, GLM-5.3 Flash W4A8 TP8, GLM-5.2 INT8 TP16,
Kimi-K2.6 W4A8 TP16,
Qwen3.5 BF16 TP2 and INT8 TP8/EP8, and MiniMax M3 BF16 TP16.
Run it inside a prepared SAIL SDK container with the
installed, compatible vLLM and vLLM SAIL wheels. It does not install packages or
reserve hardware. Select idle cards before running; the full catalog needs 16
visible 810E devices. It changes the HTTP bind address to loopback and assigns a
served model name; model flags remain those in the preset catalog.

```bash
python scripts/ci/ppu_sanity.py --list
python scripts/ci/ppu_sanity.py --dry-run
python scripts/ci/ppu_sanity.py \
  --models qwen3.5-35b_bf16_tp2 \
  --checkpoint-root /ppusw/datasets/checkpoints \
  --results-dir /tmp/810e-sanity-qwen
```

Omit `--models` to run all presets sequentially. To use a checkpoint mounted at a
different path, pass `--checkpoint KEY=/absolute/checkpoint/path`; the resolved
path and config hash are recorded. Missing checkpoints or insufficient devices
fail the selected case. The results directory must be new.
The Qwen 397B INT8 preset uses the checkpoint directory suffix `W8A8-INT8`.
The `glm-5.3-flash_w4a8_tp8` preset uses the GLM-5.3 Flash checkpoint under
`LLM/T-HEAD/v5.3`, with INT4/INT8 MoE and INT8/INT8 dense weights and activations,
TP8 and prefix caching disabled. Its HTTP server binds to loopback, as do the
other sanity presets.
The `deepseek-v4.1-flash_1017_int8_tp8` preset uses
`LLM/T-HEAD/v4.1/DeepSeek-V4.1-Flash-1017-Quant-W-INT8-PerChannel-A-INT8-PerToken-Engram-MXFP8`
under the checkpoint root and serves as `DeepSeek-V4.1-Flash-1017`. It enables
the Rust frontend, Engram CPU offload, a 524288-token context, prefix caching,
chunked prefill and dspark with seven speculative tokens. Other presets retain
the Python frontend. To run this case with the reference command's port:

```bash
python scripts/ci/ppu_sanity.py \
  --models deepseek-v4.1-flash_1017_int8_tp8 \
  --port 8999 \
  --results-dir /tmp/810e-sanity-dsv41
```

Each case checks `/health` and `/v1/models`, sends a chat request and a repeated
request, and requires nonempty content, generated tokens and a normal `stop`
finish reason. Truncation at the output limit fails; the default budget is
4096 output tokens per request (`--max-tokens` overrides it). DP16 uses a
16-request burst after the first request; this does not establish per-rank
routing. Requests use seed 0. Most presets use temperature 0; Qwen3.5-35B uses
temperature 1.0, matching its checkpoint's sampled-generation default and
[official model guidance](https://huggingface.co/Qwen/Qwen3.5-35B-A3B).
Responses with reasoning but empty content fail. The runner records exact commands,
package paths and versions, device capabilities, startup logs, requests,
responses and JSON results. It terminates its own server process group after
each case. Any failed case makes the command exit nonzero. Startup and request
timeouts default to 3600 and 600 seconds and can be overridden.
The startup budget also sets vLLM's engine-ready timeout, so the multi-frontend
DP launch cannot terminate earlier at its shorter internal default.

These presets are test inputs, not a declaration of model support. Passing
establishes inference for the recorded checkpoint and configuration; it does
not establish dataset accuracy, tool-call correctness or benchmark performance.

### Zhenwu M890 serving sanity

Use the same runner with `--device 890` to select
[ppu_sanity_890.json](../../scripts/ci/ppu_sanity_890.json) and require capability 8.9
(`ZW-M890P` identifies Zhenwu M890). The default remains 810E, capability 8.0.
The M890 catalog covers Qwen3.8-Flash-Next MXFP4/FP8 TP4 with MTP,
GLM-5.3-Flash MXFP4/FP8 TP8 with MTP, and DeepSeek V4.1-Flash-1017
MXFP4/FP8 TP8/EP8 with MXFP8 Engram. DeepSeek enables MegaMoE
(`--moe-backend deep_gemm_mega_moe`) and the MXFP4 sparse indexer
(`--attention-config '{"indexer_sparse_logits":true,"indexer_kv_dtype":"mxfp4"}'`).
It uses the Rust frontend, `deepseek_v41` parsers, CPU Engram offload,
and disables prefix caching.

```bash
python scripts/ci/ppu_sanity.py --device 890 --list
python scripts/ci/ppu_sanity.py --device 890 --dry-run
python scripts/ci/ppu_sanity.py --device 890 \
  --results-dir /tmp/890-sanity
```

Use eight visible M890 devices for the complete catalog. The same lifecycle,
generation checks, checkpoint overrides and evidence files described above apply.
These are single-host serving checks; they do not validate multi-host DP16 or
prefill/decode disaggregation.

## 5. Report correctness and performance

Include the exact command, checkpoint identifier and revision, dtype,
quantization, context length, device count and parallelism. For benchmarks, also
include the workload or dataset, input/output lengths, concurrency, warm-up,
throughput and latency measurements.

Useful tools include vLLM's `vllm bench --help` and the repository's
[`benchmarks/kernels`](../../benchmarks/kernels). Compare baseline and candidate
with the same software, hardware and workload. Explain the numerical reference
and tolerance used for kernel comparisons.

Report which checks passed, failed, skipped or were not run. Attach concise
logs and a minimal reproducer through the repository's issue or pull request
template; remove credentials and private model or host details before posting.
