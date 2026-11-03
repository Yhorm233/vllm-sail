# HGGC Kernel Development

Native device sources compile with `hgcc` using cmake-hgcc's `HG` language.
Host bindings compile with the C++ compiler. `setup.py` resolves prerequisites
and source selection, then delegates compilation and linking to CMake.

Start with the [installation guide](../getting_started/installation.md) for the
required SDK environment.

## Source ownership

| Input | Generator | Committed build input |
| --- | --- | --- |
| Upstream vLLM sources selected by [`csrc/upstream/manifest.toml`](../../csrc/upstream/manifest.toml) | `tools/port_upstream_kernels.py` | `csrc/upstream/` |
| Plugin HGGC kernels selected by [`csrc/plugin/manifest.toml`](../../csrc/plugin/manifest.toml) | None | Authored `.hg` sources and headers under `csrc/plugin/` |
| Plugin host bindings | None | Authored `.cpp` files under `csrc/plugin/` |

Plugin kernels have one source of truth: edit the HGGC implementation directly
in `csrc/plugin/`. The manifest supplies the `.hg` device sources and `.cpp`
bindings directly to CMake, which compiles them with the appropriate compiler.
Plugin headers resolve within that directory before searching upstream headers.

The upstream generator uses [sailify](https://github.com/t-head/sailify) and
declared transformations to produce HGGC build inputs. Upstream source
filenames are retained except that compiled `.cu` sources become `.hg`, so
CMake's HG language compiles every translation unit directly. Edit upstream
inputs or transformations, then regenerate. Never
hand-edit the generated upstream corpus or `excluded_ops.toml`; its manifest
is the editable file inside `csrc/upstream/`.

Selection is explicit. A newly added upstream operation is not included until
its source, binding and dependencies are reviewed and selected by the manifest.
Excluded operations retain capability diagnostics. Moving an operation into
the compiled set also requires device validation.

## Regenerate upstream kernels

Use a vLLM checkout at the manifest's `upstream_ref` (currently `v0.31.0`) and
make sailify available.

```bash
export PYTHONPATH=/path/to/sailify${PYTHONPATH:+:$PYTHONPATH}

python tools/port_upstream_kernels.py --vllm-src /path/to/vllm
python tools/port_upstream_kernels.py --vllm-src /path/to/vllm --check
```

Review and commit input changes together with the generated diff. Generation
checks require upstream source inputs and sailify, but no device or SAIL
PyTorch. A normal wheel build compiles the authored plugin sources and the
committed upstream corpus without running the generator.

Keep local source transformations explicit and deterministic. When the SDK or
sailify implements the same behavior, remove the corresponding transformation
only after regeneration, compilation and device correctness checks pass.

## Build settings

| Variable | Meaning |
| --- | --- |
| `PPU_SDK` | SDK root, including HGGC headers, runtime libraries and `bin/hgcc` |
| `HGCC` | Explicit compiler path; takes precedence over SDK and `PATH` discovery. A path that is not a file fails the build gate instead of falling back |
| `CMAKE_HGCC_DIR` | Optional path to a cmake-hgcc checkout; without usable local modules or an installed package, CMake fetches the pinned upstream revision |
| `PYTORCH_SAIL_ARCH` | Architecture list; default `ppu_15;ppu_10` |
| `VLLM_SAIL_HG_ARCH` | Plugin architecture list; must select the same targets as `PYTORCH_SAIL_ARCH` when both are nonempty |
| `VLLM_SAIL_HG_STD` | HG language standard; default `20` |
| `VLLM_SAIL_HG_FLAGS` | Extra HG compiler flags |
| `VLLM_SAIL_SKIP_EXT` | Explicitly omit native extensions for Python-only development |
| `VLLM_SAIL_PREBUILT_EXT_DIR` | Package already built native extensions from this directory without running CMake or `hgcc`; each extension file must appear exactly once. Mutually exclusive with `VLLM_SAIL_SKIP_EXT` |
| `MAX_JOBS` | Native build parallelism; an explicit setuptools `-j` takes precedence, and the default is the CPU count |
| `CMAKE_BUILD_TYPE` | CMake build type; default `Release` |

An empty plugin architecture setting uses `PYTORCH_SAIL_ARCH` or the default
list. An empty standard uses `20`, and empty flags add no options.

Native extensions are enabled by default. The build gate checks that PyTorch
is importable and that `hgcc` and CMake are available, and fails with the first
missing prerequisite. Only `VLLM_SAIL_SKIP_EXT=1` produces a Python-only
package. Run the preflight in the installation guide before building a package
for inference.

## Build a wheel

In the prepared SDK environment, from the repository root:

```bash
python -m pip wheel --no-build-isolation --no-deps --wheel-dir dist .
```

The wheel must contain these native libraries:

| Library | Purpose |
| --- | --- |
| `vllm_sail._upstream_C` | Selected upstream activation, normalization, quantization, cache and model operations |
| `vllm_sail._upstream_moe_C` | Selected upstream MoE operations |
| `vllm_sail._C` | Plugin sampler operations |
| `vllm_sail._moe_C` | Plugin expert-parallel scatter operations |

Inspect the artifact, replacing the path with the wheel just built:

```bash
python - /path/to/vllm_sail.whl <<'PY'
import sys
import zipfile

with zipfile.ZipFile(sys.argv[1]) as wheel:
    names = wheel.namelist()
for module in ('_upstream_C', '_upstream_moe_C', '_C', '_moe_C'):
    assert any(name.startswith(f'vllm_sail/{module}.') and name.endswith('.so')
               for name in names), f'missing native extension: {module}'
print('All four native extensions are present')
PY
```

Install with `python -m pip install --no-deps /path/to/vllm_sail.whl` in the
prepared runtime environment. Reinstalling an existing version requires
`--force-reinstall`. Restart workers after replacing native libraries.

## Validate a kernel change

1. Run the affected CPU unit tests; also run the generation check when changing
   upstream kernel inputs.
2. Build for each architecture the change is intended to support.
3. Check extension loading and operation registration in a fresh process.
4. Compare device results with an independent reference over relevant dtypes,
   shapes, padding and boundary cases.
5. Run the affected model path and benchmark when performance is part of the change.

Follow [Verification](../user_guide/verification_guide.md) when recording results.
Host-compiler checks in the CPU suite exercise build configuration; they do not
substitute for an HGGC compilation or a device test.
