# Third-party notices

vLLM SAIL's original code is licensed under the Apache License 2.0 in
`LICENSE`. Adapted third-party code retains its applicable license terms.

## PyTorch stable-ABI error-checking macros

`csrc/torch/csrc/stable/macros.h` adapts `STD_CUDA_CHECK` and
`STD_CUDA_KERNEL_LAUNCH_CHECK` from PyTorch, changing the runtime error type
handling and the launch-error query for the SAIL SDK.

- Source: [PyTorch v2.10.0 macros.h](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/csrc/stable/macros.h).
- Upstream license: BSD 3-Clause; the complete copyright notices, conditions
  and disclaimer are reproduced in `LICENSES/PyTorch.txt` from the
  [PyTorch v2.10.0 license](https://github.com/pytorch/pytorch/blob/v2.10.0/LICENSE).
- Project modifications: Apache License 2.0.

Both license texts and this notice are included in source distributions and
wheel license metadata, including wheels containing native extensions built
with these macros.
