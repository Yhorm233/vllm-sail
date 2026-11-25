# SPDX-License-Identifier: Apache-2.0
"""Upstream fixes with explicit installation and dependency-free imports."""


def install() -> None:
    from vllm_sail.patch.bugfix import (
        async_llm_shutdown,
        dp_idle_mtp_dummy_batch,
        qwen4_exp_ple_quant_scheme,
    )

    async_llm_shutdown.install()
    dp_idle_mtp_dummy_batch.install()
    qwen4_exp_ple_quant_scheme.install()
