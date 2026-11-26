# SPDX-License-Identifier: Apache-2.0
"""Upstream fixes with explicit installation and dependency-free imports."""


def install() -> None:
    from vllm_sail.patch.bugfix import (
        async_llm_shutdown,
        dp_idle_mtp_dummy_batch,
        glm5_next_int8_mla_dequant,
        numa_bind_default,
        pp_empty_kv_cache_group,
        pynccl_uint16_gather,
        qwen4_exp_ple_quant_scheme,
    )

    async_llm_shutdown.install()
    dp_idle_mtp_dummy_batch.install()
    glm5_next_int8_mla_dequant.install()
    numa_bind_default.install()
    pp_empty_kv_cache_group.install()
    pynccl_uint16_gather.install()
    qwen4_exp_ple_quant_scheme.install()
