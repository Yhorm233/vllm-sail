# SPDX-License-Identifier: Apache-2.0
"""Upstream fixes with explicit installation and dependency-free imports."""


def install() -> None:
    from vllm_sail.patch.bugfix import (
        async_llm_shutdown,
        glm5_next_int8_mla_dequant,
        mega_gate_padding_mask,
    )

    async_llm_shutdown.install()
    glm5_next_int8_mla_dequant.install()
    mega_gate_padding_mask.install()
