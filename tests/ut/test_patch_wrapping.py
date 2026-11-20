# SPDX-License-Identifier: Apache-2.0
"""Explicit instrumentation wrappers preserve the runtime patch chain."""

from __future__ import annotations

import functools
import sys
import types

import pytest


@pytest.mark.parametrize("class_method", [False, True], ids=["module", "class"])
def test_explicit_wrapper_preserves_patch_chain(
    monkeypatch: pytest.MonkeyPatch,
    patch_utils_module: types.ModuleType,
    class_method: bool,
) -> None:
    target = types.ModuleType("_patch_wrapping_test")
    monkeypatch.setitem(sys.modules, target.__name__, target)

    class Target:
        pass

    owner = Target if class_method else target
    target.Target = Target
    attribute = "Target.operation" if class_method else "operation"
    metadata = dict(
        reason="exercise explicit patch composition",
        affected_versions=">=0.33.0,<0.34.0",
        remove_when="upstream provides instrumentation hooks",
    )

    def original(*args):
        return "original"

    owner.operation = original

    @patch_utils_module.patch(target.__name__, attribute, **metadata)
    def first(*args):
        return original(*args) + ":first"

    with pytest.raises(RuntimeError, match="already patched"):
        patch_utils_module.patch(target.__name__, attribute, **metadata)(
            functools.wraps(first)(lambda *args: first(*args))
        )

    @patch_utils_module.patch(
        target.__name__, attribute, allow_wrapping=True, **metadata
    )
    @functools.wraps(first)
    def second(*args):
        return first(*args) + ":second"

    full_target = f"{target.__name__}.{attribute}"
    assert owner.operation() == "original:first:second"
    assert patch_utils_module.original_of(second, full_target) is first
    assert patch_utils_module.original_of(first, full_target) is original
    assert len(patch_utils_module.PATCH_REGISTRY) == 2
    with pytest.raises(RuntimeError, match="already patched"):
        patch_utils_module.patch(
            target.__name__, attribute, allow_wrapping=True, **metadata
        )(second)

    # Opt-in cannot replace an existing patch with an unrelated function
    # or a wrapper of an obsolete layer.
    for unrelated in (lambda: None, functools.wraps(first)(lambda: first())):
        with pytest.raises(RuntimeError, match="already patched"):
            patch_utils_module.patch(
                target.__name__, attribute, allow_wrapping=True, **metadata
            )(unrelated)
