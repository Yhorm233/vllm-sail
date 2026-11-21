# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
CI_DIR = ROOT / "scripts" / "ci"
RECIPES_DIR = ROOT / "tests" / "ppu" / "recipes"
POLICY_DIR = ROOT / "tests" / "ppu" / "policy"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ppu_recipe = _load_module("ppu_recipe", CI_DIR / "ppu_recipe.py")
ppu_recipe_runner = _load_module("ppu_recipe_runner", CI_DIR / "ppu_recipe_runner.py")
ppu_e2e_select = _load_module("ppu_e2e_select", CI_DIR / "ppu_e2e_select.py")


def _sample_vllm_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vllm serve")
    parser.add_argument("model_tag")
    parser.add_argument("--served-model-name")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--tensor-parallel-size", type=int)
    parser.add_argument("--data-parallel-size", type=int)
    parser.add_argument("--enable-expert-parallel", action="store_true")
    parser.add_argument("--all2all-backend")
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--max-model-len")
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--kv-cache-dtype")
    parser.add_argument("--block-size", type=int)
    parser.add_argument("--distributed-timeout-seconds", type=int)
    parser.add_argument("--cpu-distributed-timeout-seconds", type=int)
    parser.add_argument("--enable-auto-tool-choice", action="store_true")
    parser.add_argument("--tool-call-parser")
    parser.add_argument("--tokenizer-mode")
    parser.add_argument("--reasoning-parser")
    parser.add_argument("--reasoning-config")
    parser.add_argument("--speculative-config")
    parser.add_argument("--compilation-config")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--no-enable-prefix-caching", action="store_true")
    parser.add_argument("--load-format")
    parser.add_argument("--hf-overrides")
    return parser


def _yaml_files() -> list[Path]:
    return sorted(
        p
        for p in (ROOT / "tests" / "ppu").rglob("*.yaml")
        if not p.name.startswith(".")
    )


def test_checked_in_recipes_and_policies_validate_cleanly() -> None:
    files = _yaml_files()
    assert files
    assert len(
        ppu_recipe.validate_paths(files, vllm_parser_mode="skip")["validated_files"]
    ) == len(files)
    # The CPU stub covers the pilot flags. The runtime parser validates new recipes.
    report = ppu_recipe.validate_paths(
        [
            RECIPES_DIR / "zai-org" / "GLM-5.3.yaml",
            RECIPES_DIR / "deepseek-ai" / "DeepSeek-V4-Flash.yaml",
            RECIPES_DIR / "moonshotai" / "Kimi-K2.6.yaml",
        ],
        vllm_parser_mode="required",
        parser=_sample_vllm_parser(),
    )
    assert report["parser_checked"] is True
    assert report["checked_flag_sets"] >= 10


def test_glm53_flash_matches_legacy_tp2_and_default_fast_sanity() -> None:
    recipe = RECIPES_DIR / "zai-org" / "GLM-5.3-Flash.yaml"
    policy = POLICY_DIR / "models" / "glm-5.3-flash.yaml"
    legacy = next(
        case
        for case in json.loads((CI_DIR / "ppu_e2e_models.json").read_text())
        if case["key"] == "glm-5.3-flash-mxfp4-tp2"
    )
    resolved = ppu_recipe.resolve_recipe(
        recipe, policy_path=policy, tier="model_sanity"
    )
    assert resolved["ppu_count"] == legacy["ppu_count"] == 2
    assert resolved["hf_model_id"] == legacy["model_id"]
    assert resolved["golden_model_key"] == legacy["model_key"]
    assert {
        flag: value
        for flag, value, _ in ppu_recipe._parse_flag_entries(resolved["flags"])
        if flag != "--enable-prefix-caching"
    } == {
        flag: value
        for flag, value, _ in ppu_recipe._parse_flag_entries(legacy["server_cmd"][9:])
    }
    fast = ppu_e2e_select.build_recipe_matrix("", "fast_sanity")["include"]
    assert any(
        item["key"] == "glm-5.3-flash" and item["ppu_count"] == 2 for item in fast
    )


def test_resolve_glm53_matches_catalog_and_supports_presets() -> None:
    recipe = RECIPES_DIR / "zai-org" / "GLM-5.3.yaml"
    policy = POLICY_DIR / "models" / "glm-5.3.yaml"
    catalog = {
        entry["key"]: entry
        for entry in json.loads((CI_DIR / "ppu_e2e_models.json").read_text())
    }
    cat_flags = catalog["glm-5.3-mxfp4-tp4"]["server_cmd"][9:]

    resolved_ms = ppu_recipe.resolve_recipe(
        recipe, policy_path=policy, tier="model_sanity"
    )
    assert resolved_ms["ppu_count"] == 4
    assert resolved_ms["runtime_env"] == {"VLLM_SERVER_DEV_MODE": "1"}
    assert resolved_ms["flags"][:-1] == [
        "--trust-remote-code",
        "--tensor-parallel-size",
        "4",
        "--max-model-len",
        "16384",
        "--distributed-timeout-seconds",
        "3600",
        "--cpu-distributed-timeout-seconds",
        "3600",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "glm47",
        "--reasoning-parser",
        "glm45",
    ]
    assert resolved_ms["flags"][-1] == "--enable-prefix-caching"
    assert set(cat_flags) <= set(resolved_ms["flags"])

    resolved_fs = ppu_recipe.resolve_recipe(
        recipe, policy_path=policy, tier="fast_sanity"
    )
    assert "--load-format" in resolved_fs["flags"]
    assert "dummy" in resolved_fs["flags"]
    idx_ov = resolved_fs["flags"].index("--hf-overrides")
    assert json.loads(resolved_fs["flags"][idx_ov + 1]) == {"num_hidden_layers": 2}

    resolved_fs_full = ppu_recipe.resolve_recipe(
        recipe, policy_path=policy, tier="fast_sanity", layer_override=None
    )
    assert "--hf-overrides" not in resolved_fs_full["flags"]
    assert resolved_fs_full["config_digest"] != resolved_fs["config_digest"]

    resolved_tp8 = ppu_recipe.resolve_recipe(recipe, preset="tp8", cache="enabled")
    assert resolved_tp8["ppu_count"] == 8
    assert "--tensor-parallel-size" in resolved_tp8["flags"]
    idx_tp = resolved_tp8["flags"].index("--tensor-parallel-size")
    assert resolved_tp8["flags"][idx_tp + 1] == "8"


def test_resolve_dsv4_matches_catalog_and_dpep_preset() -> None:
    recipe = RECIPES_DIR / "deepseek-ai" / "DeepSeek-V4-Flash.yaml"
    policy = POLICY_DIR / "models" / "deepseek-v4-flash.yaml"
    resolved_default = ppu_recipe.resolve_recipe(
        recipe, policy_path=policy, tier="model_sanity"
    )
    assert resolved_default["ppu_count"] == 8
    assert resolved_default["flags"] == [
        "--tensor-parallel-size",
        "8",
        "--gpu-memory-utilization",
        "0.9",
        "--max-model-len",
        "16384",
        "--trust-remote-code",
        "--seed",
        "0",
        "--enable-prefix-caching",
    ]

    resolved_dpep = ppu_recipe.resolve_recipe(
        recipe, preset="dp_ep_low_latency", cache="disabled"
    )
    assert resolved_dpep["ppu_count"] == 8
    assert resolved_dpep["server_env"] == {"VLLM_ENGINE_READY_TIMEOUT_S": "6000"}
    assert "--tensor-parallel-size" not in resolved_dpep["flags"]
    assert "--seed" not in resolved_dpep["flags"]
    assert "--data-parallel-size" in resolved_dpep["flags"]
    assert "--no-enable-prefix-caching" in resolved_dpep["flags"]
    assert "--enable-prefix-caching" not in resolved_dpep["flags"]


def test_resolve_minimax_reproduces_the_retired_catalog_case() -> None:
    recipe = RECIPES_DIR / "MiniMaxAI" / "MiniMax-M3.yaml"
    policy = POLICY_DIR / "models" / "minimax-m3.yaml"
    catalog = {
        entry["key"]: entry
        for entry in json.loads((CI_DIR / "ppu_e2e_models.json").read_text())
    }
    legacy = catalog["minimax-m3-fp8"]
    policy_doc = ppu_recipe._load_yaml(policy)
    assert policy_doc["replaces_legacy_cases"] == ["minimax-m3-fp8"]
    # The shared recipe also carries the MXFP4 variant, so the FP8 policy has to
    # pin the default variant explicitly instead of inheriting whatever is first.
    assert policy_doc["variant"] == "default"
    resolved_ms = ppu_recipe.resolve_recipe(
        recipe, policy_path=policy, tier="model_sanity"
    )
    assert resolved_ms["variant"] == "default"
    assert resolved_ms["ppu_count"] == legacy["ppu_count"] == 8
    assert resolved_ms["hf_model_id"] == legacy["model_id"]
    assert resolved_ms["golden_model_key"] == legacy["model_key"]
    assert resolved_ms["served_name"] == legacy["served_name"]
    assert resolved_ms["runtime_env"] == {"VLLM_SERVER_DEV_MODE": "1"}
    # The recipe follows the P0 single-node serving baseline (FP8 KV cache,
    # block size 128, M3 tool-call/reasoning parsers), so the legacy argv is no
    # longer reproduced flag by flag. Retirement only requires the TP layout,
    # timeout and cache contract below; the accuracy claim rests on the golden
    # band of the Nightly model_sanity run.
    flags = resolved_ms["flags"]
    assert flags[flags.index("--tensor-parallel-size") + 1] == "8"
    assert flags[flags.index("--distributed-timeout-seconds") + 1] == "3600"
    assert flags[flags.index("--cpu-distributed-timeout-seconds") + 1] == "3600"
    assert "--enable-prefix-caching" in flags
    assert "--no-enable-prefix-caching" not in flags

    golden_path = (
        ROOT
        / "tests"
        / "e2e"
        / "models"
        / "_golden"
        / "accuracy_minimax-m3-fp8_890p.json"
    )
    golden = json.loads(golden_path.read_text())
    model_sanity = policy_doc["tiers"]["model_sanity"]["stages"]
    stages = {stage["id"]: stage for stage in model_sanity}
    assert stages["eval-gsm8k"]["generation"] == golden["generation"]

    resolved_fs = ppu_recipe.resolve_recipe(
        recipe, policy_path=policy, tier="fast_sanity"
    )
    assert resolved_fs["ppu_count"] == 8
    assert resolved_fs["dummy_weights"] is True
    # MoE starts at layer 3, so the override has to reach past the dense stack.
    assert resolved_fs["layer_override"] == 4
    assert "--load-format" in resolved_fs["flags"]
    assert "dummy" in resolved_fs["flags"]
    idx_ov = resolved_fs["flags"].index("--hf-overrides")
    assert json.loads(resolved_fs["flags"][idx_ov + 1]) == {"num_hidden_layers": 4}


def test_model_sanity_policy_selects_eight_card_dp_ep_preset() -> None:
    base_slug = "deepseek-v4-flash"
    slug = f"{base_slug}-dp-ep"
    legacy_case = "deepseek-v4-mxfp4-dp-ep"
    matrix = ppu_e2e_select.build_recipe_matrix(slug, "model_sanity")
    (entry,) = matrix["include"]
    recipe = ROOT / entry["recipe_path"]
    resolved = ppu_recipe.resolve_recipe(
        recipe,
        policy_path=ROOT / entry["policy_path"],
        tier="model_sanity",
        preset=entry["preset"],
    )
    base = ppu_recipe.resolve_recipe(
        recipe,
        policy_path=POLICY_DIR / "models" / f"{base_slug}.yaml",
        tier="model_sanity",
    )
    legacy = next(
        case
        for case in json.loads((CI_DIR / "ppu_e2e_models.json").read_text())
        if case["key"] == legacy_case
    )
    assert entry["ppu_count"] == resolved["ppu_count"] == 8
    assert entry["preset"] == resolved["preset"] == "dp_ep_low_latency"
    assert entry["config_digest"] == resolved["config_digest"] != base["config_digest"]
    assert set(legacy["server_env"].items()) <= set(resolved["server_env"].items())

    def flags(args):
        return {name: value for name, value, _ in ppu_recipe._parse_flag_entries(args)}

    effective = flags(resolved["flags"])
    assert "--tensor-parallel-size" not in effective
    for key, value in flags(legacy["server_cmd"][9:]).items():
        if key.endswith("-config"):
            assert json.loads(effective[key]) == json.loads(value)
        else:
            assert effective[key] == value
    fast_matrix = ppu_e2e_select.build_recipe_matrix("", "fast_sanity")
    assert slug not in {item["key"] for item in fast_matrix["include"]}


def test_resolve_and_validate_variant_overrides_and_policy_variant(
    tmp_path: Path,
) -> None:
    recipe = RECIPES_DIR / "MiniMaxAI" / "MiniMax-M3.yaml"
    policy = POLICY_DIR / "models" / "minimax-m3-mxfp4.yaml"

    resolved_default = ppu_recipe.resolve_recipe(recipe, cache="enabled")
    assert resolved_default["variant"] == "default"
    assert (
        resolved_default["hf_model_id"]
        == "T-HEAD/MiniMax-M3-Quant-W-FP8-PerChannel-A-FP8-PerToken"
    )
    assert resolved_default["golden_model_key"] == "minimax-m3-fp8"
    assert (
        resolved_default["served_name"]
        == "MiniMax-M3-Quant-W-FP8-PerChannel-A-FP8-PerToken"
    )

    resolved_mxfp4 = ppu_recipe.resolve_recipe(
        recipe, policy_path=policy, tier="fast_sanity"
    )
    assert resolved_mxfp4["variant"] == "mxfp4_fp8"
    assert resolved_mxfp4["hf_model_id"] == "T-HEAD/MiniMax-M3-MXFP4-FP8"
    assert resolved_mxfp4["golden_model_key"] == "minimax-m3-mxfp4-fp8"
    assert resolved_mxfp4["served_name"] == "minimax-m3-mxfp4-fp8"
    assert resolved_mxfp4["ppu_count"] == 8
    assert resolved_mxfp4["config_digest"] != resolved_default["config_digest"]

    resolved_dpep = ppu_recipe.resolve_recipe(
        recipe, variant="mxfp4_fp8", preset="dp_ep_low_latency", cache="disabled"
    )
    assert resolved_dpep["hf_model_id"] == "T-HEAD/MiniMax-M3-MXFP4-FP8"
    assert resolved_dpep["ppu_count"] == 8
    assert resolved_dpep["server_env"]["HGGC_EMBEDDED_COPY_THRESHOLD"] == "0x10000"
    assert "--data-parallel-size" in resolved_dpep["flags"]
    assert "--tensor-parallel-size" not in resolved_dpep["flags"]

    matrix = ppu_e2e_select.build_recipe_matrix("minimax-m3-mxfp4", tier="fast_sanity")
    assert len(matrix["include"]) == 1
    entry = matrix["include"][0]
    assert entry["variant"] == "mxfp4_fp8"
    assert entry["model_id"] == "T-HEAD/MiniMax-M3-MXFP4-FP8"
    assert entry["golden_model_key"] == "minimax-m3-mxfp4-fp8"

    # Reject unknown or blank policy variant.
    bad_pol = yaml.safe_load(policy.read_text())
    bad_pol["variant"] = "nonexistent"
    bad_pol_path = tmp_path / "bad_var_pol.yaml"
    bad_pol_path.write_text(yaml.safe_dump(bad_pol), encoding="utf-8")
    with pytest.raises(ppu_recipe.ValidationError, match="unknown variant"):
        ppu_recipe.validate_paths([bad_pol_path], vllm_parser_mode="skip")

    bad_pol["variant"] = "   "
    bad_pol_path.write_text(yaml.safe_dump(bad_pol), encoding="utf-8")
    with pytest.raises(ppu_recipe.ValidationError, match="must be non-empty string"):
        ppu_recipe.validate_paths([bad_pol_path], vllm_parser_mode="skip")

    # Reject invalid variant metadata and sensitive leaks.
    base_rec = yaml.safe_load(recipe.read_text())
    for mutator, match in (
        (
            lambda d: d["variants"]["mxfp4_fp8"].update({"hf_model_id": ""}),
            "hf_model_id must be non-empty string",
        ),
        (
            lambda d: d["variants"]["mxfp4_fp8"].update(
                {"hf_model_id": "T-HEAD/" + "a" * 97}
            ),
            "max 96 chars",
        ),
        (
            lambda d: d["variants"]["mxfp4_fp8"].update({"assets": {"model": "only"}}),
            "missing required key",
        ),
        (
            lambda d: d["variants"]["mxfp4_fp8"].update(
                {"served_name": "/data/secret"}
            ),
            "local_path",
        ),
    ):
        bad_rec = copy.deepcopy(base_rec)
        mutator(bad_rec)
        bad_rec_path = tmp_path / "bad_var_rec.yaml"
        bad_rec_path.write_text(yaml.safe_dump(bad_rec), encoding="utf-8")
        with pytest.raises(ppu_recipe.ValidationError, match=match):
            ppu_recipe.validate_paths([bad_rec_path], vllm_parser_mode="skip")


def test_resolve_deepseek_v41_and_v4_pro_recipes() -> None:
    v41 = ppu_recipe.resolve_recipe(
        RECIPES_DIR / "deepseek-ai" / "DeepSeek-V4.1-Flash.yaml",
        policy_path=POLICY_DIR / "models" / "deepseek-v4.1-flash.yaml",
        tier="fast_sanity",
    )
    assert v41["ppu_count"] == 8
    assert v41["layer_override"] is None
    assert "--hf-overrides" not in v41["flags"]
    assert v41["hf_model_id"].endswith("Engram-MXFP8")
    assert "--engram-config" in v41["flags"]
    assert "--mm-encoder-tp-mode" in v41["flags"]
    assert v41["server_env"]["VLLM_ENGINE_READY_TIMEOUT_S"] == "3600"

    v4p = ppu_recipe.resolve_recipe(
        RECIPES_DIR / "deepseek-ai" / "DeepSeek-V4-Pro.yaml",
        policy_path=POLICY_DIR / "models" / "deepseek-v4-pro.yaml",
        tier="fast_sanity",
    )
    assert v4p["ppu_count"] == 8
    assert "--kv-cache-dtype" in v4p["flags"]
    assert "--compilation-config" in v4p["flags"]
    assert "--block-size" in v4p["flags"]
    assert json.loads(v4p["flags"][v4p["flags"].index("--hf-overrides") + 1]) == {
        "num_hidden_layers": 3
    }


def test_resolve_kimi_k26_matches_catalog_and_sampling_contract() -> None:
    recipe = RECIPES_DIR / "moonshotai" / "Kimi-K2.6.yaml"
    policy = POLICY_DIR / "models" / "kimi-k2.6.yaml"
    catalog = {
        entry["key"]: entry
        for entry in json.loads((CI_DIR / "ppu_e2e_models.json").read_text())
    }
    legacy = catalog["kimi-k2.6-mxfp4-tp8"]
    cat_flags = legacy["server_cmd"][9:]

    resolved_ms = ppu_recipe.resolve_recipe(
        recipe, policy_path=policy, tier="model_sanity"
    )
    assert resolved_ms["ppu_count"] == 8
    assert resolved_ms["server_env"] == legacy["server_env"]
    # The executor injects prefix caching from the policy; the recipe itself
    # must not hardcode it, and the resolved command still covers every
    # legacy serve flag.
    assert resolved_ms["flags"][-1] == "--enable-prefix-caching"
    assert set(cat_flags) <= set(resolved_ms["flags"])
    recipe_doc = yaml.safe_load(recipe.read_text())
    assert "--enable-prefix-caching" not in recipe_doc["model"]["base_args"]
    assert "--load-format" not in recipe_doc["model"]["base_args"]
    assert set(recipe_doc["ppu"]["external_env"]) == set(legacy["server_env"])

    resolved_fs = ppu_recipe.resolve_recipe(
        recipe, policy_path=policy, tier="fast_sanity"
    )
    assert "--load-format" in resolved_fs["flags"]
    assert "dummy" in resolved_fs["flags"]
    idx_ov = resolved_fs["flags"].index("--hf-overrides")
    assert json.loads(resolved_fs["flags"][idx_ov + 1]) == {"num_hidden_layers": 2}

    # The eval generation block is the sampling contract the golden gate
    # compares key by key: any drift from the committed 890p golden fails
    # the run before a score is even compared.
    policy_doc = yaml.safe_load(policy.read_text())
    eval_stage = next(
        stage
        for stage in policy_doc["tiers"]["model_sanity"]["stages"]
        if stage["kind"] == "evalscope"
    )
    golden = json.loads(
        (
            ROOT
            / "tests"
            / "e2e"
            / "models"
            / "_golden"
            / "accuracy_kimi_k2_6_mxfp4_890p.json"
        ).read_text()
    )
    assert eval_stage["generation"] == golden["generation"]
    assert resolved_ms["golden_model_key"] == golden["model"]
    assert resolved_ms["hf_model_id"] == golden["checkpoint_tail"]


def test_config_digest_ignores_ephemeral_endpoint_and_tracks_changes() -> None:
    recipe = RECIPES_DIR / "zai-org" / "GLM-5.3.yaml"
    policy = POLICY_DIR / "models" / "glm-5.3.yaml"
    r1 = ppu_recipe.resolve_recipe(
        recipe,
        policy_path=policy,
        tier="model_sanity",
        port=9985,
        served_model_name="a",
    )
    r2 = ppu_recipe.resolve_recipe(
        recipe,
        policy_path=policy,
        tier="model_sanity",
        port=9999,
        served_model_name="b",
    )
    r3 = ppu_recipe.resolve_recipe(
        recipe, policy_path=policy, tier="fast_sanity", port=9985, served_model_name="a"
    )
    assert r1["config_digest"] == r2["config_digest"]
    assert r1["config_digest"] != r3["config_digest"]


def test_validate_rejects_unknown_keys_bad_env_json_and_leaks(tmp_path: Path) -> None:
    base_recipe = yaml.safe_load((RECIPES_DIR / "zai-org" / "GLM-5.3.yaml").read_text())

    def _check_bad_recipe(mutator, match: str) -> None:
        bad = copy.deepcopy(base_recipe)
        mutator(bad)
        path = tmp_path / "bad_recipe.yaml"
        path.write_text(yaml.safe_dump(bad), encoding="utf-8")
        with pytest.raises(ppu_recipe.ValidationError, match=match):
            ppu_recipe.validate_paths([path], vllm_parser_mode="skip")

    _check_bad_recipe(lambda d: d.update({"unknown_top": 1}), "unknown key")
    _check_bad_recipe(
        lambda d: d["ppu"].update({"visibility": "internal_restricted"}),
        "internal_restricted",
    )
    _check_bad_recipe(
        lambda d: d["model"]["base_env"].update({"HF_HUB_OFFLINE": "1"}),
        "runtime-mode env",
    )
    _check_bad_recipe(
        lambda d: d["model"]["base_env"].update({"CUSTOM_VAR": "1"}),
        "allowed prefixes",
    )
    _check_bad_recipe(
        lambda d: d["model"]["base_env"].update({"DG_USE_MOE_DYNAMIC_TILE": "1"}),
        "ppu.external_env",
    )
    _check_bad_recipe(
        lambda d: d["model"]["base_args"].extend(["--hf-overrides", "not-json"]),
        "invalid JSON",
    )
    _check_bad_recipe(
        lambda d: d["model"]["base_args"].extend(["--hf-overrides", '{"a": "$(id)"}']),
        "shell substitution",
    )
    _check_bad_recipe(
        lambda d: d.update({"notes": "/nas_aisw/datasets/secret"}),
        "local_path",
    )
    _check_bad_recipe(
        lambda d: d.update({"notes": "host.eng.t-head.cn"}),
        "internal_domain",
    )
    _check_bad_recipe(
        lambda d: d.update({"notes": "10.1.2.3"}),
        "ipv4_literal",
    )
    prohibited = "".join(chr(c) for c in (102, 108, 121, 116, 105, 103, 101, 114))
    _check_bad_recipe(
        lambda d: d.update({"notes": prohibited}),
        "prohibited_token",
    )


def test_validate_policy_guardrails_and_redaction_gate(tmp_path: Path) -> None:
    base_pol = yaml.safe_load((POLICY_DIR / "models" / "glm-5.3.yaml").read_text())

    def _check_bad_pol(mutator, match: str) -> None:
        bad = copy.deepcopy(base_pol)
        mutator(bad)
        path = tmp_path / "bad_pol.yaml"
        path.write_text(yaml.safe_dump(bad), encoding="utf-8")
        with pytest.raises(ppu_recipe.ValidationError, match=match):
            ppu_recipe.validate_paths([path], vllm_parser_mode="skip")

    _check_bad_pol(
        lambda d: d["tiers"]["model_sanity"]["stages"][1].update({"cache": "disabled"}),
        "requires cache='enabled'",
    )
    _check_bad_pol(
        lambda d: d["tiers"]["model_sanity"]["stages"][2].__delitem__("cache_mode"),
        "requires cache_mode",
    )
    _check_bad_pol(
        lambda d: d["tiers"]["model_sanity"]["stages"][2]["input"].update(
            {"range_ratio": 1.0}
        ),
        "range_ratio must be in",
    )
    _check_bad_pol(
        lambda d: d["tiers"]["model_sanity"]["stages"][2].update({"kind": "evalscope"}),
        "duplicate stage kind",
    )
    _check_bad_pol(
        lambda d: d["tiers"]["model_sanity"]["stages"].pop(),
        "missing required stage kind",
    )
    _check_bad_pol(
        lambda d: d["tiers"]["model_sanity"]["stages"][1].update(
            {"dataset": "missing@2026-11"}
        ),
        "referenced dataset definition not found",
    )

    with pytest.raises(ppu_recipe.ValidationError, match="redaction gate blocked"):
        ppu_recipe.redact_and_verify_public_payload({"err": "token ghp_1234567890ab"})
    with pytest.raises(ppu_recipe.ValidationError, match="redaction gate blocked"):
        ppu_recipe.redact_and_verify_public_payload({"path": "/nas_aisw/datasets/foo"})
    ppu_recipe.redact_and_verify_public_payload(
        {"url": "http://127.0.0.1:9985"}, allow_loopback=True
    )


def test_validate_flags_with_parser_rejects_bad_flag_and_type() -> None:
    parser = _sample_vllm_parser()
    with pytest.raises(ppu_recipe.ValidationError, match="unknown vLLM serve flag"):
        ppu_recipe.validate_flags_with_parser(
            parser, ["--nonexistent-flag"], context="test"
        )
    with pytest.raises(ppu_recipe.ValidationError, match="parser rejected args"):
        ppu_recipe.validate_flags_with_parser(
            parser, ["--tensor-parallel-size", "not-an-int"], context="test"
        )


def test_recipe_selector_matrix_and_cli() -> None:
    matrix = ppu_e2e_select.build_recipe_matrix(
        "glm-5.3,deepseek-v4-flash", tier="fast_sanity"
    )
    assert [item["key"] for item in matrix["include"]] == [
        "glm-5.3",
        "deepseek-v4-flash",
    ]
    assert [item["ppu_count"] for item in matrix["include"]] == [4, 8]
    with pytest.raises(ValueError, match="unknown recipe policy key"):
        ppu_e2e_select.build_recipe_matrix("unknown-model")


@pytest.mark.parametrize("claims", [[], "legacy", [""], [" "], [1], ["same", "same"]])
def test_replacement_policy_requires_unique_nonempty_case_ids(claims) -> None:
    policy = yaml.safe_load((POLICY_DIR / "models" / "glm-5.3.yaml").read_text())
    policy["replaces_legacy_cases"] = claims
    with pytest.raises(ValueError, match="nonempty list of unique case IDs"):
        ppu_recipe.validate_policy_structure(policy, context="test policy")


def test_replacement_policy_requires_model_sanity() -> None:
    policy = yaml.safe_load((POLICY_DIR / "models" / "glm-5.3.yaml").read_text())
    del policy["tiers"]["model_sanity"]
    with pytest.raises(ValueError, match="requires tiers.model_sanity"):
        ppu_recipe.validate_policy_structure(policy, context="test policy")


def test_recipe_defaults_include_every_policy_defining_the_requested_tier(
    tmp_path, monkeypatch
) -> None:
    shutil.copytree(ROOT / "tests/ppu", tmp_path / "tests/ppu")
    monkeypatch.setattr(ppu_e2e_select, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(ppu_recipe, "REPO_ROOT", tmp_path)
    policy_dir = tmp_path / "tests/ppu/policy/models"
    pilot = yaml.safe_load((policy_dir / "glm-5.3.yaml").read_text())
    for tier in ("fast_sanity", "model_sanity"):
        single_tier = {**pilot, "tiers": {tier: pilot["tiers"][tier]}}
        single_tier.pop("replaces_legacy_cases")
        (policy_dir / f"{tier}-only.yaml").write_text(yaml.safe_dump(single_tier))
    for tier, other in (
        ("fast_sanity", "model_sanity"),
        ("model_sanity", "fast_sanity"),
    ):
        expected = [
            path.stem
            for path in sorted(policy_dir.glob("*.yaml"))
            if tier in yaml.safe_load(path.read_text()).get("tiers", {})
        ]
        assert tier + "-only" in expected
        assert other + "-only" not in expected
        assert [
            entry["key"]
            for entry in ppu_e2e_select.build_recipe_matrix("", tier)["include"]
        ] == expected
        with pytest.raises(ValueError, match=f"{other}-only.*has no tier {tier!r}"):
            ppu_e2e_select.build_recipe_matrix(f"{other}-only", tier)


def test_qwen_fast_sanity_is_selected_at_tp2() -> None:
    matrix = ppu_e2e_select.build_recipe_matrix("", tier="fast_sanity")
    qwen = next(item for item in matrix["include"] if item["key"] == "qwen3.8-27b")
    assert qwen["ppu_count"] == 2
    assert qwen["model_id"] == "Qwen/Qwen3.8-27B"
    nightly = ppu_e2e_select.build_recipe_matrix("", tier="model_sanity")
    assert "qwen3.8-27b" not in {item["key"] for item in nightly["include"]}


@pytest.mark.parametrize("value", [False, {"max_tokens": True}, {"max_tokens": 129}])
def test_fast_sanity_stream_policy_checks_token_budget(value) -> None:
    policy = yaml.safe_load((POLICY_DIR / "models" / "qwen3.8-27b.yaml").read_text())
    policy["tiers"]["fast_sanity"]["probe"]["stream"] = value
    with pytest.raises(ppu_recipe.ValidationError, match=r"fast_sanity\.probe\.stream"):
        ppu_recipe.validate_policy_structure(policy, context="test policy")


def _stream_event(delta, finish=None):
    chunk = {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    return f"data: {json.dumps(chunk)}\r\n\r\n".encode()


_CONTENT = _stream_event({"role": "assistant", "content": "42"})
_FINISH = _stream_event({}, finish="stop")
_DONE = b"data: [DONE]\r\n\r\n"


@pytest.fixture
def probe_server():
    calls = []

    def start(stream_body, *, status=200, keep_sending=False):
        stopped = threading.Event()

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                calls.append((self.path, body))
                if not body.get("stream"):
                    completed = {
                        "choices": [
                            {"message": {"content": "42"}, "finish_reason": "stop"}
                        ],
                        "usage": {"completion_tokens": 1},
                    }
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(json.dumps(completed).encode())
                    return
                self.send_response(status)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.end_headers()
                try:
                    self.wfile.write(stream_body)
                    if keep_sending:
                        while not stopped.wait(0.01):
                            self.wfile.write(b": keepalive\n\n")
                except OSError:
                    pass

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        thread.start()
        return server, thread, stopped

    servers = []

    def listen(*args, **kwargs):
        server, thread, stopped = start(*args, **kwargs)
        servers.append((server, thread, stopped))
        return server.server_port, calls

    yield listen
    for server, thread, stopped in servers:
        stopped.set()
        server.shutdown()
        thread.join()
        server.server_close()


@pytest.mark.parametrize(
    ("body", "status", "reason"),
    [
        (_CONTENT + _FINISH + _DONE, 200, None),
        (_CONTENT + _FINISH, 200, "missing_done"),
        (_CONTENT, 200, "truncated_stream"),
        (_CONTENT + _DONE, 200, "truncated_stream"),
        (
            _stream_event({"reasoning": "42", "content": "  "}) + _FINISH + _DONE,
            200,
            "empty_stream_content",
        ),
        (b"data: {synthetic-private-response}\n\n", 200, "invalid_stream_chunk"),
        (b": " + b"x" * 65536, 200, "stream_response_too_large"),
        (b"synthetic-private-response", 500, "stream_http_error"),
    ],
)
def test_stream_chat_probe_validates_loopback_http(probe_server, body, status, reason):
    port, calls = probe_server(body, status=status)
    kwargs = {
        "port": port,
        "served_name": "dummy",
        "probe": {
            "prompt": "Say 42",
            "stream": {"max_tokens": 64, "enable_thinking": False},
        },
    }
    if reason is None:
        result = ppu_recipe_runner._send_stream_chat_probe(**kwargs)
        assert result["status"] == "pass"
        assert result["content_chars"] == 2
    else:
        with pytest.raises(ppu_recipe_runner._StreamProbeError, match=f"^{reason}$"):
            ppu_recipe_runner._send_stream_chat_probe(**kwargs)
    assert calls == [
        (
            "/v1/chat/completions",
            {
                "model": "dummy",
                "messages": [{"role": "user", "content": "Say 42"}],
                "stream": True,
                "temperature": 0.0,
                "max_tokens": 64,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
    ]


def test_stream_probe_deadline_includes_keepalive_traffic(probe_server):
    port, _ = probe_server(_CONTENT, keep_sending=True)
    started = time.monotonic()
    with pytest.raises(ppu_recipe_runner._StreamProbeError, match="^stream_timeout$"):
        ppu_recipe_runner._send_stream_chat_probe(
            port=port,
            served_name="dummy",
            probe={"prompt": "Say 42", "stream": {"max_tokens": 16}},
            timeout=0.15,
        )
    assert time.monotonic() - started < 2


def test_missing_done_fails_fast_sanity_without_restarting_server(
    tmp_path, monkeypatch, probe_server
):
    port, calls = probe_server(
        _stream_event({"content": "synthetic-private-value"}) + _FINISH
    )
    launches = []
    monkeypatch.setattr(
        ppu_recipe_runner, "_launch_resolved_server", lambda *a, **k: launches.append(a)
    )
    monkeypatch.setattr(ppu_recipe_runner, "_wait_server_ready", lambda *a, **k: 10.0)
    phases = tmp_path / "phase.txt"
    monkeypatch.setenv("PPU_RECIPE_PHASE_FILE", str(phases))

    exit_code = ppu_recipe_runner.main(
        [
            "--recipe",
            str(RECIPES_DIR / "Qwen" / "Qwen3.8-27B.yaml"),
            "--policy",
            str(POLICY_DIR / "models" / "qwen3.8-27b.yaml"),
            "--tier",
            "fast_sanity",
            "--results-dir",
            str(tmp_path),
            "--port",
            str(port),
            "--vllm-parser",
            "skip",
        ]
    )
    assert exit_code == 1
    assert len(launches) == 1
    assert [body.get("stream", False) for _, body in calls] == [False, True]
    assert calls[0][1]["model"] == calls[1][1]["model"]
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["status"] == "fail"
    assert summary["attempts"][0]["stream_probe"]["error"] == "missing_done"
    assert summary["timings"]["cold_start_s"] == 10.0
    assert "stream_s" in summary["timings"] and "nonstream_s" in summary["timings"]
    assert phases.read_text().strip() == "missing_done"
    assert "synthetic-private-value" not in json.dumps(summary)
    ppu_recipe.redact_and_verify_public_payload(summary, allow_loopback=True)


def test_runner_fast_sanity_falls_back_from_2l_to_full_layers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int | None] = []

    def fake_attempt(*, layer_override, label, **kwargs):
        calls.append(layer_override)
        if layer_override == 2:
            return {
                "mode": label,
                "layer_override": 2,
                "status": "fail",
                "cold_start_s": 12.5,
                "eval_s": 0.0,
                "config_digest": "a" * 64,
                "ppu_count": 4,
                "error": ppu_recipe_runner._sanitize_error_text(
                    "2-layer failed at /nas_aisw/datasets/x"
                ),
            }
        return {
            "mode": label,
            "layer_override": None,
            "status": "pass",
            "cold_start_s": 45.0,
            "eval_s": 1.2,
            "config_digest": "b" * 64,
            "ppu_count": 4,
            "probe": {"status": "pass", "elapsed_s": 1.2},
        }

    monkeypatch.setattr(ppu_recipe_runner, "_attempt_fast_sanity_once", fake_attempt)
    res = ppu_recipe_runner.run_fast_sanity(
        recipe_path=RECIPES_DIR / "zai-org" / "GLM-5.3.yaml",
        policy_path=POLICY_DIR / "models" / "glm-5.3.yaml",
        results_dir=tmp_path / "out",
        parser_mode="skip",
        queue_s=3.5,
    )
    assert calls == [2, None]
    assert res["status"] == "pass"
    assert res["timings"] == {
        "queue_s": 3.5,
        "cold_start_s": 45.0,
        "eval_s": 1.2,
        "bench_s": 0.0,
        "nonstream_s": 1.2,
    }
    assert (tmp_path / "out" / "summary.json").is_file()


def test_runner_fast_sanity_uses_policy_variant_and_null_layer_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[str | None, int | None]] = []

    def fake_attempt(*, variant, layer_override, label, **kwargs):
        seen.append((variant, layer_override))
        return {
            "mode": label,
            "layer_override": layer_override,
            "status": "pass",
            "cold_start_s": 30.0,
            "eval_s": 1.0,
            "config_digest": "c" * 64,
            "ppu_count": 8,
            "probe": {"status": "pass", "elapsed_s": 1.0},
        }

    pol_doc = yaml.safe_load(
        (POLICY_DIR / "models" / "minimax-m3-mxfp4.yaml").read_text()
    )
    pol_doc["tiers"]["fast_sanity"]["layer_override"] = None
    pol_path = tmp_path / "minimax-full.yaml"
    pol_path.write_text(yaml.safe_dump(pol_doc), encoding="utf-8")

    monkeypatch.setattr(ppu_recipe_runner, "_attempt_fast_sanity_once", fake_attempt)
    res = ppu_recipe_runner.run_fast_sanity(
        recipe_path=RECIPES_DIR / "MiniMaxAI" / "MiniMax-M3.yaml",
        policy_path=pol_path,
        results_dir=tmp_path / "out",
        parser_mode="skip",
    )
    assert seen == [("mxfp4_fp8", None)]
    assert res["status"] == "pass"
    assert res["variant"] == "mxfp4_fp8"


@pytest.mark.parametrize(
    ("score", "failed_file", "counts", "failed_phase"),
    [
        (0.9689, None, {}, None),
        (0.9, None, {}, None),
        (0.9689, "bench_c16.json", {"completed": 0, "failed": 32}, "measure"),
        (0.9689, "bench_c16.json", {"completed": 31, "failed": 1}, "measure"),
        (0.9689, "bench_c16.json", {"completed": 32}, "measure"),
        (0.9689, "warmup_c16.json", {"completed": 7, "failed": 1}, "warmup"),
    ],
)
def test_runner_model_sanity_end_to_end_with_checked_in_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    score: float,
    failed_file: str | None,
    counts: dict[str, int],
    failed_phase: str | None,
) -> None:
    """Exercise run_model_sanity against checked-in GLM-5.3 recipe/policy and golden."""
    fake_proc = type("FakeProc", (), {"pid": 999999})()
    monkeypatch.setenv("GSM8K_DATASET_PATH", str(tmp_path / "gsm8k_unused"))
    monkeypatch.setattr(
        ppu_recipe_runner, "_launch_resolved_server", lambda *a, **kw: fake_proc
    )
    monkeypatch.setattr(ppu_recipe_runner, "_stop_server", lambda *a, **kw: None)
    monkeypatch.setattr(ppu_recipe_runner, "_wait_server_ready", lambda *a, **kw: 12.34)
    monkeypatch.setattr(
        ppu_recipe_runner,
        "_send_chat_probe",
        lambda **kw: {
            "status": "pass",
            "elapsed_s": 0.4,
            "finish_reason": "stop",
            "content_chars": 2,
            "completion_tokens": 1,
        },
    )
    monkeypatch.setattr(
        ppu_recipe_runner,
        "_drain_and_reset_cache",
        lambda port: {"cache_reset_evidence": "endpoint_ack_only", "http_status": 200},
    )

    class _FakeEvalModule:
        @staticmethod
        def _ensure_launched_server_alive(*a, **kw):
            return None

        @staticmethod
        def run_evalscope(
            served_name,
            port,
            gsm8k_path,
            *,
            generation_config,
            limit,
            work_dir,
            seed,
            eval_batch_size,
            dataset_options,
        ):
            assert (seed, eval_batch_size) == (42, 128)
            assert dataset_options == {
                "subset_list": ["main"],
                "few_shot_num": 4,
                "few_shot_random": False,
                "filters": {"remove_until": "</think>"},
            }
            _write_review(Path(work_dir), _review_fixture()[:2], model=served_name)
            return {"score": score, "num": 1319}

        @staticmethod
        def extract_score(result):
            return float(result["score"]), int(result["num"])

    real_import = importlib.import_module

    def fake_import(name: str):
        if name == "scripts.eval.ppu_gsm8k_eval":
            return _FakeEvalModule
        return real_import(name)

    monkeypatch.setattr(ppu_recipe_runner.importlib, "import_module", fake_import)

    bench_calls: list[dict[str, object]] = []

    def fake_bench(**kw):
        bench_calls.append(kw)
        return {
            **(
                counts
                if kw["result_filename"] == failed_file
                else {"completed": kw["num_prompts"], "failed": 0}
            ),
            "request_throughput": 12.5,
            "output_throughput": 420.0,
            "mean_ttft_ms": 80.0,
            "mean_tpot_ms": 15.0,
        }

    monkeypatch.setattr(ppu_recipe_runner, "_run_vllm_bench", fake_bench)

    out_dir = tmp_path / "ms_out"
    res = ppu_recipe_runner.run_model_sanity(
        recipe_path=RECIPES_DIR / "zai-org" / "GLM-5.3.yaml",
        policy_path=POLICY_DIR / "models" / "glm-5.3.yaml",
        results_dir=out_dir,
        parser_mode="skip",
        queue_s=1.5,
        post_ready_sleep=0,
    )
    acc_ok = score == 0.9689
    assert res["status"] == ("pass" if acc_ok and not failed_phase else "fail")
    assert res["accuracy"]["score"] == score
    assert res["accuracy"]["num"] == 1319
    assert res["accuracy"]["expected_num"] == 1319
    assert res["accuracy"]["minimum_score"] == pytest.approx(0.9489)
    assert res["accuracy"]["maximum_score"] == pytest.approx(0.9889)
    assert res["accuracy"]["within_band"] is acc_ok
    assert res["micro_e2e_passed"] is (failed_phase is None)
    assert res["accuracy"]["per_question"]["count"] == 2
    assert res["accuracy"]["per_question"]["status"] == "baseline_incomplete"
    assert set(res["bench"].keys()) == {"c1", "c16"}
    assert res["bench"]["c16"]["failed_phase"] == failed_phase
    expected_prompts = [2, 8, 8] if failed_phase == "warmup" else [2, 8, 8, 32]
    assert [c["num_prompts"] for c in bench_calls] == expected_prompts
    assert all(c["range_ratio"] == 0.0 for c in bench_calls)
    assert res["timings"]["queue_s"] == 1.5
    assert res["timings"]["cold_start_s"] == 12.34
    assert (out_dir / "summary.json").is_file()
    assert (out_dir / "gsm8k_per_question.jsonl").is_file()


def test_stop_server_kills_outliving_process_group_worker(tmp_path: Path) -> None:
    """Verify _stop_server SIGKILLs workers in the pgid even after the leader exits."""
    import os
    import time

    pid_file = tmp_path / "worker.pid"
    leader_code = (
        "import os, pathlib, signal, subprocess, sys, time; "
        "worker_code = 'import os, pathlib, signal, sys, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); "
        "time.sleep(60)'; "
        "w = subprocess.Popen([sys.executable, '-c', worker_code, sys.argv[1]]); "
        "signal.signal(signal.SIGTERM, lambda *a: sys.exit(0)); "
        "time.sleep(60)"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", leader_code, str(pid_file)],
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 5.0
        raw_pid = ""
        while time.monotonic() < deadline:
            if pid_file.exists():
                raw_pid = pid_file.read_text(encoding="utf-8").strip()
                if raw_pid:
                    break
            time.sleep(0.05)
        assert raw_pid, "detached worker did not start"
        worker_pid = int(raw_pid)

        ppu_recipe_runner._stop_server(proc, grace_s=0.3)
        assert proc.poll() is not None
        kill_deadline = time.monotonic() + 2.0
        while time.monotonic() < kill_deadline:
            try:
                os.kill(worker_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            pytest.fail(f"worker {worker_pid} survived _stop_server")
    finally:
        if proc.poll() is None:
            proc.kill()


def test_recipe_workflows_follow_security_and_shell_contracts() -> None:
    for wf_name in (
        "_ppu-recipe-fast-sanity.yaml",
        "ppu-recipe-fast-sanity.yaml",
        "_ppu-recipe-model-sanity.yaml",
        "ppu-accuracy-nightly.yaml",
    ):
        wf = yaml.safe_load((ROOT / ".github" / "workflows" / wf_name).read_text())
        assert wf["cache-mode"] == "none"
        assert wf["permissions"] == {"contents": "read", "actions": "read"}
    fast_wf = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "_ppu-recipe-fast-sanity.yaml").read_text()
    )
    fast_recipe = next(
        step
        for step in fast_wf["jobs"]["sanity"]["steps"]
        if step.get("id") == "recipe"
    )
    assert fast_recipe["env"]["TIER"] == "fast_sanity"
    model_wf = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "_ppu-recipe-model-sanity.yaml").read_text()
    )
    model_recipe = next(
        step
        for step in model_wf["jobs"]["sanity"]["steps"]
        if step.get("id") == "recipe"
    )
    assert model_recipe["env"]["TIER"] == "model_sanity"
    assert model_recipe["env"]["PRESET"] == "${{ matrix.model.preset || '' }}"
    pr_wf = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "ppu-build-and-test.yaml").read_text()
    )
    assert (
        pr_wf["jobs"]["recipe-sanity"]["uses"]
        == "./.github/workflows/_ppu-recipe-fast-sanity.yaml"
    )
    nightly_wf = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "ppu-accuracy-nightly.yaml").read_text()
    )
    assert (
        nightly_wf["jobs"]["recipe-model-sanity"]["uses"]
        == "./.github/workflows/_ppu-recipe-model-sanity.yaml"
    )
    script = CI_DIR / "ppu_e2e_recipe_sanity.sh"
    res = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


@pytest.mark.parametrize(
    ("tier", "expected_install"),
    [
        ("model_sanity", True),
        ("fast_sanity", False),
    ],
)
def test_recipe_evalscope_install_preserves_runtime_numpy(
    tmp_path: Path, tier: str, expected_install: bool
) -> None:
    script = (CI_DIR / "ppu_e2e_recipe_sanity.sh").read_text()
    constraints = tmp_path / "constraints.txt"
    constraints.write_text("torch===2.13.0\n")
    python = tmp_path / "python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        "from pathlib import Path\n"
        "if sys.argv[1] == '-c' and 'version(\"numpy\")' in sys.argv[2]:\n"
        "    print('1.26.4')\n"
        "elif sys.argv[1] == '-c':\n"
        "    assert sys.argv[2] == 'from evalscope import TaskConfig, run_task'\n"
        "    print('import evalscope')\n"
        "else:\n"
        "    assert sys.argv[1:4] == ['-m', 'pip', 'install']\n"
        "    print(sys.argv[-1], Path(os.environ['PIP_CONSTRAINT']).read_text().split())\n"
    )
    python.chmod(0o755)
    start = script.index('--constraints "${PIP_CONSTRAINT}"')
    start = script.index("\n", start) + 1
    end = script.index('python "${ROOT_DIR}/scripts/ci/ppu_image_stack.py" verify')
    result = subprocess.run(
        ["bash", "-euc", script[start:end]],
        env={
            "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
            "PIP_CONSTRAINT": str(constraints),
            "TIER": tier,
            "ROOT_DIR": str(ROOT),
            "vllm_wheel": "vllm.whl",
            "sail_wheel": "sail.whl",
        },
        check=True,
        capture_output=True,
        text=True,
    )
    expected = ["torch===2.13.0"]
    if expected_install:
        expected.extend(["numpy===1.26.4", "pyarrow<26"])
    packages = ["vllm.whl", "sail.whl", str(ROOT / "requirements/ci-runtime.txt")]
    if expected_install:
        packages.append("evalscope==1.14.1")
    expected_output = [f"{package} {expected}" for package in packages]
    if expected_install:
        expected_output.append("import evalscope")
    assert result.stdout.splitlines() == expected_output


def test_prediction_only_file_cannot_supply_baseline(tmp_path: Path) -> None:
    (tmp_path / "predictions.jsonl").write_text(
        json.dumps({"id": "q0", "prediction": "text must not be published"}) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "per_question.jsonl"
    result = ppu_recipe_runner._extract_per_question_evalscope(
        tmp_path, output, expected_count=1, aggregate_score=1.0, aggregate_num=1
    )
    assert result["count"] == 0
    assert result["status"] == "baseline_incomplete"
    assert output.read_text(encoding="utf-8") == ""


def _review_fixture(run: str = "a") -> list[dict]:
    path = (
        ROOT
        / "tests"
        / "ut"
        / "fixtures"
        / (f"evalscope_1_14_1_gsm8k_main_reviews_{run}.jsonl")
    )
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_review(root: Path, rows: list[dict], model: str = "fixture") -> Path:
    path = root / "20261118_102334" / "reviews" / model / "gsm8k_main.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _extract_review(root: Path) -> tuple[dict, list[dict]]:
    report = json.loads(
        (ROOT / "tests/ut/fixtures/evalscope_1_14_1_gsm8k_report_v2.json").read_text()
    )
    primary = next(
        metric
        for metric in report["metrics"]
        if metric["identity"] == report["primary_metric_identity"]
    )
    output = root / "per_question.jsonl"
    result = ppu_recipe_runner._extract_per_question_evalscope(
        root,
        output,
        expected_count=12,
        aggregate_score=primary["score"],
        aggregate_num=primary["num"],
    )
    records = [json.loads(line) for line in output.read_text().splitlines()]
    return result, records


@pytest.mark.parametrize("generation", ["plain answer", "answer\u2028step\u0085result"])
def test_extract_review_uses_only_jsonl_newlines(
    tmp_path: Path, generation: str
) -> None:
    rows = _review_fixture()
    rows[0]["prediction"] = {"content": generation}
    source = _write_review(tmp_path, rows)
    assert generation in source.read_text(encoding="utf-8")

    result, records = _extract_review(tmp_path)

    assert result["status"] == "baseline_established"
    assert result["count"] == result["unique_ids"] == len(rows)
    assert result["unscored"] == 0
    assert records[0] == {"question_id": "6", "correct": True, "score": 1.0}


@pytest.mark.parametrize("prediction_first", [False, True])
def test_extract_real_review_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prediction_first: bool
) -> None:
    monkeypatch.setattr(
        ppu_recipe_runner.importlib.metadata, "version", lambda name: "1.14.1"
    )
    source = _write_review(tmp_path, _review_fixture())
    if prediction_first:
        (tmp_path / "000_predictions.jsonl").write_text(
            json.dumps({"id": "wrong", "prediction": "not a scored review"}) + "\n"
        )
        wrong_subset = source.with_name("gsm8k_other.jsonl")
        wrong_subset.write_text(source.read_text())
        wrong_dataset = source.with_name("aime24_main.jsonl")
        wrong_dataset.write_text(source.read_text())
    result, records = _extract_review(tmp_path)
    assert result["status"] == "baseline_established"
    assert result["reasons"] == []
    assert result["count"] == result["unique_ids"] == 12
    assert result["unscored"] == 0
    assert result["correct_sum"] == 6
    assert result["source_file"] == str(source.relative_to(tmp_path))
    assert result["evalscope_version"] == "1.14.1"
    assert result["dataset"] == "gsm8k"
    assert result["subset"] == "main"
    assert records[0] == {"question_id": "6", "correct": True, "score": 1.0}
    assert all(set(item) == {"question_id", "correct", "score"} for item in records)


@pytest.mark.parametrize("shape", ["accuracy", "acc", "numeric"])
def test_extract_binary_accuracy_shapes(tmp_path: Path, shape: str) -> None:
    rows = _review_fixture()
    for row in rows:
        score = row["sample_score"]["score"]
        value = score["value"]["accuracy"]
        score["value"] = value if shape == "numeric" else {shape: value}
    _write_review(tmp_path, rows)
    result, records = _extract_review(tmp_path)
    assert result["status"] == "baseline_established"
    assert sum(item["correct"] for item in records) == 6


@pytest.mark.parametrize(
    "case", ["prediction", "null", "fractional", "wrong_metric", "error", "malformed"]
)
def test_extract_unscored_rows_are_excluded(tmp_path: Path, case: str) -> None:
    rows = _review_fixture()
    if case == "prediction":
        rows[0].pop("sample_score")
        rows[0]["prediction"] = "not scored"
    elif case == "null":
        rows[0]["sample_score"]["score"]["value"] = None
    elif case == "fractional":
        rows[0]["sample_score"]["score"]["value"] = {"accuracy": 0.75}
    elif case == "wrong_metric":
        rows[0]["sample_score"]["score"]["value"] = {"latency": 1.0}
    elif case == "error":
        rows[0]["sample_score"]["score"]["status"] = "error"
    source = _write_review(tmp_path, rows)
    if case == "malformed":
        source.write_text(source.read_text() + "{broken json\n")
    result, records = _extract_review(tmp_path)
    assert result["status"] == "baseline_incomplete"
    assert result["unscored"] == 1
    assert any("存在未评分题" in reason for reason in result["reasons"])
    assert len(records) == (12 if case == "malformed" else 11)
    assert all(item["correct"] is not None for item in records)


@pytest.mark.parametrize("candidates", [0, 2])
def test_review_source_must_be_unique(tmp_path: Path, candidates: int) -> None:
    for i in range(candidates):
        _write_review(tmp_path, _review_fixture(), model=f"fixture-{i}")
    result, records = _extract_review(tmp_path)
    assert result["status"] == "baseline_incomplete"
    assert result["source_file"] is None
    assert any("评分文件" in reason for reason in result["reasons"])
    assert records == []


def test_review_ids_pair_across_real_run_row_orders(tmp_path: Path) -> None:
    assert _review_fixture("a") != _review_fixture("b")
    paired = []
    for run in ("a", "b"):
        root = tmp_path / run
        _write_review(root, _review_fixture(run))
        result, records = _extract_review(root)
        assert result["status"] == "baseline_established"
        paired.append({item["question_id"]: item["correct"] for item in records})
    assert paired[0] == paired[1]


@pytest.mark.parametrize("missing", [False, True])
def test_review_uses_dataset_index_not_line_number(
    tmp_path: Path, missing: bool
) -> None:
    rows = _review_fixture()
    rows[0]["sample_score"].pop("sample_id")
    if missing:
        rows[0].pop("index")
    _write_review(tmp_path, rows)
    result, records = _extract_review(tmp_path)
    if missing:
        assert result["status"] == "baseline_incomplete"
        assert any("缺少稳定题目 ID" in reason for reason in result["reasons"])
        assert len(records) == 11
    else:
        assert result["status"] == "baseline_established"
        assert records[0]["question_id"] == "6"


def test_review_output_drops_all_text(tmp_path: Path) -> None:
    rows = _review_fixture()
    for row in rows:
        row["messages"] = [{"role": "user", "content": "private question"}]
        row["prediction"] = "private prediction"
        row["sample_score"]["score"]["explanation"] = "private explanation"
    _write_review(tmp_path, rows)
    _, records = _extract_review(tmp_path)
    assert all(set(item) == {"question_id", "correct", "score"} for item in records)
    assert "private" not in (tmp_path / "per_question.jsonl").read_text()


def _baseline_records(count: int = 10) -> list[dict[str, object]]:
    return [
        {"question_id": str(i), "correct": i % 2 == 0, "score": float(i % 2 == 0)}
        for i in range(count)
    ]


def test_per_question_baseline_established() -> None:
    records = _baseline_records()
    original = copy.deepcopy(records)
    result = ppu_recipe_runner.assess_per_question_baseline(
        records, expected_count=10, aggregate_score=0.5, aggregate_num=10
    )
    assert result == {
        "status": "baseline_established",
        "reasons": [],
        "count": 10,
        "unique_ids": 10,
        "unscored": 0,
        "correct_sum": 5,
    }
    assert records == original


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("duplicate", "重复 ID"),
        ("missing_id", "缺少稳定题目 ID"),
        ("empty_id", "缺少稳定题目 ID"),
        ("count", "数量不符"),
        ("unscored", "存在未评分题"),
        ("null", "存在未评分题"),
        ("aggregate_count", "与汇总不一致"),
        ("aggregate_score", "与汇总不一致"),
        ("nonfinite", "与汇总不一致"),
    ],
)
def test_per_question_baseline_incomplete(case: str, reason: str) -> None:
    records = _baseline_records()
    expected_count, aggregate_num, aggregate_score, unscored = 10, 10, 0.5, 0
    if case == "duplicate":
        records[1]["question_id"] = records[0]["question_id"]
    elif case == "missing_id":
        records[0]["question_id"] = None
    elif case == "empty_id":
        records[0]["question_id"] = ""
    elif case == "count":
        records = _baseline_records(1318)
        expected_count, aggregate_num, aggregate_score = 1319, 1318, 0.5
    elif case == "unscored":
        unscored = 1
    elif case == "null":
        records[1]["correct"] = None
    elif case == "aggregate_count":
        aggregate_num = 11
    elif case == "aggregate_score":
        aggregate_score = 0.6
    elif case == "nonfinite":
        aggregate_score = float("nan")
    result = ppu_recipe_runner.assess_per_question_baseline(
        records,
        expected_count=expected_count,
        aggregate_score=aggregate_score,
        aggregate_num=aggregate_num,
        unscored=unscored,
    )
    assert result["status"] == "baseline_incomplete"
    assert any(reason in item for item in result["reasons"])
    if case in {"unscored", "null"}:
        assert result["unscored"] == 1


def test_per_question_baseline_reconciles_rounded_full_report() -> None:
    records = [
        {"question_id": str(i), "correct": i < 1278, "score": float(i < 1278)}
        for i in range(1319)
    ]
    result = ppu_recipe_runner.assess_per_question_baseline(
        records, expected_count=1319, aggregate_score=0.9689, aggregate_num=1319
    )
    assert result["status"] == "baseline_established"
    assert result["correct_sum"] == 1278
    records[0]["correct"] = False
    records[0]["score"] = 0.0
    result = ppu_recipe_runner.assess_per_question_baseline(
        records, expected_count=1319, aggregate_score=0.9689, aggregate_num=1319
    )
    assert abs(1277 / 1319 - 0.9689) < 1 / 1319
    assert result["status"] == "baseline_incomplete"
    assert "与汇总不一致" in result["reasons"]


def test_empty_per_question_baseline_is_incomplete() -> None:
    result = ppu_recipe_runner.assess_per_question_baseline(
        [], expected_count=1319, aggregate_score=0.0, aggregate_num=0
    )
    assert result["status"] == "baseline_incomplete"
    assert result["count"] == result["unique_ids"] == result["correct_sum"] == 0


def test_resolve_recipe_reports_enabled_features() -> None:
    dsv4_recipe = RECIPES_DIR / "deepseek-ai" / "DeepSeek-V4-Flash.yaml"
    dsv4_policy = POLICY_DIR / "models" / "deepseek-v4-flash.yaml"
    assert (
        ppu_recipe.resolve_recipe(
            dsv4_recipe, policy_path=dsv4_policy, tier="model_sanity"
        )["enabled_features"]
        == []
    )
    assert ppu_recipe.resolve_recipe(
        dsv4_recipe, preset="dp_ep_low_latency", cache="enabled"
    )["enabled_features"] == ["tool_calling", "reasoning", "spec_decoding"]

    glm_recipe = RECIPES_DIR / "zai-org" / "GLM-5.3.yaml"
    glm_policy = POLICY_DIR / "models" / "glm-5.3.yaml"
    assert ppu_recipe.resolve_recipe(
        glm_recipe, policy_path=glm_policy, tier="fast_sanity"
    )["enabled_features"] == ["tool_calling", "reasoning"]


def test_checked_in_recipe_matrix_doc_is_up_to_date() -> None:
    expected = ppu_recipe.render_ci_matrix_markdown()
    assert ppu_recipe.DEFAULT_MATRIX_DOC.is_file()
    actual = ppu_recipe.DEFAULT_MATRIX_DOC.read_text(encoding="utf-8")
    assert actual == expected, (
        "tests/ppu/README.md is out of date; regenerate with: "
        "python3 scripts/ci/ppu_recipe.py render-matrix --output tests/ppu/README.md"
    )
    assert ppu_recipe._scan_text(actual, context="tests/ppu/README.md") == []
    assert "## PR CI (`fast_sanity`)" in actual
    assert "## Nightly CI — Recipe (`model_sanity`)" in actual
    assert "## Nightly CI — Remaining Legacy Accuracy Cases" in actual
    assert "`default` (none)" in actual
    assert "`tool_calling`, `reasoning`" in actual
    legacy_section = actual.split("## Nightly CI — Remaining Legacy Accuracy Cases", 1)[
        1
    ]
    assert "| `deepseek-v4-mxfp4` |" not in legacy_section
    assert "| `glm-5.3-mxfp4-tp4` |" not in legacy_section
    assert "| `deepseek-v4-mxfp4-dp-ep` |" not in legacy_section
    assert "| `deepseek-v4-int8-dp-ep` |" in legacy_section


def test_render_matrix_cli_check_passes_and_detects_drift(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert ppu_recipe.main(["render-matrix", "--check"]) == 0

    missing = tmp_path / "missing.md"
    assert ppu_recipe.main(["render-matrix", "--output", str(missing), "--check"]) == 1
    assert "matrix doc missing" in capsys.readouterr().err

    target = tmp_path / "README.md"
    target.write_text("# stale\n", encoding="utf-8")
    assert ppu_recipe.main(["render-matrix", "--output", str(target), "--check"]) == 1
    assert "matrix doc out of date" in capsys.readouterr().err

    assert ppu_recipe.main(["render-matrix", "--output", str(target)]) == 0
    assert ppu_recipe.main(["render-matrix", "--output", str(target), "--check"]) == 0
    assert target.read_text(encoding="utf-8") == ppu_recipe.render_ci_matrix_markdown()


@pytest.mark.requires_runtime
def test_runtime_vllm_parser_validates_checked_in_recipes() -> None:
    pytest.importorskip("vllm")
    files = _yaml_files()
    report = ppu_recipe.validate_paths(files, vllm_parser_mode="required")
    assert report["parser_checked"] is True
    assert report["checked_flag_sets"] >= 10
