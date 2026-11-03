# SPDX-License-Identifier: Apache-2.0
"""Opt-in V4.1 call auditing without operator replacement; not intended for performance timing."""

import hashlib
import importlib.metadata
import inspect
import json
import os
import sys
import time
import types
from pathlib import Path

# Identify executed frames so provider/consumer aliases cannot bypass auditing.
TARGETS = {
    (
        "vllm_sail.models.deepseek_v41.ops.indexer_tilelang",
        "fused_indexer_q_rope_quant_int8",
    ): "q",
    (
        "vllm_sail.models.deepseek_v41.ops.indexer_tilelang",
        "indexer_k_norm_rope_store_int8",
    ): "k",
    ("vllm_sail.utils.deep_gemm", "int8_mqa_logits"): "logits_prefill",
    ("vllm_sail.utils.deep_gemm", "int8_paged_mqa_logits"): "logits_decode",
    ("vllm_sail.attention.ops.mla_sparse", "ppu_sparse_attn_indexer"): "indexer",
    (
        "vllm.model_executor.kernels.attention.dsa.candidate_blocks",
        "select_candidate_blocks",
    ): "candidate_write",
    (
        "vllm.model_executor.kernels.attention.dsa.candidate_blocks",
        "apply_candidate_mask",
    ): "candidate_mask",
    (
        "vllm.models.deepseek_v41.nvidia.flashmla",
        "_forward_prefill",
    ): "attention_prefill",
    ("vllm.models.deepseek_v41.nvidia.flashmla", "_forward_decode"): "attention_decode",
    ("vllm_sail.models.deepseek_v41.flashmla", "_o_proj"): "o_proj",
    ("vllm_sail.models.deepseek_v41.ops.engram_lookup", "lookup"): "engram_lookup",
    (
        "vllm_sail.models.deepseek_v41.ops.latent_store",
        "rope_quant_insert",
    ): "main_cache_store",
    (
        "vllm_sail.patch.enhancement.attention.flashmla_ops",
        "flash_mla_sparse_fwd",
    ): "flashmla_prefill_call",
    (
        "vllm_sail.patch.enhancement.attention.flashmla_ops",
        "flash_mla_with_kvcache",
    ): "flashmla_decode_call",
    (
        "vllm_sail.patch.enhancement.attention.mhc_delayed",
        "mhc_pre_delayed_tilelang",
    ): "mhc_pre_delayed",
    (
        "vllm_sail.patch.enhancement.attention.mhc_delayed",
        "mhc_fused_post_pre_delayed_tilelang",
    ): "mhc_fused_delayed",
}
NATIVE = {
    "_C::persistent_topk": "topk_decode",
    "_C::top_k_per_row_prefill": "topk_prefill",
    "_C::cp_gather_indexer_k_quant_cache": "indexer_gather",
    "_C::fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert": "qnorm_rope_swa",
    "vllm::ppu_deepseek_v4_dequant_gather": "main_cache_gather",
}


def file_identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return dict(path=str(path), sha256=digest.hexdigest())


def opened_file_identity(handle):
    """Hash the same opened object and reject writes during sampling."""
    before = os.fstat(handle.fileno())
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    after = os.fstat(handle.fileno())
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, key) != getattr(after, key) for key in fields):
        raise RuntimeError("Operator binary changed during sampling")
    return dict(
        device=f"{os.major(after.st_dev):x}:{os.minor(after.st_dev):x}",
        inode=after.st_ino,
        size=after.st_size,
        sha256=digest.hexdigest(),
    )


def native_mapping_identity(parts):
    """Container overlay maps and stat may use different device IDs; verify the mapped object."""
    path = parts[5]
    if path.endswith(" (deleted)"):
        raise RuntimeError("Loaded operator binary was replaced or deleted: " + path)
    record = dict(path=path, device=parts[3], inode=int(parts[4]))
    with Path(path).open("rb") as handle:
        file = opened_file_identity(handle)
        record.update(file=file, identity_method="proc_maps_stat")
        device = tuple(int(value, 16) for value in parts[3].split(":"))
        file_device = tuple(int(value, 16) for value in file["device"].split(":"))
        if file["inode"] != record["inode"] or file_device != device:
            # If device IDs differ, open the mapped backing file through the kernel's map_files.
            mapped_path = Path("/proc/self/map_files") / parts[0]
            with mapped_path.open("rb") as mapped:
                mapped_file = opened_file_identity(mapped)
            if mapped_file != file:
                raise RuntimeError(
                    "Mapped operator binary and disk file are different objects: "
                    + path
                )
            record.update(identity_method="proc_map_files", mapped_file=mapped_file)
        current = Path(path).stat()
        opened = os.fstat(handle.fileno())
        if (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino):
            raise RuntimeError(
                "Operator binary path was replaced during sampling: " + path
            )
    return record


def loaded_runtime():
    """Sample only loaded modules and mappings without importing operators."""
    prefixes = ("vllm", "vllm_sail", "deep_gemm", "flash_mla", "tilelang", "triton")
    files, modules, native = {}, {}, []

    def remember(path):
        identity = file_identity(path)
        files[identity["path"]] = identity["sha256"]
        return identity["path"]

    for name, module in tuple(sys.modules.items()):
        if name.split(".")[0] not in prefixes and name != "torch._ops":
            continue
        path = getattr(module, "__file__", None)
        if path:
            modules[name] = remember(path)
    seen = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        parts = line.split(None, 5)
        if len(parts) != 6 or not parts[5].startswith("/"):
            continue
        path = parts[5]
        if ".so" not in path or not any("/" + name + "/" in path for name in prefixes):
            continue
        key = (path, parts[3], parts[4])
        if key in seen:
            continue
        record = native_mapping_identity(parts)
        resolved = str(Path(path).resolve())
        if resolved in files and files[resolved] != record["file"]["sha256"]:
            raise RuntimeError(
                "Operator binary changed between module and mapping sampling: " + path
            )
        files[resolved] = record["file"]["sha256"]
        native.append(record)
        seen.add(key)
    distribution = importlib.metadata.distribution("vllm-sail")
    records = [
        path for path in distribution.files if str(path).endswith(".dist-info/RECORD")
    ]
    if len(records) != 1:
        raise RuntimeError("Cannot uniquely identify the installed plugin RECORD")
    return dict(
        modules=modules,
        files=files,
        native_mappings=native,
        plugin_record=file_identity(distribution.locate_file(records[0])),
        plugin_version=distribution.version,
        python=sys.version,
        uid=os.getuid(),
        gid=os.getgid(),
    )


def code_fingerprint(code):
    """Serialize code structure stably; marshal reference/interning markers vary with process state."""

    def canonical(value):
        if isinstance(value, types.CodeType):
            fields = (
                "co_argcount",
                "co_posonlyargcount",
                "co_kwonlyargcount",
                "co_nlocals",
                "co_stacksize",
                "co_flags",
                "co_code",
                "co_consts",
                "co_names",
                "co_varnames",
                "co_filename",
                "co_name",
                "co_qualname",
                "co_firstlineno",
                "co_linetable",
                "co_exceptiontable",
                "co_freevars",
                "co_cellvars",
            )
            return [
                "code",
                [[field, canonical(getattr(value, field, None))] for field in fields],
            ]
        if isinstance(value, tuple | frozenset):
            items = [canonical(item) for item in value]
            return [
                type(value).__name__,
                sorted(items, key=repr) if isinstance(value, frozenset) else items,
            ]
        return [
            type(value).__name__,
            value.hex() if isinstance(value, bytes) else repr(value),
        ]

    return hashlib.sha256(
        json.dumps(canonical(code), ensure_ascii=True, separators=(",", ":")).encode()
    ).hexdigest()


def native_name(op):
    # OpOverloadPacket and its .default OpOverload use different identity fields.
    return getattr(op, "_qualified_op_name", None) or getattr(
        getattr(op, "_schema", None), "name", None
    )


def tensor_layout(value):
    return dict(
        shape=list(value.shape),
        stride=list(value.stride()),
        dtype=str(value.dtype),
        device=str(value.device),
        storage_offset=value.storage_offset(),
    )


def process_start():
    return int(Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19])


def mapped_region(tensor):
    """Locate mmap by the data address rather than inferring sharing from matching filenames."""
    pointer = tensor.data_ptr()
    for line in Path("/proc/self/maps").read_text().splitlines():
        parts = line.split(None, 5)
        begin, end = (int(item, 16) for item in parts[0].split("-"))
        if begin <= pointer < end:
            return dict(
                begin=begin,
                end=end,
                permissions=parts[1],
                offset=parts[2],
                device=parts[3],
                inode=parts[4],
                path=parts[5] if len(parts) == 6 else "",
                data_offset=pointer - begin,
            )
    raise RuntimeError("Engram data address is not in the process mappings")


def worker_identity(model, config):
    import torch
    from vllm.distributed import parallel_state as parallel

    groups = {}
    for name in ("tp", "dp", "ep"):
        if name == "ep" and not config.parallel_config.enable_expert_parallel:
            groups[name] = dict(enabled=False, size=1, rank=0)
        else:
            group = getattr(parallel, "get_" + name + "_group")()
            groups[name] = dict(
                enabled=True,
                size=group.world_size,
                rank=group.rank_in_group,
                ranks=list(group.ranks),
            )
    device = torch.cuda.current_device()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if device >= len(visible) or not visible[device].isdigit():
        raise RuntimeError(
            "Runtime auditing requires an explicit numeric physical-device mapping"
        )
    classes = {}
    for obj in (model, model.language_model):
        cls = type(obj)
        classes[cls.__module__ + "." + cls.__name__] = file_identity(
            inspect.getfile(cls)
        )
    engram = {}
    for name, module in model.named_modules():
        if not hasattr(module, "dp_shared_memory") or not hasattr(
            module, "weight_scale_inv"
        ):
            continue
        tensors = {}
        for key in ("weight", "weight_scale_inv"):
            value = getattr(module, key)
            tensors[key] = dict(
                tensor_layout(value),
                pinned=value.is_pinned(),
                mapping=mapped_region(value) if value.device.type == "cpu" else None,
            )
        engram[name] = dict(
            shared=module.dp_shared_memory,
            cpu_offload=module.cpu_offload,
            head_start=module.head_start,
            head_end=module.head_start + module.part_n_hash_cols,
            vocab_start=module.vocab_start_idx,
            vocab_end=module.vocab_end_idx,
            tensors=tensors,
        )
    return dict(
        groups=groups,
        logical_device=device,
        physical_device=int(visible[device]),
        visible_devices=visible,
        device_name=torch.cuda.get_device_name(device),
        model_classes=classes,
        model=config.model_config.model,
        engram=engram,
        parallel_config=repr(config.parallel_config),
        cache_config=repr(config.cache_config),
    )


def candidate_sample(values):
    """Read only the last row to avoid copying full long-context logits; synchronize only during auditing."""
    import torch

    logits = values["logits"]
    if not logits.shape[0] or not logits.shape[1]:
        return dict(empty=True)
    row = logits.shape[0] - 1
    bound = row // values["row_repeat"]
    starts = values["row_ks"]
    start = 0 if starts is None else int(starts[bound].item())
    end = int(values["row_ke"][bound].item())
    size = values["block_size"]
    valid = logits[row, start : min(end, logits.shape[1])]
    result = dict(
        row=row,
        start=start,
        end=end,
        block_size=size,
        valid_blocks=max(0, (end - start + size - 1) // size),
        finite_valid=int(torch.isfinite(valid).sum().item()),
        outside_unmasked=int((~torch.isneginf(logits[row, :start])).sum().item())
        + int((~torch.isneginf(logits[row, end:])).sum().item()),
        latest_visible_finite=int(
            torch.isfinite(valid[-((end - start - 1) % size + 1) :]).sum().item()
        )
        if end > start
        else 0,
    )
    candidates = values.get("candidate_blocks", values.get("out"))
    if candidates is not None:
        blocks = candidates[row].detach().cpu().tolist()
        live = [int(block) for block in blocks if block >= 0]
        result.update(
            candidate_slots=candidates.shape[1],
            live_candidates=len(live),
            candidate_sha256=hashlib.sha256(json.dumps(blocks).encode()).hexdigest(),
            latest_kept=(end > start and (end - start - 1) // size in live),
        )
    return result


class ForwardTrace:
    """Record each successful forward independently; reject exceptions and asynchronous device errors."""

    def __init__(self):
        self.events = {}
        self.pending = {}
        self.files = {}

    def _identity(self, frame):
        module = frame.f_globals.get("__name__", "")
        kind = TARGETS.get((module, frame.f_code.co_name))
        if module == "torch._ops" and frame.f_code.co_name == "__call__":
            obj = frame.f_locals.get("self")
            kind = NATIVE.get(native_name(obj))
        return kind

    def __call__(self, frame, event, arg):
        if event not in ("call", "return"):
            return
        kind = self._identity(frame)
        if kind is None:
            return
        if event == "call":
            # Do not read uninitialized out; record before/after valid-element counts only for masking.
            if kind == "candidate_mask":
                self.pending[id(frame)] = candidate_sample(frame.f_locals)
            return
        values = frame.f_locals
        ratio = values.get("compress_ratio") if kind == "k" else None
        key = kind + (str(ratio) if ratio is not None else "")
        details = {}
        for name in (
            "q",
            "q_quant",
            "k",
            "k_cache",
            "kv_cache",
            "logits",
            "weights",
            "attn_out",
            "swa_kv_cache",
            "compressed_k_cache",
            "swa_k_cache",
            "out",
            "scales",
        ):
            value = values.get(name)
            if hasattr(value, "shape") and hasattr(value, "stride"):
                details[name] = tensor_layout(value)
        for name in (
            "compress_ratio",
            "num_actual_tokens",
            "candidate_write",
            "candidate_block_size",
            "k_cache_prefix",
        ):
            value = values.get(name)
            if isinstance(value, str | bool | int):
                details[name] = value
        if kind.startswith("candidate_"):
            details["after"] = candidate_sample(values)
            if kind == "candidate_mask":
                details["before"] = self.pending.pop(id(frame), None)
        if kind in NATIVE.values():
            obj = values["self"]
            details["op"] = native_name(obj)
            details["arguments"] = [
                tensor_layout(value)
                if hasattr(value, "shape") and hasattr(value, "stride")
                else value
                if isinstance(value, int | float | bool | str | type(None))
                else type(value).__name__
                for value in values.get("args", ())
            ]
            details["schema"] = str(
                getattr(obj, "_schemas", getattr(obj, "_schema", None))
            )
        source = frame.f_code.co_filename
        if source not in self.files:
            self.files[source] = file_identity(source)
        item = self.events.setdefault(key, dict(count=0, samples=[]))
        item["count"] += 1
        sample = dict(
            details=details,
            source=self.files[source],
            function=frame.f_globals.get("__name__", "") + "." + frame.f_code.co_name,
            code_filename=frame.f_code.co_filename,
            code_sha256=code_fingerprint(frame.f_code),
        )
        # Retain first/last samples and count all calls to bound per-layer/per-token log growth.
        if len(item["samples"]) < 2:
            item["samples"].append(sample)
        else:
            item["samples"][-1] = sample


def execute_traced(fn, args, kwargs, synchronize):
    if sys.getprofile() is not None or sys.gettrace() is not None:
        raise RuntimeError(
            "V4.1 runtime auditing cannot run with another profiler/trace"
        )
    trace = ForwardTrace()
    try:
        sys.setprofile(trace)
        output = fn(*args, **kwargs)
        synchronize()
    finally:
        sys.setprofile(None)
    return output, trace.events


def forward_token_counts(metadata, padding):
    """DP dummy forwards may have positive metadata token counts; distinguish them using the padding mask."""
    if not isinstance(metadata, dict):
        return None
    metadata_tokens = max(
        (getattr(value, "num_actual_tokens", 0) for value in metadata.values()),
        default=0,
    )
    if metadata_tokens <= 0:
        return None
    # The pinned GPUModelRunner provides this mask for both real and dummy forwards.
    if padding is None or padding.ndim != 1 or str(padding.dtype) != "torch.bool":
        raise RuntimeError(
            "Runtime auditing requires the runner's 1D bool padding mask"
        )
    padded_tokens = padding.numel()
    if padded_tokens < metadata_tokens:
        raise RuntimeError("Padding mask is shorter than attention metadata")
    padding_tokens = int(padding.sum().item())
    actual_tokens = padded_tokens - padding_tokens
    if actual_tokens > metadata_tokens or not bool(
        padding[metadata_tokens:].all().item()
    ):
        raise RuntimeError("Actual tokens exceed the attention metadata bounds")
    return dict(
        source="forward_context.is_padding",
        metadata_tokens=metadata_tokens,
        padded_tokens=padded_tokens,
        padding_tokens=padding_tokens,
        actual_tokens=actual_tokens,
    )


class RuntimeAudit:
    def __init__(self, directory, model, config):
        self.root = Path(directory)
        if not self.root.is_absolute() or not self.root.is_dir():
            raise ValueError(
                "The runtime audit directory must be absolute and created before startup"
            )
        self.pid = os.getpid()
        self.start = process_start()
        self.config = config
        # Constructor allocation does not imply that weights have been loaded.
        self.model = model
        self.identity = worker_identity(model, config)
        self.captured_sessions = {}
        self.idle_sessions = set()
        self.sequence = 0
        self.log = self.root / f"worker_{self.pid}_{self.start}.jsonl"
        with self.log.open("x") as handle:
            handle.write(
                json.dumps(
                    self.record(
                        "constructed", identity=self.identity, runtime=loaded_runtime()
                    )
                )
                + "\n"
            )

    def record(self, event, **values):
        return dict(
            schema=1,
            event=event,
            pid=self.pid,
            start_ticks=self.start,
            utc_ns=time.time_ns(),
            **values,
        )

    def run(self, fn, *args, **kwargs):
        control = self.root / "capture.json"
        if (self.root / "stop").exists() or not control.exists():
            return fn(*args, **kwargs)
        session = json.loads(control.read_text())
        if session.get("schema") != 1 or not isinstance(session.get("session_id"), str):
            raise ValueError("Invalid runtime audit session")
        import torch
        from vllm.forward_context import get_forward_context

        context = get_forward_context()
        # Exclude startup dummy/profile forwards and DP ranks without actual tokens.
        counts = forward_token_counts(context.attn_metadata, context.is_padding)
        if counts is None:
            return fn(*args, **kwargs)
        actual = counts["actual_tokens"]
        if actual == 0 and session["session_id"] in self.idle_sessions:
            return fn(*args, **kwargs)
        self.sequence += 1
        try:
            if actual:
                output, events = execute_traced(
                    fn, args, kwargs, torch.cuda.synchronize
                )
            else:
                # Run and synchronize the original model for idle ranks; record identity once without operator coverage.
                output = fn(*args, **kwargs)
                torch.cuda.synchronize()
                events = {}
            observed = {
                (kind, sample["source"]["path"])
                for kind, item in events.items()
                for sample in item["samples"]
            }
            known = self.captured_sessions.get(session["session_id"])
            if known is None or observed - known:
                # Refresh loaded-file identities on the first forward or a new decode/delayed path.
                identity = worker_identity(self.model, self.config)
                if identity["groups"] != self.identity["groups"]:
                    raise RuntimeError(
                        "Worker parallel groups changed during forward execution"
                    )
                with self.log.open("a") as handle:
                    handle.write(
                        json.dumps(
                            self.record(
                                "runtime_loaded",
                                session_id=session["session_id"],
                                sequence=self.sequence,
                                identity=identity,
                                runtime=loaded_runtime(),
                            )
                        )
                        + "\n"
                    )
                self.captured_sessions[session["session_id"]] = (
                    known or set()
                ) | observed
        except BaseException as exc:
            with self.log.open("a") as handle:
                handle.write(
                    json.dumps(
                        self.record(
                            "forward_failed",
                            session_id=session["session_id"],
                            sequence=self.sequence,
                            error=repr(exc),
                        )
                    )
                    + "\n"
                )
            raise
        with self.log.open("a") as handle:
            handle.write(
                json.dumps(
                    self.record(
                        "forward" if actual else "idle_forward",
                        session_id=session["session_id"],
                        sequence=self.sequence,
                        actual_tokens=actual,
                        token_counts=counts,
                        groups=self.identity["groups"],
                        events=events,
                    )
                )
                + "\n"
            )
        if not actual:
            self.idle_sessions.add(session["session_id"])
        return output
