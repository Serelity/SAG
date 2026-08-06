"""Auditable work-order semantic extraction orchestration.

The module is safe to import on development machines: Transformers and Torch are
imported only inside ``load_transformers_generator``.  Unit tests inject a fake
generator and never load a model.
"""

import argparse
import hashlib
import json
import os
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from ragflow_style_pipeline.sag_semantic_prompt import build_repair_prompt, build_semantic_prompt
from ragflow_style_pipeline.sag_semantic_schema import ENTITY_GROUPS, GROUP_LIMITS, parse_semantic_json
from ragflow_style_pipeline.sag_semantic_versions import (
    CANDIDATE_LEDGER_VERSION,
    DECISION_LEDGER_VERSION,
    DECODER_CONTRACT_VERSION,
    PROJECTION_VERSION,
    VALIDATOR_VERSION,
)
from ragflow_style_pipeline.sag_semantic_validation import (
    enrich_semantic_output,
    sanitize_semantic_output,
    validate_semantic_output,
)
from ragflow_style_pipeline.work_order_input import read_work_orders


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _config_hash(config):
    encoded = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _environment_bool(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid_boolean_environment:{name}")


def _identity(doc_id, content_hash, prompt_version, model_id):
    return "\u241f".join((doc_id, content_hash, prompt_version, model_id))


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        output.flush()
        os.fsync(output.fileno())


def _append_jsonl(path, rows):
    rows = list(rows)
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        output.flush()
        os.fsync(output.fileno())


def _read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line_{line_number}:invalid_record")
            rows.append(value)
    return rows


def _jsonl_record_count(path):
    if not path or not Path(path).exists():
        return 0
    with Path(path).open("r", encoding="utf-8") as source:
        return sum(1 for line in source if line.strip())


def _percentile(values, fraction):
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def _normalize_generation_result(value, max_new_tokens):
    if isinstance(value, str):
        return {
            "text": value,
            "input_tokens": 0,
            "output_tokens": 0,
            "finish_reason": "stop",
            "latency_ms": 0,
        }
    if not isinstance(value, dict):
        raise TypeError("generator_result_must_be_string_or_object")
    output_tokens = int(value.get("output_tokens") or 0)
    finish_reason = str(value.get("finish_reason") or ("length" if output_tokens >= max_new_tokens else "stop"))
    return {
        "text": str(value.get("text") or ""),
        "input_tokens": int(value.get("input_tokens") or 0),
        "output_tokens": output_tokens,
        "finish_reason": finish_reason,
        "latency_ms": float(value.get("latency_ms") or 0),
    }


def _bucket_orders(orders, boundaries):
    boundaries = sorted(int(value) for value in (boundaries or []) if int(value) >= 0)

    def bucket(order):
        length = len(order.get("chunk_text", ""))
        return next((index for index, boundary in enumerate(boundaries) if length <= boundary), len(boundaries))

    return sorted(enumerate(orders), key=lambda pair: (bucket(pair[1]), len(pair[1].get("chunk_text", "")), pair[0]))


def _record(order, semantic, validation, generation, config, repair_attempted):
    return {
        "schema_version": str(config.get("schema_version", "2.0")),
        "artifact_versions": {
            "validator": VALIDATOR_VERSION,
            "projection": PROJECTION_VERSION,
            "decoder_contract": str(
                config.get("decoder_contract_version", DECODER_CONTRACT_VERSION)
            ),
        },
        "doc_id": order["doc_id"],
        "content_hash": order["content_hash"],
        "event": {
            "summary": semantic.get("event_summary", ""),
            "evidence_fields": [
                field
                for field in ("title_clean", "case_content_clean", "case_goal_clean", "address_detail_clean")
                if order.get(field)
            ],
        },
        "entities": semantic.get("entities", {}),
        "discourse": semantic.get("discourse", {}),
        "validation": {
            "validator_version": VALIDATOR_VERSION,
            "status": validation["status"],
            "warnings": validation["warnings"],
            "repair_fields": validation["repair_fields"],
            "repair_attempted": bool(repair_attempted),
        },
        "model_run": {
            "model": str(config.get("model_id", "Qwen/Qwen3-4B")),
            "prompt_version": str(config.get("prompt_version", "sag_semantic_v7")),
            "decoder_contract_version": str(
                config.get("decoder_contract_version", DECODER_CONTRACT_VERSION)
            ),
            "backend": str(config.get("backend", "transformers")),
            "input_tokens": generation["input_tokens"],
            "output_tokens": generation["output_tokens"],
            "finish_reason": generation["finish_reason"],
            "latency_ms": generation["latency_ms"],
        },
    }


def _validate_with_sanitation(order, semantic, parse_warnings):
    semantic, enrichment_actions = enrich_semantic_output(order, semantic, parse_warnings)
    validation = validate_semantic_output(order, semantic, parse_warnings)
    trace = {
        "parse_warnings": list(parse_warnings or []),
        "validation_before": validation,
        "sanitation_warnings": list(enrichment_actions),
    }
    if validation["status"] != "repair_required":
        overflow = any(
            isinstance(semantic.get("entities", {}).get(group), list)
            and len(semantic["entities"][group]) > limit
            for group, limit in GROUP_LIMITS.items()
        )
        final_actions = list(enrichment_actions)
        if overflow:
            semantic, limit_actions = sanitize_semantic_output(
                semantic, validation["warnings"], order=order
            )
            for action in limit_actions:
                if action not in final_actions:
                    final_actions.append(action)
            validation = validate_semantic_output(order, semantic, parse_warnings)
        if final_actions:
            final_warnings = list(final_actions)
            for warning in validation["warnings"]:
                if warning not in final_warnings:
                    final_warnings.append(warning)
            validation = {
                **validation,
                "status": "accepted_with_warnings",
                "warnings": final_warnings,
            }
        trace["sanitation_warnings"] = final_actions
        trace["validation_after"] = validation
        return semantic, validation, trace
    if "json_parse_failed" in validation["warnings"] or "possible_history_contamination" in validation["warnings"]:
        trace["validation_after"] = validation
        return semantic, validation, trace

    cleaned = semantic
    cleaned_validation = validation
    sanitation_warnings = list(enrichment_actions)
    for _pass in range(3):
        next_cleaned, actions = sanitize_semantic_output(
            cleaned, cleaned_validation["warnings"], order=order
        )
        for action in actions:
            if action not in sanitation_warnings:
                sanitation_warnings.append(action)
        if not actions or next_cleaned == cleaned:
            break
        cleaned = next_cleaned
        cleaned_validation = validate_semantic_output(order, cleaned, parse_warnings)
        if cleaned_validation["status"] in {"accepted", "accepted_with_warnings"}:
            break
        if "json_parse_failed" in cleaned_validation["warnings"] or "possible_history_contamination" in cleaned_validation["warnings"]:
            break

    trace["sanitation_warnings"] = sanitation_warnings
    if cleaned_validation["status"] in {"accepted", "accepted_with_warnings"} and sanitation_warnings:
        final_warnings = list(sanitation_warnings)
        for warning in cleaned_validation["warnings"]:
            if warning not in final_warnings:
                final_warnings.append(warning)
        cleaned_validation = {
            **cleaned_validation,
            "status": "accepted_with_warnings",
            "warnings": final_warnings,
        }
    trace["validation_after"] = cleaned_validation
    return cleaned, cleaned_validation, trace


def _diagnostic_identity(order):
    value = f"{order.get('doc_id', '')}:{order.get('content_hash', '')}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _semantic_counts(semantic):
    entities = semantic.get("entities") if isinstance(semantic, dict) else {}
    discourse = semantic.get("discourse") if isinstance(semantic, dict) else {}
    return {
        "entities": {
            group: len(entities.get(group, [])) if isinstance(entities, dict) and isinstance(entities.get(group), list) else 0
            for group in ENTITY_GROUPS
        },
        "intents": len(discourse.get("intents", [])) if isinstance(discourse, dict) and isinstance(discourse.get("intents"), list) else 0,
        "emotions": len(discourse.get("emotions", [])) if isinstance(discourse, dict) and isinstance(discourse.get("emotions"), list) else 0,
        "satisfaction": (
            discourse.get("satisfaction", {}).get("label", "unknown")
            if isinstance(discourse, dict) and isinstance(discourse.get("satisfaction"), dict) else "unknown"
        ),
        "urgency": (
            discourse.get("urgency", {}).get("level", "normal")
            if isinstance(discourse, dict) and isinstance(discourse.get("urgency"), dict) else "normal"
        ),
    }


def _private_candidate_entry(
    order, phase, semantic, parse_warnings, generation, config,
    run_attempt_id, ledger_sequence,
):
    return {
        "schema": CANDIDATE_LEDGER_VERSION,
        "private": True,
        "doc_id": order.get("doc_id", ""),
        "content_hash": order.get("content_hash", ""),
        "phase": phase,
        "run_attempt_id": run_attempt_id,
        "ledger_sequence": ledger_sequence,
        "model": str(config.get("model_id", "Qwen/Qwen3-4B")),
        "prompt_version": str(config.get("prompt_version", "sag_semantic_v7")),
        "decoder_contract_version": str(
            config.get("decoder_contract_version", DECODER_CONTRACT_VERSION)
        ),
        "candidate": semantic,
        "parse_warnings": list(parse_warnings or []),
        "generation": {
            "input_tokens": generation.get("input_tokens", 0),
            "output_tokens": generation.get("output_tokens", 0),
            "finish_reason": generation.get("finish_reason", "unknown"),
            "latency_ms": generation.get("latency_ms", 0),
        },
    }


def _private_decision_entry(
    order, phase, final_semantic, validation, trace,
    run_attempt_id, ledger_sequence,
):
    before = trace.get("validation_before") if isinstance(trace, dict) else {}
    after = trace.get("validation_after") if isinstance(trace, dict) else {}
    return {
        "schema": DECISION_LEDGER_VERSION,
        "private": True,
        "validator_version": VALIDATOR_VERSION,
        "doc_id": order.get("doc_id", ""),
        "content_hash": order.get("content_hash", ""),
        "phase": phase,
        "run_attempt_id": run_attempt_id,
        "ledger_sequence": ledger_sequence,
        "parse_warnings": list(trace.get("parse_warnings", [])) if isinstance(trace, dict) else [],
        "validation_before": before,
        "actions": list(trace.get("sanitation_warnings", [])) if isinstance(trace, dict) else [],
        "validation_after": after or validation,
        "final_counts": _semantic_counts(final_semantic),
    }


def _append_diagnostic(path, event):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
        output.flush()


def _run_generator(generator, prompts, config, max_new_tokens=None):
    token_limit = int(max_new_tokens if max_new_tokens is not None else config.get("max_new_tokens", 512))
    results = generator(
        prompts,
        max_new_tokens=token_limit,
        temperature=float(config.get("temperature", 0.0)),
    )
    if isinstance(results, (str, dict)):
        results = [results]
    results = list(results)
    if len(results) != len(prompts):
        raise RuntimeError(f"generator_result_count:{len(results)}:{len(prompts)}")
    return [_normalize_generation_result(value, token_limit) for value in results]


def _run_generator_with_diagnostics(
    generator, prompts, config, diagnostic_path, phase, batch_start, max_new_tokens=None,
):
    token_limit = int(max_new_tokens if max_new_tokens is not None else config.get("max_new_tokens", 512))
    _append_diagnostic(diagnostic_path, {
        "event": "model_call_started", "ts": _utc_now(), "phase": phase,
        "batch_start": batch_start, "order_count": len(prompts),
        "max_new_tokens": token_limit,
    })
    try:
        return _run_generator(generator, prompts, config, max_new_tokens=token_limit)
    except Exception as error:
        _append_diagnostic(diagnostic_path, {
            "event": "model_call_failed", "ts": _utc_now(), "phase": phase,
            "batch_start": batch_start, "order_count": len(prompts),
            "exception_type": type(error).__name__,
        })
        raise


def load_transformers_generator(
    model_path, enable_thinking=False, attn_implementation="sdpa", cache_implementation="dynamic",
):
    """Load the server-local Qwen backend. Heavy imports stay inside this function."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Local model path not found: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype_name = os.environ.get("SEMANTIC_LLM_DTYPE", "").lower()
    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }.get(dtype_name, torch.float16 if torch.cuda.is_available() else torch.float32)
    model_kwargs = {
        "torch_dtype": dtype,
        "device_map": "auto",
        "local_files_only": True,
        "trust_remote_code": True,
    }
    if attn_implementation:
        model_kwargs["attn_implementation"] = str(attn_implementation)
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    def generate(prompts, max_new_tokens=512, temperature=0.0):
        messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
        input_texts = []
        for item in messages:
            try:
                input_texts.append(tokenizer.apply_chat_template(
                    item, tokenize=False, add_generation_prompt=True,
                    enable_thinking=bool(enable_thinking),
                ))
            except TypeError:
                input_texts.append(tokenizer.apply_chat_template(item, tokenize=False, add_generation_prompt=True))
        encoded = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        started = time.perf_counter()
        kwargs = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": float(temperature) > 0,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            "use_cache": True,
        }
        if cache_implementation and str(cache_implementation).lower() != "dynamic":
            kwargs["cache_implementation"] = str(cache_implementation)
        if kwargs["do_sample"]:
            kwargs["temperature"] = float(temperature)
        with torch.inference_mode():
            generated = model.generate(**encoded, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000
        continuation = generated[:, encoded.input_ids.shape[-1]:]
        texts = tokenizer.batch_decode(continuation, skip_special_tokens=True)
        input_lengths = encoded.attention_mask.sum(dim=1).tolist()
        rows = []
        for text, token_ids, input_tokens in zip(texts, continuation, input_lengths):
            output_tokens = int((token_ids != tokenizer.pad_token_id).sum().item())
            rows.append({
                "text": text,
                "input_tokens": int(input_tokens),
                "output_tokens": output_tokens,
                "finish_reason": "length" if output_tokens >= int(max_new_tokens) else "stop",
                "latency_ms": round(elapsed_ms / max(1, len(texts)), 3),
            })
        del encoded, generated, continuation
        return rows

    def empty_cache():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def memory_stats():
        if not torch.cuda.is_available():
            return {
                "current_allocated_gb": 0.0, "current_reserved_gb": 0.0,
                "peak_allocated_gb": 0.0, "peak_reserved_gb": 0.0,
            }
        scale = 1024 ** 3
        return {
            "current_allocated_gb": round(torch.cuda.memory_allocated() / scale, 3),
            "current_reserved_gb": round(torch.cuda.memory_reserved() / scale, 3),
            "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / scale, 3),
            "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / scale, 3),
        }

    generate.empty_cache = empty_cache
    generate.memory_stats = memory_stats
    generate.attn_implementation = str(attn_implementation or "default")
    generate.cache_implementation = str(cache_implementation or "dynamic")
    generate.prefix_caching = False
    generate.chunked_prefill = False
    generate.enforce_eager = False
    return generate


def load_vllm_generator(
    model_path, enable_thinking=False, gpu_memory_utilization=0.85,
    max_model_len=4096, max_num_seqs=64, enable_prefix_caching=False,
    enable_chunked_prefill=False, enforce_eager=False,
):
    """Load an offline vLLM backend with a safe V100 compatibility fallback."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Local model path not found: {model_path}")
    # This deployment targets V100: force the compatible engine and attention backend
    # before importing vLLM or initializing CUDA. Explicit environment values still win.
    os.environ.setdefault("VLLM_USE_V1", "0")
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "XFORMERS")
    import torch
    from vllm import LLM, SamplingParams

    engine = LLM(
        model=str(model_path),
        tokenizer=str(model_path),
        trust_remote_code=True,
        dtype="float16",
        tensor_parallel_size=1,
        gpu_memory_utilization=float(gpu_memory_utilization),
        max_model_len=int(max_model_len),
        max_num_seqs=int(max_num_seqs),
        enable_prefix_caching=bool(enable_prefix_caching),
        enable_chunked_prefill=bool(enable_chunked_prefill),
        enforce_eager=bool(enforce_eager),
    )
    tokenizer = engine.get_tokenizer()

    def generate(prompts, max_new_tokens=512, temperature=0.0):
        input_texts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            try:
                input_texts.append(tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=bool(enable_thinking),
                ))
            except TypeError:
                input_texts.append(tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                ))
        sampling = SamplingParams(
            temperature=float(temperature),
            max_tokens=int(max_new_tokens),
        )
        started = time.perf_counter()
        outputs = engine.generate(input_texts, sampling, use_tqdm=False)
        elapsed_ms = (time.perf_counter() - started) * 1000
        rows = []
        for output in outputs:
            choice = output.outputs[0]
            output_tokens = len(choice.token_ids)
            finish_reason = str(choice.finish_reason or (
                "length" if output_tokens >= int(max_new_tokens) else "stop"
            ))
            rows.append({
                "text": choice.text,
                "input_tokens": len(output.prompt_token_ids or []),
                "output_tokens": output_tokens,
                "finish_reason": finish_reason,
                "latency_ms": round(elapsed_ms / max(1, len(outputs)), 3),
            })
        return rows

    def empty_cache():
        # vLLM owns a paged KV-cache pool; emptying the PyTorch allocator here hurts throughput.
        return None

    def memory_stats():
        if not torch.cuda.is_available():
            return {
                "current_allocated_gb": 0.0, "current_reserved_gb": 0.0,
                "peak_allocated_gb": 0.0, "peak_reserved_gb": 0.0,
            }
        scale = 1024 ** 3
        return {
            "current_allocated_gb": round(torch.cuda.memory_allocated() / scale, 3),
            "current_reserved_gb": round(torch.cuda.memory_reserved() / scale, 3),
            "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / scale, 3),
            "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / scale, 3),
        }

    generate.empty_cache = empty_cache
    generate.memory_stats = memory_stats
    generate.attn_implementation = os.environ.get("VLLM_ATTENTION_BACKEND", "vllm-auto").lower()
    generate.cache_implementation = "paged"
    generate.prefix_caching = bool(enable_prefix_caching)
    generate.chunked_prefill = bool(enable_chunked_prefill)
    generate.enforce_eager = bool(enforce_eager)
    return generate


def _load_configured_generator(model_path, config):
    backend = str(config.get("backend", "transformers")).strip().lower()
    if backend == "transformers":
        return load_transformers_generator(
            model_path,
            enable_thinking=bool(config.get("enable_thinking", False)),
            attn_implementation=str(config.get("attn_implementation", "sdpa")),
            cache_implementation=str(config.get("cache_implementation", "dynamic")),
        )
    if backend == "vllm":
        return load_vllm_generator(
            model_path,
            enable_thinking=bool(config.get("enable_thinking", False)),
            gpu_memory_utilization=float(os.environ.get(
                "VLLM_GPU_MEMORY_UTILIZATION", config.get("vllm_gpu_memory_utilization", 0.85)
            )),
            max_model_len=int(os.environ.get(
                "VLLM_MAX_MODEL_LEN", config.get("vllm_max_model_len", 4096)
            )),
            max_num_seqs=int(os.environ.get(
                "VLLM_MAX_NUM_SEQS", config.get("vllm_max_num_seqs", 64)
            )),
            enable_prefix_caching=_environment_bool(
                "VLLM_ENABLE_PREFIX_CACHING", config.get("vllm_enable_prefix_caching", False)
            ),
            enable_chunked_prefill=_environment_bool(
                "VLLM_ENABLE_CHUNKED_PREFILL", config.get("vllm_enable_chunked_prefill", False)
            ),
            enforce_eager=_environment_bool(
                "VLLM_ENFORCE_EAGER", config.get("vllm_enforce_eager", False)
            ),
        )
    raise ValueError(f"unsupported_backend:{backend}")


def _quality_report(records, rejects):
    status_counts = Counter(row.get("validation", {}).get("status", "unknown") for row in records)
    warning_counts = Counter(
        warning for row in records for warning in row.get("validation", {}).get("warnings", [])
    )
    group_counts = {group: [len(row.get("entities", {}).get(group, [])) for row in records] for group in ENTITY_GROUPS}
    empty_entity_records = sum(
        not any(row.get("entities", {}).get(group, []) for group in ENTITY_GROUPS)
        for row in records
    )
    intent_records = sum(bool(row.get("discourse", {}).get("intents", [])) for row in records)
    return {
        "records": len(records),
        "rejects": len(rejects),
        "status_counts": dict(status_counts),
        "warning_counts": dict(warning_counts),
        "group_coverage": {
            group: sum(count > 0 for count in counts) / len(records) if records else 0.0
            for group, counts in group_counts.items()
        },
        "entity_count_distributions": {group: dict(Counter(counts)) for group, counts in group_counts.items()},
        "all_entities_empty_count": empty_entity_records,
        "all_entities_empty_rate": empty_entity_records / len(records) if records else 0.0,
        "intent_coverage": intent_records / len(records) if records else 0.0,
        "repair_attempted_count": sum(
            bool(row.get("validation", {}).get("repair_attempted")) for row in records
        ),
        "canonical_differs_from_surface": sum(
            item.get("canonical") != item.get("surface")
            for row in records for group in ENTITY_GROUPS for item in row.get("entities", {}).get(group, [])
        ),
        "intent_conflict_count": 0,
        "json_recovery_count": sum(
            count for warning, count in warning_counts.items()
            if warning.startswith("json_recovered_")
        ),
        "semantic_gap_counts": {
            warning.split(":", 1)[1]: count
            for warning, count in warning_counts.items()
            if warning.startswith("semantic_gap:")
        },
        "template_politeness_warning_count": warning_counts.get("template_politeness_as_satisfaction", 0),
    }


def run_semantic_extraction(
    input_path, output_path, rejects_path, run_report_path, quality_report_path,
    model_path, config, limit=None, resume=False, retry_rejected=False, generator=None,
    doc_ids=None, diagnostic_path=None, candidate_ledger_path=None,
    decision_ledger_path=None,
):
    """Run one primary request per order and at most one selective repair."""
    del retry_rejected  # Rejected rows are selected with doc_ids for an explicit safe rerun.
    started_wall = time.time()
    started_at = _utc_now()
    run_attempt_id = hashlib.sha256(
        f"{started_at}:{os.getpid()}:{time.time_ns()}".encode("utf-8")
    ).hexdigest()[:16]
    config = dict(config)
    model_id = str(config.get("model_id", "Qwen/Qwen3-4B"))
    prompt_version = str(config.get("prompt_version", "sag_semantic_v7"))
    backend = str(config.get("backend", "transformers")).strip().lower()
    output_path = Path(output_path)
    rejects_path = Path(rejects_path)
    partial_path = Path(str(output_path) + ".partial.jsonl")
    checkpoint_path = Path(str(output_path) + ".checkpoint.json")
    diagnostic_path = Path(diagnostic_path or (str(output_path) + ".diagnostics.jsonl"))
    candidate_ledger_path = Path(candidate_ledger_path) if candidate_ledger_path else None
    decision_ledger_path = Path(decision_ledger_path) if decision_ledger_path else None
    if not resume:
        diagnostic_path.unlink(missing_ok=True)
        if candidate_ledger_path:
            _write_jsonl(candidate_ledger_path, [])
        if decision_ledger_path:
            _write_jsonl(decision_ledger_path, [])
    _append_diagnostic(diagnostic_path, {
        "event": "run_initialized", "ts": _utc_now(),
        "schema": "privacy_safe_diagnostics_v1", "resume": bool(resume),
        "run_attempt_id": run_attempt_id,
    })
    stage_seconds = {
        "input_read": 0.0,
        "model_load": 0.0,
        "prompt_build": 0.0,
        "generation_wall": 0.0,
        "validation": 0.0,
        "artifact_write": 0.0,
    }
    stage_started = time.perf_counter()
    try:
        orders = read_work_orders(input_path, limit=limit)
        stage_seconds["input_read"] += time.perf_counter() - stage_started
    except Exception as error:
        _append_diagnostic(diagnostic_path, {
            "event": "run_failed", "ts": _utc_now(), "stage": "input_read",
            "exception_type": type(error).__name__,
        })
        raise
    if doc_ids is not None:
        selected = set(doc_ids)
        orders = [order for order in orders if order["doc_id"] in selected]

    existing = _read_jsonl(output_path) if resume and output_path.exists() else _read_jsonl(partial_path) if resume else []
    completed = {
        _identity(row.get("doc_id", ""), row.get("content_hash", ""),
                  row.get("model_run", {}).get("prompt_version", ""), row.get("model_run", {}).get("model", ""))
        for row in existing
    }
    pending = [
        order for order in orders
        if _identity(order["doc_id"], order["content_hash"], prompt_version, model_id) not in completed
    ]
    records = list(existing)
    rejects = []
    batch_size = max(1, int(config.get("batch_size", 8)))
    repair_batch_size = max(1, int(config.get("repair_batch_size", batch_size)))
    checkpoint_every = max(1, int(config.get("checkpoint_every", 50)))
    if resume and output_path.exists():
        _write_jsonl(partial_path, existing)
    elif not resume:
        _write_jsonl(partial_path, [])
    elif not partial_path.exists():
        _write_jsonl(partial_path, existing)
    persisted_record_count = len(records)
    processed_since_checkpoint = 0
    primary_requests = 0
    repair_requests = 0
    primary_batches = 0
    repair_batches = 0
    generation_rows = []
    pending_candidate_entries = []
    pending_decision_entries = []
    candidate_entries_before_run = _jsonl_record_count(candidate_ledger_path)
    decision_entries_before_run = _jsonl_record_count(decision_ledger_path)
    candidate_ledger_sequence = candidate_entries_before_run
    decision_ledger_sequence = decision_entries_before_run
    candidate_entries_written = 0
    decision_entries_written = 0
    ordered_pending = _bucket_orders(pending, config.get("length_bucket_boundaries", [600, 1400]))
    _append_diagnostic(diagnostic_path, {
        "event": "run_started", "ts": _utc_now(), "schema": "privacy_safe_diagnostics_v1",
        "orders_input": len(orders), "orders_pending": len(pending), "batch_size": batch_size,
        "repair_batch_size": repair_batch_size,
        "max_new_tokens": int(config.get("max_new_tokens", 512)),
        "repair_max_new_tokens": int(config.get("repair_max_new_tokens", config.get("max_new_tokens", 512))),
        "prompt_version": prompt_version, "model": model_id, "backend": backend,
        "validator_version": VALIDATOR_VERSION,
        "projection_version": PROJECTION_VERSION,
        "decoder_contract_version": str(
            config.get("decoder_contract_version", DECODER_CONTRACT_VERSION)
        ),
    })
    if generator is None:
        model_load_started = time.perf_counter()
        try:
            generator = _load_configured_generator(model_path, config)
            stage_seconds["model_load"] += time.perf_counter() - model_load_started
        except Exception as error:
            _append_diagnostic(diagnostic_path, {
                "event": "run_failed", "ts": _utc_now(), "stage": "model_load",
                "exception_type": type(error).__name__,
            })
            raise
    backend_memory_reader = getattr(generator, "memory_stats", None)
    backend_memory = backend_memory_reader() if callable(backend_memory_reader) else {}
    _append_diagnostic(diagnostic_path, {
        "event": "backend_ready", "ts": _utc_now(),
        "attn_implementation": str(getattr(generator, "attn_implementation", config.get("attn_implementation", "unknown"))),
        "cache_implementation": str(getattr(generator, "cache_implementation", config.get("cache_implementation", "unknown"))),
        "prefix_caching": bool(getattr(generator, "prefix_caching", False)),
        "chunked_prefill": bool(getattr(generator, "chunked_prefill", False)),
        "enforce_eager": bool(getattr(generator, "enforce_eager", False)),
        **backend_memory,
    })

    repair_queue = []

    def queue_private_audit(order, phase, candidate, parse_warnings, generation, final_semantic, validation, trace):
        nonlocal candidate_ledger_sequence, decision_ledger_sequence
        if candidate_ledger_path:
            candidate_ledger_sequence += 1
            pending_candidate_entries.append(
                _private_candidate_entry(
                    order, phase, candidate, parse_warnings, generation, config,
                    run_attempt_id, candidate_ledger_sequence,
                )
            )
        if decision_ledger_path:
            decision_ledger_sequence += 1
            pending_decision_entries.append(
                _private_decision_entry(
                    order, phase, final_semantic, validation, trace,
                    run_attempt_id, decision_ledger_sequence,
                )
            )

    def flush_private_audit():
        nonlocal candidate_entries_written, decision_entries_written
        write_started = time.perf_counter()
        if candidate_ledger_path and pending_candidate_entries:
            _append_jsonl(candidate_ledger_path, pending_candidate_entries)
            candidate_entries_written += len(pending_candidate_entries)
            pending_candidate_entries.clear()
        if decision_ledger_path and pending_decision_entries:
            _append_jsonl(decision_ledger_path, pending_decision_entries)
            decision_entries_written += len(pending_decision_entries)
            pending_decision_entries.clear()
        stage_seconds["artifact_write"] += time.perf_counter() - write_started

    def flush_repairs(force=False):
        nonlocal repair_requests, repair_batches
        while len(repair_queue) >= repair_batch_size or (force and repair_queue):
            queue_size = min(repair_batch_size, len(repair_queue))
            queued = repair_queue[:queue_size]
            del repair_queue[:queue_size]
            repair_batch_start = repair_requests
            prompt_started = time.perf_counter()
            repair_prompts = [
                build_repair_prompt(order, primary["text"], validation["warnings"], config)
                for order, primary, _semantic, validation in queued
            ]
            stage_seconds["prompt_build"] += time.perf_counter() - prompt_started
            generation_started = time.perf_counter()
            repair_results = _run_generator_with_diagnostics(
                generator, repair_prompts, config, diagnostic_path, "repair", repair_batch_start,
                max_new_tokens=int(config.get("repair_max_new_tokens", config.get("max_new_tokens", 512)))
            )
            stage_seconds["generation_wall"] += time.perf_counter() - generation_started
            repair_requests += len(queued)
            repair_batches += 1
            generation_rows.extend(repair_results)
            for (order, primary, _semantic, first_validation), repaired in zip(queued, repair_results):
                validation_started = time.perf_counter()
                candidate, parse_warnings = parse_semantic_json(
                    repaired["text"], preserve_overflow=True
                )
                semantic, validation, trace = _validate_with_sanitation(order, candidate, parse_warnings)
                stage_seconds["validation"] += time.perf_counter() - validation_started
                queue_private_audit(
                    order, "repair", candidate, parse_warnings, repaired,
                    semantic, validation, trace,
                )
                _append_diagnostic(diagnostic_path, {
                    "event": "model_result", "ts": _utc_now(), "phase": "repair",
                    "order_ref": _diagnostic_identity(order),
                    "source_chars": len(order.get("chunk_text", "")),
                    "input_tokens": repaired["input_tokens"], "output_tokens": repaired["output_tokens"],
                    "finish_reason": repaired["finish_reason"], "latency_ms": repaired["latency_ms"],
                    **trace, "semantic_counts": _semantic_counts(semantic),
                    "repair_requested": False,
                })
                if validation["status"] in {"accepted", "accepted_with_warnings"}:
                    records.append(_record(order, semantic, validation, repaired, config, True))
                else:
                    final_warnings = list(validation["warnings"])
                    if "repair_failed" not in final_warnings:
                        final_warnings.append("repair_failed")
                    rejects.append({
                        "doc_id": order["doc_id"], "content_hash": order["content_hash"],
                        "validation": {**validation, "status": "rejected", "warnings": final_warnings},
                        "primary_validation": first_validation,
                        "primary_response": primary["text"], "repair_response": repaired["text"],
                    })

    for batch_start in range(0, len(ordered_pending), batch_size):
        indexed_batch = ordered_pending[batch_start:batch_start + batch_size]
        batch = [order for _index, order in indexed_batch]
        prompt_started = time.perf_counter()
        primary_prompts = [build_semantic_prompt(order, config) for order in batch]
        stage_seconds["prompt_build"] += time.perf_counter() - prompt_started
        generation_started = time.perf_counter()
        generated = _run_generator_with_diagnostics(
            generator, primary_prompts, config,
            diagnostic_path, "primary", batch_start,
        )
        stage_seconds["generation_wall"] += time.perf_counter() - generation_started
        primary_requests += len(batch)
        primary_batches += 1
        generation_rows.extend(generated)
        for order, result in zip(batch, generated):
            validation_started = time.perf_counter()
            candidate, parse_warnings = parse_semantic_json(
                result["text"], preserve_overflow=True
            )
            semantic, validation, trace = _validate_with_sanitation(order, candidate, parse_warnings)
            stage_seconds["validation"] += time.perf_counter() - validation_started
            queue_private_audit(
                order, "primary", candidate, parse_warnings, result,
                semantic, validation, trace,
            )
            repair_requested = validation["status"] == "repair_required" and int(config.get("max_repairs_per_order", 1)) > 0
            _append_diagnostic(diagnostic_path, {
                "event": "model_result", "ts": _utc_now(), "phase": "primary",
                "order_ref": _diagnostic_identity(order),
                "source_chars": len(order.get("chunk_text", "")),
                "input_tokens": result["input_tokens"], "output_tokens": result["output_tokens"],
                "finish_reason": result["finish_reason"], "latency_ms": result["latency_ms"],
                **trace, "semantic_counts": _semantic_counts(semantic),
                "repair_requested": repair_requested,
            })
            if repair_requested:
                repair_queue.append((order, result, semantic, validation))
            elif validation["status"] in {"accepted", "accepted_with_warnings"}:
                records.append(_record(order, semantic, validation, result, config, False))
            else:
                rejects.append({
                    "doc_id": order["doc_id"], "content_hash": order["content_hash"],
                    "validation": validation, "primary_response": result["text"],
                })

        flush_repairs(force=batch_start + batch_size >= len(ordered_pending))

        if bool(config.get("empty_cache_between_batches", True)):
            cleanup = getattr(generator, "empty_cache", None)
            if callable(cleanup):
                cleanup()
        memory_reader = getattr(generator, "memory_stats", None)
        if callable(memory_reader):
            _append_diagnostic(diagnostic_path, {
                "event": "batch_memory", "ts": _utc_now(),
                "batch_start": batch_start, **memory_reader(),
            })

        processed_since_checkpoint += len(batch)
        if processed_since_checkpoint >= checkpoint_every or batch_start + batch_size >= len(ordered_pending):
            # Persist private attempts before marking semantic records complete.
            # A crash may cause a later duplicate attempt in the append-only
            # ledger, but it must never leave a checkpointed record unaudited.
            flush_private_audit()
            write_started = time.perf_counter()
            _append_jsonl(partial_path, records[persisted_record_count:])
            stage_seconds["artifact_write"] += time.perf_counter() - write_started
            persisted_record_count = len(records)
            processed_since_checkpoint = 0
            _atomic_json(checkpoint_path, {
                "completed": len(records), "rejects": len(rejects),
                "prompt_version": prompt_version, "model": model_id, "updated_at": _utc_now(),
            })

    flush_private_audit()
    write_started = time.perf_counter()
    _append_jsonl(partial_path, records[persisted_record_count:])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(partial_path, output_path)
    _write_jsonl(rejects_path, rejects)
    stage_seconds["artifact_write"] += time.perf_counter() - write_started
    elapsed = time.time() - started_wall
    finish_reasons = Counter(row["finish_reason"] for row in generation_rows)
    input_tokens = [row["input_tokens"] for row in generation_rows]
    output_tokens = [row["output_tokens"] for row in generation_rows]
    memory_reader = getattr(generator, "memory_stats", None)
    gpu_memory = memory_reader() if callable(memory_reader) else {
        "current_allocated_gb": 0.0, "current_reserved_gb": 0.0,
        "peak_allocated_gb": 0.0, "peak_reserved_gb": 0.0,
    }
    run_report = {
        "model": model_id,
        "run_attempt_id": run_attempt_id,
        "prompt_version": prompt_version,
        "validator_version": VALIDATOR_VERSION,
        "projection_version": PROJECTION_VERSION,
        "decoder_contract_version": str(
            config.get("decoder_contract_version", DECODER_CONTRACT_VERSION)
        ),
        "backend": backend,
        "dtype": os.environ.get("SEMANTIC_LLM_DTYPE", "auto"),
        "batch_size": batch_size,
        "repair_batch_size": repair_batch_size,
        "attn_implementation": str(getattr(generator, "attn_implementation", config.get("attn_implementation", "unknown"))),
        "cache_implementation": str(getattr(generator, "cache_implementation", config.get("cache_implementation", "unknown"))),
        "prefix_caching": bool(getattr(generator, "prefix_caching", False)),
        "chunked_prefill": bool(getattr(generator, "chunked_prefill", False)),
        "enforce_eager": bool(getattr(generator, "enforce_eager", False)),
        "config_hash": _config_hash(config),
        "orders_input": len(orders),
        "orders_processed": len(pending),
        "records_written": len(records),
        "rejects_written": len(rejects),
        "primary_requests": primary_requests,
        "repair_requests": repair_requests,
        "primary_batches": primary_batches,
        "repair_batches": repair_batches,
        "input_tokens_total": sum(input_tokens),
        "output_tokens_total": sum(output_tokens),
        "input_tokens_p50": _percentile(input_tokens, 0.5),
        "input_tokens_p95": _percentile(input_tokens, 0.95),
        "output_tokens_p50": _percentile(output_tokens, 0.5),
        "output_tokens_p95": _percentile(output_tokens, 0.95),
        "finish_reason_counts": dict(finish_reasons),
        "truncation_count": finish_reasons.get("length", 0),
        "elapsed_seconds": round(elapsed, 3),
        "orders_per_second": round(len(pending) / elapsed, 4) if elapsed else 0.0,
        "model_latency_seconds": round(sum(row["latency_ms"] for row in generation_rows) / 1000, 3),
        "stage_seconds": {
            name: round(seconds, 3) for name, seconds in stage_seconds.items()
        },
        "unaccounted_seconds": round(max(0.0, elapsed - sum(stage_seconds.values())), 3),
        "output_tokens_per_second": round(
            sum(output_tokens) / (sum(row["latency_ms"] for row in generation_rows) / 1000), 3
        ) if generation_rows and sum(row["latency_ms"] for row in generation_rows) else 0.0,
        "started_at": started_at,
        "ended_at": _utc_now(),
        "checkpoint": str(checkpoint_path),
        "resumed_records": len(existing),
        "gpu_current_allocated_gb": gpu_memory["current_allocated_gb"],
        "gpu_current_reserved_gb": gpu_memory["current_reserved_gb"],
        "gpu_peak_allocated_gb": gpu_memory["peak_allocated_gb"],
        "gpu_peak_reserved_gb": gpu_memory["peak_reserved_gb"],
        "empty_cache_between_batches": bool(config.get("empty_cache_between_batches", True)),
        "diagnostic_log": str(diagnostic_path),
        "private_candidate_ledger": str(candidate_ledger_path) if candidate_ledger_path else "",
        "private_decision_ledger": str(decision_ledger_path) if decision_ledger_path else "",
        "candidate_entries_before_run": candidate_entries_before_run,
        "decision_entries_before_run": decision_entries_before_run,
        "candidate_entries_written": candidate_entries_written,
        "decision_entries_written": decision_entries_written,
    }
    _append_diagnostic(diagnostic_path, {
        "event": "run_completed", "ts": _utc_now(),
        "records_written": len(records), "rejects_written": len(rejects),
        "primary_requests": primary_requests, "repair_requests": repair_requests,
        "primary_batches": primary_batches, "repair_batches": repair_batches,
        "elapsed_seconds": round(elapsed, 3),
        "stage_seconds": {
            name: round(seconds, 3) for name, seconds in stage_seconds.items()
        },
        "run_attempt_id": run_attempt_id,
        "candidate_entries_before_run": candidate_entries_before_run,
        "decision_entries_before_run": decision_entries_before_run,
        "candidate_entries_written": candidate_entries_written,
        "decision_entries_written": decision_entries_written,
        **gpu_memory,
    })
    _atomic_json(run_report_path, run_report)
    _atomic_json(quality_report_path, _quality_report(records, rejects))
    return run_report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Extract auditable semantics from desensitized work-order JSONL.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rejects", required=True)
    parser.add_argument("--run-report", required=True)
    parser.add_argument("--quality-report", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--repair-batch-size", type=int)
    parser.add_argument("--backend", choices=("transformers", "vllm"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-rejected", action="store_true")
    parser.add_argument("--doc-id-file")
    parser.add_argument("--diagnostic-log", default="")
    parser.add_argument(
        "--candidate-ledger", default="",
        help="Optional private pre-sanitation candidate JSONL; contains evidence and must not be shared.",
    )
    parser.add_argument(
        "--decision-ledger", default="",
        help="Optional private validator decision JSONL; must not be shared.",
    )
    parser.add_argument("--allow-raw-tsv", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if Path(args.input).suffix.lower() == ".tsv":
        raise SystemExit("Semantic extraction requires desensitized multiview JSONL; raw TSV remains on the legacy controlled path.")
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.batch_size is not None:
        if args.batch_size < 1:
            raise SystemExit("--batch-size must be at least 1")
        config["batch_size"] = args.batch_size
    if args.repair_batch_size is not None:
        if args.repair_batch_size < 1:
            raise SystemExit("--repair-batch-size must be at least 1")
        config["repair_batch_size"] = args.repair_batch_size
    if args.backend is not None:
        config["backend"] = args.backend
    doc_ids = None
    if args.doc_id_file:
        doc_ids = [line.strip() for line in Path(args.doc_id_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    report = run_semantic_extraction(
        args.input, args.output, args.rejects, args.run_report, args.quality_report,
        args.model_path, config, limit=args.limit, resume=args.resume,
        retry_rejected=args.retry_rejected, doc_ids=doc_ids,
        diagnostic_path=args.diagnostic_log or None,
        candidate_ledger_path=args.candidate_ledger or None,
        decision_ledger_path=args.decision_ledger or None,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
