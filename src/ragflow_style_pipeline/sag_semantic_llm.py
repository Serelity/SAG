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
from ragflow_style_pipeline.sag_semantic_schema import ENTITY_GROUPS, parse_semantic_json
from ragflow_style_pipeline.sag_semantic_validation import validate_semantic_output
from ragflow_style_pipeline.work_order_input import read_work_orders


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _config_hash(config):
    encoded = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


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
            "status": validation["status"],
            "warnings": validation["warnings"],
            "repair_fields": validation["repair_fields"],
            "repair_attempted": bool(repair_attempted),
        },
        "model_run": {
            "model": str(config.get("model_id", "Qwen/Qwen3-4B")),
            "prompt_version": str(config.get("prompt_version", "sag_semantic_v2")),
            "backend": str(config.get("backend", "transformers")),
            "input_tokens": generation["input_tokens"],
            "output_tokens": generation["output_tokens"],
            "finish_reason": generation["finish_reason"],
            "latency_ms": generation["latency_ms"],
        },
    }


def _run_generator(generator, prompts, config):
    results = generator(
        prompts,
        max_new_tokens=int(config.get("max_new_tokens", 512)),
        temperature=float(config.get("temperature", 0.0)),
    )
    if isinstance(results, (str, dict)):
        results = [results]
    results = list(results)
    if len(results) != len(prompts):
        raise RuntimeError(f"generator_result_count:{len(results)}:{len(prompts)}")
    return [_normalize_generation_result(value, int(config.get("max_new_tokens", 512))) for value in results]


def load_transformers_generator(model_path, enable_thinking=False):
    """Load the server-local Qwen backend.  Heavy imports stay inside this function."""
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
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True,
    )

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
        }
        if kwargs["do_sample"]:
            kwargs["temperature"] = float(temperature)
        with torch.inference_mode():
            generated = model.generate(**encoded, **kwargs)
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
        return rows

    return generate


def _quality_report(records, rejects):
    status_counts = Counter(row.get("validation", {}).get("status", "unknown") for row in records)
    warning_counts = Counter(
        warning for row in records for warning in row.get("validation", {}).get("warnings", [])
    )
    group_counts = {group: [len(row.get("entities", {}).get(group, [])) for row in records] for group in ENTITY_GROUPS}
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
        "canonical_differs_from_surface": sum(
            item.get("canonical") != item.get("surface")
            for row in records for group in ENTITY_GROUPS for item in row.get("entities", {}).get(group, [])
        ),
        "intent_conflict_count": 0,
        "template_politeness_warning_count": warning_counts.get("template_politeness_as_satisfaction", 0),
    }


def run_semantic_extraction(
    input_path, output_path, rejects_path, run_report_path, quality_report_path,
    model_path, config, limit=None, resume=False, retry_rejected=False, generator=None,
    doc_ids=None,
):
    """Run one primary request per order and at most one selective repair."""
    del retry_rejected  # Rejected rows are selected with doc_ids for an explicit safe rerun.
    started_wall = time.time()
    started_at = _utc_now()
    config = dict(config)
    model_id = str(config.get("model_id", "Qwen/Qwen3-4B"))
    prompt_version = str(config.get("prompt_version", "sag_semantic_v2"))
    output_path = Path(output_path)
    rejects_path = Path(rejects_path)
    partial_path = Path(str(output_path) + ".partial.jsonl")
    checkpoint_path = Path(str(output_path) + ".checkpoint.json")
    orders = read_work_orders(input_path, limit=limit)
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
    if generator is None:
        generator = load_transformers_generator(model_path, enable_thinking=bool(config.get("enable_thinking", False)))

    records = list(existing)
    rejects = []
    primary_requests = 0
    repair_requests = 0
    generation_rows = []
    batch_size = max(1, int(config.get("batch_size", 8)))
    checkpoint_every = max(1, int(config.get("checkpoint_every", 50)))
    ordered_pending = _bucket_orders(pending, config.get("length_bucket_boundaries", [600, 1400]))

    for batch_start in range(0, len(ordered_pending), batch_size):
        indexed_batch = ordered_pending[batch_start:batch_start + batch_size]
        batch = [order for _index, order in indexed_batch]
        generated = _run_generator(generator, [build_semantic_prompt(order, config) for order in batch], config)
        primary_requests += len(batch)
        generation_rows.extend(generated)
        repair_queue = []
        for order, result in zip(batch, generated):
            semantic, parse_warnings = parse_semantic_json(result["text"])
            validation = validate_semantic_output(order, semantic, parse_warnings)
            if validation["status"] == "repair_required" and int(config.get("max_repairs_per_order", 1)) > 0:
                repair_queue.append((order, result, semantic, validation))
            elif validation["status"] in {"accepted", "accepted_with_warnings"}:
                records.append(_record(order, semantic, validation, result, config, False))
            else:
                rejects.append({
                    "doc_id": order["doc_id"], "content_hash": order["content_hash"],
                    "validation": validation, "primary_response": result["text"],
                })

        if repair_queue:
            repair_results = _run_generator(generator, [
                build_repair_prompt(order, primary["text"], validation["warnings"], config)
                for order, primary, _semantic, validation in repair_queue
            ], config)
            repair_requests += len(repair_queue)
            generation_rows.extend(repair_results)
            for (order, primary, _semantic, first_validation), repaired in zip(repair_queue, repair_results):
                semantic, parse_warnings = parse_semantic_json(repaired["text"])
                validation = validate_semantic_output(order, semantic, parse_warnings)
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

        if len(records) % checkpoint_every == 0 or batch_start + batch_size >= len(ordered_pending):
            _write_jsonl(partial_path, records)
            _atomic_json(checkpoint_path, {
                "completed": len(records), "rejects": len(rejects),
                "prompt_version": prompt_version, "model": model_id, "updated_at": _utc_now(),
            })

    _write_jsonl(partial_path, records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(partial_path, output_path)
    _write_jsonl(rejects_path, rejects)
    elapsed = time.time() - started_wall
    finish_reasons = Counter(row["finish_reason"] for row in generation_rows)
    input_tokens = [row["input_tokens"] for row in generation_rows]
    output_tokens = [row["output_tokens"] for row in generation_rows]
    run_report = {
        "model": model_id,
        "backend": str(config.get("backend", "transformers")),
        "dtype": os.environ.get("SEMANTIC_LLM_DTYPE", "auto"),
        "config_hash": _config_hash(config),
        "orders_input": len(orders),
        "orders_processed": len(pending),
        "records_written": len(records),
        "rejects_written": len(rejects),
        "primary_requests": primary_requests,
        "repair_requests": repair_requests,
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
        "started_at": started_at,
        "ended_at": _utc_now(),
        "checkpoint": str(checkpoint_path),
        "resumed_records": len(existing),
    }
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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-rejected", action="store_true")
    parser.add_argument("--doc-id-file")
    parser.add_argument("--allow-raw-tsv", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if Path(args.input).suffix.lower() == ".tsv":
        raise SystemExit("Semantic extraction requires desensitized multiview JSONL; raw TSV remains on the legacy controlled path.")
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    doc_ids = None
    if args.doc_id_file:
        doc_ids = [line.strip() for line in Path(args.doc_id_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    report = run_semantic_extraction(
        args.input, args.output, args.rejects, args.run_report, args.quality_report,
        args.model_path, config, limit=args.limit, resume=args.resume,
        retry_rejected=args.retry_rejected, doc_ids=doc_ids,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
