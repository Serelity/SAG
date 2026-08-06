"""Summarize privacy-safe semantic extraction diagnostics."""

import argparse
import json
from collections import Counter
from pathlib import Path


def summarize(path):
    events = Counter()
    phases = Counter()
    finish_reasons = Counter()
    before_statuses = Counter()
    after_statuses = Counter()
    before_warnings = Counter()
    after_warnings = Counter()
    sanitation = Counter()
    entity_counts = Counter()
    repairs = 0
    input_tokens = 0
    output_tokens = 0
    latency_ms = 0.0
    memory_rows = []
    failures = Counter()
    completed = {}
    last_event = ""

    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            event = str(row.get("event", "unknown"))
            last_event = event
            events[event] += 1
            if event == "model_result":
                phase = str(row.get("phase", "unknown"))
                phases[phase] += 1
                repairs += int(bool(row.get("repair_requested")))
                finish_reasons[str(row.get("finish_reason", "unknown"))] += 1
                input_tokens += int(row.get("input_tokens") or 0)
                output_tokens += int(row.get("output_tokens") or 0)
                latency_ms += float(row.get("latency_ms") or 0)
                before = row.get("validation_before") if isinstance(row.get("validation_before"), dict) else {}
                after = row.get("validation_after") if isinstance(row.get("validation_after"), dict) else {}
                before_statuses[str(before.get("status", "unknown"))] += 1
                after_statuses[str(after.get("status", "unknown"))] += 1
                before_warnings.update(str(value) for value in before.get("warnings", []))
                after_warnings.update(str(value) for value in after.get("warnings", []))
                sanitation.update(str(value) for value in row.get("sanitation_warnings", []))
                counts = row.get("semantic_counts") if isinstance(row.get("semantic_counts"), dict) else {}
                entity_counts.update({key: int(value) for key, value in (counts.get("entities") or {}).items()})
            elif event == "batch_memory":
                memory_rows.append({
                    key: row.get(key, 0)
                    for key in ("current_allocated_gb", "current_reserved_gb", "peak_allocated_gb", "peak_reserved_gb")
                })
            elif event in {"model_call_failed", "run_failed"}:
                failure_key = ":".join(filter(None, (
                    str(row.get("stage", "")), str(row.get("phase", "")),
                    str(row.get("exception_type", "unknown")),
                )))
                failures[failure_key] += 1
            elif event == "run_completed":
                completed = {
                    key: row.get(key)
                    for key in (
                        "records_written", "rejects_written", "primary_requests", "repair_requests",
                        "primary_batches", "repair_batches", "elapsed_seconds",
                        "stage_seconds", "run_attempt_id",
                        "candidate_entries_before_run", "decision_entries_before_run",
                        "candidate_entries_written", "decision_entries_written",
                        "current_allocated_gb", "current_reserved_gb",
                        "peak_allocated_gb", "peak_reserved_gb",
                    )
                    if key in row
                }

    result = {
        "event_counts": dict(events),
        "phase_counts": dict(phases),
        "finish_reason_counts": dict(finish_reasons),
        "validation_before_statuses": dict(before_statuses),
        "validation_after_statuses": dict(after_statuses),
        "validation_before_warnings": dict(before_warnings),
        "validation_after_warnings": dict(after_warnings),
        "sanitation_actions": dict(sanitation),
        "entity_counts_after_sanitation": dict(entity_counts),
        "repair_requested_by_primary": repairs,
        "input_tokens_total": input_tokens,
        "output_tokens_total": output_tokens,
        "model_latency_ms_total": round(latency_ms, 3),
        "memory_by_batch": memory_rows,
        "failure_counts": dict(failures),
        "last_event": last_event,
        "run_completed": completed,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Summarize privacy-safe semantic extraction diagnostics.")
    parser.add_argument("--input", required=True, help="Diagnostics JSONL path")
    return parser.parse_args(argv)


if __name__ == "__main__":
    summarize(parse_args().input)
