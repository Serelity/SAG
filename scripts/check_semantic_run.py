"""Privacy-safe integrity checks for server semantic extraction artifacts."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


FORBIDDEN_REPORT_KEYS = {"prompt", "prompts", "raw_response", "primary_response", "repair_response", "chunk_text", "evidence"}


def digest(path):
    value = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return "sha256:" + value


def read_jsonl(path):
    path = Path(path)
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError(f"missing_final_newline:{path}")
    rows = []
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid_json:{path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"invalid_record:{path}:{line_number}")
        rows.append(value)
    return rows


def walk_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk_keys(item)


def check(args):
    semantic = read_jsonl(args.semantic)
    rejects = read_jsonl(args.rejects)
    run = json.loads(Path(args.run_report).read_text(encoding="utf-8"))
    quality = json.loads(Path(args.quality_report).read_text(encoding="utf-8"))
    leaked_keys = sorted((set(walk_keys(run)) | set(walk_keys(quality))) & FORBIDDEN_REPORT_KEYS)
    if leaked_keys:
        raise ValueError("report_contains_sensitive_keys:" + ",".join(leaked_keys))

    identities = []
    statuses = Counter()
    warnings = Counter()
    repairs = 0
    finish_reasons = Counter()
    for row in semantic:
        model_run = row.get("model_run") if isinstance(row.get("model_run"), dict) else {}
        validation = row.get("validation") if isinstance(row.get("validation"), dict) else {}
        identities.append((row.get("doc_id"), row.get("content_hash"), model_run.get("prompt_version"), model_run.get("model")))
        statuses[str(validation.get("status", "unknown"))] += 1
        warnings.update(str(value) for value in validation.get("warnings", []))
        repairs += int(bool(validation.get("repair_attempted")))
        finish_reasons[str(model_run.get("finish_reason", "unknown"))] += 1
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate_semantic_identity")
    expected = run.get("records_written")
    if expected is not None and int(expected) != len(semantic):
        raise ValueError(f"record_count_mismatch:{expected}:{len(semantic)}")
    if run.get("rejects_written") is not None and int(run["rejects_written"]) != len(rejects):
        raise ValueError("reject_count_mismatch")

    result = {
        "semantic_records": len(semantic),
        "reject_records": len(rejects),
        "status_counts": dict(statuses),
        "warning_counts": dict(warnings),
        "repair_attempts": repairs,
        "finish_reason_counts": dict(finish_reasons),
        "hashes": {
            "semantic": digest(args.semantic),
            "rejects": digest(args.rejects),
            "run_report": digest(args.run_report),
            "quality_report": digest(args.quality_report),
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Check semantic run counts and hashes without printing work-order text.")
    parser.add_argument("--semantic", required=True)
    parser.add_argument("--rejects", required=True)
    parser.add_argument("--run-report", required=True)
    parser.add_argument("--quality-report", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    check(parse_args())
