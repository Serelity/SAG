"""Privacy-safe integrity checks for server semantic extraction artifacts."""

import argparse
import hashlib
import json
import re
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
    output_schemas = Counter()
    validator_versions = Counter()
    projection_versions = Counter()
    for row in semantic:
        model_run = row.get("model_run") if isinstance(row.get("model_run"), dict) else {}
        validation = row.get("validation") if isinstance(row.get("validation"), dict) else {}
        identities.append((row.get("doc_id"), row.get("content_hash"), model_run.get("prompt_version"), model_run.get("model")))
        statuses[str(validation.get("status", "unknown"))] += 1
        warnings.update(str(value) for value in validation.get("warnings", []))
        output_schemas[str(row.get("output_schema_version", "legacy"))] += 1
        validator_versions[str(validation.get("validator_version", "unknown"))] += 1
        artifact_versions = row.get("artifact_versions") if isinstance(row.get("artifact_versions"), dict) else {}
        projection_versions[str(artifact_versions.get("projection", "unknown"))] += 1
        repairs += int(bool(validation.get("repair_attempted")))
        finish_reasons[str(model_run.get("finish_reason", "unknown"))] += 1
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate_semantic_identity")
    if len(output_schemas) > 1 or len(validator_versions) > 1 or len(projection_versions) > 1:
        raise ValueError("mixed_semantic_contract_versions")
    if semantic:
        output_schema = next(iter(output_schemas))
        validator_version = next(iter(validator_versions))
        projection_version = next(iter(projection_versions))
        if str(run.get("output_schema_version", "legacy")) != output_schema:
            raise ValueError("output_schema_version_mismatch")
        if str(run.get("validator_version", "unknown")) != validator_version:
            raise ValueError("validator_version_mismatch")
        if str(run.get("projection_version", "unknown")) != projection_version:
            raise ValueError("projection_version_mismatch")
    expected = run.get("records_written")
    if expected is not None and int(expected) != len(semantic):
        raise ValueError(f"record_count_mismatch:{expected}:{len(semantic)}")
    if run.get("rejects_written") is not None and int(run["rejects_written"]) != len(rejects):
        raise ValueError("reject_count_mismatch")
    manifest_check = {"enabled": bool(run.get("identity_manifest_enabled"))}
    if manifest_check["enabled"]:
        manifest_records = int(run.get("identity_manifest_records", -1))
        orders_input = int(run.get("orders_input", -1))
        scanned = int(run.get("input_records_scanned", -1))
        ignored_invalid = int(run.get("ignored_invalid_input_records", -1))
        manifest_sha256 = str(run.get("identity_manifest_sha256", ""))
        if manifest_records < 1 or manifest_records != orders_input:
            raise ValueError("identity_manifest_count_mismatch")
        if scanned < orders_input or ignored_invalid < 0:
            raise ValueError("identity_manifest_scan_counts_invalid")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", manifest_sha256):
            raise ValueError("identity_manifest_hash_invalid")
        if len(semantic) + len(rejects) != orders_input:
            raise ValueError("identity_manifest_output_count_mismatch")
        manifest_check.update({
            "records": manifest_records,
            "input_records_scanned": scanned,
            "ignored_invalid_input_records": ignored_invalid,
            "sha256": manifest_sha256,
        })

    optional_hashes = {}
    for name in ("candidate_ledger", "decision_ledger", "diagnostics"):
        path = getattr(args, name, "")
        if path:
            optional_hashes[name] = digest(path)

    result = {
        "semantic_records": len(semantic),
        "reject_records": len(rejects),
        "status_counts": dict(statuses),
        "warning_counts": dict(warnings),
        "repair_attempts": repairs,
        "finish_reason_counts": dict(finish_reasons),
        "semantic_contract": {
            "output_schema_versions": dict(output_schemas),
            "validator_versions": dict(validator_versions),
            "projection_versions": dict(projection_versions),
        },
        "identity_manifest": manifest_check,
        "hashes": {
            "semantic": digest(args.semantic),
            "rejects": digest(args.rejects),
            "run_report": digest(args.run_report),
            "quality_report": digest(args.quality_report),
            **optional_hashes,
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
    parser.add_argument("--candidate-ledger", default="")
    parser.add_argument("--decision-ledger", default="")
    parser.add_argument("--diagnostics", default="")
    return parser.parse_args(argv)


if __name__ == "__main__":
    check(parse_args())
