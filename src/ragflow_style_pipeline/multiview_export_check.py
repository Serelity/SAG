"""Aggregate-only integrity checker for desensitized multiview JSONL exports."""

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from ragflow_style_pipeline.sag_semantic_versions import MULTIVIEW_INPUT_VERSION
from ragflow_style_pipeline.work_order_input import CLEAN_FIELDS, WorkOrderInputError, normalize_work_order

_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}")


def _file_sha256_and_size(path):
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return "sha256:" + digest.hexdigest(), size


def check_multiview_export(input_path, quality_report_path=""):
    """Validate schema, identities, hashes and quality provenance without text output."""
    input_path = Path(input_path)
    counts = Counter()
    clean_field_presence = Counter()
    doc_ids = set()
    identities = set()

    with input_path.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                counts["blank_lines"] += 1
                continue
            counts["records"] += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                counts["invalid_json"] += 1
                continue
            if not isinstance(row, dict):
                counts["record_not_object"] += 1
                continue
            counts["invalid_schema"] += row.get("input_schema") != MULTIVIEW_INPUT_VERSION
            counts["missing_or_invalid_clean_field"] += any(
                field not in row or not isinstance(row.get(field), str)
                for field in CLEAN_FIELDS
            )
            for field in CLEAN_FIELDS:
                clean_field_presence[field] += bool(row.get(field))
            metadata = row.get("metadata")
            if not isinstance(metadata, dict):
                counts["metadata_not_object"] += 1
                metadata = {}
            counts["metadata_order_id_present"] += "order_id" in metadata
            doc_id = row.get("doc_id")
            supplied_hash = row.get("content_hash")
            if not isinstance(doc_id, str) or not doc_id:
                counts["missing_doc_id"] += 1
            elif doc_id in doc_ids:
                counts["duplicate_doc_id"] += 1
            if not isinstance(supplied_hash, str) or not _SHA256_RE.fullmatch(supplied_hash):
                counts["invalid_content_hash"] += 1
            identity = (doc_id, supplied_hash)
            if identity in identities:
                counts["duplicate_identity"] += 1
            doc_ids.add(doc_id)
            identities.add(identity)
            try:
                normalized = normalize_work_order(row)
            except WorkOrderInputError:
                counts["normalization_error"] += 1
                continue
            counts["content_hash_mismatch"] += supplied_hash != normalized["content_hash"]

    output_sha256, output_bytes = _file_sha256_and_size(input_path)
    quality_errors = Counter()
    quality = {}
    if quality_report_path:
        try:
            quality = json.loads(Path(quality_report_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            quality_errors["quality_report_unreadable"] += 1
        if quality:
            quality_errors["quality_schema_mismatch"] += quality.get("schema") != MULTIVIEW_INPUT_VERSION
            quality_errors["quality_records_mismatch"] += quality.get("documents_written") != counts["records"]
            quality_errors["quality_bytes_mismatch"] += quality.get("output_bytes") != output_bytes
            quality_errors["quality_sha256_mismatch"] += quality.get("output_sha256") != output_sha256
            quality_errors["quality_clean_fields_mismatch"] += quality.get("clean_fields") != list(CLEAN_FIELDS)

    errors = {
        key: value for key, value in sorted(counts.items())
        if key not in {"records"} and value
    }
    errors.update({key: value for key, value in sorted(quality_errors.items()) if value})
    return {
        "schema": "sag_multiview_export_check_v1",
        "input_schema": MULTIVIEW_INPUT_VERSION,
        "private_input": True,
        "records": counts["records"],
        "clean_field_presence": {
            field: {
                "count": clean_field_presence[field],
                "rate": round(clean_field_presence[field] / counts["records"], 6)
                if counts["records"] else 0.0,
            }
            for field in CLEAN_FIELDS
        },
        "output_bytes": output_bytes,
        "output_sha256": output_sha256,
        "error_counts": errors,
        "errors_present": bool(errors),
    }
