"""Build a minimal private Qwen input packet from exact manifest identities."""

import hashlib
import json
import os
import re
from pathlib import Path

from ragflow_style_pipeline.sag_semantic_versions import (
    EVAL_MANIFEST_VERSION,
    INFERENCE_PACKET_VERSION,
    MULTIVIEW_INPUT_VERSION,
)
from ragflow_style_pipeline.work_order_input import (
    CLEAN_FIELDS,
    WorkOrderInputError,
    normalize_work_order,
)


def _hash_bytes(value):
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _load_manifest(path):
    rows = []
    doc_ids = set()
    identities = set()
    with Path(path).open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError("identity_manifest_invalid_json") from error
            if (
                not isinstance(row, dict)
                or row.get("schema") != EVAL_MANIFEST_VERSION
                or not isinstance(row.get("doc_id"), str)
                or not row["doc_id"]
                or not isinstance(row.get("content_hash"), str)
                or not re.fullmatch(r"sha256:[0-9a-f]{64}", row["content_hash"])
            ):
                raise ValueError("identity_manifest_invalid_record")
            identity = (row["doc_id"], row["content_hash"])
            if row["doc_id"] in doc_ids:
                raise ValueError("identity_manifest_duplicate_doc_id")
            if identity in identities:
                raise ValueError("identity_manifest_duplicate_identity")
            doc_ids.add(row["doc_id"])
            identities.add(identity)
            rows.append(row)
    if not rows:
        raise ValueError("identity_manifest_empty")
    return rows


def build_inference_packet(input_path, manifest_path, output_path):
    """Select manifest records exactly and atomically write only model-required input."""
    input_path = Path(input_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    output_path = Path(output_path).resolve()
    if len({input_path, manifest_path, output_path}) != 3:
        raise ValueError("inference_packet_paths_must_differ")
    manifest = _load_manifest(manifest_path)
    targets = {row["doc_id"]: row["content_hash"] for row in manifest}
    selected = {}
    records_scanned = 0
    ignored_invalid_records = 0
    with Path(input_path).open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            records_scanned += 1
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                ignored_invalid_records += 1
                continue
            raw_doc_id = raw.get("doc_id") if isinstance(raw, dict) else None
            if raw_doc_id not in targets:
                try:
                    normalize_work_order(raw)
                except WorkOrderInputError:
                    ignored_invalid_records += 1
                continue
            if raw_doc_id in selected:
                raise ValueError("target_doc_id_duplicated_in_source")
            try:
                order = normalize_work_order(raw)
            except WorkOrderInputError as error:
                raise ValueError("target_record_invalid") from error
            if order["content_hash"] != targets[raw_doc_id]:
                raise ValueError("target_content_hash_mismatch")
            selected[raw_doc_id] = {
                "input_schema": MULTIVIEW_INPUT_VERSION,
                "inference_packet_schema": INFERENCE_PACKET_VERSION,
                "doc_id": order["doc_id"],
                "content_hash": order["content_hash"],
                **{field: order[field] for field in CLEAN_FIELDS},
                "metadata": order["metadata"],
            }

    if len(selected) != len(manifest):
        raise ValueError("identity_manifest_target_missing")
    output_rows = [selected[row["doc_id"]] for row in manifest]
    encoded = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in output_rows
    ).encode("utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    try:
        with temporary.open("wb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "schema": INFERENCE_PACKET_VERSION,
        "private_output": True,
        "source_sha256": _file_hash(input_path),
        "manifest_sha256": _file_hash(manifest_path),
        "records_scanned": records_scanned,
        "ignored_invalid_records": ignored_invalid_records,
        "manifest_records": len(manifest),
        "records_written": len(output_rows),
        "output_bytes": len(encoded),
        "output_sha256": _hash_bytes(encoded),
        "fields": [
            "input_schema", "inference_packet_schema", "doc_id", "content_hash",
            *CLEAN_FIELDS, "metadata",
        ],
    }
