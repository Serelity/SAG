"""Loopback-only workbench for editing private SAG issue annotations."""

import hashlib
import json
import os
import re
import secrets
import shutil
import threading
from copy import deepcopy
from pathlib import Path

from ragflow_style_pipeline.sag_semantic_audit import validate_gold_annotations, validate_gold_record
from ragflow_style_pipeline.sag_semantic_versions import ANNOTATION_ROUND_VERSION

_EDITABLE_KEYS = (
    "issues", "declared_intents", "direct_emotions", "satisfaction", "urgency",
)
_METADATA_KEYS = (
    "service_object_type", "type1", "type2", "type3", "call_month",
    "area_code_area", "area_code_street", "order_source", "order_type", "order_status",
)


def _sha256_bytes(value):
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _stable_jsonl(rows):
    return "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")


def _atomic_bytes(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as output:
        output.write(value)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


class AnnotationStoreError(ValueError):
    """Expected workbench state or validation failure with a safe error code."""


class AnnotationStore:
    """Own one private A/B annotation file and save validated edits atomically."""

    def __init__(self, path, expected_annotator=""):
        self.path = Path(path).resolve()
        self.backup_path = self.path.with_name(self.path.name + ".bak")
        self.expected_annotator = str(expected_annotator or "").strip()
        self._lock = threading.RLock()
        self._rows = []
        self._revision = ""
        self.annotator = ""
        self._load()

    def _load(self):
        if not self.path.is_file() or ".private." not in self.path.name:
            raise AnnotationStoreError("private_annotation_file_required")
        report = validate_gold_annotations(
            self.path, expected_annotator=self.expected_annotator
        )
        if report["errors_present"] or not report["records_read"]:
            raise AnnotationStoreError("annotation_file_invalid")
        rows = []
        with self.path.open("r", encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise AnnotationStoreError("annotation_file_invalid")
                    rows.append(value)
        annotators = {
            str(row.get("annotation", {}).get("annotator", "")).strip()
            for row in rows if isinstance(row.get("annotation"), dict)
        }
        statuses = {
            str(row.get("annotation", {}).get("status", ""))
            for row in rows if isinstance(row.get("annotation"), dict)
        }
        round_provenance = {
            json.dumps(
                row.get("annotation_round_provenance"),
                ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            )
            for row in rows
        }
        if (
            len(annotators) != 1 or not next(iter(annotators))
            or not statuses <= {"in_progress", "completed"}
            or len(round_provenance) != 1
            or next(iter(round_provenance)) in {"null", "{}"}
        ):
            raise AnnotationStoreError("annotation_round_file_required")
        provenance = rows[0].get("annotation_round_provenance")
        if (
            not isinstance(provenance, dict)
            or provenance.get("schema") != ANNOTATION_ROUND_VERSION
            or not isinstance(provenance.get("round_id"), str)
            or not provenance["round_id"]
            or not re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                str(provenance.get("source_packet_sha256", "")),
            )
        ):
            raise AnnotationStoreError("annotation_round_provenance_invalid")
        self._rows = rows
        self.annotator = next(iter(annotators))
        self._revision = _sha256_bytes(self.path.read_bytes())

    @property
    def revision(self):
        return self._revision

    def _assert_not_changed(self):
        if not self.path.is_file() or _sha256_bytes(self.path.read_bytes()) != self._revision:
            raise AnnotationStoreError("annotation_file_changed_externally")

    def summary(self):
        with self._lock:
            counts = {"in_progress": 0, "completed": 0}
            for row in self._rows:
                status = row["annotation"]["status"]
                counts[status] = counts.get(status, 0) + 1
            return {
                "records": len(self._rows),
                "status_counts": counts,
                "revision": self._revision,
            }

    def record(self, index):
        with self._lock:
            if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(self._rows):
                raise AnnotationStoreError("record_index_out_of_range")
            row = self._rows[index]
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            result = {
                "index": index,
                "records": len(self._rows),
                "subset": row.get("subset", ""),
                "clean_fields": deepcopy(row.get("clean_fields", {})),
                "metadata": {
                    key: metadata[key] for key in _METADATA_KEYS
                    if key in metadata and isinstance(metadata[key], (str, int, float, bool))
                },
                "annotation": {
                    "status": row.get("annotation", {}).get("status", ""),
                    "notes": row.get("annotation", {}).get("notes", ""),
                },
                "revision": self._revision,
            }
            for key in _EDITABLE_KEYS:
                result[key] = deepcopy(row.get(key))
            return result

    def save(self, index, revision, payload):
        with self._lock:
            self._assert_not_changed()
            if revision != self._revision:
                raise AnnotationStoreError("annotation_revision_conflict")
            if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(self._rows):
                raise AnnotationStoreError("record_index_out_of_range")
            if not isinstance(payload, dict) or set(payload) != {*_EDITABLE_KEYS, "status", "notes"}:
                raise AnnotationStoreError("annotation_payload_invalid")
            status = payload.get("status")
            notes = payload.get("notes")
            if status not in {"in_progress", "completed"} or not isinstance(notes, str):
                raise AnnotationStoreError("annotation_payload_invalid")

            candidate = deepcopy(self._rows[index])
            for key in _EDITABLE_KEYS:
                candidate[key] = deepcopy(payload[key])
            candidate["annotation"] = {
                "annotator": self.annotator,
                "status": status,
                "notes": notes,
            }
            validation = validate_gold_record(
                candidate,
                require_complete=status == "completed",
                expected_annotator=self.annotator,
            )
            if validation["errors"]:
                return {
                    "saved": False,
                    "validation": validation,
                    "revision": self._revision,
                }

            updated = list(self._rows)
            updated[index] = candidate
            encoded = _stable_jsonl(updated)
            backup_temporary = self.backup_path.with_name(self.backup_path.name + ".tmp")
            with self.path.open("rb") as source, backup_temporary.open("wb") as backup:
                shutil.copyfileobj(source, backup)
                backup.flush()
                os.fsync(backup.fileno())
            os.replace(backup_temporary, self.backup_path)
            _atomic_bytes(self.path, encoded)
            self._rows = updated
            self._revision = _sha256_bytes(encoded)
            return {
                "saved": True,
                "validation": validation,
                "revision": self._revision,
                "status_counts": self.summary()["status_counts"],
            }


def new_session_token():
    return secrets.token_urlsafe(32)
