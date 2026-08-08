"""Strict streaming preparation of redacted 12345 work-order documents."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Iterator

from .constants import (
    CLEAN_FIELDS,
    CLEAN_FIELD_LABELS,
    CLEAN_FIELD_SOURCES,
    DOC_ID_NAMESPACE,
    DOCUMENT_PRIVATE_NAME,
    DOCUMENT_SCHEMA_VERSION,
    METADATA_LABELS,
    METADATA_SOURCES,
    PII_REDACTION_VERSION,
    PIPELINE_VERSION,
    PREPARE_SAFE_NAME,
    REQUIRED_TSV_COLUMNS,
    SOURCE_ID_COLUMNS,
)
from .pii_redactor import redact_text, residual_pii_codes


NULLISH_VALUES = frozenset({"", "NULL", "null", "None", "none", "\\N"})
_CALL_TIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}$")
_LITERAL_TAB_RE = re.compile(r"\\[tTrRnN]")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
MAX_HEADER_BYTES = 1024 * 1024
MAX_PHYSICAL_LINE_BYTES = 4 * 1024 * 1024
MAX_SEMANTIC_FIELD_CHARS = 200_000
MAX_SOURCE_ID_CHARS = 256
MAX_STRUCTURED_FIELD_CHARS = {
    "service_object_type": 64,
    "area_code_city": 64,
    "area_code_area": 64,
    "area_code_street": 128,
    "case_accord_type_one_name": 128,
    "case_accord_type_two_name": 128,
    "case_accord_type_three_name": 192,
    "order_source": 64,
    "order_type": 64,
    "order_status": 64,
    "call_time": 32,
}
_SENTENCE_PUNCTUATION_RE = re.compile(r"[。！？!?]")


class PrepareError(ValueError):
    """Fatal input or publication error represented by a text-free code."""


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def file_sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return "sha256:" + digest.hexdigest(), size


def atomic_write_json(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".safe-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as target:
            target.write(canonical_json_bytes(value).decode("utf-8"))
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def stable_doc_id(source_id: str) -> str:
    digest = hashlib.sha256((DOC_ID_NAMESPACE + "\0" + source_id).encode("utf-8")).hexdigest()
    return "order_" + digest[:32]


def clean_content_hash(clean_fields: dict[str, str]) -> str:
    payload = {field: clean_fields[field] for field in CLEAN_FIELDS}
    return "sha256:" + hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _clean_cell(value: str) -> str:
    stripped = value.strip()
    return "" if stripped in NULLISH_VALUES else stripped


def _source_identity(row: dict[str, str]) -> str:
    for column in SOURCE_ID_COLUMNS:
        value = _clean_cell(row[column])
        if value:
            return value
    return ""


def _polluted(value: str) -> bool:
    return (
        "\t" in value
        or "\r" in value
        or "\n" in value
        or bool(_LITERAL_TAB_RE.search(value))
        or bool(_CONTROL_RE.search(value))
    )


def _structured_polluted(source_name: str, value: str) -> bool:
    if _polluted(value) or len(value) > MAX_STRUCTURED_FIELD_CHARS[source_name]:
        return True
    # Long sentence-like values in categorical columns are a conservative shift signal.
    return source_name != "call_time" and len(value) > 32 and bool(
        _SENTENCE_PUNCTUATION_RE.search(value)
    )


def _redact_fields(
    row: dict[str, str], mapping: dict[str, str]
) -> tuple[dict[str, str], Counter[str], bool]:
    result: dict[str, str] = {}
    counts: Counter[str] = Counter()
    residual = False
    for output_name, source_name in mapping.items():
        cleaned = _clean_cell(row[source_name])
        redacted, field_counts = redact_text(cleaned)
        result[output_name] = redacted
        counts.update(field_counts)
        residual = residual or bool(residual_pii_codes(redacted))
    return result, counts, residual


def build_rag_text(clean_fields: dict[str, str], metadata: dict[str, str]) -> str:
    lines = [
        f"{CLEAN_FIELD_LABELS[field]}：{clean_fields[field]}"
        for field in CLEAN_FIELDS
        if clean_fields[field]
    ]
    lines.extend(
        f"{METADATA_LABELS[field]}：{metadata[field]}"
        for field in METADATA_SOURCES
        if metadata[field]
    )
    return "\n".join(lines)


def _row_to_document(
    row: dict[str, str], seen_source_hashes: set[bytes]
) -> tuple[dict | None, str | None, Counter[str]]:
    source_id = _source_identity(row)
    if not source_id:
        return None, "missing_source_id", Counter()
    if _polluted(source_id) or len(source_id) > MAX_SOURCE_ID_CHARS:
        return None, "polluted_structured_field", Counter()
    source_hash = hashlib.sha256(source_id.encode("utf-8")).digest()
    if source_hash in seen_source_hashes:
        return None, "duplicate_source_id", Counter()
    # Reserve identity even when later validation rejects the row, so a second row cannot assume it.
    seen_source_hashes.add(source_hash)

    for source_name in METADATA_SOURCES.values():
        if _structured_polluted(source_name, _clean_cell(row[source_name])):
            return None, "polluted_structured_field", Counter()
    call_time = _clean_cell(row[METADATA_SOURCES["call_time"]])
    if call_time and not _CALL_TIME_RE.fullmatch(call_time):
        return None, "polluted_structured_field", Counter()

    for source_name in CLEAN_FIELD_SOURCES.values():
        semantic_value = _clean_cell(row[source_name])
        if _polluted(semantic_value) or len(semantic_value) > MAX_SEMANTIC_FIELD_CHARS:
            return None, "suspected_field_shift", Counter()

    clean_fields, clean_counts, clean_residual = _redact_fields(row, CLEAN_FIELD_SOURCES)
    if not any(clean_fields.values()):
        return None, "missing_semantic_text", Counter()
    metadata, metadata_counts, metadata_residual = _redact_fields(row, METADATA_SOURCES)
    redactions = clean_counts + metadata_counts
    if clean_residual or metadata_residual:
        return None, "pii_residual", Counter()

    document = {
        "schema_version": DOCUMENT_SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "redaction_version": PII_REDACTION_VERSION,
        "doc_id": stable_doc_id(source_id),
        "content_hash": clean_content_hash(clean_fields),
        **clean_fields,
        "rag_text": build_rag_text(clean_fields, metadata),
        "metadata": {
            **metadata,
            "call_month": call_time[:7] if call_time else "",
        },
    }
    return document, None, redactions


def iter_prepared_documents(
    input_path: Path,
    *,
    limit: int | None = None,
    counters: Counter[str] | None = None,
    redactions: Counter[str] | None = None,
) -> Iterator[dict]:
    """Yield valid documents while aggregating bad-row codes without leaking row data."""
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise PrepareError("invalid_limit")
    counters = counters if counters is not None else Counter()
    redactions = redactions if redactions is not None else Counter()
    seen_source_hashes: set[bytes] = set()

    try:
        source = Path(input_path).open("rb")
    except OSError as exc:
        raise PrepareError("input_unavailable") from exc
    with source:
        header_bytes = source.readline(MAX_HEADER_BYTES + 1)
        if not header_bytes:
            raise PrepareError("missing_header")
        if len(header_bytes) > MAX_HEADER_BYTES or not header_bytes.endswith((b"\n", b"\r")):
            raise PrepareError("invalid_header_size")
        try:
            header_line = header_bytes.decode("utf-8-sig")
        except UnicodeError as exc:
            raise PrepareError("invalid_header_utf8") from exc
        header = header_line.rstrip("\r\n").split("\t")
        if not header or any(not name for name in header):
            raise PrepareError("empty_header_column")
        if len(header) != len(set(header)):
            raise PrepareError("duplicate_header_column")
        missing = [name for name in REQUIRED_TSV_COLUMNS if name not in header]
        if missing:
            raise PrepareError("missing_required_columns")

        while True:
            raw_bytes = source.readline(MAX_PHYSICAL_LINE_BYTES + 1)
            if not raw_bytes:
                break
            if limit is not None and counters["rows_read"] >= limit:
                break
            counters["rows_read"] += 1
            oversized = len(raw_bytes) > MAX_PHYSICAL_LINE_BYTES
            if oversized and not raw_bytes.endswith((b"\n", b"\r")):
                while raw_bytes and not raw_bytes.endswith((b"\n", b"\r")):
                    raw_bytes = source.readline(MAX_PHYSICAL_LINE_BYTES + 1)
            if oversized:
                counters["bad_field_count"] += 1
                continue
            try:
                raw_line = raw_bytes.decode("utf-8")
            except UnicodeError:
                counters["bad_field_count"] += 1
                continue
            if not raw_line.endswith(("\n", "\r")):
                counters["bad_field_count"] += 1
                continue
            cells = raw_line.rstrip("\r\n").split("\t")
            if len(cells) != len(header):
                counters["bad_field_count"] += 1
                continue
            row = dict(zip(header, cells))
            document, error_code, row_redactions = _row_to_document(row, seen_source_hashes)
            redactions.update(row_redactions)
            if error_code:
                counters[error_code] += 1
                continue
            counters["documents_written"] += 1
            yield document


def prepare(input_path: Path, run_dir: Path, limit: int | None = None) -> dict:
    """Create an atomic, immutable redacted document snapshot and safe aggregate report."""
    run_dir = Path(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise PrepareError("run_dir_not_fresh")
    run_dir.mkdir(parents=True, exist_ok=True)

    counters: Counter[str] = Counter()
    redactions: Counter[str] = Counter()
    documents_path = run_dir / DOCUMENT_PRIVATE_NAME
    descriptor, temporary_name = tempfile.mkstemp(prefix=".documents-", dir=run_dir)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as target:
            for document in iter_prepared_documents(
                Path(input_path), limit=limit, counters=counters, redactions=redactions
            ):
                target.write(canonical_json_bytes(document).decode("utf-8"))
                target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        if counters["documents_written"] == 0:
            raise PrepareError("no_documents")
        output_sha256, output_bytes = file_sha256(Path(temporary_name))
        os.replace(temporary_name, documents_path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        try:
            run_dir.rmdir()
        except OSError:
            pass
        raise

    rejection_codes = (
        "bad_field_count",
        "missing_source_id",
        "duplicate_source_id",
        "missing_semantic_text",
        "suspected_field_shift",
        "polluted_structured_field",
        "pii_residual",
    )
    report = {
        "schema_version": "prepare_safe_v1",
        "pipeline_version": PIPELINE_VERSION,
        "document_schema_version": DOCUMENT_SCHEMA_VERSION,
        "redaction_version": PII_REDACTION_VERSION,
        "limit_applied": limit is not None,
        "rows_read": counters["rows_read"],
        "documents_written": counters["documents_written"],
        "rows_rejected": sum(counters[code] for code in rejection_codes),
        "rejection_counts": {code: counters[code] for code in rejection_codes},
        "redaction_counts": dict(sorted(redactions.items())),
        "output_bytes": output_bytes,
        "output_sha256": output_sha256,
    }
    atomic_write_json(run_dir / PREPARE_SAFE_NAME, report)
    return report
