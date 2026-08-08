"""End-to-end integrity and privacy checks producing aggregate-only reports."""

from __future__ import annotations

from collections import Counter
from itertools import zip_longest
import json
import math
from pathlib import Path
import re

from .constants import (
    CLEAN_FIELDS,
    CONTRACT_PRIVATE_NAME,
    DIAGNOSTICS_SAFE_NAME,
    DOCUMENT_PRIVATE_NAME,
    DOCUMENT_SCHEMA_VERSION,
    ENTITIES_PRIVATE_NAME,
    ENTITY_ROLES,
    ENTITY_SCHEMA_VERSION,
    GROUNDING_VERSION,
    LINK_SCHEMA_VERSION,
    LINKS_PRIVATE_NAME,
    PIPELINE_VERSION,
    PREPARE_SAFE_NAME,
    REJECTS_PRIVATE_NAME,
    REJECT_SCHEMA_VERSION,
    RUN_SAFE_NAME,
    SAFE_FORBIDDEN_KEYS,
)
from .grounding import ground_surface_mentions, stable_issue_id, valid_surface
from .pipeline import canonical_hash, documents_manifest, iter_jsonl
from .projection import iter_projected_links
from .work_order import atomic_write_json, clean_content_hash, file_sha256


_ENTITY_KEYS = {
    "schema_version",
    "grounding_version",
    "doc_id",
    "content_hash",
    "issues",
    "grounding_stats",
    "pipeline_version",
    "contract_hash",
    "document_ordinal",
    "attempt_count",
    "generation_diagnostics",
}
_SAFE_CODE_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,160}$")
_DIAGNOSTIC_KEYS = {
    "attempt",
    "outcome",
    "finish_reason",
    "input_tokens",
    "output_tokens",
    "latency_share_ms",
    "gpu_peak_allocated_gb",
    "gpu_peak_reserved_gb",
}


class CheckError(ValueError):
    """A safe integrity failure code."""


def assert_safe_value(value: object) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() in SAFE_FORBIDDEN_KEYS:
                raise CheckError("unsafe_safe_key")
            assert_safe_value(child)
    elif isinstance(value, list):
        for child in value:
            assert_safe_value(child)
    elif isinstance(value, str):
        lowered = value.lower()
        if ".private." in lowered or lowered.endswith(".private"):
            raise CheckError("unsafe_safe_value")


def _read_safe_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CheckError("invalid_safe_json") from exc
    if not isinstance(value, dict):
        raise CheckError("safe_object_required")
    assert_safe_value(value)
    return value


def _document_index(run_dir: Path) -> dict[str, dict]:
    documents = {}
    path = run_dir / DOCUMENT_PRIVATE_NAME
    try:
        source = path.open("rb")
    except OSError as exc:
        raise CheckError("documents_unavailable") from exc
    with source:
        ordinal = 0
        while True:
            offset = source.tell()
            raw_line = source.readline()
            if not raw_line:
                break
            try:
                document = json.loads(raw_line.decode("utf-8"))
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise CheckError("invalid_document_jsonl") from exc
            if not isinstance(document, dict):
                raise CheckError("invalid_document")
            doc_id = document.get("doc_id")
            if (
                not isinstance(doc_id, str)
                or doc_id in documents
                or document.get("schema_version") != DOCUMENT_SCHEMA_VERSION
            ):
                raise CheckError("invalid_document")
            clean_fields = {field: document.get(field) for field in CLEAN_FIELDS}
            if any(not isinstance(value, str) for value in clean_fields.values()):
                raise CheckError("invalid_clean_fields")
            if clean_content_hash(clean_fields) != document.get("content_hash"):
                raise CheckError("document_content_hash_mismatch")
            documents[doc_id] = {
                "content_hash": document["content_hash"],
                "ordinal": ordinal,
                "offset": offset,
            }
            ordinal += 1
    if not documents:
        raise CheckError("empty_documents")
    return documents


def _read_document_at(source, entry: dict, expected_doc_id: str) -> dict:
    source.seek(entry["offset"])
    try:
        document = json.loads(source.readline().decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CheckError("invalid_document_jsonl") from exc
    if document.get("doc_id") != expected_doc_id:
        raise CheckError("document_index_mismatch")
    return {
        "content_hash": document["content_hash"],
        "clean_fields": {field: document[field] for field in CLEAN_FIELDS},
    }


def _validate_entity(entity: dict, document: dict, doc_id: str) -> tuple[int, int]:
    if (
        set(entity) != _ENTITY_KEYS
        or entity.get("schema_version") != ENTITY_SCHEMA_VERSION
        or entity.get("grounding_version") != GROUNDING_VERSION
        or entity.get("pipeline_version") != PIPELINE_VERSION
        or entity.get("content_hash") != document["content_hash"]
        or not isinstance(entity.get("issues"), list)
        or not entity["issues"]
    ):
        raise CheckError("invalid_entity_contract")
    issue_ids = set()
    member_count = 0
    for issue in entity["issues"]:
        if not isinstance(issue, dict) or set(issue) != {"issue_id", *ENTITY_ROLES}:
            raise CheckError("invalid_grounded_issue")
        issue_id = issue.get("issue_id")
        if not isinstance(issue_id, str) or issue_id in issue_ids:
            raise CheckError("invalid_issue_id")
        issue_members = 0
        for role in ENTITY_ROLES:
            if not isinstance(issue[role], list):
                raise CheckError("invalid_grounded_role")
            seen_surfaces = set()
            for member in issue[role]:
                if not isinstance(member, dict) or set(member) != {"text", "mentions"}:
                    raise CheckError("invalid_grounded_member")
                surface = member["text"]
                mentions = member["mentions"]
                if (
                    not valid_surface(surface)
                    or surface in seen_surfaces
                    or not isinstance(mentions, list)
                    or not mentions
                ):
                    raise CheckError("invalid_grounded_member")
                seen_surfaces.add(surface)
                seen_mentions = set()
                for mention in mentions:
                    if not isinstance(mention, dict) or set(mention) != {
                        "field",
                        "start",
                        "end",
                        "evidence",
                    }:
                        raise CheckError("invalid_grounded_mention")
                    field = mention["field"]
                    start = mention["start"]
                    end = mention["end"]
                    evidence = mention["evidence"]
                    signature = (field, start, end)
                    if (
                        field not in CLEAN_FIELDS
                        or type(start) is not int
                        or type(end) is not int
                        or start < 0
                        or end <= start
                        or signature in seen_mentions
                        or document["clean_fields"][field][start:end] != surface
                        or evidence != surface
                    ):
                        raise CheckError("ungrounded_mention")
                    seen_mentions.add(signature)
                if mentions != ground_surface_mentions(
                    {**document["clean_fields"]}, surface
                ):
                    raise CheckError("incomplete_grounding_mentions")
                issue_members += 1
                member_count += 1
        if issue_members == 0:
            raise CheckError("empty_grounded_issue")
        if issue_id != stable_issue_id(doc_id, issue):
            raise CheckError("invalid_issue_id")
        issue_ids.add(issue_id)
    stats = entity.get("grounding_stats")
    if (
        not isinstance(stats, dict)
        or set(stats) != {
            "input_candidates",
            "grounded_candidates",
            "dropped_candidates",
            "duplicate_candidates",
            "empty_issues",
            "duplicate_issues",
        }
        or any(type(value) is not int or value < 0 for value in stats.values())
        or stats["grounded_candidates"] != member_count
    ):
        raise CheckError("invalid_grounding_stats")
    return len(issue_ids), member_count


def _validate_attempts(row: dict) -> None:
    diagnostics = row.get("generation_diagnostics")
    attempts = row.get("attempt_count")
    if (
        type(attempts) is not int
        or attempts not in {1, 2}
        or not isinstance(diagnostics, list)
        or len(diagnostics) != attempts
    ):
        raise CheckError("invalid_attempt_diagnostics")
    expected = ["primary"] if attempts == 1 else ["primary", "repair"]
    if [item.get("attempt") for item in diagnostics if isinstance(item, dict)] != expected:
        raise CheckError("invalid_attempt_sequence")
    for item in diagnostics:
        if (
            not isinstance(item, dict)
            or not {"attempt", "outcome"} <= set(item) <= _DIAGNOSTIC_KEYS
            or not isinstance(item.get("attempt"), str)
            or not isinstance(item.get("outcome"), str)
            or not _SAFE_CODE_RE.fullmatch(item["attempt"])
            or not _SAFE_CODE_RE.fullmatch(item["outcome"])
        ):
            raise CheckError("invalid_attempt_diagnostics")
        for key, value in item.items():
            if key in {"attempt", "outcome", "finish_reason"}:
                if not isinstance(value, str) or not _SAFE_CODE_RE.fullmatch(value):
                    raise CheckError("invalid_attempt_diagnostics")
            elif type(value) not in (int, float) or not math.isfinite(value):
                raise CheckError("invalid_attempt_diagnostics")
        assert_safe_value(item)


def check(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    prepare_report = _read_safe_json(run_dir / PREPARE_SAFE_NAME)
    expected_prepare_keys = {
        "schema_version",
        "pipeline_version",
        "document_schema_version",
        "redaction_version",
        "limit_applied",
        "rows_read",
        "documents_written",
        "rows_rejected",
        "rejection_counts",
        "redaction_counts",
        "output_bytes",
        "output_sha256",
    }
    if set(prepare_report) != expected_prepare_keys:
        raise CheckError("invalid_prepare_safe_contract")
    documents = _document_index(run_dir)
    try:
        stored_contract = json.loads(
            (run_dir / CONTRACT_PRIVATE_NAME).read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CheckError("invalid_run_contract") from exc
    expected_contract_keys = {
        "pipeline_version",
        "entity_schema_version",
        "documents",
        "model_fingerprint",
        "model_fingerprint_method",
        "prompt_fingerprint",
        "config_fingerprint",
        "contract_hash",
    }
    if (
        not isinstance(stored_contract, dict)
        or set(stored_contract) != expected_contract_keys
        or stored_contract.get("pipeline_version") != PIPELINE_VERSION
        or stored_contract.get("entity_schema_version") != ENTITY_SCHEMA_VERSION
    ):
        raise CheckError("invalid_run_contract")
    contract_body = {key: value for key, value in stored_contract.items() if key != "contract_hash"}
    if stored_contract["contract_hash"] != canonical_hash(contract_body):
        raise CheckError("run_contract_hash_mismatch")
    if stored_contract.get("documents") != documents_manifest(run_dir / DOCUMENT_PRIVATE_NAME):
        raise CheckError("run_contract_documents_mismatch")
    terminal = {}
    contract_hashes = set()
    issue_count = 0
    member_count = 0
    entity_count = 0
    reject_count = 0
    diagnostics_iterator = (
        row for _line_number, row in iter_jsonl(run_dir / DIAGNOSTICS_SAFE_NAME)
    )
    diagnostic_count = 0
    generation_attempts: Counter[str] = Counter()
    finish_reasons: Counter[str] = Counter()
    reject_error_codes: Counter[str] = Counter()
    input_tokens = 0
    output_tokens = 0
    generation_wall_share_ms = 0.0
    gpu_peak_allocated_gb = -1.0
    gpu_peak_reserved_gb = -1.0

    with (run_dir / DOCUMENT_PRIVATE_NAME).open("rb") as document_source:
        terminal_sources = (
            (ENTITIES_PRIVATE_NAME, ENTITY_SCHEMA_VERSION, "entity"),
            (REJECTS_PRIVATE_NAME, REJECT_SCHEMA_VERSION, "reject"),
        )
        for name, expected_schema, status in terminal_sources:
            for _line_number, row in iter_jsonl(run_dir / name):
                doc_id = row.get("doc_id")
                if doc_id not in documents or doc_id in terminal:
                    raise CheckError("invalid_terminal_identity")
                if (
                    row.get("schema_version") != expected_schema
                    or row.get("pipeline_version") != PIPELINE_VERSION
                ):
                    raise CheckError("invalid_terminal_schema")
                document = _read_document_at(document_source, documents[doc_id], doc_id)
                if row.get("content_hash") != documents[doc_id]["content_hash"]:
                    raise CheckError("terminal_content_mismatch")
                ordinal = row.get("document_ordinal")
                if type(ordinal) is not int or ordinal != documents[doc_id]["ordinal"]:
                    raise CheckError("terminal_ordinal_mismatch")
                contract_hash = row.get("contract_hash")
                if not isinstance(contract_hash, str):
                    raise CheckError("terminal_contract_missing")
                contract_hashes.add(contract_hash)
                _validate_attempts(row)
                for diagnostic in row["generation_diagnostics"]:
                    expected_diagnostic = {
                        "schema_version": "generation_diagnostic_safe_v1",
                        "terminal_status": status,
                        **diagnostic,
                    }
                    try:
                        actual_diagnostic = next(diagnostics_iterator)
                    except StopIteration as exc:
                        raise CheckError("diagnostics_mismatch") from exc
                    assert_safe_value(actual_diagnostic)
                    if actual_diagnostic != expected_diagnostic:
                        raise CheckError("diagnostics_mismatch")
                    diagnostic_count += 1
                    generation_attempts[diagnostic["attempt"]] += 1
                    if "finish_reason" in diagnostic:
                        finish_reasons[diagnostic["finish_reason"]] += 1
                    input_tokens += int(diagnostic.get("input_tokens", 0))
                    output_tokens += int(diagnostic.get("output_tokens", 0))
                    generation_wall_share_ms += float(diagnostic.get("latency_share_ms", 0.0))
                    gpu_peak_allocated_gb = max(
                        gpu_peak_allocated_gb,
                        float(diagnostic.get("gpu_peak_allocated_gb", -1.0)),
                    )
                    gpu_peak_reserved_gb = max(
                        gpu_peak_reserved_gb,
                        float(diagnostic.get("gpu_peak_reserved_gb", -1.0)),
                    )
                if status == "entity":
                    issues, members = _validate_entity(row, document, doc_id)
                    issue_count += issues
                    member_count += members
                    entity_count += 1
                else:
                    allowed = {
                        "schema_version",
                        "pipeline_version",
                        "doc_id",
                        "content_hash",
                        "contract_hash",
                        "document_ordinal",
                        "error_codes",
                        "attempt_count",
                        "generation_diagnostics",
                    }
                    error_codes = row.get("error_codes")
                    if (
                        set(row) != allowed
                        or not isinstance(error_codes, list)
                        or len(error_codes) != 2
                        or any(
                            not isinstance(code, str) or not _SAFE_CODE_RE.fullmatch(code)
                            for code in error_codes
                        )
                    ):
                        raise CheckError("invalid_reject_contract")
                    reject_error_codes.update(error_codes)
                    reject_count += 1
                terminal[doc_id] = status

    if (
        set(terminal) != set(documents)
        or contract_hashes != {stored_contract["contract_hash"]}
    ):
        raise CheckError("terminal_set_mismatch")
    output_sha256, output_bytes = file_sha256(run_dir / DOCUMENT_PRIVATE_NAME)
    if (
        prepare_report.get("schema_version") != "prepare_safe_v1"
        or prepare_report.get("pipeline_version") != PIPELINE_VERSION
        or prepare_report.get("documents_written") != len(documents)
        or prepare_report.get("output_sha256") != output_sha256
        or prepare_report.get("output_bytes") != output_bytes
    ):
        raise CheckError("prepare_publication_mismatch")

    try:
        next(diagnostics_iterator)
    except StopIteration:
        pass
    else:
        raise CheckError("diagnostics_mismatch")

    link_count = 0
    actual_links = (row for _line, row in iter_jsonl(run_dir / LINKS_PRIVATE_NAME))
    sentinel = object()
    for actual, expected in zip_longest(
        actual_links, iter_projected_links(run_dir), fillvalue=sentinel
    ):
        if actual is sentinel or expected is sentinel or actual != expected:
            raise CheckError("projection_mismatch")
        if actual.get("schema_version") != LINK_SCHEMA_VERSION:
            raise CheckError("link_schema_mismatch")
        link_count += 1

    report = {
        "schema_version": "entity_extraction_run_safe_v1",
        "pipeline_version": PIPELINE_VERSION,
        "contract_hash": next(iter(contract_hashes)),
        "document_count": len(documents),
        "entity_document_count": entity_count,
        "reject_count": reject_count,
        "issue_count": issue_count,
        "member_count": member_count,
        "link_count": link_count,
        "generation_count": diagnostic_count,
        "generation_attempt_distribution": dict(sorted(generation_attempts.items())),
        "finish_reason_distribution": dict(sorted(finish_reasons.items())),
        "input_tokens_total": input_tokens,
        "output_tokens_total": output_tokens,
        "generation_wall_share_seconds": round(generation_wall_share_ms / 1000, 3),
        "gpu_peak_allocated_gb": gpu_peak_allocated_gb,
        "gpu_peak_reserved_gb": gpu_peak_reserved_gb,
        "reject_error_counts": dict(sorted(reject_error_codes.items())),
        "checks": {
            "document_hashes": True,
            "terminal_identity": True,
            "grounding": True,
            "issue_identity": True,
            "projection_replay": True,
            "diagnostics_privacy": True,
        },
        "report_fingerprint": canonical_hash(
            [len(documents), entity_count, reject_count, issue_count, member_count, link_count]
        ),
    }
    assert_safe_value(report)
    atomic_write_json(run_dir / RUN_SAFE_NAME, report)
    return report
