"""Deterministic flat member links rebuilt from authoritative grounded entities."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile

from .constants import (
    ENTITIES_PRIVATE_NAME,
    ENTITY_ROLES,
    LINK_SCHEMA_VERSION,
    LINKS_PRIVATE_NAME,
    PIPELINE_VERSION,
    PROJECTION_VERSION,
)
from .pipeline import PipelineError, iter_jsonl
from .work_order import canonical_json_bytes


def stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()
    return prefix + digest[:32]


def iter_projected_links(run_dir: Path):
    entities_path = Path(run_dir) / ENTITIES_PRIVATE_NAME
    seen_docs = set()
    seen_issues = set()
    for _line_number, entity in iter_jsonl(entities_path):
        doc_id = entity.get("doc_id")
        if not isinstance(doc_id, str) or doc_id in seen_docs:
            raise PipelineError("projection_invalid_document")
        seen_docs.add(doc_id)
        issues = entity.get("issues")
        if not isinstance(issues, list) or not issues:
            raise PipelineError("projection_invalid_issues")
        for issue in issues:
            issue_id = issue.get("issue_id")
            if not isinstance(issue_id, str) or issue_id in seen_issues:
                raise PipelineError("projection_invalid_issue")
            seen_issues.add(issue_id)
            event_id = stable_id("event_", doc_id, issue_id)
            issue_member_count = 0
            for role in ENTITY_ROLES:
                members = issue.get(role)
                if not isinstance(members, list):
                    raise PipelineError("projection_missing_role")
                for member in members:
                    surface = member.get("text") if isinstance(member, dict) else None
                    mentions = member.get("mentions") if isinstance(member, dict) else None
                    if not isinstance(surface, str) or not surface or not isinstance(mentions, list):
                        raise PipelineError("projection_invalid_member")
                    member_id = stable_id("member_", issue_id, role, surface)
                    issue_member_count += 1
                    yield {
                        "schema_version": LINK_SCHEMA_VERSION,
                        "pipeline_version": PIPELINE_VERSION,
                        "projection_version": PROJECTION_VERSION,
                        "contract_hash": entity["contract_hash"],
                        "doc_id": doc_id,
                        "content_hash": entity["content_hash"],
                        "event_id": event_id,
                        "issue_id": issue_id,
                        "member_id": member_id,
                        "role": role,
                        "surface": surface,
                        "mentions": mentions,
                    }
            if issue_member_count == 0:
                raise PipelineError("projection_empty_issue")


def project(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    target = run_dir / LINKS_PRIVATE_NAME
    descriptor, temporary_name = tempfile.mkstemp(prefix=".links-", dir=run_dir)
    link_count = 0
    event_count = 0
    entity_document_count = 0
    previous_event_id = None
    previous_doc_id = None
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            for row in iter_projected_links(run_dir):
                output.write(canonical_json_bytes(row).decode("utf-8") + "\n")
                link_count += 1
                if row["event_id"] != previous_event_id:
                    event_count += 1
                    previous_event_id = row["event_id"]
                if row["doc_id"] != previous_doc_id:
                    entity_document_count += 1
                    previous_doc_id = row["doc_id"]
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, target)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return {
        "schema_version": "projection_summary_safe_v1",
        "pipeline_version": PIPELINE_VERSION,
        "entity_document_count": entity_document_count,
        "event_count": event_count,
        "link_count": link_count,
    }
