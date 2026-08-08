"""Deterministic exact grounding of untrusted model string candidates."""

from __future__ import annotations

import hashlib
import json
import re

from .constants import (
    CLEAN_FIELDS,
    ENTITY_ROLES,
    ENTITY_SCHEMA_VERSION,
    GROUNDING_VERSION,
)
from .pii_redactor import PLACEHOLDERS


_PUNCTUATION_ONLY_RE = re.compile(r"^[\W_]+$", re.UNICODE)


def _all_non_overlapping_occurrences(haystack: str, needle: str) -> list[tuple[int, int]]:
    occurrences = []
    offset = 0
    while needle and offset <= len(haystack) - len(needle):
        start = haystack.find(needle, offset)
        if start < 0:
            break
        end = start + len(needle)
        occurrences.append((start, end))
        offset = end
    return occurrences


def valid_surface(surface: object) -> bool:
    if not isinstance(surface, str) or not surface:
        return False
    if surface != surface.strip() or _PUNCTUATION_ONLY_RE.fullmatch(surface):
        return False
    return not any(placeholder in surface for placeholder in PLACEHOLDERS)


def _placeholder_spans(text: str) -> list[tuple[int, int]]:
    spans = []
    for placeholder in PLACEHOLDERS:
        spans.extend(_all_non_overlapping_occurrences(text, placeholder))
    return spans


def ground_surface_mentions(document: dict, surface: str) -> list[dict]:
    mentions = []
    for field in CLEAN_FIELDS:
        field_text = document.get(field, "")
        pii_spans = _placeholder_spans(field_text)
        for start, end in _all_non_overlapping_occurrences(field_text, surface):
            if any(start < pii_end and end > pii_start for pii_start, pii_end in pii_spans):
                continue
            evidence = field_text[start:end]
            if evidence == surface:
                mentions.append(
                    {
                        "field": field,
                        "start": start,
                        "end": end,
                        "evidence": evidence,
                    }
                )
    return mentions


def _canonical_issue_members(issue: dict) -> dict[str, list[str]]:
    return {
        role: sorted({member["text"] for member in issue[role]})
        for role in ENTITY_ROLES
    }


def stable_issue_id(doc_id: str, issue: dict) -> str:
    canonical = json.dumps(
        _canonical_issue_members(issue),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256((doc_id + "\0" + canonical).encode("utf-8")).hexdigest()
    return "issue_" + digest[:32]


def ground_payload(document: dict, payload: dict) -> dict:
    """Ground all candidates and drop only invalid candidates or empty issues."""
    grounded_issues = []
    seen_issue_members = set()
    input_candidates = 0
    dropped_candidates = 0
    duplicate_candidates = 0
    empty_issues = 0
    duplicate_issues = 0

    for candidate_issue in payload["issues"]:
        issue = {role: [] for role in ENTITY_ROLES}
        for role in ENTITY_ROLES:
            seen_surfaces = set()
            for surface in candidate_issue[role]:
                input_candidates += 1
                if surface in seen_surfaces:
                    duplicate_candidates += 1
                    continue
                seen_surfaces.add(surface)
                if not valid_surface(surface):
                    dropped_candidates += 1
                    continue
                mentions = ground_surface_mentions(document, surface)
                if not mentions:
                    dropped_candidates += 1
                    continue
                issue[role].append({"text": surface, "mentions": mentions})
        if not any(issue[role] for role in ENTITY_ROLES):
            empty_issues += 1
            continue
        canonical = json.dumps(
            _canonical_issue_members(issue),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if canonical in seen_issue_members:
            duplicate_issues += 1
            continue
        seen_issue_members.add(canonical)
        grounded_issues.append({"issue_id": stable_issue_id(document["doc_id"], issue), **issue})

    return {
        "schema_version": ENTITY_SCHEMA_VERSION,
        "grounding_version": GROUNDING_VERSION,
        "doc_id": document["doc_id"],
        "content_hash": document["content_hash"],
        "issues": grounded_issues,
        "grounding_stats": {
            "input_candidates": input_candidates,
            "grounded_candidates": sum(
                len(issue[role]) for issue in grounded_issues for role in ENTITY_ROLES
            ),
            "dropped_candidates": dropped_candidates,
            "duplicate_candidates": duplicate_candidates,
            "empty_issues": empty_issues,
            "duplicate_issues": duplicate_issues,
        },
    }
