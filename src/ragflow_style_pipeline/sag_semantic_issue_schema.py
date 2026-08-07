"""Issue-aware model output contract for sag_semantic_v8 development."""

from __future__ import annotations

from ragflow_style_pipeline.sag_semantic_schema import (
    EMOTIONS,
    EMOTIONS_LIMIT,
    INTENTS,
    INTENTS_LIMIT,
    SATISFACTION_LABELS,
    URGENCY_LEVELS,
    _first_balanced_object,
    _load_complete_object,
    _strip_single_fence,
)
from ragflow_style_pipeline.sag_semantic_versions import ISSUE_OUTPUT_SCHEMA_VERSION

ISSUE_LIMIT = 8
ISSUE_GROUPS = (
    "objects", "problem_behaviors", "question_focus", "request_actions", "locations",
)
ISSUE_GROUP_LIMITS = {
    "objects": 4,
    "problem_behaviors": 4,
    "question_focus": 3,
    "request_actions": 3,
    "locations": 5,
}
TIME_SCOPES = {"current", "historical"}
LOCATION_TYPES = {"road", "intersection", "poi"}


def _text(value):
    return value if isinstance(value, str) else ""


def _warn(warnings, code):
    if code not in warnings:
        warnings.append(code)


def _without_confidence(value):
    if isinstance(value, dict):
        return {
            key: _without_confidence(item)
            for key, item in value.items()
            if key not in {"confidence", "canonical", "issue_id"}
        }
    if isinstance(value, list):
        return [_without_confidence(item) for item in value]
    return value


def _source_field(item):
    value = item.get("source_field")
    if not isinstance(value, str):
        value = item.get("field")
    if isinstance(value, str) and (
        value == "case_content_windows" or value.startswith("case_content_windows.")
    ):
        return "case_content_clean"
    return _text(value)


def _member(item):
    if not isinstance(item, dict):
        return None
    return {
        "surface": _text(item.get("surface") or item.get("text")),
        "source_field": _source_field(item),
        "evidence": _text(item.get("evidence")),
    }


def _members(value, warnings, path, limit, preserve_overflow):
    if not isinstance(value, list):
        _warn(warnings, f"malformed_issue_group:{path}")
        return []
    output = []
    for index, item in enumerate(value):
        normalized = _member(item)
        if normalized is None:
            _warn(warnings, f"malformed_issue_member:{path}:{index}")
            continue
        output.append(normalized)
    if len(output) > limit:
        _warn(warnings, f"issue_group_limit_exceeded:{path}")
    return output if preserve_overflow else output[:limit]


def _locations(value, warnings, path, preserve_overflow):
    if not isinstance(value, list):
        _warn(warnings, f"malformed_issue_group:{path}")
        return []
    output = []
    for index, item in enumerate(value):
        normalized = _member(item)
        if normalized is None:
            _warn(warnings, f"malformed_issue_member:{path}:{index}")
            continue
        location_type = _text(item.get("type"))
        if location_type not in LOCATION_TYPES:
            _warn(warnings, f"invalid_location_type:{path}:{index}")
            continue
        output.append({"type": location_type, **normalized})
    limit = ISSUE_GROUP_LIMITS["locations"]
    if len(output) > limit:
        _warn(warnings, f"issue_group_limit_exceeded:{path}")
    return output if preserve_overflow else output[:limit]


def _issues(value, warnings, preserve_overflow):
    raw = value.get("issues")
    if not isinstance(raw, list):
        _warn(warnings, "malformed_issues")
        return []
    output = []
    for index, issue in enumerate(raw):
        if not isinstance(issue, dict):
            _warn(warnings, f"malformed_issue:{index}")
            continue
        scope = _text(issue.get("time_scope"))
        if scope not in TIME_SCOPES:
            scope = "current"
            _warn(warnings, f"invalid_time_scope:issues.{index}")
        normalized = {"time_scope": scope}
        for group in ISSUE_GROUPS[:-1]:
            normalized[group] = _members(
                issue.get(group, []), warnings, f"issues.{index}.{group}",
                ISSUE_GROUP_LIMITS[group], preserve_overflow,
            )
        normalized["locations"] = _locations(
            issue.get("locations", []), warnings, f"issues.{index}.locations",
            preserve_overflow,
        )
        output.append(normalized)
    if len(output) > ISSUE_LIMIT:
        _warn(warnings, "issue_limit_exceeded")
    return output if preserve_overflow else output[:ISSUE_LIMIT]


def _grounded_label(item, include_intensity=False):
    result = {
        "label": _text(item.get("label")),
        "source_field": _source_field(item),
        "evidence": _text(item.get("evidence")),
    }
    if include_intensity:
        intensity = item.get("intensity")
        result["intensity"] = intensity if type(intensity) is int and intensity in (1, 2, 3) else 1
    return result


def _intents(value, warnings):
    if not isinstance(value, list):
        _warn(warnings, "malformed_intents")
        return []
    output = []
    for item in value:
        if not isinstance(item, dict) or item.get("label") not in INTENTS:
            _warn(warnings, "invalid_intent_label")
            continue
        output.append(_grounded_label(item))
    if len(output) > INTENTS_LIMIT:
        _warn(warnings, "intents_limit_exceeded")
    return output[:INTENTS_LIMIT]


def _emotions(value, warnings):
    if not isinstance(value, list):
        _warn(warnings, "malformed_emotions")
        return []
    output = []
    for item in value:
        if not isinstance(item, dict) or item.get("label") not in EMOTIONS:
            _warn(warnings, "invalid_emotion_label")
            continue
        if type(item.get("intensity")) is not int or item.get("intensity") not in (1, 2, 3):
            _warn(warnings, "invalid_emotion_intensity")
        output.append(_grounded_label(item, include_intensity=True))
    if len(output) > EMOTIONS_LIMIT:
        _warn(warnings, "emotions_limit_exceeded")
    return output[:EMOTIONS_LIMIT]


def _satisfaction(value, warnings):
    if not isinstance(value, dict):
        _warn(warnings, "malformed_satisfaction")
        value = {}
    label = value.get("label", "unknown")
    if label not in SATISFACTION_LABELS:
        label = "unknown"
        _warn(warnings, "invalid_satisfaction_label")
    return {
        "label": label,
        "target": _text(value.get("target")),
        "source_field": _source_field(value),
        "evidence": _text(value.get("evidence")),
    }


def _urgency(value, warnings):
    if not isinstance(value, dict):
        _warn(warnings, "malformed_urgency")
        value = {}
    level = value.get("level", "normal")
    if level not in URGENCY_LEVELS:
        level = "normal"
        _warn(warnings, "invalid_urgency_level")
    field = _source_field(value)
    evidence = _text(value.get("evidence"))
    if level == "normal" and (field or evidence):
        field = evidence = ""
        _warn(warnings, "cleared_normal_urgency_grounding")
    return {"level": level, "source_field": field, "evidence": evidence}


def _normalize(value, preserve_overflow=False):
    warnings = []
    value = _without_confidence(value) if isinstance(value, dict) else {}
    discourse = value.get("discourse")
    if not isinstance(discourse, dict):
        if discourse is not None:
            _warn(warnings, "malformed_discourse")
        discourse = {}
    return {
        "output_schema": ISSUE_OUTPUT_SCHEMA_VERSION,
        "event_summary": _text(value.get("event_summary")),
        "issues": _issues(value, warnings, preserve_overflow),
        "discourse": {
            "intents": _intents(discourse.get("intents", []), warnings),
            "emotions": _emotions(discourse.get("emotions", []), warnings),
            "satisfaction": _satisfaction(discourse.get("satisfaction", {}), warnings),
            "urgency": _urgency(discourse.get("urgency", {}), warnings),
        },
    }, warnings


def normalize_issue_semantic_output(value):
    normalized, _warnings = _normalize(value)
    return normalized


def parse_issue_semantic_json(text, preserve_overflow=False):
    """Use the proven v7 tolerant JSON loader, then normalize the issue contract."""
    if not isinstance(text, str):
        return normalize_issue_semantic_output({}), ["json_parse_failed"]
    for candidate in _first_balanced_object(_strip_single_fence(text)):
        value, recovery_warning = _load_complete_object(candidate)
        if value is None:
            continue
        normalized, warnings = _normalize(value, preserve_overflow=bool(preserve_overflow))
        if recovery_warning:
            warnings.insert(0, recovery_warning)
        return normalized, warnings
    return normalize_issue_semantic_output({}), ["json_parse_failed"]


def flatten_issue_counts(semantic):
    """Return stable aggregate counts without flattening issue relationships."""
    counts = {
        "issues": 0, "objects": 0, "problem_behaviors": 0,
        "question_focus": 0, "request_actions": 0,
        "roads": 0, "intersections": 0, "pois": 0,
    }
    issues = semantic.get("issues") if isinstance(semantic, dict) else None
    if not isinstance(issues, list):
        return counts
    counts["issues"] = len(issues)
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        for group in ISSUE_GROUPS[:-1]:
            values = issue.get(group)
            if isinstance(values, list):
                counts[group] += len(values)
        for location in issue.get("locations", []) if isinstance(issue.get("locations"), list) else []:
            if isinstance(location, dict):
                target = {"road": "roads", "intersection": "intersections", "poi": "pois"}.get(location.get("type"))
                if target:
                    counts[target] += 1
    return counts
