"""Deterministic validation and candidate-level sanitation for v8 issues."""

from __future__ import annotations

from copy import deepcopy

from ragflow_style_pipeline.sag_semantic_issue_schema import (
    ISSUE_GROUPS,
    ISSUE_GROUP_LIMITS,
    ISSUE_LIMIT,
    LOCATION_TYPES,
    TIME_SCOPES,
)
from ragflow_style_pipeline.sag_semantic_schema import SOURCE_FIELDS, SOURCE_FIELD_ORDER
from ragflow_style_pipeline.sag_semantic_validation import (
    _ANOMALY_MARKERS,
    _CURRENT_MARKERS,
    _GENERIC,
    _HISTORY_MARKERS,
    _INVALID_SATISFACTION_TARGETS,
    _OBJECT_ATTITUDE_MARKERS,
    _POI_GAP_HINTS,
    _add,
    _contains,
    _emotion_evidence_supported,
    _emotion_fallback,
    _evidence_field,
    _intent_evidence_supported,
    _is_invalid_poi_shape,
    _is_named_road,
    _is_normal_service_action_behavior,
    _is_request_only_behavior,
    _is_strict_intersection,
    _primary_intent_fallback,
    _satisfaction_evidence_supported,
    _text,
)

STATUSES = {"accepted", "accepted_with_warnings", "repair_required", "rejected"}
REJECT_PREFIXES = ("missing_doc_id", "empty_semantic_text", "repair_failed")
WHOLE_RECORD_REPAIR_PREFIXES = (
    "json_parse_failed", "empty_event_summary", "empty_issues", "empty_issue:",
    "malformed_issues", "malformed_issue:", "possible_history_contamination",
)
OPTIONAL_MEMBER_CODES = {
    "invalid_source_field", "missing_evidence", "surface_evidence_mismatch",
    "generic_entity", "road_poi_conflict", "intersection_shape_conflict",
    "poi_shape_conflict", "request_action_as_behavior", "normal_service_action_as_behavior",
}


def _source_field(item):
    if not isinstance(item, dict):
        return ""
    return _text(item.get("source_field") or item.get("field"))


def _path(issue_index, group, member_index):
    return f"issues.{issue_index}.{group}.{member_index}"


def _repair_field(warning):
    if ":" in warning:
        path = warning.split(":", 1)[1]
        if path.startswith(("issues.", "discourse.")):
            return path
    if warning in {"empty_event_summary", "possible_history_contamination"}:
        return "event_summary"
    if warning in {"empty_issues", "malformed_issues"} or warning.startswith("empty_issue:"):
        return "issues"
    return ""


def _grounding_errors(order, item, path, warnings, require_surface=True):
    if not isinstance(item, dict):
        _add(warnings, f"malformed_issue_member:{path}")
        return
    field = _source_field(item)
    evidence = _text(item.get("evidence"))
    surface = _text(item.get("surface"))
    source = _text(order.get(field)) if field in SOURCE_FIELDS else ""
    if field not in SOURCE_FIELDS:
        _add(warnings, f"invalid_source_field:{path}")
    if not _contains(source, evidence):
        _add(warnings, f"missing_evidence:{path}")
    if require_surface and (not surface or not evidence or surface not in evidence):
        _add(warnings, f"surface_evidence_mismatch:{path}")


def _deduplicate(issue, issue_index, actions):
    for group in ISSUE_GROUPS:
        items = issue.get(group)
        if not isinstance(items, list):
            continue
        seen, kept = set(), []
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                kept.append(item)
                continue
            key = (
                _text(item.get("type")) if group == "locations" else "",
                _text(item.get("surface")), _source_field(item), _text(item.get("evidence")),
            )
            if key in seen:
                _add(actions, f"deduplicated_issue_member:issues.{issue_index}.{group}.{index}")
                continue
            seen.add(key)
            kept.append(item)
        issue[group] = kept


def enrich_issue_semantic_output(order, semantic, parse_warnings=None):
    """Recover only source fields and direct discourse proven by clean text."""
    del parse_warnings
    order = order if isinstance(order, dict) else {}
    cleaned = deepcopy(semantic if isinstance(semantic, dict) else {})
    actions = []
    issues = cleaned.get("issues") if isinstance(cleaned.get("issues"), list) else []
    for issue_index, issue in enumerate(issues):
        if not isinstance(issue, dict):
            continue
        for group in ISSUE_GROUPS:
            items = issue.get(group)
            if not isinstance(items, list):
                continue
            for member_index, item in enumerate(items):
                if not isinstance(item, dict):
                    continue
                evidence = _text(item.get("evidence"))
                if _source_field(item) not in SOURCE_FIELDS and evidence:
                    field = _evidence_field(order, evidence)
                    if field:
                        item["source_field"] = field
                        _add(actions, f"recovered_issue_source:{_path(issue_index, group, member_index)}")
        _deduplicate(issue, issue_index, actions)

    discourse = cleaned.get("discourse") if isinstance(cleaned.get("discourse"), dict) else {}
    for group in ("intents", "emotions"):
        items = discourse.get(group) if isinstance(discourse.get(group), list) else []
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            evidence = _text(item.get("evidence"))
            if _source_field(item) not in SOURCE_FIELDS and evidence:
                field = _evidence_field(order, evidence)
                if field:
                    item["source_field"] = field
                    _add(actions, f"recovered_discourse_source:discourse.{group}.{index}")
    intents = discourse.get("intents") if isinstance(discourse.get("intents"), list) else []
    if not intents:
        label, evidence = _primary_intent_fallback(order)
        if label and evidence:
            intents.append({
                "label": label, "source_field": _evidence_field(order, evidence),
                "evidence": evidence,
            })
            _add(actions, f"recovered_explicit_intent:{label}")
    emotions = discourse.get("emotions") if isinstance(discourse.get("emotions"), list) else []
    if not emotions:
        for label in ("愤怒", "不满", "焦虑", "无奈", "悲伤"):
            evidence, intensity = _emotion_fallback(order, label)
            if evidence:
                emotions.append({
                    "label": label, "intensity": intensity,
                    "source_field": _evidence_field(order, evidence), "evidence": evidence,
                })
                _add(actions, f"recovered_explicit_emotion:{label}")
                break
    discourse["intents"], discourse["emotions"] = intents, emotions
    for name in ("satisfaction", "urgency"):
        item = discourse.get(name)
        if not isinstance(item, dict):
            continue
        evidence = _text(item.get("evidence"))
        if _source_field(item) not in SOURCE_FIELDS and evidence:
            field = _evidence_field(order, evidence)
            if field:
                item["source_field"] = field
                _add(actions, f"recovered_discourse_source:discourse.{name}")
    cleaned["issues"], cleaned["discourse"] = issues, discourse
    return cleaned, actions


def sanitize_issue_semantic_output(semantic, warnings, order=None):
    """Drop invalid optional members; never invent issue relationships."""
    order = order if isinstance(order, dict) else {}
    cleaned = deepcopy(semantic if isinstance(semantic, dict) else {})
    issues = cleaned.get("issues") if isinstance(cleaned.get("issues"), list) else []
    discourse = cleaned.get("discourse") if isinstance(cleaned.get("discourse"), dict) else {}
    drops = {}
    drop_discourse = {"intents": set(), "emotions": set()}
    reset_satisfaction = reset_urgency = False
    actions = []
    for warning in warnings or []:
        if warning in {
            "satisfaction_missing_target_or_evidence", "invalid_satisfaction_target",
            "unsupported_satisfaction_evidence", "template_politeness_as_satisfaction",
        }:
            reset_satisfaction = True
            continue
        if warning in {"urgency_missing_evidence", "template_priority_as_urgency"}:
            reset_urgency = True
            continue
        if not isinstance(warning, str) or ":" not in warning:
            continue
        code, path = warning.split(":", 1)
        parts = path.split(".")
        if code in OPTIONAL_MEMBER_CODES and len(parts) == 4 and parts[0] == "issues":
            try:
                issue_index, member_index = int(parts[1]), int(parts[3])
            except ValueError:
                continue
            group = parts[2]
            item = None
            if 0 <= issue_index < len(issues) and isinstance(issues[issue_index], dict):
                values = issues[issue_index].get(group)
                if isinstance(values, list) and 0 <= member_index < len(values):
                    item = values[member_index]
            if code == "invalid_source_field" and isinstance(item, dict):
                field = _evidence_field(order, _text(item.get("evidence")))
                if field:
                    item["source_field"] = field
                    _add(actions, f"recovered_issue_source:{path}")
                    continue
            drops.setdefault((issue_index, group), set()).add(member_index)
        elif code in {"missing_evidence", "unsupported_intent_evidence"} and len(parts) == 3 and parts[:2] == ["discourse", "intents"]:
            try: drop_discourse["intents"].add(int(parts[2]))
            except ValueError: pass
        elif code in {"missing_evidence", "unsupported_emotion_evidence", "object_attitude_as_emotion"} and len(parts) == 3 and parts[:2] == ["discourse", "emotions"]:
            try: drop_discourse["emotions"].add(int(parts[2]))
            except ValueError: pass
        elif code == "missing_evidence" and path == "discourse.satisfaction":
            reset_satisfaction = True
        elif code == "missing_evidence" and path == "discourse.urgency":
            reset_urgency = True

    for (issue_index, group), indexes in sorted(drops.items(), reverse=True):
        if not (0 <= issue_index < len(issues)) or not isinstance(issues[issue_index], dict):
            continue
        values = issues[issue_index].get(group)
        if not isinstance(values, list):
            continue
        for index in sorted(indexes, reverse=True):
            if 0 <= index < len(values):
                del values[index]
                _add(actions, f"dropped_invalid_candidate:issues.{issue_index}.{group}.{index}")
    for issue_index, issue in enumerate(issues):
        if not isinstance(issue, dict):
            continue
        _deduplicate(issue, issue_index, actions)
        for group, limit in ISSUE_GROUP_LIMITS.items():
            values = issue.get(group)
            if isinstance(values, list) and len(values) > limit:
                del values[limit:]
                _add(actions, f"truncated_issue_group:issues.{issue_index}.{group}")
    if len(issues) > ISSUE_LIMIT:
        del issues[ISSUE_LIMIT:]
        _add(actions, "truncated_issues_to_limit")
    for group, indexes in drop_discourse.items():
        values = discourse.get(group)
        if not isinstance(values, list):
            continue
        for index in sorted(indexes, reverse=True):
            if 0 <= index < len(values):
                del values[index]
                _add(actions, f"dropped_unverified_evidence:discourse.{group}.{index}")
    if reset_satisfaction:
        discourse["satisfaction"] = {
            "label": "unknown", "target": "", "source_field": "", "evidence": "",
        }
        _add(actions, "reset_unverified_satisfaction")
    if reset_urgency:
        discourse["urgency"] = {"level": "normal", "source_field": "", "evidence": ""}
        _add(actions, "reset_unverified_urgency")
    cleaned["issues"], cleaned["discourse"] = issues, discourse
    return cleaned, actions


def validate_issue_semantic_output(order, semantic, parse_warnings=None):
    order = order if isinstance(order, dict) else {}
    semantic = semantic if isinstance(semantic, dict) else {}
    warnings = []
    for warning in parse_warnings or []:
        _add(warnings, _text(warning))
    if not _text(order.get("doc_id")) and "doc_id" in order:
        _add(warnings, "missing_doc_id")
    if not any(_text(order.get(field)) for field in SOURCE_FIELDS):
        _add(warnings, "empty_semantic_text")
    if not _text(semantic.get("event_summary")):
        _add(warnings, "empty_event_summary")
    issues = semantic.get("issues") if isinstance(semantic.get("issues"), list) else []
    if not issues:
        _add(warnings, "empty_issues")
    if len(issues) > ISSUE_LIMIT:
        _add(warnings, "issue_limit_exceeded")
    for issue_index, issue in enumerate(issues):
        if not isinstance(issue, dict):
            _add(warnings, f"malformed_issue:{issue_index}")
            continue
        if _text(issue.get("time_scope")) not in TIME_SCOPES:
            _add(warnings, f"invalid_time_scope:issues.{issue_index}")
        count, seen = 0, set()
        for group in ISSUE_GROUPS:
            values = issue.get(group)
            if not isinstance(values, list):
                _add(warnings, f"malformed_issue_group:issues.{issue_index}.{group}")
                continue
            if len(values) > ISSUE_GROUP_LIMITS[group]:
                _add(warnings, f"issue_group_limit_exceeded:issues.{issue_index}.{group}")
            for member_index, item in enumerate(values):
                path = _path(issue_index, group, member_index)
                _grounding_errors(order, item, path, warnings)
                if not isinstance(item, dict):
                    continue
                count += 1
                surface, evidence = _text(item.get("surface")), _text(item.get("evidence"))
                field = _source_field(item)
                key = (_text(item.get("type")) if group == "locations" else group, surface, field, evidence)
                if key in seen:
                    _add(warnings, f"duplicate_issue_member:{path}")
                seen.add(key)
                if group in {"objects", "problem_behaviors"} and surface in _GENERIC:
                    _add(warnings, f"generic_entity:{path}")
                if group == "problem_behaviors":
                    if _is_request_only_behavior(order, field, evidence, surface, surface):
                        _add(warnings, f"request_action_as_behavior:{path}")
                    elif _is_normal_service_action_behavior(evidence, surface, surface):
                        _add(warnings, f"normal_service_action_as_behavior:{path}")
                if group == "locations":
                    location_type = _text(item.get("type"))
                    if location_type not in LOCATION_TYPES:
                        _add(warnings, f"invalid_location_type:{path}")
                    elif location_type == "road" and not _is_named_road(surface):
                        _add(warnings, f"road_poi_conflict:{path}")
                    elif location_type == "intersection" and not _is_strict_intersection(evidence or surface):
                        _add(warnings, f"intersection_shape_conflict:{path}")
                    elif location_type == "poi" and _is_invalid_poi_shape(surface):
                        _add(warnings, f"poi_shape_conflict:{path}")
        if not count:
            _add(warnings, f"empty_issue:{issue_index}")

    discourse = semantic.get("discourse") if isinstance(semantic.get("discourse"), dict) else {}
    for group in ("intents", "emotions"):
        values = discourse.get(group) if isinstance(discourse.get(group), list) else []
        for index, item in enumerate(values):
            path = f"discourse.{group}.{index}"
            _grounding_errors(order, item, path, warnings, require_surface=False)
            label = _text(item.get("label")) if isinstance(item, dict) else ""
            evidence = _text(item.get("evidence")) if isinstance(item, dict) else ""
            if group == "intents" and not _intent_evidence_supported(order, label, evidence):
                _add(warnings, f"unsupported_intent_evidence:{path}")
            if group == "emotions":
                if any(marker in evidence for marker in _OBJECT_ATTITUDE_MARKERS):
                    _add(warnings, f"object_attitude_as_emotion:{path}")
                elif not _emotion_evidence_supported(order, label, evidence):
                    _add(warnings, f"unsupported_emotion_evidence:{path}")

    satisfaction = discourse.get("satisfaction") if isinstance(discourse.get("satisfaction"), dict) else {}
    sat_label = _text(satisfaction.get("label")) or "unknown"
    sat_target, sat_field = _text(satisfaction.get("target")), _source_field(satisfaction)
    sat_evidence = _text(satisfaction.get("evidence"))
    if sat_label == "unknown":
        if sat_target or sat_field or sat_evidence:
            _add(warnings, "unsupported_satisfaction_evidence")
    else:
        if not sat_target or not sat_evidence:
            _add(warnings, "satisfaction_missing_target_or_evidence")
        if sat_target in _INVALID_SATISFACTION_TARGETS:
            _add(warnings, "invalid_satisfaction_target")
        if sat_evidence in {"谢谢", "感谢", "感谢转交", "请优先处理，谢谢"}:
            _add(warnings, "template_politeness_as_satisfaction")
        if not _satisfaction_evidence_supported(sat_label, sat_evidence):
            _add(warnings, "unsupported_satisfaction_evidence")
        if sat_field not in SOURCE_FIELDS or not _contains(_text(order.get(sat_field)), sat_evidence):
            _add(warnings, "missing_evidence:discourse.satisfaction")

    urgency = discourse.get("urgency") if isinstance(discourse.get("urgency"), dict) else {}
    urgency_level, urgency_field = _text(urgency.get("level")) or "normal", _source_field(urgency)
    urgency_evidence = _text(urgency.get("evidence"))
    if urgency_level == "normal":
        if urgency_field or urgency_evidence:
            _add(warnings, "template_priority_as_urgency")
    else:
        if not urgency_evidence:
            _add(warnings, "urgency_missing_evidence")
        elif urgency_field not in SOURCE_FIELDS or not _contains(_text(order.get(urgency_field)), urgency_evidence):
            _add(warnings, "missing_evidence:discourse.urgency")
        if urgency_evidence in {"优先处理", "请优先处理"}:
            _add(warnings, "template_priority_as_urgency")

    content = _text(order.get("case_content_clean"))
    summary = _text(semantic.get("event_summary"))
    searchable = "".join(_text(order.get(field)) for field in SOURCE_FIELD_ORDER)
    objects = sum((issue.get("objects", []) for issue in issues if isinstance(issue, dict)), [])
    behaviors = sum((issue.get("problem_behaviors", []) for issue in issues if isinstance(issue, dict)), [])
    pois = [
        item for issue in issues if isinstance(issue, dict)
        for item in issue.get("locations", []) if isinstance(item, dict) and item.get("type") == "poi"
    ]
    if any(marker in searchable for marker in _ANOMALY_MARKERS) and not objects:
        _add(warnings, "semantic_gap:problem_objects")
    if any(marker in searchable for marker in _ANOMALY_MARKERS) and not behaviors:
        _add(warnings, "semantic_gap:problem_behaviors")
    if any(hint in searchable for hint in _POI_GAP_HINTS) and not pois:
        _add(warnings, "semantic_gap:pois")
    if any(marker in content for marker in _HISTORY_MARKERS) and any(marker in content for marker in _CURRENT_MARKERS):
        if any(token in summary for token in ("已处理", "已解决")) and not any(token in summary for token in ("不认可", "仍未", "再次")):
            _add(warnings, "possible_history_contamination")

    if any(warning.startswith(REJECT_PREFIXES) for warning in warnings):
        status = "rejected"
    elif any(warning.startswith(WHOLE_RECORD_REPAIR_PREFIXES) for warning in warnings):
        status = "repair_required"
    elif warnings:
        status = "accepted_with_warnings"
    else:
        status = "accepted"
    repair_fields = []
    for warning in warnings:
        field = _repair_field(warning)
        _add(repair_fields, field)
    return {"status": status, "warnings": warnings, "repair_fields": repair_fields}
