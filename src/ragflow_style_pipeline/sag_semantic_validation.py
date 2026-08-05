"""Deterministic quality gates for normalized work-order semantics."""

from copy import deepcopy

from ragflow_style_pipeline.sag_semantic_schema import ENTITY_GROUPS, GROUP_LIMITS, SOURCE_FIELDS

STATUSES = {"accepted", "accepted_with_warnings", "repair_required", "rejected"}
REPAIR_PREFIXES = (
    "json_parse_failed", "empty_event_summary", "missing_evidence:", "invalid_source_field:",
    "surface_evidence_mismatch:", "empty_canonical:", "generic_entity:",
    "duplicate_entity:", "road_poi_conflict:", "intersection_shape_conflict:",
    "request_action_as_behavior:", "canonical_evidence_conflict:",
    "satisfaction_missing_target_or_evidence", "template_politeness_as_satisfaction",
    "urgency_missing_evidence", "template_priority_as_urgency",
    "possible_history_contamination",
)
REJECT_PREFIXES = ("missing_doc_id", "empty_semantic_text", "repair_failed")
_GENERIC = {"问题", "情况", "事情", "相关部门", "工作人员", "道路", "马路边", "小区", "地点"}
_REQUEST_ACTIONS = {"处理", "清理", "维修", "修剪", "拆除", "解决", "调查", "协调", "整改", "回复", "答复"}
_POI_HINTS = ("小区", "新村", "花园", "家园", "公园", "学校", "医院", "市场", "商场", "广场", "北门", "南门", "东门", "西门")
_ROAD_SUFFIXES = ("路", "街", "大道", "巷", "弄", "线")
_INTERSECTION_HINTS = ("路口", "交叉口", "交界处", "交汇处", "与")
_HISTORY_MARKERS = ("部门答复", "处理结果", "前期反映", "原工单", "答复如下")
_CURRENT_MARKERS = ("其不认可", "仍未解决", "现服务对象表示", "再次要求", "现再次反映")


def _text(value):
    return value if isinstance(value, str) else ""


def _add(items, value):
    if value and value not in items:
        items.append(value)


def _source_field(item):
    return _text(item.get("source_field") or item.get("field"))


def _repair_field(warning):
    if ":" in warning:
        suffix = warning.split(":", 1)[1]
        if suffix.startswith("entities.") or suffix.startswith("discourse."):
            return suffix
    if warning.startswith("satisfaction_") or "satisfaction" in warning:
        return "discourse.satisfaction"
    if "urgency" in warning:
        return "discourse.urgency"
    if warning in {"empty_event_summary", "possible_history_contamination"}:
        return "event_summary"
    return ""


def _contains(source, evidence):
    return bool(evidence and evidence in source)


def sanitize_semantic_output(semantic, warnings):
    """Conservatively remove invalid optional candidates without losing an event.

    A bad optional entity or discourse attribute should not reject an otherwise
    useful work order.  Parse failures and history contamination are not
    sanitized because they require a model repair or rejection.
    """
    cleaned = deepcopy(semantic if isinstance(semantic, dict) else {})
    entities = cleaned.get("entities") if isinstance(cleaned.get("entities"), dict) else {}
    discourse = cleaned.get("discourse") if isinstance(cleaned.get("discourse"), dict) else {}
    drop_entities = {}
    drop_discourse = {"intents": set(), "emotions": set()}
    reset_satisfaction = False
    reset_urgency = False
    sanitation_warnings = []
    align_surfaces = {}
    align_canonicals = {}
    entity_drop_codes = {
        "invalid_source_field", "missing_evidence", "empty_canonical",
        "generic_entity", "duplicate_entity", "road_poi_conflict",
        "intersection_shape_conflict", "request_action_as_behavior",
    }

    for warning in warnings or []:
        if not isinstance(warning, str) or ":" not in warning:
            if warning in {"satisfaction_missing_target_or_evidence", "template_politeness_as_satisfaction"}:
                reset_satisfaction = True
            elif warning in {"urgency_missing_evidence", "template_priority_as_urgency"}:
                reset_urgency = True
            continue
        code, path = warning.split(":", 1)
        parts = path.split(".")
        if code in entity_drop_codes and len(parts) == 3 and parts[0] == "entities":
            try:
                drop_entities.setdefault(parts[1], set()).add(int(parts[2]))
            except ValueError:
                pass
        elif code == "surface_evidence_mismatch" and len(parts) == 3 and parts[0] == "entities":
            try:
                align_surfaces.setdefault(parts[1], set()).add(int(parts[2]))
            except ValueError:
                pass
        elif code == "canonical_evidence_conflict" and len(parts) == 3 and parts[0] == "entities":
            try:
                align_canonicals.setdefault(parts[1], set()).add(int(parts[2]))
            except ValueError:
                pass
        elif code == "missing_evidence" and len(parts) == 3 and parts[0] == "discourse":
            if parts[1] in drop_discourse:
                try:
                    drop_discourse[parts[1]].add(int(parts[2]))
                except ValueError:
                    pass
        elif code == "missing_evidence" and path == "discourse.satisfaction":
            reset_satisfaction = True
        elif code == "missing_evidence" and path == "discourse.urgency":
            reset_urgency = True

    for group, indexes in align_surfaces.items():
        items = entities.get(group)
        dropped = drop_entities.get(group, set())
        if not isinstance(items, list):
            continue
        for index in sorted(indexes):
            if index in dropped or not (0 <= index < len(items)) or not isinstance(items[index], dict):
                continue
            evidence = _text(items[index].get("evidence"))
            if evidence:
                items[index]["surface"] = evidence
                _add(sanitation_warnings, f"aligned_surface_to_evidence:entities.{group}.{index}")

    for group, indexes in align_canonicals.items():
        items = entities.get(group)
        dropped = drop_entities.get(group, set())
        if not isinstance(items, list):
            continue
        for index in sorted(indexes):
            if index in dropped or not (0 <= index < len(items)) or not isinstance(items[index], dict):
                continue
            surface = _text(items[index].get("surface")) or _text(items[index].get("evidence"))
            if surface:
                items[index]["canonical"] = surface
                _add(sanitation_warnings, f"aligned_canonical_to_surface:entities.{group}.{index}")

    for group, indexes in drop_entities.items():
        items = entities.get(group)
        if not isinstance(items, list):
            continue
        for index in sorted(indexes, reverse=True):
            if 0 <= index < len(items):
                del items[index]
                _add(sanitation_warnings, f"dropped_invalid_candidate:entities.{group}.{index}")

    for group, indexes in drop_discourse.items():
        items = discourse.get(group)
        if not isinstance(items, list):
            continue
        for index in sorted(indexes, reverse=True):
            if 0 <= index < len(items):
                del items[index]
                _add(sanitation_warnings, f"dropped_unverified_evidence:discourse.{group}.{index}")

    if reset_satisfaction:
        discourse["satisfaction"] = {"label": "unknown", "target": "", "evidence": ""}
        _add(sanitation_warnings, "reset_unverified_satisfaction")
    if reset_urgency:
        discourse["urgency"] = {"level": "normal", "evidence": ""}
        _add(sanitation_warnings, "reset_unverified_urgency")
    cleaned["entities"] = entities
    cleaned["discourse"] = discourse
    return cleaned, sanitation_warnings


def validate_semantic_output(order, semantic, parse_warnings=None):
    """Validate normalized semantics and return stable status/warnings/repair paths."""
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

    entities = semantic.get("entities") if isinstance(semantic.get("entities"), dict) else {}
    seen = set()
    for group in ENTITY_GROUPS:
        items = entities.get(group, [])
        if not isinstance(items, list):
            _add(warnings, f"malformed_entity_group:{group}")
            continue
        if len(items) > GROUP_LIMITS[group]:
            _add(warnings, f"group_limit_exceeded:{group}")
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                _add(warnings, f"malformed_entity_item:{group}:{index}")
                continue
            path = f"entities.{group}.{index}"
            field = _source_field(item)
            evidence = _text(item.get("evidence"))
            surface = _text(item.get("surface"))
            canonical = _text(item.get("canonical"))
            if field not in SOURCE_FIELDS:
                _add(warnings, f"invalid_source_field:{path}")
                source = ""
            else:
                source = _text(order.get(field))
            if not _contains(source, evidence):
                _add(warnings, f"missing_evidence:{path}")
            if surface and evidence and surface not in evidence and evidence not in surface:
                _add(warnings, f"surface_evidence_mismatch:{path}")
            if not canonical:
                _add(warnings, f"empty_canonical:{path}")
            if canonical in _GENERIC or surface in _GENERIC:
                _add(warnings, f"generic_entity:{path}")
            key = (group, canonical, evidence)
            if key in seen:
                _add(warnings, f"duplicate_entity:{path}")
            seen.add(key)
            if group == "roads":
                if any(hint in (surface + evidence) for hint in _POI_HINTS) or not canonical.endswith(_ROAD_SUFFIXES):
                    _add(warnings, f"road_poi_conflict:{path}")
            if group == "intersections":
                text = surface + evidence + canonical
                if not any(hint in text for hint in _INTERSECTION_HINTS):
                    _add(warnings, f"intersection_shape_conflict:{path}")
            if group == "problem_behaviors":
                compact = canonical.strip("希望请求要求建议尽快予以进行")
                goal_only = field == "case_goal_clean" or (evidence and evidence in _text(order.get("case_goal_clean")))
                if goal_only and (canonical in _REQUEST_ACTIONS or compact in _REQUEST_ACTIONS):
                    _add(warnings, f"request_action_as_behavior:{path}")
            comparison = surface + evidence
            if canonical and evidence and len(canonical) > 2 and not any(ch in comparison for ch in canonical):
                _add(warnings, f"canonical_evidence_conflict:{path}")

    discourse = semantic.get("discourse") if isinstance(semantic.get("discourse"), dict) else {}
    for group in ("intents", "emotions"):
        for index, item in enumerate(discourse.get(group, []) if isinstance(discourse.get(group), list) else []):
            if not isinstance(item, dict):
                continue
            evidence = _text(item.get("evidence"))
            if not evidence or not any(evidence in _text(order.get(field)) for field in SOURCE_FIELDS):
                _add(warnings, f"missing_evidence:discourse.{group}.{index}")

    satisfaction = discourse.get("satisfaction") if isinstance(discourse.get("satisfaction"), dict) else {}
    sat_label = _text(satisfaction.get("label")) or "unknown"
    sat_target = _text(satisfaction.get("target"))
    sat_evidence = _text(satisfaction.get("evidence"))
    if sat_label != "unknown" and (not sat_target or not sat_evidence):
        _add(warnings, "satisfaction_missing_target_or_evidence")
    if sat_label != "unknown" and sat_evidence in {"谢谢", "感谢", "感谢转交", "请优先处理，谢谢"}:
        _add(warnings, "template_politeness_as_satisfaction")
    if sat_evidence and not any(sat_evidence in _text(order.get(field)) for field in SOURCE_FIELDS):
        _add(warnings, "missing_evidence:discourse.satisfaction")

    urgency = discourse.get("urgency") if isinstance(discourse.get("urgency"), dict) else {}
    urgency_level = _text(urgency.get("level")) or "normal"
    urgency_evidence = _text(urgency.get("evidence"))
    if urgency_level != "normal" and not urgency_evidence:
        _add(warnings, "urgency_missing_evidence")
    if urgency_evidence and not any(urgency_evidence in _text(order.get(field)) for field in SOURCE_FIELDS):
        _add(warnings, "missing_evidence:discourse.urgency")
    if urgency_level != "normal" and urgency_evidence in {"优先处理", "请优先处理"}:
        _add(warnings, "template_priority_as_urgency")

    content = _text(order.get("case_content_clean"))
    summary = _text(semantic.get("event_summary"))
    if any(marker in content for marker in _HISTORY_MARKERS) and any(marker in content for marker in _CURRENT_MARKERS):
        if any(token in summary for token in ("已处理", "已解决")) and not any(token in summary for token in ("不认可", "仍未", "再次")):
            _add(warnings, "possible_history_contamination")

    if any(warning.startswith(REJECT_PREFIXES) for warning in warnings):
        status = "rejected"
    elif any(warning.startswith(REPAIR_PREFIXES) for warning in warnings):
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
