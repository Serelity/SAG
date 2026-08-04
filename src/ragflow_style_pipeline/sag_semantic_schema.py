"""Tolerant parsing and stable normalization for work-order semantic output."""

import json
import re


ENTITY_GROUPS = (
    "problem_objects",
    "problem_behaviors",
    "roads",
    "intersections",
    "pois",
)
SOURCE_FIELDS = {
    "title_clean",
    "case_content_clean",
    "case_goal_clean",
    "address_detail_clean",
}
INTENTS = {"投诉", "举报", "求助", "咨询", "建议", "表扬", "催办", "反馈", "其他"}
EMOTIONS = {"愤怒", "不满", "焦虑", "无奈", "悲伤", "感谢", "认可"}
SATISFACTION_LABELS = {"satisfied", "dissatisfied", "mixed", "unknown"}
URGENCY_LEVELS = {"normal", "high", "critical"}
GROUP_LIMITS = {
    "problem_objects": 3,
    "problem_behaviors": 4,
    "roads": 4,
    "intersections": 2,
    "pois": 4,
}
INTENTS_LIMIT = 3
EMOTIONS_LIMIT = 2

_FENCED_JSON = re.compile(
    r"^```(?:json)?[ \t]*(?:\r?\n)?(?P<body>.*?)(?:\r?\n)?```$",
    re.IGNORECASE | re.DOTALL,
)


def _text(value):
    return value if isinstance(value, str) else ""


def _without_confidence(value):
    if isinstance(value, dict):
        return {
            key: _without_confidence(item)
            for key, item in value.items()
            if key != "confidence"
        }
    if isinstance(value, list):
        return [_without_confidence(item) for item in value]
    return value


def _warn(warnings, code):
    if code not in warnings:
        warnings.append(code)


def _normalize_entity_item(item):
    source_field = item.get("source_field")
    if not isinstance(source_field, str):
        source_field = item.get("field")
    if isinstance(source_field, str) and (
        source_field == "case_content_windows"
        or source_field.startswith("case_content_windows.")
    ):
        source_field = "case_content_clean"
    return {
        "surface": _text(item.get("surface")),
        "canonical": _text(item.get("canonical")),
        "source_field": _text(source_field),
        "evidence": _text(item.get("evidence")),
    }


def _normalize_entities(value, warnings):
    entities = value if isinstance(value, dict) else {}
    normalized = {}
    for group in ENTITY_GROUPS:
        items = entities.get(group, [])
        if not isinstance(items, list):
            _warn(warnings, f"malformed_entity_group:{group}")
            normalized[group] = []
            continue

        group_items = []
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                _warn(warnings, f"malformed_entity_item:{group}:{index}")
                continue
            group_items.append(_normalize_entity_item(item))

        limit = GROUP_LIMITS[group]
        if len(items) > limit:
            _warn(warnings, f"group_limit_exceeded:{group}")
        normalized[group] = group_items[:limit]
    return normalized


def _normalize_intents(value, warnings):
    if not isinstance(value, list):
        _warn(warnings, "malformed_intents")
        return []

    normalized = []
    for item in value:
        label = item.get("label") if isinstance(item, dict) else None
        if label not in INTENTS:
            _warn(warnings, "invalid_intent_label")
            continue
        normalized.append({
            "label": label,
            "evidence": _text(item.get("evidence")),
        })
    if len(value) > INTENTS_LIMIT:
        _warn(warnings, "intents_limit_exceeded")
    return normalized[:INTENTS_LIMIT]


def _normalize_emotions(value, warnings):
    if not isinstance(value, list):
        _warn(warnings, "malformed_emotions")
        return []

    normalized = []
    for item in value:
        label = item.get("label") if isinstance(item, dict) else None
        if label not in EMOTIONS:
            _warn(warnings, "invalid_emotion_label")
            continue
        intensity = item.get("intensity")
        if type(intensity) is not int or intensity not in (1, 2, 3):
            intensity = 1
            _warn(warnings, "invalid_emotion_intensity")
        normalized.append({
            "label": label,
            "intensity": intensity,
            "evidence": _text(item.get("evidence")),
        })
    if len(value) > EMOTIONS_LIMIT:
        _warn(warnings, "emotions_limit_exceeded")
    return normalized[:EMOTIONS_LIMIT]


def _normalize_satisfaction(value, warnings):
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
        "evidence": _text(value.get("evidence")),
    }


def _normalize_urgency(value, warnings):
    if not isinstance(value, dict):
        _warn(warnings, "malformed_urgency")
        value = {}
    level = value.get("level", "normal")
    if level not in URGENCY_LEVELS:
        level = "normal"
        _warn(warnings, "invalid_urgency_level")
    return {
        "level": level,
        "evidence": _text(value.get("evidence")),
    }


def _normalize(value):
    warnings = []
    value = _without_confidence(value) if isinstance(value, dict) else {}
    entities = value.get("entities", {})
    if not isinstance(entities, dict):
        _warn(warnings, "malformed_entities")
        entities = {}
    discourse = value.get("discourse", {})
    if not isinstance(discourse, dict):
        _warn(warnings, "malformed_discourse")
        discourse = {}

    normalized = {
        "event_summary": _text(value.get("event_summary")),
        "entities": _normalize_entities(entities, warnings),
        "discourse": {
            "intents": _normalize_intents(discourse.get("intents", []), warnings),
            "emotions": _normalize_emotions(discourse.get("emotions", []), warnings),
            "satisfaction": _normalize_satisfaction(
                discourse.get("satisfaction", {}), warnings
            ),
            "urgency": _normalize_urgency(discourse.get("urgency", {}), warnings),
        },
    }
    return normalized, warnings


def normalize_semantic_output(value):
    """Return the stable business schema, discarding repair-oriented warnings."""
    normalized, _ = _normalize(value)
    return normalized


def _strip_single_fence(text):
    stripped = text.strip()
    match = _FENCED_JSON.fullmatch(stripped)
    return match.group("body") if match else stripped


def _first_json_object(text):
    search_from = 0
    while True:
        start = text.find("{", search_from)
        if start < 0:
            return None

        depth = 0
        in_string = False
        escaped = False
        for end in range(start, len(text)):
            current = text[end]
            if in_string:
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == '"':
                    in_string = False
                continue
            if current == '"':
                in_string = True
            elif current == "{":
                depth += 1
            elif current == "}":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(text[start:end + 1])
                    except json.JSONDecodeError:
                        search_from = end + 1
                        break
                    if isinstance(value, dict):
                        return value
                    search_from = end + 1
                    break
        else:
            return None


def parse_semantic_json(text):
    """Parse the first complete JSON object and return normalization warnings."""
    if not isinstance(text, str):
        return normalize_semantic_output({}), ["json_parse_failed"]
    value = _first_json_object(_strip_single_fence(text))
    if value is None:
        return normalize_semantic_output({}), ["json_parse_failed"]
    return _normalize(value)
