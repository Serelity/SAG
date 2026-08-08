"""Conservative parser for the sole Qwen entity-output contract."""

from __future__ import annotations

import json
import re

from .constants import ENTITY_ROLES


MAX_RESPONSE_CHARS = 32_000
MAX_ISSUES = 16
MAX_VALUES_PER_ROLE = 24
MAX_VALUE_CHARS = 256
_JSON_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*([\s\S]*?)\s*```\s*$", re.IGNORECASE)


class EntitySchemaError(ValueError):
    """A model response failed the fixed schema; message is always a safe code."""


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise EntitySchemaError("duplicate_json_key")
        result[key] = value
    return result


def _single_balanced_object(value: str) -> str:
    text = value.strip()
    if not text.startswith("{"):
        raise EntitySchemaError("json_object_required")
    depth = 0
    in_string = False
    escaped = False
    end = None
    for index, character in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
        elif character in "]}":
            depth -= 1
            if depth < 0:
                raise EntitySchemaError("unbalanced_json")
            if depth == 0:
                end = index + 1
                break
    if in_string or depth != 0 or end is None:
        raise EntitySchemaError("truncated_json")
    if text[end:].strip():
        raise EntitySchemaError("trailing_content")
    return text[:end]


def _remove_trailing_commas(value: str) -> str:
    result = []
    in_string = False
    escaped = False
    index = 0
    while index < len(value):
        character = value[index]
        if in_string:
            result.append(character)
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            index += 1
            continue
        if character == '"':
            in_string = True
            result.append(character)
            index += 1
            continue
        if character == ",":
            lookahead = index + 1
            while lookahead < len(value) and value[lookahead].isspace():
                lookahead += 1
            if lookahead < len(value) and value[lookahead] in "}]":
                index += 1
                continue
        result.append(character)
        index += 1
    return "".join(result)


def validate_entity_payload(value: object) -> dict:
    if not isinstance(value, dict) or set(value) != {"issues"}:
        raise EntitySchemaError("invalid_root")
    issues = value["issues"]
    if not isinstance(issues, list):
        raise EntitySchemaError("issues_not_array")
    if len(issues) > MAX_ISSUES:
        raise EntitySchemaError("too_many_issues")
    for issue in issues:
        if not isinstance(issue, dict) or set(issue) != set(ENTITY_ROLES):
            raise EntitySchemaError("invalid_issue_keys")
        for role in ENTITY_ROLES:
            candidates = issue[role]
            if not isinstance(candidates, list):
                raise EntitySchemaError("role_not_array")
            if len(candidates) > MAX_VALUES_PER_ROLE:
                raise EntitySchemaError("too_many_role_values")
            for candidate in candidates:
                if not isinstance(candidate, str):
                    raise EntitySchemaError("role_value_not_string")
                if len(candidate) > MAX_VALUE_CHARS:
                    raise EntitySchemaError("role_value_too_long")
    return value


def parse_model_output(raw_response: str) -> dict:
    """Parse JSON with only fenced-object and trailing-comma recovery."""
    if not isinstance(raw_response, str):
        raise EntitySchemaError("response_not_string")
    if len(raw_response) > MAX_RESPONSE_CHARS:
        raise EntitySchemaError("response_too_large")
    fenced = _JSON_FENCE_RE.fullmatch(raw_response)
    candidate = fenced.group(1) if fenced else raw_response
    candidate = _remove_trailing_commas(_single_balanced_object(candidate))
    try:
        value = json.loads(
            candidate,
            object_pairs_hook=_unique_object,
            parse_constant=lambda _token: (_ for _ in ()).throw(
                EntitySchemaError("nonfinite_number")
            ),
        )
    except EntitySchemaError:
        raise
    except (json.JSONDecodeError, RecursionError) as exc:
        raise EntitySchemaError("invalid_json") from exc
    return validate_entity_payload(value)
