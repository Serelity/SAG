"""SAG retrieval entity schema and validation for LLM candidates."""

from ragflow_style_pipeline.sag_entities import clean_value, normalize_entity_value


ALLOWED_LLM_ENTITY_TYPES = {
    "problem_object",
    "problem_behavior",
    "area",
    "street",
    "road",
    "intersection",
    "poi",
}

SOURCE_TEXT_FIELDS = ["title_clean", "case_content_clean", "case_goal_clean", "address_detail_clean"]

PROBLEM_OBJECT_ALIASES = {
    "卖菜摊子": "流动摊贩",
    "卖菜摊": "流动摊贩",
    "卖菜的": "流动摊贩",
    "路边摊": "流动摊贩",
    "流摊": "流动摊贩",
    "摊子": "流动摊贩",
    "摊点": "流动摊贩",
}

PROBLEM_BEHAVIOR_ALIASES = {
    "挡住人行道": "占道经营",
    "堵住人行道": "占道经营",
    "占用道路": "占道经营",
    "占用人行道": "占道经营",
    "占路经营": "占道经营",
    "影响通行": "影响通行",
}

GENERIC_ENTITY_VALUES = {
    "路",
    "街",
    "桥",
    "线",
    "巷",
    "弄",
    "道路",
    "关于道路",
    "政风热线",
    "常州12345热线",
    "12345热线",
    "小区",
    "关于小区",
    "导致小区",
    "该小区",
    "现小区",
    "市场",
    "要求市场",
    "本人要求市场",
    "其表示街道",
    "根据街道",
    "我街道",
    "关于街道",
}

GENERIC_PREFIXES = (
    "关于",
    "反映",
    "服务对象",
    "其表示",
    "本人要求",
    "希望",
    "要求",
    "导致",
    "现",
)


def normalize_llm_entity_value(entity_type, value):
    """Normalize an LLM entity candidate to the dictionary identity used by SAG."""
    entity_type = clean_value(entity_type)
    normalized = normalize_entity_value(value)
    if entity_type == "problem_object":
        return PROBLEM_OBJECT_ALIASES.get(normalized, normalized)
    if entity_type == "problem_behavior":
        return PROBLEM_BEHAVIOR_ALIASES.get(normalized, normalized)
    return normalized


def is_generic_entity_value(entity_type, value):
    """Return True when a value is too generic or fragmentary to be a SAG join key."""
    entity_type = clean_value(entity_type)
    normalized = normalize_llm_entity_value(entity_type, value)
    if not normalized:
        return True
    if normalized in GENERIC_ENTITY_VALUES:
        return True
    if entity_type in {"road", "street", "poi"} and len(normalized) <= 1:
        return True
    if entity_type == "road" and normalized in {"路", "街", "桥", "线", "巷", "弄"}:
        return True
    if entity_type == "poi":
        for prefix in GENERIC_PREFIXES:
            if normalized.startswith(prefix) and len(normalized) <= len(prefix) + 4:
                return True
    if entity_type == "street":
        for prefix in GENERIC_PREFIXES:
            if normalized.startswith(prefix):
                return True
    return False


def _source_text(order):
    return "\n".join(clean_value(order.get(field)) for field in SOURCE_TEXT_FIELDS if clean_value(order.get(field)))


def evidence_exists(candidate, order):
    """Return True when the candidate evidence text occurs in one source text field."""
    source_field = clean_value(candidate.get("source_field"))
    evidence_span = clean_value(candidate.get("evidence_span") or candidate.get("matched_text"))
    if not evidence_span:
        return False
    if source_field:
        return evidence_span in clean_value(order.get(source_field))
    return evidence_span in _source_text(order)


def validate_llm_candidate(candidate, order, config):
    """Validate one LLM candidate before converting it into a SAG entity link."""
    entity_type = clean_value(candidate.get("entity_type"))
    if entity_type not in ALLOWED_LLM_ENTITY_TYPES:
        return False, "unsupported_entity_type"
    entity_value = normalize_llm_entity_value(entity_type, candidate.get("entity_value"))
    if not entity_value:
        return False, "empty_entity_value"
    if is_generic_entity_value(entity_type, entity_value):
        return False, "generic_entity_value"
    confidence = float(candidate.get("confidence") or 0.0)
    min_confidence = float(config.get("min_confidence", 0.55))
    if confidence < min_confidence:
        return False, "low_confidence"
    if not evidence_exists(candidate, order):
        return False, "missing_evidence_span"
    return True, "ok"
