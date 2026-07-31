"""Build RAG documents from 12345 order rows."""

from collections import Counter
import hashlib

from ragflow_style_pipeline.pii_redactor import redact_text


NULLISH_VALUES = {"", "NULL", "null", "None", "none", "\\N"}


def is_nullish(value):
    """Return True when a TSV value should be treated as empty."""
    if value is None:
        return True
    return str(value).strip() in NULLISH_VALUES


def clean_value(value):
    """Normalize one TSV cell into a safe plain string."""
    if is_nullish(value):
        return ""
    return str(value).strip()


def _join_non_empty(values, separator):
    return separator.join(value for value in values if value)


def build_case_content(row):
    """Return the normalized raw complaint/request content."""
    return clean_value(row.get("case_content"))


def build_case_goal(row):
    """Return the normalized request goal."""
    return clean_value(row.get("case_goal"))


def build_embedding_text_from_parts(case_content, case_goal):
    """Build the text that dense embedding models should encode."""
    lines = []
    if case_content:
        lines.append(f"诉求内容：{case_content}")
    if case_goal:
        lines.append(f"诉求目标：{case_goal}")
    return "\n".join(lines)


def build_category(row):
    """Return the normalized business category path."""
    return _join_non_empty(
        [
            clean_value(row.get("case_accord_type_one_name")),
            clean_value(row.get("case_accord_type_two_name")),
            clean_value(row.get("case_accord_type_three_name")),
        ],
        " / ",
    )


def build_area(row):
    """Return the normalized area path."""
    return _join_non_empty(
        [
            clean_value(row.get("area_code_city")),
            clean_value(row.get("area_code_area")),
            clean_value(row.get("area_code_street")),
        ],
        " / ",
    )


def build_display_text_from_parts(
    service_object_type,
    case_content,
    case_goal,
    category,
    area,
    call_time,
    order_source,
):
    """Create display text from already normalized or redacted parts."""
    lines = []

    if service_object_type:
        lines.append(f"诉求类型：{service_object_type}")
    if case_content:
        lines.append(f"诉求内容：{case_content}")
    if case_goal:
        lines.append(f"诉求目标：{case_goal}")
    if category:
        lines.append(f"业务分类：{category}")
    if area:
        lines.append(f"所属区域：{area}")
    if call_time:
        lines.append(f"来电时间：{call_time}")
    if order_source:
        lines.append(f"来源渠道：{order_source}")

    return "\n".join(lines)


def build_display_text(row):
    """Create the text shown to humans and later provided to LLM context."""
    return build_display_text_from_parts(
        service_object_type=clean_value(row.get("service_object_type")),
        case_content=build_case_content(row),
        case_goal=build_case_goal(row),
        category=build_category(row),
        area=build_area(row),
        call_time=clean_value(row.get("call_time")),
        order_source=clean_value(row.get("order_source")),
    )


def build_text(row):
    """Backward-compatible full display text for one order row."""
    return build_display_text(row)


def build_derived():
    """Return reserved derived fields for future text analytics."""
    return {
        "topic_tags": [],
        "keywords": [],
        "semantic_cluster_id": "",
        "problem_object": "",
        "problem_behavior": "",
        "location_mention": "",
        "appeal_action": "",
    }


def build_metadata(row):
    """Create structured filters for one order row."""
    call_time = clean_value(row.get("call_time"))
    metadata = {
        "order_id": clean_value(row.get("order_id")),
        "service_object_type": clean_value(row.get("service_object_type")),
        "area_code_city": clean_value(row.get("area_code_city")),
        "area_code_area": clean_value(row.get("area_code_area")),
        "area_code_street": clean_value(row.get("area_code_street")),
        "type1": clean_value(row.get("case_accord_type_one_name")),
        "type2": clean_value(row.get("case_accord_type_two_name")),
        "type3": clean_value(row.get("case_accord_type_three_name")),
        "order_source": clean_value(row.get("order_source")),
        "order_type": clean_value(row.get("order_type")),
        "order_status": clean_value(row.get("order_status")),
        "call_time": call_time,
        "call_month": call_time[:7] if len(call_time) >= 7 else "",
    }
    return metadata


def build_doc_id(row):
    """Create a stable document id without exposing raw source identifiers."""
    source_id = clean_value(row.get("id")) or clean_value(row.get("order_id"))
    if not source_id:
        source_id = build_text(row)
    digest = hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:16]
    letter_digest = digest.translate(str.maketrans("0123456789abcdef", "abcdefghijklmnop"))
    return f"order_{letter_digest}"


def build_document(row):
    """Build one JSONL-ready multi-view RAG document and redaction statistics."""
    counts = Counter()

    case_content, case_content_counts = redact_text(build_case_content(row))
    counts.update(case_content_counts)

    case_goal, case_goal_counts = redact_text(build_case_goal(row))
    counts.update(case_goal_counts)

    display_parts = {}
    for key, value in {
        "service_object_type": clean_value(row.get("service_object_type")),
        "category": build_category(row),
        "area": build_area(row),
        "call_time": clean_value(row.get("call_time")),
        "order_source": clean_value(row.get("order_source")),
    }.items():
        redacted_value, value_counts = redact_text(value)
        display_parts[key] = redacted_value
        counts.update(value_counts)

    display_text = build_display_text_from_parts(
        service_object_type=display_parts["service_object_type"],
        case_content=case_content,
        case_goal=case_goal,
        category=display_parts["category"],
        area=display_parts["area"],
        call_time=display_parts["call_time"],
        order_source=display_parts["order_source"],
    )

    embedding_text = build_embedding_text_from_parts(case_content, case_goal)

    metadata = build_metadata(row)
    redacted_metadata = {}
    for key, value in metadata.items():
        redacted_value, value_counts = redact_text(value)
        redacted_metadata[key] = redacted_value
        counts.update(value_counts)

    return (
        {
            "doc_id": build_doc_id(row),
            "case_content_clean": case_content,
            "case_goal_clean": case_goal,
            "embedding_text": embedding_text,
            "display_text": display_text,
            "text": display_text,
            "metadata": redacted_metadata,
            "derived": build_derived(),
        },
        counts,
    )
