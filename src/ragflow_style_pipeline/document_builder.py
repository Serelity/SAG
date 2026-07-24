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


def build_text(row):
    """Create the retrievable text body for one order row."""
    lines = []

    service_object_type = clean_value(row.get("service_object_type"))
    case_content = clean_value(row.get("case_content"))
    case_goal = clean_value(row.get("case_goal"))

    if service_object_type:
        lines.append(f"诉求类型：{service_object_type}")
    if case_content:
        lines.append(f"诉求内容：{case_content}")
    if case_goal:
        lines.append(f"诉求目标：{case_goal}")

    category = _join_non_empty(
        [
            clean_value(row.get("case_accord_type_one_name")),
            clean_value(row.get("case_accord_type_two_name")),
            clean_value(row.get("case_accord_type_three_name")),
        ],
        " / ",
    )
    if category:
        lines.append(f"业务分类：{category}")

    area = _join_non_empty(
        [
            clean_value(row.get("area_code_city")),
            clean_value(row.get("area_code_area")),
            clean_value(row.get("area_code_street")),
        ],
        " / ",
    )
    if area:
        lines.append(f"所属区域：{area}")

    call_time = clean_value(row.get("call_time"))
    if call_time:
        lines.append(f"来电时间：{call_time}")

    order_source = clean_value(row.get("order_source"))
    if order_source:
        lines.append(f"来源渠道：{order_source}")

    return "\n".join(lines)


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
    """Build one JSONL-ready RAG document and redaction statistics."""
    counts = Counter()
    text, text_counts = redact_text(build_text(row))
    counts.update(text_counts)

    metadata = build_metadata(row)
    redacted_metadata = {}
    for key, value in metadata.items():
        redacted_value, value_counts = redact_text(value)
        redacted_metadata[key] = redacted_value
        counts.update(value_counts)

    return {"doc_id": build_doc_id(row), "text": text, "metadata": redacted_metadata}, counts
