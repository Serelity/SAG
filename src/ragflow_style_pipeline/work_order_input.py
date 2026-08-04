import hashlib
import json
import re
from pathlib import Path

from ragflow_style_pipeline.sag_entities import clean_value

CLEAN_FIELDS = ("title_clean", "case_content_clean", "case_goal_clean", "address_detail_clean")
LEGACY_LABELS = {
    "title_clean": "标题",
    "case_content_clean": "诉求内容",
    "case_goal_clean": "诉求目标",
    "address_detail_clean": "地址详情",
}


class WorkOrderInputError(ValueError):
    pass


def _legacy_field(text, label):
    labels = "|".join(re.escape(value) for value in ["标题", "诉求类型", "诉求内容", "诉求目标", "业务分类", "所属区域", "来电时间", "来源渠道", "地址详情"])
    match = re.search(rf"(?:^|\n){re.escape(label)}：(.*?)(?=\n(?:{labels})：|$)", text, re.S)
    return clean_value(match.group(1)) if match else ""


def content_hash(order):
    payload = {field: clean_value(order.get(field)) for field in CLEAN_FIELDS}
    payload["metadata"] = order.get("metadata") if isinstance(order.get("metadata"), dict) else {}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def normalize_work_order(document):
    if not isinstance(document, dict):
        raise WorkOrderInputError("invalid_document_type")
    metadata = document.get("metadata") if isinstance(document.get("metadata"), dict) else {}
    legacy_text = clean_value(document.get("text"))
    order = {
        "doc_id": clean_value(document.get("doc_id")),
        "metadata": metadata,
    }
    for field in CLEAN_FIELDS:
        value = clean_value(document.get(field))
        if not value and legacy_text:
            value = _legacy_field(legacy_text, LEGACY_LABELS[field])
        order[field] = value
    if not order["doc_id"]:
        raise WorkOrderInputError("missing_doc_id")
    if not any(order[field] for field in CLEAN_FIELDS):
        raise WorkOrderInputError("empty_semantic_text")
    order["chunk_text"] = "\n".join(
        f"{label}：{order[field]}" for field, label in LEGACY_LABELS.items() if order[field]
    )
    order["content_hash"] = content_hash(order)
    return order


def read_work_orders(path, limit=None):
    if limit is not None and limit < 0:
        raise WorkOrderInputError("invalid_limit")
    if limit == 0:
        return []
    rows = []
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                rows.append(normalize_work_order(json.loads(line)))
            except (json.JSONDecodeError, WorkOrderInputError) as exc:
                raise WorkOrderInputError(f"line_{line_number}:{exc}") from exc
            if limit is not None and len(rows) >= limit:
                break
    return rows
