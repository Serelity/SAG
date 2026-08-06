"""Build a pure SAG-lite DuckDB database from 12345 work-order rows."""

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

from ragflow_style_pipeline.sag_entities import (
    SagEntityLink,
    clean_value,
    deduplicate_entity_links,
    extract_entities_from_order,
)


SOURCE_ORDER_COLUMNS = [
    "doc_id",
    "raw_id_hash",
    "order_id_hash",
    "title_clean",
    "case_content_clean",
    "case_goal_clean",
    "address_detail_clean",
    "call_time",
    "call_month",
    "area_code_city",
    "area_code_area",
    "area_code_street",
    "case_lnglat",
    "type1",
    "type2",
    "type3",
    "type4",
    "type5",
    "case_accord_code",
    "order_source",
    "order_type",
    "order_status",
    "service_object_type",
    "belong_dept",
    "deptName",
    "orgName",
]

SAG_EVENT_COLUMNS = [
    "event_id",
    "doc_id",
    "event_text",
    "event_time",
    "event_month",
    "event_source",
    "event_status",
    "projection_version",
]

SAG_ENTITY_COLUMNS = ["entity_id", "entity_type", "entity_value", "normalized_value"]

SAG_LINK_COLUMNS = [
    "event_id",
    "doc_id",
    "entity_id",
    "entity_type",
    "entity_value",
    "surface_form",
    "normalized_value",
    "source_field",
    "source_channel",
    "confidence",
    "matched_text",
    "validation_status",
    "prompt_version",
    "projection_version",
]

SAG_DISCOURSE_COLUMNS = [
    "event_id",
    "doc_id",
    "declared_intent",
    "inferred_intents_json",
    "intent_conflict",
    "emotions_json",
    "satisfaction",
    "satisfaction_target",
    "satisfaction_evidence",
    "urgency",
    "urgency_evidence",
    "projection_version",
]


def stable_hash(value):
    """Return a short stable hash for sensitive source identifiers."""
    return hashlib.sha256(clean_value(value).encode("utf-8")).hexdigest()[:16]


def _doc_id_from_row(row):
    doc_id = clean_value(row.get("doc_id"))
    if doc_id:
        return doc_id
    source_id = clean_value(row.get("id")) or clean_value(row.get("order_id")) or json.dumps(row, sort_keys=True)
    return f"order_{stable_hash(source_id)}"


def _metadata_from_multiview(row):
    metadata = row.get("metadata") or {}
    return metadata if isinstance(metadata, dict) else {}


def _field(row, metadata, raw_name, metadata_name=None, fallback_name=None):
    for name in [raw_name, metadata_name, fallback_name]:
        if name:
            value = clean_value(row.get(name))
            if value:
                return value
            value = clean_value(metadata.get(name))
            if value:
                return value
    return ""


def source_order_row(row):
    """Map one raw TSV row or multiview JSON document into source_orders shape."""
    metadata = _metadata_from_multiview(row)
    doc_id = _doc_id_from_row(row)
    call_time = _field(row, metadata, "call_time")
    call_month = _field(row, metadata, "call_month") or (call_time[:7] if len(call_time) >= 7 else "")
    raw_id = clean_value(row.get("id")) or doc_id
    order_id = clean_value(row.get("order_id")) or clean_value(metadata.get("order_id"))

    return {
        "doc_id": doc_id,
        "raw_id_hash": stable_hash(raw_id),
        "order_id_hash": stable_hash(order_id) if order_id else "",
        "title_clean": clean_value(row.get("title_clean")) or clean_value(row.get("title")),
        "case_content_clean": clean_value(row.get("case_content_clean")) or clean_value(row.get("case_content")),
        "case_goal_clean": clean_value(row.get("case_goal_clean")) or clean_value(row.get("case_goal")),
        "address_detail_clean": clean_value(row.get("address_detail_clean")) or clean_value(row.get("address_detail")),
        "call_time": call_time,
        "call_month": call_month,
        "area_code_city": _field(row, metadata, "area_code_city"),
        "area_code_area": _field(row, metadata, "area_code_area"),
        "area_code_street": _field(row, metadata, "area_code_street"),
        "case_lnglat": _field(row, metadata, "case_lnglat"),
        "type1": _field(row, metadata, "case_accord_type_one_name", "type1"),
        "type2": _field(row, metadata, "case_accord_type_two_name", "type2"),
        "type3": _field(row, metadata, "case_accord_type_three_name", "type3"),
        "type4": _field(row, metadata, "case_accord_type_four_name", "type4"),
        "type5": _field(row, metadata, "case_accord_type_five_name", "type5"),
        "case_accord_code": _field(row, metadata, "case_accord_code"),
        "order_source": _field(row, metadata, "order_source"),
        "order_type": _field(row, metadata, "order_type"),
        "order_status": _field(row, metadata, "order_status"),
        "service_object_type": _field(row, metadata, "service_object_type"),
        "belong_dept": _field(row, metadata, "belong_dept"),
        "deptName": _field(row, metadata, "deptName"),
        "orgName": _field(row, metadata, "orgName"),
    }


def _join_labeled_lines(items):
    lines = [f"{label}：{value}" for label, value in items if clean_value(value)]
    return "\n".join(lines)


def event_row(source_order):
    """Build one SAG event row from one normalized source order."""
    event_text = _join_labeled_lines(
        [
            ("标题", source_order.get("title_clean")),
            ("诉求内容", source_order.get("case_content_clean")),
            ("诉求目标", source_order.get("case_goal_clean")),
            ("地址详情", source_order.get("address_detail_clean")),
            ("区域", " / ".join(
                value
                for value in [
                    source_order.get("area_code_city"),
                    source_order.get("area_code_area"),
                    source_order.get("area_code_street"),
                ]
                if clean_value(value)
            )),
            ("分类", " / ".join(
                value
                for value in [
                    source_order.get("type1"),
                    source_order.get("type2"),
                    source_order.get("type3"),
                    source_order.get("type4"),
                    source_order.get("type5"),
                ]
                if clean_value(value)
            )),
            ("来电时间", source_order.get("call_time")),
        ]
    )
    return {
        "event_id": f"event_{stable_hash(source_order.get('doc_id'))}",
        "doc_id": source_order.get("doc_id", ""),
        "event_text": event_text,
        "event_time": source_order.get("call_time", ""),
        "event_month": source_order.get("call_month", ""),
        "event_source": "t_order_master",
        "event_status": source_order.get("order_status", ""),
        "projection_version": "",
    }


def _source_row_from_multiview(document):
    row = {
        "doc_id": document.get("doc_id", ""),
        "case_content_clean": document.get("case_content_clean", ""),
        "case_goal_clean": document.get("case_goal_clean", ""),
        "address_detail_clean": document.get("address_detail_clean", ""),
        "title_clean": document.get("title_clean", ""),
    }
    metadata = document.get("metadata") or {}
    if isinstance(metadata, dict):
        row.update(metadata)
    return source_order_row(row)


def read_source_rows(input_path, limit=None):
    """Read .tsv or .jsonl and return normalized source order rows."""
    input_path = Path(input_path)
    rows = []
    if input_path.suffix.lower() == ".jsonl":
        with input_path.open("r", encoding="utf-8") as input_file:
            for line in input_file:
                if line.strip():
                    rows.append(_source_row_from_multiview(json.loads(line)))
                    if limit and len(rows) >= limit:
                        break
        return rows

    with input_path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file, delimiter="\t")
        for raw_row in reader:
            rows.append(source_order_row(raw_row))
            if limit and len(rows) >= limit:
                break
    return rows


def _entity_id(entity_type, normalized_value):
    return "entity_" + stable_hash(f"{entity_type}:{normalized_value}")


def _create_table(conn, table_name, columns):
    column_sql = ", ".join(f"{column} varchar" for column in columns)
    conn.execute(f"drop table if exists {table_name}")
    conn.execute(f"create table {table_name} ({column_sql})")


def _insert_rows(conn, table_name, columns, rows):
    if not rows:
        return
    placeholders = ", ".join(["?"] * len(columns))
    conn.executemany(
        f"insert into {table_name} values ({placeholders})",
        [[row.get(column, "") for column in columns] for row in rows],
    )


def load_entity_links_jsonl(path):
    """Load legacy or semantic entity-link JSONL rows grouped by doc_id."""
    path = Path(path)
    links_by_doc = {}
    if not path.exists():
        return links_by_doc
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            row = json.loads(line)
            doc_id = clean_value(row.get("doc_id"))
            entity_type = clean_value(row.get("entity_type"))
            normalized_value = clean_value(row.get("normalized_value")) or clean_value(row.get("entity_value"))
            if doc_id and entity_type and normalized_value:
                links_by_doc.setdefault(doc_id, []).append(row)
    return links_by_doc


def load_rows_jsonl_by_doc(path):
    """Load object JSONL keyed by doc_id without printing source content."""
    rows = {}
    path = Path(path)
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not clean_value(row.get("doc_id")):
                raise ValueError(f"line_{line_number}:missing_doc_id")
            rows[clean_value(row["doc_id"])] = row
    return rows


def _link_value(link, name, default=""):
    if isinstance(link, dict):
        return link.get(name, default)
    return getattr(link, name, default)


def build_sag_db_from_orders(
    source_orders,
    db_path,
    extra_entity_links_by_doc=None,
    semantic_events_by_doc=None,
    discourse_by_doc=None,
):
    """Create DuckDB SAG tables from normalized orders and optional semantic projections."""
    import duckdb

    started_at = time.time()
    source_orders = list(source_orders)
    events = [event_row(order) for order in source_orders]
    for event in events:
        override = (semantic_events_by_doc or {}).get(event["doc_id"], {})
        validation = override.get("validation") if isinstance(override.get("validation"), dict) else {}
        semantic_event = override.get("event") if isinstance(override.get("event"), dict) else {}
        status = clean_value(override.get("validation_status")) or clean_value(validation.get("status"))
        summary = clean_value(override.get("event_text")) or clean_value(semantic_event.get("summary"))
        if summary and status in {"accepted", "accepted_with_warnings"}:
            event["event_text"] = summary
        event["projection_version"] = clean_value(override.get("projection_version"))
    event_by_doc = {event["doc_id"]: event for event in events}

    entity_rows_by_key = {}
    link_rows = []
    for order in source_orders:
        rule_links = extract_entities_from_order(order)
        extra_links = (extra_entity_links_by_doc or {}).get(order["doc_id"], [])
        # Rule links remain dataclasses; external semantic links remain dictionaries so
        # provenance fields and the deliberate absence of model confidence survive.
        combined = list(deduplicate_entity_links(rule_links)) + list(extra_links)
        seen_links = set()
        for link in combined:
            doc_id = clean_value(_link_value(link, "doc_id")) or order["doc_id"]
            entity_type = clean_value(_link_value(link, "entity_type"))
            entity_value = clean_value(_link_value(link, "entity_value"))
            normalized_value = clean_value(_link_value(link, "normalized_value")) or entity_value
            source_field = clean_value(_link_value(link, "source_field"))
            source_channel = clean_value(_link_value(link, "source_channel")) or "llm"
            matched_text = clean_value(_link_value(link, "matched_text")) or entity_value
            key = (doc_id, entity_type, normalized_value, source_field, source_channel, matched_text)
            if not entity_type or not normalized_value or key in seen_links:
                continue
            seen_links.add(key)
            entity_id = _entity_id(entity_type, normalized_value)
            entity_key = (entity_type, normalized_value)
            entity_rows_by_key.setdefault(entity_key, {
                "entity_id": entity_id,
                "entity_type": entity_type,
                "entity_value": entity_value or normalized_value,
                "normalized_value": normalized_value,
            })
            event = event_by_doc.get(doc_id, {})
            confidence = _link_value(link, "confidence", None)
            link_rows.append({
                "event_id": event.get("event_id", ""),
                "doc_id": doc_id,
                "entity_id": entity_id,
                "entity_type": entity_type,
                "entity_value": entity_value or normalized_value,
                "surface_form": clean_value(_link_value(link, "surface_form")) or entity_value or normalized_value,
                "normalized_value": normalized_value,
                "source_field": source_field,
                "source_channel": source_channel,
                "confidence": "" if confidence is None else str(confidence),
                "matched_text": matched_text,
                "validation_status": clean_value(_link_value(link, "validation_status")),
                "prompt_version": clean_value(_link_value(link, "prompt_version")),
                "projection_version": clean_value(_link_value(link, "projection_version")),
            })

    discourse_rows = []
    for doc_id, row in (discourse_by_doc or {}).items():
        if doc_id not in event_by_doc:
            continue
        discourse_rows.append({
            **{column: clean_value(row.get(column)) for column in SAG_DISCOURSE_COLUMNS},
            "event_id": event_by_doc[doc_id]["event_id"],
            "doc_id": doc_id,
        })

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with duckdb.connect(str(db_path)) as conn:
        _create_table(conn, "source_orders", SOURCE_ORDER_COLUMNS)
        _create_table(conn, "sag_events", SAG_EVENT_COLUMNS)
        _create_table(conn, "sag_entities", SAG_ENTITY_COLUMNS)
        _create_table(conn, "sag_event_entity_links", SAG_LINK_COLUMNS)
        _create_table(conn, "sag_event_discourse", SAG_DISCOURSE_COLUMNS)

        _insert_rows(conn, "source_orders", SOURCE_ORDER_COLUMNS, source_orders)
        _insert_rows(conn, "sag_events", SAG_EVENT_COLUMNS, events)
        _insert_rows(conn, "sag_entities", SAG_ENTITY_COLUMNS, list(entity_rows_by_key.values()))
        _insert_rows(conn, "sag_event_entity_links", SAG_LINK_COLUMNS, link_rows)
        _insert_rows(conn, "sag_event_discourse", SAG_DISCOURSE_COLUMNS, discourse_rows)

        conn.execute("create index if not exists idx_source_orders_doc_id on source_orders(doc_id)")
        conn.execute("create index if not exists idx_sag_events_event_id on sag_events(event_id)")
        conn.execute("create index if not exists idx_sag_events_doc_id on sag_events(doc_id)")
        conn.execute("create index if not exists idx_sag_events_month on sag_events(event_month)")
        conn.execute("create index if not exists idx_sag_entities_key on sag_entities(entity_type, normalized_value)")
        conn.execute("create index if not exists idx_sag_links_doc_id on sag_event_entity_links(doc_id)")
        conn.execute("create index if not exists idx_sag_links_event_id on sag_event_entity_links(event_id)")
        conn.execute("create index if not exists idx_sag_links_entity_id on sag_event_entity_links(entity_id)")
        conn.execute("create index if not exists idx_sag_links_entity_type on sag_event_entity_links(entity_type)")
        conn.execute("create index if not exists idx_sag_discourse_doc_id on sag_event_discourse(doc_id)")
        conn.execute("create index if not exists idx_sag_discourse_event_id on sag_event_discourse(event_id)")
        conn.execute("create index if not exists idx_sag_discourse_satisfaction on sag_event_discourse(satisfaction)")
        conn.execute("create index if not exists idx_sag_discourse_urgency on sag_event_discourse(urgency)")

    return {
        "db_path": str(db_path),
        "source_orders_loaded": len(source_orders),
        "events_loaded": len(events),
        "entities_loaded": len(entity_rows_by_key),
        "links_loaded": len(link_rows),
        "discourse_loaded": len(discourse_rows),
        "build_seconds": round(time.time() - started_at, 3),
    }


def build_sag_db(
    input_path, db_path, limit=None, entity_links_jsonl="",
    semantic_events_jsonl="", discourse_jsonl="",
):
    """Read source rows and build SAG tables without importing model code."""
    rows = read_source_rows(input_path, limit=limit)
    extra_links = load_entity_links_jsonl(entity_links_jsonl) if entity_links_jsonl else None
    semantic_events = load_rows_jsonl_by_doc(semantic_events_jsonl) if semantic_events_jsonl else None
    discourse = load_rows_jsonl_by_doc(discourse_jsonl) if discourse_jsonl else None
    report = build_sag_db_from_orders(
        rows, db_path, extra_entity_links_by_doc=extra_links,
        semantic_events_by_doc=semantic_events, discourse_by_doc=discourse,
    )
    report["input_path"] = str(input_path)
    if entity_links_jsonl:
        report["entity_links_jsonl"] = str(entity_links_jsonl)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build pure SAG-lite DuckDB database from work-order data.")
    parser.add_argument("--input", required=True, help="Input t_order_master.tsv or multiview JSONL.")
    parser.add_argument("--db", required=True, help="Output DuckDB database path.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit.")
    parser.add_argument("--entity-links-jsonl", default="", help="Optional legacy or semantic entity-link JSONL.")
    parser.add_argument("--semantic-events-jsonl", default="", help="Optional projected semantic event JSONL.")
    parser.add_argument("--discourse-jsonl", default="", help="Optional projected event discourse JSONL.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = build_sag_db(
        args.input, args.db, args.limit, args.entity_links_jsonl,
        args.semantic_events_jsonl, args.discourse_jsonl,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
