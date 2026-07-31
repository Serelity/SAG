"""Build a DuckDB analysis database from multi-view work-order JSONL."""

import argparse
import json
from pathlib import Path


WORK_ORDER_COLUMNS = [
    "doc_id",
    "case_content_clean",
    "case_goal_clean",
    "embedding_text",
    "display_text",
    "call_time",
    "call_month",
    "area_code_city",
    "area_code_area",
    "area_code_street",
    "type1",
    "type2",
    "type3",
    "order_source",
    "order_type",
    "order_status",
    "service_object_type",
]


def document_row(document):
    """Flatten one multi-view document into a DuckDB row dictionary."""
    metadata = document.get("metadata", {})
    display_text = str(document.get("display_text") or document.get("text") or "")
    return {
        "doc_id": document.get("doc_id", ""),
        "case_content_clean": document.get("case_content_clean", ""),
        "case_goal_clean": document.get("case_goal_clean", ""),
        "embedding_text": document.get("embedding_text", ""),
        "display_text": display_text,
        "call_time": metadata.get("call_time", ""),
        "call_month": metadata.get("call_month", ""),
        "area_code_city": metadata.get("area_code_city", ""),
        "area_code_area": metadata.get("area_code_area", ""),
        "area_code_street": metadata.get("area_code_street", ""),
        "type1": metadata.get("type1", ""),
        "type2": metadata.get("type2", ""),
        "type3": metadata.get("type3", ""),
        "order_source": metadata.get("order_source", ""),
        "order_type": metadata.get("order_type", ""),
        "order_status": metadata.get("order_status", ""),
        "service_object_type": metadata.get("service_object_type", ""),
    }


def read_rows(jsonl_path):
    """Read multi-view JSONL documents into flattened row dictionaries."""
    rows = []
    with Path(jsonl_path).open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if line.strip():
                rows.append(document_row(json.loads(line)))
    return rows


def build_analysis_db(jsonl_path, db_path):
    """Create or replace the work_orders table from a JSONL document file."""
    import duckdb

    rows = read_rows(jsonl_path)
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with duckdb.connect(str(db_path)) as conn:
        conn.execute("drop table if exists work_orders")
        column_sql = ", ".join(f"{column} varchar" for column in WORK_ORDER_COLUMNS)
        conn.execute(f"create table work_orders ({column_sql})")
        if rows:
            placeholders = ", ".join(["?"] * len(WORK_ORDER_COLUMNS))
            conn.executemany(
                f"insert into work_orders values ({placeholders})",
                [[row[column] for column in WORK_ORDER_COLUMNS] for row in rows],
            )
        conn.execute("create index if not exists idx_work_orders_doc_id on work_orders(doc_id)")
        conn.execute("create index if not exists idx_work_orders_call_month on work_orders(call_month)")
        conn.execute("create index if not exists idx_work_orders_area on work_orders(area_code_area)")
        conn.execute("create index if not exists idx_work_orders_street on work_orders(area_code_street)")
        conn.execute("create index if not exists idx_work_orders_type3 on work_orders(type3)")

    return {"jsonl": str(jsonl_path), "db": str(db_path), "documents_loaded": len(rows)}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build DuckDB analysis database from work-order JSONL.")
    parser.add_argument("--input", required=True, help="Input multi-view document JSONL.")
    parser.add_argument("--db", required=True, help="Output DuckDB database path.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = build_analysis_db(args.input, args.db)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
