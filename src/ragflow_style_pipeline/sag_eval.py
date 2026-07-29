"""Evaluation helpers for pure SAG-lite retrieval."""

import argparse
import json
from pathlib import Path


WEAK_GOLD_TYPE3 = {"无照经营游商", "店外经营", "无证照餐饮店"}
WEAK_GOLD_KEYWORDS = ["流动摊贩", "游商摊贩", "摆摊", "设摊", "占道经营", "无照经营", "店外经营"]


def _connect(db_path):
    import duckdb

    return duckdb.connect(str(db_path), read_only=False)


def precision_at_k(results, gold_doc_ids, k):
    """Return fraction of top-k results whose doc_id is in gold_doc_ids."""
    if k <= 0 or not results:
        return 0.0
    top_results = results[:k]
    if not top_results:
        return 0.0
    hits = sum(1 for result in top_results if result.get("doc_id") in gold_doc_ids)
    return hits / len(top_results)


def recall_at_k(results, gold_doc_ids, k):
    """Return fraction of gold_doc_ids covered by top-k results."""
    if k <= 0 or not gold_doc_ids:
        return 0.0
    top_doc_ids = {result.get("doc_id") for result in results[:k]}
    hits = len(top_doc_ids & set(gold_doc_ids))
    return hits / len(gold_doc_ids)


def build_weak_gold_doc_ids(db_path, config=None):
    """Build weak gold doc ids for stall-like work orders."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            select doc_id, title_clean, case_content_clean, case_goal_clean, type3
            from source_orders
            """
        ).fetchall()

    gold = set()
    for doc_id, title, case_content, case_goal, type3 in rows:
        text = f"{title or ''}\n{case_content or ''}\n{case_goal or ''}"
        if type3 in WEAK_GOLD_TYPE3 or any(keyword in text for keyword in WEAK_GOLD_KEYWORDS):
            gold.add(doc_id)
    return gold


def evaluate_sag_results(db_path, config, results):
    """Return weak-label precision/recall metrics."""
    gold = build_weak_gold_doc_ids(db_path, config)
    return {
        "weak_gold_count": len(gold),
        "weak_precision@10": precision_at_k(results, gold, 10),
        "weak_precision@50": precision_at_k(results, gold, 50),
        "weak_precision@100": precision_at_k(results, gold, 100),
        "weak_recall@100": recall_at_k(results, gold, 100),
        "weak_recall@500": recall_at_k(results, gold, 500),
        "weak_recall@1000": recall_at_k(results, gold, 1000),
    }


def _rows_by_doc(conn, doc_ids):
    if not doc_ids:
        return {}
    placeholders = ", ".join(["?"] * len(doc_ids))
    rows = conn.execute(f"select * from source_orders where doc_id in ({placeholders})", doc_ids).fetchall()
    columns = [desc[0] for desc in conn.description]
    return {row[columns.index("doc_id")]: dict(zip(columns, row)) for row in rows}


def build_manual_eval_samples(db_path, results, limit=100):
    """Return JSONL-ready manual evaluation samples."""
    selected = results[:limit]
    doc_ids = [result["doc_id"] for result in selected]
    with _connect(db_path) as conn:
        rows_by_doc = _rows_by_doc(conn, doc_ids)

    samples = []
    for result in selected:
        row = rows_by_doc.get(result["doc_id"], {})
        samples.append(
            {
                "doc_id": result.get("doc_id", ""),
                "rank": result.get("rank", ""),
                "match_stage": result.get("match_stage", ""),
                "score": result.get("score", 0.0),
                "case_content": row.get("case_content_clean", ""),
                "case_goal": row.get("case_goal_clean", ""),
                "metadata_area": row.get("area_code_area", ""),
                "metadata_street": row.get("area_code_street", ""),
                "matched_entities": result.get("matched_entities", {}),
                "explanation": result.get("explanation", {}),
                "label": "",
                "label_reason": "",
            }
        )
    return samples


def build_entity_eval_samples(db_path, limit=200):
    """Return entity extraction samples for manual review."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            select l.doc_id, l.entity_type, l.entity_value, l.source_field, l.source_channel,
                   l.confidence, l.matched_text, s.case_content_clean, s.address_detail_clean
            from sag_event_entity_links l
            left join source_orders s on s.doc_id = l.doc_id
            order by l.entity_type, l.doc_id
            limit ?
            """,
            [int(limit)],
        ).fetchall()

    return [
        {
            "doc_id": row[0],
            "entity_type": row[1],
            "entity_value": row[2],
            "source_field": row[3],
            "source_channel": row[4],
            "confidence": float(row[5] or 0.0),
            "matched_text": row[6],
            "case_content": row[7],
            "address_detail": row[8],
            "label": "",
            "label_reason": "",
        }
        for row in rows
    ]


def _write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Export pure SAG-lite evaluation samples.")
    parser.add_argument("--db", required=True, help="SAG-lite DuckDB database path.")
    parser.add_argument("--query-report", required=True, help="SAG query report JSON.")
    parser.add_argument("--manual-samples", required=True, help="Output manual evaluation JSONL.")
    parser.add_argument("--entity-samples", required=True, help="Output entity evaluation JSONL.")
    parser.add_argument("--manual-limit", type=int, default=100, help="Manual sample limit.")
    parser.add_argument("--entity-limit", type=int, default=200, help="Entity sample limit.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = json.loads(Path(args.query_report).read_text(encoding="utf-8"))
    results = report.get("results") or report.get("retrieved_results") or []
    if not results:
        results = [
            {
                "doc_id": case.get("doc_id", ""),
                "rank": case.get("rank", index + 1),
                "match_stage": case.get("match_stage", ""),
                "score": case.get("score", 0.0),
                "matched_entities": case.get("matched_entities", {}),
                "explanation": case.get("explanation", {}),
            }
            for index, case in enumerate(report.get("representative_cases", []))
        ]

    manual_samples = build_manual_eval_samples(args.db, results, args.manual_limit)
    entity_samples = build_entity_eval_samples(args.db, args.entity_limit)
    _write_jsonl(args.manual_samples, manual_samples)
    _write_jsonl(args.entity_samples, entity_samples)
    print(
        json.dumps(
            {
                "manual_samples": str(args.manual_samples),
                "manual_samples_written": len(manual_samples),
                "entity_samples": str(args.entity_samples),
                "entity_samples_written": len(entity_samples),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
