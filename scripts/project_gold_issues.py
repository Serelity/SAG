"""Project private issue gold into flat or issue-aware SAG audit rows."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from ragflow_style_pipeline.sag_semantic_audit import (
    project_gold_issues,
    validate_gold_annotations,
)


def _atomic_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def _digest(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Project private issue gold for oracle SAG flat-vs-issue comparison."
    )
    parser.add_argument("--gold", required=True)
    parser.add_argument("--mode", choices=("flat", "issue-aware"), required=True)
    parser.add_argument("--order-events", required=True)
    parser.add_argument("--issue-events", required=True)
    parser.add_argument("--member-links", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    validation = validate_gold_annotations(args.gold, require_complete=True)
    if validation["errors_present"] or not validation["ready_for_evaluation"]:
        raise SystemExit(
            "Gold annotations are incomplete or invalid; run validate_semantic_gold.py."
        )
    orders, issues, links = project_gold_issues(args.gold, flat=args.mode == "flat")
    _atomic_jsonl(args.order_events, orders)
    _atomic_jsonl(args.issue_events, issues)
    _atomic_jsonl(args.member_links, links)
    print(json.dumps({
        "private": True,
        "mode": args.mode,
        "order_events": len(orders),
        "issue_events": len(issues),
        "member_links": len(links),
        "hashes": {
            "order_events": _digest(args.order_events),
            "issue_events": _digest(args.issue_events),
            "member_links": _digest(args.member_links),
        },
        "warning": "Private gold projection; do not share, package, or commit.",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
