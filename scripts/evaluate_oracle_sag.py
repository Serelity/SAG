"""Compare local flat and issue-aware Oracle SAG retrieval."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from ragflow_style_pipeline.sag_oracle import evaluate_oracle_retrieval


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate flat versus issue-aware Oracle SAG with private relevance gold."
    )
    parser.add_argument("--flat-db", required=True)
    parser.add_argument("--issue-db", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--traces", default="")
    parser.add_argument("--cutoffs", default="5,10")
    return parser.parse_args(argv)


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def main(argv=None):
    args = parse_args(argv)
    try:
        cutoffs = tuple(int(value.strip()) for value in args.cutoffs.split(",") if value.strip())
    except ValueError as error:
        raise SystemExit("--cutoffs must be comma-separated positive integers") from error
    report, traces = evaluate_oracle_retrieval(
        args.flat_db, args.issue_db, args.queries, cutoffs=cutoffs
    )
    _atomic_json(args.output, report)
    summary = dict(report)
    if args.traces:
        _atomic_jsonl(args.traces, traces)
        summary["private_traces_written"] = len(traces)
        summary["private_traces_sha256"] = (
            "sha256:" + hashlib.sha256(Path(args.traces).read_bytes()).hexdigest()
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
