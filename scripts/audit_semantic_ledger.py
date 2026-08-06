"""Audit private semantic candidate/decision ledgers against adjudicated issue gold."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from ragflow_style_pipeline.sag_semantic_audit import audit_candidate_ledger_against_gold


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Replay the current validator and audit pre/post mentions against private gold."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--decisions", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--traces", default="")
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
    report, traces = audit_candidate_ledger_against_gold(
        args.input,
        args.gold,
        args.candidates,
        decision_ledger_path=args.decisions or None,
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
