"""Merge double annotations and explicit conflict resolutions into private gold."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from ragflow_style_pipeline.sag_semantic_audit import (
    merge_adjudicated_gold,
    validate_gold_annotations,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Create final adjudicated gold from two annotations and resolved conflicts."
    )
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--conflicts", required=True)
    parser.add_argument("--adjudicator", required=True)
    parser.add_argument("--left-annotator", default="")
    parser.add_argument("--right-annotator", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    return parser.parse_args(argv)


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


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main(argv=None):
    args = parse_args(argv)
    rows, report = merge_adjudicated_gold(
        args.left,
        args.right,
        args.conflicts,
        adjudicator=args.adjudicator,
        left_annotator=args.left_annotator,
        right_annotator=args.right_annotator,
    )
    _atomic_jsonl(args.output, rows)
    validation = validate_gold_annotations(
        args.output, require_complete=True, expected_annotator=args.adjudicator
    )
    if not validation["ready_for_evaluation"]:
        Path(args.output).unlink(missing_ok=True)
        raise SystemExit("Merged adjudicated gold failed final validation.")
    report["output_sha256"] = (
        "sha256:" + hashlib.sha256(Path(args.output).read_bytes()).hexdigest()
    )
    report["validation"] = validation
    _atomic_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
