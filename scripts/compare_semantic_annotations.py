"""Compare two complete private SAG issue annotation files."""

import argparse
import hashlib
import json
from pathlib import Path

from ragflow_style_pipeline.sag_semantic_audit import compare_gold_annotations


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Write aggregate agreement metrics and an optional private conflict packet."
    )
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--left-annotator", default="")
    parser.add_argument("--right-annotator", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--conflicts", default="")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report, conflicts = compare_gold_annotations(
        args.left,
        args.right,
        left_annotator=args.left_annotator,
        right_annotator=args.right_annotator,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = dict(report)
    if args.conflicts:
        conflicts_path = Path(args.conflicts)
        conflicts_path.parent.mkdir(parents=True, exist_ok=True)
        with conflicts_path.open("w", encoding="utf-8", newline="\n") as target:
            for row in conflicts:
                target.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary["private_conflicts_written"] = len(conflicts)
        summary["private_conflicts_sha256"] = (
            "sha256:" + hashlib.sha256(conflicts_path.read_bytes()).hexdigest()
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
