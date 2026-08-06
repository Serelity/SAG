"""Validate private SAG issue annotations without printing identifiers or text."""

import argparse
import json
from pathlib import Path

from ragflow_style_pipeline.sag_semantic_audit import validate_gold_annotations


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate private issue gold and write an aggregate-only safe report."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--annotator", default="")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = validate_gold_annotations(
        args.input,
        require_complete=args.require_complete,
        expected_annotator=args.annotator,
    )
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["errors_present"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
