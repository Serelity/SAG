"""Evaluate grounded mentions and issue-level SAG hyperedge membership."""

import argparse
import json
from pathlib import Path

from ragflow_style_pipeline.sag_semantic_audit import (
    evaluate_semantic_gold,
    validate_gold_annotations,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate semantic predictions against private issue-level gold."
    )
    parser.add_argument("--gold", required=True)
    parser.add_argument("--predictions", default="")
    parser.add_argument("--oracle-flat", action="store_true")
    parser.add_argument("--output", default="")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.oracle_flat and not args.predictions:
        raise SystemExit("--predictions is required unless --oracle-flat is set")
    validation = validate_gold_annotations(args.gold, require_complete=True)
    if validation["errors_present"] or not validation["ready_for_evaluation"]:
        raise SystemExit(
            "Gold annotations are incomplete or invalid; run validate_semantic_gold.py."
        )
    report = evaluate_semantic_gold(
        args.gold,
        prediction_path=args.predictions or None,
        oracle_flat=args.oracle_flat,
    )
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
