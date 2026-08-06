"""Check a private multiview export and print only aggregate diagnostics."""

import argparse
import json
from pathlib import Path

from ragflow_style_pipeline.multiview_export_check import check_multiview_export


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Check private multiview JSONL schema, hashes and identity uniqueness."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--quality-report", default="")
    parser.add_argument("--output", default="")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = check_multiview_export(args.input, args.quality_report)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["errors_present"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
