"""Write aggregate-only, privacy-safe statistics for semantic input JSONL."""

import argparse
import json
from pathlib import Path

from ragflow_style_pipeline.sag_semantic_audit import profile_semantic_input


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Profile desensitized semantic input without printing record text or identifiers."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-input-chars", type=int, default=2200)
    parser.add_argument("--head-size", type=int, default=32)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = profile_semantic_input(
        args.input,
        max_input_chars=max(0, args.max_input_chars),
        head_size=max(0, args.head_size),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema": report["schema"],
        "records_read": report["records_read"],
        "valid_records": report["valid_records"],
        "invalid_records": report["invalid_records"],
        "output": str(output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
