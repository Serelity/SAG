"""Build a minimal private Qwen input packet from a frozen identity manifest."""

import argparse
import json
from pathlib import Path

from ragflow_style_pipeline.semantic_inference_packet import build_inference_packet


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Build an exact private inference packet without duplicate display text."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    input_path = Path(args.input).resolve()
    manifest_path = Path(args.manifest).resolve()
    output = Path(args.output).resolve()
    report_path = Path(args.report).resolve()
    if len({input_path, manifest_path, output, report_path}) != 4:
        raise SystemExit("input, manifest, output, and report paths must differ")
    report = build_inference_packet(args.input, args.manifest, output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
