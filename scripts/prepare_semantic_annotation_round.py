"""Create two isolated private annotation files from a pristine packet."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from ragflow_style_pipeline.sag_semantic_audit import prepare_annotation_round


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Prepare deterministic A/B annotation copies without exposing source text."
    )
    parser.add_argument("--packet", required=True)
    parser.add_argument("--left-annotator", required=True)
    parser.add_argument("--right-annotator", required=True)
    parser.add_argument("--left-output", required=True)
    parser.add_argument("--right-output", required=True)
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
    packet = Path(args.packet).resolve()
    outputs = [Path(args.left_output).resolve(), Path(args.right_output).resolve()]
    report_path = Path(args.report).resolve()
    if len({packet, *outputs, report_path}) != 4:
        raise SystemExit("packet, left output, right output, and report paths must differ")
    left, right, report = prepare_annotation_round(
        args.packet, args.left_annotator, args.right_annotator
    )
    _atomic_jsonl(outputs[0], left)
    _atomic_jsonl(outputs[1], right)
    report["left_output_sha256"] = (
        "sha256:" + hashlib.sha256(outputs[0].read_bytes()).hexdigest()
    )
    report["right_output_sha256"] = (
        "sha256:" + hashlib.sha256(outputs[1].read_bytes()).hexdigest()
    )
    _atomic_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
