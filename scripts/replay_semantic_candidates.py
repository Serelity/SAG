"""Replay the current validator against a private pre-sanitation candidate ledger."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from ragflow_style_pipeline.sag_semantic_audit import replay_candidate_ledger


def _atomic_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Replay deterministic semantic validation without loading a model."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rows, report = replay_candidate_ledger(args.input, args.candidates)
    _atomic_jsonl(args.output, rows)
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        **report,
        "output_sha256": "sha256:" + hashlib.sha256(Path(args.output).read_bytes()).hexdigest(),
        "warning": "Replay output contains desensitized evidence and must remain private.",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
