"""Build a private annotation packet without printing desensitized work-order text."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from ragflow_style_pipeline.sag_semantic_audit import build_private_annotation_packet


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
        description="Join a text-free manifest to private desensitized annotation records."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rows = build_private_annotation_packet(args.input, args.manifest)
    _atomic_jsonl(args.output, rows)
    digest = "sha256:" + hashlib.sha256(Path(args.output).read_bytes()).hexdigest()
    print(json.dumps({
        "private": True,
        "records": len(rows),
        "output": str(args.output),
        "sha256": digest,
        "warning": "Contains desensitized source text; do not share, package, or commit.",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
