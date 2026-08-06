"""Build deterministic, text-free production and challenge evaluation manifests."""

import argparse
import json
import os
from pathlib import Path

from ragflow_style_pipeline.sag_semantic_audit import build_eval_manifest


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


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
        description="Build a deterministic evaluation manifest without work-order text."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--production-size", type=int, default=200)
    parser.add_argument("--challenge-size", type=int, default=64)
    parser.add_argument("--seed", default="sag-eval-v1")
    parser.add_argument(
        "--semantic",
        default="",
        help="Optional semantic JSONL used only for repair/gap challenge strata.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    manifest, report = build_eval_manifest(
        args.input,
        production_size=max(0, args.production_size),
        challenge_size=max(0, args.challenge_size),
        seed=args.seed,
        semantic_path=args.semantic or None,
    )
    _atomic_jsonl(args.manifest, manifest)
    _atomic_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
