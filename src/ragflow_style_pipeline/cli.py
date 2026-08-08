"""Safe server-only CLI for prepare, extract, project and check."""

from __future__ import annotations

import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import re
import sys

from .check import check
from .constants import PIPELINE_VERSION
from .pipeline import extract, load_config, model_manifest
from .projection import project
from .vllm_backend import validate_environment
from .work_order import prepare


_SAFE_ERROR_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,160}$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="entity-extraction-v1")
    commands = parser.add_subparsers(dest="command", required=True)

    preflight = commands.add_parser("preflight")
    preflight.add_argument("--config", type=Path, required=True)
    preflight.add_argument("--model", type=Path, required=True)
    preflight.add_argument("--input", type=Path, required=True)

    preparation = commands.add_parser("prepare")
    preparation.add_argument("--input", type=Path, required=True)
    preparation.add_argument("--run-dir", type=Path, required=True)
    preparation.add_argument("--limit", type=int)

    extraction = commands.add_parser("extract")
    extraction.add_argument("--config", type=Path, required=True)
    extraction.add_argument("--model", type=Path, required=True)
    extraction.add_argument("--run-dir", type=Path, required=True)
    extraction.add_argument("--resume", action="store_true")

    projection = commands.add_parser("project")
    projection.add_argument("--run-dir", type=Path, required=True)

    checking = commands.add_parser("check")
    checking.add_argument("--run-dir", type=Path, required=True)
    return parser


def preflight(config_path: Path, model_path: Path, input_path: Path) -> dict:
    config = load_config(config_path)
    validate_environment()
    if not os.environ.get("CONDA_PREFIX"):
        raise RuntimeError("conda_environment_not_active")
    if not Path(input_path).is_file():
        raise RuntimeError("input_file_missing")
    try:
        vllm_version = metadata.version("vllm")
        transformers_version = metadata.version("transformers")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError("required_package_missing") from exc
    if vllm_version != "0.8.5" or transformers_version != "4.51.3":
        raise RuntimeError("package_version_mismatch")

    try:
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise RuntimeError("cuda_gpu_required")
        device_count = int(torch.cuda.device_count())
        capability = torch.cuda.get_device_capability(0)
        if capability != (7, 0):
            raise RuntimeError("v100_compute_capability_required")
    except ImportError as exc:
        raise RuntimeError("torch_missing") from exc

    manifest = model_manifest(model_path)
    return {
        "schema_version": "preflight_safe_v1",
        "pipeline_version": PIPELINE_VERSION,
        "status": "ok",
        "config_version": config["pipeline_version"],
        "vllm_version": vllm_version,
        "transformers_version": transformers_version,
        "cuda_device_count": device_count,
        "cuda_compute_capability": "7.0",
        "model_fingerprint": manifest["fingerprint"],
        "model_fingerprint_method": manifest["method"],
    }


def _safe_error_code(exc: Exception) -> str:
    value = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
    return value if _SAFE_ERROR_RE.fullmatch(value) else "unspecified_failure"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "preflight":
            result = preflight(args.config, args.model, args.input)
        elif args.command == "prepare":
            result = prepare(args.input, args.run_dir, args.limit)
        elif args.command == "extract":
            result = extract(
                args.run_dir,
                load_config(args.config),
                args.model,
                resume=args.resume,
            )
        elif args.command == "project":
            result = project(args.run_dir)
        else:
            result = check(args.run_dir)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        return 0
    except Exception as exc:
        print("ERROR:" + exc.__class__.__name__ + ":" + _safe_error_code(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
