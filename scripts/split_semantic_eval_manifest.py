"""Deterministically split a private identity manifest without reading work-order text."""

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

ALLOWED_SUBSETS = ("production", "challenge")


def _digest(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _score(record, seed):
    value = "\u241f".join((seed, record["doc_id"], record["content_hash"]))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read(path):
    rows = []
    identities = set()
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line_{line_number}:invalid_record")
            identity = (str(value.get("doc_id", "")), str(value.get("content_hash", "")))
            if not all(identity):
                raise ValueError(f"line_{line_number}:missing_identity")
            if identity in identities:
                raise ValueError("duplicate_identity")
            subset = str(value.get("subset", ""))
            if subset not in ALLOWED_SUBSETS:
                raise ValueError(f"line_{line_number}:invalid_subset")
            identities.add(identity)
            rows.append(value)
    if not rows:
        raise ValueError("empty_manifest")
    return rows


def _allocation(counts, dev_size):
    total = sum(counts.values())
    if dev_size < 1 or dev_size >= total:
        raise ValueError("dev_size_must_leave_nonempty_holdout")
    exact = {key: dev_size * counts[key] / total for key in ALLOWED_SUBSETS}
    selected = {key: min(counts[key], int(exact[key])) for key in ALLOWED_SUBSETS}
    remaining = dev_size - sum(selected.values())
    order = sorted(
        ALLOWED_SUBSETS,
        key=lambda key: (-(exact[key] - int(exact[key])), key),
    )
    for key in order:
        if remaining and selected[key] < counts[key]:
            selected[key] += 1
            remaining -= 1
    if remaining:
        raise ValueError("cannot_allocate_dev_size")
    if any(selected[key] >= counts[key] for key in ALLOWED_SUBSETS if counts[key]):
        raise ValueError("dev_size_exhausts_subset")
    return selected


def _atomic_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as target:
            for row in rows:
                target.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as target:
            json.dump(value, target, ensure_ascii=False, indent=2, sort_keys=True)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def split_manifest(source_path, dev_path, holdout_path, report_path, dev_size=16, seed="sag-v8-split-v1"):
    paths = [Path(value).resolve() for value in (source_path, dev_path, holdout_path, report_path)]
    if len(set(paths)) != len(paths):
        raise ValueError("manifest_paths_must_be_distinct")
    rows = _read(source_path)
    counts = Counter(str(row["subset"]) for row in rows)
    allocation = _allocation(counts, int(dev_size))
    dev, holdout = [], []
    for subset in ALLOWED_SUBSETS:
        group = sorted(
            (row for row in rows if row["subset"] == subset),
            key=lambda row: (_score(row, seed), row["doc_id"], row["content_hash"]),
        )
        dev.extend(group[:allocation[subset]])
        holdout.extend(group[allocation[subset]:])
    dev.sort(key=lambda row: (ALLOWED_SUBSETS.index(row["subset"]), _score(row, seed)))
    holdout.sort(key=lambda row: (ALLOWED_SUBSETS.index(row["subset"]), _score(row, seed)))
    _atomic_jsonl(dev_path, dev)
    _atomic_jsonl(holdout_path, holdout)
    report = {
        "schema": "sag_semantic_eval_split_report_v1",
        "seed": seed,
        "source_records": len(rows),
        "source_subset_counts": {key: counts[key] for key in ALLOWED_SUBSETS},
        "development_records": len(dev),
        "development_subset_counts": dict(Counter(row["subset"] for row in dev)),
        "holdout_records": len(holdout),
        "holdout_subset_counts": dict(Counter(row["subset"] for row in holdout)),
        "identity_overlap": len(
            {(row["doc_id"], row["content_hash"]) for row in dev}
            & {(row["doc_id"], row["content_hash"]) for row in holdout}
        ),
        "hashes": {
            "source": _digest(source_path),
            "development": _digest(dev_path),
            "holdout": _digest(holdout_path),
        },
    }
    _atomic_json(report_path, report)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Split a private semantic manifest into development and holdout identities.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--development", required=True)
    parser.add_argument("--holdout", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--dev-size", type=int, default=16)
    parser.add_argument("--seed", default="sag-v8-split-v1")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    result = split_manifest(
        args.source, args.development, args.holdout, args.report,
        dev_size=args.dev_size, seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
