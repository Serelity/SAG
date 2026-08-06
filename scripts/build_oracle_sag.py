"""Build a local private flat or issue-aware Oracle SAG DuckDB."""

import argparse
import json
import os
from pathlib import Path

from ragflow_style_pipeline.sag_oracle import build_oracle_sag_db


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Build a private Oracle SAG graph from completed issue gold."
    )
    parser.add_argument("--gold", required=True)
    parser.add_argument("--mode", choices=("flat", "issue-aware"), required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--report", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = db_path.with_name(db_path.name + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        report = build_oracle_sag_db(args.gold, temporary, args.mode)
        os.replace(temporary, db_path)
    finally:
        temporary.unlink(missing_ok=True)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
