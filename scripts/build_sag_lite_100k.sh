#!/usr/bin/env bash
set -euo pipefail

INPUT_TSV="${INPUT_TSV:-data/t_order_master.tsv}"
OUTPUT_DB="${OUTPUT_DB:-outputs/sag_lite.100k.duckdb}"
LIMIT="${LIMIT:-100000}"

mkdir -p outputs

PYTHONPATH=src PYTHONIOENCODING=utf-8 python -m ragflow_style_pipeline.sag_db \
  --input "${INPUT_TSV}" \
  --db "${OUTPUT_DB}" \
  --limit "${LIMIT}"
