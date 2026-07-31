#!/usr/bin/env bash
set -euo pipefail

INPUT_JSONL="${INPUT_JSONL:-outputs/t_order_master.100k.multiview.jsonl}"
OUTPUT_DB="${OUTPUT_DB:-outputs/work_orders.100k.duckdb}"

mkdir -p outputs

PYTHONPATH=src PYTHONIOENCODING=utf-8 python -m ragflow_style_pipeline.analysis_db \
  --input "${INPUT_JSONL}" \
  --db "${OUTPUT_DB}"
