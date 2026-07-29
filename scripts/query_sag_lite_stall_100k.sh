#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/sag_query_stall.json}"
INPUT_DB="${INPUT_DB:-outputs/sag_lite.100k.duckdb}"
OUTPUT_JSON="${OUTPUT_JSON:-outputs/sag_lite.query.stall.100k.json}"

mkdir -p outputs

PYTHONPATH=src PYTHONIOENCODING=utf-8 python -m ragflow_style_pipeline.sag_query \
  --db "${INPUT_DB}" \
  --config "${CONFIG}" \
  --output "${OUTPUT_JSON}"
