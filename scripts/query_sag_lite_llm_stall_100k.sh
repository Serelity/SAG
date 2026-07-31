#!/usr/bin/env bash
set -euo pipefail

INPUT_DB="${INPUT_DB:-outputs/sag_lite.llm.100k.duckdb}"
CONFIG="${CONFIG:-configs/sag_query_stall.json}"
OUTPUT_REPORT="${OUTPUT_REPORT:-outputs/sag_lite.query.stall.llm.100k.json}"

PYTHONPATH=src PYTHONIOENCODING=utf-8 python -m ragflow_style_pipeline.sag_query \
  --db "${INPUT_DB}" \
  --config "${CONFIG}" \
  --output "${OUTPUT_REPORT}"
