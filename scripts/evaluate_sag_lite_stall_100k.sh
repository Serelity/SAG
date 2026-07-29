#!/usr/bin/env bash
set -euo pipefail

INPUT_DB="${INPUT_DB:-outputs/sag_lite.100k.duckdb}"
QUERY_REPORT="${QUERY_REPORT:-outputs/sag_lite.query.stall.100k.json}"
MANUAL_SAMPLES="${MANUAL_SAMPLES:-outputs/sag_lite.eval_samples.stall.100k.jsonl}"
ENTITY_SAMPLES="${ENTITY_SAMPLES:-outputs/sag_lite.entity_eval_samples.100k.jsonl}"

mkdir -p outputs

PYTHONPATH=src PYTHONIOENCODING=utf-8 python -m ragflow_style_pipeline.sag_eval \
  --db "${INPUT_DB}" \
  --query-report "${QUERY_REPORT}" \
  --manual-samples "${MANUAL_SAMPLES}" \
  --entity-samples "${ENTITY_SAMPLES}"
