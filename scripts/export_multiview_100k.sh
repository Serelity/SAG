#!/usr/bin/env bash
set -euo pipefail

INPUT_TSV="${INPUT_TSV:-data/t_order_master.tsv}"
OUTPUT_JSONL="${OUTPUT_JSONL:-outputs/t_order_master.100k.multiview.jsonl}"
QUALITY_REPORT="${QUALITY_REPORT:-outputs/t_order_master.100k.multiview.quality.json}"
LIMIT="${LIMIT:-100000}"

mkdir -p outputs

PYTHONPATH=src PYTHONIOENCODING=utf-8 python -m ragflow_style_pipeline.export_jsonl \
  --input "${INPUT_TSV}" \
  --output "${OUTPUT_JSONL}" \
  --quality-report "${QUALITY_REPORT}" \
  --limit "${LIMIT}"
