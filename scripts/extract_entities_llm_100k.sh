#!/usr/bin/env bash
set -euo pipefail

INPUT_TSV="${INPUT_TSV:-data/t_order_master.tsv}"
CONFIG="${CONFIG:-configs/sag_entity_extraction_qwen3_4b.json}"
MODEL_PATH="${MODEL_PATH:-models/Qwen3-4B}"
OUTPUT_LINKS="${OUTPUT_LINKS:-outputs/sag_lite.entity_links.llm.100k.jsonl}"
OUTPUT_REJECTS="${OUTPUT_REJECTS:-outputs/sag_lite.entity_links.llm.rejects.100k.jsonl}"
LIMIT="${LIMIT:-100000}"

mkdir -p outputs

PYTHONPATH=src PYTHONIOENCODING=utf-8 python -m ragflow_style_pipeline.sag_entity_llm \
  --input "${INPUT_TSV}" \
  --output "${OUTPUT_LINKS}" \
  --rejects "${OUTPUT_REJECTS}" \
  --config "${CONFIG}" \
  --model-path "${MODEL_PATH}" \
  --limit "${LIMIT}"
