#!/usr/bin/env bash
set -euo pipefail

INPUT_TSV="${INPUT_TSV:-data/t_order_master.tsv}"
ENTITY_LINKS_JSONL="${ENTITY_LINKS_JSONL:-outputs/sag_lite.entity_links.llm.100k.jsonl}"
OUTPUT_DB="${OUTPUT_DB:-outputs/sag_lite.llm.100k.duckdb}"
LIMIT="${LIMIT:-100000}"

mkdir -p outputs

PYTHONPATH=src PYTHONIOENCODING=utf-8 python -m ragflow_style_pipeline.sag_db \
  --input "${INPUT_TSV}" \
  --db "${OUTPUT_DB}" \
  --limit "${LIMIT}" \
  --entity-links-jsonl "${ENTITY_LINKS_JSONL}"
