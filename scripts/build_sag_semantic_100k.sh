#!/usr/bin/env bash
set -euo pipefail

INPUT_JSONL="${INPUT_JSONL:-data/t_order_master.100k.multiview.jsonl}"
EVENTS="${EVENTS:-outputs/sag_events.qwen3_4b.jsonl}"
LINKS="${LINKS:-outputs/sag_event_entity_links.qwen3_4b.jsonl}"
DISCOURSE="${DISCOURSE:-outputs/sag_event_discourse.qwen3_4b.jsonl}"
OUTPUT_DB="${OUTPUT_DB:-outputs/sag_semantic.qwen3_4b.100k.duckdb}"
LIMIT="${LIMIT:-100000}"

mkdir -p outputs
PYTHONPATH=src PYTHONIOENCODING=utf-8 python -m ragflow_style_pipeline.sag_db \
  --input "$INPUT_JSONL" --db "$OUTPUT_DB" --limit "$LIMIT" \
  --entity-links-jsonl "$LINKS" --semantic-events-jsonl "$EVENTS" \
  --discourse-jsonl "$DISCOURSE"
