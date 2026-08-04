#!/usr/bin/env bash
set -euo pipefail

INPUT_JSONL="${INPUT_JSONL:-data/t_order_master.100k.multiview.jsonl}"
SEMANTICS="${SEMANTICS:-outputs/work_order_semantics.qwen3_4b.jsonl}"
LINKS="${LINKS:-outputs/sag_event_entity_links.qwen3_4b.jsonl}"
DISCOURSE="${DISCOURSE:-outputs/sag_event_discourse.qwen3_4b.jsonl}"
EVENTS="${EVENTS:-outputs/sag_events.qwen3_4b.jsonl}"

[[ -f "$INPUT_JSONL" ]] || { echo "Missing orders: $INPUT_JSONL" >&2; exit 2; }
[[ -f "$SEMANTICS" ]] || { echo "Missing semantics: $SEMANTICS" >&2; exit 2; }
mkdir -p outputs
PYTHONPATH=src PYTHONIOENCODING=utf-8 python -m ragflow_style_pipeline.sag_semantic_projection \
  --input "$SEMANTICS" --orders "$INPUT_JSONL" --links "$LINKS" --discourse "$DISCOURSE" --events "$EVENTS"
