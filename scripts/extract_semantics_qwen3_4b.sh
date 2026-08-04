#!/usr/bin/env bash
set -euo pipefail

INPUT_JSONL="${INPUT_JSONL:-data/t_order_master.100k.multiview.jsonl}"
CONFIG="${CONFIG:-configs/sag_semantic_extraction_qwen3_4b.json}"
MODEL_PATH="${MODEL_PATH:-models/Qwen3-4B}"
OUTPUT="${OUTPUT:-outputs/work_order_semantics.qwen3_4b.jsonl}"
REJECTS="${REJECTS:-outputs/work_order_semantics.rejects.jsonl}"
RUN_REPORT="${RUN_REPORT:-outputs/work_order_semantics.run.json}"
QUALITY_REPORT="${QUALITY_REPORT:-outputs/work_order_semantics.quality.json}"
LIMIT="${LIMIT:-100000}"

[[ "$INPUT_JSONL" == *.jsonl ]] || { echo "INPUT_JSONL must be desensitized multiview JSONL" >&2; exit 2; }
[[ -f "$INPUT_JSONL" ]] || { echo "Missing input: $INPUT_JSONL" >&2; exit 2; }
[[ -f "$CONFIG" ]] || { echo "Missing config: $CONFIG" >&2; exit 2; }
[[ -d "$MODEL_PATH" ]] || { echo "Missing server-local model: $MODEL_PATH" >&2; exit 2; }
mkdir -p outputs

df -h "$(dirname "$OUTPUT")"
printf 'input=%s\nconfig=%s\nmodel=%s\nlimit=%s\n' "$INPUT_JSONL" "$CONFIG" "$MODEL_PATH" "$LIMIT"

args=(
  --input "$INPUT_JSONL" --config "$CONFIG" --model-path "$MODEL_PATH"
  --output "$OUTPUT" --rejects "$REJECTS" --run-report "$RUN_REPORT"
  --quality-report "$QUALITY_REPORT" --limit "$LIMIT"
)
[[ "${RESUME:-0}" == "1" ]] && args+=(--resume)
[[ "${RETRY_REJECTED:-0}" == "1" ]] && args+=(--retry-rejected)
[[ -n "${DOC_ID_FILE:-}" ]] && args+=(--doc-id-file "$DOC_ID_FILE")

PYTHONPATH=src PYTHONIOENCODING=utf-8 python -m ragflow_style_pipeline.sag_semantic_llm "${args[@]}"
