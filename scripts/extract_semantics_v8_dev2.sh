#!/usr/bin/env bash
set -euo pipefail

[[ "${RESUME:-0}" != "1" ]] || {
  echo "v8_dev2 comparisons require a fresh RUN_DIR; RESUME=1 is forbidden" >&2
  exit 2
}
[[ -n "${RUN_DIR:-}" ]] || {
  echo "Set RUN_DIR to a new output directory, e.g. outputs/v8-dev2-smoke-001" >&2
  exit 2
}
[[ ! -e "$RUN_DIR" ]] || {
  echo "RUN_DIR already exists; use a fresh directory: $RUN_DIR" >&2
  exit 2
}
[[ -n "${INPUT_JSONL:-}" && -f "$INPUT_JSONL" ]] || {
  echo "Set INPUT_JSONL to a desensitized inference packet" >&2
  exit 2
}
[[ -n "${IDENTITY_MANIFEST:-}" && -f "$IDENTITY_MANIFEST" ]] || {
  echo "Set IDENTITY_MANIFEST to the same frozen development manifest used by v8_dev1" >&2
  exit 2
}
[[ -n "${MODEL_PATH:-}" && -d "$MODEL_PATH" ]] || {
  echo "Set MODEL_PATH to the server-local Qwen3-4B directory" >&2
  exit 2
}

export CONFIG="${CONFIG:-configs/sag_semantic_extraction_qwen3_4b_v8_dev2.json}"
[[ -f "$CONFIG" ]] || { echo "Missing config: $CONFIG" >&2; exit 2; }
mkdir -p "$RUN_DIR"
export OUTPUT="${OUTPUT:-$RUN_DIR/semantic.private.jsonl}"
export REJECTS="${REJECTS:-$RUN_DIR/rejects.private.jsonl}"
export RUN_REPORT="${RUN_REPORT:-$RUN_DIR/run.safe.json}"
export QUALITY_REPORT="${QUALITY_REPORT:-$RUN_DIR/quality.safe.json}"
export DIAGNOSTIC_LOG="${DIAGNOSTIC_LOG:-$RUN_DIR/diagnostics.safe.jsonl}"
export CANDIDATE_LEDGER="${CANDIDATE_LEDGER:-$RUN_DIR/candidates.private.jsonl}"
export DECISION_LEDGER="${DECISION_LEDGER:-$RUN_DIR/decisions.private.jsonl}"

if [[ "${BACKEND:-}" == "vllm" ]]; then
  export VLLM_USE_V1=0
  export VLLM_ATTENTION_BACKEND=XFORMERS
  export VLLM_ENABLE_PREFIX_CACHING=0
  export VLLM_ENABLE_CHUNKED_PREFILL=0
  export VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"
  export SEMANTIC_LLM_DTYPE=float16
fi

bash scripts/extract_semantics_qwen3_4b.sh

PYTHONPATH=src PYTHONIOENCODING=utf-8 python scripts/summarize_semantic_diagnostics.py \
  --input "$DIAGNOSTIC_LOG" > "$RUN_DIR/diagnostics-summary.safe.json"

PYTHONPATH=src PYTHONIOENCODING=utf-8 python scripts/check_semantic_run.py \
  --semantic "$OUTPUT" --rejects "$REJECTS" \
  --run-report "$RUN_REPORT" --quality-report "$QUALITY_REPORT" \
  --candidate-ledger "$CANDIDATE_LEDGER" --decision-ledger "$DECISION_LEDGER" \
  --diagnostics "$DIAGNOSTIC_LOG" > "$RUN_DIR/check.safe.json"

printf 'v8_dev2 run complete: %s\n' "$RUN_DIR"
