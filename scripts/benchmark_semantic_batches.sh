#!/usr/bin/env bash
set -euo pipefail

INPUT_JSONL="${INPUT_JSONL:-outputs/input/t_order_master.smoke50.multiview.jsonl}"
CONFIG="${CONFIG:-configs/sag_semantic_extraction_qwen3_4b.json}"
MODEL_PATH="${MODEL_PATH:-models/Qwen3-4B}"
LIMIT="${LIMIT:-32}"
BATCH_SIZES="${BATCH_SIZES:-4 8 16}"
BACKEND="${BACKEND:-transformers}"
BENCH_ROOT="${BENCH_ROOT:-outputs/semantic-benchmark-${BACKEND}-$(date +%Y%m%d-%H%M%S)}"

[[ -f "$INPUT_JSONL" ]] || { echo "Missing input: $INPUT_JSONL" >&2; exit 2; }
[[ -f "$CONFIG" ]] || { echo "Missing config: $CONFIG" >&2; exit 2; }
[[ -d "$MODEL_PATH" ]] || { echo "Missing model: $MODEL_PATH" >&2; exit 2; }
mkdir -p "$BENCH_ROOT"

printf 'benchmark_root=%s\nlimit=%s\nbatch_sizes=%s\nbackend=%s\n' \
  "$BENCH_ROOT" "$LIMIT" "$BATCH_SIZES" "$BACKEND"

for batch_size in $BATCH_SIZES; do
  run_dir="$BENCH_ROOT/batch-$batch_size"
  mkdir -p "$run_dir"
  echo "===== batch_size=$batch_size ====="
  set +e
  SEMANTIC_LLM_DTYPE="${SEMANTIC_LLM_DTYPE:-float16}" \
  INPUT_JSONL="$INPUT_JSONL" CONFIG="$CONFIG" MODEL_PATH="$MODEL_PATH" \
  BATCH_SIZE="$batch_size" BACKEND="$BACKEND" LIMIT="$LIMIT" \
  OUTPUT="$run_dir/semantic.jsonl" REJECTS="$run_dir/rejects.jsonl" \
  RUN_REPORT="$run_dir/run.json" QUALITY_REPORT="$run_dir/quality.json" \
  DIAGNOSTIC_LOG="$run_dir/diagnostics.jsonl" \
  bash scripts/extract_semantics_qwen3_4b.sh >"$run_dir/console.log" 2>&1
  status=$?
  set -e
  echo "status=$status"
  if [[ "$status" -ne 0 ]]; then
    tail -n 40 "$run_dir/console.log"
    continue
  fi

  python scripts/check_semantic_run.py \
    --semantic "$run_dir/semantic.jsonl" --rejects "$run_dir/rejects.jsonl" \
    --run-report "$run_dir/run.json" --quality-report "$run_dir/quality.json" \
    >"$run_dir/checker.json"
  python scripts/summarize_semantic_diagnostics.py \
    --input "$run_dir/diagnostics.jsonl" >"$run_dir/diagnostics.summary.json"
  python - "$run_dir/run.json" <<'PY'
import json
import sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
keys = (
    "prompt_version", "backend", "batch_size", "repair_batch_size", "records_written", "rejects_written",
    "primary_batches", "repair_requests", "repair_batches",
    "truncation_count", "elapsed_seconds", "orders_per_second",
    "output_tokens_per_second", "gpu_peak_allocated_gb", "gpu_peak_reserved_gb",
    "attn_implementation", "cache_implementation", "prefix_caching",
    "chunked_prefill", "enforce_eager",
)
print(json.dumps({key: report.get(key) for key in keys}, ensure_ascii=False))
PY
done

echo "benchmark_root=$BENCH_ROOT"
