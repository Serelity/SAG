#!/usr/bin/env bash
set -euo pipefail

fail() { printf 'ERROR:%s\n' "$1" >&2; exit 2; }
require_env() {
  local name="$1" expected="$2" current="${!name-}"
  [[ -z "$current" || "$current" == "$expected" ]] || fail "unsafe_environment:${name}"
  export "$name=$expected"
}

[[ -n "${CONDA_PREFIX:-}" ]] || fail 'activate_existing_conda_environment_first'
[[ -n "${RUN_DIR:-}" ]] || fail 'RUN_DIR_required'

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$ROOT/configs/entity_extraction_v1.json"
DATA_PATH="${DATA_PATH:-${INPUT_TSV:-}}"
MODEL_PATH="${MODEL_PATH:-${MODEL_DIR:-}}"
if [[ -z "$DATA_PATH" ]]; then
  [[ -f "$ROOT/data/t_order_master.tsv" ]] && DATA_PATH="$ROOT/data/t_order_master.tsv" || DATA_PATH="$ROOT/../data/t_order_master.tsv"
fi
if [[ -z "$MODEL_PATH" ]]; then
  [[ -d "$ROOT/models/Qwen3-4B" ]] && MODEL_PATH="$ROOT/models/Qwen3-4B" || MODEL_PATH="$ROOT/../models/Qwen3-4B"
fi

[[ -f "$DATA_PATH" ]] || fail 'input_file_missing'
[[ -d "$MODEL_PATH" ]] || fail 'model_directory_missing'
[[ -f "$CONFIG" ]] || fail 'config_missing'

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
require_env VLLM_USE_V1 0
require_env VLLM_ATTENTION_BACKEND XFORMERS
require_env VLLM_ENABLE_PREFIX_CACHING 0
require_env VLLM_ENABLE_CHUNKED_PREFILL 0
require_env VLLM_ENFORCE_EAGER 0
require_env VLLM_LOGGING_LEVEL WARNING

python -m ragflow_style_pipeline.cli preflight \
  --config "$CONFIG" --model "$MODEL_PATH" --input "$DATA_PATH"

if [[ "${RESUME:-0}" == "1" ]]; then
  [[ -d "$RUN_DIR" ]] || fail 'resume_run_directory_missing'
  [[ -z "${LIMIT:-}" ]] || fail 'LIMIT_not_allowed_with_resume'
  python -m ragflow_style_pipeline.cli extract \
    --config "$CONFIG" --model "$MODEL_PATH" --run-dir "$RUN_DIR" --resume
else
  [[ "${RESUME:-0}" == "0" ]] || fail 'RESUME_must_be_0_or_1'
  [[ ! -e "$RUN_DIR" ]] || fail 'fresh_RUN_DIR_already_exists'
  prepare=(python -m ragflow_style_pipeline.cli prepare \
    --input "$DATA_PATH" --run-dir "$RUN_DIR")
  [[ -z "${LIMIT:-}" ]] || prepare+=(--limit "$LIMIT")
  "${prepare[@]}"
  python -m ragflow_style_pipeline.cli extract \
    --config "$CONFIG" --model "$MODEL_PATH" --run-dir "$RUN_DIR"
fi

python -m ragflow_style_pipeline.cli project --run-dir "$RUN_DIR"
python -m ragflow_style_pipeline.cli check --run-dir "$RUN_DIR"
