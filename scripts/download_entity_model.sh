#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B}"
MODEL_DIR="${MODEL_DIR:-models/Qwen3-4B}"

mkdir -p "$(dirname "${MODEL_DIR}")"

export MODEL_ID MODEL_DIR
python - <<'PY'
import os
from modelscope import snapshot_download

model_id = os.environ["MODEL_ID"]
model_dir = os.environ["MODEL_DIR"]
snapshot_download(model_id, local_dir=model_dir)
print({"model_id": model_id, "model_dir": model_dir, "backend": "modelscope"})
PY
