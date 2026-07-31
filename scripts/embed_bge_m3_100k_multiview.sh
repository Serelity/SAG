#!/usr/bin/env bash
set -euo pipefail

INPUT_JSONL="${INPUT_JSONL:-outputs/t_order_master.100k.multiview.jsonl}"
VECTOR_OUTPUT="${VECTOR_OUTPUT:-outputs/embeddings.100k.multiview.bge-m3.npy}"
META_OUTPUT="${META_OUTPUT:-outputs/embeddings.100k.multiview.bge-m3.meta.jsonl}"
BGE_M3_MODEL="${BGE_M3_MODEL:-.cache/models/BAAI/bge-m3}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_LENGTH="${MAX_LENGTH:-1024}"

mkdir -p outputs

PYTHONPATH=src PYTHONIOENCODING=utf-8 python -m ragflow_style_pipeline.bge_m3_embed \
  --input "${INPUT_JSONL}" \
  --vectors "${VECTOR_OUTPUT}" \
  --meta "${META_OUTPUT}" \
  --model "${BGE_M3_MODEL}" \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
  --max-length "${MAX_LENGTH}"
