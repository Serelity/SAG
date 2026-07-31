#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/topic_analysis_stall.json}"
VECTOR_INPUT="${VECTOR_INPUT:-outputs/embeddings.100k.multiview.bge-m3.npy}"
META_INPUT="${META_INPUT:-outputs/embeddings.100k.multiview.bge-m3.meta.jsonl}"
OUTPUT_JSON="${OUTPUT_JSON:-outputs/topic_analysis.stall.100k.json}"
BGE_M3_MODEL="${BGE_M3_MODEL:-.cache/models/BAAI/bge-m3}"
DEVICE="${DEVICE:-cuda}"

mkdir -p outputs

PYTHONPATH=src PYTHONIOENCODING=utf-8 python -m ragflow_style_pipeline.topic_analysis \
  --config "${CONFIG}" \
  --vectors "${VECTOR_INPUT}" \
  --meta "${META_INPUT}" \
  --model "${BGE_M3_MODEL}" \
  --device "${DEVICE}" \
  --output "${OUTPUT_JSON}"
