#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"

python -m unittest discover -s tests -p 'test_*.py' -v
python -m compileall -q src tests
bash -n scripts/run_v1.sh scripts/verify_v1.sh

# Production source must not regain removed compatibility backends or private fixtures.
if grep -RInE \
  --include='*.py' \
  'from transformers|import transformers|import duckdb|import sqlite3|sag_semantic|annotation_workbench|oracle_sag' \
  src tests; then
  printf 'ERROR:forbidden_legacy_runtime_detected\n' >&2
  exit 2
fi
if find tests -type f \( -name '*.tsv' -o -name '*.jsonl' -o -name '*.duckdb' \) -print -quit | grep -q .; then
  printf 'ERROR:private_or_fixture_data_committed\n' >&2
  exit 2
fi

git diff --check
printf 'server_static_and_synthetic_verification_ok\n'
