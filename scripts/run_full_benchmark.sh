#!/usr/bin/env bash
# Full benchmark across every dataset, then cross-treebank figures.
#   scripts/run_full_benchmark.sh
# Honors the same MODELS / ANALYZERS / BATCHES / LIMIT env overrides.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

TOK="${TOK_EVAL:-.venv/bin/tok-eval}"

./scripts/run_full_eval.sh config.yaml          # UD Japanese (UniDic convention)
./scripts/run_full_eval.sh config.kwdlc.yaml    # KWDLC (JUMAN convention, web)

echo "==================== cross-treebank figures ===================="
"$TOK" --config config.yaml plot --all-treebanks
