#!/usr/bin/env bash
# Full tokenization + lemma + batching pass for ONE dataset config.
#
#   scripts/run_full_eval.sh [config.yaml]
#
# Environment overrides:
#   MODELS     comma-separated model names   (default: full matrix incl. remote)
#   ANALYZERS  comma-separated baselines     (default: mecab,sudachi)
#   BATCHES    batch sizes to sweep          (default: 1 2 4 8)
#   LIMIT      sentence cap for a smoke run  (default: unset = all)
#   TOK_EVAL   path to the CLI               (default: .venv/bin/tok-eval)
#
# Everything is cached per (model, prompt, params, sentence), so re-running this
# script resumes rather than re-querying. Remote models are skipped when their
# API key is absent (see .env).
set -euo pipefail

CONFIG="${1:-config.yaml}"
TOK="${TOK_EVAL:-.venv/bin/tok-eval}"
MODELS="${MODELS:-llama3.1:8b,mistral-small3.1:24b,gemma3:27b,qwen2.5:32b,opencode-go/deepseek-v4.1-flash,sakana-namazu}"
ANALYZERS="${ANALYZERS:-mecab,sudachi}"
BATCHES="${BATCHES:-1 2 4 8}"

LIMIT_ARG=()
if [[ -n "${LIMIT:-}" ]]; then LIMIT_ARG=(--limit "$LIMIT"); fi

echo "==================== dataset: ${CONFIG} ===================="
"$TOK" --config "$CONFIG" prepare "${LIMIT_ARG[@]}"
"$TOK" --config "$CONFIG" baseline --analyzers "$ANALYZERS"

echo "---- reference: tokenization ----"
"$TOK" --config "$CONFIG" llm --models "$MODELS" "${LIMIT_ARG[@]}"
echo "---- reference: tokenization + lemma ----"
"$TOK" --config "$CONFIG" llm --models "$MODELS" --lemmas "${LIMIT_ARG[@]}"

for n in $BATCHES; do
  echo "---- batch N=${n}: tokenization ----"
  "$TOK" --config "$CONFIG" llm --models "$MODELS" --batch-size "$n" "${LIMIT_ARG[@]}"
  echo "---- batch N=${n}: tokenization + lemma ----"
  "$TOK" --config "$CONFIG" llm --models "$MODELS" --batch-size "$n" --lemmas "${LIMIT_ARG[@]}"
done

echo "---- scoring + comparison + figures ----"
"$TOK" --config "$CONFIG" score
"$TOK" --config "$CONFIG" compare
"$TOK" --config "$CONFIG" plot
