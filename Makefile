# Convenience targets for the full benchmark. Requires `ollama` models pulled
# (see README) and, for KWDLC, a cloned corpus (see README).
TOK ?= .venv/bin/tok-eval
MODELS ?=

.PHONY: full ud kwdlc cross

full:      ## full pass on every dataset: tokenization + lemma + batching
	./scripts/run_full_benchmark.sh

ud:        ## full pass on UD Japanese (ja_gsd)
	MODELS="$(MODELS)" ./scripts/run_full_eval.sh config.yaml

kwdlc:     ## full pass on KWDLC
	MODELS="$(MODELS)" ./scripts/run_full_eval.sh config.kwdlc.yaml

cross:     ## cross-treebank figures over the merged scores
	$(TOK) --config config.yaml plot --all-treebanks
