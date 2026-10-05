"""MeCab baseline (literal MeCab via fugashi) and optional Sudachi."""

from __future__ import annotations

from typing import Any

from .config import Config
from .io import read_jsonl, write_jsonl
from .text import align_spans_paired, norm_lemma

_TAGGER = None
_SUDACHI = None


def _tagger():
    global _TAGGER
    if _TAGGER is None:
        import fugashi  # imported lazily so --help works without it

        _TAGGER = fugashi.Tagger()
    return _TAGGER


def mecab_version() -> str:
    from importlib.metadata import version

    tagger = _tagger()
    info = ""
    try:
        dic = tagger.dictionary_info
        if dic:
            info = getattr(dic[0], "filename", "")
    except Exception:
        pass
    return f"fugashi={version('fugashi')} dic={info}"


def segment_mecab(text: str) -> tuple[list[tuple[int, int]], list[str]]:
    tagger = _tagger()
    toks = [
        (w.surface, norm_lemma(getattr(w.feature, "lemma", None))) for w in tagger(text)
    ]
    paired = align_spans_paired(text, [t[0] for t in toks])
    spans = [p[0] for p in paired]
    lemmas = [toks[i][1] for _, i in paired]
    return spans, lemmas


def segment_sudachi(text: str) -> tuple[list[tuple[int, int]], list[str]] | None:
    global _SUDACHI
    if _SUDACHI is None:
        try:
            from sudachipy import dictionary, tokenizer
        except ImportError:
            return None
        tok = dictionary.Dictionary(dict_type="core").create()
        _SUDACHI = (tok, tokenizer.Tokenizer.SplitMode.C)
    tok, mode = _SUDACHI
    toks = [
        (m.surface(), norm_lemma(m.dictionary_form())) for m in tok.tokenize(text, mode)
    ]
    paired = align_spans_paired(text, [t[0] for t in toks])
    spans = [p[0] for p in paired]
    lemmas = [toks[i][1] for _, i in paired]
    return spans, lemmas


def _analyzer_fn(name: str):
    if name == "mecab":
        return segment_mecab
    if name == "sudachi":
        return segment_sudachi
    raise KeyError(name)


def run(cfg: Config, analyzers: list[str] | None = None) -> None:
    analyzers = analyzers or ["mecab"]
    gold = read_jsonl(cfg.gold_path())
    if not gold:
        raise SystemExit("No gold data. Run `tok-eval prepare` first.")

    for analyzer in analyzers:
        fn = _analyzer_fn(analyzer)
        records: list[dict[str, Any]] = []
        version = ""
        for sent in gold:
            result = fn(sent["text"])
            if result is None:
                print(f"[baseline] {analyzer} unavailable, skipping")
                records = []
                break
            spans, lemmas = result
            records.append(
                {
                    "id": sent["id"],
                    "analyzer": analyzer,
                    "spans": [[s, e] for s, e in spans],
                    "lemmas": lemmas,
                    "parse_ok": True,
                }
            )
        if not records:
            continue
        if analyzer == "mecab":
            version = mecab_version()
        out = cfg.predictions_path(analyzer)
        write_jsonl(out, records)
        print(
            f"[baseline] {analyzer}: {len(records)} sentences -> {out.relative_to(cfg.root)}"
            + (f" ({version})" if version else "")
        )
