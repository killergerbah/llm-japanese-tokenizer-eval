"""Build gold word-segmentation data from CoNLL-U, KWDLC (KNP), or the Hub.

Gold is reconstructed from the word/morpheme units of a corpus so it is
internally consistent with a single segmentation convention. Multi-word tokens
and empty nodes are skipped; their surface forms are recovered from the
constituent words. ``SpaceAfter=No`` in MISC is honoured so Latin content keeps
its spaces.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import Config
from .io import write_jsonl


def _sentence_from_rows(
    words: list[str], upos: list[str], misc: list[str] | None, lemmas: list[str] | None
) -> dict[str, Any]:
    misc = misc if misc is not None else ["SpaceAfter=No"] * len(words)
    lemmas = lemmas if lemmas is not None else words
    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    pos = 0
    for i, word in enumerate(words):
        spans.append((pos, pos + len(word)))
        parts.append(word)
        pos += len(word)
        no_space = "SpaceAfter=No" in misc[i]
        if not no_space and i != len(words) - 1:
            parts.append(" ")
            pos += 1
    return {
        "text": "".join(parts),
        "spans": spans,
        "words": words,
        "upos": upos,
        "lemmas": lemmas,
    }


def parse_conllu(text: str) -> list[dict[str, Any]]:
    sentences: list[dict[str, Any]] = []
    for block in text.split("\n\n"):
        words: list[str] = []
        upos: list[str] = []
        misc: list[str] = []
        lemmas: list[str] = []
        for line in block.split("\n"):
            if not line or line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) < 10:
                continue
            tid = cols[0]
            if "-" in tid or "." in tid:
                continue  # MWT range / empty node
            words.append(cols[1])
            lemmas.append(cols[2])
            upos.append(cols[3])
            misc.append(cols[9])
        if words:
            sentences.append(_sentence_from_rows(words, upos, misc, lemmas))
    return sentences


def read_conllu_path(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        sentences: list[dict[str, Any]] = []
        for f in sorted(path.glob("*.conllu")):
            sentences.extend(parse_conllu(f.read_text(encoding="utf-8")))
        return sentences
    return parse_conllu(path.read_text(encoding="utf-8"))


# --- KWDLC / Kyoto Corpus (KNP format) ------------------------------------

_JUMAN_UPOS = {
    "動詞": "VERB",
    "形容詞": "ADJ",
    "形容動詞": "ADJ",
    "副詞": "ADV",
    "連体詞": "DET",
    "接続詞": "CCONJ",
    "感動詞": "INTJ",
    "助詞": "ADP",
    "助動詞": "AUX",
    "接頭詞": "NOUN",
    "接尾辞": "NOUN",
    "接尾": "NOUN",
    "記号": "PUNCT",
    "名詞": "NOUN",
    "未定義語": "X",
    "その他": "X",
}
_PROPN_FINE = {"人名", "地名", "組織名", "固有", "固有名詞"}


def _map_upos(pos: str, fine: str) -> str:
    if pos == "名詞":
        if fine in _PROPN_FINE:
            return "PROPN"
        if fine in {"数詞", "数量詞"}:
            return "NUM"
        return "NOUN"
    if pos == "助詞" and fine == "接続助詞":
        return "SCONJ"
    return _JUMAN_UPOS.get(pos, "X")


def _doc_id(sid: str) -> str:
    head, _, tail = sid.rpartition("-")
    return head if head and tail.isdigit() else sid


def _sentence_from_knp(
    morphemes: list[tuple[str, str, str, str]]
) -> dict[str, Any]:
    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    words: list[str] = []
    lemmas: list[str] = []
    upos: list[str] = []
    pos = 0
    for surface, lemma, p, fine in morphemes:
        if surface.strip() == "":  # whitespace token -> separator, not a word
            if parts:
                parts.append(" ")
                pos += 1
            continue
        spans.append((pos, pos + len(surface)))
        parts.append(surface)
        pos += len(surface)
        words.append(surface)
        lemmas.append(lemma if lemma not in ("", "*", "_") else surface)
        upos.append(_map_upos(p, fine))
    return {"text": "".join(parts), "spans": spans, "words": words, "upos": upos, "lemmas": lemmas}


def parse_knp(text: str) -> list[dict[str, Any]]:
    """Parse KNP-format annotations (KWDLC / Kyoto Corpus) into sentences.

    Morpheme lines are ``surface reading lemma POS pos_id finePOS ...``. ``*``
    (bunsetsu) and ``+`` (basic phrase) lines are ignored.
    """
    sents: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None
    for line in text.splitlines():
        if line.startswith("#"):
            if line.startswith("# S-ID:"):
                cur = {"sid": line[7:].strip().split()[0] if line[7:].strip() else "", "mor": []}
                sents.append(cur)
            continue
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == "EOS":
            cur = None
            continue
        if stripped[0] in "*+":
            continue
        if cur is None:
            cur = {"sid": "", "mor": []}
            sents.append(cur)
        cols = stripped.split()
        if len(cols) < 4:
            continue
        surface = cols[0]
        lemma = cols[2] if len(cols) > 2 else surface
        pos = cols[3]
        fine = cols[5] if len(cols) > 5 else "*"
        cur["mor"].append((surface, lemma, pos, fine))

    out: list[dict[str, Any]] = []
    for sent in sents:
        if not sent["mor"]:
            continue
        d = _sentence_from_knp(sent["mor"])
        if not d["words"]:
            continue
        d["doc_id"] = _doc_id(sent["sid"])
        d["sent_id"] = sent["sid"]
        out.append(d)
    return out


def read_knp_path(path: Path, ids_path: Path | None = None) -> list[dict[str, Any]]:
    ids: set[str] | None = None
    if ids_path is not None:
        ids = set(ids_path.read_text(encoding="utf-8").split())
    sentences: list[dict[str, Any]] = []
    files = sorted(path.rglob("*.knp")) if path.is_dir() else [path]
    for f in files:
        for sent in parse_knp(f.read_text(encoding="utf-8")):
            if ids is None or sent.get("doc_id") in ids:
                sentences.append(sent)
    return sentences


def _load_hf(cfg: Config) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "The [gold] extra is required for source: hf. "
            "Install with `pip install -e '.[gold]'` or use source: conllu."
        ) from exc

    ds = load_dataset(
        cfg.dataset.name,
        cfg.dataset.config,
        split=cfg.dataset.split,
        revision=cfg.dataset.revision,
    )
    sentences: list[dict[str, Any]] = []
    for ex in ds:
        words = list(ex["tokens"])
        upos = list(ex.get("upos") or ["_"] * len(words))
        raw_misc = ex.get("misc")
        misc = [str(m) for m in raw_misc] if raw_misc is not None else None
        raw_lemmas = ex.get("lemmas")
        if raw_lemmas is None:
            raw_lemmas = ex.get("lemma")
        lemmas = [str(l) for l in raw_lemmas] if raw_lemmas is not None else None
        sentences.append(_sentence_from_rows(words, upos, misc, lemmas))
    return sentences


def build_gold(cfg: Config, limit: int | None = None) -> list[dict[str, Any]]:
    if cfg.dataset.source == "conllu":
        if not cfg.dataset.path:
            raise SystemExit("dataset.path is required when source: conllu")
        raw = read_conllu_path(cfg.root / cfg.dataset.path)
    elif cfg.dataset.source == "knp":
        if not cfg.dataset.path:
            raise SystemExit("dataset.path is required when source: knp")
        ids_path = (cfg.root / cfg.dataset.ids_path) if cfg.dataset.ids_path else None
        raw = read_knp_path(cfg.root / cfg.dataset.path, ids_path)
    elif cfg.dataset.source == "hf":
        raw = _load_hf(cfg)
    else:
        raise SystemExit(f"unknown dataset.source: {cfg.dataset.source!r}")

    if limit is not None:
        raw = raw[:limit]

    records: list[dict[str, Any]] = []
    for i, sent in enumerate(raw):
        records.append(
            {
                "id": f"{cfg.treebank}-{i:06d}",
                "treebank": cfg.treebank,
                "text": sent["text"],
                "spans": [[s, e] for s, e in sent["spans"]],
                "words": sent["words"],
                "upos": sent["upos"],
                "lemmas": sent.get("lemmas", sent["words"]),
            }
        )
    return records


def run(cfg: Config, limit: int | None = None) -> Path:
    records = build_gold(cfg, limit=limit)
    out = cfg.gold_path()
    write_jsonl(out, records)
    print(f"[prepare] wrote {len(records)} sentences -> {out.relative_to(cfg.root)}")
    return out
