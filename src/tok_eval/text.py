"""Character-span utilities for word segmentation.

A segmentation of a string is represented as a list of ``(start, end)``
character offsets. Offsets are into the raw (un-normalised) sentence.
"""

from __future__ import annotations

import unicodedata

Span = tuple[int, int]


def tokens_to_spans(tokens: list[str]) -> list[Span]:
    """Concatenate tokens into contiguous spans (no separators assumed)."""
    spans: list[Span] = []
    pos = 0
    for tok in tokens:
        spans.append((pos, pos + len(tok)))
        pos += len(tok)
    return spans


def spans_to_tokens(text: str, spans: list[Span]) -> list[str]:
    return [text[s:e] for s, e in spans]


def validate_spans(text: str, spans: list[Span]) -> bool:
    """True if spans tile ``text`` exactly from 0 to len(text)."""
    pos = 0
    for s, e in spans:
        if s != pos or e < s or e > len(text):
            return False
        pos = e
    return pos == len(text)


def align_surfaces(text: str, surfaces: list[str]) -> list[Span]:
    """Locate each surface in ``text`` in order, tolerating dropped separators.

    MeCab and LLM output can omit or normalise whitespace; this walks the text
    and finds each surface sequentially, allowing gaps (e.g. spaces).
    """
    spans: list[Span] = []
    pos = 0
    for surface in surfaces:
        if not surface:
            continue
        idx = text.find(surface, pos)
        if idx < 0:
            idx = text.find(surface)
        if idx < 0:
            continue
        spans.append((idx, idx + len(surface)))
        pos = idx + len(surface)
    return spans


def align_spans_paired(text: str, tokens: list[str]) -> list[tuple[Span, int]]:
    """Like :func:`align_surfaces` but keeps each token's original index.

    Returns ``(span, index)`` pairs so parallel fields (e.g. lemmas) stay
    aligned with the located surface.
    """
    out: list[tuple[Span, int]] = []
    pos = 0
    for i, tok in enumerate(tokens):
        if not tok:
            continue
        idx = text.find(tok, pos)
        if idx < 0:
            idx = text.find(tok)
        if idx < 0:
            continue
        out.append(((idx, idx + len(tok)), i))
        pos = idx + len(tok)
    return out


def norm_lemma(value: str | None) -> str:
    """Normalise a lemma for comparison (``_``/``*``/empty mean 'unset')."""
    if value is None:
        return ""
    value = str(value).strip()
    return "" if value in {"_", "*"} else value


def boundaries(spans: list[Span]) -> set[int]:
    """Word-start offsets, excluding sentence start (0)."""
    return {s for s, _ in spans if s != 0}


def boundary_counts(gold: list[Span], pred: list[Span]) -> tuple[int, int, int]:
    gb = boundaries(gold)
    pb = boundaries(pred)
    return len(gb & pb), len(pb - gb), len(gb - pb)


def word_counts(gold: list[Span], pred: list[Span]) -> tuple[int, int, int]:
    """Exact word-span matching: a predicted word must equal a gold word."""
    gset, pset = set(gold), set(pred)
    return len(gset & pset), len(pset - gset), len(gset - pset)


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def is_katakana(s: str) -> bool:
    if not s:
        return False
    letters = [c for c in s if not unicodedata.category(c).startswith("P")]
    if not letters:
        return False
    return all("\u30a0" <= c <= "\u30ff" for c in letters)
