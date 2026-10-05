from tok_eval.text import (
    align_spans_paired,
    align_surfaces,
    boundary_counts,
    is_katakana,
    norm_lemma,
    prf,
    tokens_to_spans,
    validate_spans,
    word_counts,
)


def test_tokens_to_spans():
    assert tokens_to_spans(["東京都", "に", "住む"]) == [(0, 3), (3, 4), (4, 6)]


def test_validate_spans():
    text = "東京都に住む"
    assert validate_spans(text, [(0, 3), (3, 4), (4, 6)])
    assert not validate_spans(text, [(0, 3), (4, 6)])


def test_align_surfaces_with_gap():
    text = "hello world"
    assert align_surfaces(text, ["hello", "world"]) == [(0, 5), (6, 11)]


def test_boundary_counts():
    gold = [(0, 2), (2, 3), (3, 5)]
    pred = [(0, 2), (2, 5)]
    assert boundary_counts(gold, pred) == (1, 0, 1)
    assert word_counts(gold, pred) == (1, 1, 2)


def test_prf():
    p, r, f = prf(1, 1, 1)
    assert abs(p - 0.5) < 1e-9
    assert abs(f - 0.5) < 1e-9


def test_is_katakana():
    assert is_katakana("コンピュータ")
    assert not is_katakana("東京")
    assert not is_katakana("abc")


def test_align_spans_paired_keeps_indices():
    text = "東京都"
    # second token deliberately not found; indices must track originals
    pairs = align_spans_paired(text, ["東京", "XXX", "都"])
    assert pairs == [((0, 2), 0), ((2, 3), 2)]


def test_norm_lemma():
    assert norm_lemma(None) == ""
    assert norm_lemma("_") == ""
    assert norm_lemma("*") == ""
    assert norm_lemma(" 住む ") == "住む"
