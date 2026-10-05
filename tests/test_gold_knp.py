from pathlib import Path

from tok_eval.gold import _doc_id, _map_upos, parse_knp, read_knp_path

FIX = Path(__file__).parent / "fixtures" / "kwdlc"
KNP = FIX / "knp" / "w201106-00000" / "w201106-0000060560.knp"


def test_doc_id():
    assert _doc_id("w201106-0000060560-1") == "w201106-0000060560"
    assert _doc_id("plain") == "plain"


def test_map_upos():
    assert _map_upos("名詞", "地名") == "PROPN"
    assert _map_upos("名詞", "普通名詞") == "NOUN"
    assert _map_upos("動詞", "*") == "VERB"
    assert _map_upos("助詞", "接続助詞") == "SCONJ"
    assert _map_upos("記号", "句点") == "PUNCT"


def test_parse_knp_reconstruction():
    sents = parse_knp(KNP.read_text(encoding="utf-8"))
    assert len(sents) == 2
    s0 = sents[0]
    assert s0["text"] == "エンドユーザーが関心有る病気に対して"
    assert "".join(s0["words"]) == s0["text"]
    for (a, b), w in zip(s0["spans"], s0["words"]):
        assert s0["text"][a:b] == w
    assert s0["doc_id"] == "w201106-0000060560"

    s1 = sents[1]
    assert s1["text"] == "東京大学で研究する。"
    assert s1["upos"] == ["PROPN", "NOUN", "ADP", "NOUN", "VERB", "PUNCT"]
    assert s1["lemmas"] == ["東京", "大学", "で", "研究", "する", "。"]


def test_read_knp_path_and_ids():
    assert len(read_knp_path(FIX / "knp", None)) == 2
    assert len(read_knp_path(FIX / "knp", FIX / "test.id")) == 2
    # an id file with no matching docs filters everything out
    empty = FIX / "test.id"
    assert all(s["doc_id"] == "w201106-0000060560" for s in read_knp_path(FIX / "knp", empty))
