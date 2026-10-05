from tok_eval.config import Config, DatasetCfg, LLMCfg, ModelCfg
from tok_eval.compare import _batch_of, _mcnemar, _pairs
from tok_eval.llm import (
    _call_openai,
    analyzer_id,
    batch_block,
    cache_key,
    choose_delim,
    choose_sep,
    ollama_timings,
    parse_segmentation,
    parse_segmentation_batch,
    parse_segmentation_lemma,
    strip_reasoning,
)

# Assemble the closing tag from parts so the literal never appears contiguously.
_THINK_CLOSE = "<" + "/think" + ">"
_THINK_OPEN = "<think" + ">"


def test_choose_delim_avoids_present():
    text = "a／b"
    d = choose_delim(text)
    assert d not in text


def test_parse_exact_delimiter():
    text = "東京都に住む"
    spans, ok = parse_segmentation(text, "東京／都／に／住む", "／")
    assert ok
    assert spans == [(0, 2), (2, 3), (3, 4), (4, 6)]


def test_parse_strips_code_fence():
    text = "すしを食べた"
    content = "```\nすし／を／食べた\n```"
    spans, ok = parse_segmentation(text, content, "／")
    assert ok
    assert spans == [(0, 2), (2, 3), (3, 6)]


def test_parse_rejects_altered_text():
    spans, ok = parse_segmentation("東京都", "東京／dou", "／")
    assert not ok


def test_parse_tolerates_delimiter_spaces():
    text = "すしを食べた"
    spans, ok = parse_segmentation(text, "すし ／ を ／ 食べた", "／")
    assert ok
    assert spans == [(0, 2), (2, 3), (3, 6)]


def test_parse_strips_reasoning_block():
    text = "すしを食べた"
    content = _THINK_OPEN + "Let me segment this." + _THINK_CLOSE + "すし／を／食べた"
    spans, ok = parse_segmentation(text, content, "／")
    assert ok
    assert spans == [(0, 2), (2, 3), (3, 6)]


def test_strip_reasoning_removes_all_blocks():
    content = _THINK_OPEN + "a" + _THINK_CLOSE + "x" + _THINK_OPEN + "b" + _THINK_CLOSE
    assert strip_reasoning(content) == "x"


def test_choose_sep_avoids_present():
    assert choose_sep("a→b", "／") != "→"


def test_parse_lemma_pairs():
    text = "東京都に住む"
    content = "東京→東京／都→都／に→に／住む→住む"
    spans, lemmas, ok = parse_segmentation_lemma(text, content, "／", "→")
    assert ok
    assert spans == [(0, 2), (2, 3), (3, 4), (4, 6)]
    assert lemmas == ["東京", "都", "に", "住む"]


def test_parse_lemma_rejects_altered_text():
    spans, lemmas, ok = parse_segmentation_lemma("東京都", "東京→東京／dou→dou", "／", "→")
    assert not ok


def test_ollama_timings_extraction():
    payload = {
        "eval_count": 100,
        "eval_duration": 2_000_000_000,
        "prompt_eval_count": 50,
        "prompt_eval_duration": 1_000_000_000,
        "total_duration": 5,
        "load_duration": 1,
        "done_reason": "stop",
    }
    t = ollama_timings(payload)
    assert abs(t["eval_tokens_per_s"] - 50.0) < 1e-9
    assert abs(t["prompt_tokens_per_s"] - 50.0) < 1e-9
    assert t["done_reason"] == "stop"


def test_ollama_timings_guards_missing_or_zero():
    assert ollama_timings({})["eval_tokens_per_s"] is None
    assert ollama_timings({"eval_count": 5, "eval_duration": 0})["eval_tokens_per_s"] is None


def test_batch_block_and_analyzer_id():
    assert batch_block(["a", "b"]) == "S1: a\nS2: b"
    assert analyzer_id("m", None) == "m"
    assert analyzer_id("m", 4) == "m@b4"
    assert analyzer_id("m", None, True) == "m@lem"
    assert analyzer_id("m", 2, True) == "m@b2@lem"


def test_prompt_path_batch_lemma():
    from pathlib import Path

    cfg = Config(root=Path("/tmp"), dataset=DatasetCfg(), models=[])
    assert cfg.prompt_path().name == "segment_ja.txt"
    assert cfg.prompt_path(lemma=True).name == "segment_ja_lemma.txt"
    assert cfg.prompt_path(batch=True).name == "segment_ja_batch.txt"
    assert cfg.prompt_path(lemma=True, batch=True).name == "segment_ja_batch_lemma.txt"


def test_parse_batch_lines():
    texts = ["東京都", "すしを食べた"]
    content = "S1: 東京／都\nS2: すし／を／食べた"
    res = parse_segmentation_batch(texts, content, "／", "")
    assert [ok for _, _, ok in res] == [True, True]
    assert res[0][0] == [(0, 2), (2, 3)]
    assert res[1][0] == [(0, 2), (2, 3), (3, 6)]


def test_parse_batch_missing_line_isolated():
    texts = ["東京都", "すしを食べた"]
    res = parse_segmentation_batch(texts, "S1: 東京／都", "／", "")
    assert res[0][2] is True
    assert res[1][2] is False


def test_parse_batch_reordered_lines():
    texts = ["東京都", "すし"]
    res = parse_segmentation_batch(texts, "S2: すし\nS1: 東京／都", "／", "")
    assert res[0][0] == [(0, 2), (2, 3)]
    assert res[1][0] == [(0, 2)]


def test_parse_batch_lemma_lines():
    texts = ["東京都", "すし"]
    content = "S1: 東京→東京／都→都\nS2: すし→寿司"
    res = parse_segmentation_batch(texts, content, "／", "→")
    assert [ok for _, _, ok in res] == [True, True]
    assert res[0][1] == ["東京", "都"]
    assert res[1][1] == ["寿司"]


def test_compare_helpers():
    assert _batch_of("m@b8") == 8
    assert _batch_of("m") == 1
    assert _batch_of("m@b2@lem") == 2
    assert _mcnemar(0, 0) == 1.0
    assert 0.0 <= _mcnemar(5, 0) <= 1.0
    assert ("m@b8", "m@b1") in _pairs(["m", "m@b1", "m@b8"])
    assert ("m@b1", "m") in _pairs(["m", "m@b1", "m@b8"])
    assert ("m@b8", "m") in _pairs(["m", "m@b1", "m@b8"])
    # lemma arms group separately from token-only arms
    lemma_pairs = _pairs(["m@lem", "m@b1@lem", "m@b2@lem"])
    assert ("m@b2@lem", "m@b1@lem") in lemma_pairs
    assert ("m@b1@lem", "m@lem") in lemma_pairs
    assert ("m@b1", "m@lem") not in _pairs(["m", "m@lem", "m@b1"])


def test_cache_key_varies_with_variant():
    a = cache_key("m", "u", "p", 0.0, "t", "ollama|low")
    b = cache_key("m", "u", "p", 0.0, "t", "ollama|none")
    assert a != b
    assert cache_key("m", "u", "p", 0.0, "t", "") == cache_key("m", "u", "p", 0.0, "t")


def test_native_base_url_strips_v1():
    assert (
        ModelCfg(name="x", base_url="http://localhost:11434/v1").resolved_native_base_url()
        == "http://localhost:11434"
    )
    assert (
        ModelCfg(name="x", base_url="http://h:1234/v1/").resolved_native_base_url()
        == "http://h:1234"
    )


def test_model_usable_requires_api_key():
    missing = ModelCfg(
        name="x", base_url="http://h/v1", api_key_env="DEFINITELY_UNSET_KEY", requires_api_key=True
    )
    assert missing.usable() is False
    optional = ModelCfg(
        name="x", base_url="http://h/v1", api_key_env="DEFINITELY_UNSET_KEY"
    )
    assert optional.usable() is True
    no_url = ModelCfg(name="x", base_url_env="DEFINITELY_UNSET_URL")
    assert no_url.usable() is False


def test_per_model_overrides():
    m = ModelCfg(name="x", max_tokens=64, num_ctx=4096, extra_body={"chat_template_kwargs": {"thinking": False}})
    assert m.max_tokens == 64
    assert m.num_ctx == 4096
    eb = m.extra_body or {}
    assert eb["chat_template_kwargs"]["thinking"] is False


def test_api_model_falls_back_to_name():
    assert ModelCfg(name="llama3.1:8b").api_model() == "llama3.1:8b"
    m = ModelCfg(name="opencode-go/deepseek-v4.1-flash", model="deepseek-v4.1-flash")
    assert m.api_model() == "deepseek-v4.1-flash"


def test_extra_headers_roundtrip():
    m = ModelCfg(name="x", extra_headers={"User-Agent": "llm-tok-eval/0.1.0"})
    assert (m.extra_headers or {}).get("User-Agent") == "llm-tok-eval/0.1.0"


class _FakeCompletions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        import types

        self.kwargs = kwargs
        msg = types.SimpleNamespace(content="東京／都", model_extra={})
        choice = types.SimpleNamespace(message=msg)
        usage = types.SimpleNamespace(completion_tokens=2, prompt_tokens=5)
        return types.SimpleNamespace(
            choices=[choice], model="deepseek-v4.1-flash", usage=usage
        )


class _FakeClient:
    def __init__(self):
        import types

        self.chat = types.SimpleNamespace(completions=_FakeCompletions())


def test_openai_call_uses_api_model_and_headers():
    from pathlib import Path

    cfg = Config(
        root=Path("/tmp"),
        dataset=DatasetCfg(),
        models=[],
        llm=LLMCfg(max_tokens=123),
    )
    model = ModelCfg(
        name="opencode-go/deepseek-v4.1-flash",
        model="deepseek-v4.1-flash",
        extra_headers={"User-Agent": "llm-tok-eval/0.1.0", "x-opencode-session": "llm-tok-eval"},
    )
    client = _FakeClient()
    prompt = {"system": "s", "user": "{text} | {delim} | {sep}"}
    res = _call_openai(client, model, prompt, "東京都", "／", "", cfg, None)
    sent = client.chat.completions.kwargs
    assert sent["model"] == "deepseek-v4.1-flash"
    assert sent["max_tokens"] == 123
    assert sent["extra_headers"]["x-opencode-session"] == "llm-tok-eval"
    assert res["content"] == "東京／都"
