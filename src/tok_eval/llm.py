"""Prompt-based LLM segmentation over an OpenAI-compatible endpoint.

Every raw completion is cached to ``results/raw/<treebank>/<analyzer>.jsonl``
keyed by a hash of (analyzer, base_url, prompt, temperature, request variant,
text). Re-running never re-queries a cached sentence, so results stay
reproducible even if a remote alias silently changes.
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
import time
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from .config import Config, ModelCfg
from .io import append_jsonl, read_jsonl, write_jsonl
from .text import align_surfaces, norm_lemma, tokens_to_spans, validate_spans

_DELIMS = ["\uff0f", "|", "\u2502", "/", "\t"]
_SEPS = ["\u2192", "\u21d2", "=>", ":", "\t", "|"]

# Reasoning models (qwen3, gpt-oss, ...) may wrap their chain-of-thought in
# think blocks. The tags are assembled from split literals so the closing tag is
# never present as one contiguous token in this source file.
_THINK_OPEN = "<" "think" ">"
_THINK_CLOSE = "<" "/think" ">"
_THINK_RE = re.compile(
    re.escape(_THINK_OPEN) + r".*?" + re.escape(_THINK_CLOSE),
    re.IGNORECASE | re.DOTALL,
)


def strip_reasoning(content: str) -> str:
    """Remove think blocks and surrounding whitespace from model output."""
    return _THINK_RE.sub("", content or "").strip()


def load_prompt(path: Path) -> tuple[dict[str, str], str]:
    raw = path.read_bytes()
    prompt = yaml.safe_load(raw.decode("utf-8"))
    sha = hashlib.sha256(raw).hexdigest()
    return prompt, sha


def choose_delim(text: str) -> str:
    for d in _DELIMS:
        if d not in text:
            return d
    return "\uE000"


def choose_sep(text: str, delim: str) -> str:
    for s in _SEPS:
        if s not in text and s != delim:
            return s
    return "\uE001"


def _clean(content: str) -> str:
    content = strip_reasoning((content or "").strip())
    if content.startswith("```"):
        content = re.sub(r"^```[a-zA-Z0-9]*\n?", "", content)
        content = re.sub(r"\n?```$", "", content)
    return content.strip()


def cache_key(
    model: str, base_url: str, prompt_sha: str, temp: float, text: str, variant: str = ""
) -> str:
    h = hashlib.sha256()
    for part in (model, base_url, prompt_sha, repr(temp), variant, text):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def parse_segmentation(text: str, content: str, delim: str) -> tuple[list[tuple[int, int]], bool]:
    content = _clean(content)
    parts = content.split(delim)
    if "".join(parts) == text:
        return tokens_to_spans(parts), True
    trimmed = [p.strip() for p in parts if p.strip()]
    if "".join(trimmed) == text:
        return tokens_to_spans(trimmed), True
    spans = align_surfaces(text, trimmed or parts)
    return spans, bool(spans) and validate_spans(text, spans)


def parse_segmentation_lemma(
    text: str, content: str, delim: str, sep: str
) -> tuple[list[tuple[int, int]], list[str], bool]:
    """Parse ``surface<sep>lemma`` tokens joined by ``delim``."""
    content = _clean(content)
    surfaces: list[str] = []
    lemmas: list[str] = []
    for part in content.split(delim):
        surface, found, lemma = part.partition(sep)
        surfaces.append(surface)
        lemmas.append(norm_lemma(lemma) if found else "")
    if "".join(surfaces) == text:
        return tokens_to_spans(surfaces), lemmas, True
    return [], [], False


def _render_user(prompt: dict[str, str], text: str, delim: str, sep: str) -> str:
    return (
        prompt["user"]
        .replace("{delim}", delim)
        .replace("{sep}", sep)
        .replace("{text}", text)
    )


_BATCH_LINE = re.compile(r"^\s*S(\d+)\s*[:：]\s*(.*)$")


def batch_block(texts: list[str]) -> str:
    """Render the labeled input block for the batch prompt."""
    return "\n".join(f"S{i + 1}: {t}" for i, t in enumerate(texts))


def analyzer_id(name: str, batch_size: int | None, lemma: bool = False) -> str:
    """Local analyzer identity.

    ``<model>`` (reference) with optional ``@b<N>`` batch and ``@lem`` lemma
    suffixes, e.g. ``gemma3:27b@b2@lem``. Lemma runs get distinct raw/prediction
    files so they never overwrite the plain segmentation results.
    """
    suffix = ""
    if batch_size is not None:
        suffix += f"@b{batch_size}"
    if lemma:
        suffix += "@lem"
    return f"{name}{suffix}"


def parse_segmentation_batch(
    texts: list[str], content: str, delim: str, sep: str
) -> list[tuple[list[tuple[int, int]], list[str] | None, bool]]:
    """Parse labeled ``S<k>:`` lines into per-sentence (spans, lemmas, ok)."""
    cleaned = _clean(content)
    found: dict[int, str] = {}
    for line in cleaned.splitlines():
        m = _BATCH_LINE.match(line)
        if m:
            found[int(m.group(1)) - 1] = m.group(2)
    results: list[tuple[list[tuple[int, int]], list[str] | None, bool]] = []
    for i, text in enumerate(texts):
        seg = found.get(i)
        if seg is None:
            results.append(([], None, False))
            continue
        if sep:
            spans, lemmas, ok = parse_segmentation_lemma(text, seg, delim, sep)
            results.append((spans, lemmas, ok))
        else:
            spans, ok = parse_segmentation(text, seg, delim)
            results.append((spans, None, ok))
    return results


def _rate(count: int | None, duration_ns: int | None) -> float | None:
    if not count or not duration_ns:
        return None
    return count / (duration_ns / 1e9)


def ollama_timings(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract throughput fields from an Ollama /api/chat response."""
    ec = payload.get("eval_count")
    ed = payload.get("eval_duration")
    pc = payload.get("prompt_eval_count")
    pd = payload.get("prompt_eval_duration")
    return {
        "eval_count": ec,
        "eval_duration_ns": ed,
        "prompt_eval_count": pc,
        "prompt_eval_duration_ns": pd,
        "total_duration_ns": payload.get("total_duration"),
        "load_duration_ns": payload.get("load_duration"),
        "done_reason": payload.get("done_reason"),
        "eval_tokens_per_s": _rate(ec, ed),
        "prompt_tokens_per_s": _rate(pc, pd),
    }


def _post_json(url: str, payload: dict[str, Any], timeout: float = 1800.0) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _call_openai(
    client,
    model: ModelCfg,
    prompt: dict[str, str],
    text: str,
    delim: str,
    sep: str,
    cfg: Config,
    effort: str | None,
) -> dict[str, Any]:
    extra_body: dict[str, Any] = dict(model.extra_body or {})
    if effort:
        extra_body.setdefault("reasoning_effort", effort)
    kwargs: dict[str, Any] = {
        "model": model.api_model(),
        "messages": [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": _render_user(prompt, text, delim, sep)},
        ],
        "temperature": cfg.llm.temperature,
        "top_p": cfg.llm.top_p,
        "max_tokens": model.max_tokens or cfg.llm.max_tokens,
    }
    if extra_body:
        kwargs["extra_body"] = extra_body
    if model.extra_headers:
        kwargs["extra_headers"] = model.extra_headers
    t0 = time.perf_counter()
    resp = client.chat.completions.create(**kwargs)
    wall = time.perf_counter() - t0
    msg = resp.choices[0].message
    content = msg.content or ""
    extra = getattr(msg, "model_extra", None) or {}
    reasoning = (
        getattr(msg, "reasoning", None)
        or extra.get("reasoning")
        or extra.get("reasoning_content")
    )
    usage = getattr(resp, "usage", None)
    timings = {
        "wall_s": wall,
        "eval_count": getattr(usage, "completion_tokens", None) if usage else None,
        "prompt_eval_count": getattr(usage, "prompt_tokens", None) if usage else None,
    }
    return {
        "content": content,
        "version": getattr(resp, "model", model.api_model()),
        "timings": timings,
        "thinking": reasoning,
        "content_empty": content.strip() == "",
    }


def _call_ollama(
    base_url: str,
    model: ModelCfg,
    prompt: dict[str, str],
    text: str,
    delim: str,
    sep: str,
    cfg: Config,
    effort: str | None,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/api/chat"
    options: dict[str, Any] = {
        "temperature": cfg.llm.temperature,
        "top_p": cfg.llm.top_p,
        "num_predict": model.max_tokens or cfg.llm.max_tokens,
    }
    if model.num_ctx:
        options["num_ctx"] = model.num_ctx
    payload: dict[str, Any] = {
        "model": model.api_model(),
        "messages": [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": _render_user(prompt, text, delim, sep)},
        ],
        "stream": False,
        "options": options,
    }
    if model.extra_body:
        payload.update(model.extra_body)
    if effort == "none":
        payload["think"] = False
    elif effort:
        payload["reasoning_effort"] = effort
    t0 = time.perf_counter()
    resp = _post_json(url, payload)
    wall = time.perf_counter() - t0
    msg = resp.get("message") or {}
    content = msg.get("content") or ""
    timings = ollama_timings(resp)
    timings["wall_s"] = wall
    return {
        "content": content,
        "version": resp.get("model", model.api_model()),
        "timings": timings,
        "thinking": msg.get("thinking"),
        "content_empty": content.strip() == "",
    }


def _invoke(
    model: ModelCfg,
    client,
    prompt: dict[str, str],
    text: str,
    delim: str,
    sep: str,
    cfg: Config,
    effort: str | None,
) -> dict[str, Any]:
    if model.api == "ollama":
        base = model.resolved_native_base_url() or model.resolved_base_url() or ""
        return _call_ollama(base, model, prompt, text, delim, sep, cfg, effort)
    return _call_openai(client, model, prompt, text, delim, sep, cfg, effort)


def run(
    cfg: Config,
    models: list[str] | None = None,
    offline: bool = False,
    limit: int | None = None,
    batch_size: int | None = None,
) -> dict[str, int]:
    from openai import OpenAI

    gold = read_jsonl(cfg.gold_path())
    if not gold:
        raise SystemExit("No gold data. Run `tok-eval prepare` first.")
    if limit is not None:
        gold = gold[:limit]

    if batch_size is None:
        batch_size = cfg.llm.batch_size
    batch_mode = batch_size is not None
    n_batch = batch_size or 1

    lemma_mode = cfg.llm.lemmas
    prompt, prompt_sha = load_prompt(
        cfg.prompt_path(lemma=lemma_mode, batch=batch_mode)
    )
    if batch_mode:
        prompt = {**prompt, "user": prompt["user"].replace("{n}", str(n_batch))}

    if models is None:
        selected = [m for m in cfg.models if m.enabled]
    else:
        # explicit --models opt-in runs a model even if disabled by default
        selected = [m for m in cfg.models if m.name in models]
    if not selected:
        print("[llm] no models selected")
        return {}

    summary: dict[str, int] = {}
    for model in selected:
        if model.resolved_base_url() is None:
            print(f"[llm] skip {model.name}: no base_url/credentials")
            continue
        if model.requires_api_key and not model.has_api_key():
            print(f"[llm] skip {model.name}: missing {model.api_key_env}")
            continue

        base_url = model.resolved_base_url() or ""
        effort = model.reasoning_effort or cfg.llm.reasoning_effort
        analyzer = analyzer_id(model.name, batch_size, lemma_mode)
        base_cap = model.max_tokens or cfg.llm.max_tokens
        cap = base_cap * n_batch if batch_mode else base_cap
        call_model = replace(model, max_tokens=cap)
        if batch_mode:
            variant = (
                f"{model.api}|{effort}|{model.extra_body}|{model.num_ctx}"
                f"|{model.max_tokens}|b{n_batch}|cap{cap}|lem{lemma_mode}"
            )
        else:
            # single-sentence reference: variant without batch fields
            variant = f"{model.api}|{effort}|{model.extra_body}|{model.num_ctx}|{model.max_tokens}"
        raw_path = cfg.raw_path(analyzer)
        cache = {r["key"]: r for r in read_jsonl(raw_path)}
        client = (
            OpenAI(base_url=base_url, api_key=model.resolved_api_key())
            if model.api == "openai"
            else None
        )
        preds: list[dict[str, Any]] = []
        n_ok = 0
        n_err = 0
        n_empty = 0
        n_skipped = 0
        rates: list[float] = []
        n_batches = 0
        n_batch_ok = 0

        units = (
            [gold[i : i + n_batch] for i in range(0, len(gold), n_batch)]
            if batch_mode
            else [[g] for g in gold]
        )

        for bi, unit in enumerate(tqdm(units, desc=f"llm:{analyzer}", leave=False)):
            texts = [s["text"] for s in unit]
            joined = "".join(texts)
            delim = choose_delim(joined)
            sep = choose_sep(joined, delim) if lemma_mode else ""
            render_text = batch_block(texts) if batch_mode else texts[0]
            key = cache_key(
                analyzer, base_url, prompt_sha, cfg.llm.temperature, render_text, variant
            )

            # Cached errors (e.g. a bad/expired key) and empty/truncated
            # responses (e.g. all tokens spent reasoning) are retried when online.
            cached = cache.get(key)
            if (
                cached is not None
                and not cached.get("error")
                and not cached.get("content_empty")
            ):
                rec = cached
            elif offline:
                n_skipped += len(unit)
                continue
            else:
                rec = {
                    "key": key,
                    "model": model.name,
                    "analyzer": analyzer,
                    "base_url": base_url,
                    "api": model.api,
                    "reasoning_effort": effort,
                    "lemma_mode": lemma_mode,
                    "prompt_sha256": prompt_sha,
                    "temperature": cfg.llm.temperature,
                    "batch_size": n_batch,
                    "batch_index": bi,
                    "texts": texts,
                    "text": render_text,
                    "delim": delim,
                    "sep": sep,
                    "response": None,
                    "thinking": None,
                    "model_version": None,
                    "content_empty": None,
                    "error": None,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                try:
                    res = _invoke(
                        call_model, client, prompt, render_text, delim, sep, cfg, effort
                    )
                    rec["response"] = res["content"]
                    rec["model_version"] = res["version"]
                    rec["thinking"] = res.get("thinking")
                    rec["content_empty"] = res["content_empty"]
                    rec.update(res.get("timings") or {})
                except Exception as exc:  # noqa: BLE001 - report and continue
                    rec["error"] = f"{type(exc).__name__}: {exc}"
                append_jsonl(raw_path, [rec])
                cache[key] = rec

            if rec.get("error"):
                n_err += 1
                continue
            if rec.get("eval_tokens_per_s"):
                rates.append(rec["eval_tokens_per_s"])
            response = rec.get("response") or ""
            if not response.strip():
                n_empty += 1

            if batch_mode:
                item_results = parse_segmentation_batch(
                    texts, response, rec["delim"], rec.get("sep") or ""
                )
                n_batches += 1
                n_batch_ok += int(all(ok for _, _, ok in item_results))
            else:
                if lemma_mode:
                    spans, lemmas, ok = parse_segmentation_lemma(
                        texts[0], response, rec["delim"], rec.get("sep") or ""
                    )
                else:
                    spans, ok = parse_segmentation(texts[0], response, rec["delim"])
                    lemmas = None
                item_results = [(spans, lemmas, ok)]

            eval_count = rec.get("eval_count")
            wall_s = rec.get("wall_s")
            for pos, (sent, (spans, lemmas, ok)) in enumerate(zip(unit, item_results)):
                n_ok += int(ok)
                pred: dict[str, Any] = {
                    "id": sent["id"],
                    "analyzer": analyzer,
                    "tier": model.tier,
                    "spans": [[s, e] for s, e in spans],
                    "parse_ok": ok,
                    "prompt_sha256": prompt_sha,
                    "model_version": rec.get("model_version"),
                    "eval_tokens_per_s": rec.get("eval_tokens_per_s"),
                    "content_empty": bool(rec.get("content_empty")),
                    "lemma_mode": lemma_mode,
                    "batch_size": n_batch,
                    "batch_index": bi if batch_mode else None,
                    "position_in_batch": pos if batch_mode else None,
                }
                if eval_count:
                    pred["tokens_per_sentence"] = eval_count / n_batch
                if wall_s:
                    pred["wall_s_per_sentence"] = wall_s / n_batch
                if lemma_mode:
                    pred["lemmas"] = lemmas
                preds.append(pred)

        if not preds:
            print(
                f"[llm] {analyzer}: no predictions "
                f"(errors={n_err}, offline-skipped={n_skipped}); "
                "keeping any existing predictions file"
            )
            summary[analyzer] = 0
            continue

        write_jsonl(cfg.predictions_path(analyzer), preds)
        summary[analyzer] = n_ok
        speed = f", eval={statistics.median(rates):.1f} tok/s" if rates else ""
        batch_note = (
            f", batch_parse={n_batch_ok}/{n_batches}" if batch_mode and n_batches else ""
        )
        print(
            f"[llm] {analyzer}: {n_ok}/{len(preds)} parsed ok"
            + batch_note
            + (f", {n_empty} empty-content" if n_empty else "")
            + (f", {n_err} errors" if n_err else "")
            + (f", {n_skipped} offline-skipped" if n_skipped else "")
            + speed
            + f" -> {cfg.predictions_path(analyzer).relative_to(cfg.root)}"
        )
    return summary
