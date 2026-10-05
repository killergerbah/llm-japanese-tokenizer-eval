"""Score segmentations against gold.

Two complementary metrics are reported:

* ``boundary_*``  - character-boundary precision/recall/F1. Robust to
  segmentation-convention differences between analyzers.
* ``word_*``      - exact word-span precision/recall/F1. Stricter.

Micro scores over all sentences are reported with bootstrap confidence
intervals; a ``word_recall`` breakdown is reported per gold-word category.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from typing import Any, Callable

import numpy as np

from .config import Config
from .io import read_jsonl, write_jsonl
from .text import boundary_counts, is_katakana, norm_lemma, prf, word_counts

FAMILIES = ("boundary", "word")
METRICS = ("precision", "recall", "f1")


def _metric_value(metric: str, tp: int, fp: int, fn: int) -> float:
    p, r, f = prf(tp, fp, fn)
    return {"precision": p, "recall": r, "f1": f}[metric]


def _config_hash(cfg: Config) -> str:
    payload = {
        "treebank": cfg.treebank,
        "dataset": vars(cfg.dataset),
        "models": [vars(m) for m in cfg.models if m.enabled],
        "temperature": cfg.llm.temperature,
        "lemmas": cfg.llm.lemmas,
        "seed": cfg.seed,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _category(word: str, upos: str) -> str:
    if upos == "PROPN":
        return "proper_noun"
    if is_katakana(word):
        return "katakana"
    return "other"


def _sentence_stats(gold: dict[str, Any], pred: dict[str, Any]) -> dict[str, Any]:
    gold_spans = [tuple(s) for s in gold["spans"]]
    pred_spans = [tuple(s) for s in pred.get("spans", [])]
    b_tp, b_fp, b_fn = boundary_counts(gold_spans, pred_spans)
    w_tp, w_fp, w_fn = word_counts(gold_spans, pred_spans)

    gold_lemmas = gold.get("lemmas") or list(gold["words"])
    pred_lemmas = pred.get("lemmas")
    pred_lemma_by_span = (
        {tuple(sp): norm_lemma(l) for sp, l in zip(pred_spans, pred_lemmas)}
        if pred_lemmas is not None
        else None
    )

    pred_set = set(pred_spans)
    cat_hit: dict[str, int] = defaultdict(int)
    cat_tot: dict[str, int] = defaultdict(int)
    lemma_hit = 0
    lemma_pred_tot = 0
    for (s, e), word, upos, gl in zip(
        gold_spans, gold["words"], gold["upos"], gold_lemmas
    ):
        cat = _category(word, upos)
        cat_tot[cat] += 1
        span = (s, e)
        if span in pred_set:
            cat_hit[cat] += 1
            if pred_lemma_by_span is not None and span in pred_lemma_by_span:
                lemma_pred_tot += 1
                if pred_lemma_by_span[span] == norm_lemma(gl):
                    lemma_hit += 1

    return {
        "b": (b_tp, b_fp, b_fn),
        "w": (w_tp, w_fp, w_fn),
        "cat_hit": dict(cat_hit),
        "cat_tot": dict(cat_tot),
        "lemma_hit": lemma_hit,
        "lemma_pred_tot": lemma_pred_tot,
        "gold_tokens": len(gold_spans),
        "parse_ok": bool(pred.get("parse_ok", True)),
        "pos": pred.get("position_in_batch"),
    }


def _bootstrap(
    stats: list[dict[str, Any]],
    counts_fn: Callable[[dict[str, Any]], tuple[int, int, int]],
    metric: str,
    rng: np.random.Generator,
    n_boot: int,
) -> tuple[float, float, float]:
    arr = np.array([counts_fn(s) for s in stats], dtype=float)  # (n, 3)
    point = _metric_value(metric, *arr.sum(axis=0).astype(int))
    n = len(stats)
    if n == 0:
        return point, float("nan"), float("nan")
    vals = np.empty(n_boot)
    idx = rng.integers(0, n, size=(n_boot, n))
    for i in range(n_boot):
        summed = arr[idx[i]].sum(axis=0).astype(int)
        vals[i] = _metric_value(metric, *summed)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, float(lo), float(hi)


def _category_recall(
    stats: list[dict[str, Any]], category: str, rng: np.random.Generator, n_boot: int
) -> tuple[float, float, float]:
    values = np.array(
        [(s["cat_hit"].get(category, 0), s["cat_tot"].get(category, 0)) for s in stats],
        dtype=float,
    )
    tot = values[:, 1].sum()
    point = values[:, 0].sum() / tot if tot else 0.0
    n = len(stats)
    vals = np.empty(n_boot)
    idx = rng.integers(0, n, size=(n_boot, n))
    for i in range(n_boot):
        sample = values[idx[i]]
        t = sample[:, 1].sum()
        vals[i] = sample[:, 0].sum() / t if t else 0.0
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def _bootstrap_ratio(
    stats: list[dict[str, Any]],
    num_key: str,
    den_key: str,
    rng: np.random.Generator,
    n_boot: int,
) -> tuple[float, float, float]:
    arr = np.array([(s[num_key], s[den_key]) for s in stats], dtype=float)
    tot = arr[:, 1].sum()
    point = arr[:, 0].sum() / tot if tot else 0.0
    n = len(stats)
    if n == 0:
        return float(point), float("nan"), float("nan")
    vals = np.empty(n_boot)
    idx = rng.integers(0, n, size=(n_boot, n))
    for i in range(n_boot):
        sample = arr[idx[i]]
        t = sample[:, 1].sum()
        vals[i] = sample[:, 0].sum() / t if t else 0.0
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def _analyzers(cfg: Config) -> list[tuple[str, str, int]]:
    """Return (analyzer_name, tier, batch_size) from the predictions directory."""
    pred_dir = cfg.root / cfg.results_dir / "predictions" / cfg.treebank
    out: list[tuple[str, str, int]] = []
    tier_by_name = {m.name: m.tier for m in cfg.models}
    if not pred_dir.exists():
        return out
    for path in sorted(pred_dir.glob("*.jsonl")):
        records = read_jsonl(path)
        if not records:
            continue
        name = records[0].get("analyzer", path.stem)
        tier = records[0].get("tier") or tier_by_name.get(name, "baseline")
        batch = int(records[0].get("batch_size") or 1)
        out.append((name, tier, batch))
    return out


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _timing_records(cfg: Config, ts: str, cfg_hash: str) -> list[dict[str, Any]]:
    """Aggregate per-call throughput from the raw cache.

    The raw cache is the source of truth for timings (predictions only carry a
    copy). Records are de-duplicated by cache key, keeping the latest.
    """
    raw_dir = cfg.root / cfg.results_dir / "raw" / cfg.treebank
    if not raw_dir.exists():
        return []
    tier_by_name = {m.name: m.tier for m in cfg.models}
    out: list[dict[str, Any]] = []

    for path in sorted(raw_dir.glob("*.jsonl")):
        recs = read_jsonl(path)
        if not recs:
            continue
        by_key: dict[str, dict[str, Any]] = {}
        for i, r in enumerate(recs):
            by_key[r.get("key") or f"__{i}"] = r
        rows = list(by_key.values())
        model = rows[0].get("model", path.stem)
        analyzer = rows[0].get("analyzer", model)
        batch = int(rows[0].get("batch_size") or 1)
        tier = rows[0].get("tier") or tier_by_name.get(model, "baseline")
        prompt_sha = next(
            (r.get("prompt_sha256") for r in rows if r.get("prompt_sha256")), None
        )
        model_version = next(
            (r.get("model_version") for r in rows if r.get("model_version")), None
        )

        def add(metric: str, value: float, lo: float, hi: float, n: int) -> None:
            out.append(
                {
                    "model": analyzer,
                    "tier": tier,
                    "treebank": cfg.treebank,
                    "metric": metric,
                    "category": "all",
                    "value": value,
                    "ci_low": lo,
                    "ci_high": hi,
                    "n": n,
                    "n_boot": 0,
                    "seed": cfg.seed,
                    "batch_size": batch,
                    "lemma_mode": analyzer.endswith("@lem"),
                    "prompt_sha256": prompt_sha,
                    "model_version": model_version,
                    "config_hash": cfg_hash,
                    "timestamp": ts,
                }
            )

        def add_dist(metric: str, values: list[float]) -> None:
            if not values:
                return
            lo, hi = np.percentile(values, [10, 90])
            add(metric, float(np.median(values)), float(lo), float(hi), len(values))

        add_dist(
            "eval_tokens_per_s",
            [v for v in (_to_float(r.get("eval_tokens_per_s")) for r in rows) if v],
        )
        add_dist(
            "prompt_tokens_per_s",
            [v for v in (_to_float(r.get("prompt_tokens_per_s")) for r in rows) if v],
        )
        # Normalize per-call cost by batch size so batch arms are comparable.
        add_dist(
            "tokens_per_sentence",
            [ec / batch for ec in (_to_float(r.get("eval_count")) for r in rows) if ec],
        )
        add_dist(
            "wall_s_per_sentence",
            [w / batch for w in (_to_float(r.get("wall_s")) for r in rows) if w],
        )

        if any("content_empty" in r for r in rows):
            n_empty = sum(1 for r in rows if r.get("content_empty"))
            add("empty_content_rate", n_empty / len(rows), float("nan"), float("nan"), len(rows))

    return out


def run(cfg: Config, n_boot: int | None = None) -> list[dict[str, Any]]:
    gold = read_jsonl(cfg.gold_path())
    if not gold:
        raise SystemExit("No gold data. Run `tok-eval prepare` first.")
    n_boot = n_boot or cfg.bootstrap
    rng = np.random.default_rng(cfg.seed)
    cfg_hash = _config_hash(cfg)
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")

    records: list[dict[str, Any]] = []
    for analyzer, tier, batch in _analyzers(cfg):
        preds = {r["id"]: r for r in read_jsonl(cfg.predictions_path(analyzer))}
        stats = [
            _sentence_stats(g, preds[g["id"]]) for g in gold if g["id"] in preds
        ]
        if not stats:
            continue
        prompt_sha = next(
            (p.get("prompt_sha256") for p in preds.values() if p.get("prompt_sha256")),
            None,
        )
        model_version = next(
            (p.get("model_version") for p in preds.values() if p.get("model_version")),
            None,
        )

        def emit(
            metric: str,
            value: float,
            lo: float,
            hi: float,
            category: str = "all",
            n: int | None = None,
        ):
            records.append(
                {
                    "model": analyzer,
                    "tier": tier,
                    "treebank": cfg.treebank,
                    "metric": metric,
                    "category": category,
                    "value": value,
                    "ci_low": lo,
                    "ci_high": hi,
                    "n": n if n is not None else len(stats),
                    "n_boot": n_boot,
                    "seed": cfg.seed,
                    "batch_size": batch,
                    "lemma_mode": analyzer.endswith("@lem"),
                    "prompt_sha256": prompt_sha,
                    "model_version": model_version,
                    "config_hash": cfg_hash,
                    "timestamp": ts,
                }
            )

        headline: dict[str, float] = {}
        for family in FAMILIES:
            key = "b" if family == "boundary" else "w"
            for metric in METRICS:
                v, lo, hi = _bootstrap(
                    stats, lambda s, k=key: s[k], metric, rng, n_boot
                )
                emit(f"{family}_{metric}", v, lo, hi)
                headline[f"{family}_{metric}"] = v

        emit(
            "parse_rate",
            float(np.mean([s["parse_ok"] for s in stats])),
            float("nan"),
            float("nan"),
        )

        for category in ("proper_noun", "katakana", "other"):
            v, lo, hi = _category_recall(stats, category, rng, n_boot)
            emit("word_recall", v, lo, hi, category=category)

        if sum(s["lemma_pred_tot"] for s in stats) > 0:
            v, lo, hi = _bootstrap_ratio(stats, "lemma_hit", "lemma_pred_tot", rng, n_boot)
            emit("lemma_accuracy", v, lo, hi)
            headline["lemma_accuracy"] = v
            v2, lo2, hi2 = _bootstrap_ratio(
                stats, "lemma_pred_tot", "gold_tokens", rng, n_boot
            )
            emit("lemma_coverage", v2, lo2, hi2)

        # Batch-specific diagnostics: whole-batch parse success and position effect.
        batch_groups: dict[Any, list[bool]] = {}
        for p in preds.values():
            bi = p.get("batch_index")
            if bi is not None:
                batch_groups.setdefault(bi, []).append(bool(p.get("parse_ok", True)))
        if batch_groups:
            rate = sum(1 for v in batch_groups.values() if all(v)) / len(batch_groups)
            emit("batch_parse_rate", rate, float("nan"), float("nan"), n=len(batch_groups))

        positions = sorted({s["pos"] for s in stats if s["pos"] is not None})
        for pos in positions:
            sub = [s for s in stats if s["pos"] == pos]
            for fam, key in (("boundary", "b"), ("word", "w")):
                v, lo, hi = _bootstrap(sub, lambda s, k=key: s[k], "f1", rng, n_boot)
                emit(f"{fam}_f1", v, lo, hi, category=f"pos{pos}", n=len(sub))

        lemma_note = (
            f" lemma_acc={headline['lemma_accuracy']:.4f}"
            if "lemma_accuracy" in headline
            else ""
        )
        print(
            f"[score] {analyzer}: word_f1={headline['word_f1']:.4f} "
            f"boundary_f1={headline['boundary_f1']:.4f}{lemma_note} (n={len(stats)})"
        )

    scores_dir = cfg.root / cfg.results_dir / "scores"
    records.extend(_timing_records(cfg, ts, cfg_hash))
    # Keep rows from other treebanks so scoring one dataset does not discard another.
    existing = read_jsonl(scores_dir / "scores.jsonl")
    merged = [r for r in existing if r.get("treebank") != cfg.treebank] + records
    write_jsonl(scores_dir / "scores.jsonl", merged)
    _write_tables(merged, scores_dir)
    print(
        f"[score] wrote {len(records)} rows for {cfg.treebank}; "
        f"scores.jsonl now holds {len(merged)} -> {scores_dir.relative_to(cfg.root)}"
    )
    return merged


def _write_tables(records: list[dict[str, Any]], out_dir) -> None:
    try:
        import pandas as pd
    except ImportError:
        return
    if not records:
        return
    df = pd.DataFrame(records)
    df.to_csv(out_dir / "scores.csv", index=False)
    try:
        df.to_parquet(out_dir / "scores.parquet", index=False)
    except Exception:
        pass
