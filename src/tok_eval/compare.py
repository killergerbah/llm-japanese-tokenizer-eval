"""Paired comparison between analyzer variants on the same gold sentences.

Compares each batch arm against the N=1 batch arm and against the
single-sentence reference, using paired per-sentence deltas (bootstrap CI) and
a McNemar test on exact-segmentation outcomes. Written to
``results/comparisons/<treebank>/``.
"""

from __future__ import annotations

import re
import time
from math import comb
from typing import Any

import numpy as np

from .config import Config
from .io import read_jsonl, write_jsonl
from .text import boundary_counts, prf, word_counts


def _parse(name: str) -> tuple[str, int, bool, bool]:
    """Split an analyzer id into (base, batch, lemma, has_batch_suffix)."""
    lemma = name.endswith("@lem")
    core = name[:-4] if lemma else name
    m = re.search(r"@b(\d+)$", core)
    if m:
        return core[: m.start()], int(m.group(1)), lemma, True
    return core, 1, lemma, False


def _batch_of(name: str) -> int:
    return _parse(name)[1]


def _load(cfg: Config):
    gold = {g["id"]: g for g in read_jsonl(cfg.gold_path())}
    pred_dir = cfg.root / cfg.results_dir / "predictions" / cfg.treebank
    preds: dict[str, dict[str, dict[str, Any]]] = {}
    if pred_dir.exists():
        for path in sorted(pred_dir.glob("*.jsonl")):
            recs = read_jsonl(path)
            if not recs:
                continue
            name = recs[0].get("analyzer", path.stem)
            preds[name] = {r["id"]: r for r in recs}
    return gold, preds


def _per_sentence(gold: dict[str, Any], pred: dict[str, Any]) -> dict[str, Any]:
    gs = [tuple(x) for x in gold["spans"]]
    ps = [tuple(x) for x in pred.get("spans", [])]
    _, _, bf = prf(*boundary_counts(gs, ps))
    _, _, wf = prf(*word_counts(gs, ps))
    return {"bf": bf, "wf": wf, "exact": wf == 1.0}


def _pairs(analyzers: list[str]) -> list[tuple[str, str]]:
    # Group by (base model, lemma mode) so lemma and token-only arms never mix.
    groups: dict[tuple[str, bool], list[tuple[str, bool]]] = {}
    for name in analyzers:
        base, _, lemma, suffixed = _parse(name)
        groups.setdefault((base, lemma), []).append((name, suffixed))
    pairs: list[tuple[str, str]] = []
    for (base, lemma), members in groups.items():
        names = [m[0] for m in members]
        ref = next((n for n, suffixed in members if not suffixed), None)
        b1 = f"{base}@b1" + ("@lem" if lemma else "")
        has_b1 = b1 in names
        for name, suffixed in members:
            if not suffixed:  # reference (no @b) has nothing to compare against
                continue
            if name == b1:
                if ref:
                    pairs.append((b1, ref))  # batch template N=1 vs reference
                continue
            if has_b1:
                pairs.append((name, b1))  # batch arm vs N=1 batch arm
            if ref:
                pairs.append((name, ref))  # arm vs single-sentence reference
    return pairs


def _delta_ci(
    a: list[float], b: list[float], rng: np.random.Generator, n_boot: int
) -> tuple[float, float, float]:
    d = np.array(a) - np.array(b)
    n = len(d)
    point = float(d.mean())
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def _mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, p)


def run(cfg: Config, n_boot: int | None = None) -> list[dict[str, Any]]:
    n_boot = n_boot or cfg.bootstrap
    rng = np.random.default_rng(cfg.seed)
    gold, preds_by_analyzer = _load(cfg)
    if not gold or len(preds_by_analyzer) < 2:
        print("[compare] need gold and at least two analyzers")
        return []

    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    out: list[dict[str, Any]] = []
    for arm_a, arm_b in _pairs(sorted(preds_by_analyzer)):
        pa, pb = preds_by_analyzer[arm_a], preds_by_analyzer[arm_b]
        ids = [i for i in gold if i in pa and i in pb]
        if not ids:
            continue
        bf_a, bf_b, wf_a, wf_b = [], [], [], []
        b_disc = c_disc = 0
        for i in ids:
            sa = _per_sentence(gold[i], pa[i])
            sb = _per_sentence(gold[i], pb[i])
            bf_a.append(sa["bf"])
            bf_b.append(sb["bf"])
            wf_a.append(sa["wf"])
            wf_b.append(sb["wf"])
            if sa["exact"] and not sb["exact"]:
                b_disc += 1
            elif sb["exact"] and not sa["exact"]:
                c_disc += 1
        p = _mcnemar(b_disc, c_disc)
        for metric, a, b in (
            ("word_f1_delta", wf_a, wf_b),
            ("boundary_f1_delta", bf_a, bf_b),
        ):
            v, lo, hi = _delta_ci(a, b, rng, n_boot)
            out.append(
                {
                    "model": _parse(arm_a)[0],
                    "arm_a": arm_a,
                    "arm_b": arm_b,
                    "batch_size_a": _batch_of(arm_a),
                    "batch_size_b": _batch_of(arm_b),
                    "metric": metric,
                    "value": v,
                    "ci_low": lo,
                    "ci_high": hi,
                    "p_value": p,
                    "n_pairs": len(ids),
                    "timestamp": ts,
                }
            )
        print(
            f"[compare] {arm_a} vs {arm_b}: "
            f"word_f1_delta={out[-2]['value']:+.4f} "
            f"boundary_f1_delta={out[-1]['value']:+.4f} (p={p:.3g}, n={len(ids)})"
        )

    comp_dir = cfg.root / cfg.results_dir / "comparisons" / cfg.treebank
    write_jsonl(comp_dir / "comparisons.jsonl", out)
    try:
        import pandas as pd

        pd.DataFrame(out).to_csv(comp_dir / "comparisons.csv", index=False)
    except Exception:
        pass
    print(f"[compare] wrote {len(out)} rows -> {comp_dir.relative_to(cfg.root)}")
    return out
