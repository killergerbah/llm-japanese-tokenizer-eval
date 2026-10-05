"""Render score figures (SVG + PNG) with matplotlib/seaborn."""

from __future__ import annotations

from pathlib import Path

from .config import Config
from .io import read_jsonl

_TIER_COLORS = {"local": "#4c72b0", "remote": "#dd8452", "baseline": "#55a868"}


def _load_df(cfg: Config):
    import pandas as pd

    scores = read_jsonl(cfg.root / cfg.results_dir / "scores" / "scores.jsonl")
    if not scores:
        return None
    return pd.DataFrame(scores)


def _bar_metric(
    df, metric: str, category: str, out: Path, title: str, ylim=(0, 1)
) -> bool:
    sub = df[(df["metric"] == metric) & (df["category"] == category)].copy()
    if sub.empty:
        return False
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    sub = sub.sort_values("value", ascending=False).reset_index(drop=True)
    x = np.arange(len(sub))
    colors = [_TIER_COLORS.get(t, "#999999") for t in sub["tier"]]
    yerr = None
    if sub["ci_low"].notna().all():
        yerr = np.vstack(
            [(sub["value"] - sub["ci_low"]).to_numpy(), (sub["ci_high"] - sub["value"]).to_numpy()]
        )

    fig, ax = plt.subplots(figsize=(max(6, len(sub) * 1.1), 4.5))
    ax.bar(x, sub["value"], color=colors, yerr=yerr, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(sub["model"], rotation=30, ha="right")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".svg"))
    fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)
    return True


def _grouped_recall(df, out: Path) -> bool:
    sub = df[(df["metric"] == "word_recall") & (df["category"] != "all")].copy()
    if sub.empty:
        return False
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    models = sorted(sub["model"].unique())
    categories = ["proper_noun", "katakana", "other"]
    width = 0.8 / len(categories)
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.3), 4.5))
    for i, cat in enumerate(categories):
        vals = []
        for m in models:
            row = sub[(sub["model"] == m) & (sub["category"] == cat)]
            vals.append(float(row["value"].iloc[0]) if not row.empty else 0.0)
        ax.bar(x + i * width, vals, width, label=cat)
    ax.set_xticks(x + width * (len(categories) - 1) / 2)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("word_recall")
    ax.set_title("Word recall by gold-word category")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".svg"))
    fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)
    return True


def _batch_lines(df, metric: str, out: Path, title: str) -> bool:
    if "batch_size" not in df.columns:
        return False
    sub = df[
        (df["metric"] == metric)
        & (df["category"] == "all")
        & (df["model"].astype(str).str.contains("@b"))
    ].copy()
    if sub.empty:
        return False
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub["base"] = sub["model"].astype(str).str.split("@b").str[0]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for base, g in sub.groupby("base"):
        g = g.sort_values("batch_size")
        ax.plot(g["batch_size"], g["value"], marker="o", label=base)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(sub["batch_size"].unique()))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("sentences per prompt (batch size)")
    ax.set_ylim(0, 1)
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".svg"))
    fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)
    return True


def _position_lines(df, metric: str, out: Path, title: str) -> bool:
    if "category" not in df.columns:
        return False
    sub = df[(df["metric"] == metric) & (df["category"].astype(str).str.startswith("pos"))].copy()
    if sub.empty:
        return False
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub["pos"] = sub["category"].astype(str).str.replace("pos", "", regex=False).astype(int)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, g in sub.groupby("model"):
        g = g.sort_values("pos")
        ax.plot(g["pos"], g["value"], marker="o", label=name)
    ax.set_xlabel("position within batch")
    ax.set_ylim(0, 1)
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".svg"))
    fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)
    return True


def _cross_treebank(df, metric: str, out: Path, title: str) -> bool:
    if "treebank" not in df.columns:
        return False
    sub = df[(df["metric"] == metric) & (df["category"] == "all")].copy()
    sub = sub.drop_duplicates(subset=["treebank", "model"])
    if sub.empty or sub["treebank"].nunique() < 2:
        return False
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    treebanks = sorted(sub["treebank"].unique())
    models = sorted(sub["model"].unique())
    width = 0.8 / len(treebanks)
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.3), 4.5))
    for i, tb in enumerate(treebanks):
        vals = []
        for m in models:
            row = sub[(sub["treebank"] == tb) & (sub["model"] == m)]
            vals.append(float(row["value"].iloc[0]) if not row.empty else 0.0)
        ax.bar(x + i * width, vals, width, label=tb)
    ax.set_xticks(x + width * (len(treebanks) - 1) / 2)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="treebank")
    fig.tight_layout()
    fig.savefig(out.with_suffix(".svg"))
    fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)
    return True


def run(cfg: Config, all_treebanks: bool = False) -> None:
    df = _load_df(cfg)
    if df is None:
        raise SystemExit("No scores. Run `tok-eval score` first.")
    fig_dir = cfg.root / cfg.results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    cur = df[df["treebank"] == cfg.treebank] if "treebank" in df.columns else df
    if cur.empty:
        print(f"[plot] no scores for treebank {cfg.treebank}; using all rows")
        cur = df

    made = []
    if _bar_metric(cur, "word_f1", "all", fig_dir / "word_f1", "Segmenter exact-word F1"):
        made.append("word_f1")
    if _bar_metric(
        cur, "boundary_f1", "all", fig_dir / "boundary_f1", "Segmenter boundary F1"
    ):
        made.append("boundary_f1")
    if _bar_metric(
        cur, "lemma_accuracy", "all", fig_dir / "lemma_accuracy", "Lemma accuracy (aligned tokens)"
    ):
        made.append("lemma_accuracy")
    if _bar_metric(
        cur,
        "eval_tokens_per_s",
        "all",
        fig_dir / "eval_tokens_per_s",
        "Generation throughput (eval tok/s, median with p10-p90)",
        ylim=None,
    ):
        made.append("eval_tokens_per_s")
    if _bar_metric(
        cur, "empty_content_rate", "all", fig_dir / "empty_content_rate", "Empty-content rate"
    ):
        made.append("empty_content_rate")
    if _grouped_recall(cur, fig_dir / "word_recall_by_category"):
        made.append("word_recall_by_category")
    if _batch_lines(cur, "word_f1", fig_dir / "batch_word_f1", "Exact-word F1 vs batch size"):
        made.append("batch_word_f1")
    if _batch_lines(
        cur, "boundary_f1", fig_dir / "batch_boundary_f1", "Boundary F1 vs batch size"
    ):
        made.append("batch_boundary_f1")
    if _batch_lines(
        cur,
        "tokens_per_sentence",
        fig_dir / "batch_tokens_per_sentence",
        "Generated tokens per sentence vs batch size",
    ):
        made.append("batch_tokens_per_sentence")
    if _position_lines(
        cur, "boundary_f1", fig_dir / "position_boundary_f1", "Boundary F1 by position in batch"
    ):
        made.append("position_boundary_f1")
    if all_treebanks:
        if _cross_treebank(df, "word_f1", fig_dir / "cross_word_f1", "Exact-word F1 by treebank"):
            made.append("cross_word_f1")
        if _cross_treebank(
            df, "boundary_f1", fig_dir / "cross_boundary_f1", "Boundary F1 by treebank"
        ):
            made.append("cross_boundary_f1")

    print(f"[plot] wrote {len(made)} figure(s) -> {fig_dir.relative_to(cfg.root)}: {', '.join(made)}")
