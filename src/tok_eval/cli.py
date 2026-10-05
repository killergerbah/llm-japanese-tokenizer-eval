"""Command-line interface: ``tok-eval <prepare|baseline|llm|score|plot|compare|all>``."""

from __future__ import annotations

import argparse

from . import compare, gold, llm, plot, score
from . import mecab as baseline
from .config import load_config


def _split(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tok-eval", description=__doc__)
    p.add_argument("--config", default="config.yaml", help="path to config.yaml")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("prepare", help="build gold.jsonl")
    sp.add_argument("--limit", type=int, default=None)

    sb = sub.add_parser("baseline", help="run MeCab/Sudachi baselines")
    sb.add_argument("--analyzers", default="mecab", help="comma list, e.g. mecab,sudachi")

    sl = sub.add_parser("llm", help="run LLM segmenters")
    sl.add_argument("--models", default=None, help="comma list of model names")
    sl.add_argument("--offline", action="store_true", help="use cache only")
    sl.add_argument("--limit", type=int, default=None)
    sl.add_argument("--lemmas", action="store_true", help="also predict lemmas")
    sl.add_argument(
        "--reasoning-effort",
        default=None,
        choices=["none", "low", "medium", "high"],
        help="override llm.reasoning_effort (models with their own setting win)",
    )
    sl.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="sentences per prompt (batch ablation); omit for the single-sentence reference",
    )

    ss = sub.add_parser("score", help="score predictions against gold")
    ss.add_argument("--n-boot", type=int, default=None)

    splot = sub.add_parser("plot", help="render figures from scores")
    splot.add_argument(
        "--all-treebanks",
        action="store_true",
        help="also emit cross-treebank comparison figures",
    )

    sc = sub.add_parser("compare", help="paired comparison between analyzer variants")
    sc.add_argument("--n-boot", type=int, default=None)

    sa = sub.add_parser("all", help="prepare -> baseline -> llm -> score -> plot")
    sa.add_argument("--limit", type=int, default=None)
    sa.add_argument("--analyzers", default="mecab")
    sa.add_argument("--models", default=None)
    sa.add_argument("--offline", action="store_true")
    sa.add_argument("--skip-llm", action="store_true")
    sa.add_argument("--lemmas", action="store_true", help="also predict lemmas")
    sa.add_argument(
        "--reasoning-effort",
        default=None,
        choices=["none", "low", "medium", "high"],
        help="override llm.reasoning_effort (models with their own setting win)",
    )
    sa.add_argument("--batch-size", type=int, default=None)

    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    if getattr(args, "lemmas", False):
        cfg.llm.lemmas = True
    if getattr(args, "reasoning_effort", None):
        cfg.llm.reasoning_effort = args.reasoning_effort

    if args.command == "prepare":
        gold.run(cfg, limit=args.limit)
    elif args.command == "baseline":
        baseline.run(cfg, analyzers=_split(args.analyzers))
    elif args.command == "llm":
        llm.run(
            cfg,
            models=_split(args.models),
            offline=args.offline,
            limit=args.limit,
            batch_size=args.batch_size,
        )
    elif args.command == "score":
        score.run(cfg, n_boot=args.n_boot)
    elif args.command == "plot":
        plot.run(cfg, all_treebanks=args.all_treebanks)
    elif args.command == "compare":
        compare.run(cfg, n_boot=args.n_boot)
    elif args.command == "all":
        gold.run(cfg, limit=args.limit)
        baseline.run(cfg, analyzers=_split(args.analyzers))
        if not args.skip_llm:
            llm.run(
                cfg,
                models=_split(args.models),
                offline=args.offline,
                limit=args.limit,
                batch_size=args.batch_size,
            )
        score.run(cfg)
        plot.run(cfg)


if __name__ == "__main__":
    main()
