#!/usr/bin/env python3
"""Mini convergence-speed bar chart for extended abstract.

Same methodology and layout as plot_convergence_speed.py (grouped bars across
thresholds, one bar per algo per threshold, error bars across seeds), but
restricted to thresholds 0.75 / 0.5 / 0.25 and rendered mini-sized with
the legend below the figure.

Default output dir: <cache_dir>/../convergence_speed/ (alongside the big plot).

Usage:
    python scripts/plot_mini_convergence_speed.py \\
        --cache_dir results/convergence_same_net/cache \\
        --runs <12 source dirs in order> \\
        --group TQC TQC TQC SAC SAC SAC CROSSQ CROSSQ CROSSQ TQC-CROSSQ TQC-CROSSQ TQC-CROSSQ
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GROUP_COLORS = {
    "TQC":        "tab:blue",
    "SAC":        "tab:orange",
    "CROSSQ":     "tab:green",
    "TQC-CROSSQ": "tab:red",
}

DEFAULT_THRESHOLDS = [0.75, 0.5, 0.25]


def _resolve_run_id(result_dir):
    cfg_path = os.path.join(result_dir, "run_config.json")
    if not os.path.exists(cfg_path):
        sys.exit(f"ERROR: no run_config.json in {result_dir}")
    with open(cfg_path) as f:
        cfg = json.load(f)
    rid = cfg.get("wandb_run_id") or cfg.get("wandb_run_name")
    if not rid:
        sys.exit(f"ERROR: no wandb_run_id in {cfg_path}")
    return rid


def _load_cache(cache_dir, run_id):
    path = os.path.join(cache_dir, f"{run_id.replace('/', '_')}_convergence.csv")
    if not os.path.exists(path):
        print(f"  WARNING: cache missing for {run_id}: {path}")
        return None
    return pd.read_csv(path)


def _steps_to_threshold(df, col, threshold):
    """Same definition as plot_convergence_speed.py: first step where col < threshold."""
    if col not in df.columns:
        return float("nan")
    hit = df[df[col] < threshold]
    if hit.empty:
        return float("nan")
    return float(hit["step"].iloc[0])


def main():
    parser = argparse.ArgumentParser(description="Mini convergence-speed bar chart")
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--group", nargs="+", required=True)
    parser.add_argument("--metric", choices=["cost", "discomfort"], default="discomfort",
                        help="Which EMA ratio to threshold (default: discomfort, matches the big plot)")
    parser.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--output_dir", default=None,
                        help="Default: <cache_dir>/../convergence_speed (alongside the big plot)")
    parser.add_argument("--format", choices=["pdf", "png", "svg"], default="pdf")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--figwidth", type=float, default=None,
                        help="Figure width in inches. Default: auto-scaled to content "
                             "(inches_per_bar * n_groups * n_thresh + side_margin).")
    parser.add_argument("--figheight", type=float, default=2.2)
    parser.add_argument("--bar_width", type=float, default=0.18,
                        help="Bar slot width in x-axis units. Controls cluster span and the "
                             "inter-cluster gap (1 - n_groups*bar_width). Bar positions are fixed by this.")
    parser.add_argument("--bar_fill", type=float, default=0.85,
                        help="Fraction of its slot the visible bar fills (default 0.85). "
                             "Lower = more white space between bars within a cluster.")
    parser.add_argument("--inches_per_bar", type=float, default=0.30,
                        help="When --figwidth is auto, allot this many inches per bar.")
    parser.add_argument("--ylim", type=float, nargs=2, default=None,
                        help="Manual y-axis range in training steps")
    args = parser.parse_args()

    if len(args.runs) != len(args.group):
        sys.exit(f"ERROR: --runs has {len(args.runs)} entries, --group has {len(args.group)}")

    output_dir = args.output_dir or os.path.normpath(
        os.path.join(args.cache_dir, "..", "convergence_speed"))
    os.makedirs(output_dir, exist_ok=True)

    metric_col = "ema_cost_ratio" if args.metric == "cost" else "ema_dis_ratio"
    thresholds = sorted(args.thresholds, reverse=True)

    # Load caches grouped by algo
    by_group = {}
    for rdir, g in zip(args.runs, args.group):
        rid = _resolve_run_id(rdir)
        df = _load_cache(args.cache_dir, rid)
        if df is None:
            continue
        by_group.setdefault(g, []).append(df)
        print(f"  {g:12s}  {rid}  ({len(df)} steps)")

    if not by_group:
        sys.exit("ERROR: no cache CSVs found")

    group_order = []
    for g in args.group:
        if g not in group_order and g in by_group:
            group_order.append(g)

    # data[group][threshold] = list of per-seed steps (NaN if never reached)
    data = {g: {t: [_steps_to_threshold(df, metric_col, t) for df in by_group[g]]
                for t in thresholds}
            for g in group_order}

    n_thresh = len(thresholds)
    n_groups = len(group_order)
    bar_width = args.bar_width
    x = np.arange(n_thresh)

    # Auto-size width to content unless overridden
    figwidth = args.figwidth
    if figwidth is None:
        # Side margin covers y-axis label + ticks + right pad
        figwidth = args.inches_per_bar * n_groups * n_thresh + 1.4

    fig, ax = plt.subplots(figsize=(figwidth, args.figheight))

    # Tight top — fit to tallest bar
    bar_tops = [np.mean([v for v in data[g][t] if not np.isnan(v)])
                for g in group_order for t in thresholds
                if any(not np.isnan(v) for v in data[g][t])]
    y_max = max(bar_tops) * 1.03 if bar_tops else 1.0

    drawn_width = bar_width * args.bar_fill
    for gi, g in enumerate(group_order):
        offset = (gi - n_groups / 2 + 0.5) * bar_width
        for ti, t in enumerate(thresholds):
            xi = x[ti] + offset
            seeds = data[g][t]
            valid = [v for v in seeds if not np.isnan(v)]
            if not valid:
                ax.bar(xi, y_max, width=drawn_width,
                       color=GROUP_COLORS.get(g, "lightgray"), alpha=0.25,
                       hatch="///", edgecolor=GROUP_COLORS.get(g, "lightgray"),
                       linewidth=0.5)
            else:
                mean = float(np.mean(valid))
                std = float(np.std(valid)) if len(valid) > 1 else 0.0
                ax.bar(xi, mean, width=drawn_width,
                       color=GROUP_COLORS.get(g, "lightgray"), alpha=0.8)

    if args.ylim is not None:
        ax.set_ylim(args.ylim[0], args.ylim[1])
    else:
        ax.set_ylim(0, y_max)
    ax.set_xticks(x)
    ax.set_xticklabels([f"< {t:g}" for t in thresholds], fontsize=8)
    metric_word = "cost" if args.metric == "cost" else "discomfort"
    ax.set_xlabel(f"{metric_word.capitalize()} ratio reached", fontsize=8)
    ax.set_ylabel("Training steps", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", alpha=0.25)

    # Steps in k / M
    ax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda y, _: f"{int(y/1000)}k" if y < 1e6 else f"{y/1e6:.1f}M"))

    # Legend below
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=GROUP_COLORS.get(g, "lightgray"), alpha=0.8, label=g)
               for g in group_order]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.05),
               ncol=len(group_order), frameon=False, fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(output_dir,
                             f"mini_convergence_speed_{args.metric}.{args.format}")
    plt.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
