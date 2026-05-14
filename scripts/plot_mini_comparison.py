#!/usr/bin/env python3
"""Mini model-comparison plot for extended abstract.

Thin boxplots: one box per algo group, mean cost ratio across pooled episodes.
Legend below the figure.

Usage:
    python scripts/plot_mini_comparison.py \\
        --runs /home/warre/Documents/THESIS/results/long_tqc_seed42_... ... \\
        --group TQC TQC TQC SAC SAC SAC CROSSQ CROSSQ CROSSQ TQC-CROSSQ TQC-CROSSQ TQC-CROSSQ \\
        --output_dir results/model_comparison_same_net_1p5M/mini
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


def _load_episode_metrics(result_dir, period, subdir):
    for folder in [subdir, "compare"]:
        path = os.path.join(result_dir, folder, period, f"{period}_episode_metrics.csv")
        if os.path.exists(path):
            return pd.read_csv(path)
    return None


def _add_ratios(df):
    df = df.copy()
    if "baseline_energy_cost_usd" in df.columns:
        df["cost_ratio"] = df["rl_energy_cost_usd"] / df["baseline_energy_cost_usd"].replace(0, np.nan)
    if "baseline_discomfort_deg_h" in df.columns:
        df["dis_ratio"] = df["rl_discomfort_deg_h"] / df["baseline_discomfort_deg_h"].replace(0, np.nan)
    return df


def main():
    parser = argparse.ArgumentParser(description="Mini model-comparison plot")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--group", nargs="+", required=True)
    parser.add_argument("--period", choices=["val", "test"], default="test")
    parser.add_argument("--compare_subdir", default="compare_reeval")
    parser.add_argument("--metric", choices=["cost", "discomfort", "both"], default="both")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--format", choices=["pdf", "png", "svg"], default="pdf")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--figwidth", type=float, default=6.0)
    parser.add_argument("--figheight", type=float, default=2.0)
    parser.add_argument("--ylim", type=float, nargs=2, default=None,
                        help="Manual y-axis range (applies to all panels). "
                             "Default: auto-fit to box whiskers with small pad.")
    args = parser.parse_args()

    if len(args.runs) != len(args.group):
        sys.exit(f"ERROR: --runs has {len(args.runs)} entries, --group has {len(args.group)}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load per-episode data, pooled per group
    by_group = {}
    for rdir, g in zip(args.runs, args.group):
        df = _load_episode_metrics(rdir, args.period, args.compare_subdir)
        if df is None:
            print(f"  WARNING: skipping {rdir} — no {args.period}_episode_metrics.csv")
            continue
        df = _add_ratios(df)
        by_group.setdefault(g, []).append(df)
        print(f"  {g:12s}  {os.path.basename(rdir)}  ({len(df)} episodes)")

    if not by_group:
        sys.exit("ERROR: no per-episode data loaded")

    group_order = []
    for g in args.group:
        if g not in group_order and g in by_group:
            group_order.append(g)

    pooled = {g: pd.concat(by_group[g], ignore_index=True) for g in group_order}

    metrics_to_plot = []
    if args.metric in ("cost", "both"):
        metrics_to_plot.append(("cost_ratio",  "Cost ratio"))
    if args.metric in ("discomfort", "both"):
        metrics_to_plot.append(("dis_ratio",   "Discomfort ratio"))

    n = len(metrics_to_plot)
    fig, axes = plt.subplots(1, n, figsize=(args.figwidth, args.figheight))
    if n == 1:
        axes = [axes]

    for ax, (col, ylabel) in zip(axes, metrics_to_plot):
        data = [pooled[g][col].dropna().to_numpy() for g in group_order]
        bp = ax.boxplot(
            data,
            tick_labels=group_order,
            patch_artist=True,
            widths=0.35,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.2},
            boxprops={"linewidth": 0.8},
            whiskerprops={"linewidth": 0.8},
            capprops={"linewidth": 0.8},
        )
        for box, g in zip(bp["boxes"], group_order):
            box.set_facecolor(GROUP_COLORS.get(g, "lightgray"))
            box.set_alpha(0.7)

        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(axis="x", labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", alpha=0.25)
        # Hide x tick labels — legend below carries the info
        ax.set_xticklabels([])

        # Tight y-limits fit to the data only (no forced y=1 baseline line)
        if args.ylim is not None:
            ax.set_ylim(args.ylim[0], args.ylim[1])
        else:
            all_vals = np.concatenate([d for d in data if len(d) > 0])
            lo = float(np.percentile(all_vals, 2))
            hi = float(np.percentile(all_vals, 98))
            pad = (hi - lo) * 0.08 or 0.02
            ax.set_ylim(lo - pad, hi + pad)

    # Legend below the figure — proxy patches
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=GROUP_COLORS.get(g, "lightgray"), alpha=0.7, label=g)
               for g in group_order]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.05),
               ncol=len(group_order), frameon=False, fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(args.output_dir, f"mini_comparison_{args.period}.{args.format}")
    plt.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
