#!/usr/bin/env python3
"""Mini convergence plot for extended abstract.

Reads cached convergence CSVs (produced by plot_convergence.py) and the
matching source result_dirs (for the run_id -> group mapping via run_config.json),
then draws ONE small panel: cost ratio EMA vs steps, group mean only,
legend below the figure.

Usage:
    python scripts/plot_mini_convergence.py \\
        --cache_dir results/convergence_same_net/cache \\
        --runs /home/warre/Documents/THESIS/results/long_tqc_seed42_... \\
               /home/warre/Documents/THESIS/results/long_tqc_seed123_... \\
               ... (12 dirs in order) ... \\
        --group TQC TQC TQC SAC SAC SAC CROSSQ CROSSQ CROSSQ TQC-CROSSQ TQC-CROSSQ TQC-CROSSQ \\
        --output_dir results/convergence_same_net/mini
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


def main():
    parser = argparse.ArgumentParser(description="Mini convergence plot for extended abstract")
    parser.add_argument("--cache_dir", required=True,
                        help="Directory with <run_id>_convergence.csv files")
    parser.add_argument("--runs", nargs="+", required=True,
                        help="Source result dirs (must contain run_config.json), one per cache CSV")
    parser.add_argument("--group", nargs="+", required=True,
                        help="Group name per dir (e.g. TQC TQC TQC SAC ...)")
    parser.add_argument("--metric", choices=["cost", "discomfort"], default="cost",
                        help="Which ratio to plot (default: cost)")
    parser.add_argument("--output_dir", default=None,
                        help="Default: <cache_dir>/.. /mini")
    parser.add_argument("--x_min", type=int, default=None,
                        help="Minimum training step on x-axis")
    parser.add_argument("--format", choices=["pdf", "png", "svg"], default="pdf")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--figwidth", type=float, default=6.0,
                        help="Figure width in inches (default 6.0, mini-sized)")
    parser.add_argument("--figheight", type=float, default=3.2,
                        help="Figure height in inches (default 3.2)")
    args = parser.parse_args()

    if len(args.runs) != len(args.group):
        sys.exit(f"ERROR: --runs has {len(args.runs)} entries, --group has {len(args.group)}")

    output_dir = args.output_dir or os.path.join(os.path.dirname(args.cache_dir.rstrip("/")), "mini")
    os.makedirs(output_dir, exist_ok=True)

    # Build {group: [df, df, ...]}
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

    # Preserve group order from CLI
    group_order = []
    for g in args.group:
        if g not in group_order and g in by_group:
            group_order.append(g)

    metric_col = "ema_cost_ratio" if args.metric == "cost" else "ema_dis_ratio"
    ylabel = ("Energy cost ratio (RL / Baseline)" if args.metric == "cost"
              else "Discomfort ratio (RL / Baseline)")

    fig, ax = plt.subplots(figsize=(args.figwidth, args.figheight))

    for g in group_order:
        members = by_group[g]
        all_steps = sorted(set().union(*[set(m["step"]) for m in members]))
        if args.x_min is not None:
            all_steps = [s for s in all_steps if s >= args.x_min]
        # Step-wise asof interpolation, then mean across seeds
        vals = []
        for m in members:
            s = m.set_index("step")[metric_col]
            vals.append([s.asof(st) if st >= s.index.min() else np.nan for st in all_steps])
        arr = np.array(vals, dtype=float)
        mean = np.nanmean(arr, axis=0)
        color = GROUP_COLORS.get(g, None)
        ax.plot(all_steps, mean, color=color, linewidth=1.6, label=g)

    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
    if args.x_min is not None:
        ax.set_xlim(left=args.x_min)
    if args.metric == "discomfort":
        ax.set_yscale("symlog", linthresh=1.0, linscale=1.0)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0])
        ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1", "2", "5", "10"])

    ax.set_xlabel("Training steps")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)

    # Steps on x-axis as e.g. "500k"
    ax.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, _: f"{int(x/1000)}k" if x < 1e6 else f"{x/1e6:.1f}M"))

    # Legend below the figure
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=len(group_order), frameon=False, fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"mini_convergence_{args.metric}.{args.format}")
    plt.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
