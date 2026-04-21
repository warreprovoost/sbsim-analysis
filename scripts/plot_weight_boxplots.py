#!/usr/bin/env python3
"""Plot energy cost and discomfort boxplots grouped by energy weight.

Usage:
    python scripts/plot_weight_boxplots.py \\
        results/long_tqc_seed42_ew10_... \\
        results/long_tqc_seed42_ew15_... \\
        results/long_tqc_seed42_ew20_... \\
        results/long_tqc_seed42_ew30_... \\
        results/long_tqc_seed42_ew50_... \\
        --period test
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PASS_COLOR  = "#4c9be8"
BL_COLOR    = "#f4a261"


def _load_run(result_dir, period, compare_subdir):
    config_path = os.path.join(result_dir, "run_config.json")
    if not os.path.exists(config_path):
        print(f"  WARNING: skipping {result_dir} — no run_config.json")
        return None
    with open(config_path) as f:
        config = json.load(f)

    for folder in [compare_subdir, "compare"]:
        path = os.path.join(result_dir, folder, period, f"{period}_episode_metrics.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            weight = float(config.get("energy_weight", 1.0))
            return {"dir": result_dir, "config": config, "df": df, "weight": weight}

    print(f"  WARNING: skipping {result_dir} — no {period}_episode_metrics.csv")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dirs", nargs="+")
    parser.add_argument("--period", default="test", choices=["val", "test"])
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--compare_subdir", default="compare_reeval")
    args = parser.parse_args()

    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(args.result_dirs[0]), f"weight_boxplots_{timestamp}"
    )
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "command.txt"), "w") as f:
        f.write(" ".join(sys.argv) + "\n")

    # Load all runs
    runs = []
    for rdir in args.result_dirs:
        r = _load_run(rdir, args.period, args.compare_subdir)
        if r is not None:
            runs.append(r)

    if not runs:
        print("ERROR: no runs loaded")
        sys.exit(1)

    # Group by weight
    groups = {}
    for r in runs:
        groups.setdefault(r["weight"], []).append(r)
    weights = sorted(groups.keys())
    print(f"Weights found: {weights}")

    # Pool episodes per weight
    weight_data = {}
    for w in weights:
        pooled = pd.concat([r["df"] for r in groups[w]], ignore_index=True)
        weight_data[w] = {
            "rl_cost":    pooled["rl_energy_cost_usd"].dropna().to_numpy(),
            "bl_cost":    pooled["baseline_energy_cost_usd"].dropna().to_numpy(),
            "rl_dis":     pooled["rl_discomfort_deg_h"].dropna().to_numpy(),
            "bl_dis":     pooled["baseline_discomfort_deg_h"].dropna().to_numpy(),
            "n_seeds":    len(groups[w]),
            "n_episodes": len(pooled),
        }

    x = np.arange(len(weights))
    labels = [str(w) for w in weights]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(8, len(weights) * 1.5), 10), sharex=False)
    fig.suptitle(f"Effect of Energy Weight on Performance",
                 fontsize=13, fontweight="bold")

    for ax, rl_key, ylabel, fmt in [
        (ax1, "rl_cost", "Energy cost (USD / episode)",    ".3f"),
        (ax2, "rl_dis",  "Discomfort (°C·h / episode)",   ".2f"),
    ]:
        rl_data = [weight_data[w][rl_key] for w in weights]

        bp = ax.boxplot(rl_data, positions=x, widths=0.5,
                        patch_artist=True, manage_ticks=False)
        for box in bp["boxes"]:
            box.set_facecolor(PASS_COLOR); box.set_alpha(0.8)

        # Annotate median value to the right of the median line
        for i, (vals, median_line) in enumerate(zip(rl_data, bp["medians"])):
            median = float(np.median(vals))
            mx = median_line.get_xdata()[1]  # right end of median line
            ax.annotate(f"{median:{fmt}}", (mx, median),
                        textcoords="offset points", xytext=(5, 0),
                        va="center", fontsize=8, color="black")

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Energy weight")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    path = os.path.join(output_dir, f"{args.period}_weight_boxplots.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # Print summary
    print(f"\n{'='*70}")
    print(f"{'Weight':>8} {'Seeds':>6} {'Episodes':>9} {'RL Cost':>10} {'BL Cost':>10} {'RL Dis':>10} {'BL Dis':>10}")
    print("-" * 70)
    for w in weights:
        d = weight_data[w]
        print(f"{w:>8.2f} {d['n_seeds']:>6} {d['n_episodes']:>9} "
              f"{np.mean(d['rl_cost']):>10.3f} {np.mean(d['bl_cost']):>10.3f} "
              f"{np.mean(d['rl_dis']):>10.3f} {np.mean(d['bl_dis']):>10.3f}")

    print(f"\nDone. Output: {output_dir}")


if __name__ == "__main__":
    main()
