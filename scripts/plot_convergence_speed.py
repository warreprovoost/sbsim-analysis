#!/usr/bin/env python3
"""Plot convergence speed: steps-to-threshold for each algo across seeds.

Produces one figure with two subplots: discomfort ratio and energy cost ratio.

Usage:
    python scripts/plot_convergence_speed.py \\
        --runs results/sac_s42 results/sac_s123 results/sac_s456 \\
               results/tqc_s42 results/tqc_s123 results/tqc_s456 \\
               results/crossq_s42 results/crossq_s123 results/crossq_s456 \\
        --group SAC SAC SAC TQC TQC TQC CROSSQ CROSSQ CROSSQ \\
        --cache_dir results/convergence_sac_tqc_crossq/cache \\
        --output_dir results/convergence_speed
"""
import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_THRESHOLDS = [1.0, 0.8, 0.75, 0.5, 0.25]

METRICS = [
    ("ema_dis_ratio", "Discomfort ratio (RL / Baseline)"),
]


def _steps_to_threshold(df: pd.DataFrame, col: str, threshold: float) -> float:
    if col not in df.columns:
        return float("nan")
    hit = df[df[col] < threshold]
    if hit.empty:
        return float("nan")
    return float(hit["step"].iloc[0])


def _draw_bars(ax, group_names, color_map, thresholds, data, title):
    n_thresh = len(thresholds)
    n_groups = len(group_names)
    bar_width = 0.8 / n_groups
    x = np.arange(n_thresh)

    # Compute plot ceiling from all valid values so "did not reach" bars can fill to top
    all_valid = [v for g in group_names for t in thresholds for v in data[g][t] if not np.isnan(v)]
    y_max = max(all_valid) * 1.15 if all_valid else 1.0

    ax.set_title(title, fontsize=11, fontweight="bold")
    for gi, g in enumerate(group_names):
        means, stds, seed_vals = [], [], []
        for t in thresholds:
            vals = [v for v in data[g][t] if not np.isnan(v)]
            means.append(np.mean(vals) if vals else np.nan)
            stds.append(np.std(vals) if len(vals) > 1 else 0)
            seed_vals.append(data[g][t])

        offset = (gi - n_groups / 2 + 0.5) * bar_width

        for ti, (mean, std, seeds) in enumerate(zip(means, stds, seed_vals)):
            xi = x[ti] + offset
            valid = [v for v in seeds if not np.isnan(v)]
            nan_count = len(seeds) - len(valid)
            all_nan = len(valid) == 0

            if all_nan:
                # Hatched full-height bar = "never reached this threshold"
                ax.bar(xi, y_max, width=bar_width * 0.9,
                       color=color_map[g], alpha=0.3, hatch="///",
                       edgecolor=color_map[g])
                ax.text(xi, y_max * 0.5, "never\nreached", ha="center", va="center",
                        fontsize=7, color="black", rotation=90)
            else:
                yerr = std if len(valid) > 1 else 0
                ax.bar(xi, mean, width=bar_width * 0.9,
                       color=color_map[g], alpha=0.8,
                       yerr=yerr, capsize=4, error_kw={"linewidth": 1.5})
                ax.scatter([xi] * len(valid), valid,
                           color="black", zorder=5, s=30, marker="o", alpha=0.8)
                if nan_count > 0:
                    ax.annotate(f"{nan_count}✗", xy=(xi, mean), fontsize=7,
                                color="red", ha="center", va="bottom")

        # Legend proxy
        ax.bar(0, 0, color=color_map[g], alpha=0.8, label=g)

    ax.set_ylim(0, y_max)
    ax.set_xticks(x)
    ax.set_xticklabels([f"< {t}" for t in thresholds], fontsize=10)
    ax.set_xlabel("threshold")
    ax.set_ylabel("Training steps")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True,
                        help="Result dirs, same order as --group")
    parser.add_argument("--group", nargs="+", required=True,
                        help="Group name per dir (e.g. SAC SAC SAC TQC TQC TQC)")
    parser.add_argument("--cache_dir", required=True,
                        help="Path to cache dir from plot_convergence.py output")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS)
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.cache_dir, "..", "convergence_speed")
    output_dir = os.path.normpath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    runs = []
    for rdir, group in zip(args.runs, args.group):
        config_path = os.path.join(rdir, "run_config.json")
        if not os.path.exists(config_path):
            print(f"WARNING: no run_config.json in {rdir}, skipping")
            continue
        with open(config_path) as f:
            cfg = json.load(f)
        run_id = cfg.get("wandb_run_id") or cfg.get("wandb_run_name")
        if not run_id:
            print(f"WARNING: no wandb_run_id in {rdir}, skipping")
            continue
        cache_path = os.path.join(args.cache_dir, f"{run_id}_convergence.csv")
        if not os.path.exists(cache_path):
            print(f"WARNING: no cache CSV for run_id={run_id} in {args.cache_dir}, skipping")
            continue
        df = pd.read_csv(cache_path)
        runs.append({"group": group, "df": df, "run_id": run_id, "dir": rdir})
        print(f"  Loaded: {group} {run_id} ({len(df)} steps)")

    if not runs:
        print("ERROR: no runs loaded")
        sys.exit(1)

    seen = []
    for r in runs:
        if r["group"] not in seen:
            seen.append(r["group"])
    group_names = seen

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(group_names), 3)))
    color_map = {g: colors[i] for i, g in enumerate(group_names)}
    thresholds = sorted(args.thresholds, reverse=True)

    # Build data per metric
    all_data = {}
    for col, _ in METRICS:
        d = {g: {t: [] for t in thresholds} for g in group_names}
        for r in runs:
            for t in thresholds:
                d[r["group"]][t].append(_steps_to_threshold(r["df"], col, t))
        all_data[col] = d

    # ── Bar chart ──
    fig, ax = plt.subplots(figsize=(max(10, len(thresholds) * 2.5), 6))
    col, label = METRICS[0]
    _draw_bars(ax, group_names, color_map, thresholds, all_data[col],
               f"Convergence speed — steps to reach discomfort ratio threshold\n"
               f"(lower is faster;  bar = mean across seeds,  dots = individual seeds)")

    plt.tight_layout()
    path = os.path.join(output_dir, "convergence_speed.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # ── Summary tables ──
    for col, label in METRICS:
        rows = []
        for t in thresholds:
            row = {"threshold": f"< {t}"}
            for g in group_names:
                vals = [v for v in all_data[col][g][t] if not np.isnan(v)]
                nan_count = sum(1 for v in all_data[col][g][t] if np.isnan(v))
                entry = f"{np.mean(vals):.0f} ± {np.std(vals):.0f}" if vals else "N/A"
                if nan_count:
                    entry += f" ({nan_count} NaN)"
                row[g] = entry
            rows.append(row)

        summary_df = pd.DataFrame(rows)
        slug = col.replace("ema_", "")
        print(f"\n── {label} ──")
        print(summary_df.to_string(index=False))
        summary_df.to_csv(os.path.join(output_dir, f"convergence_speed_{slug}.csv"), index=False)
        with open(os.path.join(output_dir, f"convergence_speed_{slug}.txt"), "w") as f:
            f.write(f"{label}\n" + summary_df.to_string(index=False) + "\n")

    print(f"\nSaved to: {output_dir}/")


if __name__ == "__main__":
    main()
