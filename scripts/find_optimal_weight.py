#!/usr/bin/env python3
"""Find the optimal energy weight for TQC by analysing trained models.

For each weight value, pools episodes across seeds (if multiple runs given),
tests whether RL comfort is statistically better than baseline (Mann-Whitney
p < 0.05, RL mean < baseline mean), then finds the lowest energy cost among
weights that pass the comfort test.

Usage:
    python scripts/find_optimal_weight.py \\
        results/long_tqc_seed42_ew10_... \\
        results/long_tqc_seed42_ew15_... \\
        results/long_tqc_seed42_ew20_... \\
        results/long_tqc_seed42_ew25_... \\
        results/long_tqc_seed42_ew30_... \\
        --period test

    # Multiple seeds per weight — pass all dirs, script groups by weight automatically:
    python scripts/find_optimal_weight.py \\
        results/long_tqc_seed42_ew15_... \\
        results/long_tqc_seed123_ew15_... \\
        results/long_tqc_seed456_ew15_... \\
        results/long_tqc_seed42_ew20_... \\
        --period test
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


PASS_COLOR = "#2ecc71"
FAIL_COLOR = "#e74c3c"
OPTIMAL_COLOR = "#2980b9"


# ── data loading ──────────────────────────────────────────────────────────────

def _load_run(result_dir: str, period: str, compare_subdir: str):
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


def _group_by_weight(runs: list[dict]) -> dict[float, list[dict]]:
    groups = {}
    for r in runs:
        w = r["weight"]
        groups.setdefault(w, []).append(r)
    return dict(sorted(groups.items()))


# ── analysis ──────────────────────────────────────────────────────────────────

def _analyse_weight(weight: float, runs: list[dict], p_threshold: float) -> dict:
    pooled = pd.concat([r["df"] for r in runs], ignore_index=True)
    n_seeds = len(runs)
    n_eps = len(pooled)

    rl_dis  = pooled["rl_discomfort_deg_h"].dropna().to_numpy()
    bl_dis  = pooled["baseline_discomfort_deg_h"].dropna().to_numpy()
    rl_cost = pooled["rl_energy_cost_usd"].dropna().to_numpy()
    bl_cost = pooled["baseline_energy_cost_usd"].dropna().to_numpy()

    # Comfort test: RL must be statistically better (lower) than baseline
    stat, pval = stats.mannwhitneyu(rl_dis, bl_dis, alternative="less")
    rl_dis_mean  = float(np.mean([r["df"]["rl_discomfort_deg_h"].mean() for r in runs]))
    bl_dis_mean  = float(pooled["baseline_discomfort_deg_h"].mean())
    rl_cost_mean = float(np.mean([r["df"]["rl_energy_cost_usd"].mean() for r in runs]))
    bl_cost_mean = float(pooled["baseline_energy_cost_usd"].mean())

    comfort_passes = (pval < p_threshold) and (rl_dis_mean < bl_dis_mean)
    dis_improvement_pct  = (bl_dis_mean  - rl_dis_mean)  / bl_dis_mean  * 100 if bl_dis_mean  > 0 else 0.0
    cost_improvement_pct = (bl_cost_mean - rl_cost_mean) / bl_cost_mean * 100 if bl_cost_mean > 0 else 0.0

    # Std across seed means (if multiple seeds)
    rl_cost_std = float(np.std([r["df"]["rl_energy_cost_usd"].mean() for r in runs]))
    rl_dis_std  = float(np.std([r["df"]["rl_discomfort_deg_h"].mean() for r in runs]))

    return {
        "weight":               weight,
        "n_seeds":              n_seeds,
        "n_episodes":           n_eps,
        "rl_cost_mean":         rl_cost_mean,
        "rl_cost_std":          rl_cost_std,
        "bl_cost_mean":         bl_cost_mean,
        "cost_improvement_pct": cost_improvement_pct,
        "rl_dis_mean":          rl_dis_mean,
        "rl_dis_std":           rl_dis_std,
        "bl_dis_mean":          bl_dis_mean,
        "dis_improvement_pct":  dis_improvement_pct,
        "comfort_pval":         pval,
        "comfort_passes":       comfort_passes,
    }


# ── plotting ──────────────────────────────────────────────────────────────────

def _plot(results: list[dict], output_dir: str, period: str, p_threshold: float):
    weights      = [r["weight"]       for r in results]
    costs        = [r["rl_cost_mean"] for r in results]
    cost_stds    = [r["rl_cost_std"]  for r in results]
    dis          = [r["rl_dis_mean"]  for r in results]
    dis_stds     = [r["rl_dis_std"]   for r in results]
    bl_costs     = [r["bl_cost_mean"] for r in results]
    bl_dis       = [r["bl_dis_mean"]  for r in results]
    passes       = [r["comfort_passes"] for r in results]
    pvals        = [r["comfort_pval"]   for r in results]

    passing_weights = [w for w, p in zip(weights, passes) if p]
    optimal = min(passing_weights, key=lambda w: results[weights.index(w)]["rl_cost_mean"]) \
              if passing_weights else None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    fig.suptitle(f"Energy Weight Optimisation — TQC, {period.capitalize()} split\n"
                 f"(comfort test: Mann-Whitney p < {p_threshold}, RL < baseline)",
                 fontsize=12, fontweight="bold")

    colors = [PASS_COLOR if p else FAIL_COLOR for p in passes]

    # ── Top panel: energy cost ──
    ax1.plot(weights, bl_costs, color="gray", linewidth=1.5, linestyle="--",
             label="Baseline cost", zorder=1)
    ax1.errorbar(weights, costs, yerr=cost_stds,
                 fmt="none", color="black", capsize=4, linewidth=1.2, zorder=2)
    for w, c, col in zip(weights, costs, colors):
        ax1.scatter(w, c, color=col, s=90, zorder=3)
    if optimal is not None:
        ax1.axvline(optimal, color=OPTIMAL_COLOR, linewidth=1.5, linestyle=":",
                    label=f"Optimal weight = {optimal}")
    ax1.set_ylabel("Mean energy cost (USD / episode)")
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=9)

    # ── Bottom panel: discomfort ──
    ax2.plot(weights, bl_dis, color="gray", linewidth=1.5, linestyle="--",
             label="Baseline discomfort", zorder=1)
    ax2.errorbar(weights, dis, yerr=dis_stds,
                 fmt="none", color="black", capsize=4, linewidth=1.2, zorder=2)
    for w, d, col in zip(weights, dis, colors):
        ax2.scatter(w, d, color=col, s=90, zorder=3)
    if optimal is not None:
        ax2.axvline(optimal, color=OPTIMAL_COLOR, linewidth=1.5, linestyle=":")
    # Annotate p-values
    for w, pv, d in zip(weights, pvals, dis):
        sig = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else "ns"
        ax2.annotate(sig, (w, d), textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=8)
    ax2.set_xlabel("Energy weight")
    ax2.set_ylabel("Mean discomfort (°C·h / episode)")
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=9)

    # Shared legend for pass/fail
    legend_patches = [
        mpatches.Patch(color=PASS_COLOR, label="Comfort test passed (p < 0.05)"),
        mpatches.Patch(color=FAIL_COLOR, label="Comfort test failed"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=2,
               fontsize=9, bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    path = os.path.join(output_dir, f"{period}_weight_optimisation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return optimal


# ── summary table ─────────────────────────────────────────────────────────────

def _print_summary(results: list[dict], optimal, p_threshold: float):
    print(f"\n{'='*90}")
    print(f"  WEIGHT OPTIMISATION SUMMARY  (comfort threshold: p < {p_threshold})")
    print(f"{'='*90}")
    header = f"{'Weight':>8} {'Seeds':>6} {'RL Cost':>10} {'BL Cost':>10} "
    header += f"{'Cost Δ%':>8} {'RL Dis':>10} {'BL Dis':>10} {'Dis Δ%':>8} {'p-val':>8} {'Pass':>5}"
    print(header)
    print("-" * 90)
    for r in results:
        sig = "***" if r["comfort_pval"] < 0.001 else "**" if r["comfort_pval"] < 0.01 \
              else "*" if r["comfort_pval"] < 0.05 else "ns"
        mark = "✓" if r["comfort_passes"] else "✗"
        optimal_mark = " ← OPTIMAL" if r["weight"] == optimal else ""
        print(f"{r['weight']:>8.2f} {r['n_seeds']:>6} "
              f"{r['rl_cost_mean']:>10.3f} {r['bl_cost_mean']:>10.3f} "
              f"{r['cost_improvement_pct']:>+8.1f}% "
              f"{r['rl_dis_mean']:>10.3f} {r['bl_dis_mean']:>10.3f} "
              f"{r['dis_improvement_pct']:>+8.1f}% "
              f"{r['comfort_pval']:>7.4f}{sig:3} {mark}{optimal_mark}")

    if optimal is not None:
        print(f"\n  → Optimal energy weight: {optimal}")
    else:
        print(f"\n  → No weight passed the comfort test (p < {p_threshold})")
        # Find the closest to passing
        best = min(results, key=lambda r: r["comfort_pval"])
        print(f"     Closest to passing: weight={best['weight']} (p={best['comfort_pval']:.4f})")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dirs", nargs="+",
                        help="Result dirs (run_config.json must contain energy_weight)")
    parser.add_argument("--period", default="test", choices=["val", "test"])
    parser.add_argument("--p_threshold", type=float, default=0.05,
                        help="p-value threshold for comfort test (default: 0.05)")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--compare_subdir", default="compare_reeval")
    args = parser.parse_args()

    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(args.result_dirs[0]), f"weight_optimisation_{timestamp}"
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
    groups = _group_by_weight(runs)
    print(f"\nLoaded {len(runs)} runs across {len(groups)} weight values: "
          f"{list(groups.keys())}")

    # Analyse each weight
    results = []
    for weight, weight_runs in groups.items():
        r = _analyse_weight(weight, weight_runs, args.p_threshold)
        results.append(r)
        status = "PASS" if r["comfort_passes"] else "FAIL"
        print(f"  w={weight}: cost={r['rl_cost_mean']:.3f} dis={r['rl_dis_mean']:.3f} "
              f"p={r['comfort_pval']:.4f} [{status}]")

    # Plot and print
    optimal = _plot(results, output_dir, args.period, args.p_threshold)
    _print_summary(results, optimal, args.p_threshold)

    # Save results CSV
    pd.DataFrame(results).to_csv(
        os.path.join(output_dir, f"{args.period}_weight_optimisation.csv"), index=False
    )
    print(f"\nDone. Output: {output_dir}")


if __name__ == "__main__":
    main()
