#!/usr/bin/env python3
"""Compare multiple trained models side-by-side.

Pass result directories as positional arguments. Labels are auto-generated
from run_config.json (algo + energy_weight + action_design).

Usage:
    python scripts/compare_models.py results/dir_sac_ew05 results/dir_sac_ew20 results/dir_td3_ew20
    python scripts/compare_models.py results/dir_* --output_dir results/comparison_plot
    python scripts/compare_models.py results/dir_a results/dir_b --period test --label "SAC w=0.5" "SAC w=2.0"
"""
import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_label(config: dict) -> str:
    """Auto-generate a short label from run_config."""
    algo = config.get("algo", "?").upper()
    ew = config.get("energy_weight", "?")
    ad = config.get("action_design", "")
    parts = [algo, f"w={ew}"]
    if ad and ad != "reheat_per_zone":
        parts.append(ad)
    return " ".join(parts)


def _load_episode_metrics(result_dir: str, period: str, subdir: str) -> pd.DataFrame:
    """Try to load episode metrics from compare_reeval/ first, then compare/."""
    for folder in [subdir, "compare"]:
        path = os.path.join(result_dir, folder, period, f"{period}_episode_metrics.csv")
        if os.path.exists(path):
            return pd.read_csv(path)
    return None


def main():
    parser = argparse.ArgumentParser(description="Compare multiple trained RL models")
    parser.add_argument("result_dirs", nargs="+", help="Result directories to compare")
    parser.add_argument("--output_dir", default=None, help="Where to save plots (default: results/model_comparison)")
    parser.add_argument("--period", default="test", choices=["val", "test"], help="Which period to compare")
    parser.add_argument("--label", nargs="*", default=None, help="Custom labels (one per dir, overrides auto)")
    parser.add_argument("--compare_subdir", default="compare_reeval", help="Subdirectory name for comparison results")
    args = parser.parse_args()

    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(args.result_dirs[0]), f"model_comparison_{timestamp}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Save comparison config
    compare_config = {
        "timestamp": timestamp,
        "period": args.period,
        "result_dirs": args.result_dirs,
        "custom_labels": args.label,
        "compare_subdir": args.compare_subdir,
        "output_dir": output_dir,
    }
    with open(os.path.join(output_dir, "compare_config.json"), "w") as f:
        json.dump(compare_config, f, indent=2)

    # Load data for each model
    models = []
    for i, rdir in enumerate(args.result_dirs):
        config_path = os.path.join(rdir, "run_config.json")
        if not os.path.exists(config_path):
            print(f"WARNING: skipping {rdir} — no run_config.json")
            continue
        with open(config_path) as f:
            config = json.load(f)

        df = _load_episode_metrics(rdir, args.period, args.compare_subdir)
        if df is None:
            print(f"WARNING: skipping {rdir} — no {args.period}_episode_metrics.csv found")
            continue

        label = args.label[i] if args.label and i < len(args.label) else _make_label(config)
        models.append({"label": label, "config": config, "df": df, "dir": rdir})
        print(f"  Loaded: {label} ({len(df)} episodes) from {rdir}")

    if len(models) < 2:
        print("ERROR: need at least 2 models to compare")
        sys.exit(1)

    n = len(models)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n, 3)))

    # ── 1. RL-only boxplot comparison (4 panels) ──
    _plot_rl_boxplots(models, colors, output_dir, args.period)

    # ── 2. RL vs Baseline improvement (bar chart) ──
    _plot_improvement_bars(models, colors, output_dir, args.period)

    # ── 3. Cost vs Comfort scatter (Pareto front) ──
    _plot_cost_vs_comfort(models, colors, output_dir, args.period)

    # ── 4. Summary table ──
    _print_summary_table(models, output_dir, args.period)

    print(f"\nDone. Plots saved to: {output_dir}")


def _plot_rl_boxplots(models, colors, output_dir, period):
    """Side-by-side boxplots of RL performance across models."""
    metrics = [
        ("rl_energy_cost_usd", "Energy cost (USD/ep)"),
        ("rl_discomfort_deg_h", "Discomfort (°C·h/ep)"),
        ("rl_pct_outside_comfort", "Outside comfort (%)"),
        ("rl_max_temp_deviation_c", "Max deviation (°C)"),
    ]
    available = [(col, label) for col, label in metrics if all(col in m["df"].columns for m in models)]
    if not available:
        return

    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 6))
    if len(available) == 1:
        axes = [axes]
    fig.suptitle(f"RL Agent Comparison — {period.upper()} period", fontsize=14, fontweight="bold")

    labels = [m["label"] for m in models]
    for ax, (col, ylabel) in zip(axes, available):
        data = [m["df"][col].to_numpy() for m in models]
        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6)
        for j, box in enumerate(bp["boxes"]):
            box.set_facecolor(colors[j])
            box.set_alpha(0.7)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    path = os.path.join(output_dir, f"{period}_rl_comparison_boxplots.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_improvement_bars(models, colors, output_dir, period):
    """Bar chart showing % improvement over baseline for cost and comfort."""
    labels = [m["label"] for m in models]

    cost_improve = []
    comfort_improve = []
    for m in models:
        df = m["df"]
        bl_cost = df["baseline_energy_cost_usd"].mean()
        rl_cost = df["rl_energy_cost_usd"].mean()
        cost_improve.append((bl_cost - rl_cost) / bl_cost * 100 if bl_cost > 0 else 0)

        bl_dis = df["baseline_discomfort_deg_h"].mean()
        rl_dis = df["rl_discomfort_deg_h"].mean()
        comfort_improve.append((bl_dis - rl_dis) / bl_dis * 100 if bl_dis > 0 else 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Improvement over Baseline — {period.upper()} period", fontsize=14, fontweight="bold")

    x = np.arange(len(labels))
    bars1 = ax1.bar(x, cost_improve, color=[colors[i] for i in range(len(labels))], alpha=0.8)
    ax1.set_ylabel("Cost reduction (%)")
    ax1.set_title("Energy cost")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right")
    ax1.axhline(0, color="k", linewidth=0.5)
    ax1.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars1, cost_improve):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=10)

    bars2 = ax2.bar(x, comfort_improve, color=[colors[i] for i in range(len(labels))], alpha=0.8)
    ax2.set_ylabel("Discomfort reduction (%)")
    ax2.set_title("Thermal discomfort")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha="right")
    ax2.axhline(0, color="k", linewidth=0.5)
    ax2.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars2, comfort_improve):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    path = os.path.join(output_dir, f"{period}_improvement_bars.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_cost_vs_comfort(models, colors, output_dir, period):
    """Scatter: mean energy cost vs mean discomfort per model (Pareto-style)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(f"Cost vs Comfort Trade-off — {period.upper()} period", fontsize=13, fontweight="bold")

    for i, m in enumerate(models):
        df = m["df"]
        cost_mean = df["rl_energy_cost_usd"].mean()
        dis_mean = df["rl_discomfort_deg_h"].mean()
        cost_std = df["rl_energy_cost_usd"].std()
        dis_std = df["rl_discomfort_deg_h"].std()

        ax.errorbar(cost_mean, dis_mean, xerr=cost_std, yerr=dis_std,
                     fmt="o", color=colors[i], markersize=10, capsize=5,
                     label=m["label"])

    # Also plot baseline (should be similar across models, use first)
    df0 = models[0]["df"]
    bl_cost = df0["baseline_energy_cost_usd"].mean()
    bl_dis = df0["baseline_discomfort_deg_h"].mean()
    ax.plot(bl_cost, bl_dis, "kX", markersize=14, label="Baseline", zorder=5)

    ax.set_xlabel("Energy cost (USD / episode)")
    ax.set_ylabel("Discomfort (°C·h / episode)")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, f"{period}_cost_vs_comfort.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _print_summary_table(models, output_dir, period):
    """Print and save a summary table."""
    rows = []
    for m in models:
        df = m["df"]
        bl_cost = df["baseline_energy_cost_usd"].mean()
        rl_cost = df["rl_energy_cost_usd"].mean()
        bl_dis = df["baseline_discomfort_deg_h"].mean()
        rl_dis = df["rl_discomfort_deg_h"].mean()

        row = {
            "Model": m["label"],
            "RL Cost (USD)": f"{rl_cost:.2f}",
            "BL Cost (USD)": f"{bl_cost:.2f}",
            "Cost Δ (%)": f"{(bl_cost - rl_cost) / bl_cost * 100:.1f}" if bl_cost > 0 else "N/A",
            "RL Discomfort (°C·h)": f"{rl_dis:.1f}",
            "BL Discomfort (°C·h)": f"{bl_dis:.1f}",
            "Comfort Δ (%)": f"{(bl_dis - rl_dis) / bl_dis * 100:.1f}" if bl_dis > 0 else "N/A",
            "RL Reward": f"{df['rl_reward'].mean():.1f}",
            "BL Reward": f"{df['baseline_reward'].mean():.1f}",
        }
        if "rl_pct_outside_comfort" in df.columns:
            row["Outside (%)"] = f"{df['rl_pct_outside_comfort'].mean():.1f}"
        if "rl_max_temp_deviation_c" in df.columns:
            row["Max Dev (°C)"] = f"{df['rl_max_temp_deviation_c'].mean():.2f}"
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    print(f"\n{'='*80}")
    print(f"  SUMMARY — {period.upper()} period")
    print(f"{'='*80}")
    print(summary_df.to_string(index=False))

    path = os.path.join(output_dir, f"{period}_summary_table.csv")
    summary_df.to_csv(path, index=False)
    print(f"\n  Saved: {path}")


if __name__ == "__main__":
    main()
