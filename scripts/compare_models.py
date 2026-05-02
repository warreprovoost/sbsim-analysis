#!/usr/bin/env python3
"""Compare multiple trained models side-by-side.

Can operate in two modes:
  1. Individual model comparison (default): one box per result dir
  2. Grouped comparison (--group): group dirs by algorithm, pool episodes across seeds

Usage:
    # Individual comparison
    python scripts/compare_models.py results/sac_1 results/td3_1 --period test

    # Grouped comparison: 3 SAC runs vs 3 TQC runs
    python scripts/compare_models.py \\
        results/sac_1 results/sac_2 results/sac_3 \\
        results/tqc_1 results/tqc_2 results/tqc_3 \\
        --group SAC SAC SAC TQC TQC TQC \\
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


METRICS = [
    ("rl_energy_cost_usd",      "Energy cost (USD/ep)"),
    ("rl_discomfort_deg_h",     "Discomfort (°C·h/ep)"),
    ("rl_pct_outside_comfort",  "Outside comfort (%)"),
    ("rl_max_temp_deviation_c", "Max deviation (°C)"),
]

NORM_METRICS = [
    ("cost_ratio",    "Energy cost ratio (RL / Baseline)"),
    ("dis_ratio",     "Discomfort ratio (RL / Baseline)"),
]


def _add_normalized_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-episode rl/baseline ratio columns."""
    df = df.copy()
    if "baseline_energy_cost_usd" in df.columns:
        df["cost_ratio"] = df["rl_energy_cost_usd"] / df["baseline_energy_cost_usd"].replace(0, np.nan)
    if "baseline_discomfort_deg_h" in df.columns:
        df["dis_ratio"] = df["rl_discomfort_deg_h"] / df["baseline_discomfort_deg_h"].replace(0, np.nan)
    return df


def _make_label(config: dict) -> str:
    algo = config.get("algo", "?").upper()
    ew = config.get("energy_weight", "?")
    ad = config.get("action_design", "")
    parts = [algo, f"w={ew}"]
    if ad and ad != "reheat_per_zone":
        parts.append(ad)
    return " ".join(parts)


def _load_episode_metrics(result_dir: str, period: str, subdir: str) -> pd.DataFrame:
    for folder in [subdir, "compare"]:
        path = os.path.join(result_dir, folder, period, f"{period}_episode_metrics.csv")
        if os.path.exists(path):
            return pd.read_csv(path)
    return None


def _load_models(result_dirs, labels, period, compare_subdir):
    models = []
    for i, rdir in enumerate(result_dirs):
        config_path = os.path.join(rdir, "run_config.json")
        if not os.path.exists(config_path):
            print(f"WARNING: skipping {rdir} — no run_config.json")
            continue
        with open(config_path) as f:
            config = json.load(f)

        df = _load_episode_metrics(rdir, period, compare_subdir)
        if df is None:
            print(f"WARNING: skipping {rdir} — no {period}_episode_metrics.csv found")
            continue

        label = labels[i] if labels and i < len(labels) else _make_label(config)
        df = _add_normalized_metrics(df)
        models.append({"label": label, "config": config, "df": df, "dir": rdir})
        print(f"  Loaded: {label} ({len(df)} episodes) from {rdir}")
    return models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dirs", nargs="+")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--period", default="test", choices=["val", "test"])
    parser.add_argument("--label", nargs="*", default=None, help="Custom label per dir")
    parser.add_argument("--group", nargs="*", default=None,
                        help="Group name per dir (e.g. SAC SAC SAC TQC TQC TQC)")
    parser.add_argument("--vs_baseline", action="store_true",
                        help="Pool all dirs as one algorithm group and compare against baseline")
    parser.add_argument("--compare_subdir", default="compare_reeval")
    args = parser.parse_args()

    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(args.result_dirs[0]), f"model_comparison_{timestamp}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Save the exact command used to produce these results
    cmd_path = os.path.join(output_dir, "command.txt")
    with open(cmd_path, "w") as f:
        f.write(" ".join(sys.argv) + "\n")
    print(f"  Command saved to: {cmd_path}")

    models = _load_models(args.result_dirs, args.label, args.period, args.compare_subdir)
    if not models:
        print("ERROR: no models loaded")
        sys.exit(1)

    if args.vs_baseline:
        algo = models[0]["config"].get("algo", "RL").upper()
        _vs_baseline_comparison(models, algo, output_dir, args.period)
    elif args.group:
        # Attach group to each model
        for i, m in enumerate(models):
            m["group"] = args.group[i] if i < len(args.group) else m["label"]
        _grouped_comparison(models, output_dir, args.period)
    else:
        if len(models) < 2:
            print("ERROR: need at least 2 models to compare")
            sys.exit(1)
        _individual_comparison(models, output_dir, args.period)

    print(f"\nDone. Plots saved to: {output_dir}")


# ── Individual comparison (one box per model) ────────────────────────────────

def _individual_comparison(models, output_dir, period):
    n = len(models)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n, 3)))
    _plot_rl_boxplots(models, colors, output_dir, period)
    _plot_improvement_bars(models, colors, output_dir, period)
    _plot_cost_vs_comfort(models, colors, output_dir, period)
    _print_summary_table(models, output_dir, period)


def _plot_rl_boxplots(models, colors, output_dir, period):
    available = [(col, lbl) for col, lbl in METRICS
                 if all(col in m["df"].columns for m in models)]
    if not available:
        return
    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 6))
    if len(available) == 1:
        axes = [axes]
    algo_names = " vs ".join(dict.fromkeys(m["label"].split()[0] for m in models))
    fig.suptitle(f"{algo_names} — {period.capitalize()} split",
                 fontsize=14, fontweight="bold")
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
    labels = [m["label"] for m in models]
    cost_improve, comfort_improve = [], []
    for m in models:
        df = m["df"]
        bl_cost = df["baseline_energy_cost_usd"].mean()
        rl_cost = df["rl_energy_cost_usd"].mean()
        cost_improve.append((bl_cost - rl_cost) / bl_cost * 100 if bl_cost > 0 else 0)
        bl_dis = df["baseline_discomfort_deg_h"].mean()
        rl_dis = df["rl_discomfort_deg_h"].mean()
        comfort_improve.append((bl_dis - rl_dis) / bl_dis * 100 if bl_dis > 0 else 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Improvement over Baseline — {period.capitalize()} split",
                 fontsize=14, fontweight="bold")
    x = np.arange(len(labels))
    for ax, data, ylabel, title in [
        (ax1, cost_improve, "Cost reduction (%)", "Energy cost"),
        (ax2, comfort_improve, "Discomfort reduction (%)", "Thermal discomfort"),
    ]:
        bars = ax.bar(x, data, color=[colors[i] for i in range(len(labels))], alpha=0.8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.axhline(0, color="k", linewidth=0.5)
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, data):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    path = os.path.join(output_dir, f"{period}_improvement_bars.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_cost_vs_comfort(models, colors, output_dir, period):
    fig, ax = plt.subplots(figsize=(8, 6))
    algo_names = " vs ".join(dict.fromkeys(m["label"].split()[0] for m in models))
    ax.set_title(f"{algo_names} — Cost vs Comfort, {period.capitalize()} split",
                 fontsize=13, fontweight="bold")
    for i, m in enumerate(models):
        df = m["df"]
        ax.errorbar(df["rl_energy_cost_usd"].mean(), df["rl_discomfort_deg_h"].mean(),
                    xerr=df["rl_energy_cost_usd"].std(), yerr=df["rl_discomfort_deg_h"].std(),
                    fmt="o", color=colors[i], markersize=10, capsize=5, label=m["label"])
    df0 = models[0]["df"]
    ax.plot(df0["baseline_energy_cost_usd"].mean(), df0["baseline_discomfort_deg_h"].mean(),
            "kX", markersize=14, label="Baseline", zorder=5)
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
        }
        if "rl_pct_outside_comfort" in df.columns:
            row["Outside (%)"] = f"{df['rl_pct_outside_comfort'].mean():.1f}"
        if "rl_max_temp_deviation_c" in df.columns:
            row["Max Dev (°C)"] = f"{df['rl_max_temp_deviation_c'].mean():.2f}"
        rows.append(row)
    summary_df = pd.DataFrame(rows)
    print(f"\n{'='*80}\n  SUMMARY — {period.upper()} period\n{'='*80}")
    print(summary_df.to_string(index=False))
    path = os.path.join(output_dir, f"{period}_summary_table.csv")
    summary_df.to_csv(path, index=False)
    print(f"\n  Saved: {path}")


# ── Single algorithm vs Baseline (pool seeds, Mann-Whitney vs baseline) ──────

def _vs_baseline_comparison(models, algo, output_dir, period):
    pooled = pd.concat([m["df"] for m in models], ignore_index=True)
    n_seeds = len(models)
    n_eps = len(pooled)

    rl_color, bl_color = "#4c9be8", "#f4a261"

    # --- Boxplots: RL vs Baseline side by side for each metric ---
    BL_METRICS = [
        ("rl_energy_cost_usd",     "baseline_energy_cost_usd",     "Energy cost (USD/ep)"),
        ("rl_discomfort_deg_h",    "baseline_discomfort_deg_h",    "Discomfort (°C·h/ep)"),
        ("rl_pct_outside_comfort", "baseline_pct_outside_comfort", "Outside comfort (%)"),
        ("rl_max_temp_deviation_c","baseline_max_temp_deviation_c","Max deviation (°C)"),
    ]
    available = [(rc, bc, lbl) for rc, bc, lbl in BL_METRICS
                 if rc in pooled.columns and bc in pooled.columns]

    if available:
        fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 6))
        if len(available) == 1:
            axes = [axes]
        fig.suptitle(f"{algo} vs Baseline — {period.capitalize()} split"
                     f"  ({n_seeds} seeds, {n_eps} episodes pooled)",
                     fontsize=13, fontweight="bold")
        for ax, (rl_col, bl_col, ylabel) in zip(axes, available):
            rl_data = pooled[rl_col].dropna().to_numpy()
            bl_data = pooled[bl_col].dropna().to_numpy()
            bp = ax.boxplot([rl_data, bl_data], labels=[algo, "Baseline"],
                            patch_artist=True, widths=0.6)
            bp["boxes"][0].set_facecolor(rl_color);  bp["boxes"][0].set_alpha(0.8)
            bp["boxes"][1].set_facecolor(bl_color);  bp["boxes"][1].set_alpha(0.8)
            # Mann-Whitney U test
            stat, pval = stats.mannwhitneyu(rl_data, bl_data, alternative="two-sided")
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
            ax.set_title(f"p={pval:.3f} {sig}", fontsize=9)
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", alpha=0.3)
            # Seed mean diamonds
            for j, means in enumerate([[m["df"][rl_col].mean() for m in models],
                                        [m["df"][bl_col].mean() for m in models]]):
                ax.scatter([j + 1] * len(means), means, color="black",
                           zorder=5, s=40, marker="D")
        axes[0].scatter([], [], color="black", s=40, marker="D", label="Seed mean")
        axes[0].legend(fontsize=8)
        plt.tight_layout()
        path = os.path.join(output_dir, f"{period}_vs_baseline_boxplots.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

    # --- Summary table ---
    print(f"\n{'='*70}\n  {algo} vs BASELINE — {period.upper()} split\n{'='*70}")
    rows = []
    for rc, bc, lbl in BL_METRICS:
        if rc not in pooled.columns or bc not in pooled.columns:
            continue
        rl_mean = np.mean([m["df"][rc].mean() for m in models])
        rl_std  = np.std( [m["df"][rc].mean() for m in models])
        bl_mean = pooled[bc].mean()
        delta   = (bl_mean - rl_mean) / bl_mean * 100 if bl_mean > 0 else float("nan")
        stat, pval = stats.mannwhitneyu(pooled[rc].dropna(), pooled[bc].dropna(),
                                        alternative="two-sided")
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
        rows.append({"Metric": lbl,
                     f"{algo} (mean±std)": f"{rl_mean:.3f} ± {rl_std:.3f}",
                     "Baseline (mean)": f"{bl_mean:.3f}",
                     "Δ (%)": f"{delta:+.1f}",
                     "p-value": f"{pval:.4f} {sig}"})
    summary_df = pd.DataFrame(rows)
    print(summary_df.to_string(index=False))
    path = os.path.join(output_dir, f"{period}_vs_baseline_summary.csv")
    summary_df.to_csv(path, index=False)
    print(f"\n  Saved: {path}")


# ── Grouped comparison (pool episodes across seeds per group) ─────────────────

def _grouped_comparison(models, output_dir, period):
    # Collect unique groups in order
    seen = []
    for m in models:
        if m["group"] not in seen:
            seen.append(m["group"])
    group_names = seen

    n_groups = len(group_names)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_groups, 3)))
    color_map = {g: colors[i] for i, g in enumerate(group_names)}

    # Build group data: pooled episodes and per-seed means
    groups = {}
    for g in group_names:
        members = [m for m in models if m["group"] == g]
        pooled = pd.concat([m["df"] for m in members], ignore_index=True)
        seed_means = {col: [m["df"][col].mean() for m in members]
                      for col, _ in METRICS if col in members[0]["df"].columns}
        groups[g] = {"members": members, "pooled": pooled, "seed_means": seed_means}

    _plot_grouped_normalized_boxplots(groups, group_names, color_map, output_dir, period)
    _plot_grouped_normalized_scatter(groups, group_names, color_map, output_dir, period)
    _print_grouped_summary(groups, group_names, color_map, output_dir, period)


def _plot_grouped_boxplots(groups, group_names, color_map, output_dir, period):
    available = [(col, lbl) for col, lbl in METRICS
                 if all(col in groups[g]["pooled"].columns for g in group_names)]
    if not available:
        return

    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 6))
    if len(available) == 1:
        axes = [axes]
    fig.suptitle(f"{' vs '.join(group_names)} — {period.capitalize()} split",
                 fontsize=13, fontweight="bold")

    for ax, (col, ylabel) in zip(axes, available):
        data = [groups[g]["pooled"][col].to_numpy() for g in group_names]
        bp = ax.boxplot(data, labels=group_names, patch_artist=True, widths=0.6)
        for j, (box, g) in enumerate(zip(bp["boxes"], group_names)):
            box.set_facecolor(color_map[g])
            box.set_alpha(0.7)

        # Overlay seed means as dots
        for j, g in enumerate(group_names):
            seed_vals = groups[g]["seed_means"].get(col, [])
            ax.scatter([j + 1] * len(seed_vals), seed_vals,
                       color="black", zorder=5, s=40, marker="D",
                       label="Seed mean" if j == 0 else "")

        # Mann-Whitney U test between first two groups
        if len(group_names) == 2:
            d0 = groups[group_names[0]]["pooled"][col].dropna()
            d1 = groups[group_names[1]]["pooled"][col].dropna()
            stat, pval = stats.mannwhitneyu(d0, d1, alternative="two-sided")
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
            ax.set_title(f"p={pval:.3f} {sig}", fontsize=9)

        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)

    # Add legend for seed means
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(output_dir, f"{period}_grouped_boxplots.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_grouped_normalized_boxplots(groups, group_names, color_map, output_dir, period):
    available = [(col, lbl) for col, lbl in NORM_METRICS
                 if all(col in groups[g]["pooled"].columns for g in group_names)]
    if not available:
        return

    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 6))
    if len(available) == 1:
        axes = [axes]
    fig.suptitle(f"{' vs '.join(group_names)} — Normalised by Baseline (lower is better)"
                 f" — {period.capitalize()} split", fontsize=13, fontweight="bold")

    for ax, (col, ylabel) in zip(axes, available):
        data = [groups[g]["pooled"][col].dropna().to_numpy() for g in group_names]
        bp = ax.boxplot(data, labels=group_names, patch_artist=True, widths=0.6)
        for j, (box, g) in enumerate(zip(bp["boxes"], group_names)):
            box.set_facecolor(color_map[g])
            box.set_alpha(0.7)

        # Seed mean diamonds
        for j, g in enumerate(group_names):
            members = groups[g]["members"]
            seed_vals = [m["df"][col].mean() for m in members if col in m["df"].columns]
            ax.scatter([j + 1] * len(seed_vals), seed_vals,
                       color="black", zorder=5, s=40, marker="D",
                       label="Seed mean" if j == 0 else "")

        ax.axhline(1.0, color="gray", linewidth=1.0, linestyle="--", label="Baseline (=1)")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)

        # Mann-Whitney between all pairs
        if len(group_names) >= 2:
            for i in range(len(group_names)):
                for j in range(i + 1, len(group_names)):
                    g0, g1 = group_names[i], group_names[j]
                    d0 = groups[g0]["pooled"][col].dropna()
                    d1 = groups[g1]["pooled"][col].dropna()
                    _, pval = stats.mannwhitneyu(d0, d1, alternative="two-sided")
                    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
                    print(f"  {g0} vs {g1} | {ylabel}: p={pval:.4f} {sig}")

    axes[0].legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(output_dir, f"{period}_grouped_normalized_boxplots.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_grouped_normalized_scatter(groups, group_names, color_map, output_dir, period):
    if not all("cost_ratio" in groups[g]["pooled"].columns and
               "dis_ratio" in groups[g]["pooled"].columns for g in group_names):
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(f"{' vs '.join(group_names)} — Normalised Cost vs Discomfort"
                 f" — {period.capitalize()} split  (lower is better)",
                 fontsize=13, fontweight="bold")

    for g in group_names:
        members = groups[g]["members"]
        cost_means = [m["df"]["cost_ratio"].mean() for m in members]
        dis_means  = [m["df"]["dis_ratio"].mean()  for m in members]
        ax.scatter(cost_means, dis_means, color=color_map[g],
                   s=60, alpha=0.6, zorder=5, marker="o",
                   edgecolors="white", linewidths=0.5)
        ax.errorbar(np.mean(cost_means), np.mean(dis_means),
                    xerr=np.std(cost_means), yerr=np.std(dis_means),
                    fmt="o", color=color_map[g], markersize=10,
                    capsize=5, capthick=1.5, linewidth=1.5,
                    markeredgecolor="black", markeredgewidth=0.8,
                    label=g, zorder=6)

    ax.plot(1.0, 1.0, "kX", markersize=14, markeredgewidth=2,
            label="Baseline (1, 1)", zorder=7)
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axvline(1.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Energy cost ratio (RL / Baseline)")
    ax.set_ylabel("Discomfort ratio (RL / Baseline)")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, f"{period}_grouped_normalized_scatter.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_grouped_cost_vs_comfort(groups, group_names, color_map, output_dir, period):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(f"{' vs '.join(group_names)} — Cost vs Comfort, {period.capitalize()} split",
                 fontsize=13, fontweight="bold")

    for g in group_names:
        members = groups[g]["members"]
        # Per-seed means
        cost_means = [m["df"]["rl_energy_cost_usd"].mean() for m in members]
        dis_means  = [m["df"]["rl_discomfort_deg_h"].mean() for m in members]
        cx, cy = np.mean(cost_means), np.mean(dis_means)
        ex, ey = np.std(cost_means), np.std(dis_means)

        # Individual seed points
        ax.scatter(cost_means, dis_means, color=color_map[g],
                   s=50, alpha=0.6, zorder=5, marker="o",
                   edgecolors="white", linewidths=0.5)

        # Group mean with std error bars
        ax.errorbar(cx, cy, xerr=ex, yerr=ey,
                    fmt="o", color=color_map[g], markersize=9,
                    capsize=5, capthick=1.5, linewidth=1.5,
                    markeredgecolor="black", markeredgewidth=0.8,
                    label=g, zorder=6)

    # Baseline
    pooled0 = groups[group_names[0]]["pooled"]
    bl_cost = pooled0["baseline_energy_cost_usd"].mean()
    bl_dis  = pooled0["baseline_discomfort_deg_h"].mean()
    ax.plot(bl_cost, bl_dis, "kX", markersize=10, markeredgewidth=2,
            label="Baseline", zorder=7)

    ax.set_xlabel("Energy cost (USD / episode)")
    ax.set_ylabel("Discomfort (°C·h / episode)")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, f"{period}_grouped_cost_vs_comfort.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _print_grouped_summary(groups, group_names, color_map, output_dir, period):
    print(f"\n{'='*90}\n  GROUPED SUMMARY — {period.upper()} period\n{'='*90}")

    rows = []
    for g in group_names:
        members = groups[g]["members"]
        pooled  = groups[g]["pooled"]
        n_seeds = len(members)
        n_eps   = len(pooled)

        cost_ratio_mean = np.mean([m["df"]["cost_ratio"].mean() for m in members]) if "cost_ratio" in members[0]["df"].columns else float("nan")
        cost_ratio_std  = np.std( [m["df"]["cost_ratio"].mean() for m in members]) if "cost_ratio" in members[0]["df"].columns else float("nan")
        dis_ratio_mean  = np.mean([m["df"]["dis_ratio"].mean()  for m in members]) if "dis_ratio"  in members[0]["df"].columns else float("nan")
        dis_ratio_std   = np.std( [m["df"]["dis_ratio"].mean()  for m in members]) if "dis_ratio"  in members[0]["df"].columns else float("nan")

        row = {
            "Group":             g,
            "Seeds":             n_seeds,
            "Episodes":          n_eps,
            "Cost ratio (↓)":    f"{cost_ratio_mean:.3f} ± {cost_ratio_std:.3f}",
            "Discomfort ratio (↓)": f"{dis_ratio_mean:.3f} ± {dis_ratio_std:.3f}",
        }
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    print(summary_df.to_string(index=False))

    # Statistical tests between each pair of groups
    stat_lines = []
    if len(group_names) >= 2:
        stat_lines.append("Mann-Whitney U tests (two-sided):")
        for i in range(len(group_names)):
            for j in range(i + 1, len(group_names)):
                g0, g1 = group_names[i], group_names[j]
                for col, lbl in NORM_METRICS:
                    if col not in groups[g0]["pooled"].columns:
                        continue
                    d0 = groups[g0]["pooled"][col].dropna()
                    d1 = groups[g1]["pooled"][col].dropna()
                    stat, pval = stats.mannwhitneyu(d0, d1, alternative="two-sided")
                    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
                    line = f"  {g0} vs {g1} | {lbl:<40} p={pval:.4f} {sig}"
                    stat_lines.append(line)
                    print(line)

    path = os.path.join(output_dir, f"{period}_grouped_summary.csv")
    summary_df.to_csv(path, index=False)

    txt_path = os.path.join(output_dir, f"{period}_grouped_summary.txt")
    with open(txt_path, "w") as f:
        f.write(f"GROUPED SUMMARY — {period.upper()} period\n")
        f.write("=" * 90 + "\n")
        f.write(summary_df.to_string(index=False) + "\n\n")
        f.write("\n".join(stat_lines) + "\n")
    print(f"\n  Saved: {path}")
    print(f"  Saved: {txt_path}")


if __name__ == "__main__":
    main()
