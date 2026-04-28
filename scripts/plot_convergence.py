#!/usr/bin/env python3
"""Plot baseline-normalised convergence curves from W&B training logs.

For each W&B run this script:
  1. Pulls the val/energy_cost_usd and val/start_timestamps history.
  2. Runs the thermostat baseline on the exact same episode start timestamps
     that were logged during training.
  3. Computes the normalised cost ratio  rl_cost / baseline_cost  per chunk.
  4. Plots rolling-minimum of the ratio vs training steps for all runs,
     grouped by algorithm.

Usage:
    python scripts/plot_convergence.py \\
        --runs results/sac_s42 results/sac_s43 results/sac_s44 \\
               results/tqc_s42 results/tqc_s43 results/tqc_s44 \\
        --group SAC SAC SAC TQC TQC TQC \\
        --wandb_project my-wandb-project \\
        --output_dir results/convergence

Each result dir must contain run_config.json (with wandb_run_id or the run
name that was used during training).  Alternatively supply --run_ids directly.
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from smart_control_analysis.building_factory import building_factory, get_base_params
from smart_control_analysis.rl_trainer import BuildingRLTrainer
from smart_control_analysis.baseline_controller import ThermostatBaselineController
from smart_control_analysis.eval_plotter import _run_episode_trace


BASELINE_CACHE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "results", ".baseline_cache.json"
)

# ── helpers ──────────────────────────────────────────────────────────────────

def _build_base_params(run_config: dict) -> dict:
    base = get_base_params().copy()
    base["weather_source"] = "replay"
    base["weather_csv_path"] = run_config["weather_csv"]
    base["time_zone"] = "America/Los_Angeles"
    base["time_step_sec"] = int(base.get("time_step_sec", 300))
    base["floorplan"] = run_config.get("floorplan", "office_4room")
    base["energy_weight"] = run_config.get("energy_weight", 2.0)
    base["action_design"] = run_config.get("action_design", "reheat_per_zone")
    episode_days = run_config.get("episode_days", 7)
    base["max_steps"] = int(episode_days * 24 * 3600 / base["time_step_sec"])
    return base, episode_days


def _baseline_metrics_for_timestamps(
    timestamps: list[str],
    base: dict,
    episode_days: int,
    training_mode: str,
) -> tuple[list[float], list[float]]:
    """Run the thermostat baseline from each given start timestamp.

    Returns (costs, discomforts) — one value per timestamp.
    """
    costs, discomforts = [], []
    for ts_str in timestamps:
        p = base.copy()
        p["start_timestamp"] = ts_str
        comfort_band_k = p.get("comfort_band_k", (294.15, 295.15))
        p["initial_temp_celsius"] = 0.5 * (comfort_band_k[0] + comfort_band_k[1]) - 273.15

        trainer = BuildingRLTrainer(
            building_factory_fn=building_factory,
            base_params=p,
            default_factory_kwargs={"training_mode": training_mode},
        )
        env = trainer.create_env(params=p, training_mode=training_mode)
        baseline = ThermostatBaselineController(
            comfort_band_k=env.comfort_band_k,
            working_hours=env.working_hours,
            night_setback_k=getattr(env, "night_setback_k", 0.0),
            night_off=False,
        )
        _action_design = getattr(env, "action_design", "reheat_per_zone")
        _, b_m, _ = _run_episode_trace(
            env,
            lambda obs, env_: baseline.get_action(obs, env_),
            "baseline",
            seed=0,
            action_design=_action_design,
        )
        env.close()
        costs.append(b_m["energy_cost_usd"])
        discomforts.append(b_m["discomfort_deg_h"])
    return costs, discomforts


def _fetch_wandb_history(wandb_project: str, run_id: str) -> pd.DataFrame:
    """Return W&B history df with val metrics and start timestamps."""
    import wandb
    api = wandb.Api()
    run = api.run(f"{wandb_project}/{run_id}")
    history = run.history(
        keys=["val/energy_cost_usd", "val/discomfort_deg_h", "val/start_timestamps"],
        x_axis="_step",
        pandas=True,
    )
    history = history.rename(columns={"_step": "step"})
    history = history.dropna(subset=["val/start_timestamps"])
    return history


# ── per-run convergence ───────────────────────────────────────────────────────

def _load_baseline_cache() -> dict:
    if os.path.exists(BASELINE_CACHE_PATH):
        with open(BASELINE_CACHE_PATH) as f:
            raw = json.load(f)
        # stored as {key: [cost, dis]}, convert back to tuples
        return {k: tuple(v) for k, v in raw.items()}
    return {}


def _save_baseline_cache(cache: dict) -> None:
    os.makedirs(os.path.dirname(BASELINE_CACHE_PATH), exist_ok=True)
    with open(BASELINE_CACHE_PATH, "w") as f:
        json.dump({k: list(v) for k, v in cache.items()}, f)


def _baseline_cache_key(timestamps: list[str]) -> str:
    """Stable cache key based on the sorted episode start timestamps."""
    import hashlib
    key = ",".join(sorted(timestamps))
    return hashlib.md5(key.encode()).hexdigest()[:12]


def _compute_convergence(result_dir: str, wandb_project: str, run_id: str,
                         cache_dir: str,
                         baseline_cache: dict) -> pd.DataFrame | None:
    """Return df with columns: step, cost_ratio, dis_ratio, ema_*.

    baseline_cache is a shared dict {cache_key: (bl_cost_mean, bl_dis_mean)}
    so identical timestamp sets across runs are only simulated once.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{run_id.replace('/', '_')}_convergence.csv")
    if os.path.exists(cache_path):
        print(f"    [cache] {cache_path}")
        return pd.read_csv(cache_path)

    # Load run config
    config_path = os.path.join(result_dir, "run_config.json")
    if not os.path.exists(config_path):
        print(f"  WARNING: no run_config.json in {result_dir}, skipping")
        return None
    with open(config_path) as f:
        run_config = json.load(f)

    base, episode_days = _build_base_params(run_config)
    training_mode = run_config.get("eval_training_mode",
                                   run_config.get("training_mode", "full"))

    # Fetch W&B history
    print(f"    Fetching W&B history for run {run_id} ...")
    try:
        history = _fetch_wandb_history(wandb_project, run_id)
    except Exception as e:
        print(f"  WARNING: could not fetch W&B history: {e}")
        return None

    if history.empty:
        print(f"  WARNING: no val rows in W&B history for {run_id}")
        return None

    rows = []
    for _, row in history.iterrows():
        step = int(row["step"])
        rl_cost = float(row["val/energy_cost_usd"])
        rl_discomfort = float(row.get("val/discomfort_deg_h", np.nan))
        timestamps = json.loads(row["val/start_timestamps"])

        # Use shared baseline cache to avoid recomputing identical episode windows
        bkey = _baseline_cache_key(timestamps)
        if bkey in baseline_cache:
            bl_cost_mean, bl_dis_mean = baseline_cache[bkey]
            print(f"    step={step}: baseline from shared cache (key={bkey})")
        else:
            print(f"    step={step}: running baseline on {len(timestamps)} episodes ...")
            bl_costs, bl_discomforts = _baseline_metrics_for_timestamps(
                timestamps, base, episode_days, training_mode
            )
            bl_cost_mean = float(np.mean(bl_costs))
            bl_dis_mean  = float(np.mean(bl_discomforts))
            baseline_cache[bkey] = (bl_cost_mean, bl_dis_mean)

        cost_ratio = rl_cost / bl_cost_mean if bl_cost_mean > 0 else np.nan
        dis_ratio  = rl_discomfort / bl_dis_mean if bl_dis_mean > 0 else np.nan
        rows.append({"step": step,
                     "rl_cost": rl_cost, "baseline_cost": bl_cost_mean,
                     "rl_discomfort": rl_discomfort, "baseline_discomfort": bl_dis_mean,
                     "cost_ratio": cost_ratio, "dis_ratio": dis_ratio})

    df = pd.DataFrame(rows).sort_values("step").reset_index(drop=True)
    df["ema_cost_ratio"] = df["cost_ratio"].ewm(span=5, adjust=False).mean()
    df["ema_dis_ratio"]  = df["dis_ratio"].ewm(span=5, adjust=False).mean()

    df.to_csv(cache_path, index=False)
    print(f"    Cached to: {cache_path}")
    return df


# ── plotting ──────────────────────────────────────────────────────────────────

def _plot_convergence(all_runs: list[dict], group_names: list[str],
                      color_map: dict, output_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{' vs '.join(group_names)} — Convergence (RL / Baseline, lower is better)",
                 fontsize=13, fontweight="bold")

    panels = [
        ("ema_cost_ratio", "Energy cost ratio  (RL / Baseline)", axes[0], False),
        ("ema_dis_ratio",  "Discomfort ratio  (RL / Baseline)",  axes[1], True),
    ]

    for col, ylabel, ax, symlog in panels:
        ax.axhline(1.0, color="gray", linewidth=1.0, linestyle="--", label="Baseline (= 1)")
        ax.set_xlabel("Training steps")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)

        # Plot group mean EMA as solid line + std band — cleaner than individual seed lines
        for g in group_names:
            members = [r for r in all_runs if r["group"] == g and r.get("df") is not None]
            if not members:
                continue
            all_steps = sorted(set().union(*[set(r["df"]["step"]) for r in members]))
            interp_vals = []
            for r in members:
                s = r["df"].set_index("step")[col]
                interped = [s.asof(st) if st >= s.index.min() else np.nan for st in all_steps]
                interp_vals.append(interped)
            arr = np.array(interp_vals, dtype=float)
            mean = np.nanmean(arr, axis=0)
            std  = np.nanstd(arr,  axis=0)
            color = color_map[g]
            ax.plot(all_steps, mean, color=color, linewidth=2.0, label=g, zorder=3)
            ax.fill_between(all_steps, mean - std, mean + std, color=color, alpha=0.2)

        if symlog:
            # Linear between 0 and 1 (where differences matter), log above 1
            ax.set_yscale("symlog", linthresh=1.0, linscale=1.0)
            ax.yaxis.set_minor_locator(plt.NullLocator())
            # Subtle shading below 1 to highlight the better-than-baseline region
            ax.axhspan(0, 1.0, alpha=0.04, color="green", zorder=0)
            # Custom y-tick labels: 0.5, 0.75, 1, 10 instead of 10^0, 10^1 etc.
            ax.set_yticks([0.25, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0])
            ax.set_yticklabels(["0.25","0.5", "0.75", "1", "2", "5", "10"])
        ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, "convergence_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True,
                        help="Result dirs, one per run (must contain run_config.json)")
    parser.add_argument("--group", nargs="*", default=None,
                        help="Group name per dir (e.g. SAC SAC SAC TQC TQC TQC)")
    parser.add_argument("--run_ids", nargs="*", default=None,
                        help="W&B run IDs, one per dir (overrides auto-detection from run_config)")
    parser.add_argument("--wandb_project", default="warrepro-universiteit-gent/building-rl-thesis",
                        help="W&B project in format entity/project")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--plot_only", action="store_true",
                        help="Skip W&B fetch and baseline simulation; replot from existing cache")
    args = parser.parse_args()

    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(args.runs[0]), f"convergence_{timestamp}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Save command
    with open(os.path.join(output_dir, "command.txt"), "w") as f:
        f.write(" ".join(sys.argv) + "\n")

    cache_dir = os.path.join(output_dir, "cache")

    # Resolve group names
    n = len(args.runs)
    groups = args.group or ["run"] * n
    if len(groups) < n:
        groups += [groups[-1]] * (n - len(groups))

    # Unique group order
    seen = []
    for g in groups:
        if g not in seen:
            seen.append(g)
    group_names = seen
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(group_names), 3)))
    color_map = {g: colors[i] for i, g in enumerate(group_names)}

    if args.plot_only:
        # Load directly from cache — no W&B or simulation needed
        supplied_ids = args.run_ids or []
        all_runs = []
        for i, (rdir, group) in enumerate(zip(args.runs, groups)):
            # Resolve run_id: --run_ids first, then run_config, then unknown
            if i < len(supplied_ids):
                run_id = supplied_ids[i]
            else:
                run_id = "unknown"
                config_path = os.path.join(rdir, "run_config.json")
                if os.path.exists(config_path):
                    with open(config_path) as f:
                        cfg = json.load(f)
                    run_id = cfg.get("wandb_run_id") or cfg.get("wandb_run_name", "unknown")

            cache_path = os.path.join(cache_dir, f"{run_id.replace('/', '_')}_convergence.csv")
            if os.path.exists(cache_path):
                df = pd.read_csv(cache_path)
                print(f"  Loaded cache: {cache_path}")
            else:
                print(f"  WARNING: no cache found for {rdir} (run_id={run_id}), skipping")
                df = None
            all_runs.append({"dir": rdir, "group": group, "run_id": run_id, "df": df})
    else:
        # Resolve run IDs: prefer --run_ids, then run_config wandb_run_id field
        run_ids = args.run_ids or []
        for i, rdir in enumerate(args.runs):
            if i < len(run_ids):
                continue  # already supplied
            config_path = os.path.join(rdir, "run_config.json")
            if os.path.exists(config_path):
                with open(config_path) as f:
                    cfg = json.load(f)
                rid = cfg.get("wandb_run_id") or cfg.get("wandb_run_name")
                if rid:
                    run_ids.append(rid)
                    continue
            print(f"ERROR: could not find W&B run ID for {rdir}. "
                  f"Supply --run_ids or add wandb_run_id to run_config.json.")
            sys.exit(1)

        baseline_cache = _load_baseline_cache()
        print(f"  Loaded {len(baseline_cache)} baseline cache entries from {BASELINE_CACHE_PATH}")
        all_runs = []
        for rdir, run_id, group in zip(args.runs, run_ids, groups):
            print(f"\nProcessing: {rdir}  (group={group}, run_id={run_id})")
            df = _compute_convergence(rdir, args.wandb_project, run_id, cache_dir, baseline_cache)
            all_runs.append({"dir": rdir, "group": group, "run_id": run_id, "df": df})
        _save_baseline_cache(baseline_cache)
        print(f"  Saved {len(baseline_cache)} baseline cache entries to {BASELINE_CACHE_PATH}")

    _plot_convergence(all_runs, group_names, color_map, output_dir)
    print(f"\nDone. Output: {output_dir}")


if __name__ == "__main__":
    main()
