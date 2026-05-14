#!/usr/bin/env python3
"""Run trained model(s) for a single episode and produce a chosen plot.

Plot modes:
  full      — 6-panel trace + matching baseline 6-panel (default)
  temp_only — single panel: room temperature for one model (+ baseline if not skipped)
  compare   — overlay mean room temperature across multiple result_dirs on one plot
  zones     — one model, one line per zone

Usage:
    python scripts/plot_model.py results/sac_seed42_ew15
    python scripts/plot_model.py results/sac_seed42_ew15 --mode temp_only --start 2024-01-15
    python scripts/plot_model.py results/sac_seed42_ew15 results/td3_seed123_ew15 results/tqc_seed42_ew125 \
        --mode compare --start 2024-01-15
    python scripts/plot_model.py results/sac_seed42_ew15 --mode zones --start 2024-01-15
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from smart_control_analysis.building_factory import building_factory, get_base_params
from smart_control_analysis.rl_trainer import BuildingRLTrainer
from smart_control_analysis.runner import _sample_start_in_period, _run_episode_trace
from smart_control_analysis.eval_plotter import _plot_episode_trace_6panel
from smart_control_analysis.baseline_controller import ThermostatBaselineController


DEFAULT_WEATHER_CSV = "/home/warre/Documents/THESIS/weather_data/belgium_weather_multiyear.csv"

# Output format/dpi for saved figures — set in main() from CLI flags.
# Default: vector PDF (best for thesis). DPI only matters for raster formats.
SAVE_DPI = 300
SAVE_EXT = "pdf"


def _with_ext(fname):
    """Replace the extension on a filename with the current SAVE_EXT."""
    base, _ = os.path.splitext(fname)
    return f"{base}.{SAVE_EXT}"


def load_trainer(result_dir, weather_csv_override=None):
    """Reconstruct trainer + load weights from a result directory."""
    config_path = os.path.join(result_dir, "run_config.json")
    if not os.path.exists(config_path):
        sys.exit(f"ERROR: run_config.json not found in {result_dir}")
    with open(config_path) as f:
        run_config = json.load(f)

    weather_csv = weather_csv_override or run_config["weather_csv"]
    if not os.path.exists(weather_csv):
        if os.path.exists(DEFAULT_WEATHER_CSV):
            print(f"  weather_csv '{weather_csv}' not found, falling back to {DEFAULT_WEATHER_CSV}")
            weather_csv = DEFAULT_WEATHER_CSV
        else:
            sys.exit(f"ERROR: weather_csv not found: {weather_csv}")

    base = get_base_params().copy()
    base["weather_source"] = "replay"
    base["weather_csv_path"] = weather_csv
    base["time_zone"] = "Europe/Brussels"
    base["time_step_sec"] = int(base.get("time_step_sec", 300))
    base["floorplan"] = run_config.get("floorplan", "single_room")
    base["energy_weight"] = run_config.get("energy_weight", 2.0)
    base["action_design"] = run_config.get("action_design", "reheat_per_zone")

    training_mode = run_config.get("eval_training_mode", run_config.get("training_mode", "full"))

    trainer = BuildingRLTrainer(
        building_factory_fn=building_factory,
        base_params=base,
        default_factory_kwargs={"training_mode": training_mode},
    )

    algo = run_config["algo"]
    model_path = os.path.join(result_dir, f"{algo}_2024_model.zip")
    if not os.path.exists(model_path):
        model_path = os.path.join(result_dir, f"{algo}_2024_model")
    trainer.load_model(model_path, algo)

    vn_path = os.path.join(result_dir, f"{algo}_2024_model_vecnormalize.pkl")
    if os.path.exists(vn_path):
        trainer.load_vec_normalize(vn_path)

    return trainer, run_config, training_mode


def _episode_params(trainer, start_timestamp, episode_days):
    p = trainer.base_params.copy()
    p["start_timestamp"] = start_timestamp
    p["max_steps"] = int(episode_days * 24 * 3600 / p["time_step_sec"])
    comfort_band_k = p.get("comfort_band_k", (294.15, 295.15))
    p["initial_temp_celsius"] = 0.5 * (comfort_band_k[0] + comfort_band_k[1]) - 273.15
    return p


def run_rl_episode(trainer, training_mode, start_timestamp, episode_days, seed, deterministic=True):
    """Roll out one RL episode and return (trace_df, comfort_band_c, action_design)."""
    p = _episode_params(trainer, start_timestamp, episode_days)
    env = trainer.create_env(params=p, training_mode=training_mode)
    vn = trainer.vec_normalize

    def policy(obs, env):
        if vn is not None:
            obs = vn.normalize_obs(obs)
        return trainer.model.predict(obs, deterministic=deterministic)[0]

    action_design = getattr(env, "action_design", "reheat_per_zone")
    df, _metrics, comfort_band_c = _run_episode_trace(
        env, policy, "rl", seed=seed, action_design=action_design,
    )
    env.close()
    return df, comfort_band_c, action_design


def run_baseline_episode(trainer, training_mode, start_timestamp, episode_days, seed,
                         baseline_night_off=False):
    """Roll out one thermostat-baseline episode."""
    p = _episode_params(trainer, start_timestamp, episode_days)
    env = trainer.create_env(params=p, training_mode=training_mode)
    baseline = ThermostatBaselineController(
        comfort_band_k=env.comfort_band_k,
        working_hours=env.working_hours,
        night_setback_k=getattr(env, "night_setback_k", 0.0),
        night_off=baseline_night_off,
    )
    action_design = getattr(env, "action_design", "reheat_per_zone")
    df, _metrics, comfort_band_c = _run_episode_trace(
        env,
        lambda obs, env: baseline.get_action(obs, env),
        "baseline",
        seed=seed,
        action_design=action_design,
    )
    env.close()
    return df, comfort_band_c, action_design


def make_six_panel(df, comfort_band_c, action_design, output_dir, title, fname):
    """Calls eval_plotter's 6-panel routine. It hardcodes dpi=150, so we
    monkeypatch plt.savefig for the duration of the call to honour SAVE_DPI."""
    fig_path = os.path.join(output_dir, _with_ext(fname))
    original_savefig = plt.savefig

    def patched_savefig(*args, **kwargs):
        kwargs["dpi"] = SAVE_DPI
        return original_savefig(*args, **kwargs)

    plt.savefig = patched_savefig
    try:
        _plot_episode_trace_6panel(
            df, comfort_band_c,
            title=title,
            fig_path=fig_path,
            action_design=action_design,
        )
    finally:
        plt.savefig = original_savefig
    return fig_path


def _format_time_axis(ax, df):
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %m-%d %H:%M"))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=3))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    if not df.empty:
        ax.set_xlim(df["timestamp"].min(),
                    df["timestamp"].max() + pd.Timedelta(hours=3))


def _draw_comfort_band(ax, df, comfort_band_c):
    """Draw the comfort band, but only over working hours (when the setpoint is active).

    Detects working hours via the per-step comfort_low_c column: night setback lowers it,
    so daytime rows are those at the max comfort_low_c value. fill_between with `where=`
    automatically breaks the band into separate daytime segments.
    """
    if "comfort_low_c" in df.columns and df["comfort_low_c"].notna().any():
        cb_low = df["comfort_low_c"].to_numpy()
        cb_high = df["comfort_high_c"].to_numpy()
        day_low = float(np.nanmax(cb_low))
        day_mask = cb_low >= day_low - 0.01
    else:
        cb_low = np.full(len(df), comfort_band_c[0])
        cb_high = np.full(len(df), comfort_band_c[1])
        day_mask = np.ones(len(df), dtype=bool)

    ax.fill_between(df["timestamp"], cb_low, cb_high,
                    where=day_mask,
                    alpha=0.12, color="green", label="Comfort band")


def make_temp_only(df, comfort_band_c, output_dir, title, fname,
                    baseline_df=None):
    """Single-panel temperature plot. Mean line + min/max band, optional baseline overlay."""
    fig, ax = plt.subplots(figsize=(14, 5))
    _draw_comfort_band(ax, df, comfort_band_c)

    if "room_temp_min_c" in df.columns and "room_temp_max_c" in df.columns:
        ax.fill_between(df["timestamp"], df["room_temp_min_c"], df["room_temp_max_c"],
                        alpha=0.18, color="tab:blue", label="RL zone range")
    ax.plot(df["timestamp"], df["room_temp_c"], color="tab:blue", linewidth=1.8,
            label="RL zone mean")

    if baseline_df is not None and not baseline_df.empty:
        if "room_temp_min_c" in baseline_df.columns:
            ax.fill_between(baseline_df["timestamp"],
                            baseline_df["room_temp_min_c"], baseline_df["room_temp_max_c"],
                            alpha=0.12, color="tab:gray", label="Baseline zone range")
        ax.plot(baseline_df["timestamp"], baseline_df["room_temp_c"],
                color="black", linewidth=1.5, linestyle="-.", label="Baseline zone mean")

    ax.set_ylabel("Room temp (°C)")
    ax.set_xlabel("Timestamp")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    _format_time_axis(ax, df)
    plt.tight_layout()

    fig_path = os.path.join(output_dir, _with_ext(fname))
    plt.savefig(fig_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved temp-only plot: {fig_path}")
    return fig_path


def make_compare(traces, comfort_band_c, output_dir, title, fname,
                  baseline_df=None):
    """Overlay multiple algos' mean room temperature on a single plot.

    traces: list of dicts {"label": str, "df": pd.DataFrame}
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    # Use the first trace's df for comfort band timestamps
    ref_df = traces[0]["df"]
    _draw_comfort_band(ax, ref_df, comfort_band_c)

    colors = plt.cm.tab10.colors
    for i, tr in enumerate(traces):
        df = tr["df"]
        ax.plot(df["timestamp"], df["room_temp_c"],
                color=colors[i % len(colors)], linewidth=1.8, label=tr["label"])

    if baseline_df is not None and not baseline_df.empty:
        ax.plot(baseline_df["timestamp"], baseline_df["room_temp_c"],
                color="black", linewidth=1.5, linestyle="-.", label="Baseline")

    ax.set_ylabel("Room temp (°C)")
    ax.set_xlabel("Timestamp")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
    _format_time_axis(ax, ref_df)
    plt.tight_layout()

    fig_path = os.path.join(output_dir, _with_ext(fname))
    plt.savefig(fig_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved compare plot: {fig_path}")
    return fig_path


def make_zones(df, comfort_band_c, output_dir, title, fname):
    """Per-zone temperature lines for one model."""
    zone_cols = sorted([c for c in df.columns if c.startswith("zone_temp_c_")],
                       key=lambda s: int(s.rsplit("_", 1)[-1]))
    if not zone_cols:
        print("  WARN: no per-zone columns in trace, falling back to mean only")
        zone_cols = ["room_temp_c"]

    fig, ax = plt.subplots(figsize=(14, 5))
    _draw_comfort_band(ax, df, comfort_band_c)

    colors = plt.cm.tab20.colors
    for i, col in enumerate(zone_cols):
        zone_idx = col.rsplit("_", 1)[-1]
        ax.plot(df["timestamp"], df[col],
                color=colors[i % len(colors)], linewidth=1.3,
                label=f"Zone {zone_idx}")

    ax.set_ylabel("Zone temp (°C)")
    ax.set_xlabel("Timestamp")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    _format_time_axis(ax, df)
    plt.tight_layout()

    fig_path = os.path.join(output_dir, _with_ext(fname))
    plt.savefig(fig_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved zones plot: {fig_path}")
    return fig_path


def resolve_start(args, time_zone):
    if args.start:
        return args.start
    rng = np.random.default_rng(args.seed)
    return _sample_start_in_period(
        rng=rng,
        start=args.period_start,
        end=args.period_end,
        episode_days=args.episode_days,
        time_zone=time_zone,
        start_hour=0,
    ).isoformat()


def main():
    parser = argparse.ArgumentParser(description="Run model(s) and generate a chosen plot")
    parser.add_argument("result_dirs", nargs="+",
                        help="One or more result directories (each contains run_config.json). "
                             "Multiple required for --mode compare.")
    parser.add_argument("--mode", choices=["full", "temp_only", "compare", "zones"],
                        default="full",
                        help="Which plot to produce (default: full = 6-panel + baseline)")
    parser.add_argument("--output_dir", default=None,
                        help="Where to save plots (default: <first_result_dir>/plots)")
    parser.add_argument("--start", default=None,
                        help="Episode start (e.g. 2024-01-15). If unset, sampled from --period_start/--period_end.")
    parser.add_argument("--period_start", default="2023-12-01")
    parser.add_argument("--period_end", default="2024-03-24")
    parser.add_argument("--episode_days", type=int, default=7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stochastic", action="store_true",
                        help="Sample actions stochastically instead of deterministic argmax")
    parser.add_argument("--weather_csv", default=None,
                        help=f"Override weather CSV path (default: from run_config, "
                             f"or {DEFAULT_WEATHER_CSV} if missing)")
    parser.add_argument("--no_baseline", action="store_true",
                        help="Skip the baseline rollout/overlay")
    parser.add_argument("--baseline_night_off", action="store_true",
                        help="Baseline turns off all heating at night")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Raster DPI when --format png (default: 300). Ignored for vector formats.")
    parser.add_argument("--format", choices=["pdf", "png", "svg"], default="pdf",
                        help="Output format (default: pdf — vector, best for thesis)")
    args = parser.parse_args()

    global SAVE_DPI, SAVE_EXT
    SAVE_DPI = args.dpi
    SAVE_EXT = args.format

    if args.mode != "compare" and len(args.result_dirs) > 1:
        print(f"NOTE: --mode {args.mode} only uses the first result_dir; "
              f"ignoring {len(args.result_dirs) - 1} others")
    if args.mode == "compare" and len(args.result_dirs) < 2:
        print("NOTE: --mode compare with one result_dir — overlay will have one line")

    output_dir = args.output_dir or os.path.join(args.result_dirs[0], "plots")
    os.makedirs(output_dir, exist_ok=True)

    # Resolve start_ts once using the first trainer's time zone (all use Europe/Brussels)
    first_trainer, first_cfg, first_tm = load_trainer(
        args.result_dirs[0], weather_csv_override=args.weather_csv)
    start_ts = resolve_start(args, first_trainer.base_params["time_zone"])
    print(f"Episode start: {start_ts}  days={args.episode_days}  seed={args.seed}")
    print(f"Output: {output_dir}\n")

    def _run(trainer, training_mode, label):
        print(f"  rolling out {label} ...")
        return run_rl_episode(
            trainer=trainer,
            training_mode=training_mode,
            start_timestamp=start_ts,
            episode_days=args.episode_days,
            seed=args.seed,
            deterministic=not args.stochastic,
        )

    def _run_baseline(trainer, training_mode):
        print("  rolling out baseline ...")
        return run_baseline_episode(
            trainer=trainer,
            training_mode=training_mode,
            start_timestamp=start_ts,
            episode_days=args.episode_days,
            seed=args.seed,
            baseline_night_off=args.baseline_night_off,
        )

    date_tag = start_ts[:10]

    if args.mode == "full":
        first_algo_label = first_cfg['algo'].upper()
        df, comfort_band_c, action_design = _run(first_trainer, first_tm, first_algo_label)
        make_six_panel(
            df, comfort_band_c, action_design, output_dir,
            title=f"{first_algo_label} — start {date_tag} ({args.episode_days}d)",
            fname="episode_6panel_rl.png",
        )
        if not args.no_baseline:
            b_df, b_comfort, b_ad = _run_baseline(first_trainer, first_tm)
            make_six_panel(
                b_df, b_comfort, b_ad, output_dir,
                title=f"Thermostat baseline — start {date_tag} ({args.episode_days}d)",
                fname="episode_6panel_baseline.png",
            )

    elif args.mode == "temp_only":
        first_algo_label = first_cfg['algo'].upper()
        df, comfort_band_c, _ = _run(first_trainer, first_tm, first_algo_label)
        b_df = None
        if not args.no_baseline:
            b_df, _, _ = _run_baseline(first_trainer, first_tm)
        make_temp_only(
            df, comfort_band_c, output_dir,
            title=f"{first_algo_label} — room temperature — {date_tag} ({args.episode_days}d)",
            fname=f"temp_only_{first_cfg['algo']}_{date_tag}.png",
            baseline_df=b_df,
        )

    elif args.mode == "compare":
        traces = []
        comfort_band_c = None
        # First trainer already loaded
        df0, cb0, _ = _run(first_trainer, first_tm, first_cfg['algo'].upper())
        traces.append({"label": first_cfg['algo'].upper(), "df": df0})
        comfort_band_c = cb0

        for rdir in args.result_dirs[1:]:
            print(f"\n[{rdir}]")
            trainer_i, cfg_i, tm_i = load_trainer(rdir, weather_csv_override=args.weather_csv)
            df_i, _, _ = _run(trainer_i, tm_i, cfg_i['algo'].upper())
            traces.append({"label": cfg_i['algo'].upper(), "df": df_i})

        b_df = None
        if not args.no_baseline:
            b_df, _, _ = _run_baseline(first_trainer, first_tm)

        algos_tag = "_".join(t["label"].lower() for t in traces)
        make_compare(
            traces, comfort_band_c, output_dir,
            title=f"Algorithm comparison — room temp — {date_tag} ({args.episode_days}d)",
            fname=f"compare_{algos_tag}_{date_tag}.png",
            baseline_df=b_df,
        )

    elif args.mode == "zones":
        first_algo_label = first_cfg['algo'].upper()
        df, comfort_band_c, _ = _run(first_trainer, first_tm, first_algo_label)
        make_zones(
            df, comfort_band_c, output_dir,
            title=f"{first_algo_label} — per-zone temperature — {date_tag} ({args.episode_days}d)",
            fname=f"zones_{first_cfg['algo']}_{date_tag}.png",
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
