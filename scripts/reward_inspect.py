#!/usr/bin/env python3
"""Inspect reward breakdown from a TRAINED model's actual behavior.

Runs real episodes with the trained policy and shows per-step reward
components so you can see exactly how comfort, energy, and smoothness
penalties interact during the model's actual control strategy.

Usage:
    python scripts/reward_inspect.py --model_dir results/long_sac_seed42_ew10_.../
    python scripts/reward_inspect.py --model_dir results/... --energy_weight 3.0   # override weight
    python scripts/reward_inspect.py --model_dir results/... --start_timestamp 2023-01-15T00:00:00
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../sbsim"))

from smart_control_analysis.building_factory import building_factory, get_base_params
from smart_control_analysis.rl_trainer import BuildingRLTrainer

SMOOTHNESS_WEIGHT = 0.5


def run_episode(env, model, vec_normalize, seed=0):
    obs, _ = env.reset(seed=seed)
    done = False
    rows = []
    prev_action = None

    while not done:
        _obs = vec_normalize.normalize_obs(obs) if vec_normalize is not None else obs
        action = model.predict(_obs, deterministic=True)[0]

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        comfort_penalty = float(info.get("comfort_penalty", 0.0))
        energy_rate     = float(info.get("energy_rate", 0.0))
        energy_cost_usd = float(info.get("energy_cost_usd", 0.0))

        energy_penalty = env.energy_weight * energy_rate / max(env._energy_norm, 1.0)

        if prev_action is not None:
            delta = float(np.mean(np.abs(action - prev_action)))
            smoothness = SMOOTHNESS_WEIGHT * delta
        else:
            delta = 0.0
            smoothness = 0.0
        prev_action = action.copy()

        # Get zone temps
        zone_temps = env.sim.building.get_zone_average_temps()
        temps_c = [zone_temps[zid] - 273.15 for zid in env.zone_ids]

        local_ts = env.sim.current_timestamp.tz_convert(env.time_zone)
        hour = local_ts.hour + local_ts.minute / 60.0
        is_working = env.working_hours[0] <= hour < env.working_hours[1]

        rows.append({
            "step":             env.step_count,
            "hour":             hour,
            "is_working":       is_working,
            "comfort_penalty":  comfort_penalty,
            "energy_penalty":   energy_penalty,
            "smoothness":       smoothness,
            "action_delta":     delta,
            "reward":           float(reward),
            "energy_rate_w":    energy_rate,
            "energy_cost_usd":  energy_cost_usd,
            **{f"zone{i}_c": t for i, t in enumerate(temps_c)},
        })

    return pd.DataFrame(rows)


def print_section(title):
    print(f"\n{'─'*70}")
    print(f"  {title}")
    print(f"{'─'*70}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="Path to trained model result dir")
    parser.add_argument("--energy_weight", type=float, default=None, help="Override energy weight (default: from run_config)")
    parser.add_argument("--n_episodes", type=int, default=3)
    parser.add_argument("--start_timestamp", default=None, help="Override start (default: cold winter week)")
    parser.add_argument("--weather_csv", default="/user/gent/453/vsc45342/thesis/weather_data/belgium_weather_multiyear.csv")
    args = parser.parse_args()

    # Load run config
    config_path = os.path.join(args.model_dir, "run_config.json")
    with open(config_path) as f:
        run_config = json.load(f)

    algo = run_config["algo"]
    ew = args.energy_weight if args.energy_weight is not None else run_config.get("energy_weight", 1.0)
    floorplan = run_config.get("floorplan", "office_4room")

    # Load model
    trainer = BuildingRLTrainer(building_factory_fn=building_factory)
    trainer.load_model(os.path.join(args.model_dir, f"{algo}_2024_model.zip"), algo)
    vn_path = os.path.join(args.model_dir, f"{algo}_2024_model_vecnormalize.pkl")
    if os.path.exists(vn_path):
        trainer.load_vec_normalize(vn_path)

    params = get_base_params().copy()
    params["weather_source"] = "replay"
    params["weather_csv_path"] = args.weather_csv
    params["floorplan"] = floorplan
    params["energy_weight"] = ew
    params["max_steps"] = int(7 * 24 * 3600 / params["time_step_sec"])
    params["action_design"] = run_config.get("action_design", "reheat_per_zone")

    # Run episodes with different start dates
    starts = [
        args.start_timestamp or "2023-01-15T00:00:00",  # cold winter
        "2023-03-10T00:00:00",  # spring transition
        "2022-12-20T00:00:00",  # deep winter
    ][:args.n_episodes]

    print(f"\n{'='*70}")
    print(f"  REWARD BALANCE — TRAINED MODEL ANALYSIS")
    print(f"  Model:         {args.model_dir}")
    print(f"  Algorithm:     {algo.upper()}")
    print(f"  Energy weight: {ew}")
    print(f"  Floorplan:     {floorplan}")
    print(f"  Smoothness:    {SMOOTHNESS_WEIGHT}")
    print(f"{'='*70}")

    all_dfs = []
    for i, start in enumerate(starts):
        params["start_timestamp"] = start
        _, env = building_factory(params, training_mode="full")
        df = run_episode(env, trainer.model, trainer.vec_normalize, seed=i)
        df["episode"] = i
        df["start_date"] = start
        all_dfs.append(df)
        env.close()

        # Per-episode summary
        day = df[df["is_working"]]
        night = df[~df["is_working"]]

        print_section(f"Episode {i}: {start}")
        print(f"  {'':>20}  {'Comfort':>10}  {'Energy':>10}  {'Smooth':>10}  {'Reward':>10}")

        for label, subset in [("DAY (working)", day), ("NIGHT", night), ("ALL", df)]:
            if len(subset) == 0:
                continue
            print(f"  {label:<20}  {subset['comfort_penalty'].mean():>10.4f}  "
                  f"{subset['energy_penalty'].mean():>10.4f}  "
                  f"{subset['smoothness'].mean():>10.4f}  "
                  f"{subset['reward'].mean():>+10.4f}")

        print(f"\n  Total energy cost: ${df['energy_cost_usd'].sum():.2f}")
        print(f"  Total energy rate: mean={df['energy_rate_w'].mean():.0f}W, peak={df['energy_rate_w'].max():.0f}W")

        # Zone temps
        zone_cols = [c for c in df.columns if c.startswith("zone") and c.endswith("_c")]
        if zone_cols:
            print(f"\n  Zone temps during working hours:")
            for col in zone_cols:
                d = day[col]
                print(f"    {col}: mean={d.mean():.1f}°C  min={d.min():.1f}°C  max={d.max():.1f}°C")

    # Aggregate across episodes
    df_all = pd.concat(all_dfs, ignore_index=True)
    day_all = df_all[df_all["is_working"]]
    night_all = df_all[~df_all["is_working"]]

    print_section("AGGREGATE (all episodes)")
    print(f"\n  Per-step reward components (MEAN):")
    print(f"  {'':>20}  {'Comfort':>10}  {'Energy':>10}  {'Smooth':>10}  {'Reward':>10}")
    for label, subset in [("DAY", day_all), ("NIGHT", night_all), ("ALL", df_all)]:
        if len(subset) == 0:
            continue
        print(f"  {label:<20}  {subset['comfort_penalty'].mean():>10.4f}  "
              f"{subset['energy_penalty'].mean():>10.4f}  "
              f"{subset['smoothness'].mean():>10.4f}  "
              f"{subset['reward'].mean():>+10.4f}")

    print(f"\n  Per-step reward components (P95 — peaks):")
    print(f"  {'':>20}  {'Comfort':>10}  {'Energy':>10}  {'Smooth':>10}")
    for label, subset in [("DAY", day_all), ("NIGHT", night_all), ("ALL", df_all)]:
        if len(subset) == 0:
            continue
        print(f"  {label:<20}  {subset['comfort_penalty'].quantile(0.95):>10.4f}  "
              f"{subset['energy_penalty'].quantile(0.95):>10.4f}  "
              f"{subset['smoothness'].quantile(0.95):>10.4f}")

    print(f"\n  Per-step reward components (MAX — worst case):")
    print(f"  {'':>20}  {'Comfort':>10}  {'Energy':>10}  {'Smooth':>10}")
    for label, subset in [("DAY", day_all), ("ALL", df_all)]:
        if len(subset) == 0:
            continue
        print(f"  {label:<20}  {subset['comfort_penalty'].max():>10.4f}  "
              f"{subset['energy_penalty'].max():>10.4f}  "
              f"{subset['smoothness'].max():>10.4f}")

    # Key ratios
    mean_c = day_all["comfort_penalty"].mean()
    mean_e = day_all["energy_penalty"].mean()
    peak_c = day_all["comfort_penalty"].quantile(0.95)
    peak_e = day_all["energy_penalty"].quantile(0.95)

    print_section("BALANCE RATIOS")
    if mean_c > 1e-6:
        print(f"  Day mean:  energy/comfort = {mean_e/mean_c:.2f}x")
    else:
        print(f"  Day mean:  comfort ≈ 0 (policy keeps comfort), energy = {mean_e:.4f}")
    if peak_c > 1e-6:
        print(f"  Day p95:   energy/comfort = {peak_e/peak_c:.2f}x")
    print(f"\n  Mean action delta: {df_all['action_delta'].mean():.3f} (bang-bang=2.0, smooth<0.2)")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
