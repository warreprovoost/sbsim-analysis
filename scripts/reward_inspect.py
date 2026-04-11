#!/usr/bin/env python3
"""Inspect reward breakdown during actual training episodes.

Runs episodes with a random/fixed/trained policy and prints the actual
magnitudes of comfort penalty, energy penalty, and smoothness penalty
so you can verify the reward function is well balanced.

Usage:
    python scripts/reward_inspect.py
    python scripts/reward_inspect.py --energy_weight 1.0 --n_episodes 3
    python scripts/reward_inspect.py --model_dir results/long_sac_seed42_ew10_.../
    python scripts/reward_inspect.py --policy heat_max
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


def run_episode(env, policy="random", seed=0, model=None, vec_normalize=None):
    obs, _ = env.reset(seed=seed)
    done = False
    rows = []
    prev_action = None

    while not done:
        if policy == "model" and model is not None:
            _obs = vec_normalize.normalize_obs(obs) if vec_normalize is not None else obs
            action = model.predict(_obs, deterministic=True)[0]
        elif policy == "heat_max":
            action = np.ones(env.action_space.shape, dtype=np.float32)
        elif policy == "zero":
            action = np.zeros(env.action_space.shape, dtype=np.float32)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        comfort_penalty = float(info.get("comfort_penalty", 0.0))
        total_cost      = float(info.get("energy_cost_usd", 0.0))
        gas_cost        = float(info.get("gas_cost_usd", 0.0))
        elec_cost       = float(info.get("elec_cost_usd", 0.0))
        energy_rate     = (abs(info.get("blower_rate", 0.0))
                           + abs(info.get("ah_conditioning_rate", 0.0))
                           + abs(info.get("boiler_gas_rate", 0.0)))

        energy_penalty_cost  = env.energy_weight * total_cost / max(env._cost_norm, 1e-9)
        energy_penalty_watts = env.energy_weight * energy_rate / max(env._energy_norm, 1.0)

        if prev_action is not None:
            smoothness = 0.05 * float(np.mean(np.abs(action - prev_action)))
        else:
            smoothness = 0.0
        prev_action = action.copy()

        rows.append({
            "comfort_penalty":      comfort_penalty,
            "energy_penalty_cost":  energy_penalty_cost,
            "energy_penalty_watts": energy_penalty_watts,
            "smoothness_penalty":   smoothness,
            "gas_cost_usd":         gas_cost,
            "elec_cost_usd":        elec_cost,
            "total_cost_usd":       total_cost,
            "energy_rate_w":        energy_rate,
            "reward":               float(reward),
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--floorplan", default="office_4room")
    parser.add_argument("--energy_weight", type=float, default=1.0)
    parser.add_argument("--n_episodes", type=int, default=3)
    parser.add_argument("--policy", default="random", choices=["random", "heat_max", "zero", "model"])
    parser.add_argument("--model_dir", default=None, help="Path to result dir to load trained model")
    parser.add_argument("--start_timestamp", default="2022-01-10T00:00:00", help="Cold week to test on")
    parser.add_argument("--weather_csv", default="/user/gent/453/vsc45342/thesis/weather_data/belgium_weather_multiyear.csv")
    args = parser.parse_args()

    # Load model if requested
    model = None
    vec_normalize = None
    if args.model_dir:
        from smart_control_analysis.rl_trainer import BuildingRLTrainer
        config_path = os.path.join(args.model_dir, "run_config.json")
        with open(config_path) as f:
            run_config = json.load(f)
        algo = run_config["algo"]
        trainer = BuildingRLTrainer(building_factory_fn=building_factory)
        trainer.load_model(os.path.join(args.model_dir, f"{algo}_2024_model.zip"), algo)
        vn_path = os.path.join(args.model_dir, f"{algo}_2024_model_vecnormalize.pkl")
        if os.path.exists(vn_path):
            trainer.load_vec_normalize(vn_path)
        model = trainer.model
        vec_normalize = trainer.vec_normalize
        args.policy = "model"
        args.energy_weight = run_config.get("energy_weight", args.energy_weight)
        args.floorplan = run_config.get("floorplan", args.floorplan)

    params = get_base_params().copy()
    params["weather_source"] = "replay"
    params["weather_csv_path"] = args.weather_csv
    params["floorplan"] = args.floorplan
    params["energy_weight"] = args.energy_weight
    params["use_cost_reward"] = True
    params["max_steps"] = int(7 * 24 * 3600 / params["time_step_sec"])
    params["start_timestamp"] = args.start_timestamp

    _, env = building_factory(params, training_mode="step")

    print(f"\n{'='*65}")
    print(f"  Floorplan:     {args.floorplan}  ({env.n_zones} zones)")
    print(f"  Energy weight: {args.energy_weight}")
    print(f"  Policy:        {args.policy}")
    print(f"  Start:         {args.start_timestamp}")
    print(f"  energy_norm:   {env._energy_norm:.1f} W")
    print(f"  cost_norm:     ${env._cost_norm:.4f} / step")
    print(f"{'='*65}\n")

    all_dfs = []
    for ep in range(args.n_episodes):
        df = run_episode(env, policy=args.policy, seed=ep, model=model, vec_normalize=vec_normalize)
        all_dfs.append(df)
    env.close()

    df = pd.concat(all_dfs, ignore_index=True)

    print(f"{'Metric':<35} {'mean':>10} {'max':>10} {'p95':>10}")
    print("-" * 65)
    for col, label in [
        ("comfort_penalty",      "Comfort penalty"),
        ("energy_penalty_cost",  "Energy penalty (cost mode)"),
        ("energy_penalty_watts", "Energy penalty (watts mode)"),
        ("smoothness_penalty",   "Smoothness penalty"),
        ("gas_cost_usd",         "Gas cost (USD/step)"),
        ("elec_cost_usd",        "Elec cost (USD/step)"),
        ("total_cost_usd",       "Total cost (USD/step)"),
        ("energy_rate_w",        "Energy rate (W)"),
    ]:
        print(f"  {label:<33} {df[col].mean():>10.4f} {df[col].max():>10.4f} {df[col].quantile(0.95):>10.4f}")

    print(f"\n  Total reward mean: {df['reward'].mean():.3f}")
    print(f"  Total reward min:  {df['reward'].min():.3f}")

    mean_comfort = df["comfort_penalty"].mean()
    mean_energy  = df["energy_penalty_cost"].mean()
    print(f"\n--- RATIO CHECK (energy_weight={args.energy_weight}) ---")
    if mean_comfort > 1e-6:
        ratio = mean_energy / mean_comfort
        print(f"  energy / comfort = {ratio:.2f}x")
        if ratio < 0.3:
            print("  WARNING: energy penalty too small — agent will ignore energy cost")
        elif ratio > 3.0:
            print("  WARNING: energy penalty too large — agent will ignore comfort")
        else:
            print("  OK: reasonably balanced")
    else:
        print("  comfort_penalty ~ 0 — policy keeps comfort well")

    print(f"\n  Gas fraction of total cost: {df['gas_cost_usd'].mean() / max(df['total_cost_usd'].mean(), 1e-9) * 100:.1f}%")


if __name__ == "__main__":
    main()
