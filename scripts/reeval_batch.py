#!/usr/bin/env python3
"""Reeval multiple models in one Python process — imports happen once.

Usage:
    python scripts/reeval_batch.py \\
        results/model_1 results/model_2 results/model_3 \\
        --seed 141 --no_val
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from smart_control_analysis.building_factory import building_factory, get_base_params
from smart_control_analysis.rl_trainer import BuildingRLTrainer
from smart_control_analysis.eval_plotter import compare_rl_vs_baseline, _compare_period_rl_vs_baseline


def reeval_one(result_dir, seed, episode_days_override, training_mode_override,
               no_val, val_only, test_start, test_end, val_start, val_end, n_episodes,
               baseline_cache=None):
    config_path = os.path.join(result_dir, "run_config.json")
    if not os.path.exists(config_path):
        print(f"  ERROR: no run_config.json in {result_dir}, skipping")
        return

    with open(config_path) as f:
        run_config = json.load(f)

    print(f"\n{'='*60}")
    print(f"  {result_dir}")
    print(f"  algo={run_config['algo']}  weight={run_config.get('energy_weight')}  seed={seed}")

    base = get_base_params().copy()
    base["weather_source"] = "replay"
    base["weather_csv_path"] = run_config["weather_csv"]
    base["time_zone"] = "Europe/Brussels"
    base["time_step_sec"] = int(base.get("time_step_sec", 300))
    base["floorplan"] = run_config.get("floorplan", "single_room")
    base["energy_weight"] = run_config.get("energy_weight", 2.0)
    base["action_design"] = run_config.get("action_design", "reheat_per_zone")

    episode_days = episode_days_override or run_config.get("episode_days", 7)
    base["max_steps"] = int(episode_days * 24 * 3600 / base["time_step_sec"])

    training_mode = training_mode_override or run_config.get(
        "eval_training_mode", run_config.get("training_mode", "full"))

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

    output_dir = os.path.join(result_dir, "compare_reeval")

    os.makedirs(output_dir, exist_ok=True)

    if val_only:
        _compare_period_rl_vs_baseline(
            trainer=trainer,
            params_template=base,
            period_name="val",
            period_start=val_start,
            period_end=val_end,
            output_dir=output_dir,
            n_episodes=n_episodes,
            episode_days=episode_days,
            deterministic=True,
            seed=seed,
            training_mode=training_mode,
            n_plot_episodes=0,
            verbose=True,
            baseline_cache=baseline_cache,
        )
    elif no_val:
        _compare_period_rl_vs_baseline(
            trainer=trainer,
            params_template=base,
            period_name="test",
            period_start=test_start,
            period_end=test_end,
            output_dir=output_dir,
            n_episodes=n_episodes,
            episode_days=episode_days,
            deterministic=True,
            seed=seed,
            training_mode=training_mode,
            n_plot_episodes=0,
            verbose=True,
            baseline_cache=baseline_cache,
        )
    else:
        compare_rl_vs_baseline(
            trainer=trainer,
            output_dir=output_dir,
            n_episodes=n_episodes,
            episode_days=episode_days,
            seed=seed,
            deterministic=True,
            training_mode=training_mode,
            n_plot_episodes=0,
            verbose=True,
            val_period_start=val_start,
            val_period_end=val_end,
            test_period_start=test_start,
            test_period_end=test_end,
            baseline_cache=baseline_cache,
        )

    print(f"  Saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dirs", nargs="+")
    parser.add_argument("--seed", type=int, default=141)
    parser.add_argument("--n_episodes", type=int, default=30)
    parser.add_argument("--episode_days", type=int, default=None)
    parser.add_argument("--training_mode", default=None)
    parser.add_argument("--no_val", action="store_true")
    parser.add_argument("--val_only", action="store_true")
    parser.add_argument("--val_start", default="2022-10-01")
    parser.add_argument("--val_end", default="2023-03-24")
    parser.add_argument("--test_start", default="2023-12-01")
    parser.add_argument("--test_end", default="2024-03-24")
    args = parser.parse_args()

    # Shared baseline cache — keyed by start_timestamp so identical episodes across
    # models are only simulated once (baseline is deterministic given the same start date)
    baseline_cache = {}

    total = len(args.result_dirs)
    for i, rdir in enumerate(args.result_dirs):
        print(f"\n[{i+1}/{total}] {rdir}")
        try:
            reeval_one(
                result_dir=rdir,
                seed=args.seed,
                episode_days_override=args.episode_days,
                training_mode_override=args.training_mode,
                no_val=args.no_val,
                val_only=args.val_only,
                test_start=args.test_start,
                test_end=args.test_end,
                val_start=args.val_start,
                val_end=args.val_end,
                n_episodes=args.n_episodes,
                baseline_cache=baseline_cache,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    print(f"\n=== Done: {total} models reevaluated ===")


if __name__ == "__main__":
    main()
