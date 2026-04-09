#!/usr/bin/env python3
"""Re-run compare_rl_vs_baseline on a previously trained model.

Usage:
    python scripts/reeval.py /path/to/result_dir
    python scripts/reeval.py /path/to/result_dir --n_episodes 20 --n_plot_episodes 5
    python scripts/reeval.py /path/to/result_dir --val_start 2022-11-01 --val_end 2023-02-28
    python scripts/reeval.py /path/to/result_dir --save_traces
"""
# import torch  # noqa: F401  — must be first for CUDA init

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from smart_control_analysis.building_factory import building_factory, get_base_params
from smart_control_analysis.rl_trainer import BuildingRLTrainer, compare_rl_vs_baseline


def main():
    parser = argparse.ArgumentParser(description="Re-evaluate a trained RL model")
    parser.add_argument("result_dir", help="Path to the result directory (contains run_config.json)")
    parser.add_argument("--output_dir", default=None, help="Output dir for comparison (default: <result_dir>/compare_reeval)")
    parser.add_argument("--n_episodes", type=int, default=10)
    parser.add_argument("--n_plot_episodes", type=int, default=1)
    parser.add_argument("--episode_days", type=int, default=None, help="Override episode length (default: from run_config)")
    parser.add_argument("--seed", type=int, default=141)
    parser.add_argument("--training_mode", default=None, help="Override eval training mode")
    parser.add_argument("--save_traces", action="store_true", help="Save per-step CSV traces")
    parser.add_argument("--no_val", action="store_true", help="Skip validation period, only run test")
    parser.add_argument("--val_start", default="2022-10-01")
    parser.add_argument("--val_end", default="2023-03-24")
    parser.add_argument("--test_start", default="2023-10-01")
    parser.add_argument("--test_end", default="2024-03-24")
    args = parser.parse_args()

    # Load run config
    config_path = os.path.join(args.result_dir, "run_config.json")
    if not os.path.exists(config_path):
        print(f"ERROR: run_config.json not found in {args.result_dir}")
        sys.exit(1)
    with open(config_path) as f:
        run_config = json.load(f)

    print(f"Re-evaluating: {args.result_dir}")
    print(f"  algo: {run_config['algo']}")
    print(f"  floorplan: {run_config['floorplan']}")
    print(f"  energy_weight: {run_config['energy_weight']}")

    # Reconstruct base params
    base = get_base_params().copy()
    base["weather_source"] = "replay"
    base["weather_csv_path"] = run_config["weather_csv"]
    base["time_zone"] = "America/Los_Angeles"
    base["time_step_sec"] = int(base.get("time_step_sec", 300))
    base["floorplan"] = run_config.get("floorplan", "single_room")
    base["energy_weight"] = run_config.get("energy_weight", 2.0)
    base["action_design"] = run_config.get("action_design", "reheat_per_zone")

    episode_days = args.episode_days or run_config.get("episode_days", 7)
    base["max_steps"] = int(episode_days * 24 * 3600 / base["time_step_sec"])

    training_mode = args.training_mode or run_config.get("eval_training_mode", run_config.get("training_mode", "full"))

    # Create trainer and load model
    trainer = BuildingRLTrainer(
        building_factory_fn=building_factory,
        base_params=base,
        default_factory_kwargs={"training_mode": training_mode},
    )

    algo = run_config["algo"]
    model_path = os.path.join(args.result_dir, f"{algo}_2024_model.zip")
    if not os.path.exists(model_path):
        model_path = os.path.join(args.result_dir, f"{algo}_2024_model")
    trainer.load_model(model_path, algo)

    vn_path = os.path.join(args.result_dir, f"{algo}_2024_model_vecnormalize.pkl")
    if os.path.exists(vn_path):
        trainer.load_vec_normalize(vn_path)

    # Run comparison
    output_dir = args.output_dir or os.path.join(args.result_dir, "compare_reeval")
    print(f"\nOutput: {output_dir}")
    print(f"  episodes: {args.n_episodes}, days: {episode_days}, seed: {args.seed}")
    if not args.no_val:
        print(f"  val: {args.val_start} -> {args.val_end}")
    print(f"  test: {args.test_start} -> {args.test_end}")
    print()

    if args.no_val:
        from smart_control_analysis.rl_trainer import _compare_period_rl_vs_baseline
        os.makedirs(output_dir, exist_ok=True)
        base = trainer.base_params.copy()
        test_df, test_summary = _compare_period_rl_vs_baseline(
            trainer=trainer,
            params_template=base,
            period_name="test",
            period_start=args.test_start,
            period_end=args.test_end,
            output_dir=output_dir,
            n_episodes=args.n_episodes,
            episode_days=episode_days,
            deterministic=True,
            seed=args.seed,
            training_mode=training_mode,
            n_plot_episodes=args.n_plot_episodes,
            verbose=True,
            save_traces=args.save_traces,
        )
        results = {"test_results": test_summary}
    else:
        results = compare_rl_vs_baseline(
            trainer=trainer,
            output_dir=output_dir,
            n_episodes=args.n_episodes,
            episode_days=episode_days,
            seed=args.seed,
            deterministic=True,
            training_mode=training_mode,
            n_plot_episodes=args.n_plot_episodes,
            verbose=True,
            save_traces=args.save_traces,
            val_period_start=args.val_start,
            val_period_end=args.val_end,
            test_period_start=args.test_start,
            test_period_end=args.test_end,
        )

    for section in ("val_results", "test_results", "summary"):
        summary = results.get(section, {})
        if not summary:
            continue
        print(f"\n  [{section}]")
        for k, v in summary.items():
            print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    print(f"\nDone. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
