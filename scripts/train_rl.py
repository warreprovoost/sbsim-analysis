"""
RL Training Script for Building HVAC Control
Usage:
    python train_rl.py --mode short        # 5k steps, comfort_only, for testing
    python train_rl.py --mode long         # 100k steps, comfort_only
    python train_rl.py --mode full         # 100k steps, full training_mode
    python train_rl.py --mode long --algo td3  # override algorithm
    python train_rl.py --mode mini --unique_run  # add timestamp to avoid overwrites
"""

import argparse
import os
import sys
import datetime

THESIS_ROOT = os.path.expanduser("~/thesis")

# --- make sure the packages are findable ---
sys.path.insert(0, os.path.join(THESIS_ROOT, "sbsim-analysis"))
sys.path.insert(0, os.path.join(THESIS_ROOT, "sbsim"))

from smart_control_analysis.rl_trainer import run_rl_setup, compare_rl_vs_baseline

# ─────────────────────────────────────────────
# Presets
# ─────────────────────────────────────────────
PRESETS = {
    "mini": dict(
        total_timesteps=6000,
        chunk_timesteps=3000,
        episode_days=7,
        n_eval_episodes=1,
        training_mode="full",
        eval_training_mode="full",
        description="mini test",
    ),
    "short": dict(
        total_timesteps=5_000,
        chunk_timesteps=1_000,
        episode_days=3,
        n_eval_episodes=2,
        training_mode="full",
        eval_training_mode="fulll",
        description="Quick smoke test — 5k steps, full, 3-day episodes",
    ),
    "long": dict(
        total_timesteps=100_000,
        chunk_timesteps=5_000,
        episode_days=7,
        n_eval_episodes=5,
        training_mode="full",
        eval_training_mode="full",
        description="Full comfort training — 100k steps, 7-day episodes",
    ),
    "long_eval1": dict(
        total_timesteps=100_000,
        chunk_timesteps=5_000,
        episode_days=7,
        n_eval_episodes=1,
        training_mode="full",
        eval_training_mode="full",
        description="Full comfort training — 100k steps, 7-day episodes, 1 eval episode",
    ),
    "full": dict(
        total_timesteps=500_000,
        chunk_timesteps=10_000,
        episode_days=7,
        n_eval_episodes=2,
        training_mode="full",
        eval_training_mode="full",
        description="Full training with energy penalty — 500k steps, 7-day episodes, 60s timestep",
    ),
    "full_eval1": dict(
        total_timesteps=500_000,
        chunk_timesteps=5_000,
        episode_days=7,
        n_eval_episodes=5,
        training_mode="full",
        eval_training_mode="full",
        description="Full training with energy penalty — 100k steps, 7-day episodes, 1 eval episode",
    ),
    "always_occ1": dict(
        total_timesteps=100_000,
        chunk_timesteps=5_000,
        episode_days=7,
        n_eval_episodes=1,
        training_mode="always_occupied",
        eval_training_mode="always_occupied",
        description="7-day full run with constant occupancy=1.0 (no working-hours effect)",
    ),
}

# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Train SAC/TD3 on building HVAC.")
    parser.add_argument(
        "--mode",
        choices=["mini", "short", "long", "long_eval1", "full", "full_eval1", "always_occ1"],
        default="short",
        help="Training preset to use.",
    )
    parser.add_argument(
        "--algo",
        choices=["sac", "td3"],
        default="sac",
        help="RL algorithm.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory. Defaults to results/<mode>_<algo>_<seed>.",
    )
    parser.add_argument(
        "--n_envs",
        type=int,
        default=1,
        help="Number of parallel envs (1=DummyVecEnv, >1=SubprocVecEnv with spawn).",
    )
    parser.add_argument(
        "--weather_csv",
        type=str,
        default="~/thesis/weather_data/oslo_weather_multiyear.csv",
        help="Path to weather CSV.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="building-rl-thesis",
        help="W&B project name. Set to '' to disable W&B.",
    )
    parser.add_argument(
        "--no_compare",
        action="store_true",
        help="Skip RL vs baseline comparison after training.",
    )
    parser.add_argument(
        "--unique_run",
        action="store_true",
        help="Add timestamp to output dir to avoid overwrites.",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    args = parse_args()
    preset = PRESETS[args.mode]

    weather_csv = os.path.expanduser(args.weather_csv)

    if args.output_dir is None:
        base_dir = f"{args.mode}_{args.algo}_seed{args.seed}"
        if args.unique_run:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            base_dir = f"{base_dir}_{ts}"
        output_dir = os.path.join(THESIS_ROOT, "results", base_dir)
    else:
        output_dir = os.path.expanduser(args.output_dir)

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print(f"  Mode:        {args.mode}")
    print(f"  Description: {preset['description']}")
    print(f"  Algorithm:   {args.algo.upper()}")
    print(f"  Seed:        {args.seed}")
    print(f"  Output dir:  {output_dir}")
    print(f"  Weather:     {args.weather_csv}")
    print(f"  W&B project: {args.wandb_project or 'DISABLED'}")
    print("=" * 60)

    # ── W&B ──
    wandb_run = None
    if args.wandb_project:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=f"{args.mode}_{args.algo}_seed{args.seed}",
                tags=[args.mode, args.algo, f"seed{args.seed}"],
                config={
                    "mode": args.mode,
                    "algo": args.algo,
                    "seed": args.seed,
                    **preset,
                },
            )
            print(f"W&B run: {wandb_run.url}")
        except Exception as e:
            print(f"W&B init failed (continuing without it): {e}")
            wandb_run = None

    # ── Training ──
    print("\n--- TRAINING ---")
    res = run_rl_setup(
        weather_csv_path=weather_csv,
        algo=args.algo,
        total_timesteps=preset["total_timesteps"],
        chunk_timesteps=preset["chunk_timesteps"],
        episode_days=preset["episode_days"],
        seed=args.seed,
        output_dir=output_dir,
        training_mode=preset["training_mode"],
        eval_training_mode=preset["eval_training_mode"],
        n_eval_episodes=preset["n_eval_episodes"],
        n_envs=args.n_envs,
        wandb_run=wandb_run,
        wandb_finish=False,  # keep run alive for compare logging
    )

    print("\nTRAIN SUMMARY:")
    for k, v in res["summary"].items():
        print(f"  {k}: {v}")

    # ── Comparison ──
    if not args.no_compare:
        print("\n--- RL vs BASELINE COMPARISON ---")
        compare_dir = os.path.join(output_dir, "compare")
        compare_results = compare_rl_vs_baseline(
            trainer=res["trainer"],
            output_dir=compare_dir,
            n_episodes=preset["n_eval_episodes"],
            episode_days=preset["episode_days"],
            seed=args.seed + 99,
            deterministic=True,
            training_mode=preset["eval_training_mode"],
            n_plot_episodes=2 if args.mode != "short" else 1,
            verbose=True,
        )

        # only print/log dict sections that are actual metric maps
        for section in ("val_results", "test_results", "summary"):
            summary = compare_results.get(section, {})
            print(f"\n  [{section}]")
            for k, v in summary.items():
                print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

        if wandb_run is not None:
            for section in ("val_results", "test_results", "summary"):
                summary = compare_results.get(section, {})
                wandb_run.log({f"compare/{section}/{k}": v for k, v in summary.items()})

    # ── Finish ──
    if wandb_run is not None:
        import wandb
        wandb.finish()

    print(f"\nDone. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
