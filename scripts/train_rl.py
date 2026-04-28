"""
RL Training Script for Building HVAC Control
Usage:
    python train_rl.py --mode short        # 5k steps, comfort_only, for testing
    python train_rl.py --mode long         # 100k steps, comfort_only
    python train_rl.py --mode full         # 100k steps, full training_mode
    python train_rl.py --mode long --algo td3  # override algorithm
    python train_rl.py --mode mini --unique_run  # add timestamp to avoid overwrites
"""

import torch; torch.cuda.is_available()  # must be imported first before other imports hijack CUDA detection
import argparse
import os
import sys
import datetime

THESIS_ROOT = os.path.expanduser("~/thesis")

# --- make sure the packages are findable ---
sys.path.insert(0, os.path.join(THESIS_ROOT, "sbsim-analysis"))
sys.path.insert(0, os.path.join(THESIS_ROOT, "sbsim"))

from smart_control_analysis.eval_plotter import compare_rl_vs_baseline
from smart_control_analysis.runner import run_rl_setup
# ─────────────────────────────────────────────
# Presets
# ─────────────────────────────────────────────
PRESETS = {
    "mini": dict(
        total_timesteps=3000,
        chunk_timesteps=1500,
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
        total_timesteps=750_000,
        chunk_timesteps=30_000,   # ~20 episodes per chunk → 25 different start dates per 500k steps
        episode_days=7,
        n_eval_episodes=30,
        training_mode="full",
        eval_training_mode="full",
        policy_kwargs={"net_arch": [256, 256, 256]},
        # Curriculum: learn comfort first, ramp energy, then converge at fixed weight
        # 0-40%: comfort only (ew=1), 40-80%: ramp 1→15, 80-100%: constant 15
        #curriculum=[(0.3, 1.0), (0.7, 12.5), (1.0, 12.5)],
        description="Full training with energy penalty — 500k steps, 7-day episodes, 10min timestep",
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
        total_timesteps=1_000_000,
        chunk_timesteps=50_000,   # ~50 full episodes per chunk (1 episode = 1,008 steps at 600s/7days)
        episode_days=7,
        n_eval_episodes=10,
        training_mode="full",
        eval_training_mode="full",
        description="Full training with energy penalty — 1M steps, 7-day episodes, 60s timestep",
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
    # CPU-optimized: train_freq=10 batches gradient updates to reduce Python overhead.
    # batch_size=1024 uses CPU BLAS more efficiently.
    # net_arch=[64,64] matches the small obs space (28 dims) and cuts linear op time ~16x vs [256,256].
    "long_cpu": dict(
        total_timesteps=1_000_000,
        chunk_timesteps=50_000,
        episode_days=7,
        n_eval_episodes=10,
        training_mode="full",
        eval_training_mode="full",
        train_freq=10,
        gradient_steps=10,
        batch_size=1024,
        policy_kwargs={"net_arch": [64, 64]},
        description="1M steps CPU-optimised",
    ),
    "mega_cpu": dict(
        total_timesteps=10_000_000,
        chunk_timesteps=500_000,
        episode_days=7,
        n_eval_episodes=3,
        training_mode="full",
        eval_training_mode="full",
        train_freq=10,
        gradient_steps=10,
        batch_size=1024,
        policy_kwargs={"net_arch": [64, 64]},
        description="10M steps CPU-optimised",
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
        choices=["mini", "short", "long", "long_eval1", "full", "full_eval1", "always_occ1", "long_cpu", "mega_cpu"],
        default="short",
        help="Training preset to use.",
    )
    parser.add_argument(
        "--algo",
        choices=["sac", "td3", "tqc", "ppo", "ddpg", "crossq"],
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
        "--floorplan",
        choices=["single_room", "office_4room", "corporate_floor", "headquarters_floor"],
        default="single_room",
        help="Building floorplan to use.",
    )
    parser.add_argument(
        "--energy_weight",
        type=float,
        default=1.0,
        help="Comfort/energy trade-off: 0.5=comfort-focused, 1.0=balanced, 2.0=energy-focused.",
    )
    parser.add_argument(
        "--action_design",
        choices=["reheat_per_zone", "damper_per_zone", "full_per_zone"],
        default="reheat_per_zone",
        help="Action space design: reheat_per_zone (cold), damper_per_zone (warm), full_per_zone (both).",
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=None,
        help="Override preset total_timesteps.",
    )
    parser.add_argument(
        "--no_compare",
        action="store_true",
        help="Skip RL vs baseline comparison after training.",
    )
    parser.add_argument(
        "--no_val",
        action="store_true",
        help="Skip validation period in comparison, only run test.",
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
        ew_str = f"_ew{str(args.energy_weight).replace('.', '')}"
        base_dir = f"{args.mode}_{args.algo}_seed{args.seed}{ew_str}"
        if args.unique_run:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            base_dir = f"{base_dir}_{ts}"
        output_dir = os.path.join(THESIS_ROOT, "results", base_dir)
    else:
        output_dir = os.path.expanduser(args.output_dir)

    os.makedirs(output_dir, exist_ok=True)

    # Save full run config immediately so it's available even if the run crashes
    import json
    run_config = {
        "mode":         args.mode,
        "algo":         args.algo,
        "seed":         args.seed,
        "floorplan":     args.floorplan,
        "energy_weight": args.energy_weight,
        "action_design": args.action_design,
        "weather_csv":  args.weather_csv,
        "output_dir":   output_dir,
        "wandb_project": args.wandb_project,
        **{k: v for k, v in preset.items() if k != "description"},
        "total_timesteps": args.total_timesteps or preset["total_timesteps"],
        "description":  preset["description"],
        "started_at":   datetime.datetime.now().isoformat(),
    }
    with open(os.path.join(output_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    print("=" * 60)
    print(f"  Mode:        {args.mode}")
    print(f"  Description: {preset['description']}")
    print(f"  Algorithm:   {args.algo.upper()}")
    print(f"  Seed:        {args.seed}")
    print(f"  Output dir:  {output_dir}")
    print(f"  Weather:     {args.weather_csv}")
    print(f"  Floorplan:   {args.floorplan}")
    print(f"  Energy weight: {args.energy_weight}")
    print(f"  Action design: {args.action_design}")
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
            # Save run ID so plot_convergence.py can look it up later
            run_config["wandb_run_id"] = wandb_run.id
            run_config["wandb_run_name"] = wandb_run.name
            with open(os.path.join(output_dir, "run_config.json"), "w") as f:
                json.dump(run_config, f, indent=2)
        except Exception as e:
            print(f"W&B init failed (continuing without it): {e}")
            wandb_run = None

    # ── Training ──
    print("\n--- TRAINING ---")
    # Extract known run_rl_setup params from preset; remaining keys are algo kwargs
    _known = {"total_timesteps", "chunk_timesteps", "episode_days", "n_eval_episodes",
              "training_mode", "eval_training_mode", "description", "curriculum"}
    _extra_train_kwargs = {k: v for k, v in preset.items() if k not in _known}

    res = run_rl_setup(
        weather_csv_path=weather_csv,
        algo=args.algo,
        total_timesteps=args.total_timesteps or preset["total_timesteps"],
        chunk_timesteps=preset["chunk_timesteps"],
        episode_days=preset["episode_days"],
        seed=args.seed,
        output_dir=output_dir,
        training_mode=preset["training_mode"],
        eval_training_mode=preset["eval_training_mode"],
        n_eval_episodes=min(2, preset["n_eval_episodes"]),
        n_envs=args.n_envs,
        floorplan=args.floorplan,
        energy_weight=args.energy_weight,
        action_design=args.action_design,
        wandb_run=wandb_run,
        wandb_finish=False,  # keep run alive for compare logging
        train_period_end="2023-03-31",  # include 2022-2023 winter (val period no longer used)
        curriculum=preset.get("curriculum"),
        **_extra_train_kwargs,
    )

    print("\nTRAIN SUMMARY:")
    for k, v in res["summary"].items():
        print(f"  {k}: {v}")

    # ── Comparison ──
    if not args.no_compare:
        print("\n--- RL vs BASELINE COMPARISON ---")
        compare_dir = os.path.join(output_dir, "compare")
        if args.no_val:
            from smart_control_analysis.eval_plotter import _compare_period_rl_vs_baseline
            os.makedirs(compare_dir, exist_ok=True)
            test_df, test_summary = _compare_period_rl_vs_baseline(
                trainer=res["trainer"],
                params_template=res["trainer"].base_params.copy(),
                period_name="test",
                period_start="2023-12-01",
                period_end="2024-03-24",
                output_dir=compare_dir,
                n_episodes=preset["n_eval_episodes"],
                episode_days=preset["episode_days"],
                deterministic=True,
                seed=args.seed + 99,
                training_mode=preset["eval_training_mode"],
                n_plot_episodes=2 if args.mode != "short" else 1,
                verbose=True,
            )
            compare_results = {"test_results": test_summary, "summary": test_summary}
        else:
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


    # ── Finish ──
    if wandb_run is not None:
        import wandb
        wandb.finish()

    run_config["finished_at"] = datetime.datetime.now().isoformat()
    with open(os.path.join(output_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    print(f"\nDone. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
