import os
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Any, Callable, Dict, List, Optional, Tuple
from smart_control_analysis.rl_trainer import BuildingRLTrainer
from smart_control_analysis.building_factory import building_factory, get_base_params
from smart_control_analysis.runner import _sample_start_in_period, _run_episode_trace
from smart_control_analysis.baseline_controller import ThermostatBaselineController


def compare_algorithms(
    algos: List[str] = ["sac", "td3", "ddpg"],
    total_timesteps: int = 100000,
    n_seeds: int = 3,
    output_dir: str = "./results",
) -> Dict[str, Dict]:
    """
    Train and compare multiple algorithms across multiple seeds.

    Returns a dict mapping algo_name -> {seed_idx -> trainer}
    """
    results = {}

    for algo in algos:
        print(f"\n{'='*60}")
        print(f"Training {algo.upper()}")
        print(f"{'='*60}")
        results[algo] = {}

        for seed_idx in range(n_seeds):
            print(f"\nSeed {seed_idx + 1}/{n_seeds}")
            trainer = BuildingRLTrainer(building_factory_fn=building_factory)
            trainer.train(
                algo=algo,
                total_timesteps=total_timesteps,
                verbose=0,  # Less spam
            )
            results[algo][seed_idx] = trainer

            # Save intermediate results
            trainer.save_results(os.path.join(output_dir, algo))
            trainer.save_model(os.path.join(output_dir, algo, f"{algo}_seed{seed_idx}"))
            trainer.close()

    return results


def plot_comparison(
    results: Dict[str, Dict],
    output_dir: str = "./results",
    figsize: Tuple[int, int] = (14, 6),
):
    """
    Plot comparison across algorithms and seeds.

    Parameters
    ----------
    results : dict
        Dict from compare_algorithms()
    output_dir : str
        Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Aggregate data per algorithm
    algo_data = {}
    for algo, seeds_dict in results.items():
        rewards_list = []
        for seed_idx, trainer in seeds_dict.items():
            if trainer.callback:
                rewards_list.append(trainer.callback.episode_rewards)
        algo_data[algo] = rewards_list

    # Plot 1: Mean reward curves with confidence bands
    fig, ax = plt.subplots(figsize=figsize)
    colors = {"sac": "blue", "td3": "green", "ddpg": "red"}

    for algo, rewards_list in algo_data.items():
        if not rewards_list:
            continue

        # Stack rewards: (n_seeds, n_episodes)
        rewards_array = np.array(rewards_list)
        mean_rewards = np.mean(rewards_array, axis=0)
        std_rewards = np.std(rewards_array, axis=0)

        x = np.arange(len(mean_rewards))
        ax.plot(x, mean_rewards, color=colors.get(algo, "black"), linewidth=2, label=algo.upper())
        ax.fill_between(
            x,
            mean_rewards - std_rewards,
            mean_rewards + std_rewards,
            color=colors.get(algo, "black"),
            alpha=0.2
        )

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Episode Reward", fontsize=12)
    ax.set_title("Algorithm Comparison: Episode Rewards", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    plot_path = os.path.join(output_dir, "comparison_rewards.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Comparison plot saved to {plot_path}")

    # Plot 2: Box plot of final rewards
    fig, ax = plt.subplots(figsize=(10, 6))
    box_data = []
    labels = []

    for algo in sorted(algo_data.keys()):
        rewards_list = algo_data[algo]
        if rewards_list:
            final_rewards = [r[-1] if len(r) > 0 else 0 for r in rewards_list]
            box_data.append(final_rewards)
            labels.append(algo.upper())

    ax.boxplot(box_data, labels=labels)
    ax.set_ylabel("Final Episode Reward", fontsize=12)
    ax.set_title("Algorithm Comparison: Final Reward Distribution", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    plot_path = os.path.join(output_dir, "comparison_boxplot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Box plot saved to {plot_path}")

def _plot_comparison_boxplots(
    df: pd.DataFrame,
    fig_path: Optional[str] = None,
    title: str = "",
) -> plt.Figure:
    """
    2x3 grid box plot comparing RL vs Baseline per episode:
      Top row (energy):
        1. Energy cost RL vs Baseline
        2. Cost savings (baseline - RL)
      Bottom row (comfort):
        3. Discomfort (°C·h / episode)
        4. Time outside comfort (%)
        5. Max temperature deviation (°C)
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    ax_cost, ax_savings, ax_empty = axes[0]
    ax_comfort, ax_pct, ax_max = axes[1]

    rl_color, bl_color, savings_color = "#4c9be8", "#f4a261", "#2ecc71"

    # hide unused top-right cell
    ax_empty.set_visible(False)

    # --- Energy cost ---
    cost_data = [df["rl_energy_cost_usd"].to_numpy(), df["baseline_energy_cost_usd"].to_numpy()]
    bp1 = ax_cost.boxplot(cost_data, labels=["RL", "Baseline"], patch_artist=True, widths=0.5)
    bp1["boxes"][0].set_facecolor(rl_color)
    bp1["boxes"][1].set_facecolor(bl_color)
    ax_cost.set_ylabel("Energy cost (USD / episode)")
    ax_cost.set_title("Energy cost")
    ax_cost.grid(axis="y", alpha=0.3)

    # --- Cost savings (baseline - RL): positive = RL saved money ---
    savings = df["baseline_energy_cost_usd"].to_numpy() - df["rl_energy_cost_usd"].to_numpy()
    bp2 = ax_savings.boxplot([savings], labels=["RL vs Baseline"], patch_artist=True, widths=0.5)
    bp2["boxes"][0].set_facecolor(savings_color)
    ax_savings.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_savings.set_ylabel("Cost saved (USD / episode)")
    ax_savings.set_title("Energy savings")
    ax_savings.grid(axis="y", alpha=0.3)
    median_baseline = float(np.median(df["baseline_energy_cost_usd"].to_numpy()))
    median_savings = float(np.median(savings))
    if median_baseline > 0:
        pct_saved = 100.0 * median_savings / median_baseline
        ax_savings.annotate(
            f"Median: {pct_saved:+.1f}%",
            xy=(1, median_savings),
            xytext=(1.3, median_savings),
            fontsize=9,
            color="darkgreen" if pct_saved >= 0 else "red",
            va="center",
        )

    # --- Discomfort degree-hours ---
    dis_data = [df["rl_discomfort_deg_h"].to_numpy(), df["baseline_discomfort_deg_h"].to_numpy()]
    bp3 = ax_comfort.boxplot(dis_data, labels=["RL", "Baseline"], patch_artist=True, widths=0.5)
    bp3["boxes"][0].set_facecolor(rl_color)
    bp3["boxes"][1].set_facecolor(bl_color)
    ax_comfort.set_ylabel("Discomfort (°C·h / episode)")
    ax_comfort.set_title("Thermal discomfort")
    ax_comfort.grid(axis="y", alpha=0.3)

    # --- % timesteps outside comfort ---
    if "rl_pct_outside_comfort" in df.columns:
        pct_data = [df["rl_pct_outside_comfort"].to_numpy(), df["baseline_pct_outside_comfort"].to_numpy()]
        bp4 = ax_pct.boxplot(pct_data, labels=["RL", "Baseline"], patch_artist=True, widths=0.5)
        bp4["boxes"][0].set_facecolor(rl_color)
        bp4["boxes"][1].set_facecolor(bl_color)
    ax_pct.set_ylabel("Time outside comfort (%)")
    ax_pct.set_title("% Outside comfort")
    ax_pct.grid(axis="y", alpha=0.3)

    # --- Max temperature deviation ---
    if "rl_max_temp_deviation_c" in df.columns:
        max_data = [df["rl_max_temp_deviation_c"].to_numpy(), df["baseline_max_temp_deviation_c"].to_numpy()]
        bp5 = ax_max.boxplot(max_data, labels=["RL", "Baseline"], patch_artist=True, widths=0.5)
        bp5["boxes"][0].set_facecolor(rl_color)
        bp5["boxes"][1].set_facecolor(bl_color)
    ax_max.set_ylabel("Max deviation (°C)")
    ax_max.set_title("Worst-case violation")
    ax_max.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if fig_path is not None:
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Saved comparison boxplots: {fig_path}")

    return fig


def _plot_energy_prices(
    df: pd.DataFrame,
    title: str = "Energy Prices",
    fig_path: Optional[str] = None,
) -> plt.Figure:
    """Plot electricity and gas spot prices over an episode."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    fig.suptitle(title, fontsize=13)

    if "elec_price_eur_per_mwh" in df.columns:
        ax1.plot(df["timestamp"], df["elec_price_eur_per_mwh"], color="tab:blue", linewidth=1.0)
        ax1.set_ylabel("Electricity\n(EUR/MWh)")
        ax1.grid(True, alpha=0.3)

    if "gas_price_eur_per_mwh" in df.columns:
        ax2.plot(df["timestamp"], df["gas_price_eur_per_mwh"], color="tab:orange", linewidth=1.0)
        ax2.set_ylabel("Gas ZTP\n(EUR/MWh)")
        ax2.grid(True, alpha=0.3)

    ax2.set_xlabel("Time")
    fig.autofmt_xdate()
    plt.tight_layout()

    if fig_path is not None:
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Saved energy prices plot: {fig_path}")

    plt.close(fig)
    return fig


def _compare_period_rl_vs_baseline(
    trainer: BuildingRLTrainer,
    params_template: Dict[str, Any],
    period_name: str,
    period_start: str,
    period_end: str,
    output_dir: str,
    n_episodes: int,
    episode_days: int,
    deterministic: bool,
    seed: int,
    training_mode: str,
    n_plot_episodes: int,
    verbose: bool = True,
    save_traces: bool = False,
    baseline_night_off: bool = False,
    algo_label: str = "RL",
    baseline_cache: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    rows = []
    period_dir = os.path.join(output_dir, period_name)
    os.makedirs(period_dir, exist_ok=True)

    iterator = tqdm(
        range(n_episodes),
        desc=f"[compare:{period_name}]",
        leave=False,
        disable=not verbose,
    )

    for ep in iterator:
        p = params_template.copy()
        p["start_timestamp"] = _sample_start_in_period(
            rng=rng,
            start=period_start,
            end=period_end,
            episode_days=episode_days,
            time_zone=p["time_zone"],
            start_hour=0,  # always start at midnight — 8h warmup before working hours
        ).isoformat()

        # Start building at comfort band midpoint so there is no cold-start penalty spike
        comfort_band_k = p.get("comfort_band_k", (294.15, 295.15))
        p["initial_temp_celsius"] = 0.5 * (comfort_band_k[0] + comfort_band_k[1]) - 273.15

        max_steps = int(episode_days * 24 * 3600 / p["time_step_sec"])
        p["max_steps"] = max_steps

        # RL episode — normalize obs using training VecNormalize stats before predict
        env_rl = trainer.create_env(params=p, training_mode=training_mode)
        _vn = trainer.vec_normalize  # may be None if no VecNormalize was used
        def _rl_policy(obs, env, _det=deterministic, _vn=_vn):
            if _vn is not None:
                obs = _vn.normalize_obs(obs)
            return trainer.model.predict(obs, deterministic=_det)[0]
        _action_design = getattr(env_rl, "action_design", "reheat_per_zone")
        rl_df, rl_m, comfort_band_c = _run_episode_trace(
            env_rl,
            _rl_policy,
            "rl",
            seed=ep,
            action_design=_action_design,
        )
        env_rl.close()

        # Baseline episode — use cache if available to avoid re-simulating identical episodes
        ts_key = p["start_timestamp"]
        if baseline_cache is not None and ts_key in baseline_cache:
            b_m = baseline_cache[ts_key]["metrics"]
            b_df = baseline_cache[ts_key]["df"]
            b_pct_outside = baseline_cache[ts_key]["pct_outside"]
            b_max_dev = baseline_cache[ts_key]["max_dev"]
            _skip_baseline_stats = True
        else:
            env_b = trainer.create_env(params=p, training_mode=training_mode)
            baseline = ThermostatBaselineController(
                comfort_band_k=env_b.comfort_band_k,
                working_hours=env_b.working_hours,
                night_setback_k=getattr(env_b, "night_setback_k", 0.0),
                night_off=baseline_night_off,
            )
            b_df, b_m, _ = _run_episode_trace(
                env_b,
                lambda obs, env: baseline.get_action(obs, env),
                "baseline",
                seed=ep,
                action_design=_action_design,
            )
            env_b.close()
            _skip_baseline_stats = False

        if save_traces:
            rl_df.to_csv(os.path.join(period_dir, f"episode_{ep:02d}_rl_trace.csv"), index=False)
            b_df.to_csv(os.path.join(period_dir, f"episode_{ep:02d}_baseline_trace.csv"), index=False)

        if ep < n_plot_episodes:
            _plot_episode_trace_6panel(
                rl_df, comfort_band_c,
                title=f"{period_name.upper()} Episode {ep} - RL",
                fig_path=os.path.join(period_dir, f"episode_{ep:02d}_rl_plot.png"),
                action_design=_action_design,
            )
            _plot_episode_trace_6panel(
                b_df, comfort_band_c,
                title=f"{period_name.upper()} Episode {ep} - Baseline",
                fig_path=os.path.join(period_dir, f"episode_{ep:02d}_baseline_plot.png"),
                action_design=_action_design,
            )
            _plot_energy_prices(
                rl_df,
                title=f"{period_name.upper()} Episode {ep} - Energy Prices",
                fig_path=os.path.join(period_dir, f"episode_{ep:02d}_prices_plot.png"),
            )

        def _comfort_stats(trace_df):
            """Returns (pct_outside, max_deviation_c) for working hours only, per zone."""
            if trace_df.empty:
                return 0.0, 0.0
            if "comfort_low_c" not in trace_df.columns:
                return 0.0, 0.0

            # Filter to daytime (working hours) only:
            # daytime rows have comfort_low_c at its max value (no night setback applied)
            day_low = trace_df["comfort_low_c"].max()
            day_mask = trace_df["comfort_low_c"] >= day_low - 0.01
            df_day = trace_df[day_mask]
            if df_day.empty:
                return 0.0, 0.0

            low  = df_day["comfort_low_c"].to_numpy()
            high = df_day["comfort_high_c"].to_numpy()
            zone_cols = sorted([c for c in df_day.columns if c.startswith("zone_temp_c_")])

            if zone_cols:
                all_violations = np.stack([
                    np.maximum(low - df_day[col].to_numpy(), 0.0) +
                    np.maximum(df_day[col].to_numpy() - high, 0.0)
                    for col in zone_cols
                ])  # shape: (n_zones, n_steps)
                pct_outside = float((all_violations > 0).mean() * 100.0)
                max_dev = float(all_violations.max())
            else:
                t = df_day["room_temp_c"].to_numpy()
                violation = np.maximum(low - t, 0.0) + np.maximum(t - high, 0.0)
                pct_outside = float((violation > 0).mean() * 100.0)
                max_dev = float(violation.max())
            return pct_outside, max_dev

        rl_pct_outside, rl_max_dev = _comfort_stats(rl_df)
        if not _skip_baseline_stats:
            b_pct_outside, b_max_dev = _comfort_stats(b_df)
            if baseline_cache is not None:
                baseline_cache[ts_key] = {
                    "metrics": b_m,
                    "df": b_df,
                    "pct_outside": b_pct_outside,
                    "max_dev": b_max_dev,
                }

        row = {
            "episode": ep,
            "start_timestamp": p["start_timestamp"],
            "rl_reward": rl_m["reward"],
            "baseline_reward": b_m["reward"],
            "reward_diff_rl_minus_baseline": rl_m["reward"] - b_m["reward"],
            "rl_comfort_penalty": rl_m["comfort_penalty"],
            "baseline_comfort_penalty": b_m["comfort_penalty"],
            "rl_energy": rl_m["energy"],
            "baseline_energy": b_m["energy"],
            "rl_energy_cost_usd": rl_m["energy_cost_usd"],
            "baseline_energy_cost_usd": b_m["energy_cost_usd"],
            "rl_discomfort_deg_h": rl_m["discomfort_deg_h"],
            "baseline_discomfort_deg_h": b_m["discomfort_deg_h"],
            "rl_pct_outside_comfort": rl_pct_outside,
            "baseline_pct_outside_comfort": b_pct_outside,
            "rl_max_temp_deviation_c": rl_max_dev,
            "baseline_max_temp_deviation_c": b_max_dev,
            "episode_length": rl_m["length"],
        }
        rows.append(row)

        if verbose:
            iterator.set_postfix(
                rl=f"{row['rl_reward']:.1f}",
                base=f"{row['baseline_reward']:.1f}",
                diff=f"{row['reward_diff_rl_minus_baseline']:.1f}",
            )

    df = pd.DataFrame(rows)
    summary = {
        "period": period_name,
        "rl_reward_mean": float(df["rl_reward"].mean()),
        "baseline_reward_mean": float(df["baseline_reward"].mean()),
        "reward_gain_mean": float(df["reward_diff_rl_minus_baseline"].mean()),
        "rl_comfort_mean": float(df["rl_comfort_penalty"].mean()),
        "baseline_comfort_mean": float(df["baseline_comfort_penalty"].mean()),
        "rl_energy_mean": float(df["rl_energy"].mean()),
        "baseline_energy_mean": float(df["baseline_energy"].mean()),
        "rl_energy_cost_usd_mean": float(df["rl_energy_cost_usd"].mean()),
        "baseline_energy_cost_usd_mean": float(df["baseline_energy_cost_usd"].mean()),
        "rl_discomfort_deg_h_mean": float(df["rl_discomfort_deg_h"].mean()),
        "baseline_discomfort_deg_h_mean": float(df["baseline_discomfort_deg_h"].mean()),
        "rl_reward_std": float(df["rl_reward"].std(ddof=0)),
        "baseline_reward_std": float(df["baseline_reward"].std(ddof=0)),
    }

    _plot_comparison_boxplots(
        df,
        fig_path=os.path.join(period_dir, f"{period_name}_comparison_boxplots.png"),
        title=f"{period_name.capitalize()} split — {algo_label} vs Baseline",
    )

    df.to_csv(os.path.join(period_dir, f"{period_name}_episode_metrics.csv"), index=False)
    pd.DataFrame([summary]).to_csv(os.path.join(period_dir, f"{period_name}_summary.csv"), index=False)

    if verbose:
        print(
            f"[compare:{period_name}] done | "
            f"RL mean={summary['rl_reward_mean']:.2f}±{summary['rl_reward_std']:.2f}, "
            f"BASE mean={summary['baseline_reward_mean']:.2f}±{summary['baseline_reward_std']:.2f}, "
            f"gain={summary['reward_gain_mean']:.2f}"
        )

    return df, summary


def compare_rl_vs_baseline(
    trainer: BuildingRLTrainer,
    output_dir: str,
    n_episodes: int = 8,
    episode_days: int = 7,
    seed: int = 42,
    deterministic: bool = True,
    training_mode: str = "full",
    n_plot_episodes: int = 2,
    verbose: bool = True,
    save_traces: bool = False,
    val_period_start="2022-10-01",
    val_period_end="2023-03-24",
    test_period_start="2023-12-01",
    test_period_end="2024-03-24",
    baseline_night_off: bool = False,
    baseline_cache: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compare trained RL policy vs thermostat baseline on winter val/test splits.
    Saves per-episode traces, metrics CSVs, and 4-panel plots.
    """
    if trainer.model is None:
        raise ValueError("Trainer has no model. Train or load model first.")

    os.makedirs(output_dir, exist_ok=True)
    base = trainer.base_params.copy()
    algo_label = type(trainer.model).__name__.upper() if trainer.model is not None else "RL"

    val_df, val_summary = _compare_period_rl_vs_baseline(
        trainer=trainer,
        params_template=base,
        period_name="val",
        period_start=val_period_start,
        period_end=val_period_end,
        output_dir=output_dir,
        n_episodes=n_episodes,
        episode_days=episode_days,
        deterministic=deterministic,
        seed=seed,
        training_mode=training_mode,
        n_plot_episodes=n_plot_episodes,
        verbose=verbose,
        save_traces=save_traces,
        baseline_night_off=baseline_night_off,
        algo_label=algo_label,
        baseline_cache=baseline_cache,
    )
    test_df, test_summary = _compare_period_rl_vs_baseline(
        trainer=trainer,
        params_template=base,
        period_name="test",
        period_start=test_period_start,
        period_end=test_period_end,
        output_dir=output_dir,
        n_episodes=n_episodes,
        episode_days=episode_days,
        deterministic=deterministic,
        seed=seed + 1,
        training_mode=training_mode,
        n_plot_episodes=n_plot_episodes,
        verbose=verbose,
        save_traces=save_traces,
        baseline_night_off=baseline_night_off,
        algo_label=algo_label,
        baseline_cache=baseline_cache,
    )

    # Training curve plot
    if (trainer.callback is not None
            and hasattr(trainer.callback, "episode_rewards")
            and len(trainer.callback.episode_rewards) > 0):
        _plot_training_curve(
            timesteps=trainer.callback.timesteps,
            episode_rewards=trainer.callback.episode_rewards,
            val_summary=val_summary,
            test_summary=test_summary,
            fig_path=os.path.join(output_dir, "training_curve.png"),
        )

    return {
        "trainer": trainer,
        "val_results": val_summary,
        "test_results": test_summary,
        "summary": {
            "val_mean_reward": val_summary["rl_reward_mean"],
            "test_mean_reward": test_summary["rl_reward_mean"],
            "val_std_reward": val_summary["rl_reward_std"],
            "test_std_reward": test_summary["rl_reward_std"],
            "val_mean_comfort": val_summary["rl_comfort_mean"],
            "test_mean_comfort": test_summary["rl_comfort_mean"],
            "val_mean_energy": val_summary["rl_energy_mean"],
            "test_mean_energy": test_summary["rl_energy_mean"],
            "val_reward_gain_mean": val_summary["reward_gain_mean"],
            "test_reward_gain_mean": test_summary["reward_gain_mean"],
        },
    }

def _plot_training_curve(
    timesteps: List[int],
    episode_rewards: List[float],
    val_summary: Dict[str, float],
    test_summary: Dict[str, float],
    fig_path: Optional[str] = None,
    window: int = 20,
) -> plt.Figure:
    """Plot episode reward over training timesteps with smoothed trend."""
    fig, ax = plt.subplots(figsize=(14, 5))

    ts = np.array(timesteps)
    rs = np.array(episode_rewards)

    # Raw episode reward (faint)
    ax.plot(ts, rs, color="tab:blue", alpha=0.3, linewidth=0.8, label="Episode reward")

    # Smoothed moving average — window as fraction of total episodes so it scales sensibly
    w = max(5, min(window, len(rs) // 5))
    if len(rs) >= w:
        kernel = np.ones(w) / w
        smoothed = np.convolve(rs, kernel, mode="same")
        # convolve "same" distorts edges — mask first and last w//2 points
        mask = np.ones(len(rs), dtype=bool)
        mask[:w // 2] = False
        mask[-(w // 2):] = False
        ax.plot(ts[mask], smoothed[mask], color="tab:blue", linewidth=2.2,
                label=f"Smoothed (window={w})")

    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode reward")
    ax.set_title("Training curve — episode reward over time")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    plt.tight_layout()

    if fig_path is not None:
        run_id = os.path.basename(os.path.dirname(fig_path))
        fig.text(0.01, 0.002, f"run: {run_id}", fontsize=6, color="gray", va="bottom")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Saved training curve: {fig_path}")

    return fig

def _plot_episode_trace_6panel(
    df: pd.DataFrame,
    comfort_band_c: Tuple[float, float],
    title: str = "",
    fig_path: Optional[str] = None,
    figsize: Tuple[int, int] = (18, 24),
    action_design: str = "reheat_per_zone",
) -> plt.Figure:
    """
    Plot a 7-panel episode trace:
      1. Room temperature + comfort band
      2. Outside temperature
      3. Comfort penalty
      4. Energy rate (W)
      5. Energy cost (USD/step): gas, electricity, total
      6. Cumulative reward
      7. Actions
    """
    comfort_low_c, comfort_high_c = comfort_band_c

    fig, axes = plt.subplots(
        7, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.2, 1.2, 1.4, 1.4, 1.4, 1.5]},
    )
    ax_room, ax_out, ax_pen, ax_energy, ax_cost, ax_reward, ax_act = axes

    # --- 1) Room temperature ---
    # Use per-step comfort band columns if available (supports night setback), else fall back to static
    if "comfort_low_c" in df.columns and df["comfort_low_c"].notna().any():
        cb_low  = df["comfort_low_c"]
        cb_high = df["comfort_high_c"]
    else:
        cb_low  = pd.Series([comfort_low_c]  * len(df), index=df.index)
        cb_high = pd.Series([comfort_high_c] * len(df), index=df.index)

    # Comfort band (green shading)
    ax_room.fill_between(df["timestamp"], cb_low, cb_high,
                         alpha=0.12, color="green", label="Comfort band")
    ax_room.plot(df["timestamp"], cb_low,  color="green", linestyle="--", linewidth=1.0,
                 label="Comfort low (night setback)")
    ax_room.plot(df["timestamp"], cb_high, color="red",   linestyle="--", linewidth=1.0,
                 label="Comfort high")

    # Zone temperature band (min–max shading) + mean line
    if "room_temp_min_c" in df.columns and "room_temp_max_c" in df.columns:
        ax_room.fill_between(df["timestamp"], df["room_temp_min_c"], df["room_temp_max_c"],
                             alpha=0.18, color="tab:blue", label="Zone temp range")
        ax_room.plot(df["timestamp"], df["room_temp_min_c"],
                     color="tab:blue", linewidth=0.7, linestyle=":", alpha=0.6)
        ax_room.plot(df["timestamp"], df["room_temp_max_c"],
                     color="tab:blue", linewidth=0.7, linestyle=":", alpha=0.6)
    ax_room.plot(df["timestamp"], df["room_temp_c"],
                 label="Zone temp mean", linewidth=1.8, color="tab:blue")

    ax_room.set_ylabel("Room temp (°C)")
    ax_room.set_title(title, fontsize=13, fontweight="bold")
    ax_room.grid(alpha=0.25)
    ax_room.legend(loc="upper right", fontsize=8)

    # --- 2) Outside temperature ---
    ax_out.plot(df["timestamp"], df["outside_temp_c"],
                label="Outside temp (°C)", linewidth=1.8, color="tab:orange")
    ax_out.set_ylabel("Outside (°C)")
    ax_out.grid(alpha=0.25)
    ax_out.legend(loc="upper right", fontsize=8)

    # --- 3) Comfort penalty ---
    ax_pen.fill_between(df["timestamp"], 0, df["comfort_penalty"],
                        alpha=0.35, color="crimson")
    ax_pen.plot(df["timestamp"], df["comfort_penalty"],
                color="crimson", linewidth=1.2, label="Comfort penalty")
    ax_pen.set_ylabel("Comfort penalty")
    ax_pen.grid(alpha=0.25)
    ax_pen.legend(loc="upper right", fontsize=8)

    # --- 4) Energy rate ---
    if "energy_rate" in df.columns:
        ax_energy.fill_between(df["timestamp"], 0, df["energy_rate"],
                               alpha=0.3, color="tab:purple")
        ax_energy.plot(df["timestamp"], df["energy_rate"],
                       color="tab:purple", linewidth=1.2, label="Energy rate (W)")
        ax_energy.set_ylabel("Energy (W)")
    else:
        ax_energy.text(0.5, 0.5, "No energy_rate column",
                       ha="center", va="center", transform=ax_energy.transAxes)
        ax_energy.set_ylabel("Energy (W)")
    ax_energy.grid(alpha=0.25)
    ax_energy.legend(loc="upper right", fontsize=8)

    # --- 5) Energy cost (USD/hr) — 3 separate lines ---
    has_cost = "energy_cost_usd" in df.columns and df["energy_cost_usd"].notna().any()
    if has_cost:
        if len(df) > 1:
            dt_sec = (df["timestamp"].iloc[1] - df["timestamp"].iloc[0]).total_seconds()
        else:
            dt_sec = 60.0
        scale = 3600.0 / dt_sec  # USD/step → USD/hr
        ax_cost.plot(df["timestamp"], df["gas_cost_usd"] * scale,
                     color="tab:orange", linewidth=1.4, label="Gas (USD/hr)")
        ax_cost.plot(df["timestamp"], df["elec_cost_usd"] * scale,
                     color="tab:blue", linewidth=1.4, label="Electricity (USD/hr)")
        ax_cost.plot(df["timestamp"], df["energy_cost_usd"] * scale,
                     color="black", linewidth=1.8, linestyle="--", label="Total (USD/hr)")
        ax_cost.set_ylabel("Energy cost\n(USD/hr)")
    else:
        ax_cost.text(0.5, 0.5, "No cost data", ha="center", va="center",
                     transform=ax_cost.transAxes)
        ax_cost.set_ylabel("Energy cost\n(USD/hr)")
    ax_cost.grid(alpha=0.25)
    ax_cost.legend(loc="upper right", fontsize=8)

    # --- 6) Cumulative + step reward ---
    if "reward" in df.columns:
        cum_reward = df["reward"].cumsum()
        ax_reward.plot(df["timestamp"], cum_reward,
                       color="tab:green", linewidth=2, label="Cumulative reward")
        ax_r2 = ax_reward.twinx()
        ax_r2.plot(df["timestamp"], df["reward"],
                   color="tab:green", linewidth=0.8, alpha=0.4, label="Step reward")
        ax_r2.set_ylabel("Step reward", fontsize=8, color="tab:green")
        ax_r2.tick_params(axis="y", labelcolor="tab:green")
        ax_reward.set_ylabel("Cumul. reward")
        ax_reward.legend(loc="upper left", fontsize=8)
        ax_r2.legend(loc="upper right", fontsize=8)
    else:
        ax_reward.text(0.5, 0.5, "No reward column",
                       ha="center", va="center", transform=ax_reward.transAxes)
        ax_reward.set_ylabel("Reward")
    ax_reward.grid(alpha=0.25)

    # --- 7) Actions — labels depend on action_design ---
    if action_design == "reheat_per_zone":
        action_cols = {
            "action_supply":        "Supply air",
            "action_boiler":        "Boiler",
            "action_damper_shared": "Damper (shared)",
            "action_reheat_mean":   "Reheat mean (per-zone)",
        }
    elif action_design == "damper_per_zone":
        action_cols = {
            "action_supply":        "Supply air",
            "action_boiler":        "Boiler",
            "action_reheat_shared": "Reheat (shared)",
            "action_damper_mean":   "Damper mean (per-zone)",
        }
    else:  # full_per_zone
        action_cols = {
            "action_supply":      "Supply air",
            "action_boiler":      "Boiler",
            "action_reheat_mean": "Reheat mean (per-zone)",
            "action_damper_mean": "Damper mean (per-zone)",
        }
    for col, lbl in action_cols.items():
        if col in df.columns:
            ax_act.step(df["timestamp"], df[col], where="post",
                        label=lbl, linewidth=1.4)
    ax_act.axhline(0, color="black", linewidth=0.6, linestyle=":")
    ax_act.set_ylabel("Action [-1, 1]")
    ax_act.set_xlabel("Timestamp")
    ax_act.set_ylim(-1.15, 1.15)
    ax_act.grid(alpha=0.25)
    ax_act.legend(loc="upper right", ncol=2, fontsize=8)

    # x-axis ticks
    ax_act.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    ax_act.xaxis.set_major_formatter(mdates.DateFormatter("%a %m-%d %H:%M"))
    ax_act.xaxis.set_minor_locator(mdates.HourLocator(interval=3))
    plt.setp(ax_act.get_xticklabels(), rotation=30, ha="right")

    if not df.empty:
        ax_act.set_xlim(
            df["timestamp"].min(),
            df["timestamp"].max() + pd.Timedelta(hours=3),
        )

    plt.tight_layout()

    if fig_path is not None:
        # Add run id as small footer so the plot is self-identifying
        run_id = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(fig_path))))
        episode_file = os.path.basename(fig_path)
        fig.text(0.01, 0.002, f"run: {run_id}  |  {episode_file}",
                 fontsize=6, color="gray", va="bottom", ha="left")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Saved 6-panel plot: {fig_path}")

    return fig


def _shade_working_hours(
    ax: plt.Axes,
    timestamps: pd.Series,
    working_hours: Tuple[float, float] = (8.0, 18.0),
    color: str = "gold",
    alpha: float = 0.10,
    workdays: Tuple[int, ...] = (0, 1, 2, 3, 4),  # Mon-Fri
) -> None:
    """Shade daily working-hour windows on a time axis, excluding weekends by default."""
    if timestamps is None or len(timestamps) == 0:
        return

    t0 = pd.to_datetime(timestamps.min())
    t1 = pd.to_datetime(timestamps.max())
    h_start, h_end = float(working_hours[0]), float(working_hours[1])

    day0 = t0.floor("D")
    day1 = t1.ceil("D")
    days = pd.date_range(day0, day1, freq="D")

    first = True
    for d in days:
        if d.dayofweek not in workdays:
            continue

        ws = d + pd.to_timedelta(h_start, unit="h")
        we = d + pd.to_timedelta(h_end, unit="h")
        if we <= ws:
            we = we + pd.Timedelta(days=1)

        left = max(ws, t0)
        right = min(we, t1)
        if right > left:
            ax.axvspan(
                left,
                right,
                color=color,
                alpha=alpha,
                lw=0,
                label="Working hours" if first else None,
            )
            first = False
