import json
import os

import numpy as np
import pandas as pd

from typing import Any, Callable, Dict, List, Optional, Tuple
from smart_control_analysis.rl_trainer import BuildingRLTrainer
from smart_control_analysis.building_factory import building_factory, get_base_params


def _sample_start_in_period(
    rng: np.random.Generator,
    start: str,
    end: str,
    episode_days: int,
    time_zone: str,
    start_hour: int = 0,
) -> pd.Timestamp:
    """Sample a random start day in [start, end], always at start_hour local time.

    Starting at 00:00 (default) gives the building 8 hours to reach comfort temperature
    before working hours begin, avoiding a cold-start penalty spike at episode start.
    """
    s = pd.Timestamp(start, tz=time_zone).normalize()  # midnight of start date
    e = pd.Timestamp(end, tz=time_zone).normalize()
    latest = e - pd.Timedelta(days=episode_days)
    if latest <= s:
        raise ValueError("Period too short for requested episode_days.")
    # sample a random day index
    n_days = int((latest - s).days)
    day_offset = int(rng.integers(0, n_days + 1))
    return s + pd.Timedelta(days=day_offset, hours=start_hour)


def _evaluate_period_random_starts(
    trainer: BuildingRLTrainer,
    params_template: Dict[str, Any],
    period_start: str,
    period_end: str,
    n_episodes: int,
    episode_days: int,
    deterministic: bool,
    seed: int,
    factory_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    rewards, lengths = [], []

    for _ in range(n_episodes):
        p = params_template.copy()
        p["start_timestamp"] = _sample_start_in_period(
            rng=rng,
            start=period_start,
            end=period_end,
            episode_days=episode_days,
            time_zone=p["time_zone"],
        ).isoformat()

        env = trainer.create_env(params=p, factory_kwargs=factory_kwargs)
        obs, _ = env.reset()
        _vn = trainer.vec_normalize
        done = False
        ep_r = 0.0
        ep_l = 0
        while not done:
            obs_in = _vn.normalize_obs(obs) if _vn is not None else obs
            action, _ = trainer.model.predict(obs_in, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_r += float(reward)
            ep_l += 1
            done = terminated or truncated
        env.close()

        rewards.append(ep_r)
        lengths.append(ep_l)

    return {
        "episode_rewards": np.array(rewards),
        "episode_lengths": np.array(lengths),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
    }


def _available_env_workers() -> int:
    """Number of CPU workers available to this process."""
    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return max(1, int(os.cpu_count() or 1))

def run_rl_setup(
    weather_csv_path: str,
    algo: str = "sac",
    total_timesteps: int = 300_000,
    chunk_timesteps: int = 50_000,
    n_envs: Optional[int] = None,
    episode_days: int = 7,
    seed: int = 42,
    output_dir: str = "./results/rl_2024",
    training_mode: str = "full",
    eval_training_mode: Optional[str] = None,
    n_eval_episodes: int = 8,
    floorplan: str = "single_room",
    energy_weight: float = 2.0,
    action_design: str = "reheat_per_zone",
    wandb_finish: bool = True,
    train_period_start="2015-10-01",
    train_period_end="2022-03-31",
    val_period_start="2022-10-01",
    val_period_end="2023-03-24",
    test_period_start="2023-12-01",
    test_period_end="2024-03-24",
    curriculum: Optional[List[Tuple[float, float]]] = None,
    **train_kwargs,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    algo = algo.lower()
    if algo not in {"sac", "td3", "ddpg", "tqc"}:
        raise ValueError(f"Unsupported algo: {algo}")

    if eval_training_mode is None:
        eval_training_mode = training_mode

    base = get_base_params().copy()
    base["weather_source"] = "replay"
    base["weather_csv_path"] = weather_csv_path
    base["time_zone"] = "America/Los_Angeles"
    base["time_step_sec"] = int(base.get("time_step_sec", 300))
    base["floorplan"] = floorplan
    base["energy_weight"] = energy_weight
    base["action_design"] = action_design

    max_steps = int(episode_days * 24 * 3600 / base["time_step_sec"])
    base["max_steps"] = max_steps

    trainer = BuildingRLTrainer(
        building_factory_fn=building_factory,
        base_params=base,
        default_factory_kwargs={"training_mode": training_mode},
    )

    if n_envs is None:
        n_envs = min(4, _available_env_workers())
        print(f"Auto n_envs={n_envs}")
    print(f"Using n_envs={n_envs}")

    rng = np.random.default_rng(seed)
    val_rng = np.random.default_rng(seed + 99)  # separate rng so val starts are consistent
    trained = 0
    first_chunk = True
    n_val_episodes_periodic = 2  # 2 episodes per checkpoint — enough to see trend without too much overhead

    # Curriculum: list of (fraction_of_training, energy_weight) breakpoints
    # Linearly interpolates between breakpoints for smooth transitions.
    # e.g. [(0.4, 1.0), (1.0, 60.0)]
    # means: 0-40% → constant ew=1.0, 40-100% → linear ramp 1.0 → 60.0
    # If None, use constant energy_weight throughout
    current_energy_weight = energy_weight
    if curriculum is not None:
        curriculum = sorted(curriculum, key=lambda x: x[0])
        current_energy_weight = curriculum[0][1]
        print(f"Curriculum: {[(f'{frac:.0%}', ew) for frac, ew in curriculum]}")

    while trained < total_timesteps:
        # Update energy weight based on curriculum schedule (linear interpolation)
        if curriculum is not None:
            progress = trained / total_timesteps
            if progress <= curriculum[0][0]:
                current_energy_weight = curriculum[0][1]
            elif progress >= curriculum[-1][0]:
                current_energy_weight = curriculum[-1][1]
            else:
                for i in range(len(curriculum) - 1):
                    f0, w0 = curriculum[i]
                    f1, w1 = curriculum[i + 1]
                    if f0 <= progress < f1:
                        t = (progress - f0) / (f1 - f0)
                        current_energy_weight = w0 + t * (w1 - w0)
                        break

        chunk = min(chunk_timesteps, total_timesteps - trained)
        p = base.copy()
        p["energy_weight"] = current_energy_weight
        p["start_timestamp"] = _sample_start_in_period(
            rng=rng,
            start=train_period_start,
            end=train_period_end,
            episode_days=episode_days,
            time_zone=base["time_zone"],
        ).isoformat()

        if first_chunk:
            trainer.train(
                algo=algo,
                total_timesteps=chunk,
                n_envs=n_envs,
                seed=seed,
                params=p,
                verbose=0,
                training_mode=training_mode,
                **train_kwargs,
            )
            first_chunk = False
        else:
            new_env = trainer.create_vec_env(
                n_envs=n_envs,
                params=p,
                training_mode=training_mode,
            )
            if trainer.env is not None:
                try:
                    trainer.env.close()
                except Exception:
                    pass
            trainer.env = new_env
            trainer.model.set_env(new_env)
            trainer.model.learn(
                total_timesteps=chunk,
                callback=trainer.callback,
                progress_bar=True,
                reset_num_timesteps=False,
            )

        trained += chunk
        print(f"Trained {trained}/{total_timesteps} timesteps")

        # --- Periodic val evaluation: log real metrics (not normalized reward) to W&B ---
        if trainer.wandb_run is not None:
            val_discomfort, val_cost, val_comfort_penalty, val_starts = [], [], [], []
            for _ in range(n_val_episodes_periodic):
                vp = base.copy()
                start_ts = _sample_start_in_period(
                    rng=val_rng,
                    start=val_period_start,
                    end=val_period_end,
                    episode_days=episode_days,
                    time_zone=base["time_zone"],
                )
                vp["start_timestamp"] = start_ts.isoformat()
                val_starts.append(start_ts.isoformat())
                env = trainer.create_env(params=vp, factory_kwargs={"training_mode": eval_training_mode})
                _vn = trainer.vec_normalize
                obs, _ = env.reset()
                done = False
                ep_cost, ep_discomfort, ep_comfort = 0.0, 0.0, 0.0
                trace_rows = []
                while not done:
                    obs_in = _vn.normalize_obs(obs) if _vn is not None else obs
                    action, _ = trainer.model.predict(obs_in, deterministic=True)
                    obs, _, terminated, truncated, info = env.step(action)
                    ep_cost += float(info.get("energy_cost_usd", 0.0))
                    ep_comfort += float(info.get("comfort_penalty", 0.0))
                    zone_temps_c = [v - 273.15 for v in env.sim.building.get_zone_average_temps().values()]
                    trace_rows.append({
                        "room_temp_c": float(np.mean(zone_temps_c)),
                        "comfort_low_c": float(info.get("comfort_low_c", env.comfort_band_k[0] - 273.15)),
                        "comfort_high_c": float(info.get("comfort_high_c", env.comfort_band_k[1] - 273.15)),
                        **{f"zone_temp_c_{i}": float(t) for i, t in enumerate(zone_temps_c)},
                    })
                    done = terminated or truncated
                env.close()
                dt_sec = float(base.get("time_step_sec", 600))
                ep_dh = _discomfort_degree_hours(pd.DataFrame(trace_rows), dt_sec)
                val_discomfort.append(ep_dh)
                val_cost.append(ep_cost)
                val_comfort_penalty.append(ep_comfort)

            log_dict = {
                "val/discomfort_deg_h":     float(np.mean(val_discomfort)),
                "val/energy_cost_usd":      float(np.mean(val_cost)),
                "val/comfort_penalty_mean": float(np.mean(val_comfort_penalty)),
                "val/start_timestamps":     json.dumps(val_starts),
            }
            if curriculum is not None:
                log_dict["train/energy_weight"] = current_energy_weight
            trainer.wandb_run.log(log_dict, step=trained)

    trainer.save_model(os.path.join(output_dir, f"{algo}_2024_model"))
    trainer.save_results(output_dir)

    val_results = _evaluate_period_random_starts(
        trainer=trainer,
        params_template=base,
        period_start=val_period_start,
        period_end=val_period_end,
        n_episodes=n_eval_episodes,
        episode_days=episode_days,
        deterministic=True,
        seed=seed + 1,
        factory_kwargs={"training_mode": eval_training_mode},
    )
    test_results = _evaluate_period_random_starts(
        trainer=trainer,
        params_template=base,
        period_start=test_period_start,
        period_end=test_period_end,
        n_episodes=n_eval_episodes,
        episode_days=episode_days,
        deterministic=True,
        seed=seed + 2,
        factory_kwargs={"training_mode": eval_training_mode},
    )

    summary = {
        "algo": algo,
        "training_mode": training_mode,
        "eval_training_mode": eval_training_mode,
        "val_mean_reward": val_results["mean_reward"],
        "val_std_reward": val_results["std_reward"],
        "test_mean_reward": test_results["mean_reward"],
        "test_std_reward": test_results["std_reward"],
    }

    if trainer.wandb_run is not None:
        if wandb_finish:
            trainer.wandb_run.finish()
            trainer.wandb_run = None

    pd.DataFrame([summary]).to_csv(
        os.path.join(output_dir, f"{algo}_summary.csv"), index=False
    )

    return {
        "trainer": trainer,
        "val_results": val_results,
        "test_results": test_results,
        "summary": summary,
    }


def _extract_action_channels(action: np.ndarray, action_design: str = "reheat_per_zone") -> Dict[str, float]:
    """Extract action channels into named columns, accounting for action_design layout.

    Action vector layout by design:
      reheat_per_zone : [supply, boiler, shared_damper, reheat_z0, reheat_z1, ...]
      damper_per_zone : [supply, boiler, shared_reheat, damper_z0, damper_z1, ...]
      full_per_zone   : [supply, boiler, reheat_z0, ..., reheat_zN, damper_z0, ..., damper_zN]
    """
    a = np.asarray(action, dtype=np.float32).flatten()
    n_extra = max(len(a) - 2, 0)  # elements after supply + boiler

    row: Dict[str, float] = {
        "action_supply": float(a[0]) if len(a) > 0 else np.nan,
        "action_boiler": float(a[1]) if len(a) > 1 else np.nan,
    }

    if action_design == "reheat_per_zone":
        # a[2] = shared damper, a[3:] = per-zone reheat
        row["action_damper_shared"] = float(a[2]) if len(a) > 2 else np.nan
        row["action_reheat_mean"]   = float(np.mean(a[3:])) if len(a) > 3 else np.nan
    elif action_design == "damper_per_zone":
        # a[2] = shared reheat, a[3:] = per-zone dampers
        row["action_reheat_shared"] = float(a[2]) if len(a) > 2 else np.nan
        row["action_damper_mean"]   = float(np.mean(a[3:])) if len(a) > 3 else np.nan
    else:  # full_per_zone
        # a[2:2+n] = per-zone reheat, a[2+n:] = per-zone dampers  (n = n_extra // 2)
        n = n_extra // 2
        row["action_reheat_mean"] = float(np.mean(a[2:2+n])) if n > 0 else np.nan
        row["action_damper_mean"] = float(np.mean(a[2+n:])) if n > 0 else np.nan

    return row


def _discomfort_degree_hours(df: pd.DataFrame, dt_sec: float) -> float:
    """
    Compute total discomfort degree-hours for working hours only.

    Filters to daytime timesteps (comfort_low_c at its max = no night setback).
    Sums violations across all individual zones so a cold zone cannot be masked
    by a warm zone.

    Discomfort = Σ_zones Σ_daytime_steps  max(T_low - T_zone, 0) + max(T_zone - T_high, 0)  ×  dt_h
    """
    if df.empty or "comfort_low_c" not in df.columns:
        return 0.0

    # Filter to daytime only
    day_low = df["comfort_low_c"].max()
    df = df[df["comfort_low_c"] >= day_low - 0.01]
    if df.empty:
        return 0.0

    dt_h = dt_sec / 3600.0
    low  = df["comfort_low_c"].to_numpy()
    high = df["comfort_high_c"].to_numpy()
    zone_cols = sorted([c for c in df.columns if c.startswith("zone_temp_c_")])

    if zone_cols:
        total = 0.0
        for col in zone_cols:
            t = df[col].to_numpy()
            violation = np.maximum(low - t, 0.0) + np.maximum(t - high, 0.0)
            total += float(violation.sum() * dt_h)
        return total
    else:
        t = df["room_temp_c"].to_numpy()
        violation = np.maximum(low - t, 0.0) + np.maximum(t - high, 0.0)
        return float(violation.sum() * dt_h)


def _run_episode_trace(
    env,
    policy_fn,
    policy_name: str,
    seed: Optional[int] = None,
    action_design: str = "reheat_per_zone",
) -> Tuple[pd.DataFrame, Dict[str, float], Tuple[float, float]]:
    obs, _ = env.reset(seed=seed)
    done = False

    rows = []
    ep_reward = 0.0
    ep_comfort = 0.0
    ep_energy = 0.0
    ep_cost = 0.0
    ep_len = 0

    comfort_low_c = env.comfort_band_k[0] - 273.15
    comfort_high_c = env.comfort_band_k[1] - 273.15

    while not done:
        action = np.asarray(policy_fn(obs, env), dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        ts = env.sim.current_timestamp
        zone_temps_k = env.sim.building.get_zone_average_temps()
        zone_temps_c = [v - 273.15 for v in zone_temps_k.values()]
        room_temp_c     = float(np.mean(zone_temps_c))
        room_temp_min_c = float(np.min(zone_temps_c))
        room_temp_max_c = float(np.max(zone_temps_c))
        outside_temp_c = float(
            env.air_handler.get_observation("outside_air_temperature_sensor", ts) - 273.15
        )

        row = {
            "timestamp": ts,
            "policy": policy_name,
            "room_temp_c":     room_temp_c,
            "room_temp_min_c": room_temp_min_c,
            "room_temp_max_c": room_temp_max_c,
            **{f"zone_temp_c_{i}": float(t) for i, t in enumerate(zone_temps_c)},
            "outside_temp_c": outside_temp_c,
            "comfort_penalty": float(info.get("comfort_penalty", 0.0)),
            "energy_rate": float(info.get("energy_rate", 0.0)),
            "blower_rate": float(info.get("blower_rate", 0.0)),
            "ah_conditioning_rate": float(info.get("ah_conditioning_rate", 0.0)),
            "boiler_gas_rate": float(info.get("boiler_gas_rate", 0.0)),
            "pump_rate_raw": float(info.get("pump_rate_raw", 0.0)),
            "reheat_coil_rate": float(info.get("reheat_coil_rate", 0.0)),
            "energy_cost_usd": float(info.get("energy_cost_usd", 0.0)),
            "gas_cost_usd": float(info.get("gas_cost_usd", 0.0)),
            "elec_cost_usd": float(info.get("elec_cost_usd", 0.0)),
            "elec_price_eur_per_mwh": float(info.get("elec_price_eur_per_mwh", np.nan)),
            "gas_price_eur_per_mwh": float(info.get("gas_price_eur_per_mwh", np.nan)),
            "reward": float(reward),
            "reward_total": float(info.get("reward_total", reward)),
            "supply_air_sp_C": float(info.get("supply_air_sp_C", np.nan)),
            "boiler_sp_C": float(info.get("boiler_sp_C", np.nan)),
            "comfort_low_c": float(info.get("comfort_low_c", env.comfort_band_k[0] - 273.15)),
            "comfort_high_c": float(info.get("comfort_high_c", env.comfort_band_k[1] - 273.15)),
        }
        row.update(_extract_action_channels(action, action_design=action_design))
        rows.append(row)

        ep_reward += float(reward)
        ep_comfort += float(info.get("comfort_penalty", 0.0))
        ep_energy += float(info.get("energy_rate", 0.0))
        ep_cost += float(info.get("energy_cost_usd", 0.0))
        ep_len += 1
        done = terminated or truncated

    df = pd.DataFrame(rows)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    dt_sec = float(env.sim.time_step_sec)
    metrics = {
        "reward": ep_reward,
        "comfort_penalty": ep_comfort,
        "energy": ep_energy,
        "energy_cost_usd": ep_cost,
        "discomfort_deg_h": _discomfort_degree_hours(df, dt_sec),
        "length": ep_len,
    }
    return df, metrics, (comfort_low_c, comfort_high_c)
