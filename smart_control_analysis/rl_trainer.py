import csv
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import DDPG
from stable_baselines3 import SAC
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from tqdm import tqdm
import itertools

from smart_control_analysis.action_wrappers import FixedActionsWrapper
from smart_control_analysis.building_factory import building_factory, get_base_params

class TrainingProgressCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
            super().__init__(verbose)
            self.episode_rewards = []
            self.episode_lengths = []
            self.timesteps = []
            self.energy_rates = []
            self.comfort_penalties = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "energy_rate" in info:
                self.energy_rates.append(float(info["energy_rate"]))
            if "comfort_penalty" in info:
                self.comfort_penalties.append(float(info["comfort_penalty"]))

            # record finished episodes from Monitor/VecMonitor
            if "episode" in info:
                ep = info["episode"]
                self.episode_rewards.append(float(ep.get("r", 0.0)))
                self.episode_lengths.append(int(ep.get("l", 0)))
                self.timesteps.append(int(self.model.num_timesteps))
        return True

class BuildingRLTrainer:
    """Train multiple RL agents (SAC/TD3/DDPG) on the building environment."""

    def __init__(
        self,
        building_factory_fn: Callable,
        base_params: Optional[Dict[str, Any]] = None,
        default_factory_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.building_factory_fn = building_factory_fn
        self.base_params = base_params if base_params is not None else get_base_params()
        self.default_factory_kwargs = default_factory_kwargs or {}

        self.model = None
        self.callback = None
        self.env = None
        self.algo_name = None

    def create_env(
        self,
        params: Optional[Dict[str, Any]] = None,
        fixed_actions: Optional[Dict[int, float]] = None,
        factory_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if params is None:
            params = self.base_params.copy()

        kwargs = {**self.default_factory_kwargs, **(factory_kwargs or {})}
        _, env = self.building_factory_fn(params, **kwargs)

        if fixed_actions:
            env = FixedActionsWrapper(env, fixed=fixed_actions)
        return env

    def create_vec_env(
        self,
        n_envs: int = 1,
        params: Optional[Dict[str, Any]] = None,
        fixed_actions: Optional[Dict[int, float]] = None,
        factory_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if params is None:
            params = self.base_params.copy()

        kwargs = {**self.default_factory_kwargs, **(factory_kwargs or {})}

        def make_env_fn():
            _, env = self.building_factory_fn(params, **kwargs)
            if fixed_actions:
                env = FixedActionsWrapper(env, fixed=fixed_actions)
            return env

        vec_cls = SubprocVecEnv if n_envs > 1 else None
        return make_vec_env(
            make_env_fn,
            n_envs=n_envs,
            wrapper_class=None,
            vec_env_cls=vec_cls,
        )


    def train(
        self,
        algo: str = "sac",
        total_timesteps: int = 100000,
        learning_rate: float = 3e-4,
        buffer_size: int = 100000,
        batch_size: int = 256,
        n_envs: int = 1,
        params: Optional[Dict[str, Any]] = None,
        verbose: int = 1,
        fixed_actions: Optional[Dict[int, float]] = None,
        factory_kwargs: Optional[Dict[str, Any]] = None,
        **algo_kwargs
    ):
        """
        Train any supported algorithm on the building environment.

        Parameters
        ----------
        algo : str
            Algorithm to use: 'sac', 'td3', or 'ddpg'
        total_timesteps : int
            Total number of environment steps to train.
        learning_rate : float
            Learning rate for the neural networks.
        buffer_size : int
            Size of replay buffer.
        batch_size : int
            Batch size for training.
        n_envs : int
            Number of parallel environments.
        params : dict, optional
            Environment parameters.
        verbose : int
            Verbosity level.
        **algo_kwargs : dict
            Additional algorithm-specific hyperparameters.

        Returns
        -------
        Trained model
        """
        if params is None:
            params = self.base_params.copy()

        self.algo_name = algo.lower()
        if self.algo_name not in ["sac", "td3", "ddpg"]:
            raise ValueError(f"Unsupported algorithm: {algo}. Choose 'sac', 'td3', or 'ddpg'.")

        algo_kwargs.pop("fixed_actions", None)
        # Create environment
        self.env = self.create_vec_env(
            n_envs=n_envs,
            params=params,
            fixed_actions=fixed_actions,
            factory_kwargs=factory_kwargs,
        )
        # Common parameters
        common_params = {
            "learning_rate": learning_rate,
            "verbose": verbose,
            "device": "auto",
        }

        # Algo-specific parameters
        if self.algo_name == "sac":
            common_params["buffer_size"] = buffer_size
            common_params["batch_size"] = batch_size
            AlgoClass = SAC
        elif self.algo_name == "td3":
            common_params["buffer_size"] = buffer_size
            common_params["batch_size"] = batch_size
            # TD3-specific defaults
            common_params.setdefault("policy_delay", 2)
            common_params.setdefault("target_policy_noise", 0.2)
            common_params.setdefault("target_noise_clip", 0.5)
            AlgoClass = TD3
        else:  # ddpg
            common_params["buffer_size"] = buffer_size
            common_params["batch_size"] = batch_size
            AlgoClass = DDPG

        # Merge in user-provided kwargs
        common_params.update(algo_kwargs)

        # Create agent
        self.model = AlgoClass("MlpPolicy", self.env, **common_params)

        # Setup callback for progress tracking
        self.callback = TrainingProgressCallback()

        # Train
        print(f"Training {self.algo_name.upper()} for {total_timesteps} timesteps...")
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=self.callback,
            progress_bar=True
        )

        return self.model

    def evaluate(
        self,
        n_episodes: int = 5,
        deterministic: bool = True,
        params: Optional[Dict[str, Any]] = None,
        fixed_actions: Optional[Dict[int, float]] = None,
        factory_kwargs: Optional[Dict[str, Any]] = None,

    ) -> Dict[str, Any]:
        """
        Evaluate trained model on environment.

        Returns
        -------
        dict with episode_rewards, mean_reward, std_reward, episode_lengths
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        if params is None:
            params = self.base_params.copy()

        eval_env = self.create_env(
            params=params,
            fixed_actions=fixed_actions,
            factory_kwargs=factory_kwargs,
        )
        episode_rewards = []
        episode_lengths = []

        for episode in range(n_episodes):
            obs, _ = eval_env.reset()
            episode_reward = 0.0
            episode_length = 0
            done = False

            while not done:
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                episode_reward += reward
                episode_length += 1
                done = terminated or truncated

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            print(f"  Episode {episode + 1}: reward={episode_reward:.2f}, length={episode_length}")

        eval_env.close()

        return {
            "episode_rewards": np.array(episode_rewards),
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "episode_lengths": np.array(episode_lengths),
        }

    def sweep_fixed_actions(
        self,
        sweep_config: Dict[str, List[float]],
        n_episodes: int = 5,
        params: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Sweep over fixed action values and evaluate reward/comfort/energy.

        Parameters
        ----------
        sweep_config : dict
            Dict mapping action_idx -> list of values to sweep
            Example: {0: [0.0], 1: [0.0, 0.5, 1.0], 2: [1.0]}

        Returns
        -------
        DataFrame with results per sweep point
        """
        if params is None:
            params = self.base_params.copy()

        # Generate all combinations
        indices = sorted(sweep_config.keys())
        values_lists = [sweep_config[i] for i in indices]
        combinations = list(itertools.product(*values_lists))

        rows = []

        for combo in combinations:
            fixed_dict = dict(zip(indices, combo))

            # Create wrapped env
            eval_env = self.create_env(params)
            eval_env = FixedActionsWrapper(eval_env, fixed=fixed_dict)

            episode_rewards = []
            episode_energy = []
            episode_comfort = []

            for episode in range(n_episodes):
                obs, _ = eval_env.reset()
                ep_reward = 0.0
                ep_energy = 0.0
                ep_comfort = 0.0
                done = False

                while not done:
                    # random action (baseline, no agent yet)
                    action = eval_env.action_space.sample()
                    obs, reward, terminated, truncated, info = eval_env.step(action)
                    ep_reward += reward
                    ep_energy += float(info.get("energy_rate", 0.0))
                    ep_comfort += float(info.get("comfort_penalty", 0.0))
                    done = terminated or truncated

                episode_rewards.append(ep_reward)
                episode_energy.append(ep_energy)
                episode_comfort.append(ep_comfort)

            eval_env.close()

            rows.append({
                **fixed_dict,
                "mean_reward": float(np.mean(episode_rewards)),
                "std_reward": float(np.std(episode_rewards)),
                "mean_energy": float(np.mean(episode_energy)),
                "mean_comfort": float(np.mean(episode_comfort)),
            })

        return pd.DataFrame(rows)

    def plot_training_progress(
            self,
            figsize: Tuple[int, int] = (10, 4),
            show_episode_length: bool = False,
        ) -> plt.Figure:
            """Plot training progress from callback."""
            if self.callback is None or not self.callback.episode_rewards:
                raise ValueError("No training data available. Train the model first.")

            if not show_episode_length:
                fig, ax = plt.subplots(1, 1, figsize=figsize)
                ax.plot(self.callback.episode_rewards, alpha=0.7, linewidth=1, label="Raw")
                ax.set_xlabel("Episode", fontsize=12)
                ax.set_ylabel("Episode Reward", fontsize=12)
                ax.set_title(
                    f"{self.algo_name.upper()}: Episode Rewards",
                    fontsize=14,
                    fontweight="bold",
                )
                ax.grid(True, alpha=0.3)

                if len(self.callback.episode_rewards) > 10:
                    window = max(1, len(self.callback.episode_rewards) // 20)
                    smoothed = np.convolve(
                        self.callback.episode_rewards,
                        np.ones(window) / window,
                        mode="valid",
                    )
                    ax.plot(smoothed, color="red", linewidth=2, label="Moving Avg")
                    ax.legend()

                fig.tight_layout()
                return fig

            # Optional old behavior
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            axes[0].plot(self.callback.episode_rewards, alpha=0.7, linewidth=1, label="Raw")
            axes[0].set_xlabel("Episode", fontsize=12)
            axes[0].set_ylabel("Episode Reward", fontsize=12)
            axes[0].set_title(f"{self.algo_name.upper()}: Episode Rewards", fontsize=14, fontweight="bold")
            axes[0].grid(True, alpha=0.3)

            if len(self.callback.episode_rewards) > 10:
                window = max(1, len(self.callback.episode_rewards) // 20)
                smoothed = np.convolve(
                    self.callback.episode_rewards, np.ones(window) / window, mode="valid"
                )
                axes[0].plot(smoothed, color="red", linewidth=2, label="Moving Avg")
                axes[0].legend()

            axes[1].plot(self.callback.episode_lengths, alpha=0.7, linewidth=1)
            axes[1].set_xlabel("Episode", fontsize=12)
            axes[1].set_ylabel("Episode Length (steps)", fontsize=12)
            axes[1].set_title(f"{self.algo_name.upper()}: Episode Lengths", fontsize=14, fontweight="bold")
            axes[1].grid(True, alpha=0.3)

            fig.tight_layout()
            return fig


    def save_results(self, output_dir: str):
        """Save training results to CSV for later analysis."""
        if self.callback is None:
            raise ValueError("No training data to save.")

        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, f"{self.algo_name}_results.csv")

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "reward", "length", "timestep"])
            for i, (r, l, t) in enumerate(zip(
                self.callback.episode_rewards,
                self.callback.episode_lengths,
                self.callback.timesteps
            )):
                writer.writerow([i, r, l, t])

        print(f"Results saved to {csv_path}")
        return csv_path

    def save_model(self, filepath: str):
        """Save trained model."""
        if self.model is None:
            raise ValueError("No model to save. Train first.")
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str, algo: str):
        """Load trained model."""
        algo = algo.lower()
        if algo == "sac":
            self.model = SAC.load(filepath)
        elif algo == "td3":
            self.model = TD3.load(filepath)
        elif algo == "ddpg":
            self.model = DDPG.load(filepath)
        else:
            raise ValueError(f"Unknown algo: {algo}")
        self.algo_name = algo
        print(f"Model loaded from {filepath}")

    def close(self):
        """Close environments."""
        if self.env is not None:
            self.env.close()


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


def _sample_start_in_period(
    rng: np.random.Generator,
    start: str,
    end: str,
    episode_days: int,
    time_zone: str,
) -> pd.Timestamp:
    s = pd.Timestamp(start, tz=time_zone)
    e = pd.Timestamp(end, tz=time_zone)
    latest = e - pd.Timedelta(days=episode_days)
    if latest <= s:
        raise ValueError("Period too short for requested episode_days.")
    frac = rng.random()
    return s + (latest - s) * frac


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
        done = False
        ep_r = 0.0
        ep_l = 0
        while not done:
            action, _ = trainer.model.predict(obs, deterministic=deterministic)
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

def run_sac_2024_setup(
    weather_csv_path: str,
    total_timesteps: int = 300_000,
    chunk_timesteps: int = 50_000,
    n_envs: Optional[int] = None,  # ignored if provided; auto-set below
    episode_days: int = 7,
    seed: int = 42,
    output_dir: str = "./results/sac_2024",
) -> Dict[str, Any]:
    """
    Train on 2024 train split with randomized weekly starts, then evaluate on val/test.
    Splits:
      - train: 2024-01-01 .. 2024-10-01
      - val:   2024-10-01 .. 2024-12-01
      - test:  2024-12-01 .. 2025-01-01
    """
    os.makedirs(output_dir, exist_ok=True)

    base = get_base_params().copy()
    base["weather_source"] = "replay"
    base["weather_csv_path"] = weather_csv_path
    base["time_zone"] = "America/Los_Angeles"
    base["time_step_sec"] = int(base.get("time_step_sec", 300))

    # one episode = one week
    max_steps = int(episode_days * 24 * 3600 / base["time_step_sec"])
    base["max_steps"] = max_steps

    trainer = BuildingRLTrainer(
        building_factory_fn=building_factory,
        base_params=base,
        default_factory_kwargs={"training_mode": "full"},
    )

    # Always use all CPU cores available to this process
    n_envs = _available_env_workers()
    print(f"Using n_envs={n_envs} (available CPU workers for this process)")

    rng = np.random.default_rng(seed)
    trained = 0
    first_chunk = True

    while trained < total_timesteps:
        chunk = min(chunk_timesteps, total_timesteps - trained)

        p = base.copy()
        p["start_timestamp"] = _sample_start_in_period(
            rng=rng,
            start="2024-01-01",
            end="2024-10-01",
            episode_days=episode_days,
            time_zone=base["time_zone"],
        ).isoformat()

        if first_chunk:
            trainer.train(
                algo="sac",
                total_timesteps=chunk,
                n_envs=n_envs,
                params=p,
                factory_kwargs={"training_mode": "full"},
                verbose=1,
            )
            first_chunk = False
        else:
            new_env = trainer.create_vec_env(
                n_envs=n_envs,
                params=p,
                factory_kwargs={"training_mode": "full"},
            )
            if trainer.env is not None:
                trainer.env.close()
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

    # Evaluate on held-out periods
    val_results = _evaluate_period_random_starts(
        trainer=trainer,
        params_template=base,
        period_start="2024-10-01",
        period_end="2024-12-01",
        n_episodes=8,
        episode_days=episode_days,
        deterministic=True,
        seed=seed + 1,
        factory_kwargs={"training_mode": "full"},
    )
    test_results = _evaluate_period_random_starts(
        trainer=trainer,
        params_template=base,
        period_start="2024-12-01",
        period_end="2025-01-01",
        n_episodes=8,
        episode_days=episode_days,
        deterministic=True,
        seed=seed + 2,
        factory_kwargs={"training_mode": "full"},
    )

    # Save model + logs
    trainer.save_model(os.path.join(output_dir, "sac_2024_model"))
    trainer.save_results(output_dir)

    summary = {
        "val_mean_reward": val_results["mean_reward"],
        "val_std_reward": val_results["std_reward"],
        "test_mean_reward": test_results["mean_reward"],
        "test_std_reward": test_results["std_reward"],
    }
    pd.DataFrame([summary]).to_csv(os.path.join(output_dir, "summary.csv"), index=False)

    print("Validation:", summary["val_mean_reward"], "+/-", summary["val_std_reward"])
    print("Test:", summary["test_mean_reward"], "+/-", summary["test_std_reward"])

    return {
        "trainer": trainer,
        "val_results": val_results,
        "test_results": test_results,
        "summary": summary,
    }
