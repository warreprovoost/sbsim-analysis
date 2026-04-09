import csv
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Any, Callable, Dict, List, Optional, Tuple
from stable_baselines3 import DDPG
from stable_baselines3 import SAC
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from tqdm import tqdm
import itertools
from smart_control_analysis.baseline_controller import ThermostatBaselineController

try:
    import wandb
except ImportError:
    wandb = None

from smart_control_analysis.action_wrappers import FixedActionsWrapper
from smart_control_analysis.building_factory import building_factory, get_base_params

class TrainingProgressCallback(BaseCallback):
    def __init__(
        self,
        verbose: int = 0,
        wandb_run: Optional[Any] = None,
        log_every_n_steps: int = 100,
    ):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.timesteps = []
        self.energy_rates = []
        self.comfort_penalties = []
        self.wandb_run = wandb_run
        self.log_every_n_steps = max(1, int(log_every_n_steps))
        self._last_logged_step = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        current_step = int(self.model.num_timesteps)

        # aggregate step metrics
        step_energy = []
        step_comfort = []

        for info in infos:
            if "energy_rate" in info:
                val = float(info["energy_rate"])
                self.energy_rates.append(val)
                step_energy.append(val)
            if "comfort_penalty" in info:
                val = float(info["comfort_penalty"])
                self.comfort_penalties.append(val)
                step_comfort.append(val)

            if "episode" in info:
                ep = info["episode"]
                ep_r = float(ep.get("r", 0.0))
                ep_l = int(ep.get("l", 0))
                self.episode_rewards.append(ep_r)
                self.episode_lengths.append(ep_l)
                self.timesteps.append(current_step)

                if self.wandb_run is not None:
                    self.wandb_run.log(
                        {
                            "train/episode_reward": ep_r,
                            "train/episode_length": ep_l,
                        },
                        step=current_step,
                    )

        # periodic step-level logging
        if (
            self.wandb_run is not None
            and (current_step - self._last_logged_step) >= self.log_every_n_steps
        ):
            payload = {}
            if step_energy:
                payload["train/energy_rate_mean"] = float(np.mean(step_energy))
            if step_comfort:
                payload["train/comfort_penalty_mean"] = float(np.mean(step_comfort))
            # Log raw reward_total from info (not VecNormalize-scaled episode reward)
            step_rewards = [
                float(info["reward_total"])
                for info in infos
                if "reward_total" in info
            ]
            if step_rewards:
                payload["train/reward_mean"] = float(np.mean(step_rewards))
            if payload:
                self.wandb_run.log(payload, step=current_step)
            self._last_logged_step = current_step

        return True


def _make_env_worker(
    building_factory_fn: Callable,
    params: Dict[str, Any],
    factory_kwargs: Dict[str, Any],
    training_mode: Optional[str],
    rank: int,
    gpu_id: Optional[int] = None,

) -> Callable:
    """
    Returns a picklable env factory for SubprocVecEnv.
    Must be a top-level function, not a lambda or closure capturing self.
    """
    def _init():
        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""  # disable GPU, avoid TF CUDA init hang
        kwargs = factory_kwargs.copy()
        if training_mode is not None:
            kwargs["training_mode"] = training_mode

        _, env = building_factory_fn(params.copy(), **kwargs)
        _apply_training_mode_overrides(env, training_mode)
        from stable_baselines3.common.monitor import Monitor
        env = Monitor(env)
        return env

    return _init


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
        self.vec_normalize: Optional[VecNormalize] = None
        self.algo_name = None
        self.wandb_run = None

    def create_env(
        self,
        params: Optional[Dict[str, Any]] = None,
        fixed_actions: Optional[Dict[int, float]] = None,
        factory_kwargs: Optional[Dict[str, Any]] = None,
        training_mode: Optional[str] = None,
    ):
        if params is None:
            params = self.base_params.copy()

        kwargs = _merge_factory_kwargs(
            self.default_factory_kwargs, factory_kwargs, training_mode
        )
        _, env = self.building_factory_fn(params, **kwargs)

        _apply_training_mode_overrides(env, training_mode)

        if fixed_actions:
            env = FixedActionsWrapper(env, fixed=fixed_actions)
        return env


    def create_vec_env(
        self,
        n_envs: int = 1,
        params: Optional[Dict[str, Any]] = None,
        fixed_actions: Optional[Dict[int, float]] = None,
        factory_kwargs: Optional[Dict[str, Any]] = None,
        training_mode: Optional[str] = None,
    ):
        if params is None:
            params = self.base_params.copy()

        kwargs = _merge_factory_kwargs(
            self.default_factory_kwargs, factory_kwargs, training_mode
        )

        env_fns = [
            _make_env_worker(
                building_factory_fn=self.building_factory_fn,
                params=params.copy(),
                factory_kwargs=kwargs,
                training_mode=training_mode,
                gpu_id=None,
                rank=i,
            )
            for i in range(n_envs)
        ]

        if n_envs == 1:
            vec_env = DummyVecEnv(env_fns)
        else:
            vec_env = SubprocVecEnv(env_fns, start_method="spawn")
        normalized = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        self.vec_normalize = normalized
        return normalized


    def train(
        self,
        algo: str = "sac",
        total_timesteps: int = 100000,
        learning_rate: float = 3e-4,
        buffer_size: int = 500_000,
        batch_size: int = 256,
        n_envs: int = 1,
        params: Optional[Dict[str, Any]] = None,
        verbose: int = 1,
        fixed_actions: Optional[Dict[int, float]] = None,
        factory_kwargs: Optional[Dict[str, Any]] = None,
        training_mode: Optional[str] = None,
        use_wandb: bool = False,
        wandb_project: str = "smart-control-rl",
        wandb_run_name: Optional[str] = None,
        wandb_tags: Optional[List[str]] = None,
        wandb_mode: str = "online",
        wandb_log_every_n_steps: int = 100,
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

        external_wandb_run = algo_kwargs.pop("wandb_run", None)
        if external_wandb_run is not None:
            self.wandb_run = external_wandb_run

        algo_kwargs.pop("fixed_actions", None)
        # Create environment
        self.env = self.create_vec_env(
            n_envs=n_envs,
            params=params,
            fixed_actions=fixed_actions,
            factory_kwargs=factory_kwargs,
            training_mode=training_mode,
        )
        # Common parameters
        common_params = {
            "learning_rate": learning_rate,
            "verbose": verbose,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
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

        if use_wandb:
            if wandb is None:
                raise ImportError("wandb is not installed. Install with: pip install wandb")
            if self.wandb_run is None:
                self.wandb_run = wandb.init(
                    project=wandb_project,
                    name=wandb_run_name,
                    tags=wandb_tags or [],
                    mode=wandb_mode,  # "online", "offline", or "disabled"
                    config={
                        "algo": self.algo_name,
                        "total_timesteps": total_timesteps,
                        "learning_rate": learning_rate,
                        "buffer_size": buffer_size,
                        "batch_size": batch_size,
                        "n_envs": n_envs,
                        "training_mode": training_mode,
                    },
                    reinit=False,
                )

        # Setup callback for progress tracking
        self.callback = TrainingProgressCallback(
            wandb_run=self.wandb_run,
            log_every_n_steps=wandb_log_every_n_steps,
        )

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
        training_mode: Optional[str] = None,
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
            training_mode=training_mode,
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
        """Save trained model and VecNormalize stats."""
        if self.model is None:
            raise ValueError("No model to save. Train first.")
        self.model.save(filepath)
        # Also save VecNormalize statistics so eval uses the same obs normalization
        if isinstance(self.env, VecNormalize):
            norm_path = filepath + "_vecnormalize.pkl"
            self.env.save(norm_path)
            print(f"VecNormalize stats saved to {norm_path}")
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

    def load_vec_normalize(self, filepath: str):
        """Load VecNormalize stats saved alongside the model (for correct eval obs normalization)."""
        import pickle
        with open(filepath, "rb") as f:
            self.vec_normalize = pickle.load(f)
        print(f"VecNormalize stats loaded from {filepath}")

    def close(self):
        """Close environments."""
        if self.env is not None:
            self.env.close()
        if self.wandb_run is not None:
            self.wandb_run.finish()
            self.wandb_run = None


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
    train_period_start="2019-10-01",
    train_period_end="2022-03-31",
    val_period_start="2022-10-01",
    val_period_end="2023-03-24",
    test_period_start="2023-10-01",
    test_period_end="2024-03-24",
    **train_kwargs,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    algo = algo.lower()
    if algo not in {"sac", "td3", "ddpg"}:
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
    trained = 0
    first_chunk = True

    while trained < total_timesteps:
        chunk = min(chunk_timesteps, total_timesteps - trained)
        p = base.copy()
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
                params=p,
                verbose=1,
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

    # log final eval to wandb if enabled
    if trainer.wandb_run is not None:
        trainer.wandb_run.log(
            {
                "eval/val_mean_reward": summary["val_mean_reward"],
                "eval/val_std_reward": summary["val_std_reward"],
                "eval/test_mean_reward": summary["test_mean_reward"],
                "eval/test_std_reward": summary["test_std_reward"],
            },
            step=int(trainer.model.num_timesteps),
        )
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

def _merge_factory_kwargs(
    default_factory_kwargs: Optional[Dict[str, Any]],
    factory_kwargs: Optional[Dict[str, Any]],
    training_mode: Optional[str] = None,
) -> Dict[str, Any]:
    merged = {**(default_factory_kwargs or {}), **(factory_kwargs or {})}
    if training_mode is not None:
        merged["training_mode"] = training_mode
    return merged

def _extract_action_channels(action: np.ndarray) -> Dict[str, float]:
    a = np.asarray(action, dtype=np.float32).flatten()
    return {
        "action_supply":       float(a[0]) if len(a) > 0 else np.nan,
        "action_boiler":       float(a[1]) if len(a) > 1 else np.nan,
        "action_reheat":       float(a[2]) if len(a) > 2 else np.nan,   # shared reheat
        "action_damper_mean":  float(np.mean(a[3:])) if len(a) > 3 else np.nan,  # mean per-zone damper
    }


def _discomfort_degree_hours(df: pd.DataFrame, dt_sec: float) -> float:
    """
    Compute total discomfort in degree-hours for one episode trace.

    Uses per-step comfort_low_c / comfort_high_c columns (supports night setback).
    Discomfort = Σ  max(T_low - T_zone, 0) + max(T_zone - T_high, 0)  ×  dt_h

    A value of 0 means the zone was always inside the comfort band.
    A value of 1 means the zone was 1°C outside the band for 1 hour.
    """
    if df.empty:
        return 0.0
    dt_h = dt_sec / 3600.0
    t = df["room_temp_c"].to_numpy()
    low = df["comfort_low_c"].to_numpy() if "comfort_low_c" in df.columns else np.full(len(t), np.nan)
    high = df["comfort_high_c"].to_numpy() if "comfort_high_c" in df.columns else np.full(len(t), np.nan)
    violation = np.maximum(low - t, 0.0) + np.maximum(t - high, 0.0)
    return float(violation.sum() * dt_h)


def _run_episode_trace(
    env,
    policy_fn,
    policy_name: str,
    seed: Optional[int] = None,
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
        row.update(_extract_action_channels(action))
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


def _plot_comparison_boxplots(
    df: pd.DataFrame,
    fig_path: Optional[str] = None,
    title: str = "",
) -> plt.Figure:
    """
    Four-panel box plot comparing RL vs Baseline per episode:
      1. Energy cost (USD / episode)
      2. Discomfort (°C·h / episode)
      3. Time outside comfort (% of timesteps)
      4. Max temperature deviation (°C)
    """
    fig, axes = plt.subplots(1, 4, figsize=(18, 6))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    ax_cost, ax_comfort, ax_pct, ax_max = axes

    rl_color, bl_color = "#4c9be8", "#f4a261"

    # --- Energy cost ---
    cost_data = [df["rl_energy_cost_usd"].to_numpy(), df["baseline_energy_cost_usd"].to_numpy()]
    bp1 = ax_cost.boxplot(cost_data, labels=["RL", "Baseline"], patch_artist=True, widths=0.5)
    bp1["boxes"][0].set_facecolor(rl_color)
    bp1["boxes"][1].set_facecolor(bl_color)
    ax_cost.set_ylabel("Energy cost (USD / episode)")
    ax_cost.set_title("Energy cost")
    ax_cost.grid(axis="y", alpha=0.3)

    # --- Discomfort degree-hours ---
    dis_data = [df["rl_discomfort_deg_h"].to_numpy(), df["baseline_discomfort_deg_h"].to_numpy()]
    bp2 = ax_comfort.boxplot(dis_data, labels=["RL", "Baseline"], patch_artist=True, widths=0.5)
    bp2["boxes"][0].set_facecolor(rl_color)
    bp2["boxes"][1].set_facecolor(bl_color)
    ax_comfort.set_ylabel("Discomfort (°C·h / episode)")
    ax_comfort.set_title("Thermal discomfort")
    ax_comfort.grid(axis="y", alpha=0.3)

    # --- % timesteps outside comfort ---
    if "rl_pct_outside_comfort" in df.columns:
        pct_data = [df["rl_pct_outside_comfort"].to_numpy(), df["baseline_pct_outside_comfort"].to_numpy()]
        bp3 = ax_pct.boxplot(pct_data, labels=["RL", "Baseline"], patch_artist=True, widths=0.5)
        bp3["boxes"][0].set_facecolor(rl_color)
        bp3["boxes"][1].set_facecolor(bl_color)
    ax_pct.set_ylabel("Time outside comfort (%)")
    ax_pct.set_title("% Outside comfort")
    ax_pct.grid(axis="y", alpha=0.3)

    # --- Max temperature deviation ---
    if "rl_max_temp_deviation_c" in df.columns:
        max_data = [df["rl_max_temp_deviation_c"].to_numpy(), df["baseline_max_temp_deviation_c"].to_numpy()]
        bp4 = ax_max.boxplot(max_data, labels=["RL", "Baseline"], patch_artist=True, widths=0.5)
        bp4["boxes"][0].set_facecolor(rl_color)
        bp4["boxes"][1].set_facecolor(bl_color)
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
        rl_df, rl_m, comfort_band_c = _run_episode_trace(
            env_rl,
            _rl_policy,
            "rl",
            seed=ep,
        )
        env_rl.close()

        # Baseline episode (same start, same seed)
        env_b = trainer.create_env(params=p, training_mode=training_mode)
        baseline = ThermostatBaselineController(
            comfort_band_k=env_b.comfort_band_k,
            working_hours=env_b.working_hours,
            night_setback_k=getattr(env_b, "night_setback_k", 0.0),
        )
        b_df, b_m, _ = _run_episode_trace(
            env_b,
            lambda obs, env: baseline.get_action(obs, env),
            "baseline",
            seed=ep,
        )
        env_b.close()

        if save_traces:
            rl_df.to_csv(os.path.join(period_dir, f"episode_{ep:02d}_rl_trace.csv"), index=False)
            b_df.to_csv(os.path.join(period_dir, f"episode_{ep:02d}_baseline_trace.csv"), index=False)

        if ep < n_plot_episodes:
            _plot_episode_trace_6panel(
                rl_df, comfort_band_c,
                title=f"{period_name.upper()} Episode {ep} - RL",
                fig_path=os.path.join(period_dir, f"episode_{ep:02d}_rl_plot.png"),
            )
            _plot_episode_trace_6panel(
                b_df, comfort_band_c,
                title=f"{period_name.upper()} Episode {ep} - Baseline",
                fig_path=os.path.join(period_dir, f"episode_{ep:02d}_baseline_plot.png"),
            )
            _plot_energy_prices(
                rl_df,
                title=f"{period_name.upper()} Episode {ep} - Energy Prices",
                fig_path=os.path.join(period_dir, f"episode_{ep:02d}_prices_plot.png"),
            )

        def _comfort_stats(trace_df):
            """Returns (pct_outside, max_deviation_c) for an episode trace."""
            if trace_df.empty:
                return 0.0, 0.0
            t = trace_df["room_temp_c"].to_numpy()
            low = trace_df["comfort_low_c"].to_numpy() if "comfort_low_c" in trace_df.columns else np.full(len(t), np.nan)
            high = trace_df["comfort_high_c"].to_numpy() if "comfort_high_c" in trace_df.columns else np.full(len(t), np.nan)
            violation = np.maximum(low - t, 0.0) + np.maximum(t - high, 0.0)
            pct_outside = float((violation > 0).mean() * 100.0)
            max_dev = float(violation.max())
            return pct_outside, max_dev

        rl_pct_outside, rl_max_dev = _comfort_stats(rl_df)
        b_pct_outside, b_max_dev = _comfort_stats(b_df)

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
        title=f"{period_name.upper()} — RL vs Baseline",
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
    test_period_start="2023-10-01",
    test_period_end="2024-03-24",
) -> Dict[str, Any]:
    """
    Compare trained RL policy vs thermostat baseline on winter val/test splits.
    Saves per-episode traces, metrics CSVs, and 4-panel plots.
    """
    if trainer.model is None:
        raise ValueError("Trainer has no model. Train or load model first.")

    os.makedirs(output_dir, exist_ok=True)
    base = trainer.base_params.copy()

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

    # --- 6) Actions ---
    action_cols = {
        "action_supply":      "Supply air",
        "action_boiler":      "Boiler",
        "action_reheat":      "Reheat (shared)",
        "action_damper_mean": "Damper mean",
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

def _apply_training_mode_overrides(env, training_mode: Optional[str]) -> None:
    if training_mode != "always_occupied":
        return

    # 24/7 occupancy = 1.0, independent of working hours
    if hasattr(env, "occupancy_model"):
        env.occupancy_model = "constant"
    if hasattr(env, "occupancy_per_zone"):
        env.occupancy_per_zone = 1.0
    if hasattr(env, "working_hours"):
        env.working_hours = (0.0, 24.0)
    if hasattr(env, "_build_occupancies"):
        env.occupancies = env._build_occupancies(seed=None)
