import csv
import os
import torch # pyright: ignore[reportMissingImports]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, Callable, Dict, List, Optional, Tuple
from stable_baselines3 import DDPG
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import TD3
from sb3_contrib import CrossQ, TQC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
import itertools

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
        normalized = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
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
        seed: int = 42,
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
        self.seed = seed
        if self.algo_name not in ["sac", "td3", "ddpg", "tqc", "ppo", "crossq"]:
            raise ValueError(f"Unsupported algorithm: {algo}. Choose 'sac', 'td3', 'tqc', 'ddpg', 'ppo', or 'crossq'.")

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
            "seed": self.seed,
        }

        # Algo-specific parameters
        if self.algo_name == "sac":
            common_params["buffer_size"] = buffer_size
            common_params["batch_size"] = batch_size
            common_params.setdefault("learning_starts", 5000)  # ~35 simulated days of random exploration
            AlgoClass = SAC
        elif self.algo_name == "tqc":
            common_params["buffer_size"] = buffer_size
            common_params["batch_size"] = batch_size
            common_params.setdefault("learning_starts", 5000)
            AlgoClass = TQC
        elif self.algo_name == "td3":
            common_params["buffer_size"] = buffer_size
            common_params["batch_size"] = batch_size
            common_params.setdefault("policy_delay", 2)
            common_params.setdefault("target_policy_noise", 0.2)
            common_params.setdefault("target_noise_clip", 0.5)
            AlgoClass = TD3
        elif self.algo_name == "crossq":
            common_params["buffer_size"] = buffer_size
            common_params["batch_size"] = batch_size
            common_params.setdefault("learning_starts", 5000)
            algo_kwargs.pop("policy_kwargs", None)  # use CrossQ default: pi=[256,256], qf=[1024,1024]
            AlgoClass = CrossQ
        elif self.algo_name == "ppo":
            # PPO is on-policy: no replay buffer; explores via stochastic Gaussian policy
            common_params.setdefault("n_steps", 1024)   # ~1 episode (7d × 24h × 6 steps/h = 1008 steps)
            common_params.setdefault("batch_size", 512) # must divide n_steps; 1024/512 = 2 minibatches
            common_params.setdefault("n_epochs", 10)    # gradient passes per rollout
            common_params.setdefault("gamma", 0.99)
            common_params.setdefault("gae_lambda", 0.95)
            common_params.setdefault("ent_coef", 0.01)  # small entropy bonus to keep exploration alive
            common_params.setdefault("clip_range", 0.2)
            AlgoClass = PPO
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
        elif algo == "tqc":
            self.model = TQC.load(filepath)
        elif algo == "td3":
            self.model = TD3.load(filepath)
        elif algo == "ddpg":
            self.model = DDPG.load(filepath)
        elif algo == "ppo":
            self.model = PPO.load(filepath)
        elif algo == "crossq":
            self.model = CrossQ.load(filepath)
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

def _merge_factory_kwargs(
    default_factory_kwargs: Optional[Dict[str, Any]],
    factory_kwargs: Optional[Dict[str, Any]],
    training_mode: Optional[str] = None,
) -> Dict[str, Any]:
    merged = {**(default_factory_kwargs or {}), **(factory_kwargs or {})}
    if training_mode is not None:
        merged["training_mode"] = training_mode
    return merged
