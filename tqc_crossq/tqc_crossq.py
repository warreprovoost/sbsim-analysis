from typing import Any, ClassVar, TypeVar

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule

from tqc_crossq.utils import quantile_huber_loss
from tqc_crossq.policies import Actor, MlpPolicy, QuantileCritic, TQCCrossQPolicy

SelfTQCCrossQ = TypeVar("SelfTQCCrossQ", bound="TQCCrossQ")


class TQCCrossQ(OffPolicyAlgorithm):
    """
    TQC + CrossQ: Truncated Quantile Critics with BatchRenorm and no target network.

    Combines:
      * TQC (https://arxiv.org/abs/2005.04269): distributional quantile critic with
        truncation of the top-k quantiles to control overestimation bias.
      * CrossQ (https://openreview.net/forum?id=PczQtTsTIX): BatchRenorm layers in the
        actor and critic, no target network, joint forward pass over (obs, next_obs)
        so batch statistics cover the mixture distribution. Delayed actor updates
        replace the role of target networks for stability.

    :param policy: The policy model to use (MlpPolicy)
    :param env: The environment to learn from
    :param learning_rate: Learning rate
    :param buffer_size: Size of the replay buffer
    :param learning_starts: Steps before learning starts
    :param batch_size: Minibatch size
    :param gamma: Discount factor
    :param train_freq: Update the model every ``train_freq`` steps
    :param gradient_steps: How many gradient steps to do after each rollout
    :param action_noise: Action noise
    :param replay_buffer_class: Replay buffer class
    :param replay_buffer_kwargs: Replay buffer kwargs
    :param optimize_memory_usage: Memory-efficient replay buffer variant
    :param ent_coef: Entropy regularization coefficient (or ``"auto"``)
    :param target_entropy: Target entropy for automatic ent_coef
    :param top_quantiles_to_drop_per_net: Number of quantiles to drop per critic
    :param policy_delay: Update the actor (and ent_coef) every ``policy_delay`` critic updates
    :param use_sde: Whether to use gSDE
    :param sde_sample_freq: gSDE noise resampling frequency
    :param stats_window_size: Rollout logging window
    :param tensorboard_log: Tensorboard log directory
    :param policy_kwargs: Additional policy kwargs
    :param verbose: Verbosity level
    :param seed: Random seed
    :param device: Device to run on
    :param _init_setup_model: Whether to build the network at instance creation
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        # TODO: CnnPolicy / MultiInputPolicy
    }
    policy: TQCCrossQPolicy
    actor: Actor
    critic: QuantileCritic

    def __init__(
        self,
        policy: str | type[TQCCrossQPolicy],
        env: GymEnv | str,
        learning_rate: float | Schedule = 1e-3,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        gamma: float = 0.99,
        train_freq: int | tuple[int, str] = 1,
        gradient_steps: int = 1,
        action_noise: ActionNoise | None = None,
        replay_buffer_class: type[ReplayBuffer] | None = None,
        replay_buffer_kwargs: dict[str, Any] | None = None,
        optimize_memory_usage: bool = False,
        ent_coef: str | float = "auto",
        target_entropy: str | float = "auto",
        top_quantiles_to_drop_per_net: int = 2,
        policy_delay: int = 3,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        stats_window_size: int = 100,
        tensorboard_log: str | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        verbose: int = 0,
        seed: int | None = None,
        device: th.device | str = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=1.0,  # no target network — kept only to satisfy the base class
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            supported_action_spaces=(spaces.Box,),
            support_multi_env=True,
        )

        self.target_entropy = target_entropy
        self.log_ent_coef: th.Tensor | None = None
        self.ent_coef = ent_coef
        self.ent_coef_optimizer: th.optim.Adam | None = None
        self.top_quantiles_to_drop_per_net = top_quantiles_to_drop_per_net
        self.policy_delay = policy_delay

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()

        if self.target_entropy == "auto":
            self.target_entropy = float(-np.prod(self.env.action_space.shape).astype(np.float32))  # type: ignore
        else:
            self.target_entropy = float(self.target_entropy)

        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

            self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))
        else:
            self.ent_coef_tensor = th.tensor(float(self.ent_coef), device=self.device)

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # BatchRenorm: leave default training mode True; we will toggle
        # set_bn_training_mode() around forward passes to control whether running
        # stats are updated.
        self.policy.set_training_mode(True)

        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        n_target_quantiles = self.critic.quantiles_total - self.top_quantiles_to_drop_per_net * self.critic.n_critics

        for _ in range(gradient_steps):
            self._n_updates += 1
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            discounts = self.gamma

            if self.use_sde:
                self.actor.reset_noise()

            # ----- entropy coefficient -----
            if self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # ----- next actions (no grad through actor BN stats) -----
            with th.no_grad():
                self.actor.set_bn_training_mode(False)
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)

            # ----- joint forward pass through critic -----
            # Concatenate (obs, next_obs) and (actions, next_actions) so the BatchRenorm
            # layers see the mixture distribution in a single forward pass. This is
            # what allows CrossQ to drop the target network.
            all_obs = th.cat([replay_data.observations, replay_data.next_observations], dim=0)
            all_actions = th.cat([replay_data.actions, next_actions], dim=0)

            self.critic.set_bn_training_mode(True)
            # Shape: (2 * batch_size, n_critics, n_quantiles)
            all_quantiles = self.critic(all_obs, all_actions)
            self.critic.set_bn_training_mode(False)

            # Split back: each is (batch_size, n_critics, n_quantiles)
            current_quantiles, next_quantiles = th.split(all_quantiles, batch_size, dim=0)

            with th.no_grad():
                next_quantiles = next_quantiles.detach()
                # Sort over (n_critics * n_quantiles), drop the top
                # `top_quantiles_to_drop_per_net * n_critics` to control overestimation.
                next_quantiles_flat = next_quantiles.reshape(batch_size, -1)
                next_quantiles_sorted, _ = th.sort(next_quantiles_flat)
                next_quantiles_truncated = next_quantiles_sorted[:, :n_target_quantiles]

                # td error + entropy term
                target_quantiles = next_quantiles_truncated - ent_coef * next_log_prob.reshape(-1, 1)
                target_quantiles = replay_data.rewards + (1 - replay_data.dones) * discounts * target_quantiles
                # (batch_size, 1, n_target_quantiles) so it broadcasts against (batch_size, n_critics, n_quantiles)
                target_quantiles.unsqueeze_(dim=1)

            critic_loss = quantile_huber_loss(current_quantiles, target_quantiles, sum_over_quantiles=False)
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # ----- delayed actor / ent_coef update -----
            if self._n_updates % self.policy_delay == 0:
                # Actor BN stats: update in this forward pass.
                self.actor.set_bn_training_mode(True)
                actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
                log_prob = log_prob.reshape(-1, 1)
                self.actor.set_bn_training_mode(False)

                if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                    ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()  # type: ignore[operator]
                    ent_coef_losses.append(ent_coef_loss.item())

                    self.ent_coef_optimizer.zero_grad()
                    ent_coef_loss.backward()
                    self.ent_coef_optimizer.step()

                # Critic forward for the actor loss: do NOT update BN running stats here
                # (they were updated in the joint pass above on the on-policy batch).
                self.critic.set_bn_training_mode(False)
                # Mean over quantiles, then mean over critics: (batch_size, 1)
                qf_pi = self.critic(replay_data.observations, actions_pi).mean(dim=2).mean(dim=1, keepdim=True)
                actor_loss = (ent_coef * log_prob - qf_pi).mean()
                actor_losses.append(actor_loss.item())

                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def learn(
        self: SelfTQCCrossQ,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "TQCCrossQ",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfTQCCrossQ:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> list[str]:
        return [*super()._excluded_save_params(), "actor", "critic"]

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        if self.ent_coef_optimizer is not None:
            saved_pytorch_variables = ["log_ent_coef"]
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables = ["ent_coef_tensor"]
        return state_dicts, saved_pytorch_variables
