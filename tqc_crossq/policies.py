from functools import partial
from typing import Any

import torch as th
from gymnasium import spaces
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution, StateDependentNoiseDistribution
from stable_baselines3.common.policies import BaseModel, BasePolicy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    create_mlp,
    get_actor_critic_arch,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from torch import nn

from tqc_crossq.torch_layers import BatchRenorm1d

# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20


class Actor(BasePolicy):
    """
    Actor network (policy) for TQC-CrossQ.
    Identical to CrossQ's actor: a squashed Gaussian policy with BatchRenorm layers.
    The TQC-specific changes live in QuantileCritic and the train loop, not here.

    :param observation_space: Observation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE.
    :param use_expln: Use ``expln()`` instead of ``exp()`` to ensure a positive std (gSDE).
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param normalize_images: Whether to normalize images by dividing by 255.0.
    :param batch_norm_momentum: The rate of convergence for the batch renormalization statistics
    :param batch_norm_eps: A small value added to the variance to prevent division by zero
    :param renorm_warmup_steps: Number of steps to warm up BatchRenorm statistics
    """

    action_space: spaces.Box

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: list[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
        batch_norm_momentum: float = 0.01,
        batch_norm_eps: float = 0.001,
        renorm_warmup_steps: int = 100_000,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        self.use_sde = use_sde
        self.sde_features_extractor = None
        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        self.log_std_init = log_std_init
        self.use_expln = use_expln
        self.full_std = full_std
        self.clip_mean = clip_mean

        action_dim = get_action_dim(self.action_space)

        pre_linear_modules = [
            partial(
                BatchRenorm1d,
                momentum=batch_norm_momentum,
                eps=batch_norm_eps,
                warmup_steps=renorm_warmup_steps,
            )
        ]

        latent_pi_net = create_mlp(
            features_dim,
            -1,
            net_arch,
            activation_fn,
            pre_linear_modules=pre_linear_modules,  # type: ignore[arg-type]
        )

        if net_arch:
            latent_pi_net.append(pre_linear_modules[0](net_arch[-1]))

        self.latent_pi = nn.Sequential(*latent_pi_net)
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else features_dim

        if self.use_sde:
            self.action_dist = StateDependentNoiseDistribution(
                action_dim, full_std=full_std, use_expln=use_expln, learn_features=True, squash_output=True
            )
            self.mu, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=last_layer_dim, latent_sde_dim=last_layer_dim, log_std_init=log_std_init
            )
            if clip_mean > 0.0:
                self.mu = nn.Sequential(self.mu, nn.Hardtanh(min_val=-clip_mean, max_val=clip_mean))
        else:
            self.action_dist = SquashedDiagGaussianDistribution(action_dim)  # type: ignore[assignment]
            self.mu = nn.Linear(last_layer_dim, action_dim)
            self.log_std = nn.Linear(last_layer_dim, action_dim)  # type: ignore[assignment]

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                full_std=self.full_std,
                use_expln=self.use_expln,
                features_extractor=self.features_extractor,
                clip_mean=self.clip_mean,
            )
        )
        return data

    def get_std(self) -> th.Tensor:
        msg = "get_std() is only available when using gSDE"
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), msg
        return self.action_dist.get_std(self.log_std)

    def reset_noise(self, batch_size: int = 1) -> None:
        msg = "reset_noise() is only available when using gSDE"
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), msg
        self.action_dist.sample_weights(self.log_std, batch_size=batch_size)

    def get_action_dist_params(self, obs: PyTorchObs) -> tuple[th.Tensor, th.Tensor, dict[str, th.Tensor]]:
        features = self.extract_features(obs, self.features_extractor)
        latent_pi = self.latent_pi(features)
        mean_actions = self.mu(latent_pi)

        if self.use_sde:
            return mean_actions, self.log_std, dict(latent_sde=latent_pi)
        log_std = self.log_std(latent_pi)  # type: ignore[operator]
        log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean_actions, log_std, {}

    def forward(self, obs: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        return self.action_dist.actions_from_params(mean_actions, log_std, deterministic=deterministic, **kwargs)

    def action_log_prob(self, obs: PyTorchObs) -> tuple[th.Tensor, th.Tensor]:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        return self.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs)

    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        return self(observation, deterministic)

    def set_bn_training_mode(self, mode: bool) -> None:
        """
        Set the training mode of the BatchRenorm layers only.
        When training is True, the running statistics are updated.
        """
        for module in self.modules():
            if isinstance(module, BatchRenorm1d):
                module.train(mode)


class QuantileCritic(BaseModel):
    """
    Distributional quantile critic for TQC-CrossQ.
    Same head as TQC (n_quantiles outputs per critic) with BatchRenorm layers
    interleaved like CrossQ. There is no target critic — the joint forward pass
    on (obs, next_obs) provides correct batch statistics for the off-policy target.

    :param observation_space: Observation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Features extractor
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images
    :param n_quantiles: Number of quantiles per critic
    :param n_critics: Number of critic networks
    :param share_features_extractor: Whether to share the features extractor with the actor
    :param batch_norm_momentum: BatchRenorm momentum
    :param batch_norm_eps: BatchRenorm epsilon
    :param renorm_warmup_steps: BatchRenorm warmup steps
    """

    action_space: spaces.Box
    features_extractor: BaseFeaturesExtractor

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: list[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_quantiles: int = 25,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        batch_norm_momentum: float = 0.01,
        batch_norm_eps: float = 0.001,
        renorm_warmup_steps: int = 100_000,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        action_dim = get_action_dim(self.action_space)

        pre_linear_modules = [
            partial(
                BatchRenorm1d,
                momentum=batch_norm_momentum,
                eps=batch_norm_eps,
                warmup_steps=renorm_warmup_steps,
            )
        ]

        self.share_features_extractor = share_features_extractor
        self.n_quantiles = n_quantiles
        self.n_critics = n_critics
        self.quantiles_total = n_quantiles * n_critics
        self.q_networks: list[nn.Module] = []

        for idx in range(n_critics):
            qf_net_list = create_mlp(
                features_dim + action_dim,
                n_quantiles,
                net_arch,
                activation_fn,
                pre_linear_modules=pre_linear_modules,  # type: ignore[arg-type]
            )
            qf_net = nn.Sequential(*qf_net_list)
            self.add_module(f"qf{idx}", qf_net)
            self.q_networks.append(qf_net)

    def forward(self, obs: PyTorchObs, action: th.Tensor) -> th.Tensor:
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
        qvalue_input = th.cat([features, action], dim=1)
        # (batch_size, n_critics, n_quantiles)
        quantiles = th.stack(tuple(qf(qvalue_input) for qf in self.q_networks), dim=1)
        return quantiles

    def set_bn_training_mode(self, mode: bool) -> None:
        for module in self.modules():
            if isinstance(module, BatchRenorm1d):
                module.train(mode)


class TQCCrossQPolicy(BasePolicy):
    """
    Policy class (with both actor and critic) for TQC-CrossQ.

    Truncated Quantile Critics + CrossQ-style BatchRenorm. Has no target critic;
    the algorithm relies on a joint forward pass over (obs, next_obs) to
    compute batch statistics over the mixture distribution.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param batch_norm_momentum: BatchRenorm momentum
    :param batch_norm_eps: BatchRenorm epsilon
    :param renorm_warmup_steps: BatchRenorm warmup steps
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param use_expln: Use expln() instead of exp() (gSDE)
    :param clip_mean: Clip the mean output when using gSDE
    :param features_extractor_class: Features extractor to use
    :param features_extractor_kwargs: Keyword arguments for the features extractor
    :param normalize_images: Whether to normalize images
    :param optimizer_class: The optimizer to use
    :param optimizer_kwargs: Additional keyword arguments for the optimizer
    :param n_quantiles: Number of quantiles per critic
    :param n_critics: Number of critic networks
    :param share_features_extractor: Whether the actor and critic share the features extractor
    """

    actor: Actor
    critic: QuantileCritic

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: list[int] | dict[str, list[int]] | None = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        batch_norm_momentum: float = 0.01,
        batch_norm_eps: float = 0.001,
        renorm_warmup_steps: int = 100_000,
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: dict[str, Any] | None = None,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
        n_quantiles: int = 25,
        n_critics: int = 2,
        share_features_extractor: bool = False,
    ):
        if optimizer_kwargs is None:
            # CrossQ default: b1=0.5 in the original implementation
            optimizer_kwargs = {}
            if optimizer_class in [th.optim.Adam, th.optim.AdamW]:
                optimizer_kwargs["betas"] = (0.5, 0.999)

        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            # Wider critic required for stable BatchNorm (CrossQ paper).
            net_arch = {"pi": [256, 256], "qf": [1024, 1024]}

        actor_arch, critic_arch = get_actor_critic_arch(net_arch)

        self.batch_norm_params = {
            "batch_norm_momentum": batch_norm_momentum,
            "batch_norm_eps": batch_norm_eps,
            "renorm_warmup_steps": renorm_warmup_steps,
        }

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": actor_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
            **self.batch_norm_params,
        }
        self.actor_kwargs = self.net_args.copy()

        sde_kwargs = {
            "use_sde": use_sde,
            "log_std_init": log_std_init,
            "use_expln": use_expln,
            "clip_mean": clip_mean,
        }
        self.actor_kwargs.update(sde_kwargs)
        self.critic_kwargs = self.net_args.copy()
        self.critic_kwargs.update(
            {
                "n_quantiles": n_quantiles,
                "n_critics": n_critics,
                "net_arch": critic_arch,
                "share_features_extractor": share_features_extractor,
            }
        )

        self.share_features_extractor = share_features_extractor

        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        self.actor = self.make_actor()
        self.actor.optimizer = self.optimizer_class(  # type: ignore[call-arg]
            self.actor.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

        if self.share_features_extractor:
            self.critic = self.make_critic(features_extractor=self.actor.features_extractor)
            critic_parameters = [param for name, param in self.critic.named_parameters() if "features_extractor" not in name]
        else:
            self.critic = self.make_critic(features_extractor=None)
            critic_parameters = list(self.critic.parameters())

        self.critic.optimizer = self.optimizer_class(  # type: ignore[call-arg]
            critic_parameters,
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.net_args["activation_fn"],
                use_sde=self.actor_kwargs["use_sde"],
                log_std_init=self.actor_kwargs["log_std_init"],
                use_expln=self.actor_kwargs["use_expln"],
                clip_mean=self.actor_kwargs["clip_mean"],
                n_quantiles=self.critic_kwargs["n_quantiles"],
                n_critics=self.critic_kwargs["n_critics"],
                share_features_extractor=self.share_features_extractor,
                lr_schedule=self._dummy_schedule,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
                **self.batch_norm_params,
            )
        )
        return data

    def reset_noise(self, batch_size: int = 1) -> None:
        self.actor.reset_noise(batch_size=batch_size)

    def make_actor(self, features_extractor: BaseFeaturesExtractor | None = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return Actor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: BaseFeaturesExtractor | None = None) -> QuantileCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return QuantileCritic(**critic_kwargs).to(self.device)

    def forward(self, obs: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        return self.actor(observation, deterministic)

    def set_training_mode(self, mode: bool) -> None:
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.training = mode


MlpPolicy = TQCCrossQPolicy
