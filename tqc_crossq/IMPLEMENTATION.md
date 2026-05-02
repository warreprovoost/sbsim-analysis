# TQC + CrossQ Implementation Notes

This document explains the `tqc_crossq` module: what it is, why it exists, and exactly what was changed/added compared to vanilla TQC.

## What it is

`TQCCrossQ` is a hybrid of two existing algorithms in `sb3_contrib`:

- **TQC** (Truncated Quantile Critics, [Kuznetsov et al. 2020](https://arxiv.org/abs/2005.04269)) — a SAC-style off-policy actor-critic where each critic outputs a *distribution* of `n_quantiles` Q-values rather than a scalar. Overestimation bias is controlled by sorting all `n_critics * n_quantiles` predictions at the next state and dropping the top-k.
- **CrossQ** ([Bhatt et al. 2024](https://openreview.net/forum?id=PczQtTsTIX)) — a SAC-style algorithm that **removes the target network entirely** and replaces it with BatchRenorm layers in actor and critic. Stability comes from doing a single *joint forward pass* through the critic on the concatenated batch `(obs, next_obs)` so the BN running statistics see the mixture distribution. Actor updates are delayed (`policy_delay=3`) to compensate for the missing target network.

The combination is "TQC without target nets, stabilized by BatchRenorm + joint forward pass + delayed actor updates." The truncation mechanism from TQC is preserved; the `critic_target` and Polyak update from TQC are gone.

## Why these specific design choices

The CrossQ paper's central claim is that *naive* SAC + BN fails because the running statistics on the online critic and the target critic drift apart — the target critic gets out-of-distribution `next_obs` inputs that its BN layers were never updated on. Two ways to fix this:

1. Keep target nets and Polyak-update the BN buffers too. (`sb3_contrib/tqc/tqc.py` already does this with `batch_norm_stats` / `batch_norm_stats_target`.) This does not solve the core problem: the online critic's `running_mean` / `running_var` are updated on `obs`, while the target critic sees `next_obs` — a shifted distribution. Their running stats drift apart, making `next_obs` effectively out-of-distribution to the target critic's BN layers. That is the instability.
2. Drop target nets, do a joint forward pass on `(obs, next_obs)` so BN sees the mixture. There is one set of running stats, updated on the joint distribution, so no drift is possible. This is CrossQ's contribution.

We chose **option 2** for `TQCCrossQ`. BatchRenorm is not an optional component — it is the reason the target network can be dropped. If you want TQC without BatchRenorm, use the existing `TQC`.

## File layout

```
sb3_contrib/tqc_crossq/
├── __init__.py          # exports TQCCrossQ, MlpPolicy
├── policies.py          # Actor, QuantileCritic, TQCCrossQPolicy
├── tqc_crossq.py        # TQCCrossQ algorithm
└── IMPLEMENTATION.md    # this file
```

It's also re-exported from `sb3_contrib/__init__.py` as `TQCCrossQ`.

## What changed vs. vanilla TQC

### `policies.py`

#### `Actor`
- Added params: `batch_norm_momentum`, `batch_norm_eps`, `renorm_warmup_steps`.
- Builds `pre_linear_modules = [partial(BatchRenorm1d, ...)]` and threads them into `create_mlp(...)` so each `Linear` is preceded by a `BatchRenorm1d`. After the MLP, an extra `BatchRenorm1d(net_arch[-1])` is appended so the latent representation feeding `mu` / `log_std` is also normalized. (Same pattern as [crossq/policies.py:107-117](../crossq/policies.py).)
- Added `set_bn_training_mode(mode)` — toggles `train()` on every `BatchRenorm1d` submodule only, leaving everything else untouched. This is the lever the algorithm uses to control which forward passes update the running stats.

#### `QuantileCritic` (CrossQ's critic with TQC's head)
- Same BatchRenorm wiring as `Actor`, applied per-q-network.
- **Output head**: each q-network outputs `n_quantiles` values (TQC's distributional head), not a scalar (CrossQ's).
- **`forward` return shape**: CrossQ's critic returns a tuple of `(B, 1)` tensors and cats them to `(B, n_critics)`. `QuantileCritic` instead stacks to `(B, n_critics, n_quantiles)` — required so the truncation step can sort across all critics and quantiles in one operation.
- Stores `n_quantiles`, `n_critics`, and `quantiles_total = n_quantiles * n_critics` for the algorithm to read.
- Has `set_bn_training_mode(mode)`.

#### `TQCCrossQPolicy`
- No `critic_target`. The `_build` method creates only `actor` and `critic`, plus their optimizers.
- Default `net_arch = {"pi": [256, 256], "qf": [1024, 1024]}` — wider critic required for stable BatchNorm (CrossQ paper). Vanilla TQC defaulted to `[256, 256]` for both.
- Default Adam betas `(0.5, 0.999)` when the optimizer is Adam/AdamW (CrossQ convention). User-supplied `optimizer_kwargs` override this.
- Constructor params: `batch_norm_momentum`, `batch_norm_eps`, `renorm_warmup_steps` bubbled up from the actor/critic; `n_quantiles` / `n_critics` from TQC.

#### What was removed
- No `CnnPolicy` / `MultiInputPolicy` (CrossQ doesn't have these either; ports would need to handle BatchRenorm in the features extractor).

### `tqc_crossq.py`

The algorithm class. `train()` is best understood as **CrossQ's train loop with four TQC-specific lines dropped in**. Here is exactly what comes from where:

#### From CrossQ (unchanged)
- `policy_delay=3` — actor and ent_coef update every 3 critic steps, replacing the role of the target network for stability.
- The joint forward pass structure — concatenate `(obs, next_obs)`, forward through critic with BN training on, split back:

  ```python
  all_obs     = th.cat([obs, next_obs], dim=0)
  all_actions = th.cat([actions, next_actions], dim=0)
  self.critic.set_bn_training_mode(True)
  all_quantiles = self.critic(all_obs, all_actions)   # (2B, n_critics, n_quantiles)
  self.critic.set_bn_training_mode(False)
  current_quantiles, next_quantiles = th.split(all_quantiles, batch_size, dim=0)
  ```

- All `set_bn_training_mode` toggling around actor and critic forward passes.
- Entropy coefficient update, optimizer steps, logging.

#### From TQC (dropped into CrossQ's loop)
These four lines are the only TQC-specific additions. Everything else is CrossQ:

1. **`n_target_quantiles`** — how many quantiles to keep after truncation:
   ```python
   n_target_quantiles = self.critic.quantiles_total - self.top_quantiles_to_drop_per_net * self.critic.n_critics
   ```

2. **Truncation** — sort all `n_critics * n_quantiles` next-state quantiles and drop the top-k to control overestimation. `next_quantiles` here comes from the joint-pass split instead of a `critic_target(...)` call, but the logic is identical to TQC:
   ```python
   next_quantiles_flat = next_quantiles.reshape(batch_size, -1)
   next_quantiles_sorted, _ = th.sort(next_quantiles_flat)
   next_quantiles_truncated = next_quantiles_sorted[:, :n_target_quantiles]
   ```

3. **`quantile_huber_loss`** — distributional critic loss instead of MSE:
   ```python
   critic_loss = quantile_huber_loss(current_quantiles, target_quantiles, sum_over_quantiles=False)
   ```

4. **Actor loss aggregation** — mean over quantiles then over critics to get a scalar Q-value:
   ```python
   qf_pi = self.critic(obs, actions_pi).mean(dim=2).mean(dim=1, keepdim=True)
   ```

#### Removed from TQC
- `tau`, `target_update_interval`, `critic_target`, `polyak_update(...)`, `batch_norm_stats` / `batch_norm_stats_target` — all gone because there is no target network.
- `tau` is still passed to `OffPolicyAlgorithm.__init__` as `1.0` solely to satisfy the base class signature; it is never used.

#### Save / load
- `_excluded_save_params` excludes only `actor` and `critic` (no `critic_target`).
- `_get_torch_save_params` saves `policy`, `actor.optimizer`, `critic.optimizer`, plus the entropy coefficient — same shape as CrossQ's.

## Hyperparameter defaults

| Param | TQCCrossQ | TQC | CrossQ |
|---|---|---|---|
| `learning_rate` | `1e-3` | `3e-4` | `1e-3` |
| `net_arch` | `{"pi":[256,256],"qf":[1024,1024]}` | `[256,256]` | `{"pi":[256,256],"qf":[1024,1024]}` |
| `tau` | (n/a) | `0.005` | (n/a) |
| `policy_delay` | `3` | (n/a) | `3` |
| `top_quantiles_to_drop_per_net` | `2` | `2` | (n/a) |
| `n_quantiles` | `25` | `25` | (n/a) |
| Adam betas | `(0.5, 0.999)` | `(0.9, 0.999)` | `(0.5, 0.999)` |

## Smoke tests run

Both run on `Pendulum-v1` with `sb3-env`:

1. **Instantiate + 64-step learn**: builds successfully, `_n_updates` advances correctly, no shape errors. Network printout confirms BatchRenorm1d is interleaved correctly in both actor and critic.
2. **Save / load round-trip**: model is saved to a zip, reloaded, and `predict(obs, deterministic=True)` returns identical actions before and after — confirms BatchRenorm running buffers and policy params round-trip correctly.

## Things to be aware of

- **No CnnPolicy / MultiInputPolicy.** Image and dict obs spaces aren't supported. Adding them requires plumbing BatchRenorm into the features extractor — same work as it would be for CrossQ.
- **BN-mode toggling matters.** Four rules govern when BN running stats update:
  - Critic joint forward pass over `(obs, next_obs)`: BN training **ON** (the only place critic running stats update each step).
  - Critic forward pass for actor loss: BN training **OFF** (`actions_pi` are not from the replay distribution).
  - Actor `action_log_prob(obs)` for actor loss: BN training **ON** (on-policy `obs`).
  - Actor `action_log_prob(next_obs)` for target: BN training **OFF** (under `th.no_grad()`).
- **`tau=1.0`** is hardcoded in the `super().__init__` call only because `OffPolicyAlgorithm` requires it. It is never used — there's no target network to update.
- **Policy delay vs. target nets.** With `policy_delay=3` the actor/ent_coef update every 3 critic steps. Lowering this to 1 may destabilize training (no target net + immediate actor follow-up); raising it slows convergence. CrossQ tuned this empirically.
- **Wider critic is required for stable BatchNorm.** The CrossQ paper shows that a narrow critic causes BatchNorm to become unstable — the per-layer feature distributions are too constrained for the running statistics to be reliable. The default `qf=[1024,1024]` is the minimum the authors recommend.

## What's not done

- No CnnPolicy / MultiInputPolicy.
- No dedicated unit tests under `tests/`. The existing `tests/test_train_eval_mode.py` patterns for CrossQ would be the right place to add them.
- No documentation page under `docs/modules/`.
- No entry in `CHANGELOG.md`.
