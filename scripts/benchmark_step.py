"""
Benchmark SAC training to identify bottlenecks.

Measures during actual training:
  - Rollout phase: env.step() calls (building physics + obs + reward)
  - Gradient update phase: SAC replay buffer sampling + backprop
  - Breakdown of env.step() internals

Usage:
  python scripts/benchmark_step.py --steps 500 --floorplan office_4room \
      --weather_csv /user/gent/453/vsc45342/thesis/weather_data/oslo_weather_multiyear.csv
"""
import torch; print("using CUDA", torch.cuda.is_available())
import argparse
import time
import sys
import os
import io
import cProfile
import pstats
import numpy as np
from collections import defaultdict

THESIS_ROOT = os.path.expanduser("~/thesis")
sys.path.insert(0, os.path.join(THESIS_ROOT, "sbsim-analysis"))
sys.path.insert(0, os.path.join(THESIS_ROOT, "sbsim"))

from smart_control_analysis.building_factory import building_factory, get_base_params
from smart_control_analysis.gym_wrapper import BuildingGymEnv, _AlwaysZeroOccupancy
from smart_control_analysis.energy_prices import get_electricity_price_usd_per_ws, get_gas_price_usd_per_1000ft3
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer


class Timer:
    def __init__(self):
        self.totals = defaultdict(float)
        self.counts = defaultdict(int)

    def time(self, name):
        return _TimerCtx(self, name)

    def report(self, total_elapsed):
        print(f"\n{'='*65}")
        print(f"{'Section':<38} {'Total(s)':>8} {'Per step(ms)':>13} {'%wall':>7}")
        print(f"{'-'*65}")
        for name, t in sorted(self.totals.items(), key=lambda x: -x[1]):
            n = self.counts[name]
            pct = 100 * t / total_elapsed if total_elapsed > 0 else 0
            print(f"{name:<38} {t:>8.3f} {1000*t/max(n,1):>13.2f} {pct:>6.1f}%")
        print(f"{'='*65}")
        print(f"{'TOTAL wall time':<38} {total_elapsed:>8.3f}s")


class _TimerCtx:
    def __init__(self, timer, name):
        self.timer = timer
        self.name = name

    def __enter__(self):
        self._t = time.perf_counter()

    def __exit__(self, *_):
        self.timer.totals[self.name] += time.perf_counter() - self._t
        self.timer.counts[self.name] += 1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=500, help="Total training steps to run")
    p.add_argument("--floorplan", default="office_4room")
    p.add_argument("--weather_csv", required=True)
    p.add_argument("--learning_starts", type=int, default=100,
                   help="Steps before gradient updates begin (SAC warmup)")
    p.add_argument("--gradient_steps", type=int, default=1,
                   help="Gradient updates per env step (after learning_starts)")
    return p.parse_args()


def make_env(args):
    base_params = get_base_params().copy()
    base_params["weather_source"] = "replay"
    base_params["weather_csv_path"] = args.weather_csv
    base_params["floorplan"] = args.floorplan
    base_params["time_zone"] = "America/Los_Angeles"
    base_params["energy_weight"] = 1.0
    base_params["action_design"] = "reheat_per_zone"
    base_params["max_steps"] = int(7 * 24 * 3600 / base_params["time_step_sec"])
    _, env = building_factory(base_params.copy(), training_mode="full")
    return env


def main():
    args = parse_args()
    timer = Timer()

    print("Building environment...")
    env = make_env(args)
    obs, _ = env.reset()

    print("Building SAC model...")
    model = SAC(
        "MlpPolicy", env,
        verbose=0,
        learning_starts=args.learning_starts,
        gradient_steps=args.gradient_steps,
        batch_size=256,
        buffer_size=10_000,
    )

    print(f"Device: {model.device}")

    # Warmup: trigger Numba JIT before measuring
    print("Warming up (20 steps)...")
    for _ in range(20):
        a = env.action_space.sample()
        o, r, te, tr, _ = env.step(a)
        if te or tr:
            env.reset()
    env.reset()
    print("Warmup done.\n")

    # Timing callback that intercepts rollout vs training phases
    from stable_baselines3.common.callbacks import BaseCallback

    class TimingCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.t_rollout = 0.0
            self.t_gradient = 0.0
            self._t = None
            self.n_rollout_steps = 0
            self.n_gradient_steps = 0

        def _on_step(self) -> bool:
            # Called after each env.step() during rollout
            if self._t is not None:
                self.t_rollout += time.perf_counter() - self._t
                self.n_rollout_steps += 1
            self._t = time.perf_counter()
            return True

        def on_training_start(self, locals_, globals_):
            self._t = time.perf_counter()

        def on_rollout_end(self):
            # Rollout finished, gradient update about to start
            if self._t is not None:
                self.t_rollout += time.perf_counter() - self._t
            self._t = time.perf_counter()

        def on_training_end(self):
            if self._t is not None:
                self.t_gradient += time.perf_counter() - self._t

    cb = TimingCallback()

    pr = cProfile.Profile()
    pr.enable()
    t_wall_start = time.perf_counter()

    model.learn(total_timesteps=args.steps, callback=cb, progress_bar=False)

    t_wall = time.perf_counter() - t_wall_start
    pr.disable()

    print(f"\n{'='*55}")
    print(f"Total wall time:    {t_wall:.2f}s")
    print(f"Steps/sec:          {args.steps / t_wall:.1f}")
    print(f"{'='*55}")
    print(f"Rollout (env.step): {cb.t_rollout:.2f}s  ({100*cb.t_rollout/t_wall:.1f}%)")
    print(f"Gradient updates:   {cb.t_gradient:.2f}s  ({100*cb.t_gradient/t_wall:.1f}%)")
    print(f"Other (SB3 overhead):{t_wall - cb.t_rollout - cb.t_gradient:.2f}s")

    print("\n--- cProfile: top 25 by cumtime ---")
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumtime")
    ps.print_stats(25)
    print(s.getvalue())


if __name__ == "__main__":
    main()