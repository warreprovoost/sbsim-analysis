#!/usr/bin/env python3
"""Benchmark FastCPUSimulator vs SimulatorFlexibleGeometries (original).

Measures steps/second for both simulators across all floorplans.
Reports speedup factor for thesis Section 3.3.

Usage:
    python scripts/benchmark_simulator.py
    python scripts/benchmark_simulator.py --n_steps 500 --n_runs 3
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../sbsim"))

from smart_control_analysis.building_factory import building_factory, get_base_params
from smart_control_analysis.custom_sbsim.fast_cpu_simulator import FastCPUSimulator
from smart_control.simulator.simulator_flexible_floor_plan import SimulatorFlexibleGeometries


def make_env(floorplan: str, use_fast: bool, weather_csv: str):
    """Build an env using either FastCPUSimulator or the original."""
    params = get_base_params().copy()
    params["weather_source"] = "replay"
    params["weather_csv_path"] = weather_csv
    params["floorplan"] = floorplan
    params["max_steps"] = 2000

    if use_fast:
        _, env = building_factory(params, training_mode="step")
        return env
    else:
        # Patch building_factory to use the original simulator
        import smart_control_analysis.building_factory as bf_mod
        original_sim_class = bf_mod.FastCPUSimulator
        bf_mod.FastCPUSimulator = SimulatorFlexibleGeometries
        try:
            _, env = building_factory(params, training_mode="step")
        finally:
            bf_mod.FastCPUSimulator = original_sim_class
        return env


def benchmark(env, n_steps: int) -> float:
    """Run n_steps and return steps/second."""
    obs, _ = env.reset(seed=0)
    action = env.action_space.sample()

    # Warmup (JIT compile for Numba on first run)
    for _ in range(5):
        env.step(action)
    env.reset(seed=0)

    start = time.perf_counter()
    for i in range(n_steps):
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            env.reset()
    elapsed = time.perf_counter() - start

    return n_steps / elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_steps", type=int, default=1000, help="Steps per benchmark run")
    parser.add_argument("--n_runs", type=int, default=3, help="Runs to average over")
    parser.add_argument("--weather_csv", default="/user/gent/453/vsc45342/thesis/weather_data/belgium_weather_multiyear.csv")
    parser.add_argument("--floorplans", nargs="+",
                        default=["single_room", "office_4room", "corporate_floor"],
                        choices=["single_room", "office_4room", "corporate_floor", "headquarters_floor"])
    args = parser.parse_args()

    print(f"{'='*65}")
    print(f"  SIMULATOR BENCHMARK")
    print(f"  Steps per run: {args.n_steps}  |  Runs: {args.n_runs}")
    print(f"{'='*65}")
    print(f"\n  {'Floorplan':<22}  {'Original (it/s)':>16}  {'Fast (it/s)':>12}  {'Speedup':>9}")
    print(f"  {'─'*22}  {'─'*16}  {'─'*12}  {'─'*9}")

    results = []
    for floorplan in args.floorplans:

        # Benchmark original
        orig_speeds = []
        for run in range(args.n_runs):
            try:
                env = make_env(floorplan, use_fast=False, weather_csv=args.weather_csv)
                speed = benchmark(env, args.n_steps)
                orig_speeds.append(speed)
                env.close()
            except Exception as e:
                print(f"  ERROR (original, {floorplan}, run {run}): {e}")
                break

        # Benchmark fast
        fast_speeds = []
        for run in range(args.n_runs):
            try:
                env = make_env(floorplan, use_fast=True, weather_csv=args.weather_csv)
                speed = benchmark(env, args.n_steps)
                fast_speeds.append(speed)
                env.close()
            except Exception as e:
                print(f"  ERROR (fast, {floorplan}, run {run}): {e}")
                break

        if orig_speeds and fast_speeds:
            orig_mean = np.mean(orig_speeds)
            fast_mean = np.mean(fast_speeds)
            speedup = fast_mean / orig_mean
            print(f"  {floorplan:<22}  {orig_mean:>13.1f}    {fast_mean:>10.1f}    {speedup:>7.1f}x")
            results.append((floorplan, orig_mean, fast_mean, speedup))

    if results:
        print(f"\n{'─'*65}")
        avg_speedup = np.mean([r[3] for r in results])
        print(f"  Average speedup: {avg_speedup:.1f}x")
        print(f"\n  Thesis note: FastCPUSimulator uses a Numba JIT-compiled")
        print(f"  Gauss-Seidel solver in identical row-major order to the")
        print(f"  original. Speedup comes from eliminating Python loop overhead.")

    print(f"{'='*65}")


if __name__ == "__main__":
    main()
