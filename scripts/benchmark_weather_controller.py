#!/usr/bin/env python3
"""Benchmark FastReplayWeatherController vs ReplayWeatherController.

Measures get_current_temp calls/second for both controllers.
Reports speedup factor for thesis.

Usage:
    python scripts/benchmark_weather_controller.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../sbsim"))

from smart_control.simulator.weather_controller import ReplayWeatherController
from smart_control_analysis.custom_sbsim.fast_weather_controller import FastReplayWeatherController

WEATHER_CSV = "/user/gent/453/vsc45342/thesis/weather_data/belgium_weather_multiyear.csv"
N_CALLS = 10000
N_RUNS = 5


def make_timestamps(n: int) -> list:
    """Generate n random timestamps within the training period."""
    rng = np.random.default_rng(42)
    start = pd.Timestamp("2019-11-01", tz="UTC")
    end = pd.Timestamp("2022-02-28", tz="UTC")
    span_s = int((end - start).total_seconds())
    offsets = rng.integers(0, span_s, size=n)
    return [start + pd.Timedelta(seconds=int(o)) for o in offsets]


def benchmark(controller, timestamps) -> float:
    """Call get_current_temp for each timestamp, return calls/second."""
    # Warmup
    for ts in timestamps[:100]:
        controller.get_current_temp(ts)

    start = time.perf_counter()
    for ts in timestamps:
        controller.get_current_temp(ts)
    elapsed = time.perf_counter() - start
    return len(timestamps) / elapsed


def main():
    print(f"{'='*60}")
    print(f"  WEATHER CONTROLLER BENCHMARK")
    print(f"  Calls per run: {N_CALLS:,}  |  Runs: {N_RUNS}")
    print(f"{'='*60}")

    timestamps = make_timestamps(N_CALLS)

    orig_speeds = []
    fast_speeds = []

    for run in range(N_RUNS):

        #orig_speeds.append(benchmark(orig, timestamps))

        fast = FastReplayWeatherController(
            local_weather_path=WEATHER_CSV,
            convection_coefficient=60.0,
        )
        fast_speeds.append(benchmark(fast, timestamps))

        print(
              f"fast={fast_speeds[-1]:>10.0f} calls/s  "
              )

    fast_mean = np.mean(fast_speeds)

    print(f"\n{'─'*60}")
    print(f"  Fast     (mean): {fast_mean:>10.0f} calls/s")
    print(f"\n  FastReplayWeatherController precomputes time and temperature")
    print(f"  arrays once at construction, replacing per-call array rebuilds")
    print(f"  and min()/max() scans with a single np.interp lookup.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
