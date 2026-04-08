"""
Parse the Belpex hourly spot price Excel file and save a filtered parquet
covering only the training/val/test periods used in the thesis.

Output: thesis/weather_data/belpex_prices.parquet
  - Index: UTC datetime (hourly)
  - Column 'eur_per_mwh': float, spot price in EUR/MWh
  - Column 'usd_per_ws': float, converted to USD/Ws (used directly in gym_wrapper)

EUR/MWh → USD/Ws:  price_usd_per_ws = price_eur_per_mwh * EUR_TO_USD / 1e6 / 3600
"""

import pandas as pd
import numpy as np
import os

EXCEL_PATH = "/user/gent/453/vsc45342/thesis/weather_data/hourly-spot-belpex--c--elexys.xlsx"
OUT_PATH   = "/user/gent/453/vsc45342/thesis/weather_data/belpex_prices.parquet"

# Training periods to keep (inclusive)
PERIODS = [
    ("2019-10-01", "2022-03-31"),  # train
    ("2022-10-01", "2023-03-24"),  # val
    ("2023-10-01", "2024-03-24"),  # test
]

# 1 EUR ≈ 1.08 USD (2023 average)
EUR_TO_USD = 1.08


def in_any_period(ts: pd.Timestamp) -> bool:
    for start, end in PERIODS:
        if pd.Timestamp(start) <= ts <= pd.Timestamp(end):
            return True
    return False


def main():
    print("Reading Excel...")
    df = pd.read_excel(EXCEL_PATH, header=None, skiprows=2)
    df.columns = ["date", "time", "eur_per_mwh"]

    # Drop header row (row 0 after skiprows=2 is "Datum Time Euro")
    df = df[df["date"] != "Datum"].reset_index(drop=True)
    df = df.dropna(subset=["date", "time", "eur_per_mwh"])

    # Parse date: "30/12/2024" format
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")

    # Parse hour: "22u" → 22
    df["hour"] = df["time"].str.replace("u", "", regex=False).astype(int)

    # Build timestamp (Belgian local time — no DST correction, treated as naive)
    df["timestamp"] = df["date"] + pd.to_timedelta(df["hour"], unit="h")
    df = df.set_index("timestamp").sort_index()

    df["eur_per_mwh"] = pd.to_numeric(df["eur_per_mwh"], errors="coerce")
    df = df.dropna(subset=["eur_per_mwh"])

    print(f"Total rows: {len(df)}")
    print(f"Date range: {df.index.min()} → {df.index.max()}")

    # Filter to training periods only
    mask = df.index.map(in_any_period)
    df = df[mask]
    print(f"Rows after period filter: {len(df)}")

    # Convert EUR/MWh → USD/Ws
    df["usd_per_ws"] = df["eur_per_mwh"] * EUR_TO_USD / 1e6 / 3600.0

    out = df[["eur_per_mwh", "usd_per_ws"]]

    # Deduplicate (DST clock-change hours may appear twice — keep mean)
    out = out.groupby(out.index).mean()

    # Reindex to full hourly range and ffill gaps
    full_index = pd.date_range(out.index.min(), out.index.max(), freq="h")
    out = out.reindex(full_index).ffill().bfill()

    # Filter to periods only
    period_mask = out.index.map(in_any_period)
    out = out[period_mask]

    for start, end in PERIODS:
        period_df = out[start:end]
        expected = pd.date_range(start, end, freq="h")
        missing = expected.difference(period_df.index)
        if len(missing) > 0:
            print(f"WARNING: still {len(missing)} missing hours in {start}–{end}")
        else:
            print(f"OK: {start}–{end} — {len(period_df)} hours, no gaps")

    out.to_parquet(OUT_PATH)
    print(f"Saved → {OUT_PATH}")
    print(out.describe())


if __name__ == "__main__":
    main()