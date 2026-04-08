# Belgian energy prices for the HVAC RL thesis.
#
# Electricity: real Belpex hourly spot prices, loaded from parquet.
#   Prepared by scripts/prepare_belpex_prices.py
#
# Gas: real EEX ZTP monthly spot prices, loaded from parquet.
#   Prepared by scripts/prepare_ztp_gas_prices.py
#   2019 missing from source → filled with 2020 values.

import os
import pandas as pd

_DATA_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../weather_data"))

# ---------------------------------------------------------------------------
# Electricity — Belpex hourly spot (EUR/MWh → USD/Ws)
# ---------------------------------------------------------------------------

def _load_belpex() -> pd.Series:
    df = pd.read_parquet(os.path.join(_DATA_DIR, "belpex_prices.parquet"))
    return df["usd_per_ws"]

BELPEX_USD_PER_WS: pd.Series = _load_belpex()
BELPEX_FALLBACK_USD_PER_WS: float = float(BELPEX_USD_PER_WS.median())


def get_electricity_price_usd_per_ws(timestamp: pd.Timestamp) -> float:
    """Belgian Belpex spot price in USD/Ws for the given timestamp.

    Truncates to the hour. Falls back to dataset median if not found.
    """
    key = timestamp.replace(tzinfo=None).floor("h")
    val = BELPEX_USD_PER_WS.get(key, None)
    if val is None or pd.isna(val):
        return BELPEX_FALLBACK_USD_PER_WS
    return float(val)


# ---------------------------------------------------------------------------
# Gas — EEX ZTP monthly spot (EUR/MWh → USD/1000 ft³)
# ---------------------------------------------------------------------------

def _load_ztp() -> pd.DataFrame:
    return pd.read_parquet(os.path.join(_DATA_DIR, "ztp_gas_prices.parquet"))

_ZTP_DF: pd.DataFrame = _load_ztp()
_ZTP_FALLBACK_USD_PER_1000FT3: float = float(_ZTP_DF["usd_per_1000ft3"].median())
# Pre-build a plain dict for O(1) lookup instead of pandas MultiIndex .loc
_ZTP_DICT: dict = {
    (int(y), int(m)): float(v)
    for (y, m), v in _ZTP_DF["usd_per_1000ft3"].items()
}


def get_gas_price_usd_per_1000ft3(timestamp: pd.Timestamp) -> float:
    """EEX ZTP spot gas price in USD/1000 ft³ for the given timestamp's year+month."""
    return _ZTP_DICT.get((timestamp.year, timestamp.month), _ZTP_FALLBACK_USD_PER_1000FT3)


def get_gas_prices_by_month_for_year(year: int) -> tuple:
    """Return a 12-tuple of USD/1000 ft³ values for Jan–Dec of the given year.

    Used to construct NaturalGasEnergyCost for a fixed training year.
    Falls back to median for missing months.
    """
    return tuple(
        get_gas_price_usd_per_1000ft3(pd.Timestamp(year=year, month=m, day=1))
        for m in range(1, 13)
    )