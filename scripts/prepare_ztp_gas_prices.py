"""
Parse EEX ZTP Gas Spot Excel and save a monthly gas price parquet.

Output: thesis/weather_data/ztp_gas_prices.parquet
  - Index: pd.Period('YYYY-MM', freq='M')  stored as year, month int columns
  - Column 'usd_per_1000ft3': converted price used by NaturalGasEnergyCost

Conversion: EUR/MWh → USD/1000 ft³
  sbsim constant: KWH_PER_KFT3_GAS = 293.07107  →  1000 ft³ = 293.07 kWh
  So 1 MWh = 1000/293.07107 × 1000 ft³ ≈ 3.412 × 1000 ft³
  1 EUR ≈ 1.08 USD
  usd_per_1000ft3 = eur_per_mwh * 1.08 / (1000 / 293.07107)
"""

import pandas as pd
import numpy as np

EXCEL_PATH = "/user/gent/453/vsc45342/thesis/weather_data/eex-ztp-gas-spot--c--elexys.xlsx"
OUT_PATH   = "/user/gent/453/vsc45342/thesis/weather_data/ztp_gas_prices.parquet"

EUR_TO_USD = 1.08
KWH_PER_KFT3_GAS = 293.07107  # sbsim constant: 1000 ft³ = 293.07 kWh
MWH_TO_1000FT3 = 1000.0 / KWH_PER_KFT3_GAS  # ≈ 3.412 (1 MWh = 3.412 × 1000 ft³)

# Years with no real data — map to a donor year's monthly pattern.
# Gas prices are monthly so we copy the donor's 12-month profile.
YEAR_DONOR_MAP = {
    2015: 2020,
    2016: 2020,
    2017: 2020,
    2018: 2020,
    2019: 2020,  # 2019 was already filled from 2020; make it explicit
}

MONTH_MAP = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}


def parse_eur(val) -> float:
    """Parse '€ 11,19' or similar to float."""
    if pd.isna(val):
        return float("nan")
    s = str(val).replace("€", "").replace(" ", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def main():
    df = pd.read_excel(EXCEL_PATH, header=None, skiprows=2)
    # Row 0 after skiprows=2 is the header: Month, 2020, 2021, ...
    df.columns = df.iloc[0]
    df = df.iloc[1:].reset_index(drop=True)
    df = df[df["Month"].isin(MONTH_MAP)].copy()

    years = [c for c in df.columns if str(c).isdigit()]

    records = []
    for _, row in df.iterrows():
        month = MONTH_MAP[row["Month"]]
        for year in years:
            eur = parse_eur(row[year])
            if not np.isnan(eur):
                usd = eur * EUR_TO_USD / MWH_TO_1000FT3
                records.append({"year": int(year), "month": month,
                                 "eur_per_mwh": eur, "usd_per_1000ft3": usd})

    out = pd.DataFrame(records).set_index(["year", "month"]).sort_index()
    print(out.to_string())

    # Fill missing years using donor year's monthly pattern.
    for target_year, donor_year in YEAR_DONOR_MAP.items():
        for month in range(1, 13):
            if (target_year, month) not in out.index and (donor_year, month) in out.index:
                row = out.loc[(donor_year, month)].copy()
                out.loc[(target_year, month), :] = row
    out = out.sort_index()
    print(f"\nYears covered after donor fill: {sorted(out.index.get_level_values('year').unique())}")

    out.to_parquet(OUT_PATH)
    print(f"\nSaved → {OUT_PATH}")
    print(f"Years covered: {sorted(out.index.get_level_values('year').unique())}")


if __name__ == "__main__":
    main()