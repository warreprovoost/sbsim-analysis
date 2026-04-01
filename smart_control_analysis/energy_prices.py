# Oslo, Norway energy prices for monetary cost reward.

# Gas: European TTF-based, $/1000 ft³, monthly. Source: TTF spot 2023, 1 EUR ≈ 1.08 USD.
GAS_PRICE_BY_MONTH_SOURCE = (
    16.5,   # Jan
    15.5,   # Feb
    14.0,   # Mar
    10.5,   # Apr
     9.0,   # May
     8.0,   # Jun
     7.5,   # Jul
     7.5,   # Aug
     9.0,   # Sep
    11.5,   # Oct
    14.5,   # Nov
    17.0,   # Dec
)

# Electricity: Nordpool NO1 spot + Elvia commercial grid tariff, cents/kWh (USD), ~2023.
# Spot winter peak (07-19): ~80 øre/kWh; off-peak: ~60 øre/kWh.
# Grid tariff (Elvia commercial): ~40 øre/kWh fixed component included in spot estimates.
# 1 NOK ≈ 0.092 USD → peak ~120 øre ≈ 11.0 ¢/kWh; off-peak ~60 øre ≈ 5.5 ¢/kWh.
WEEKDAY_PRICE_BY_HOUR = (
    5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5,   # 00-06 off-peak
    11.0, 11.0, 11.0, 11.0, 11.0, 11.0,  # 07-12 peak
    11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0,  # 13-19 peak
    5.5, 5.5, 5.5, 5.5,                   # 20-23 off-peak
)
WEEKEND_PRICE_BY_HOUR = (5.5,) * 24
