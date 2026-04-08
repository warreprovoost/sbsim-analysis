from typing import Any, Dict, List, Optional, Tuple

from gymnasium import spaces
import gymnasium as gym
import numpy as np
import pandas as pd
import re

from smart_control.simulator.randomized_arrival_departure_occupancy import RandomizedArrivalDepartureOccupancy
from smart_control.simulator.step_function_occupancy import StepFunctionOccupancy
from smart_control_analysis.energy_prices import get_electricity_price_usd_per_ws, get_gas_price_usd_per_1000ft3


class _AlwaysZeroOccupancy:
    """
    Passed to sim.reward_info() purely to get energy data from r_obj.
    Comfort penalty is computed separately in _reward_terms_from_reward_obj
    via _zone_occupancies_now() which correctly handles tz-naive local timestamps.
    """
    def average_zone_occupancy(self, zone_id: str, start_time: Any, end_time: Any) -> float:
        return 0.0

class _ConstantOccupancy:
    """Constant occupancy independent of time/window."""
    def __init__(self, value: float = 1.0):
        self.value = float(value)

    def average_zone_occupancy(self, zone_id: str, start_time: Any, end_time: Any) -> float:
        return self.value

class BuildingGymEnv(gym.Env):
    """
    Action space depends on action_design:

    "reheat_per_zone" (default, cold climate):
      [0]:      supply_air_temp_setpoint  [-1,1] -> [12°C, 32°C]
      [1]:      boiler_setpoint           [-1,1] -> [20°C, 65°C]
      [2]:      shared_damper             [-1,1] -> [0.01, 1.0]
      [3..3+n]: per-zone reheat valve     [-1,1] -> [0.0, 1.0]
      Total: 3 + n_zones

    "damper_per_zone" (warm climate):
      [0]:      supply_air_temp_setpoint  [-1,1] -> [12°C, 32°C]
      [1]:      boiler_setpoint           [-1,1] -> [20°C, 65°C]
      [2]:      shared_reheat             [-1,1] -> [0.0, 1.0]
      [3..3+n]: per-zone damper           [-1,1] -> [0.01, 1.0]
      Total: 3 + n_zones

    "full_per_zone" (both, largest space):
      [0]:      supply_air_temp_setpoint  [-1,1] -> [12°C, 32°C]
      [1]:      boiler_setpoint           [-1,1] -> [20°C, 65°C]
      [2..2+n]: per-zone reheat valve     [-1,1] -> [0.0, 1.0]
      [2+n..2+2n]: per-zone damper        [-1,1] -> [0.01, 1.0]
      Total: 2 + 2*n_zones
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        sim: Any,
        time_zone: str = "America/Los_Angeles",
        seed: Optional[int] = None,
        max_steps: Optional[int] = None,
        comfort_band_k: Tuple[float, float] = (294.15, 295.15),  # 21C to 22C
        working_hours: Tuple[float, float] = (8.0, 18.0),
        night_setback_k: float = 2.0,  # K to lower the band outside working hours
        occupancy_model: str = "randomized",   # "randomized" or "step"
        occupancy_per_zone: float = 10.0,
        occupancy_kwargs: Optional[Dict[str, Any]] = None,
        energy_norm: float = 500.0,  # W — expected peak *total* energy for this building
        use_cost_reward: bool = False,  # if True, use monetary cost (USD/step) instead of W
        energy_weight: float = 2.0,  # comfort/energy trade-off: higher = more energy penalty
        action_design: str = "reheat_per_zone",  # "reheat_per_zone" | "damper_per_zone" | "full_per_zone"
    ):
        super().__init__()
        self.sim = sim
        self.time_zone = time_zone
        # Default to steps for 24h
        self.max_steps = max_steps if max_steps is not None else (24 * 60 * 60) // self.sim.time_step_sec
        self.step_count=  0

        self.comfort_band_k = comfort_band_k
        self.working_hours = working_hours
        self.night_setback_k = float(night_setback_k)

        self.occupancy_model = occupancy_model
        self.occupancy_per_zone = float(occupancy_per_zone)
        self.occupancy_kwargs = occupancy_kwargs or {}

        self._last_energy_components = {
            "blower_rate": float("nan"),
            "boiler_gas_rate": float("nan"),
            "pump_rate": float("nan"),
        }
        # Single running-max norm for total energy (W).
        # Seeded from energy_norm so boiler/blower/AH absolute costs are preserved.
        # EMA slowly raises the norm if the sim regularly exceeds the initial estimate.
        self._energy_norm: float = float(energy_norm)  # fixed — set from physics in building_factory
        self.use_cost_reward = use_cost_reward
        self.energy_weight = float(energy_weight)
        if action_design not in ("reheat_per_zone", "damper_per_zone", "full_per_zone"):
            raise ValueError(f"Unknown action_design: {action_design!r}")
        self.action_design = action_design

        # Energy prices: real Belgian ZTP gas and Belpex electricity, looked up per timestamp.
        # Page 6 of European Commission, Joint Research Centre. (2017).
        # Best Available Techniques (BAT) Reference Document for Large Combustion Plants. Publications Office of the EU.
        self._BOILER_EFFICIENCY: float = 0.88

        # Cost norm: peak cost per step at 60s timestep.
        # Peak boiler (113W gas, Jan) + blower (30W elec peak) for 60s ≈ $0.000161
        # With energy_weight=1.0: peak penalty = 1.0 * 0.000161/0.000161 = 1.0 (= a 1°C violation)
        self._cost_norm: float = 0.000161

        zone_temps_dict = self.sim.building.get_zone_average_temps()
        self.zone_ids = sorted(zone_temps_dict.keys())  # e.g., ["room_1", "room_2"]
        self.n_zones = len(self.zone_ids)

        self.occupancies = self._build_occupancies(seed=seed)

        self._last_reheat_cmds = np.zeros(self.n_zones, dtype=np.float32)
        self._last_damper_cmds = np.zeros(self.n_zones, dtype=np.float32)
        self._last_ambient_temp_c: float = 0.0  # for ambient trend feature
        self._last_action: Optional[np.ndarray] = None  # for action smoothness penalty

        # action space size depends on design
        if self.action_design == "full_per_zone":
            n_actions = 2 + 2 * self.n_zones
        else:
            n_actions = 3 + self.n_zones  # one shared + n per-zone
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(n_actions,),
            dtype=np.float32,
        )
        self._forecast_hours = [1, 3, 6]  # hours ahead for weather forecast features
        # obs: temps(n) + errors(n) + reheat_cmds(n) + damper_cmds(n) + time(5)
        #      + [supply_air_sp, boiler_sp, ambient, ambient_trend](4) + forecast(3)
        self.observation_space = spaces.Box(
            low=-1e3, high=1e4,
            shape=(self.n_zones * 4 + 14,),
            dtype=np.float32,
        )
        # convenience references
        self.air_handler = self.sim.hvac.air_handler
        self.boiler = self.sim.hvac.boiler
        self.vavs = self.sim.hvac.vavs  # dict: zone_id -> VAV object

    def _comfort_band_now_k(self, timestamp: Optional[pd.Timestamp] = None) -> Tuple[Tuple[float, float], bool]:
        """Return ((low_k, high_k), is_working_hours).

        At night the floor is lowered by night_setback_k; the ceiling is unchanged.
        """
        ts = timestamp if timestamp is not None else self.sim.current_timestamp
        local_ts = ts.tz_convert(self.time_zone) if ts.tzinfo is not None else ts
        hour = local_ts.hour + local_ts.minute / 60.0
        start_h, end_h = self.working_hours
        in_working_hours = start_h <= hour < end_h
        if in_working_hours or self.night_setback_k == 0.0:
            return self.comfort_band_k, True
        low, high = self.comfort_band_k
        return (low - self.night_setback_k, high), False

    def _occupancy_window_naive(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """Return [start, end] as tz-naive LOCAL timestamps for occupancy models."""
        start = self.sim.current_timestamp
        end = start + pd.to_timedelta(self.sim.time_step_sec, unit="s")

        if isinstance(start, pd.Timestamp) and start.tzinfo is not None:
            # tz_convert first to get correct local time, THEN strip tz
            start = start.tz_convert(self.time_zone).tz_localize(None)
            end = end.tz_convert(self.time_zone).tz_localize(None)

        return start, end

    def _time_features(self, start: pd.Timestamp) -> np.ndarray:
        hour_float = start.hour + start.minute / 60.0 + start.second / 3600.0
        hour_angle = 2.0 * np.pi * (hour_float / 24.0)

        dow = start.dayofweek
        dow_angle = 2.0 * np.pi * (dow / 7.0)

        is_weekend = 1.0 if dow >= 5 else 0.0

        return np.array(
            [
                np.sin(hour_angle),
                np.cos(hour_angle),
                np.sin(dow_angle),
                np.cos(dow_angle),
                is_weekend,
            ],
            dtype=np.float32,
        )


    def _get_obs(self) -> np.ndarray:
        zone_temps_dict = self.sim.building.get_zone_average_temps()
        self._cached_zone_temps_dict = zone_temps_dict  # reused in _reward_terms_from_reward_obj
        temps_c = np.array([zone_temps_dict[zid] - 273.15 for zid in self.zone_ids], dtype=np.float32)

        comfort_mid_c = (self.comfort_band_k[0] + self.comfort_band_k[1]) / 2.0 - 273.15
        temp_errors = (temps_c - comfort_mid_c).astype(np.float32)  # +warm, -cold

        local_ts = self.sim.current_timestamp.tz_convert(self.time_zone)
        time_feats = self._time_features(local_ts)

        # Supply air setpoint: action range maps to [12, 32]°C → normalise to [-1, 1]
        supply_air_sp_raw = self.air_handler.heating_air_temp_setpoint - 273.15
        supply_air_sp = np.float32((supply_air_sp_raw - 22.0) / 10.0)   # centre=22, half-range=10

        # Boiler setpoint: action range maps to [36, 65]°C → normalise to [-1, 1]
        boiler_sp_raw = self.boiler.supply_water_setpoint - 273.15
        boiler_sp = np.float32((boiler_sp_raw - 50.5) / 14.5)           # centre=50.5, half-range=14.5

        ambient_temp_c = float(
            self.air_handler.get_observation(
                "outside_air_temperature_sensor",
                self.sim.current_timestamp,
            ) - 273.15
        )
        # Ambient: Oslo winter range roughly [-20, +20]°C → normalise to [-1, 1]
        ambient_norm = np.float32(ambient_temp_c / 20.0)
        ambient_trend = np.float32(ambient_temp_c - self._last_ambient_temp_c)
        # Trend: typically within ±2°C/step → normalise to [-1, 1]
        ambient_trend_norm = np.float32(np.clip(ambient_trend / 2.0, -1.0, 1.0))
        self._last_ambient_temp_c = ambient_temp_c

        # temp_errors: comfort band is 1°C wide, violations up to ~8°C → clip/norm to [-1, 1]
        temp_errors_norm = np.clip(temp_errors / 8.0, -1.0, 1.0).astype(np.float32)

        # Weather forecast: same normalisation as ambient
        wc = getattr(self.sim, "weather_controller", None) or getattr(self.sim, "_weather_controller", None)
        if hasattr(wc, "get_forecast_temps_c"):
            forecast_raw = wc.get_forecast_temps_c(self.sim.current_timestamp, self._forecast_hours)
        else:
            forecast_raw = np.full(len(self._forecast_hours), ambient_temp_c, dtype=np.float32)
        forecast_norm = np.clip(forecast_raw / 20.0, -1.0, 1.0).astype(np.float32)

        # temps_c: absolute zone temp, centre on comfort midpoint, same ±8°C normalisation
        temps_norm = np.clip((temps_c - (comfort_mid_c)) / 8.0, -1.0, 1.0).astype(np.float32)

        # Energy prices — normalised to [-1, 1]
        # Belpex electricity: centre=150 EUR/MWh, half-range=150 (covers 0–300 EUR/MWh)
        _elec_eur_per_mwh = get_electricity_price_usd_per_ws(local_ts) * 3.6e9 / 1.08
        elec_price_norm = np.float32(np.clip(_elec_eur_per_mwh / 150.0 - 1.0, -1.0, 1.0))
        # ZTP gas: centre=50 EUR/MWh, half-range=50 (covers 0–100 EUR/MWh)
        _gas_eur_per_mwh = get_gas_price_usd_per_1000ft3(local_ts) / 1.08 * 293.07107 / 1000.0
        gas_price_norm = np.float32(np.clip(_gas_eur_per_mwh / 50.0 - 1.0, -1.0, 1.0))

        return np.concatenate([
            temps_norm,                                    # n  zone temp normalised to [-1,1]
            temp_errors_norm,                              # n  comfort error normalised to [-1,1]
            self._last_reheat_cmds.astype(np.float32),     # n  last per-zone reheat cmds [-1,1]
            self._last_damper_cmds.astype(np.float32),     # n  last per-zone damper cmds [-1,1]
            time_feats,                                    # 5  sin/cos ∈ [-1,1], weekend ∈ {0,1}
            np.array(
                [
                    supply_air_sp,                         # [-1, 1] normalised setpoint
                    boiler_sp,                             # [-1, 1] normalised setpoint
                    ambient_norm,                          # [-1, 1] outdoor temp
                    ambient_trend_norm,                    # [-1, 1] temp change since last step
                ],
                dtype=np.float32,
            ),                                             # 4
            forecast_norm,                                 # 3  forecast temps normalised to [-1,1]
            np.array([elec_price_norm, gas_price_norm], dtype=np.float32),  # 2  energy prices
        ])

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            np.random.seed(seed)
        self.sim.reset()

        self.occupancies = self._build_occupancies(seed=seed)
        self.step_count = 0
        self._last_reheat_cmds = np.zeros(self.n_zones, dtype=np.float32)
        self._last_damper_cmds = np.zeros(self.n_zones, dtype=np.float32)
        self._last_ambient_temp_c = 0.0
        self._last_action = None
        # Keep norms across resets so they accumulate over training episodes.
        # Set to nan only on the very first reset (already done in __init__).

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        prev_action = self._last_action
        action = np.asarray(action, dtype=np.float32).ravel()
        self._last_action = action.copy()

        supply_air_action = action[0]
        boiler_action     = action[1]

        if self.action_design == "reheat_per_zone":
            # [supply, boiler, shared_damper, reheat_0..n]
            shared_damper_action   = action[2]
            vav_reheat_actions     = action[3:3 + self.n_zones]
            vav_damper_actions     = np.full(self.n_zones, shared_damper_action, dtype=np.float32)
        elif self.action_design == "damper_per_zone":
            # [supply, boiler, shared_reheat, damper_0..n]
            shared_reheat_action   = action[2]
            vav_damper_actions     = action[3:3 + self.n_zones]
            vav_reheat_actions     = np.full(self.n_zones, shared_reheat_action, dtype=np.float32)
        else:  # full_per_zone
            # [supply, boiler, reheat_0..n, damper_0..n]
            vav_reheat_actions     = action[2:2 + self.n_zones]
            vav_damper_actions     = action[2 + self.n_zones:2 + 2 * self.n_zones]

        # supply air: [-1,1] -> [285.15, 305.15] K [12°C, 32°C]
        supply_air_sp = 295.15 + supply_air_action * 10.0
        ah_heat_sp = supply_air_sp - 0.5
        ah_cool_sp = supply_air_sp + 0.5

        # boiler: [-1,1] -> [36°C, 50.5°C, 65°C]
        boiler_sp = 323.65 + boiler_action * 14.5

        # clamp boiler to at least outside air temp (the sim requires it)
        ambient_temp_k = float(
            self.air_handler.get_observation(
                "outside_air_temperature_sensor",
                self.sim.current_timestamp,
            )
        )
        boiler_sp = max(boiler_sp, ambient_temp_k + 1.0, ah_heat_sp + 1.0)  # at least 1°C above outside and air_supply

        action_timestamp = self.sim.current_timestamp

        self.air_handler.set_action(
            "supply_air_heating_temperature_setpoint",
            float(ah_heat_sp),
            action_timestamp,
        )
        self.air_handler.set_action(
            "supply_air_cooling_temperature_setpoint",
            float(ah_cool_sp),
            action_timestamp,
        )
        self.boiler.set_action("supply_water_setpoint", float(boiler_sp), action_timestamp)

        min_damper = 0.00001
        reheat_cmds = np.clip((vav_reheat_actions + 1.0) * 0.5, 0.0, 1.0)
        damper_cmds = np.clip((vav_damper_actions + 1.0) * 0.5, min_damper, 1.0)
        self._last_reheat_cmds = reheat_cmds.astype(np.float32)
        self._last_damper_cmds = damper_cmds.astype(np.float32)

        for i, zone_id in enumerate(self.zone_ids):
            vav = self.vavs[zone_id]
            vav.set_action(
                "supply_air_damper_percentage_command",
                float(damper_cmds[i]),
                action_timestamp,
            )
            vav.reheat_valve_setting = float(reheat_cmds[i])

        self.sim.step_sim()
        obs = self._get_obs()

        # Use _AlwaysZeroOccupancy so sim.reward_info() never calls StepFunctionOccupancy
        # with tz-aware timestamps (which would crash or give wrong results).
        # Comfort penalty is computed from _zone_occupancies_now() in _reward_terms_from_reward_obj.
        r_obj = self.sim.reward_info(_AlwaysZeroOccupancy())
        comfort_penalty, energy_rate = self._reward_terms_from_reward_obj(r_obj)

        # Compute monetary cost per step using pre-extracted scalar prices (no pint overhead).
        # dt is always time_step_sec — no need to diff timestamps.
        dt = float(self.sim.time_step_sec)
        local_start = action_timestamp.tz_convert(self.time_zone)
        gas_rate = max(0.0, self._last_energy_components["boiler_gas_rate"])
        _gas_usd_per_j = get_gas_price_usd_per_1000ft3(local_start) / 293.07107 / 3.6e6
        gas_cost = float(_gas_usd_per_j * gas_rate / self._BOILER_EFFICIENCY * dt)
        elec_rate = abs(self._last_energy_components["blower_rate"]) + abs(self._last_energy_components["ah_conditioning_rate"])
        elec_price = get_electricity_price_usd_per_ws(local_start)
        elec_cost = float(elec_price * elec_rate * dt)
        total_cost = gas_cost + elec_cost

        total_reward = self._combine_reward_terms(comfort_penalty, energy_rate, total_cost, prev_action, action)

        self.step_count += 1
        terminated = False
        truncated = self.step_count >= self.max_steps

        (comfort_low_k, comfort_high_k), _ = self._comfort_band_now_k(action_timestamp)
        info: Dict = {
            "comfort_low_c":        comfort_low_k - 273.15,
            "comfort_high_c":       comfort_high_k - 273.15,
            "comfort_penalty":      float(comfort_penalty),
            "energy_rate":          float(energy_rate),
            "energy_cost_usd":      float(total_cost),
            "gas_cost_usd":         float(gas_cost),
            "elec_cost_usd":        float(elec_cost),
            "elec_price_eur_per_mwh": float(elec_price * 3.6e9 / 1.08),  # USD/Ws → EUR/MWh
            "gas_price_eur_per_mwh":  float(get_gas_price_usd_per_1000ft3(local_start) / 1.08 * 293.07107 / 1000.0),  # USD/1000ft3 → EUR/MWh
            "blower_rate":          self._last_energy_components["blower_rate"],
            "ah_conditioning_rate": self._last_energy_components["ah_conditioning_rate"],
            "boiler_gas_rate":      self._last_energy_components["boiler_gas_rate"],
            "pump_rate_raw":        self._last_energy_components["pump_rate_raw"],
            "reheat_coil_rate":     self._last_energy_components["reheat_coil_rate"],
            "reward_total":         float(total_reward),
            "supply_air_sp_C":      supply_air_sp - 273.15,
            "boiler_sp_C":          boiler_sp - 273.15,
        }

        return obs, float(total_reward), terminated, truncated, info

    def _reward_terms_from_reward_obj(self, r: Any) -> Tuple[float, float]:
        (comfort_low_k, comfort_high_k), is_working = self._comfort_band_now_k()

        total_comfort_penalty = 0.0

        weighted_violation_sum = 0.0
        weight_sum = 0.0

        comfort_mid_k = 0.5 * (comfort_low_k + comfort_high_k)
        comfort_half_band = 0.5 * (comfort_high_k - comfort_low_k)

        weighted_violation_sum = 0.0
        weight_sum = 0.0

        for _zone_id, ziv in r.zone_reward_infos.items():
            t = float(ziv.zone_air_temperature)
            # Linear violation in degrees C (K difference = C difference)
            violation_deg = max(comfort_low_k - t, 0.0) + max(t - comfort_high_k, 0.0)
            # Quadratic: 1°C off → 1, 4°C off → 16, 8°C off → 64
            temp_violation = violation_deg ** 2
            # Center bonus only during working hours — at night we don't want to incentivise
            # cooling toward the (shifted-down) midpoint, just avoid falling below the lower floor.
            if is_working:
                center_bonus = 0.5 * max(0.0, 1.0 - abs(t - comfort_mid_k) / comfort_half_band)
            else:
                center_bonus = 0.0

            w = 1.0  # always penalise (occupancy forced on)
            weighted_violation_sum += w * (temp_violation - center_bonus)
            weight_sum += w

        total_comfort_penalty = (
            weighted_violation_sum / (weight_sum + 1e-6)
            if weight_sum > 0.0
            else 0.0
        )

        blower_rate = sum(
            float(ainfo.blower_electrical_energy_rate)
            for ainfo in r.air_handler_reward_infos.values()
        )
        ah_conditioning_rate = sum(
            abs(float(ainfo.air_conditioning_electrical_energy_rate))
            for ainfo in r.air_handler_reward_infos.values()
        )
        boiler_gas_rate = sum(
            float(binfo.natural_gas_heating_energy_rate)
            for binfo in r.boiler_reward_infos.values()
        )
        pump_rate = sum(
            float(binfo.pump_electrical_energy_rate)
            for binfo in r.boiler_reward_infos.values()
        )

        # get supply air temp from AH after sim step — reuse cached temps from _get_obs
        _zt = getattr(self, "_cached_zone_temps_dict", None) or self.sim.building.get_zone_average_temps()
        recirculation_temp = float(np.mean(list(_zt.values())))

        ambient_temp = float(
            self.air_handler.get_observation(
                "outside_air_temperature_sensor",
                self.sim.current_timestamp,
            )
        )

        supply_air_temp = float(
            self.air_handler.get_supply_air_temp(recirculation_temp, ambient_temp)
        )

        reheat_coil_rate = sum(
            vav.compute_reheat_energy_rate(
                supply_air_temp,
                float(self.boiler.supply_water_setpoint),
            )
            for vav in self.vavs.values()
        )

        self._last_energy_components = {
            "blower_rate":        float(blower_rate),
            "ah_conditioning_rate": float(ah_conditioning_rate),
            "boiler_gas_rate":    float(boiler_gas_rate),
            "pump_rate_raw":      float(pump_rate),
            "reheat_coil_rate":   float(reheat_coil_rate),
        }

        total_energy_rate = blower_rate + ah_conditioning_rate + boiler_gas_rate + pump_rate
        return float(total_comfort_penalty), float(total_energy_rate)




    def _combine_reward_terms(self, comfort_penalty: float, energy_rate: float,
                              total_cost: float = 0.0,
                              prev_action: Optional[np.ndarray] = None,
                              action: Optional[np.ndarray] = None) -> float:
        # comfort_penalty: mean squared °C violation → 1°C=1, 4°C=16, 8°C=64
        # energy_rate (watts mode): total W normalized by physics-based peak estimate
        # total_cost (cost mode): USD/step normalized by cost_norm
        # smoothness_penalty: penalizes large action changes to discourage bang-bang control
        # energy_weight controls the comfort/energy trade-off:
        #   2.0 → peak energy penalty = 2× a 1°C violation  (energy-focused)
        #   1.0 → peak energy penalty = 1× a 1°C violation  (balanced)
        #   0.5 → peak energy penalty = 0.5× a 1°C violation (comfort-focused)
        energy_weight = self.energy_weight
        if self.use_cost_reward:
            energy_penalty = energy_weight * total_cost / max(self._cost_norm, 1e-9) if self.occupancy_model != "step" else 0.0
        else:
            energy_penalty = energy_weight * energy_rate / max(self._energy_norm, 1.0) if self.occupancy_model != "step" else 0.0

        # Small penalty on action rate-of-change (max delta per dim = 2.0)
        # Weight 0.05 → max smoothness penalty ≈ 0.1, much smaller than comfort
        if prev_action is not None and action is not None:
            delta = float(np.mean(np.abs(action - prev_action)))
            smoothness_penalty = 0.05 * delta
        else:
            smoothness_penalty = 0.0

        return float(-(comfort_penalty + energy_penalty + smoothness_penalty))

    def _reduce_reward_obj_to_scalar(self, r: Any) -> float:
        comfort_penalty, energy_rate = self._reward_terms_from_reward_obj(r)
        return self._combine_reward_terms(comfort_penalty, energy_rate, total_cost=0.0)

    def _build_occupancies(self, seed: Optional[int]) -> List[Any]:
        occs: List[Any] = []
        base_seed = None if seed is None else int(seed)

        for i, _zone_id in enumerate(self.zone_ids):
            if self.occupancy_model == "step":
                # Step-function occupancy (deterministic schedule)
                work_start_hour, work_end_hour = self.working_hours
                default_kwargs = {
                    "work_start_time": pd.Timedelta(hours=work_start_hour),
                    "work_end_time": pd.Timedelta(hours=work_end_hour),
                    "work_occupancy": self.occupancy_per_zone,
                    "nonwork_occupancy": 0.0,
                }
                default_kwargs.update(self.occupancy_kwargs)
                occs.append(StepFunctionOccupancy(**default_kwargs))
            elif self.occupancy_model in {"constant", "always_one"}:
                occs.append(_ConstantOccupancy(value=self.occupancy_per_zone))
            else:
                # Randomized occupancy (stochastic)
                default_kwargs = {
                    "zone_assignment": int(self.occupancy_per_zone),
                    "earliest_expected_arrival_hour": 7,
                    "latest_expected_arrival_hour": 9,
                    "earliest_expected_departure_hour": 16,
                    "latest_expected_departure_hour": 18,
                    "time_step_sec": self.sim.time_step_sec,
                    "seed": (None if base_seed is None else base_seed + i),
                    "time_zone": self.time_zone,
                }

                work_start_hour, work_end_hour = self.working_hours
                default_kwargs = { # OVERWRITE
                    "work_start_time": pd.Timedelta(hours=work_start_hour),
                    "work_end_time": pd.Timedelta(hours=work_end_hour),
                    "work_occupancy": self.occupancy_per_zone,
                    "nonwork_occupancy": 0.0,
                }
                default_kwargs.update(self.occupancy_kwargs)
                occs.append(StepFunctionOccupancy(**default_kwargs)) # DON'T FORGET, I disabled RandomizedArrivalDepartureOccupancy

        return occs

    def _zone_occupancies_now(self) -> Dict[str, float]:
        start, end = self._occupancy_window_naive()

        occ_dict = {}
        for occ, zone_id in zip(self.occupancies, self.zone_ids):
            occ_val = float(occ.average_zone_occupancy(zone_id, start, end))
            occ_dict[zone_id] = occ_val
        return occ_dict

    def _zone_suffix(self, zid: str) -> Optional[str]:
        """Extract numeric suffix from ids like room_1 / zone_id_1."""
        m = re.search(r"(\d+)$", str(zid))
        return m.group(1) if m else None

    def _occupancy_for_reward_zone(
        self,
        reward_zone_id: str,
        occ_by_zone: Dict[str, float],
    ) -> float:
        """Map reward zone id (e.g., zone_id_1) to occupancy zone id (e.g., room_1)."""
        # direct hit
        if reward_zone_id in occ_by_zone:
            return float(occ_by_zone[reward_zone_id])

        # suffix-based fallback (robust to room_1 vs zone_id_1 naming)
        target_suffix = self._zone_suffix(reward_zone_id)
        if target_suffix is None:
            return 0.0

        for occ_zone_id, occ in occ_by_zone.items():
            if self._zone_suffix(occ_zone_id) == target_suffix:
                return float(occ)

        return 0.0

    def render(self, mode="human"):
        self.sim.get_video()
        return

    def close(self):
        return None
