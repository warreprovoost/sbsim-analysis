from typing import Any, Dict, List, Optional, Tuple

from smart_control_analysis.custom_sbsim.mutable_schedule import MutableSetpointSchedule
from gymnasium import spaces
import gymnasium as gym
import numpy as np
import pandas as pd
import re

from smart_control.simulator.randomized_arrival_departure_occupancy import RandomizedArrivalDepartureOccupancy
from smart_control.simulator.step_function_occupancy import StepFunctionOccupancy


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
    Action space (reduced):
    - [0]:      supply_air_temp_setpoint  [-1,1] -> [12°C, 32°C]
    - [1]:      boiler_setpoint           [-1,1] -> [20°C, 65°C]
    - [2]:      shared damper             [-1,1] -> [0.01, 1.0]  (same for all zones)
    - [3..3+n]: per-zone reheat valve     [-1,1] -> [0.0, 1.0]
    Total: 3 + n_zones
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
        occupancy_model: str = "randomized",   # "randomized" or "step"
        occupancy_per_zone: float = 10.0,
        occupancy_kwargs: Optional[Dict[str, Any]] = None,
        energy_norm: float = 500.0,  # W — expected peak *total* energy for this building
    ):
        super().__init__()
        self.sim = sim
        self.time_zone = time_zone
        # Default to steps for 24h
        self.max_steps = max_steps if max_steps is not None else (24 * 60 * 60) // self.sim.time_step_sec
        self.step_count=  0

        self.comfort_band_k = comfort_band_k
        self.working_hours = working_hours

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

        zone_temps_dict = self.sim.building.get_zone_average_temps()
        self.zone_ids = sorted(zone_temps_dict.keys())  # e.g., ["room_1", "room_2"]
        self.n_zones = len(self.zone_ids)

        self.occupancies = self._build_occupancies(seed=seed)

        self._last_shared_damper_cmd = np.float32(0.5)
        self._last_reheat_cmds = np.zeros(self.n_zones, dtype=np.float32)
        self._last_ambient_temp_c: float = 0.0  # for ambient trend feature
        self._last_action: Optional[np.ndarray] = None  # for action smoothness penalty


        # action: [ah_supply, boiler, vav_1_damper, vav_2_damper, ..., vav_1_reheat, vav_2_reheat, ...]
        # 2 shared + 2 per zone
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(3 + self.n_zones,),
            dtype=np.float32,
        )
        self._forecast_hours = [1, 3, 6]  # hours ahead for weather forecast features
        self.observation_space = spaces.Box(
            low=-1e3, high=1e4,
            # temps(n) + temp_errors(n) + reheat_cmds(n) + time_feats(5)
            # + [supply_air_sp, boiler_sp, ambient, ambient_trend, damper](5)
            # + forecast_temps(3): outdoor temp at +1h, +3h, +6h
            shape=(self.n_zones * 3 + 13,),
            dtype=np.float32,
        )
        # convenience references
        self.air_handler = self.sim.hvac.air_handler
        self.boiler = self.sim.hvac.boiler
        self.vavs = self.sim.hvac.vavs  # dict: zone_id -> VAV object

    def _occupancy_window_naive(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """Return [start, end] as tz-naive LOCAL timestamps for occupancy models."""
        start = self.sim.current_timestamp
        end = start + pd.to_timedelta(self.sim.time_step_sec, unit="s")

        if isinstance(start, pd.Timestamp) and start.tzinfo is not None:
            # FIX: tz_convert first to get correct local time, THEN strip tz
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
        temps_c = np.array([zone_temps_dict[zid] - 273.15 for zid in self.zone_ids], dtype=np.float32)

        comfort_mid_c = (self.comfort_band_k[0] + self.comfort_band_k[1]) / 2.0 - 273.15
        temp_errors = (temps_c - comfort_mid_c).astype(np.float32)  # +warm, -cold

        start, _ = self._occupancy_window_naive()
        time_feats = self._time_features(start)

        supply_air_sp = np.float32(self.air_handler.heating_air_temp_setpoint - 273.15)
        boiler_sp = np.float32(self.boiler.supply_water_setpoint - 273.15)

        ambient_temp_c = float(
            self.air_handler.get_observation(
                "outside_air_temperature_sensor",
                self.sim.current_timestamp,
            ) - 273.15
        )
        ambient_trend = np.float32(ambient_temp_c - self._last_ambient_temp_c)
        self._last_ambient_temp_c = ambient_temp_c

        # Weather forecast: outdoor temp at +1h, +3h, +6h
        wc = self.sim.weather_controller
        if hasattr(wc, "get_forecast_temps_c"):
            forecast = wc.get_forecast_temps_c(self.sim.current_timestamp, self._forecast_hours)
        else:
            forecast = np.full(len(self._forecast_hours), ambient_temp_c, dtype=np.float32)

        return np.concatenate([
            temps_c,                                      # n  absolute zone temps (°C)
            temp_errors,                                  # n  zone temp - comfort midpoint
            self._last_reheat_cmds.astype(np.float32),    # n  last reheat valve positions
            time_feats,                                   # 5  sin/cos hour, sin/cos dow, weekend
            np.array(
                [
                    supply_air_sp,                        # last supply air setpoint
                    boiler_sp,                            # last boiler setpoint
                    np.float32(ambient_temp_c),           # current outdoor temp
                    ambient_trend,                        # °C change since last step (pre-heat signal)
                    self._last_shared_damper_cmd,         # last damper position
                ],
                dtype=np.float32,
            ),                                            # 5
            forecast,                                     # 3  outdoor temp at +1h, +3h, +6h
        ])

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            np.random.seed(seed)
        self.sim.reset()

        self.occupancies = self._build_occupancies(seed=seed)
        self.step_count = 0
        self._last_shared_damper_cmd = np.float32(0.5)
        self._last_reheat_cmds = np.zeros(self.n_zones, dtype=np.float32)
        self._last_ambient_temp_c = 0.0
        self._last_action = None
        # Keep norms across resets so they accumulate over training episodes.
        # Set to nan only on the very first reset (already done in __init__).

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        prev_action = self._last_action
        action = np.asarray(action, dtype=np.float32).ravel()
        assert action.size == 3 + self.n_zones
        self._last_action = action.copy()

        supply_air_action  = action[0]
        boiler_action      = action[1]
        shared_damper      = action[2]
        vav_reheat_actions = action[3:3 + self.n_zones]

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

        # shared damper — same for all zones
        min_damper = 0.00001
        damper_cmd = float(np.clip((shared_damper + 1.0) * 0.5, min_damper, 1.0))

        reheat_cmds = np.clip((vav_reheat_actions + 1.0) * 0.5, 0.0, 1.0)
        self._last_shared_damper_cmd = np.float32(damper_cmd)
        self._last_reheat_cmds = reheat_cmds.astype(np.float32)

        for i, zone_id in enumerate(self.zone_ids):
            vav = self.vavs[zone_id]
            vav.set_action(
                "supply_air_damper_percentage_command",
                damper_cmd,   # shared for all zones
                action_timestamp,
            )
            vav.reheat_valve_setting = float(reheat_cmds[i])  # per zone

        self.sim.step_sim()
        obs = self._get_obs()

        # Use _AlwaysZeroOccupancy so sim.reward_info() never calls StepFunctionOccupancy
        # with tz-aware timestamps (which would crash or give wrong results).
        # Comfort penalty is computed from _zone_occupancies_now() in _reward_terms_from_reward_obj.
        r_obj = self.sim.reward_info(_AlwaysZeroOccupancy())
        comfort_penalty, energy_rate = self._reward_terms_from_reward_obj(r_obj)
        total_reward = self._combine_reward_terms(comfort_penalty, energy_rate, prev_action, action)

        self.step_count += 1
        terminated = False
        truncated = self.step_count >= self.max_steps

        info: Dict = {
            "comfort_penalty":      float(comfort_penalty),
            "energy_rate":          float(energy_rate),
            "blower_rate":          self._last_energy_components["blower_rate"],
            "ah_conditioning_rate": self._last_energy_components["ah_conditioning_rate"],
            "boiler_gas_rate":      self._last_energy_components["boiler_gas_rate"],
            "pump_rate_raw":        self._last_energy_components["pump_rate_raw"],
            "reheat_coil_rate":     self._last_energy_components["reheat_coil_rate"],  # new
            "reward_total":         float(total_reward),
            "supply_air_sp_C":      supply_air_sp - 273.15,
            "boiler_sp_C":          boiler_sp - 273.15,
        }

        return obs, float(total_reward), terminated, truncated, info

    def _reward_terms_from_reward_obj(self, r: Any) -> Tuple[float, float]:
        comfort_low_k, comfort_high_k = self.comfort_band_k

        total_comfort_penalty = 0.0

        weighted_violation_sum = 0.0
        weight_sum = 0.0

        for _zone_id, ziv in r.zone_reward_infos.items():
            t = float(ziv.zone_air_temperature)
            # Linear violation in degrees C (K difference = C difference)
            violation_deg = max(comfort_low_k - t, 0.0) + max(t - comfort_high_k, 0.0)
            # Quadratic: 1°C off → 1, 4°C off → 16, 8°C off → 64
            temp_violation = violation_deg ** 2

            w = 1.0  # always penalise (occupancy forced on)
            weighted_violation_sum += w * temp_violation
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

        # get supply air temp from AH after sim step
        recirculation_temp = float(
            np.mean(list(self.sim.building.get_zone_average_temps().values()))
        )

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
                              prev_action: Optional[np.ndarray] = None,
                              action: Optional[np.ndarray] = None) -> float:
        # comfort_penalty: mean squared °C violation → 1°C=1, 4°C=16, 8°C=64
        # energy_rate: total W normalized by physics-based peak estimate
        # smoothness_penalty: penalizes large action changes to discourage bang-bang control
        energy_weight = 0.5
        if self.occupancy_model == "step":
            energy_weight = 0.0

        energy_penalty = energy_weight * energy_rate / max(self._energy_norm, 1.0)

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
        return self._combine_reward_terms(comfort_penalty, energy_rate)

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
