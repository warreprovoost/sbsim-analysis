from typing import Any, Dict, List, Optional, Tuple

from smart_control_analysis.custom_sbsim.mutable_schedule import MutableSetpointSchedule
from gymnasium import spaces
import gymnasium as gym
import numpy as np
import pandas as pd

from smart_control.simulator.randomized_arrival_departure_occupancy import RandomizedArrivalDepartureOccupancy
from smart_control.simulator.simulator import Simulator
from smart_control.simulator.thermostat import Thermostat


class BuildingGymEnv(gym.Env):
    """
    Action space (reduced):
    - [0]: supply_air_temp_setpoint [-1,1] -> [285.15, 305.15] K (12°C - 32°C)
    - [1]: boiler water setpoint [-1,1] -> [318, 338] K (45°C - 65°C)
    - [2:2+n_zones]: per-zone VAV damper percentage [-1,1] -> [0.01, 1]
    - [2+n_zones:2+2*n_zones]: per-zone VAV reheat valve [-1,1] -> [0, 1]
    Total action dim: 2 + 2*n_zones
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        sim: Any,
        mutable_schedule: MutableSetpointSchedule = None,
        time_zone: str = "Europe/Brussels",
        seed: Optional[int] = None,
        max_steps: Optional[int] = None,
        comfort_band_k: Tuple[float, float] = (294.15, 295.15),  # 21C to 22C
        comfort_hours: Tuple[float, float] = (8.0, 18.0),
    ):
        super().__init__()
        self.sim = sim
        self.mutable_schedule = mutable_schedule
        self.time_zone = time_zone
        # Default to steps for 24h
        self.max_steps = max_steps if max_steps is not None else (24 * 60 * 60) // self.sim.time_step_sec

        self.comfort_band_k = comfort_band_k
        self.comfort_hours = comfort_hours


        zone_temps_dict = self.sim.building.get_zone_average_temps()
        self.zone_ids = sorted(zone_temps_dict.keys())  # e.g., ["room_1", "room_2"]
        self.n_zones = len(self.zone_ids)

        # create occupancies (one per zone)
        self.occupancies: List[RandomizedArrivalDepartureOccupancy] = [
            RandomizedArrivalDepartureOccupancy(
                zone_assignment=10, # this is the number of occupants
                earliest_expected_arrival_hour=7,
                latest_expected_arrival_hour=9,
                earliest_expected_departure_hour=16,
                latest_expected_departure_hour=18,
                time_step_sec=self.sim.time_step_sec,
                seed=(None if seed is None else seed + i),
                time_zone=self.time_zone,
            )
            for i in range(self.n_zones)
        ]

        # action: [ah_heat, ah_cool, boiler, vav_1_damper, vav_2_damper, ..., vav_1_reheat, vav_2_reheat, ...]
        # 2 shared + 2 per zone
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(2 + 2 * self.n_zones,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-1e3, high=1e4,
            shape=(self.n_zones * 2 + 1,),  # temps + occupancies + time
            dtype=np.float32
)
        # convenience references
        self.air_handler = self.sim.hvac.air_handler
        self.boiler = self.sim.hvac.boiler
        self.vavs = self.sim.hvac.vavs  # dict: zone_id -> VAV object

    def _get_obs(self) -> np.ndarray:
        """Extract comprehensive observation."""
        zone_temps_dict = self.sim.building.get_zone_average_temps()
        temps = np.array([zone_temps_dict[zid] for zid in self.zone_ids], dtype=np.float32)

        start = self.sim.current_timestamp
        end = start + pd.to_timedelta(self.sim.time_step_sec, unit="s")

        occupancies = np.array(
            [
                occ.average_zone_occupancy(zone_id, start, end)
                for occ, zone_id in zip(self.occupancies, self.zone_ids)
            ],
            dtype=np.float32,
        )

        current_hour = start.hour / 24.0

        return np.concatenate([temps, occupancies, [current_hour]])

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            np.random.seed(seed)
        self.sim.reset()

        # deterministic occupancies when seed is given
        base_seed = 0 if seed is None else int(seed)
        self.occupancies = [
            RandomizedArrivalDepartureOccupancy(
                zone_assignment=10,
                earliest_expected_arrival_hour=7,
                latest_expected_arrival_hour=9,
                earliest_expected_departure_hour=16,
                latest_expected_departure_hour=18,
                time_step_sec=self.sim.time_step_sec,
                seed=base_seed + i,
                time_zone=self.time_zone,
            )
            for i in range(self.n_zones)
        ]
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).ravel()
        assert action.size == 2 + 2 * self.n_zones

        supply_air_action = action[0]
        boiler_action     = action[1]
        vav_damper_actions = action[2:2+self.n_zones]
        vav_reheat_actions = action[2+self.n_zones:2+2*self.n_zones]

        # supply air temp: [-1,1] -> [285.15, 305.15] K  (12°C to 32°C)
        supply_air_sp = 295.15 + supply_air_action * 10.0

        ah_heat_sp = supply_air_sp - 0.5
        ah_cool_sp = supply_air_sp + 0.5

        # boiler: [-1,1] -> [318.15, 338.15] K (45°C to 65°C)
        boiler_sp = 328.15 + boiler_action * 10.0

        action_timestamp = self.sim.current_timestamp

        if self.mutable_schedule is not None:
            self.mutable_schedule.set_temp_window((ah_heat_sp - 0.1, ah_heat_sp + 0.1))

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

        min_damper = 0.01
        damper_cmds = np.clip((vav_damper_actions + 1.0) * 0.5, min_damper, 1.0)
        reheat_cmds = np.clip((vav_reheat_actions + 1.0) * 0.5, 0.0, 1.0)

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

        r_obj = self.sim.reward_info(self.occupancies[0])
        comfort_penalty, energy_rate = self._reward_terms_from_reward_obj(r_obj)
        total_reward = self._combine_reward_terms(comfort_penalty, energy_rate)

        self.step_count += 1
        terminated = False
        truncated = self.step_count >= self.max_steps

        info: Dict = {
            "raw_reward_obj": r_obj,
            "comfort_penalty": float(comfort_penalty),
            "energy_rate": float(energy_rate),
            "reward_total": float(total_reward),
            "supply_air_sp_C": supply_air_sp - 273.15,
            "boiler_sp_C": boiler_sp - 273.15,
        }
        return obs, float(total_reward), terminated, truncated, info


    def _is_comfort_active(self) -> bool:
            """Return True only during configured working hours."""
            ts = self.sim.current_timestamp
            hour_of_day = ts.hour + ts.minute / 60.0 + ts.second / 3600.0

            start_hour, end_hour = self.comfort_hours

            # Normal same-day window, e.g. 8 -> 18
            if start_hour <= end_hour:
                return start_hour <= hour_of_day < end_hour

            # Overnight window, e.g. 22 -> 6
            return hour_of_day >= start_hour or hour_of_day < end_hour

    def _reward_terms_from_reward_obj(self, r: Any) -> Tuple[float, float]:
        comfort_low_k, comfort_high_k = self.comfort_band_k

        total_comfort_penalty = 0.0
        if self._is_comfort_active():
            for ziv in r.zone_reward_infos.values():
                t = float(ziv.zone_air_temperature)
                total_comfort_penalty += max(comfort_low_k - t, 0.0) + max(t - comfort_high_k, 0.0)

        total_energy_rate = (
            sum(float(ainfo.blower_electrical_energy_rate)
                for ainfo in r.air_handler_reward_infos.values()) +
            sum(float(binfo.natural_gas_heating_energy_rate) +
                float(binfo.pump_electrical_energy_rate)
                for binfo in r.boiler_reward_infos.values())
        )
        # FOR NOW ENERGY IS DISREGARDED
        # total_energy_rate = 0.0
        return float(total_comfort_penalty), float(total_energy_rate)


    def _combine_reward_terms(self, comfort_penalty: float, energy_rate: float) -> float:
      energy_weight = 1e-4
      return float(-(comfort_penalty + energy_weight * energy_rate))

    def _reduce_reward_obj_to_scalar(self, r: Any) -> float:
      comfort_penalty, energy_rate = self._reward_terms_from_reward_obj(r)
      return self._combine_reward_terms(comfort_penalty, energy_rate)

    def render(self, mode="human"):
        self.sim.get_video()
        return

    def close(self):
        return None
