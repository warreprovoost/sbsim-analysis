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
    Gymnasium wrapper that uses device.set_action(...) for controls.

    Architecture:
    - 1 air_handler: supply_air_heating_temperature_setpoint, supply_air_cooling_temperature_setpoint
    - 1 boiler: supply_water_setpoint
    - n_zones VAVs (sim.hvac.vavs dict): each has supply_air_damper_percentage_command and reheat_valve_setting

    Action space:
    - [0]: air_handler heating setpoint [-1,1] -> [273.15, 313.15] K (0°C - 40°C)
    - [1]: air_handler cooling setpoint [-1,1] -> [283.15, 303.15] K (10°C - 30°C)
    - [2]: boiler water setpoint [-1,1] -> [318, 338] K (45°C - 65°C)
    - [3:3+n_zones]: per-zone VAV damper percentage [-1,1] -> [0, 1]
    - [3+n_zones:3+2*n_zones]: per-zone VAV reheat valve [-1,1] -> [0, 1]
    Total action dim: 3 + 2*n_zones
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, sim: Any, mutable_schedule: MutableSetpointSchedule = None, time_zone: str = "Europe/Brussels", seed: Optional[int] = None,  max_steps: Optional[int] = None):
        super().__init__()
        self.sim = sim
        self.mutable_schedule = mutable_schedule
        self.time_zone = time_zone
        # Default to steps for 24h
        self.max_steps = max_steps if max_steps is not None else (24 * 60 * 60) // self.sim.time_step_sec


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
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3 + 2 * self.n_zones,), dtype=np.float32)
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
        if self.mutable_schedule is not None:
            self.mutable_schedule.set_temp_window((293, 294))

        # recreate occupancies for fresh randomness each episode
        self.occupancies = [
            RandomizedArrivalDepartureOccupancy(
                zone_assignment=10, # this is the number of occupants
                earliest_expected_arrival_hour=7,
                latest_expected_arrival_hour=9,
                earliest_expected_departure_hour=16,
                latest_expected_departure_hour=18,
                time_step_sec=self.sim.time_step_sec,
                seed=None,
                time_zone=self.time_zone,
            )
            for i in range(self.n_zones)
        ]

        self.step_count = 0
        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).ravel()
        assert action.size == 3 + 2 * self.n_zones

        # split action components
        ah_heat_action = action[0]
        ah_cool_action = action[1]
        boiler_action = action[2]
        vav_damper_actions = action[3:3+self.n_zones]
        vav_reheat_actions = action[3+self.n_zones:3+2*self.n_zones]

        # map actions to setpoints
        # air handler heating: [-1,1] -> [273.15, 313.15] K (0°C - 40°C)
        ah_heat_sp = 293.15 + ah_heat_action * 20.0  # Center at 20°C, range ±20K
        # air handler cooling: [-1,1] -> [283.15, 303.15] K (10°C - 30°C)
        ah_cool_sp = 293.15 + ah_cool_action * 10.0
        # boiler water: [-1,1] -> [318, 338] K (45°C - 65°C)
        boiler_sp = 328.0 + boiler_action * 10.0

        # get current timestamp
        action_timestamp = self.sim.current_timestamp

        # This controls the Thermostat indirectly.
        if self.mutable_schedule is not None:
          self.mutable_schedule.set_temp_window((ah_heat_sp - 0.1, ah_heat_sp + 0.1))

        # apply air handler setpoints
        self.air_handler.set_action("supply_air_heating_temperature_setpoint", float(ah_heat_sp), action_timestamp)
        self.air_handler.set_action("supply_air_cooling_temperature_setpoint", float(ah_cool_sp), action_timestamp)

        # apply boiler setpoint
        self.boiler.set_action("supply_water_setpoint", float(boiler_sp), action_timestamp)

        # map [-1,1] -> [0,1], then enforce min damper > 0 to avoid Vav.output() assert
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

        # advance simulator one timestep
        #print(self.sim.hvac.vavs["room_1"].reheat_valve_setting)
        self.sim.step_sim()
        #print(self.sim.hvac.vavs["room_1"].reheat_valve_setting)

        obs = self._get_obs()

        # call reward_info once per occupancy and sum
        total_reward = 0.0
        raw_reward_objs = []
        for occ in self.occupancies:
            r_obj = self.sim.reward_info(occ)
            raw_reward_objs.append(r_obj)
            total_reward += self._reduce_reward_obj_to_scalar(r_obj)


        self.step_count += 1
        terminated = False  # What could a termination condition be?
        truncated = self.step_count >= self.max_steps  # Truncate after max_steps

        info: Dict = {"raw_reward_objs": raw_reward_objs}
        return obs, float(total_reward), terminated, truncated, info

    def _reduce_reward_obj_to_scalar(self, r: Any) -> float:
      total_comfort_penalty = sum(
          abs(float(ziv.zone_air_temperature) -
              0.5 * (float(ziv.heating_setpoint_temperature) +
                    float(ziv.cooling_setpoint_temperature)))
          for ziv in r.zone_reward_infos.values()
      )

      total_energy_rate = (
          sum(float(ainfo.blower_electrical_energy_rate)
              for ainfo in r.air_handler_reward_infos.values()) +
          sum(float(binfo.natural_gas_heating_energy_rate) +
              float(binfo.pump_electrical_energy_rate)
              for binfo in r.boiler_reward_infos.values())
      )

      # Normalize: comfort in K, energy in W
      energy_weight = 1e-4  # Adjust based on your scaling preferences
      reward = -(total_comfort_penalty + energy_weight * total_energy_rate)

      return float(reward)

    def render(self, mode="human"):
        self.sim.get_video()
        return

    def close(self):
        return None
