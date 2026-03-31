from smart_control_analysis.custom_sbsim.mutable_schedule import MutableSetpointSchedule
import pandas as pd
from smart_control.simulator.weather_controller import WeatherController
from smart_control_analysis.custom_sbsim.fast_weather_controller import FastReplayWeatherController
from typing import Any, Dict, Optional
import pytz
import warnings
import os


from smart_control_analysis.gym_wrapper import BuildingGymEnv
from smart_control.simulator.air_handler import AirHandler
from smart_control_analysis.custom_sbsim.safe_boiler import SafeBoiler
from smart_control.simulator.building import FloorPlanBasedBuilding
from smart_control.simulator.building import MaterialProperties
from smart_control.simulator.hvac_floorplan_based import FloorPlanBasedHvac
from smart_control.simulator.setpoint_schedule import SetpointSchedule
from smart_control_analysis.custom_sbsim.fast_cpu_simulator import FastCPUSimulator


def building_factory(
        params: dict,
        training_mode: Optional[str] = None,
        ):
    """
    Create simulator and environment with given parameters.

    Parameters
    ----------
    params : dict
        Configuration dict - now includes material properties and building params
    """
    import numpy as np

    sim_tz = params.get("time_zone", "America/Los_Angeles")
    start_ts = pd.Timestamp(params.get("start_timestamp", "2023-01-15 06:00:00"))
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize(sim_tz)

    # Ensure a minimum difference between outdoor_low_temp and outdoor_high_temp
    min_delta = 0.1  # Minimum allowed difference in °C
    low = params.get("outdoor_low_temp", 5)
    high = params.get("outdoor_high_temp", 20)
    if high - low < min_delta:
        # Adjust high to ensure the minimum difference
        high = low + min_delta
        params["outdoor_high_temp"] = high

    def single_room_floorplan(room_width: int, room_height: int):
        total_width = room_width + 2
        total_height = room_height + 2
        floorplan = np.full((total_height, total_width), 2, dtype=int)
        zone_map = np.full((total_height, total_width), -1, dtype=int)
        floorplan[1:-1, 1:-1] = 0
        zone_map[1:-1, 1:-1] = 0
        return floorplan, zone_map

    floor_plan, zone_map = single_room_floorplan(room_width=10, room_height=10)

    # Material properties from params
    inside_air = MaterialProperties(
        density=params.get("inside_air_density", 1.225),
        heat_capacity=params.get("inside_air_heat_capacity", 1005),
        conductivity=params.get("inside_air_conductivity", 0.025),
    )

    inside_wall = MaterialProperties(
        density=params.get("inside_wall_density", 2400),
        heat_capacity=params.get("inside_wall_heat_capacity", 880),
        conductivity=params.get("inside_wall_conductivity", 1.4),
    )

    exterior_wall = MaterialProperties(
        density=params.get("exterior_wall_density", 2000),
        heat_capacity=params.get("exterior_wall_heat_capacity", 900),
        conductivity=params.get("exterior_wall_conductivity", 0.4),
    )

    # Suppress known simulator warning from building_utils.py (UserWarning)
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module=r"smart_control\.simulator\.building_utils",
    )
    # Building with params
    building = FloorPlanBasedBuilding(
        cv_size_cm=params.get("cv_size_cm", 100.0),
        floor_height_cm=params.get("floor_height_cm", 300.0),
        initial_temp=params.get("initial_temp_celsius", 18.0) + 273.15,
        inside_air_properties=inside_air,
        inside_wall_properties=inside_wall,
        building_exterior_properties=exterior_wall,
        floor_plan=floor_plan,
        zone_map=zone_map,
        buffer_from_walls=params.get("buffer_from_walls", 3),
        convection_simulator=None,
        include_radiative_heat_transfer=False,
        view_factor_method="ScriptF",
    )
    weather_source = params.get("weather_source", "sinusoidal")
    if weather_source == "replay":
        weather_csv_path = os.path.expanduser(params.get("weather_csv_path", ""))
        if not weather_csv_path:
            raise ValueError("weather_source='replay' requires params['weather_csv_path']")
        weather_controller = FastReplayWeatherController(
            local_weather_path=weather_csv_path,
            convection_coefficient=params.get("convection_coefficient", 60.0),
        )
    else:
        weather_controller = WeatherController(
            default_low_temp=273.15 + params.get("outdoor_low_temp", 5),
            default_high_temp=273.15 + params.get("outdoor_high_temp", 15),
            convection_coefficient=params.get("convection_coefficient", 60.0),
        )

    boiler = SafeBoiler(
        reheat_water_setpoint=273.15 + params.get("boiler_setpoint_celsius", 55),
        water_pump_differential_head=params.get("pump_head", 6.1),
        water_pump_efficiency=params.get("pump_efficiency", 1.0),
    )

    air_handler = AirHandler(
        recirculation=params.get("recirculation", 0.8),
        heating_air_temp_setpoint=273.15 + params.get("ah_heating_setpoint_celsius", 20),
        cooling_air_temp_setpoint=273.15 + params.get("ah_cooling_setpoint_celsius", 40),
        fan_differential_pressure=params.get("fan_pressure", 800),
        fan_efficiency=params.get("fan_efficiency", 0.7),
        max_air_flow_rate=params.get("max_airflow", 8.67),
        device_id="main_AHU",
        sim_weather_controller=weather_controller,
    )

    # Our sim does not use this schedule
    schedule = SetpointSchedule(
        morning_start_hour=7,
        evening_start_hour=19,
        comfort_temp_window=(273+22, 273+23.5),
        eco_temp_window=(273+18, 273+28),
        time_zone=pytz.timezone(sim_tz),  # was Europe/Brussels
    )

    hvac = FloorPlanBasedHvac(
        air_handler=air_handler,
        boiler=boiler,
        schedule=schedule,
        vav_max_air_flow_rate=params.get("vav_max_flow", 1.2),
        vav_reheat_max_water_flow_rate=params.get("vav_reheat_flow", 0.1),
        zone_identifier=["room_1"],
    )

    building_sim = FastCPUSimulator(
        building=building,
        hvac=hvac,
        weather_controller=weather_controller,
        time_step_sec=params.get("time_step_sec", 300),
        convergence_threshold=0.5,
        iteration_limit=48,
        iteration_warning=35,
        start_timestamp=start_ts,   # <- use tz-aware timestamp
    )

    occupancy_model = "randomized"
    if training_mode == "comfort_only":
        occupancy_model = "step"
    elif training_mode == "full":
        occupancy_model = "randomized"

    # Peak energy estimate from first principles (all at max VAV settings):
    # Boiler gas: flow * Cp_water * (boiler_sp - supply_air_sp)
    _delta_t = params.get("boiler_setpoint_celsius", 45) - params.get("ah_heating_setpoint_celsius", 18)
    _peak_boiler_w = params.get("vav_reheat_flow", 0.001) * 4186.0 * max(_delta_t, 1.0)
    # Blower: actual VAV flow (not rated max) * fan_pressure / fan_efficiency
    _peak_blower_w = (params.get("fan_pressure", 800) * params.get("vav_max_flow", 0.01)
                      / params.get("fan_efficiency", 0.7))
    _energy_norm = _peak_boiler_w + _peak_blower_w  # W — ~124 W for default params

    env = BuildingGymEnv(
        sim=building_sim,
        time_zone=sim_tz,
        occupancy_model=occupancy_model,
        comfort_band_k=params.get("comfort_band_k", (294.15, 295.15)),
        working_hours=params.get("working_hours", (8.0, 18.0)),
        max_steps=params.get("max_steps", None),
        occupancy_per_zone=params.get("occupancy_per_zone", 10.0),
        energy_norm=_energy_norm,
    )
    return building_sim, env


def get_base_params() -> dict:
    """Return default parameter configuration."""
    return {
        # Sim
        "time_step_sec": 60,
        "max_steps": int(7 * 24 * 3600 / 60),
        "working_hours":  (0.0, 24.0),


        # Weather
        "outdoor_low_temp": -5,
        "outdoor_high_temp": 5,
        "convection_coefficient": 50.0,    # stronger outside exchange
        "start_timestamp": "2023-01-16 06:00:00",
        "time_zone": "Europe/Oslo",
        "weather_source": "replay",
        "weather_csv_path": "~/thesis/weather_data/oslo_weather_multiyear.csv",


        # Boiler
        "boiler_setpoint_celsius": 45,
        "pump_head": 6.1,
        "pump_efficiency": 0.65,

        # Air handler
        "recirculation": 0.7,
        "ah_heating_setpoint_celsius": 18,
        "ah_cooling_setpoint_celsius": 40,
        "fan_pressure": 800,
        "fan_efficiency": 0.7,
        "max_airflow": 8.67,

        # VAV
        "vav_max_flow": 0.01, # this is flow_rate_demand when vav is turned 'on'
        "vav_reheat_flow": 0.001, # this is reheat_demand when vav is turned 'on'

        # Building geometry
        "cv_size_cm": 25.0,
        "floor_height_cm": 300.0,
        "initial_temp_celsius": 20.0,
        "buffer_from_walls": 1,

        # Material properties - inside air
        "inside_air_density": 1.225,
        "inside_air_heat_capacity": 1005,
        "inside_air_conductivity": 0.025,

        # Material properties - inside wall
        "inside_wall_density": 2400,
        "inside_wall_heat_capacity": 880,
        "inside_wall_conductivity": 1.4,

        # Material properties - exterior wall
        "exterior_wall_density": 800,
        "exterior_wall_heat_capacity": 840,
        "exterior_wall_conductivity": 2.0,
    }
