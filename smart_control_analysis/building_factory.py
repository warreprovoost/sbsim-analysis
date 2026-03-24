from smart_control_analysis.custom_sbsim.mutable_schedule import MutableSetpointSchedule
import pandas as pd
import pytz

from smart_control_analysis.gym_wrapper import BuildingGymEnv
from smart_control.simulator.air_handler import AirHandler
from smart_control.simulator.boiler import Boiler
from smart_control.simulator.building import FloorPlanBasedBuilding
from smart_control.simulator.building import MaterialProperties
from smart_control.simulator.hvac_floorplan_based import FloorPlanBasedHvac
from smart_control.simulator.setpoint_schedule import SetpointSchedule
from smart_control.simulator.weather_controller import WeatherController
from smart_control_analysis.custom_sbsim.direct_vav_tf_simulator import DirectVavTFSimulator


def building_factory(params: dict):
    """
    Create simulator and environment with given parameters.

    Parameters
    ----------
    params : dict
        Configuration dict - now includes material properties and building params
    """
    import numpy as np

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
    weather_controller = WeatherController(
        default_low_temp=273.15 + params.get("outdoor_low_temp", 5),
        default_high_temp=273.15 + params.get("outdoor_high_temp", 20),
        convection_coefficient=params.get("convection_coefficient", 60.0),  # stronger outside convection

    )

    boiler = Boiler(
        reheat_water_setpoint=273.15 + params.get("boiler_setpoint_celsius", 55),
        water_pump_differential_head=params.get("pump_head", 60000),
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

    mutable_schedule = MutableSetpointSchedule(time_zone=pytz.timezone("Europe/Brussels"))

    schedule = SetpointSchedule(
            morning_start_hour=7,                 # Start comfort mode at 07:00
            evening_start_hour=19,                # Start eco mode at 19:00
            comfort_temp_window=(273+22,273+23.5),
            eco_temp_window=(273+18, 273+28),
            time_zone=pytz.timezone("Europe/Brussels")
        )

    hvac = FloorPlanBasedHvac(
        air_handler=air_handler,
        boiler=boiler,
        schedule=schedule,
        vav_max_air_flow_rate=params.get("vav_max_flow", 1.2),
        vav_reheat_max_water_flow_rate=params.get("vav_reheat_flow", 0.1),
        zone_identifier=["room_1"],
    )

    building_sim = DirectVavTFSimulator(
        building=building,
        hvac=hvac,
        weather_controller=weather_controller,
        time_step_sec=1500,
        convergence_threshold=0.5,
        iteration_limit=48,
        iteration_warning=35,
        start_timestamp=pd.Timestamp("2025-12-15 08:00:00"),
    )

    env = BuildingGymEnv(building_sim, mutable_schedule=None)
    return building_sim, env


def get_base_params() -> dict:
    """Return default parameter configuration."""
    return {
        # Weather
        "outdoor_low_temp": 10,
        "outdoor_high_temp": 20,

        # Boiler
        "boiler_setpoint_celsius": 55,
        "pump_head": 60000,
        "pump_efficiency": 1.0,

        # Air handler
        "recirculation": 0.8,
        "ah_heating_setpoint_celsius": 18,
        "ah_cooling_setpoint_celsius": 40,
        "fan_pressure": 800,
        "fan_efficiency": 0.7,
        "max_airflow": 8.67,

        # VAV
        "vav_max_flow": 1.4, # this is flow_rate_demand when vav is turned 'on'
        "vav_reheat_flow": 0.4, # this is reheat_demand when vav is turned 'on'

        # Building geometry
        "cv_size_cm": 25.0,
        "floor_height_cm": 300.0,
        "initial_temp_celsius": 20.0,
        "buffer_from_walls": 3,

        # Material properties - inside air
        "inside_air_density": 1.225,
        "inside_air_heat_capacity": 1005,
        "inside_air_conductivity": 0.025,

        # Material properties - inside wall
        "inside_wall_density": 2400,
        "inside_wall_heat_capacity": 880,
        "inside_wall_conductivity": 1.4,

        # Material properties - exterior wall
        "exterior_wall_density": 2000,
        "exterior_wall_heat_capacity": 900,
        "exterior_wall_conductivity": 0.4,
    }

def get_base_params_with_varying_weather() -> dict:
    """Extreme params to force significant night temperature drops."""
    return {
        # Weather: very cold nights, slightly less cold days
        "outdoor_low_temp": 0,
        "outdoor_high_temp": 10,

        # Boiler (keep moderate so it can’t fully compensate)
        "boiler_setpoint_celsius": 45,  # lower reheat water setpoint

        "pump_head": 40000,
        "pump_efficiency": 0.7,

        # Air handler: less recirculation so more cold air mixes in
        "recirculation": 0.01,                 # was 0.8; more outdoor air -> colder supply
        "ah_heating_setpoint_celsius": 18,    # modest heating
        "ah_cooling_setpoint_celsius": 40,
        "fan_pressure": 600,
        "fan_efficiency": 0.6,
        "max_airflow": 6.0,

        # VAV
        "vav_max_flow": 1.0,
        "vav_reheat_flow": 0.2,

        # Building geometry: thin CV → less mass per wall cell
        "cv_size_cm": 5.0,       # 10 cm cells — thin walls
        "floor_height_cm": 300.0,
        "initial_temp_celsius": 20.0,
        "buffer_from_walls": 1,

        # Inside air
        "inside_air_density": 1.225,
        "inside_air_heat_capacity": 600,
        "inside_air_conductivity": 0.03,

        # Interior walls: almost no thermal mass, some conductivity
        "inside_wall_density": 300,
        "inside_wall_heat_capacity": 400,
        "inside_wall_conductivity": 1.0,

        # Exterior walls: extremely leaky, minimal mass
        "exterior_wall_density": 1,
        "exterior_wall_heat_capacity": 300,
        "exterior_wall_conductivity": 1000.0,   # very high → fast heat loss
    }
