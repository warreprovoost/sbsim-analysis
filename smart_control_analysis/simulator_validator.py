from typing import Any, Callable, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class SimulatorValidator:
    """Validate simulator behavior by testing control responses."""

    def __init__(
        self,
        building_factory: Callable,
        base_params: Dict[str, Any],
    ):
        """
        Parameters
        ----------
        building_factory : callable
            Function that creates (building_sim, env) from params dict.
        base_params : dict
            Base parameters for building configuration.
        """
        self.building_factory = building_factory
        self.base_params = base_params.copy()

    def test_heating_response(
        self,
        target_heating_action: float = 0.8,
        target_vav_damper_action: float = 1.0,
        target_vav_reheat_action: float = 1.0,
        n_steps: int = 24,
    ) -> dict:
        """
        Test heating by applying constant high heating action with VAV controls.

        Parameters
        ----------
        target_heating_action : float
            Air handler heating setpoint action [-1, 1]
        target_vav_damper_action : float
            VAV damper opening action [-1, 1]
        target_vav_reheat_action : float
            VAV reheat valve opening action [-1, 1]
        n_steps : int
            Number of simulation steps
        """
        building_sim, env = self.building_factory(self.base_params)
        obs, _ = env.reset()

        print(f"Action space: {env.action_space}")
        print(f"Initial temp: {float(np.mean(obs)):.2f} K")

        start_timestamp = building_sim.current_timestamp
        timestamps = [building_sim.current_timestamp]
        temps = [round(float(np.mean(obs)), 4)]

        vav = building_sim.hvac.vavs["room_1"]
        vav.reheat_valve_setting = 1

        print(f"\n=== INITIAL VAV STATE ===")
        print(f"  damper_setting: {vav.damper_setting}")
        print(f"  reheat_valve_setting: {vav.reheat_valve_setting}")
        print(f"  flow_rate_demand: {vav.flow_rate_demand}")
        print(f"  reheat_demand: {vav.reheat_demand}")
        print(f"  zone_air_temperature: {vav.zone_air_temperature:.2f} K")
        print(f"  Air handler heating setpoint: {building_sim.hvac.air_handler.heating_air_temp_setpoint:.2f} K")

        for step in range(n_steps):
            # Build action array: [ah_heat, ah_cool, boiler, vav_damper, vav_reheat]
            action = np.zeros(env.action_space.shape[0], dtype=np.float32)
            action[0] = target_heating_action      # Air handler heating
            action[1] = 0.0                         # Air handler cooling (neutral)
            action[2] = 0.0                         # Boiler (neutral)
            action[3] = target_vav_damper_action   # VAV damper
            action[4] = target_vav_reheat_action   # VAV reheat valve

            vav.reheat_valve_setting = 1
            obs, _, _, _, _ = env.step(action)
            temp = round(float(np.mean(obs)), 4)
            temps.append(temp)
            timestamps.append(building_sim.current_timestamp)

            if step == 1:
                print(f"\nAfter first step:")
                print(f"  Zone temp: {temp:.4f} K")
                print(f"  damper_setting: {vav.damper_setting}")
                print(f"  reheat_valve_setting: {vav.reheat_valve_setting}")
                print(f"  flow_rate_demand: {vav.flow_rate_demand}")
                print(f"  reheat_demand: {vav.reheat_demand}")

        # Convert timestamps to hours elapsed from start
        hours_elapsed = np.array([(ts - start_timestamp).total_seconds() / 3600.0 for ts in timestamps])

        print(f"\n=== FINAL VAV STATE ===")
        print(f"  damper_setting: {vav.damper_setting}")
        print(f"  reheat_valve_setting: {vav.reheat_valve_setting}")
        print(f"  flow_rate_demand: {vav.flow_rate_demand}")
        print(f"  reheat_demand: {vav.reheat_demand}")
        print(f"  zone_air_temperature: {vav.zone_air_temperature:.2f} K")
        print(f"  Air handler heating setpoint: {building_sim.hvac.air_handler.heating_air_temp_setpoint:.2f} K")

        return {
            "timestamps": timestamps,
            "hours_elapsed": hours_elapsed,
            "mean_temp_trajectory": np.array(temps),
        }

    def plot_response(
        self,
        result: dict,
        title: str = "Temperature Response to Heating",
        figsize: Tuple[int, int] = (12, 6),
    ) -> plt.Figure:
        """Plot temperature response with elapsed time."""
        fig, ax = plt.subplots(figsize=figsize)

        hours_elapsed = result["hours_elapsed"]
        temps = result["mean_temp_trajectory"]

        ax.plot(
            hours_elapsed,
            temps,
            linewidth=2.5,
            label="Zone Temperature",
            color="steelblue",
            marker="o",
            markersize=8,
        )
        ax.set_xlabel("Hours Elapsed", fontsize=12)
        ax.set_ylabel("Temperature [K]", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # Set y-axis to show full degree range
        min_temp = np.floor(np.min(temps))
        max_temp = np.ceil(np.max(temps))
        temp_range = max(max_temp - min_temp, 10)
        center = (min_temp + max_temp) / 2
        ax.set_ylim(center - temp_range / 2, center + temp_range / 2)

        fig.tight_layout()
        return fig
