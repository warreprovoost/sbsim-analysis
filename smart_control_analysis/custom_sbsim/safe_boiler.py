from smart_control.simulator.boiler import Boiler


class SafeBoiler(Boiler):
    """Boiler that clamps water_temp >= outside_temp in thermal dissipation.

    The base class asserts water_temp >= outside_temp, which can fail when the
    boiler tank transiently cools toward ambient (e.g. after the RL agent drives
    boiler_sp very low and return_water_temp drops below outside_temp).
    """

    def compute_thermal_energy_rate(
        self, return_water_temp: float, outside_temp: float
    ) -> float:
        # Clamp return_water_temp to [outside_temp, setpoint] to prevent
        # flow_heating_energy_rate from exploding when return temp is unphysical
        return_water_temp = float(return_water_temp)
        outside_temp = float(outside_temp)
        return_water_temp = max(return_water_temp, outside_temp)
        return_water_temp = min(return_water_temp, float(self._reheat_water_setpoint))
        return super().compute_thermal_energy_rate(return_water_temp, outside_temp)

    def compute_thermal_dissipation_rate(
        self, water_temp: float, outside_temp: float
    ) -> float:
        # Clamp: guard against NaN and water_temp < outside_temp
        water_temp = float(water_temp)
        outside_temp = float(outside_temp)
        if not (water_temp >= outside_temp):  # handles NaN too
            water_temp = outside_temp
        return super().compute_thermal_dissipation_rate(water_temp, outside_temp)
