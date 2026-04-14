import numpy as np


class ThermostatBaselineController:
    """
    Simple thermostat baseline.

    Boiler: on/off with deadband on the coldest zone.
    Reheat valves: proportional to how far each zone is below the setpoint (slow ramp).
    Dampers: close immediately if zone is too warm (aggressive cooling).
    Night setback: comfort band floor lowered outside working hours.
    """

    def __init__(self, comfort_band_k=(294.15, 295.15), working_hours=(8.0, 18.0),
                 night_setback_k=2.0, deadband_k=0.2, prop_band_k=2.0,
                 night_off=False):
        self.comfort_low_k, self.comfort_high_k = comfort_band_k
        self.working_hours = working_hours
        self.night_setback_k = float(night_setback_k)
        self.deadband_k = deadband_k    # hysteresis for boiler on/off
        self.prop_band_k = prop_band_k  # reheat ramps from 0→1 over this many °C below setpoint
        self.night_off = night_off      # if True: all actuators off at night (no heating)
        # Note: boiler turns on at comfort_low - deadband_k, reheat is full at prop_band_k below floor
        self._heating_on = False

    def _comfort_band_now_k(self, timestamp):
        hour = timestamp.hour + timestamp.minute / 60.0
        start_h, end_h = self.working_hours
        in_working_hours = start_h <= hour < end_h if start_h <= end_h else (hour >= start_h or hour < end_h)
        if in_working_hours or self.night_setback_k == 0.0:
            return self.comfort_low_k, self.comfort_high_k
        return self.comfort_low_k - self.night_setback_k, self.comfort_high_k

    def get_action(self, obs: np.ndarray, env) -> np.ndarray:
        n_zones = env.n_zones
        timestamp = env.sim.current_timestamp

        zone_temps_dict = env.sim.building.get_zone_average_temps()
        zone_temps_k = np.array([zone_temps_dict[zid] for zid in env.zone_ids], dtype=np.float32)

        comfort_low_k, comfort_high_k = self._comfort_band_now_k(timestamp)
        comfort_mid_k = 0.5 * (comfort_low_k + comfort_high_k)
        hour = timestamp.hour + timestamp.minute / 60.0
        start_h, end_h = self.working_hours
        is_working = start_h <= hour < end_h if start_h <= end_h else (hour >= start_h or hour < end_h)
        # During working hours target midpoint; at night target just above floor (save energy)
        target_k = comfort_mid_k if is_working else comfort_low_k + 0.2
        min_temp = float(np.min(zone_temps_k))

        # Night off: all actuators to -1, no heating outside working hours
        if self.night_off and not is_working:
            self._heating_on = False
            action_design = getattr(env, "action_design", "reheat_per_zone")
            if action_design == "full_per_zone":
                return np.full(2 + 2 * n_zones, -1.0, dtype=np.float32)
            else:
                return np.full(3 + n_zones, -1.0, dtype=np.float32)

        # Boiler: on/off with deadband around target
        if min_temp < target_k - self.deadband_k:
            self._heating_on = True
        elif min_temp > target_k + self.deadband_k:
            self._heating_on = False

        action_design = getattr(env, "action_design", "reheat_per_zone")
        if action_design == "full_per_zone":
            action = np.zeros(2 + 2 * n_zones, dtype=np.float32)
        else:
            action = np.zeros(3 + n_zones, dtype=np.float32)

        # Supply air and boiler
        action[0] = 0.0                                   # supply air: center
        action[1] = 0.25 if self._heating_on else -1.0    # boiler: 1.0=65°C when on, -1.0=36°C (off)

        # Per-zone reheat: proportional ramp targeting current target setpoint.
        # Output in [-1, 1]: -1=off, +1=full. Mapped to [0,1] by env as (a+1)/2.
        reheat_vals = np.array([
            float(np.clip((target_k - zone_temps_k[i]) / self.prop_band_k, 0.0, 0.5)) * 2.0 - 1.0
            for i in range(n_zones)
        ], dtype=np.float32)

        # Per-zone damper: close aggressively if too warm, open proportionally if cold.
        damper_vals = np.array([
            -1.0 if zone_temps_k[i] > comfort_high_k  # too warm: close immediately
            else float(np.clip((target_k - zone_temps_k[i]) / self.prop_band_k, 0.0, 1.0)) * 2.0 - 1.0
            for i in range(n_zones)
        ], dtype=np.float32)

        if action_design == "reheat_per_zone":
            action[2] = float(np.mean(damper_vals))
            action[3:3 + n_zones] = reheat_vals
        elif action_design == "damper_per_zone":
            action[2] = float(np.clip((target_k - min_temp) / self.prop_band_k, 0.0, 0.75)) * 2.0 - 1.0
            action[3:3 + n_zones] = damper_vals
        else:  # full_per_zone
            action[2:2 + n_zones] = reheat_vals
            action[2 + n_zones:2 + 2 * n_zones] = damper_vals

        return action
