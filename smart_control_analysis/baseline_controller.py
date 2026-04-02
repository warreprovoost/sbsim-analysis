import numpy as np


class ThermostatBaselineController:
    """
    Simple rule-based baseline: maintain comfort band during working hours.

    Strategy:
    - During working hours: heat/cool to keep zones in comfort band
    - Outside working hours: turn everything off
    - Per-zone control via reheat valves
    - Global boiler + supply air setpoints
    """

    def __init__(self, comfort_band_k=(294.15, 295.15), working_hours=(8.0, 18.0),
                 night_setback_k=2.0,
                 boiler_sp_heating_k=338.15, boiler_sp_idle_k=318.15, supply_air_sp_k=295.15):
        self.comfort_low_k, self.comfort_high_k = comfort_band_k
        self.working_hours = working_hours
        self.night_setback_k = float(night_setback_k)
        self.boiler_sp_heating_k = boiler_sp_heating_k
        self.boiler_sp_idle_k = boiler_sp_idle_k
        self.supply_air_sp_k = supply_air_sp_k
        self._boiler_on = False
        self._boiler_state_initialized = False
        self._boiler_steps_since_switch = 0

        # wider hysteresis for coarse timesteps
        self.boiler_on_margin_k = 0.6    # ON below low-0.6K
        self.boiler_off_margin_k = 0.6   # OFF above low+0.6K

        # anti-chatter dwell times (in control steps)
        self.min_on_steps = 4
        self.min_off_steps = 4


    def _comfort_band_now_k(self, timestamp):
        """Return (low_k, high_k) adjusted for night setback."""
        if self.night_setback_k == 0.0:
            return self.comfort_low_k, self.comfort_high_k
        hour = timestamp.hour + timestamp.minute / 60.0
        start_h, end_h = self.working_hours
        if start_h <= end_h:
            in_working_hours = start_h <= hour < end_h
        else:
            in_working_hours = hour >= start_h or hour < end_h
        if in_working_hours:
            return self.comfort_low_k, self.comfort_high_k
        return self.comfort_low_k - self.night_setback_k, self.comfort_high_k  # only lower the floor

    def _is_working_hours(self, timestamp) -> bool:

        return True # FORCE every hour of every day you work
        """Check if current time is within working hours."""
        hour = timestamp.hour + timestamp.minute / 60.0
        start_hour, end_hour = self.working_hours

        if start_hour <= end_hour:
            return start_hour <= hour < end_hour
        else:  # overnight shift
            return hour >= start_hour or hour < end_hour

    def get_action(self, obs: np.ndarray, env) -> np.ndarray:
        """
        Compute baseline action based on observations.

        Parameters
        ----------
        obs : np.ndarray
            Observation from env._get_obs() — only zone temps (first n_zones) are used directly.
        env : BuildingGymEnv
            Environment reference (to access zone temps and timestamps)

        Returns
        -------
        action : np.ndarray
            Action vector [supply_air, boiler, damper, reheat_1, ..., reheat_n]
        """
        n_zones = env.n_zones
        timestamp = env.sim.current_timestamp

        # Read zone temps directly from sim — never parse obs (layout changes break this)
        zone_temps_dict = env.sim.building.get_zone_average_temps()
        zone_temps_k = np.array([zone_temps_dict[zid] for zid in env.zone_ids], dtype=np.float32)
        zone_occupancies = np.ones(n_zones, dtype=np.float32)  # always occupied

        action_design = getattr(env, "action_design", "reheat_per_zone")
        if action_design == "full_per_zone":
            action = np.zeros(2 + 2 * n_zones, dtype=np.float32)
        else:
            action = np.zeros(3 + n_zones, dtype=np.float32)

        if not self._is_working_hours(timestamp):
            self._boiler_on = False
            self._boiler_state_initialized = False
            self._boiler_steps_since_switch = 0
            action[:] = -1.0  # everything off/closed
            return action

        action[0] = 0.0  # supply_air_sp = 22°C (center)

        comfort_low_k, comfort_high_k = self._comfort_band_now_k(timestamp)
        comfort_mid_k = 0.5 * (comfort_low_k + comfort_high_k)
        min_temp = float(np.min(zone_temps_k))
        on_th = comfort_mid_k - self.boiler_on_margin_k
        off_th = comfort_mid_k + self.boiler_off_margin_k

        if not self._boiler_state_initialized:
            self._boiler_on = (min_temp < on_th)
            self._boiler_state_initialized = True
            self._boiler_steps_since_switch = 0
        else:
            self._boiler_steps_since_switch += 1
            if self._boiler_on:
                if self._boiler_steps_since_switch >= self.min_on_steps and min_temp > off_th:
                    self._boiler_on = False
                    self._boiler_steps_since_switch = 0
            else:
                if self._boiler_steps_since_switch >= self.min_off_steps and min_temp < on_th:
                    self._boiler_on = True
                    self._boiler_steps_since_switch = 0

        action[1] = 1.0 if self._boiler_on else -1.0

        # Per-zone reheat: proportional to each zone's coldness
        reheat_vals = np.array([
            float(np.clip((comfort_low_k - zone_temps_k[i]) / 1.0, -1.0, 1.0))
            for i in range(n_zones)
        ], dtype=np.float32)

        # Per-zone damper: open for cold zones, closed for warm zones
        damper_vals = np.array([
            -1.0 if zone_occupancies[i] < 0.1 or zone_temps_k[i] > comfort_high_k + 1.0
            else (1.0 if zone_temps_k[i] < comfort_low_k
                  else float(np.clip((comfort_low_k - zone_temps_k[i]) / 1.0, -1.0, 1.0)))
            for i in range(n_zones)
        ], dtype=np.float32)

        if action_design == "reheat_per_zone":
            # [supply, boiler, shared_damper, reheat_0..n]
            action[2] = float(np.mean(damper_vals))   # shared damper = mean demand
            action[3:3 + n_zones] = reheat_vals
        elif action_design == "damper_per_zone":
            # [supply, boiler, shared_reheat, damper_0..n]
            action[2] = float(np.clip((comfort_low_k - min_temp) / 1.0, -1.0, 1.0))
            action[3:3 + n_zones] = damper_vals
        else:  # full_per_zone
            # [supply, boiler, reheat_0..n, damper_0..n]
            action[2:2 + n_zones] = reheat_vals
            action[2 + n_zones:2 + 2 * n_zones] = damper_vals

        return action
