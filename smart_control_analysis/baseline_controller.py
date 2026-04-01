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
                 boiler_sp_heating_k=338.15, boiler_sp_idle_k=318.15, supply_air_sp_k=295.15):
        self.comfort_low_k, self.comfort_high_k = comfort_band_k
        self.working_hours = working_hours
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

        action = np.zeros(3 + n_zones, dtype=np.float32)

        if not self._is_working_hours(timestamp):
            self._boiler_on = False
            self._boiler_state_initialized = False
            self._boiler_steps_since_switch = 0
            # Outside working hours: turn everything off
            action[0] = -1.0  # supply air low
            action[1] = -1.0  # boiler low (idle)
            action[2] = -1.0  # damper closed
            action[3:] = -1.0  # all reheat off
            return action

        # During working hours: maintain comfort band
        # Supply air: keep at neutral setpoint
        action[0] = 0.0  # supply_air_sp = 22°C (center)

        # Boiler: heat to comfort midpoint with hysteresis
        comfort_mid_k = 0.5 * (self.comfort_low_k + self.comfort_high_k)
        min_temp = float(np.min(zone_temps_k))
        on_th = comfort_mid_k - self.boiler_on_margin_k   # ON below midpoint-0.6
        off_th = comfort_mid_k + self.boiler_off_margin_k  # OFF above midpoint+0.6

        # initialize once when entering working hours
        if not self._boiler_state_initialized:
            self._boiler_on = (min_temp < on_th)
            self._boiler_state_initialized = True
            self._boiler_steps_since_switch = 0
        else:
            self._boiler_steps_since_switch += 1

            if self._boiler_on:
                # allow OFF only after minimum ON time
                if self._boiler_steps_since_switch >= self.min_on_steps and min_temp > off_th:
                    self._boiler_on = False
                    self._boiler_steps_since_switch = 0
            else:
                # allow ON only after minimum OFF time
                if self._boiler_steps_since_switch >= self.min_off_steps and min_temp < on_th:
                    self._boiler_on = True
                    self._boiler_steps_since_switch = 0

        action[1] = 1.0 if self._boiler_on else -1.0

        # Damper: open during working hours to allow airflow
        action[2] = 1.0  # damper fully open

        # Per-zone reheat: proportional control
        for i in range(n_zones):
            temp = zone_temps_k[i]
            occ = zone_occupancies[i]

            if occ < 0.1:
                # Zone unoccupied: no heating
                action[3 + i] = -1.0
            elif temp < self.comfort_low_k - 1.0:
                # Zone too cold: full reheat
                action[3 + i] = 1.0
            elif temp > self.comfort_high_k + 1.0:
                # Zone too hot: no reheat (let damper/AH cool)
                action[3 + i] = -1.0
            else:
                # Proportional control within dead band
                error = self.comfort_low_k - temp
                reheat_cmd = np.clip(error / 1.0, -1.0, 1.0)
                action[3 + i] = float(reheat_cmd)

        return action
