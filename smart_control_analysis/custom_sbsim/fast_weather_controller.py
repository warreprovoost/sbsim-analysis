import numpy as np
import pandas as pd
from smart_control.simulator.weather_controller import ReplayWeatherController, _EPOCH
from smart_control.utils import conversion_utils


class FastReplayWeatherController(ReplayWeatherController):
    """ReplayWeatherController with cached arrays for fast get_current_temp.

    The base class rebuilds np.array(index) and reads the TempF Series on every
    call to get_current_temp, and also calls min()/max() over the full Time
    column. This subclass precomputes all of those once at construction.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._min_time = self._weather_data['Time'].iloc[0]
        self._max_time = self._weather_data['Time'].iloc[-1]
        self._times_array = np.array(self._weather_data.index)
        self._temps_array = self._weather_data['TempF'].to_numpy()

    def get_forecast_temps_c(self, timestamp: pd.Timestamp, horizon_hours: list) -> np.ndarray:
        """Return outdoor temperature (°C) at current + each offset in horizon_hours.
        Clamps to available data range so episode end doesn't crash.
        Weather data on a short term is pretty reliable"""

        timestamp_utc = timestamp.tz_convert('UTC')
        out = []
        for h in horizon_hours:
            t = timestamp_utc + pd.Timedelta(hours=h)
            t = min(max(t, self._min_time), self._max_time)
            target = (t - _EPOCH).total_seconds()
            temp_f = float(np.interp(target, self._times_array, self._temps_array))
            out.append(conversion_utils.fahrenheit_to_kelvin(temp_f) - 273.15)
        return np.array(out, dtype=np.float32)

    def get_current_temp(self, timestamp: pd.Timestamp) -> float:
        timestamp = timestamp.tz_convert('UTC')
        if timestamp < self._min_time:
            raise ValueError(
                f'Attempting to get weather data at {timestamp}, before the'
                f' earliest timestamp {self._min_time}.'
            )
        if timestamp > self._max_time:
            raise ValueError(
                f'Attempting to get weather data at {timestamp}, after the'
                f' latest timestamp {self._max_time}.'
            )
        target_timestamp = (timestamp - _EPOCH).total_seconds()
        temp_f = np.interp(target_timestamp, self._times_array, self._temps_array)
        return conversion_utils.fahrenheit_to_kelvin(temp_f)
