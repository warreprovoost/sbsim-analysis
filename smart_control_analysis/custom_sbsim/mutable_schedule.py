import datetime
from typing import Optional, Set, Tuple

import pandas as pd
import pytz

from smart_control.simulator.setpoint_schedule import SetpointSchedule

TemperatureWindow = Tuple[int, int]


class MutableSetpointSchedule(SetpointSchedule):
    """SetpointSchedule that allows runtime updates of temp windows."""

    def __init__(
        self,
        morning_start_hour: int = 0, # Only use day time behaviour for now
        evening_start_hour: int= 24,
        temp_window: TemperatureWindow = (293,294),
        holidays: Optional[Set[int]] = None,
        time_zone: datetime.tzinfo = pytz.UTC,
    ):
        super().__init__(
            morning_start_hour=morning_start_hour,
            evening_start_hour=evening_start_hour,
            comfort_temp_window=temp_window,
            eco_temp_window=temp_window,
            holidays=holidays,
            time_zone=time_zone,
        )

    @staticmethod
    def _validate_window(window: TemperatureWindow) -> None:
        if not isinstance(window, tuple) or len(window) != 2:
            raise ValueError("Temperature window must be a 2-tuple (low, high).")
        low, high = window
        if low >= high:
            raise ValueError("Temperature window must have low < high.")

    def set_temp_window(self, window: TemperatureWindow) -> None:
        self._validate_window(window)
        self.comfort_temp_window = window
        self.eco_temp_window = window
