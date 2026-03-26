from smart_control.simulator.tf_simulator import TFSimulator


class DirectVavTFSimulator(TFSimulator):
  """TFSimulator variant for direct RL control (no thermostat overwrite, no video/log frames)."""

  def __init__(self, *args, disable_video: bool = True, **kwargs):
    super().__init__(*args, **kwargs)
    self._disable_video = bool(disable_video)

    if self._disable_video:
      # Disable frame logging if present
      lap = getattr(self, "_log_and_plotter", None)
      if lap is not None and hasattr(lap, "log"):
        lap.log = lambda *a, **k: None

  def setup_step_sim(self) -> None:
    # No thermostat write-back to VAVs.
    return None

  def get_video(self, *args, **kwargs):
    # Hard-disable video generation.
    if self._disable_video:
      return None
    return super().get_video(*args, **kwargs)
