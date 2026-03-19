from smart_control.simulator.tf_simulator import TFSimulator


class DirectVavTFSimulator(TFSimulator):
  """TFSimulator variant that does not let thermostat overwrite VAV commands."""

  def setup_step_sim(self) -> None:
    # Intentionally no-op:
    # Base Simulator.setup_step_sim() calls vav.update_settings(...),
    # which routes through Thermostat and overwrites direct RL actions.
    return None
