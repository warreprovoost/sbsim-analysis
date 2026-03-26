import numpy as np
import gymnasium as gym
from gymnasium import spaces


class FixedActionsWrapper(gym.ActionWrapper):
    """
    Wrapper that fixes certain action indices to constant values,
    exposing only the remaining actions to the RL agent.

    Example for 1 zone (4 actions total):
        fixed = {0: 0.0, 1: 1.0}  # fix supply_air=22C, boiler=65C
        → agent only controls actions [2, 3] (damper, reheat)

    Example for 10 zones (22 actions total):
        fixed = {0: 0.0, 1: 1.0}  # fix supply_air and boiler
        → agent controls actions [2..21] (10 dampers + 10 reheats)
    """

    def __init__(self, env: gym.Env, fixed: dict):
        """
        Parameters
        ----------
        env : gym.Env
        fixed : dict
            mapping {action_idx: fixed_value} in original action space
        """
        super().__init__(env)
        self.fixed = fixed
        self.n_total = env.action_space.shape[0]

        # indices the agent will control
        self.free_indices = [i for i in range(self.n_total) if i not in fixed]

        low  = env.action_space.low[self.free_indices]
        high = env.action_space.high[self.free_indices]
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def action(self, agent_action: np.ndarray) -> np.ndarray:
        """Expand agent action back to full action vector."""
        full = np.zeros(self.n_total, dtype=np.float32)

        # fill fixed values
        for idx, val in self.fixed.items():
            full[idx] = float(val)

        # fill agent-controlled values
        for agent_idx, env_idx in enumerate(self.free_indices):
            full[env_idx] = float(agent_action[agent_idx])

        return full
