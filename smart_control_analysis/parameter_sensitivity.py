from typing import Any, Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


class ParameterSensitivityAnalyzer:
    """Analyze how building parameters affect zone temperatures."""

    def __init__(self, building_factory: Callable, env_factory: Callable):
        """
        Parameters
        ----------
        building_factory : callable
            Function that returns (building_sim, env) given parameter dict.
            Signature: (params_dict) -> (building_sim, env)
        env_factory : callable
            Function that creates environment from building_sim.
        """
        self.building_factory = building_factory
        self.env_factory = env_factory

    def sweep_single_param(
        self,
        param_name: str,
        param_values: List[float],
        base_params: Dict[str, Any],
        n_steps: int = 50,
    ) -> Dict[str, Any]:
        """
        Sweep a single parameter and record final zone temperatures.

        Parameters
        ----------
        param_name : str
            Name of parameter to sweep (e.g., "outdoor_high_temp", "boiler_setpoint").
        param_values : list
            Values to test.
        base_params : dict
            Base parameter configuration (will be modified).
        n_steps : int
            Number of simulation steps per configuration.

        Returns
        -------
        dict
            {'param_values': [...], 'final_temps': [...], 'mean_temps': [...]}
        """
        final_temps_list = []
        mean_temps_list = []

        for val in param_values:
            params = base_params.copy()
            params[param_name] = val

            try:
                building_sim, env = self.building_factory(params)
                obs, _ = env.reset()

                # run neutral actions
                for _ in range(n_steps):
                    action = np.zeros(env.action_space.shape[0], dtype=np.float32)
                    obs, _, _, _, _ = env.step(action)

                final_temps_list.append(obs.copy())
                mean_temps_list.append(np.mean(obs))
            except Exception as e:
                print(f"  Error with {param_name}={val}: {e}")
                final_temps_list.append(np.full_like(obs, np.nan))
                mean_temps_list.append(np.nan)

        return {
            "param_name": param_name,
            "param_values": np.array(param_values),
            "final_temps": np.array(final_temps_list),
            "mean_temps": np.array(mean_temps_list),
        }

    def sweep_multiple_params(
        self,
        param_sweeps: Dict[str, List[float]],
        base_params: Dict[str, Any],
        n_steps: int = 50,
    ) -> Dict[str, Any]:
        """
        Sweep multiple parameters individually and return results.

        Parameters
        ----------
        param_sweeps : dict
            {'param_name': [val1, val2, ...], ...}
        base_params : dict
            Base configuration.
        n_steps : int
            Simulation steps per config.

        Returns
        -------
        dict
            Keys are param names, values are sweep results.
        """
        results = {}
        for param_name, param_values in param_sweeps.items():
            print(f"Sweeping {param_name}...")
            results[param_name] = self.sweep_single_param(
                param_name, param_values, base_params, n_steps
            )
        return results

    def plot_single_param_sweep(
        self,
        sweep_result: Dict[str, Any],
        figsize: Tuple[int, int] = (10, 6),
    ) -> plt.Figure:
        """Plot temperature response to single parameter sweep."""
        param_name = sweep_result["param_name"]
        param_values = sweep_result["param_values"]
        mean_temps = sweep_result["mean_temps"]

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(param_values, mean_temps, marker="o", linewidth=2, markersize=8, color="steelblue")
        ax.set_xlabel(f"{param_name}", fontsize=12)
        ax.set_ylabel("Mean Zone Temperature [K]", fontsize=12)
        ax.set_title(f"Temperature Sensitivity to {param_name}", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def plot_all_param_sweeps(
        self,
        sweep_results: Dict[str, Dict[str, Any]],
        figsize: Tuple[int, int] = (16, 12),
    ) -> plt.Figure:
        """Plot all parameter sweeps in a grid."""
        param_names = list(sweep_results.keys())
        n_params = len(param_names)
        n_cols = 3
        n_rows = (n_params + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_params > 1 else [axes]

        for idx, param_name in enumerate(param_names):
            result = sweep_results[param_name]
            param_values = result["param_values"]
            mean_temps = result["mean_temps"]

            axes[idx].plot(param_values, mean_temps, marker="o", linewidth=2, markersize=6, color="steelblue")
            axes[idx].set_xlabel(param_name, fontsize=10)
            axes[idx].set_ylabel("Mean Temp [K]", fontsize=10)
            axes[idx].set_title(f"Sensitivity to {param_name}", fontsize=11, fontweight="bold")
            axes[idx].grid(True, alpha=0.3)

        # hide unused axes
        for idx in range(n_params, len(axes)):
            axes[idx].remove()

        fig.suptitle("Parameter Sensitivity Analysis", fontsize=16, fontweight="bold")
        fig.tight_layout()
        return fig

    def plot_temp_heatmap_dual_params(
        self,
        param_sweeps: Dict[str, List[float]],
        base_params: Dict[str, Any],
        param1_name: str,
        param2_name: str,
        n_steps: int = 50,
        figsize: Tuple[int, int] = (10, 8),
    ) -> plt.Figure:
        """
        Create 2D heatmap of temperature response to two parameters.

        Parameters
        ----------
        param_sweeps : dict
            {'param1': [vals], 'param2': [vals]}
        base_params : dict
            Base configuration.
        param1_name : str
            Parameter for X-axis.
        param2_name : str
            Parameter for Y-axis.
        n_steps : int
            Simulation steps.
        figsize : tuple
            Figure size.

        Returns
        -------
        plt.Figure
        """
        param1_vals = param_sweeps[param1_name]
        param2_vals = param_sweeps[param2_name]

        temps_grid = np.zeros((len(param2_vals), len(param1_vals)))

        for i, p2_val in enumerate(param2_vals):
            for j, p1_val in enumerate(param1_vals):
                params = base_params.copy()
                params[param1_name] = p1_val
                params[param2_name] = p2_val

                try:
                    building_sim, env = self.building_factory(params)
                    obs, _ = env.reset()
                    for _ in range(n_steps):
                        action = np.zeros(env.action_space.shape[0], dtype=np.float32)
                        obs, _, _, _, _ = env.step(action)
                    temps_grid[i, j] = np.mean(obs)
                except Exception as e:
                    print(f"  Error with {param1_name}={p1_val}, {param2_name}={p2_val}: {e}")
                    temps_grid[i, j] = np.nan

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(temps_grid, cmap="RdYlBu_r", aspect="auto", origin="lower")

        ax.set_xticks(np.arange(len(param1_vals)))
        ax.set_yticks(np.arange(len(param2_vals)))
        ax.set_xticklabels([f"{v:.2f}" for v in param1_vals], rotation=45)
        ax.set_yticklabels([f"{v:.2f}" for v in param2_vals])

        ax.set_xlabel(param1_name, fontsize=12)
        ax.set_ylabel(param2_name, fontsize=12)
        ax.set_title(f"Temperature Sensitivity: {param1_name} vs {param2_name}", fontsize=14, fontweight="bold")

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Mean Zone Temperature [K]", fontsize=11)

        fig.tight_layout()
        return fig
