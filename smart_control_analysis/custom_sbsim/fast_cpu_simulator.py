import numpy as np
from smart_control.simulator.simulator_flexible_floor_plan import SimulatorFlexibleGeometries

from numba import njit



# ---------------------------------------------------------------------------
# Numba-JIT Gauss-Seidel kernel (identical ordering to the original Python loop)
# ---------------------------------------------------------------------------

def _make_gs_kernel():
    """Return a JIT-compiled Gauss-Seidel sweep function with Numba"""


    @njit(cache=True)
    def _gs_sweep(T, last_temp, input_q,
                    interior, corner, edge,
                    w_up, w_down, w_left, w_right,
                    c_self, c_amb_h, c_q,
                    denom_base, denom_h,
                    ambient_temperature, convection_coefficient,
                    nrows, ncols):
        """One full Gauss-Seidel sweep in row-major order (matches original).

        Updates T in-place.  Returns max |delta| across all cells.
        """
        max_delta = 0.0
        h = convection_coefficient
        for x in range(nrows):
            for y in range(ncols):
                if not (interior[x, y] or corner[x, y] or edge[x, y]):
                    # exterior — fix to ambient
                    old = T[x, y]
                    T[x, y] = ambient_temperature
                    d = abs(T[x, y] - old)
                    if d > max_delta:
                        max_delta = d
                    continue

                denom = denom_base[x, y] + h * denom_h[x, y]

                n_nbr = (w_up[x, y]    * T[x - 1, y] +
                            w_down[x, y]  * T[x + 1, y] +
                            w_left[x, y]  * T[x, y - 1] +
                            w_right[x, y] * T[x, y + 1])

                n_amb  = h * c_amb_h[x, y] * ambient_temperature
                n_self = c_self[x, y] * last_temp[x, y]
                n_q    = c_q[x, y] * input_q[x, y]

                new_val = (n_nbr + n_amb + n_self + n_q) / denom
                d = abs(new_val - T[x, y])
                if d > max_delta:
                    max_delta = d
                T[x, y] = new_val

        return max_delta

    return _gs_sweep


_gs_sweep = _make_gs_kernel()


class FastCPUSimulator(SimulatorFlexibleGeometries):
    """SimulatorFlexibleGeometries variant for direct RL control.

    Changes vs base class:
    - setup_step_sim() is a no-op (no thermostat write-back to VAVs)
    - Video logging disabled
    - finite_differences_timestep() uses a precomputed-coefficient Gauss-Seidel
      sweep in identical row-major order to the original, JIT-compiled with
      Numba when available (falls back to plain Python otherwise)
    """

    def __init__(self, *args, disable_video: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self._disable_video = bool(disable_video)

        if self._disable_video:
            lap = getattr(self, "_log_and_plotter", None)
            if lap is not None and hasattr(lap, "log"):
                lap.log = lambda *a, **k: None

        self._fdm_coeffs = None

    def setup_step_sim(self) -> None:
        return None

    def get_video(self, *args, **kwargs):
        if self._disable_video:
            return None
        return super().get_video(*args, **kwargs)

    def _precompute_fdm_coeffs(self) -> None:
        """Precompute static per-cell coefficient arrays.

        Called once on the first step.  Depends only on grid structure and
        material properties, which are fixed for the lifetime of the simulator.
        """
        building = self.building
        H, W = building.temp.shape
        dt = self._time_step_sec
        dx = building.cv_size_cm / 100.0
        z  = building.floor_height_cm / 100.0

        k       = building.conductivity   # (H, W)
        density = building.density        # (H, W)
        Cp      = building.heat_capacity  # (H, W)
        nbrs    = building.neighbors

        n_nbr = np.array([[len(nbrs[x][y]) for y in range(W)] for x in range(H)])

        interior = (n_nbr >= 4)
        corner   = (n_nbr == 2)
        edge     = (n_nbr == 3)

        t0_int = dx**2 * density * Cp / (dt * k)           # interior
        t0_ec  = density * dx**2 * Cp / (2.0 * dt)         # edge / corner

        w_up    = np.zeros((H, W))
        w_down  = np.zeros((H, W))
        w_left  = np.zeros((H, W))
        w_right = np.zeros((H, W))
        c_self     = np.zeros((H, W))
        c_amb_h    = np.zeros((H, W))
        c_q        = np.zeros((H, W))
        denom_base = np.ones((H, W))
        denom_h    = np.zeros((H, W))

        for x in range(H):
            for y in range(W):
                nb = nbrs[x][y]
                n  = len(nb)

                if n <= 1:
                    continue  # exterior: handled separately in the sweep

                def _set_dir(nx, ny, w, _x=x, _y=y):
                    if   nx == _x - 1: w_up[_x, _y]    = w
                    elif nx == _x + 1: w_down[_x, _y]  = w
                    elif ny == _y - 1: w_left[_x, _y]  = w
                    elif ny == _y + 1: w_right[_x, _y] = w

                if n == 2:  # corner
                    t0 = t0_ec[x, y]; ki = k[x, y]
                    for (nx, ny) in nb:
                        _set_dir(nx, ny, ki)
                    c_self[x, y]     = t0
                    c_amb_h[x, y]    = 2.0 * dx
                    denom_base[x, y] = 2.0 * ki + t0
                    denom_h[x, y]    = 2.0 * dx

                elif n == 3:  # edge
                    t0 = t0_ec[x, y]; ki = k[x, y]
                    for (nx, ny) in nb:
                        ef = 0.5 if len(nbrs[nx][ny]) < 4 else 1.0
                        _set_dir(nx, ny, ki * ef)
                    c_self[x, y]     = t0
                    c_amb_h[x, y]    = dx
                    denom_base[x, y] = 2.0 * ki + t0
                    denom_h[x, y]    = dx

                else:  # interior
                    t0 = t0_int[x, y]
                    w_up[x, y] = w_down[x, y] = w_left[x, y] = w_right[x, y] = 1.0
                    c_self[x, y]     = t0
                    c_q[x, y]        = 1.0 / (k[x, y] * z)
                    denom_base[x, y] = 4.0 + t0

        # Numba requires contiguous C arrays
        def _c(a): return np.ascontiguousarray(a)

        self._fdm_coeffs = dict(
            interior=_c(interior), corner=_c(corner), edge=_c(edge),
            w_up=_c(w_up), w_down=_c(w_down),
            w_left=_c(w_left), w_right=_c(w_right),
            c_self=_c(c_self), c_amb_h=_c(c_amb_h), c_q=_c(c_q),
            denom_base=_c(denom_base), denom_h=_c(denom_h),
            nrows=H, ncols=W,
        )

    def finite_differences_timestep(
        self, *, ambient_temperature: float, convection_coefficient: float
    ) -> bool:
        if self._fdm_coeffs is None:
            self._precompute_fdm_coeffs()

        c = self._fdm_coeffs

        # last_temp is the temperature at the START of this timestep (constant
        # across iterations — same as self.building.temp[x][y] in the original)
        last_temp = np.ascontiguousarray(self.building.temp, dtype=float)
        input_q   = np.ascontiguousarray(self.building.input_q, dtype=float)
        T         = last_temp.copy()

        converged = False
        for iteration in range(self._iteration_limit):
            max_delta = _gs_sweep(
                T, last_temp, input_q,
                c['interior'], c['corner'], c['edge'],
                c['w_up'], c['w_down'], c['w_left'], c['w_right'],
                c['c_self'], c['c_amb_h'], c['c_q'],
                c['denom_base'], c['denom_h'],
                float(ambient_temperature), float(convection_coefficient),
                c['nrows'], c['ncols'],
            )

            if iteration + 1 == self._iteration_warning:
                from absl import logging
                logging.warning(
                    'Step %d, not converged in %d steps, max_delta = %3.3f',
                    iteration, self._iteration_warning, max_delta,
                )

            if max_delta <= self._convergence_threshold:
                converged = True
                break

        self.building.temp = T
        return converged
