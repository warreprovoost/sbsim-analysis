import numpy as np

from smart_control_analysis.building_factory import building_factory, get_base_params
from smart_control_analysis.custom_sbsim.direct_vav_tf_simulator import DirectVavTFSimulator



def run_vav_check() -> dict:
    params = get_base_params()
    sim, env = building_factory(params)

    assert isinstance(sim, DirectVavTFSimulator), f"Unexpected simulator: {type(sim)}"

    reset_out = env.reset()
    _obs, _info = reset_out if isinstance(reset_out, tuple) else (reset_out, {})

    n_zones = env.n_zones
    a = np.zeros(env.action_space.shape[0], dtype=np.float32)
    a[3:3 + n_zones] = -0.4
    a[3 + n_zones:3 + 2 * n_zones] = -0.26

    env.step(a)

    expected_d = 0.5 * (a[3] + 1.0)
    expected_r = 0.5 * (a[3 + n_zones] + 1.0)

    for zid in env.zone_ids:
        v = env.vavs[zid]
        assert abs(v.damper_setting - expected_d) < 1e-3, "Damper overwritten"
        assert abs(v.reheat_valve_setting - expected_r) < 1e-3, "Reheat overwritten"

    # near-zero damper safety
    a2 = np.zeros_like(a)
    a2[3:3 + n_zones] = -1.0
    a2[3 + n_zones:3 + 2 * n_zones] = 0.0
    env.step(a2)

    dampers = [float(env.vavs[z].damper_setting) for z in env.zone_ids]
    return {
        "ok": True,
        "simulator": type(sim).__name__,
        "n_zones": int(n_zones),
        "dampers_after_close_cmd": dampers,
    }
