"""Юнит-тесты ядра без ROS (запуск: python3 -m pytest test/)."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tram_odometry.estimator import Estimator  # noqa: E402
from tram_odometry.geo import latlon_to_mgrs_local  # noqa: E402
from tram_odometry.route import Route, load_routes, load_stops, load_traction  # noqa: E402


def test_mgrs_projection_matches_pyproj_reference():
    # эталон из pyproj (EPSG:32637) минус (300000, 6100000)
    x, y = latlon_to_mgrs_local(55.809898, 37.461398)
    assert abs(x - 103574.5) < 0.1 and abs(y - 85993.1) < 0.1


def test_route_point_and_project():
    r = Route('line', [[0, 0, 0], [100, 0, 1]])
    x, y, z, yaw = r.point(25.0)
    assert (x, y) == (25.0, 0.0) and abs(z - 0.25) < 1e-9 and yaw == 0.0
    s, xt, _ = r.project(40.0, 3.0)
    assert abs(s - 40.0) < 1e-9 and abs(xt - 3.0) < 1e-9


def _drive(est, v_ms, t0=0.0, dur=100.0, notch=0):
    t = t0
    while t < t0 + dur:
        est.on_cmd(t, notch)
        if int(round(t * 20)) % 2 == 0:
            est.on_wheel(t, 'front', v_ms * 3.6)
            est.on_wheel(t + 0.001, 'rear', v_ms * 3.6)
        t += 0.05
    return t


def test_constant_speed_integration_along_route():
    r = Route('line', [[0, 0, 0], [5000, 0, 0]])
    est = Estimator([r], {'stops_enabled': False}, traction=load_traction())
    est.init_from_xy(0.0, 100.0, 0.0, 0.0)
    t = _drive(est, 10.0, dur=60.0)
    v, x, y, z, yaw, s = est.state(t)
    assert abs(v - 10.0) < 0.05
    assert abs(x - (100.0 + 600.0)) < 2.0 and abs(y) < 1e-6


def test_garbage_inputs_do_not_crash_or_poison_state():
    r = Route('line', [[0, 0, 0], [5000, 0, 0]])
    est = Estimator([r], {'stops_enabled': False}, traction=load_traction())
    est.init_from_xy(0.0, 0.0, 0.0, 0.0)
    _drive(est, 8.0, dur=10.0)
    for bad in (float('nan'), float('inf'), -50.0, 1e6):
        est.on_wheel(10.0, 'front', bad)
    est.on_gnss(10.0, 'master', float('nan'), 0.0)
    v = est.state(10.05)[0]
    assert math.isfinite(v) and abs(v - 8.0) < 0.3


def test_stuck_bogie_is_rejected():
    r = Route('line', [[0, 0, 0], [5000, 0, 0]])
    est = Estimator([r], {'stops_enabled': False}, traction=load_traction())
    est.init_from_xy(0.0, 0.0, 0.0, 0.0)
    t = _drive(est, 10.0, dur=10.0)
    for k in range(100):            # задняя тележка 10 с показывает 0, передняя исправна
        tt = t + 0.1 * k
        est.on_cmd(tt, 0)
        est.on_wheel(tt, 'front', 36.0)
        est.on_wheel(tt + 0.001, 'rear', 0.0)
    assert abs(est.state(t + 10.0)[0] - 10.0) < 0.3


def test_shipped_model_data_loads():
    routes = load_routes()
    assert len(routes) >= 2 and all(r.length > 4000 for r in routes)
    assert len(load_stops()) > 10
    tr = load_traction()
    assert len(tr['accel']) == 31 and 0.0 <= tr['lag'] < 2.0
    assert np.isfinite(np.array(tr['accel'])).all()
