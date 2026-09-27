"""ROS отдаёт топики в порядке записи; штампы разных топиков перемежаются."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tram_odometry.estimator import Estimator  # noqa: E402
from tram_odometry.route import Route, load_traction  # noqa: E402


def test_wheels_slightly_older_than_controller_are_still_used():
    r = Route('line', [[0, 0, 0], [5000, 0, 0]])
    est = Estimator([r], {'stops_enabled': False}, traction=load_traction())
    est.init_from_xy(0.0, 0.0, 0.0, 0.0)
    t = 0.0
    while t < 60.0:
        est.on_cmd(t + 0.15, 0)          # штамп контроллера на 150 мс впереди колёс
        est.on_wheel(t, 'front', 36.0)
        est.on_wheel(t, 'rear', 36.0)
        t += 0.1
    v, x = est.state(t)[:2]
    assert abs(v - 10.0) < 0.1
    assert abs(x - 600.0) < 5.0


def test_same_stream_going_back_in_time_is_ignored():
    r = Route('line', [[0, 0, 0], [5000, 0, 0]])
    est = Estimator([r], {'stops_enabled': False}, traction=load_traction())
    est.init_from_xy(0.0, 0.0, 0.0, 0.0)
    est.on_wheel(1.0, 'front', 36.0)
    est.on_wheel(0.5, 'front', 0.0)      # старше последнего показания передней тележки
    assert est.wheel['front'][0] == 1.0
