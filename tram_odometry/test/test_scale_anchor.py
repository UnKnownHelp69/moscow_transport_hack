"""Регрессия масштаба по станциям (синтетические маршруты, без bag)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tram_odometry.estimator import Estimator
from tram_odometry.route import Route


def make_estimator(**overrides):
    route = Route('line', [[0.0, 0.0, 0.0], [30000.0, 0.0, 0.0]])
    params = dict(use_model=False, anchor_scale=True, anchor_use_init=False,
                  recover_max=0.0, stop_repeat_dist=100.0, stop_dwell=5.0)
    params.update(overrides)
    est = Estimator([route], params)
    est.init_from_xy(0.0, 100.0, 0.0)
    return est


def stopped(est, s, odo, t=10.0):
    """Одно событие стоянки после уже прошедшего времени ожидания."""
    est.s, est.odo, est.v = float(s), float(odo), 0.0
    est.wheel = {'front': (t, 0.0), 'rear': (t, 0.0)}
    est.stand_t0, est.stand_used = t - 6.0, False
    est._check_stop(t)


@pytest.mark.parametrize('scale', [0.988, 1.006])
def test_affine_station_fit_recovers_both_scale_directions(scale):
    est = make_estimator()
    # большой начальный сдвиг не должен выглядеть как ошибка масштаба
    for odo in (100.0, 600.0, 1100.0, 2100.0):
        est.odo = odo
        est._anchor_scale_update(340.0 + scale * odo)
    assert est.scale == pytest.approx(scale, abs=0.0002)


def test_single_anchor_preserves_prior_and_history_is_bounded():
    est = make_estimator()
    est.odo = 100.0
    est._anchor_scale_update(800.0)
    assert est.scale == 1.0
    for k in range(1, 50):
        est.odo = 100.0 + 100.0 * k
        est._anchor_scale_update(800.0 + 99.0 * k)
    assert len(est.scale_anchors) == 32
    assert 0.989 < est.scale < 0.991


def test_same_station_not_counted_as_independent_scale_anchor():
    est = make_estimator()
    est.odo = 100.0
    est._anchor_scale_update(200.0)
    est.odo = 107.0
    est._anchor_scale_update(200.0)
    assert len(est.scale_anchors) == 1
    assert est.scale == 1.0


def test_scale_bound_protects_against_impossible_anchor_slope():
    est = make_estimator(scale_limit=0.02)
    est.odo = 0.0
    est._anchor_scale_update(100.0)
    est.odo = 1000.0
    est._anchor_scale_update(2100.0)
    assert est.scale == pytest.approx(1.02)


def test_queue_soft_correction_cannot_reset_station_scale_baseline():
    est = make_estimator()
    est.stop_s['line'] = np.array([200.0, 1200.0])
    est.queue_s['line'] = [(700.0, 2.0)]
    stopped(est, 200.0, 100.0)
    first_anchor = list(est.scale_anchors)
    est.var0, est.d_since = 1.0, 500.0
    stopped(est, 701.0, 600.0, 20.0)
    assert est.n_queue == 1
    assert est.d_since == 0.0  # фильтр очередей сам сбрасывает свою неопределённость
    assert est.scale_anchors == first_anchor
    assert est.scale == 1.0
    stopped(est, 1201.0, 1100.0, 30.0)
    assert est.n_corrections == 2
    assert len(est.scale_anchors) == 2
    assert est.scale == pytest.approx(1.0)


def test_forward_creep_does_not_reapply_same_station_position():
    est = make_estimator()
    est.stop_s['line'] = np.array([200.0, 600.0])
    stopped(est, 199.0, 100.0)
    n = est.n_corrections
    stopped(est, 207.0, 107.0, 20.0)
    assert est.n_corrections == n
    assert est.s == 207.0
    assert len(est.scale_anchors) == 1
    stopped(est, 599.0, 500.0, 30.0)
    assert est.n_corrections == n + 1
    assert len(est.scale_anchors) == 2


def test_disable_recovery_rejects_two_similar_queue_offsets():
    est = make_estimator()
    est.stop_s['line'] = np.array([200.0, 600.0])
    stopped(est, 161.0, 100.0)
    stopped(est, 561.0, 500.0, 20.0)
    assert est.n_corrections == 0
    assert est.n_recoveries == 0
    assert not est.scale_anchors
    assert est.scale == 1.0


def test_scale_adaptation_is_opt_in():
    est = make_estimator(anchor_scale=False)
    est.stop_s['line'] = np.array([200.0, 600.0])
    stopped(est, 199.0, 100.0)
    stopped(est, 598.0, 500.0, 20.0)
    assert est.n_corrections == 2
    assert not est.scale_anchors
    assert est.scale == 1.0


def test_post_initialization_gnss_does_not_change_scale_or_station_history():
    est = make_estimator()
    est.gnss_t0 = 0.0
    for odo, cp in ((100.0, 200.0), (1100.0, 1190.0)):
        est.odo = odo
        est._anchor_scale_update(cp)
    before = (est.scale, est.s, est.odo, list(est.scale_anchors))
    est.on_gnss(100.0, 'master', 55.81, 37.46)
    est.on_gnss(101.0, 'rover', 55.82, 37.47)
    assert (est.scale, est.s, est.odo, est.scale_anchors) == before
