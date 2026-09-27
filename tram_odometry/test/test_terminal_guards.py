"""Защиты станций на конечных и кэш проекций."""
import os
import sys
import numpy as np
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tram_odometry.estimator import Estimator
from tram_odometry.route import Route


def test_terminal_does_not_snap_uncertain_stationary_start():
    route = Route('line', [[0,0,0],[1000,0,0]])
    stops = [dict(x=55,y=0,n_bags=5,mad=.1,direction='terminal')]
    est = Estimator([route],dict(anchor_scale=True,anchor_use_init=False),stops)
    est.init_from_xy(0,0,0)
    est.var0=900
    est.wheel={'front':(10,0),'rear':(10,0)}
    est.stand_t0=0
    est._check_stop(10)
    assert est.s == 0
    assert not est.scale_anchors
    assert est.n_corrections == 0


def test_terminal_update_never_becomes_scale_anchor():
    route=Route('line',[[0,0,0],[1000,0,0]])
    stops=[dict(x=100,y=0,n_bags=5,mad=.1,direction='terminal')]
    est=Estimator([route],dict(anchor_scale=True,anchor_use_init=False),stops)
    est.init_from_xy(0,0,0)
    est.s,est.odo=98.,98.
    est.wheel={'front':(10,0),'rear':(10,0)}
    est.stand_t0=0
    est._check_stop(10)
    assert est.n_corrections == 1
    assert 98 < est.s < 100
    assert not est.scale_anchors


def test_manual_alignment_anchor_and_deadband():
    route=Route('line',[[0,0,0],[10000,0,0]])
    est=Estimator([route],dict(anchor_scale=True,anchor_init_var=25,anchor_deadband=.002))
    est.init_from_xy(0,100,0)
    assert est.scale_anchors == [(0.,100.,25)]
    est.odo=1000
    est._anchor_scale_update(1099)
    assert est.scale == 1.


def test_cached_projection_matches_direct_formula():
    rng=np.random.default_rng(11)
    xyz=np.cumsum(rng.normal(size=(30,3)),axis=0)
    route=Route('random',xyz)
    for x,y in rng.normal(size=(30,2))*5:
        a=xyz[:-1,:2]; d=xyz[1:,:2]-a
        l2=np.maximum((d*d).sum(1),1e-12)
        u=np.clip(((np.array([x,y])-a)*d).sum(1)/l2,0,1)
        distance=np.linalg.norm(a+u[:,None]*d-[x,y],axis=1)
        i=int(distance.argmin())
        s,xt,yaw=route.project(x,y)
        assert s == pytest.approx(route.s[i]+u[i]*np.sqrt(l2[i]))
        assert xt == pytest.approx(distance[i])
        assert yaw == pytest.approx(np.arctan2(d[i,1],d[i,0]))
