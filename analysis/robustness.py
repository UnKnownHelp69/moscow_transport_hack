"""Проверка оценщика с искусственными сбоями на прогонах val (критерий 3: юз и аномалии).

Сценарии (меняются только потоки колёс, эталон GNSS не трогается):
  stuck_rear   задняя тележка показывает 0 на ходу 30 с (x3 за рейс)
  spikes       1 % показаний передней тележки заменены выбросами +-20..60 км/ч
  dropout      обе тележки молчат 3 с каждые 60 с
  wheel_spin   обе тележки +15 % при тяге на 4 с (x10 за рейс)
  wheel_slide  обе тележки -25 % при торможении на 3 с (x10 за рейс)
  nan          5 % показаний NaN или отрицательный мусор
Запуск: python robustness.py [key=value ...]
"""
import json
import sys

import numpy as np

import evaluate as E  # noqa: задаёт TRAM_ODOM_DATA
import refdata
from tram_odometry.route import load_routes, load_stops, load_traction, load_vehicles

rng = np.random.default_rng(1)


def inject(d, kind):
    d = {k: v.copy() for k, v in d.items()}
    ft, fv, rt, rv = d['front_t'], d['front_v'], d['rear_t'], d['rear_v']
    cmd = np.interp(ft, d['cmd_t'], d['cmd_v'])
    moving = np.where(fv > 20)[0]
    if kind == 'stuck_rear':
        for c in rng.choice(moving, 3, replace=False) if len(moving) > 3 else []:
            m = (rt >= ft[c]) & (rt < ft[c] + 30)
            rv[m] = 0.0
    elif kind == 'spikes':
        m = rng.random(len(fv)) < 0.01
        fv[m] = np.clip(fv[m] + rng.choice([-1, 1], m.sum()) * rng.uniform(20, 60, m.sum()), 0, None)
    elif kind == 'dropout':
        keep_f = ((ft - ft[0]) % 60) > 3
        keep_r = ((rt - ft[0]) % 60) > 3
        d['front_t'], d['front_v'] = ft[keep_f], fv[keep_f]
        d['rear_t'], d['rear_v'] = rt[keep_r], rv[keep_r]
        return d
    elif kind in ('wheel_spin', 'wheel_slide'):
        cand = np.where((cmd >= 5) & (fv > 10))[0] if kind == 'wheel_spin' else np.where((cmd <= -5) & (fv > 15))[0]
        f = 1.15 if kind == 'wheel_spin' else 0.75
        dur = 4.0 if kind == 'wheel_spin' else 3.0
        for c in rng.choice(cand, min(10, len(cand)), replace=False) if len(cand) else []:
            t0 = ft[c]
            for tt, vv in ((ft, fv), (rt, rv)):
                m = (tt >= t0) & (tt < t0 + dur)
                ramp = np.clip((tt[m] - t0) / 0.5, 0, 1)
                vv[m] *= 1 + (f - 1) * ramp
    elif kind == 'nan':
        m = rng.random(len(fv)) < 0.05
        fv[m] = np.where(rng.random(m.sum()) < 0.5, np.nan, -30.0)
    return d


def main():
    params = {}
    for a in sys.argv[1:]:
        k, v = a.split('=')
        params[k] = json.loads(v)
    routes, stops, traction = load_routes(), load_stops(), load_traction()
    _, val = refdata.split()
    orig = refdata.load
    print('%-12s %-8s %8s %8s %8s %8s' % ('scenario', 'model', 'v_rmse', 'v_max', 'p_rmse', 'p_max'))
    for kind in ['none', 'stuck_rear', 'spikes', 'dropout', 'wheel_spin', 'wheel_slide', 'nan']:
        for use_model in (False, True):
            p = dict(params, use_model=use_model)
            veh = load_vehicles()
            rows = []
            for b in val[:16]:
                d0 = orig(b)
                if d0['master_t'][0] - d0['front_t'][0] > E.LATE_INIT:
                    continue
                dd = d0 if kind == 'none' else inject(d0, kind)
                refdata.load = lambda _b, dd=dd: dd
                try:
                    r = E.run_bag(b, routes, dict(p, wheel_scale=veh.get(b.split('_')[0], 1.0)), stops, traction)
                finally:
                    refdata.load = orig
                if r:
                    rows.append(r[0])
            vr = np.mean([m['v_rmse'] for m in rows])
            vm = np.mean([m['v_max'] for m in rows])
            pr = np.mean([m['p_rmse'] for m in rows])
            pm = np.mean([m['p_max'] for m in rows])
            print('%-12s %-8s %8.3f %8.2f %8.2f %8.2f' % (kind, use_model, vr, vm, pr, pm))


if __name__ == '__main__':
    main()
