"""График и таблица по прогонам для docs/МЕТРИКИ.md (прогоны val, модель только по train):
без коррекции по GNSS после выставки и с пачками GNSS (5 с каждые 120 с)."""
import csv
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import evaluate as E
import refdata
from tram_odometry.estimator import Estimator
from tram_odometry.route import load_queues, load_routes, load_stops, load_traction, load_vehicles

DOCS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'docs')
BLUE, INK, INK2, SURF = '#2a78d6', '#0b0b0b', '#52514e', '#fcfcfb'
BASE = dict(stops_enabled=False, use_model=False, wheel_scale=1.0)


def trace(bag, params, routes, stops, traction, gnss_burst=False):
    d = refdata.load(bag)
    est = Estimator(routes, params, stops, traction, load_queues())
    T, K, V = E.events(d)
    llh = {E.MFIX: d['master_llh'], E.RFIX: d['rover_llh']}
    ot, o = [], []
    for t, k, v in zip(T, K, V):
        if k == E.FRONT:
            est.on_wheel(t, 'front', v)
        elif k == E.REAR:
            est.on_wheel(t, 'rear', v)
        elif k == E.CMD:
            est.on_cmd(t, v)
        else:
            # после выставки GNSS отдаётся только в режиме пачек (5 с каждые 120 с)
            if est.aligning or (gnss_burst and (t - T[0]) % 120.0 <= 5.0):
                la, lo, _ = llh[k][int(v)]
                est.on_gnss(t, 'master' if k == E.MFIX else 'rover', la, lo)
            continue
        st = est.state(t)
        if st is not None:
            ot.append(t)
            o.append(st)
    ot, o = np.array(ot), np.array(o)
    rt, rp = d['ref_t'], d['ref_xyz']
    sel = rt >= ot[0]
    j = E.nearest(ot, rt[sel])
    ok = np.abs(ot[j] - rt[sel]) < 0.05
    P, Q, yaw = o[j[ok], 1:4], rp[sel][ok], o[j[ok], 4]
    dxy = P[:, :2] - Q[:, :2]
    at = dxy[:, 0] * np.cos(yaw) + dxy[:, 1] * np.sin(yaw)
    dist = np.r_[0, np.cumsum(np.abs(np.diff(o[:, 5])))][j[ok]]
    return dist, at


def main():
    routes, stops, traction, veh = load_routes(), load_stops(), load_traction(), load_vehicles()
    _, val = refdata.split()
    val = [b for b in val if refdata.load(b)['master_t'][0] - refdata.load(b)['front_t'][0] <= E.LATE_INIT]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True, facecolor=SURF)
    panels = [('Только колёса', BASE, False),
              ('Модель, без коррекции по GNSS', {}, False),
              ('Модель, коррекция по пачкам GNSS', {}, True)]
    for ax, (title, params, burst) in zip(axes, panels):
        for b in val:
            p = dict(params)
            p.setdefault('wheel_scale', veh.get(b.split('_')[0], 1.0))
            dist, at = trace(b, p, routes, stops, traction, burst)
            ax.plot(dist / 1000, at, color=BLUE, lw=1.0, alpha=0.45)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_facecolor(SURF)
        ax.set_title(title, color=INK, fontsize=11, loc='left')
        ax.set_xlabel('пройдено, км', color=INK2)
        ax.grid(axis='y', color='#e4e3df', lw=0.6)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=INK2)
    axes[0].set_ylabel('ошибка вдоль пути, м', color=INK2)
    axes[0].set_ylim(-40, 40)
    fig.suptitle('Ошибка вдоль пути, %d прогонов val (линия = прогон; пачки GNSS: 5 с каждые 2 мин)' % len(val),
                 color=INK, fontsize=12, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, 'figures', 'along_track_error.png'), dpi=130, facecolor=SURF)

    # таблица по прогонам: без коррекции по GNSS и с пачками GNSS
    rows = []
    for b in val:
        p = {'wheel_scale': veh.get(b.split('_')[0], 1.0)}
        E.GNSS_MODE = 'start'
        r, w = E.run_bag(b, routes, p, stops, traction)
        E.GNSS_MODE = 'burst'
        g, w = E.run_bag(b, routes, p, stops, traction)
        rows.append([b, '%.3f' % r['v_rmse'], '%.3f' % r['v_bias'], '%.2f' % r['p_rmse'], '%.2f' % r['p_max'],
                     '%.2f' % r['at_rmse'], '%.2f' % r['ct_rmse'], '%.3f' % r['drift_pct'], r['n_corr'],
                     '%.2f' % g['p_rmse'], '%.2f' % g['at_rmse']])
    with open(os.path.join(DOCS, 'results_val.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['bag', 'v_rmse_ms', 'v_bias_ms', 'p_rmse_m', 'p_max_m', 'along_rmse_m', 'cross_rmse_m',
                    'final_drift_pct', 'stop_corrections', 'p_rmse_gnss_m', 'along_rmse_gnss_m'])
        w.writerows(rows)
    print('saved')


if __name__ == '__main__':
    main()
