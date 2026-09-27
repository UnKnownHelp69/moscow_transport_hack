"""Офлайн-оценка: причинное проигрывание bag в порядке header.stamp.

GNSS отдаётся оценщику только первые `init_window` с после первого валидного фикса, дальше
служит эталоном. Метрики (после выставки):
  скорость: RMSE / MAE / смещение относительно |master/vel|;
  положение: 3D-ошибка относительно эталона base_link, вдоль пути (со знаком) и поперёк,
             дрейф = ошибка в последней точке / пройденный путь (%).
Запуск: python evaluate.py [val|train|all] [key=value ...]   (переопределение параметров)
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GNSS_MODE = os.environ.get('TRAM_EVAL_GNSS', 'start')   # start | burst | sparse | all
SPARSE_PERIOD = float(os.environ.get('TRAM_EVAL_GNSS_PERIOD', '30'))   # с, для sparse
# по умолчанию - поставляемая модель (только train и архив OSM)
os.environ.setdefault('TRAM_ODOM_DATA', os.path.join(HERE, '..', 'tram_odometry', 'tram_odometry', 'data'))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'tram_odometry'))
import refdata
from tram_odometry.estimator import Estimator
from tram_odometry.route import load_queues, load_routes, load_stops, load_traction, load_vehicles

FRONT, REAR, CMD, MFIX, RFIX = range(5)
LATE_INIT = 60.0   # с: bag, где GNSS появляется позже, не похож на тестовый


def events(d):
    parts = [(d['front_t'], FRONT, d['front_v']), (d['rear_t'], REAR, d['rear_v']),
             (d['cmd_t'], CMD, d['cmd_v'])]
    for code, key in ((MFIX, 'master'), (RFIX, 'rover')):
        t = d[key + '_t']
        if len(t):
            parts.append((t, code, np.arange(len(t), dtype=float)))
    t = np.concatenate([p[0] for p in parts])
    k = np.concatenate([np.full(len(p[0]), p[1]) for p in parts])
    v = np.concatenate([p[2] for p in parts])
    o = np.argsort(t, kind='stable')
    return t[o], k[o], v[o]


def run_bag(bag, routes, params, stops=None, traction=None):
    d = refdata.load(bag)
    est = Estimator(routes, params, stops, traction, load_queues())
    T, K, V = events(d)
    llh = {MFIX: d['master_llh'], RFIX: d['rover_llh']}
    out_t, out = [], []
    sparse_seen = set()
    wall = 0.0
    for t, k, v in zip(T, K, V):
        c0 = time.perf_counter()
        if k == FRONT:
            est.on_wheel(t, 'front', v)
        elif k == REAR:
            est.on_wheel(t, 'rear', v)
        elif k == CMD:
            est.on_cmd(t, v)
        else:
            if not est.aligning:
                # GNSS после выставки: 'start' - нет (как в условии по умолчанию), 'burst' - пачки
                # по 5 с каждые 120 с, 'sparse' - по фиксу антенны раз в 30 с, 'all' - весь прогон (оценка против того же GNSS оптимистична)
                if GNSS_MODE == 'start' or (GNSS_MODE == 'burst' and (t - T[0]) % 120.0 > 5.0):
                    continue
                if GNSS_MODE == 'sparse':
                    # по одному фиксу каждой антенны раз в SPARSE_PERIOD с
                    b = (k, int((t - T[0]) // SPARSE_PERIOD))
                    if b in sparse_seen:
                        continue
                    sparse_seen.add(b)
            la, lo, _ = llh[k][int(v)]
            est.on_gnss(t, 'master' if k == MFIX else 'rover', la, lo)
            continue
        st = est.state(t)
        wall += time.perf_counter() - c0
        if st is not None:
            out_t.append(t)
            out.append(st)
    if not out:
        return None
    out_t = np.array(out_t)
    out = np.array(out)
    return metrics(d, out_t, out, routes, est), wall / max(len(out_t), 1)


def nearest(ts, q):
    j = np.clip(np.searchsorted(ts, q), 1, len(ts) - 1)
    return np.where(np.abs(ts[j - 1] - q) <= np.abs(ts[j] - q), j - 1, j)


def metrics(d, ot, o, routes, est):
    m = {}
    t0 = ot[0]
    vt, vv = d['vel_t'], d['vel_v']
    sel = (vt >= t0) & (vt <= ot[-1]) & (vv < 40)
    mt = d['master_t']
    if len(mt):
        jm = nearest(mt, vt)
        sel &= np.abs(mt[jm] - vt) < 0.2          # без валидного фикса gnss/vel равна 0
    j = nearest(ot, vt[sel])
    ok = np.abs(ot[j] - vt[sel]) < 0.05
    e = o[j[ok], 0] - vv[sel][ok]
    m['v_rmse'] = float(np.sqrt(np.mean(e ** 2)))
    m['v_mae'] = float(np.mean(np.abs(e)))
    m['v_bias'] = float(np.mean(e))
    m['v_max'] = float(np.abs(e).max())
    rt, rp = d['ref_t'], d['ref_xyz']
    sel = (rt >= t0) & (rt <= ot[-1])
    j = nearest(ot, rt[sel])
    ok = np.abs(ot[j] - rt[sel]) < 0.05
    P, Q = o[j[ok], 1:4], rp[sel][ok]
    yaw = o[j[ok], 4]
    dxy = P[:, :2] - Q[:, :2]
    e3 = np.linalg.norm(P - Q, axis=1)
    at = dxy[:, 0] * np.cos(yaw) + dxy[:, 1] * np.sin(yaw)
    ct = -dxy[:, 0] * np.sin(yaw) + dxy[:, 1] * np.cos(yaw)
    dist = float(np.sum(np.abs(np.diff(o[:, 5]))))
    m.update(p_rmse=float(np.sqrt(np.mean(e3 ** 2))), p_mean=float(e3.mean()), p_max=float(e3.max()),
             p_p95=float(np.percentile(e3, 95)), at_rmse=float(np.sqrt(np.mean(at ** 2))),
             at_max=float(np.abs(at).max()), ct_rmse=float(np.sqrt(np.mean(ct ** 2))),
             z_rmse=float(np.sqrt(np.mean((P[:, 2] - Q[:, 2]) ** 2))),
             drift_pct=float(100 * e3[-1] / max(dist, 1.0)), dist_m=dist, n_ref=int(len(e3)),
             route=est.route.name, n_queue=est.n_queue, n_corr=est.n_corrections, scale=est.scale, n_rej=est.n_rejected)
    return m


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else 'val'
    params = {}
    for a in sys.argv[2:]:
        k, v = a.split('=')
        params[k] = json.loads(v)
    tr, va = refdata.split()
    bags = {'val': va, 'train': tr, 'all': tr + va}[which]
    routes = load_routes()
    stops = load_stops()
    traction = load_traction()
    vehicles = load_vehicles()
    rows = []
    walls = []
    for b in bags:
        d = refdata.load(b)
        if len(d['master_t']) and d['master_t'][0] - d['front_t'][0] > LATE_INIT:
            print('%-16s  skipped: first GNSS fix %.0f s after bag start (not a test-like case)' % (
                b, d['master_t'][0] - d['front_t'][0]))
            continue
        pb = dict(params)
        if 'wheel_scale' not in params:
            pb['wheel_scale'] = vehicles.get(b.split('_')[0], 1.0)
        r = run_bag(b, routes, pb, stops, traction)
        if r is None:
            print('%-16s  NOT ALIGNED' % b)
            continue
        m, w = r
        walls.append(w)
        rows.append((b, m))
        print('%-16s v_rmse=%.3f p_rmse=%7.2f p_max=%7.2f at_rmse=%7.2f ct=%5.2f drift=%5.2f%% corr=%2d scale=%.4f' % (
            b, m['v_rmse'], m['p_rmse'], m['p_max'], m['at_rmse'], m['ct_rmse'], m['drift_pct'],
            m['n_corr'], m['scale']))
    keys = ['v_rmse', 'v_mae', 'v_bias', 'p_rmse', 'p_mean', 'p_p95', 'p_max', 'at_rmse', 'at_max',
            'ct_rmse', 'z_rmse', 'drift_pct']
    print('\n%s (%d bags)  mean / median / worst' % (which, len(rows)))
    summ = {}
    for k in keys:
        a = np.array([m[k] for _, m in rows])
        worst = a.max() if k != 'v_bias' else a[np.abs(a).argmax()]
        summ[k] = (float(a.mean()), float(np.median(a)), float(worst))
        print('  %-10s %8.3f %8.3f %8.3f' % (k, *summ[k]))
    print('  cpu per output: %.1f us' % (1e6 * np.mean(walls)))
    return summ


if __name__ == '__main__':
    main()
