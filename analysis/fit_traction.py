"""Подбор модели тяги и торможения a = f(позиция, v) по train-прогонам (только колёса, без GNSS).

Скорость - среднее двух тележек (км/ч -> м/с) на сетке 0.1 с; ускорение - сглаженная
производная. Позиция контроллера берётся на `lag` с раньше (задержка привода и тормозов).
Результат - таблица медиан ускорения по ячейкам позиция x скорость ->
tram_odometry/.../data/traction.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'tram_odometry'))
import refdata

SPLIT = sys.argv[1] if len(sys.argv) > 1 else 'train'
from tram_odometry.route import DATA_DIR

V_BINS = [0.0, 1.0, 3.0, 6.0, 9.0, 12.0, 15.0, 30.0]   # м/с
DT = 0.1


def series(d):
    t0, t1 = max(d['front_t'][0], d['rear_t'][0]), min(d['front_t'][-1], d['rear_t'][-1])
    g = np.arange(t0, t1, DT)
    f = np.interp(g, d['front_t'], d['front_v'])
    r = np.interp(g, d['rear_t'], d['rear_v'])
    ok = np.abs(f - r) < 1.0            # пропуск сбоев датчиков
    v = 0.5 * (f + r) / 3.6
    k = np.ones(5) / 5
    vs = np.convolve(v, k, mode='same')
    a = np.gradient(vs, DT)
    return g, v, a, ok


def main():
    tr, va = refdata.split()
    data = [refdata.load(b) for b in (tr + va if SPLIT == 'all' else tr)]
    S = [series(d) for d in data]
    best = None
    for lag in np.arange(0.0, 2.01, 0.1):
        X, Y = [], []
        for d, (g, v, a, ok) in zip(data, S):
            n = np.interp(g - lag, d['cmd_t'], d['cmd_v'], left=0).round()
            m = ok & (v > 1.0)
            X.append(n[m])
            Y.append(a[m])
        X, Y = np.concatenate(X), np.concatenate(Y)
        pred = np.zeros_like(Y)
        for k in np.unique(X):
            pred[X == k] = np.median(Y[X == k])
        r2 = 1 - np.mean((Y - pred) ** 2) / np.var(Y)
        if best is None or r2 > best[0]:
            best = (r2, lag)
        print('lag %.1f s  R2(notch only) = %.3f' % (lag, r2))
    r2, lag = best
    print('best lag %.1f s (R2 %.3f)' % (lag, r2))
    # таблица позиция x интервал скорости
    N, V, A = [], [], []
    for d, (g, v, a, ok) in zip(data, S):
        n = np.interp(g - lag, d['cmd_t'], d['cmd_v'], left=0).round()
        m = ok & (v > 0.2)
        N.append(n[m]); V.append(v[m]); A.append(a[m])
    N, V, A = np.concatenate(N), np.concatenate(V), np.concatenate(A)
    notches = list(range(-15, 16))
    table = []
    for k in notches:
        row = []
        for i in range(len(V_BINS) - 1):
            m = (N == k) & (V >= V_BINS[i]) & (V < V_BINS[i + 1])
            row.append(float(np.median(A[m])) if m.sum() >= 30 else None)
        table.append(row)
    # пустые ячейки: ближайший интервал той же позиции, затем ближайшая позиция
    arr = np.array([[np.nan if x is None else x for x in r] for r in table])
    for i in range(arr.shape[0]):
        good = np.where(~np.isnan(arr[i]))[0]
        if len(good):
            for j in range(arr.shape[1]):
                if np.isnan(arr[i, j]):
                    arr[i, j] = arr[i, good[np.abs(good - j).argmin()]]
    for j in range(arr.shape[1]):
        good = np.where(~np.isnan(arr[:, j]))[0]
        for i in range(arr.shape[0]):
            if np.isnan(arr[i, j]):
                arr[i, j] = arr[good[np.abs(good - i).argmin()], j]
    pred = np.array([arr[int(n) + 15, min(np.searchsorted(V_BINS, v, side='right') - 1, len(V_BINS) - 2)]
                     for n, v in zip(N, V)])
    r2v = 1 - np.mean((A - pred) ** 2) / np.var(A)
    resid = A - pred
    print('R2(notch x speed) = %.3f, residual std = %.3f m/s^2' % (r2v, resid.std()))
    print('speed bins (m/s):', V_BINS)
    for k, row in zip(notches, arr):
        print('%+3d ' % k + ' '.join('%6.2f' % x for x in row))
    out = dict(lag=float(lag), v_bins=V_BINS, notches=notches, accel=arr.round(3).tolist(),
               resid_std=float(resid.std()), r2=float(r2v), source=SPLIT)
    with open(os.path.join(DATA_DIR, 'traction.json'), 'w') as f:
        json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
