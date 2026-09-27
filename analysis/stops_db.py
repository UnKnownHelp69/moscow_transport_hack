"""Контрольные точки-остановки по train-прогонам.

Остановка: обе тележки < 0.3 км/ч не меньше MIN_DWELL с. Её положение - эталон base_link,
спроецированный на официальную карту своего направления (длина дуги s_map). Остановки
кластеризуются по s_map отдельно по направлениям; кластеры с достаточным числом событий и
малым разбросом становятся контрольными точками: tram_odometry/tram_odometry/data/stops.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'tram_odometry'))
import reader as R
import refdata
from tram_odometry.route import DATA_DIR, Route

MIN_DWELL = 5.0
V_STOP = 0.3        # км/ч
GAP = 12.0          # м: новый кластер, если следующая остановка дальше этого
STATION_MIN_BAGS, STATION_MAX_MAD = 3, 0.5   # тот же фильтр, что в оценщике
QUEUE_TOL = 3.0     # м: остановки дальше этого от станции - кандидаты в очереди и светофоры
QUEUE_GAP = 15.0    # м: зазор кластеризации мест очередей
QUEUE_MIN_BAGS = 2


def stop_events(d):
    t, v = d['front_t'], np.maximum(d['front_v'], np.interp(d['front_t'], d['rear_t'], d['rear_v']))
    m = v < V_STOP
    e = np.diff(np.r_[0, m.astype(int), 0])
    out = []
    for a, b in zip(np.where(e == 1)[0], np.where(e == -1)[0]):
        if t[b - 1] - t[a] >= MIN_DWELL:
            out.append((t[a], t[b - 1]))
    return out


def main(split='train'):
    maps = {n: Route(n, R.load_map(n)[:, :3]) for n in R.map_names()}
    tr, va = refdata.split()
    bags = tr if split == 'train' else tr + va
    ev = {n: [] for n in maps}
    for b in bags:
        d = refdata.load(b)
        rt, rp = d['ref_t'], d['ref_xyz']
        for t0, t1 in stop_events(d):
            j = np.where((rt >= t0) & (rt <= t1))[0]
            if len(j) < 3:
                continue
            x, y = np.median(rp[j, 0]), np.median(rp[j, 1])
            best = min(((n, *m.project(x, y)) for n, m in maps.items()), key=lambda z: z[2])
            n, s, xt, _ = best
            if xt < 1.5 and 5 < s < maps[n].length - 5:
                ev[n].append((s, t1 - t0, x, y, b))
    out = []
    for n, E in ev.items():
        E.sort()
        s = np.array([e[0] for e in E])
        groups, cur = [], [0]
        for i in range(1, len(s)):
            if s[i] - s[i - 1] > GAP:
                groups.append(cur)
                cur = []
            cur.append(i)
        groups.append(cur)
        for g in groups:
            ss = s[g]
            nb = len({E[i][4] for i in g})
            out.append(dict(direction=n, s_map=float(np.median(ss)), std=float(ss.std()),
                            mad=float(np.median(np.abs(ss - np.median(ss)))), n=len(g), n_bags=nb,
                            x=float(np.median([E[i][2] for i in g])),
                            y=float(np.median([E[i][3] for i in g])),
                            dwell_med=float(np.median([E[i][1] for i in g]))))
    # места очередей и светофоров: повторяющиеся остановки вдали от станций
    queues = []
    for n, E in ev.items():
        st = np.array([c['s_map'] for c in out if c['direction'] == n
                       and c['n_bags'] >= STATION_MIN_BAGS and c['mad'] <= STATION_MAX_MAD])
        Q = [e for e in E if not len(st) or np.abs(st - e[0]).min() > QUEUE_TOL]
        Q.sort()
        if not Q:
            continue
        s = np.array([e[0] for e in Q])
        groups, cur = [], [0]
        for i in range(1, len(s)):
            if s[i] - s[i - 1] > QUEUE_GAP:
                groups.append(cur)
                cur = []
            cur.append(i)
        groups.append(cur)
        for g in groups:
            nb = len({Q[i][4] for i in g})
            if nb < QUEUE_MIN_BAGS:
                continue
            ss = s[g]
            queues.append(dict(direction=n, s_map=float(np.median(ss)), std=float(max(ss.std(), 1.0)),
                               n=len(g), n_bags=nb, x=float(np.median([Q[i][2] for i in g])),
                               y=float(np.median([Q[i][3] for i in g])),
                               dwell_med=float(np.median([Q[i][1] for i in g]))))
    for q in queues:
        print('QUEUE %-26s s=%7.1f n=%3d bags=%2d std=%5.2f dwell=%5.1f' % (
            q['direction'], q['s_map'], q['n'], q['n_bags'], q['std'], q['dwell_med']))
    nb_total = len(bags)
    for c in out:
        c['freq'] = c['n_bags'] / nb_total
        print('%-26s s=%7.1f n=%3d bags=%2d freq=%.2f std=%5.2f mad=%5.2f dwell=%5.1f' % (
            c['direction'], c['s_map'], c['n'], c['n_bags'], c['freq'], c['std'], c['mad'], c['dwell_med']))
    with open(os.path.join(DATA_DIR, 'stops.json'), 'w') as f:
        json.dump({'source': split, 'stops': out, 'queues': queues}, f, indent=1)


if __name__ == '__main__':
    main(*(sys.argv[1:2]))
