"""Входные потоки и GNSS-эталон (base_link) по каждому bag, кэш в npz.

Эталон base_link (ось передней тележки, уровень рельса) из двух антенн:
    base = master + 9.873 * unit(rover - master),  z = mean(alt) - 3.0
Эталон скорости - |master/vel linear.xy| (единственный независимый от колёс источник).
Разбиение: уникальные bag с GNSS по имени; чётный индекс -> train, нечётный -> val.
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reader as R

CACHE = os.path.join(R.REPO, 'dataset', 'cache')
MASTER_TO_BASE = 9.873
BASELINE = 12.436
ANT_Z = 3.0


def load(bag):
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, bag + '.npz')
    if os.path.exists(p):
        return dict(np.load(p))
    d = R.read_topics(bag, {R.TOPIC_FRONT: 'wheel', R.TOPIC_REAR: 'wheel', R.TOPIC_DRIVER: 'drv',
                            R.TOPIC_MASTER_FIX: 'fix', R.TOPIC_ROVER_FIX: 'fix',
                            R.TOPIC_MASTER_VEL: 'twist'})
    out = {}
    for key, topic in (('front', R.TOPIC_FRONT), ('rear', R.TOPIC_REAR), ('cmd', R.TOPIC_DRIVER)):
        out[key + '_t'], out[key + '_v'] = d[topic]
    mt, mg = d[R.TOPIC_MASTER_FIX]
    rt, rg = d[R.TOPIC_ROVER_FIX]
    out['master_t'], out['master_llh'] = mt, mg.reshape(-1, 3)
    out['rover_t'], out['rover_llh'] = rt, rg.reshape(-1, 3)
    vt, tw = d[R.TOPIC_MASTER_VEL]
    tw = tw.reshape(-1, 6)
    out['vel_t'], out['vel_v'] = vt, np.hypot(tw[:, 0], tw[:, 1])
    # эталон base_link в моменты фиксов master, если фикс rover не дальше 60 мс
    ref_t, ref_xyz = [], []
    if len(mt) and len(rt):
        mx, my = R.latlon_to_mgrs_local(mg[:, 0], mg[:, 1])
        rx, ry = R.latlon_to_mgrs_local(rg[:, 0], rg[:, 1])
        j = np.clip(np.searchsorted(rt, mt), 1, len(rt) - 1)
        j = np.where(np.abs(rt[j - 1] - mt) < np.abs(rt[j] - mt), j - 1, j)
        ok = np.abs(rt[j] - mt) < 0.06
        dx, dy = rx[j] - mx, ry[j] - my
        L = np.hypot(dx, dy)
        ok &= np.abs(L - BASELINE) < 0.5
        L[L == 0] = 1
        bx = mx + MASTER_TO_BASE * dx / L
        by = my + MASTER_TO_BASE * dy / L
        bz = 0.5 * (mg[:, 2] + rg[j, 2]) - ANT_Z
        ref_t, ref_xyz = mt[ok], np.c_[bx, by, bz][ok]
    out['ref_t'] = np.asarray(ref_t, float)
    out['ref_xyz'] = np.asarray(ref_xyz, float).reshape(-1, 3)
    np.savez(p, **out)
    return out


def split():
    """Списки (train, val) уникальных bag с пригодным GNSS-эталоном."""
    p = os.path.join(CACHE, 'split.csv')
    if not os.path.exists(p):
        uniq, _ = R.unique_bags()
        good = []
        for b in uniq:
            d = load(b)
            if len(d['ref_t']) >= 300:
                good.append(b)
        good.sort()
        with open(p, 'w', newline='') as f:
            w = csv.writer(f)
            for i, b in enumerate(good):
                w.writerow([b, 'train' if i % 2 == 0 else 'val'])
    rows = list(csv.reader(open(p)))
    return [b for b, s in rows if s == 'train'], [b for b, s in rows if s == 'val']


if __name__ == '__main__':
    tr, va = split()
    print('train %d, val %d' % (len(tr), len(va)))
