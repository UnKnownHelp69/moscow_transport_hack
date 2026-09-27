"""Масштаб колёс по вагону: истинная скорость = scale * скорость колеса (км/ч / 3.6).

Оценивается по |gnss/vel| при 8-16 м/с и валидном фиксе (префикс имени bag - номер вагона).
Запуск: python vehicle_scale.py [train|all]  -> <data>/vehicles.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'tram_odometry'))
import refdata
from tram_odometry.route import DATA_DIR

SPLIT = sys.argv[1] if len(sys.argv) > 1 else 'train'


def main():
    tr, va = refdata.split()
    bags = tr + va if SPLIT == 'all' else tr
    ratios = {}
    for b in bags:
        d = refdata.load(b)
        vt, vv, mt = d['vel_t'], d['vel_v'], d['master_t']
        j = np.clip(np.searchsorted(mt, vt), 1, len(mt) - 1)
        ok = np.abs(mt[j] - vt) < 0.2
        w = 0.5 * (np.interp(vt, d['front_t'], d['front_v']) + np.interp(vt, d['rear_t'], d['rear_v'])) / 3.6
        m = ok & (vv > 8) & (vv < 16)
        ratios.setdefault(b.split('_')[0], []).append(w[m] / vv[m])
    out = {k: round(float(1.0 / np.median(np.concatenate(v))), 5) for k, v in ratios.items()}
    print(out)
    with open(os.path.join(DATA_DIR, 'vehicles.json'), 'w') as f:
        json.dump({'source': SPLIT, 'wheel_scale': out}, f, indent=1)


if __name__ == '__main__':
    main()
