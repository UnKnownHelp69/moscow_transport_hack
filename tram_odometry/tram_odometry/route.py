"""Маршруты-полилинии: длина дуги s <-> (x, y, z) в MGRS local, м."""
import json
import os

import numpy as np

DATA_DIR = os.environ.get('TRAM_ODOM_DATA') or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


class Route:
    def __init__(self, name, xyz, meta=None):
        self.name = name
        self.xyz = np.asarray(xyz, float)
        seg = np.hypot(np.diff(self.xyz[:, 0]), np.diff(self.xyz[:, 1]))
        self.s = np.r_[0.0, np.cumsum(seg)]
        self.length = float(self.s[-1])
        self.meta = meta or {}
        # геометрия сегментов кэшируется для проекций
        self._a = self.xyz[:-1, :2]
        self._d = self.xyz[1:, :2] - self._a
        self._l2 = np.maximum((self._d * self._d).sum(1), 1e-12)

    def point(self, s):
        """(x, y, z, yaw) в точке s (s ограничивается длиной маршрута)."""
        s = min(max(s, 0.0), self.length)
        i = int(np.searchsorted(self.s, s, side='right')) - 1
        i = min(max(i, 0), len(self.s) - 2)
        ds = self.s[i + 1] - self.s[i]
        u = (s - self.s[i]) / ds if ds > 0 else 0.0
        p = self.xyz[i] + u * (self.xyz[i + 1] - self.xyz[i])
        d = self.xyz[i + 1] - self.xyz[i]
        return float(p[0]), float(p[1]), float(p[2]), float(np.arctan2(d[1], d[0]))

    def project_candidates(self, x, y, max_distance=40.0):
        """Все пространственно разные локальные проекции на самопересекающийся маршрут.

        Только ближайшая проекция теряет верную гипотезу старта, если маршрут дальше проходит
        ближе в обратном направлении.
        """
        a, d, l2 = self._a, self._d, self._l2
        u = np.clip(((x-a[:,0])*d[:,0]+(y-a[:,1])*d[:,1])/l2, 0., 1.)
        dist = np.hypot(a[:,0]+u*d[:,0]-x, a[:,1]+u*d[:,1]-y)
        local = (dist <= np.r_[np.inf,dist[:-1]]) & (dist <= np.r_[dist[1:],np.inf])
        ids = np.flatnonzero(local & (dist <= max_distance))
        out = []
        for i in sorted(ids, key=lambda j: dist[j]):
            s = float(self.s[i]+u[i]*np.sqrt(l2[i]))
            if all(abs(s-c[0]) > 10. for c in out):
                out.append((s,float(dist[i]),float(np.arctan2(d[i,1],d[i,0]))))
        return out

    def project(self, x, y):
        """Ближайшая точка полилинии: (s, поперечное расстояние, курс касательной)."""
        a, d, L2 = self._a, self._d, self._l2
        u = np.clip(((x - a[:, 0]) * d[:, 0] + (y - a[:, 1]) * d[:, 1]) / L2, 0.0, 1.0)
        px = a[:, 0] + u * d[:, 0]
        py = a[:, 1] + u * d[:, 1]
        dist = np.hypot(px - x, py - y)
        i = int(dist.argmin())
        return (float(self.s[i] + u[i] * np.sqrt(L2[i])), float(dist[i]),
                float(np.arctan2(d[i, 1], d[i, 0])))


def load_routes(path=None):
    path = path or os.path.join(DATA_DIR, 'routes.json')
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    return [Route(r['name'], r['xyz'], r.get('meta')) for r in d['routes']]


def load_traction(path=None):
    path = path or os.path.join(DATA_DIR, 'traction.json')
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def load_vehicles(path=None):
    path = path or os.path.join(DATA_DIR, 'vehicles.json')
    if not os.path.exists(path):
        return {}
    with open(path, encoding='utf-8') as f:
        return json.load(f)['wheel_scale']


def load_stops(path=None):
    path = path or os.path.join(DATA_DIR, 'stops.json')
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8') as f:
        return json.load(f)['stops']


def load_queues(path=None):
    path = path or os.path.join(DATA_DIR, 'stops.json')
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8') as f:
        return json.load(f).get('queues', [])


def save_routes(routes, path=None, extra=None):
    path = path or os.path.join(DATA_DIR, 'routes.json')
    out = {'frame': 'MGRS 37U (UTM37N - (300000, 6100000)), metres; z = rail top (base_link)',
           'routes': [{'name': r.name, 'meta': r.meta,
                       'xyz': [[round(float(v), 3) for v in p] for p in r.xyz]} for r in routes]}
    if extra:
        out.update(extra)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, separators=(',', ':'))
