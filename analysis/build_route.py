"""Варианты маршрута по train-прогонам: начало по GNSS + официальная карта + конец,
сцепленные через кольца на конечных. Читает только эталоны train.

Запуск: python3 analysis/build_route.py train --output DIR/routes.json
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tram_odometry'))
import reader as R
import refdata
from tram_odometry.route import Route, save_routes

WEST = 'shchukinskaya_tallinskaya'
EAST = 'tallinskaya_shchukinskaya'


Z_TAPER = 100.   # м: сдвиг высоты стыка затухает на этом расстоянии от карты


def arc_length(p):
    return np.r_[0., np.cumsum(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1))]


def resample(p, step=1., smooth=5):
    """Убрать дрожание на стоянках, прорядить по длине дуги XY, сохранить оба конца."""
    p = np.asarray(p, float)
    if len(p) < 2:
        return p.copy()
    # сравниваем с последней оставленной точкой, а не с предыдущим шумным фиксом
    keep = [0]
    for i in range(1, len(p)):
        if np.linalg.norm(p[i, :2] - p[keep[-1], :2]) >= .15:
            keep.append(i)
    if np.linalg.norm(p[-1, :2] - p[keep[-1], :2]) > 1e-6:
        keep.append(len(p) - 1)
    p = p[keep]
    if len(p) < 2:
        return p.copy()
    s = arc_length(p)
    u = np.r_[np.arange(0., s[-1], step), s[-1]]
    q = np.column_stack([np.interp(u, s, p[:, k]) for k in range(3)])
    if smooth > 1 and len(q) > smooth:
        h = smooth // 2
        out = q.copy()
        for i in range(1, len(q) - 1):
            a, b = max(0, i - h), min(len(q), i + h + 1)
            # линейное удаление тренда не сжимает концы и касательные
            out[i] = q[a:b].mean(0) + (u[i] - u[a:b].mean()) * (q[b-1] - q[a]) / max(u[b-1] - u[a], 1e-9)
        q = out
    return q


def project_many(route, p, batch=256):
    """Точная векторная проекция; короткие окна у конечных ограничивают память."""
    a = route.xyz[:-1, :2]
    v = np.diff(route.xyz[:, :2], axis=0)
    vv = np.maximum(np.sum(v*v, axis=1), 1e-12)
    out = []
    for k in range(0, len(p), batch):
        d = p[k:k+batch, None, :2] - a
        u = np.clip(np.sum(d*v, axis=2) / vv, 0., 1.)
        e = np.linalg.norm(d - u[:, :, None]*v, axis=2)
        j = np.argmin(e, axis=1)
        ii = np.arange(len(j))
        out.append(np.column_stack([route.s[j] + u[ii, j]*np.sqrt(vv[j]), e[ii, j]]))
    return np.vstack(out) if out else np.empty((0, 2))


def _clean_piece(t, p, prefix):
    """Оставить непрерывную часть, касающуюся карты; разрывы на стоянках допустимы."""
    gap = (np.diff(t) > 3.) & (np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1) > 2.)
    ids = np.flatnonzero(gap)
    if len(ids):
        p = p[ids[-1]+1:] if prefix else p[:ids[0]+1]
    return p


def terminal_piece(t, p, m, prefix):
    """Обрезать по плоскости, касательной к концу карты, и плавно свести к его точным XYZ.

    В отличие от простой склейки, трек не заходит обратно на первые/последние 0..20 м карты.
    Каждое направление проверяется отдельно, поэтому короткие выезды с конечной не
    классифицируются по большинству точек на карте.
    """
    end = m.xyz[0] if prefix else m.xyz[-1]
    local = Route('local', m.xyz[:100] if prefix else m.xyz[-100:])
    nearby = np.flatnonzero(np.linalg.norm(p[:, :2] - end[:2], axis=1) < 110.)
    if not len(nearby):
        return None
    proj = project_many(local, p[nearby])
    on = (proj[:, 1] < 2.) & ((proj[:, 0] < 25.) if prefix else (proj[:, 0] > local.length-25.))
    hits = nearby[on]
    if not len(hits):
        return None
    tangent = m.xyz[5, :2]-m.xyz[0, :2] if prefix else m.xyz[-1, :2]-m.xyz[-6, :2]
    tangent /= np.linalg.norm(tangent)
    along = (p[:, :2]-end[:2]) @ tangent
    # ищем настоящий проход вперёд рядом с надёжной точкой на карте
    endpoint_distance = np.linalg.norm(p[:, :2]-end[:2], axis=1)
    crossings = np.flatnonzero((along[:-1] <= 0.) & (along[1:] > 0.) &
                              (np.minimum(endpoint_distance[:-1], endpoint_distance[1:]) < 5.))
    if not len(crossings):
        return None
    anchor = hits[0] if prefix else hits[-1]
    k = int(crossings[np.argmin(np.abs(crossings-anchor))])
    u = -along[k] / max(along[k+1]-along[k], 1e-9)
    crossing = p[k]*(1-u) + p[k+1]*u
    if prefix:
        q = _clean_piece(t[:k+2], np.vstack([p[:k+1], crossing]), True)
    else:
        q = _clean_piece(t[k:], np.vstack([crossing, p[k+1:]]), False)
    if len(q) < 3 or np.max(np.linalg.norm(q[:, :2]-end[:2], axis=1)) > 800.:
        return None
    q = resample(q)
    s = arc_length(q)
    if s[-1] < 10. or s[-1] > 1200.:
        return None
    # разница z на стыке - дрейф высоты GNSS train, а не ступенька рельса:
    # убираем её постоянную часть, высоту официальной карты не меняем
    endpoint = q[-1] if prefix else q[0]
    dz = end[2]-endpoint[2]
    dist = s[-1]-s if prefix else s
    q[:, 2] += dz*np.clip(1.-dist/Z_TAPER, 0., 1.)
    w = np.clip(1.-dist/20., 0., 1.)
    shift = end[:2]-(q[-1, :2] if prefix else q[0, :2])
    q[:, :2] += w[:, None]*shift
    q[-1 if prefix else 0] = end
    return q


def distinct(polys, tol=2., min_new=12.):
    kept = []
    for bag, q in sorted(polys, key=lambda z: -arc_length(z[1])[-1]):
        sample = q[::3]
        new = np.ones(len(sample), bool)
        for _, p in kept:
            new &= project_many(Route('kept', p), sample)[:, 1] > tol
        if new.sum()*3 >= min_new:
            kept.append((bag, q))
    return kept


def tangent(p, at_end, distance=10.):
    s = arc_length(p)
    if at_end:
        a = max(0, int(np.searchsorted(s, s[-1]-distance)))
        v = p[-1, :2]-p[a, :2]
    else:
        b = min(len(p)-1, int(np.searchsorted(s, distance)))
        v = p[b, :2]-p[0, :2]
    return v/np.linalg.norm(v)


def circle_arc(p, q, t, step=.5):
    """Дуга окружности из p в q с заданной начальной единичной касательной."""
    w = q-p
    cross = t[0]*w[1]-t[1]*w[0]
    if abs(cross) < 1e-8:
        return np.linspace(p, q, max(2, int(np.linalg.norm(w)/step)+1))
    radius = np.dot(w, w)/(2.*cross)
    center = p + radius*np.array([-t[1], t[0]])
    a = np.arctan2(*(p-center)[::-1])
    b = np.arctan2(*(q-center)[::-1])
    angle = (b-a) % (2*np.pi) if radius > 0 else -((a-b) % (2*np.pi))
    if abs(angle) > np.pi+1e-5:
        raise ValueError('unphysical long circular arc')
    u = np.linspace(a, a+angle, max(2, int(abs(radius*angle)/step)+1))
    xy = center + abs(radius)*np.column_stack([np.cos(u), np.sin(u)])
    xy[0], xy[-1] = p, q
    return xy


def biarc(a, b):
    """G1-биарк с равными касательными отрезками между концами, наблюдаемыми в train."""
    p, q = a[-1, :2], b[0, :2]
    t0, t1 = tangent(a, True), tangent(b, False)
    v = q-p
    c = 2.*(1.-np.dot(t0, t1))
    h = np.dot(v, t0+t1)
    d = (-h+np.sqrt(h*h+c*np.dot(v, v)))/c if c > 1e-8 else np.dot(v, v)/(2.*h)
    if not np.isfinite(d) or d <= 0:
        raise ValueError('terminal tangents cannot form a forward biarc')
    mid = .5*(p+q+d*(t0-t1))
    xy = np.vstack([circle_arc(p, mid, t0), circle_arc(q, mid, -t1)[::-1][1:]])
    s = arc_length(xy)
    if s[-1] > 120. or np.linalg.norm(v) > 70.:
        raise ValueError('unsupported terminal gap')
    return np.column_stack([xy, np.interp(s, [0., s[-1]], [a[-1, 2], b[0, 2]])])


def join(*parts):
    """Склеить без дублирующих вершин, сохранив точные точки карты и развилок."""
    q = np.vstack(parts)
    keep = np.r_[True, np.linalg.norm(np.diff(q[:, :2], axis=0), axis=1) > 1e-7]
    return q[keep]


def complete_west(suffixes, prefixes):
    if not suffixes or not prefixes:
        return suffixes, []
    bag, a = max(suffixes, key=lambda z: arc_length(z[1])[-1])
    candidates = []
    for bp, b in prefixes:
        gap = np.linalg.norm(a[-1, :2]-b[0, :2])
        # тупик уходит на юго-восток и потом поворачивает на север: в отличие от
        # возврата через кольцо, он не антипараллелен входящей касательной
        opposing = np.dot(tangent(a, True), tangent(b, False)) < -.75
        if 5. < gap < 70. and opposing:
            try:
                bridge = biarc(a, b)
            except ValueError:
                continue
            candidates.append((arc_length(bridge)[-1], bp, b, bridge))
    if not candidates:
        return suffixes, []
    _, bp, b, bridge = min(candidates, key=lambda z: z[0])
    completed = join(a, bridge, b)
    info = dict(from_bag=bag, to_bag=bp, inferred_length=float(arc_length(bridge)[-1]),
                chord=float(np.linalg.norm(a[-1, :2]-b[0, :2])), method='equal_distance_G1_biarc')
    print('West terminal completion:', info, flush=True)
    return [(bag+'+biarc+'+bp, completed)], [info]


def extend_terminal(suffixes, distance):
    """Открытое прямое продолжение, без выдуманного соединения с обратным путём.

    Курс - по последним 15 м наблюдаемого пути, высота дальше постоянная. Это явный
    запасной вариант, а не выученная геометрия.
    """
    bag, p = max(suffixes, key=lambda z: arc_length(z[1])[-1])
    t = tangent(p, True, distance=15.)
    u = np.r_[np.arange(1., distance, 1.), distance]
    q = np.tile(p[-1], (len(u), 1))
    q[:, :2] += u[:, None]*t
    info = dict(from_bag=bag, inferred_length=float(distance), method='open_terminal_tangent_extrapolation')
    return [(bag+'+open_tangent', join(p, q))], [info]


def build(train, complete=False, terminal_extension=0.):
    maps = {n: Route(n, R.load_map(n)[:, :3]) for n in R.map_names()}
    pre, suf = {n: [] for n in maps}, {n: [] for n in maps}
    for bag in train:
        d = refdata.load(bag)
        t, p = d['ref_t'], d['ref_xyz']
        if len(p) < 3:
            continue
        found = []
        for n, m in maps.items():
            for is_pre, target in [(True, pre), (False, suf)]:
                q = terminal_piece(t, p, m, is_pre)
                if q is not None:
                    target[n].append((bag, q))
                    found.append(n[:2]+(' pre ' if is_pre else ' suf ')+str(round(arc_length(q)[-1])))
        print(bag, '; '.join(found), flush=True)
    pre = {n: distinct(v) for n, v in pre.items()}
    suf = {n: distinct(v) for n, v in suf.items()}
    inferred = []
    if complete:
        suf[WEST], inferred = complete_west(suf[WEST], pre[EAST])
    elif terminal_extension > 0. and suf[WEST]:
        suf[WEST], inferred = extend_terminal(suf[WEST], terminal_extension)
    base = {}
    for n, m in maps.items():
        tails = [(b, q, 'loop') for b, q in suf[n]] or [(None, m.xyz[-1:], 'loop')]
        if n == WEST and pre[EAST]:
            # тупик по-прежнему - развёрнутое самое длинное начало рейса на восток
            b, q = pre[EAST][0]
            # общий конец карты остаётся точным; только короткий съезд
            tails.append(('rev:'+b, join(m.xyz[-1:], q[::-1]), 'stub'))
        base[n] = []
        for bp, p in pre[n] or [(None, m.xyz[:1])]:
            for bs, q, kind in tails:
                xyz = join(p, m.xyz, q)
                plen = float(arc_length(p)[-1])
                meta = dict(direction=n, prefix_from=bp, suffix_from=bs, suffix_kind=kind,
                            prefix_len=plen, map_s0=plen, map_end=plen+m.length,
                            suffix_len=float(arc_length(q)[-1]),
                            tail_kind=kind if n == WEST else None,
                            fork_s=plen+m.length if n == WEST else None)
                base[n].append((Route(n, xyz), meta))
    routes = []
    names = list(maps)
    for n in names:
        other = base[names[1-names.index(n)]]
        for r, meta in base[n]:
            candidates = []
            end_tangent = tangent(r.xyz, True)
            for o, om in other:
                sj, xt, yaw = o.project(*r.xyz[-1, :2])
                if xt < 3. and np.dot(end_tangent, [np.cos(yaw), np.sin(yaw)]) > .5:
                    candidates.append((xt, sj, o, om))
            kinds = {}
            for xt, sj, o, om in sorted(candidates, key=lambda v: (v[0], -v[2].length)):
                key = om['tail_kind'] if meta['tail_kind'] is None else None
                kinds.setdefault(key, (sj, o, om))
            variants = []
            for kind, (sj, o, om) in kinds.items():
                k = int(np.searchsorted(o.s, sj, side='right'))
                xyz = join(r.xyz, np.asarray(o.point(sj)[:3])[None, :], o.xyz[k:])
                m2 = dict(meta, chained_at=r.length)
                if meta['tail_kind'] is None:
                    m2.update(tail_kind=kind, fork_s=(r.length+om['fork_s']-sj if om['fork_s'] is not None else None))
                variants.append((xyz, m2))
            for xyz, m2 in variants or [(r.xyz, meta)]:
                name = n+'_v'+str(len(routes))
                routes.append(Route(name, xyz, m2))
                print(name, 'length', round(routes[-1].length, 2), 'tail', m2['tail_kind'], 'chained', 'chained_at' in m2, flush=True)
    return routes, inferred


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('split', nargs='?', choices=['train'], default='train')
    parser.add_argument('--output', default=os.path.join(R.REPO, 'dataset/cache/model_v2/routes.json'))
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--complete-west', action='store_true', help='Experimental, unsupported closed-loop hypothesis')
    group.add_argument('--no-complete-west', action='store_true', help='Compatibility alias for conservative default')
    group.add_argument('--terminal-extension', type=float, default=0., metavar='METRES', help='Open terminal tangent extrapolation, not surveyed geometry')
    args = parser.parse_args(argv)
    if not 0. <= args.terminal_extension <= 200.:
        parser.error('--terminal-extension must be in [0, 200] metres')
    split_file = Path(refdata.CACHE)/'split.csv'
    if not split_file.exists():
        parser.error('Existing dataset/cache/split.csv required; parent must prepare the fixed split first.')
    with split_file.open(newline='') as f:
        rows = list(csv.reader(f))
    train = [b for b, label in rows if label == 'train']
    if not train or set(train) & {b for b, label in rows if label == 'val'}:
        parser.error('Empty or overlapping train split')
    # лучше ошибка, чем неявная загрузка сырого датасета
    missing = [b for b in train if not (Path(refdata.CACHE)/(b+'.npz')).exists()]
    if missing:
        parser.error('Training caches not ready: '+', '.join(missing))
    routes, inferred = build(train, complete=args.complete_west, terminal_extension=args.terminal_extension)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_routes(routes, path=args.output, extra=dict(source='train', builder='build_route',
                train_bags=train, inferred_connectors=inferred))
    print('Saved', args.output, flush=True)


if __name__ == '__main__':
    main()
