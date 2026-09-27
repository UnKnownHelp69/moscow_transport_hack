"""Маршруты по train-прогонам и официальной карте, западная конечная - по OSM.

Использует архив OSM в dataset/map/osm_west (ODbL, https://www.openstreetmap.org/copyright);
читает только эталоны train.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_route as B

BBOX = (37.385, 55.797, 37.394, 55.802)
CACHE = Path(B.R.REPO)/'dataset/map/osm_west'


def fetch_osm(cache=CACHE):
    """Сохранить небольшую выгрузку OSM и исходный ответ сервера без изменений."""
    cache = Path(cache)
    path = cache/'raw.json'
    provenance_path = cache/'provenance.json'
    if path.exists():
        return json.loads(path.read_text()), json.loads(provenance_path.read_text())
    query = '[out:json][timeout:30];way[railway=tram](55.797,37.385,55.802,37.394);out geom;'
    urls = ['https://api.openstreetmap.org/api/0.6/map?bbox='+','.join(map(str, BBOX)),
            'https://overpass.kumi.systems/api/interpreter?data='+urllib.parse.quote(query),
            'https://overpass-api.de/api/interpreter?data='+urllib.parse.quote(query)]
    errors = []
    for url in urls:
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'mos-trap-route-builder/1.0'})
            with urllib.request.urlopen(request, timeout=45) as f:
                raw = f.read()
            if raw.lstrip().startswith(b'<'):
                root = ET.fromstring(raw)
                nodes = {int(n.attrib['id']): dict(lat=float(n.attrib['lat']), lon=float(n.attrib['lon'])) for n in root.findall('node')}
                ways = []
                for w in root.findall('way'):
                    tags = {t.attrib['k']: t.attrib['v'] for t in w.findall('tag')}
                    if tags.get('railway') != 'tram':
                        continue
                    ids = [int(n.attrib['ref']) for n in w.findall('nd')]
                    ways.append(dict(type='way', id=int(w.attrib['id']), tags=tags, nodes=ids,
                                     geometry=[nodes[i] for i in ids], version=w.attrib.get('version'),
                                     timestamp=w.attrib.get('timestamp')))
                data = dict(elements=ways, generator=root.attrib.get('generator'))
            else:
                data = json.loads(raw)
            if not data.get('elements'):
                raise ValueError('No OSM ways returned')
            break
        except Exception as e:
            errors.append(str(e))
    else:
        raise RuntimeError('Public Overpass unavailable: '+'; '.join(errors))
    cache.mkdir(parents=True, exist_ok=True)
    (cache/('raw.osm' if raw.lstrip().startswith(b'<') else 'response.json')).write_bytes(raw)
    path.write_text(json.dumps(data)+'\n')
    provenance = dict(url=url, retrieved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      bbox=list(BBOX), license='ODbL', attribution='OpenStreetMap contributors',
                      copyright_url='https://www.openstreetmap.org/copyright',
                      osm_base_timestamp=data.get('osm3s', {}).get('timestamp_osm_base'))
    provenance_path.write_text(json.dumps(provenance, indent=2)+'\n')
    features = []
    for w in data['elements']:
        geom = w.get('geometry', [])
        if w.get('type') != 'way' or len(geom) < 2:
            continue
        features.append(dict(type='Feature', properties=dict(osm_id=w['id'], **w.get('tags', {})),
                             geometry=dict(type='LineString', coordinates=[[p['lon'], p['lat']] for p in geom])))
    (cache/'tram.geojson').write_text(json.dumps(dict(type='FeatureCollection', features=features), indent=2)+'\n')
    return data, provenance


def osm_xyz(ways, ids):
    pieces = []
    for i, wid in enumerate(ids):
        if i and ways[ids[i-1]]['nodes'][-1] != ways[wid]['nodes'][0]:
            raise ValueError('OSM directed ways are not node-connected: '+str(ids))
        g = ways[wid]['geometry']
        x, y = B.R.latlon_to_mgrs_local(np.array([p['lat'] for p in g]), np.array([p['lon'] for p in g]))
        pieces.append(np.column_stack([x, y, np.zeros(len(x))]))
    return B.resample(B.join(*pieces), step=.5, smooth=5)


def nearest_points(p, q):
    r = B.Route('osm', q)
    proj = B.project_many(r, p)
    xy = np.column_stack([np.interp(proj[:, 0], r.s, q[:, k]) for k in range(2)])
    return xy, proj[:, 1]


def fit_train_rigid(p, q):
    """Жёсткое преобразование ICP, подогнанное только к наблюдаемому пути train."""
    origin = p[:, :2].mean(0)
    source = q[:, :2]-origin
    target = p[:, :2]-origin
    rotation = np.eye(2)
    shift = np.zeros(2)
    for _ in range(40):
        current = source @ rotation + shift
        matches, dist = nearest_points(target, np.column_stack([current, np.zeros(len(current))]))
        use = dist < max(3., np.percentile(dist, 90))
        a, b = matches[use], target[use]
        ac, bc = a.mean(0), b.mean(0)
        u, _, vt = np.linalg.svd((a-ac).T @ (b-bc))
        step_r = u @ vt
        if np.linalg.det(step_r) < 0:
            u[:, -1] *= -1
            step_r = u @ vt
        step_t = bc-ac @ step_r
        rotation = rotation @ step_r
        shift = shift @ step_r + step_t
    out = q.copy()
    out[:, :2] = source @ rotation + shift + origin
    _, dist = nearest_points(p, out)
    return out, dict(rmse=float(np.sqrt(np.mean(dist**2))), median=float(np.median(dist)),
                     p95=float(np.percentile(dist, 95)), rotation_deg=float(np.degrees(np.arctan2(rotation[0, 1], rotation[0, 0]))),
                     translation_at_origin=shift.tolist(), origin_xy=origin.tolist())


def transform_public(q, fit):
    theta = np.radians(fit['rotation_deg'])
    rotation = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    origin = np.asarray(fit['origin_xy'])
    out = q.copy()
    out[:, :2] = (out[:, :2]-origin) @ rotation + np.asarray(fit['translation_at_origin']) + origin
    return out


def extend_return_prefixes(prefixes, ways, fit, topology):
    """Добавить варианты, достроенные назад, не меняя наблюдаемые начала и тупики.

    Все совмещения используют одно преобразование, подогнанное по train. Путь выбирается
    по первым 80 м каждого начала train. Фиксы val сюда не попадают.
    """
    graph = dict(ways)
    original = ways[188726183]
    for label, lo, hi in [('head', 0, 2), ('tail', 4, 6)]:
        part = dict(original)
        part['nodes'] = original['nodes'][lo:hi]
        part['geometry'] = original['geometry'][lo:hi]
        graph['188726183:'+label] = part
    # путь 4 ведёт к любой петле, путь 5 - только к внешней
    cap = [188726176] if topology == 'inner' else [1199696832, 1232873046]
    choices = [('4', cap+[188726177, 488091029], [188726177, 488091029]),
               ('5', [1199696832, '188726183:head', 188726178, '188726183:tail', 488091029],
                     [188726178, '188726183:tail', 488091029]),
               ('6', [1199696832, 188726183, 488091029], [188726183, 488091029])]
    additions, infos = [], []
    for bag, p in list(prefixes):
        observed = p[B.arc_length(p) < 80.]
        ranked = []
        for track, ids, returns in choices:
            q = transform_public(osm_xyz(graph, returns), fit)
            _, dist = nearest_points(observed, q)
            ranked.append((float(np.sqrt(np.mean(dist**2))), track, ids))
        score, track, ids = min(ranked)
        print('OSM prefix candidates', bag, ranked, flush=True)
        if score > 2.:
            continue
        q = transform_public(osm_xyz(graph, ids), fit)
        r = B.Route('return_extension', q)
        s, xt, _ = r.project(*p[0, :2])
        if xt > 3. or s < 5.:
            continue
        u = np.r_[r.s[r.s < s], s]
        extension = np.column_stack([np.interp(u, r.s, q[:, k]) for k in range(3)])
        w = np.clip(1.-(s-u)/20., 0., 1.)
        extension[:, :2] += w[:, None]*(p[0, :2]-extension[-1, :2])
        extension[:, 2] = p[0, 2]
        extension[-1] = p[0]
        additions.append((bag+'+osm_prefix_track'+track, B.join(extension, p)))
        infos.append(dict(from_bag=bag, track=track, method='public_osm_backward_prefix',
                          osm_way_ids=ids, train_rmse=score, added_length=float(B.arc_length(extension)[-1]),
                          transform=fit))
    prefixes.extend(additions)
    return infos


def osm_completion(data, provenance, topology='shortest', extend_prefixes=False):
    ways = {w['id']: w for w in data['elements'] if w.get('type') == 'way'}
    # направленная связная топология: входящий путь 3 -> внутренняя петля 7A
    # или внешняя 7 -> обратный путь 4/6; обратный путь выбирают train-наблюдения.
    # Сама петля не наблюдалась, поэтому варианты топологии задаются явно.
    paths = [[188726185, 188726176, 188726177, 488091029],
             [188726185, 188726180, 1199696832, 1232873046, 188726177, 488091029],
             [188726185, 188726180, 1199696832, 188726183, 488091029]]

    def complete(suffixes, prefixes):
        ba, a = max(suffixes, key=lambda z: B.arc_length(z[1])[-1])
        candidates = []
        for bb, b in prefixes:
            if np.dot(B.tangent(a, True), B.tangent(b, False)) > -.75:
                continue
            # только прямые участки конечной, общие с OSM, не подход по карте
            observed = np.vstack([a[B.arc_length(a) > B.arc_length(a)[-1]-70.], b[B.arc_length(b) < 80.]])
            for ids in paths:
                q, fit = fit_train_rigid(observed, osm_xyz(ways, ids))
                candidates.append((fit['rmse'], ba, a, bb, b, ids, q, fit))
        if not candidates:
            raise ValueError('No training return geometry supports public terminal topology')
        for score, _, _, bb, _, ids, _, fit in sorted(candidates, key=lambda c: c[0]):
            print('OSM train fit', bb, ids, json.dumps(fit), flush=True)
        if topology == 'inner':
            candidates = [c for c in candidates if 188726176 in c[5]]
        elif topology == 'outer':
            candidates = [c for c in candidates if 1199696832 in c[5]]
        # подгонки, различающиеся на миллиметры, не определяют невидимую петлю:
        # среди статистически равных берём кратчайший путь по OSM
        best = min(c[0] for c in candidates)
        candidates = [c for c in candidates if c[0] <= best+.05]
        _, ba, a, bb, b, ids, q, fit = min(candidates, key=lambda c: B.arc_length(c[6])[-1])
        if fit['rmse'] > 3. or abs(fit['rotation_deg']) > 5. or np.linalg.norm(fit['translation_at_origin']) > 10.:
            raise ValueError('Public geometry fails independent train-only alignment gate')
        r = B.Route('aligned_osm', q)
        s0, x0, _ = r.project(*a[-1, :2])
        s1, x1, _ = r.project(*b[0, :2])
        if not s0 < s1 or max(x0, x1) > 3.:
            raise ValueError('Public path does not connect observed endpoints in order')
        u = np.r_[s0, r.s[(r.s > s0)&(r.s < s1)], s1]
        bridge = np.column_stack([np.interp(u, r.s, q[:, k]) for k in range(3)])
        length = u[-1]-u[0]
        # остаточное смещение OSM плавно сводим к точным концам train,
        # только на 20 м у каждого стыка; саму петлю не деформируем
        w0 = np.clip(1.-(u-u[0])/20., 0., 1.)
        w1 = np.clip(1.-(u[-1]-u)/20., 0., 1.)
        bridge[:, :2] += w0[:, None]*(a[-1, :2]-bridge[0, :2]) + w1[:, None]*(b[0, :2]-bridge[-1, :2])
        bridge[:, 2] = np.interp(u, [s0, s1], [a[-1, 2], b[0, 2]])
        bridge[0], bridge[-1] = a[-1], b[0]
        info = dict(method='public_osm_train_rigid_alignment', from_bag=ba, to_bag=bb,
                    osm_way_ids=ids, osm_way_versions={str(wid): dict(version=ways[wid].get('version'), timestamp=ways[wid].get('timestamp')) for wid in ids},
                    topology_policy=topology, fit=fit, connector_length=float(B.arc_length(bridge)[-1]),
                    original_osm_connector_length=float(length), provenance=provenance)
        print('OSM selected connector', json.dumps(info), flush=True)
        prefix_info = extend_return_prefixes(prefixes, ways, fit, 'inner' if 188726176 in ids else 'outer') if extend_prefixes else []
        return [(ba+'+osm+'+bb, B.join(a, bridge, b))], [info]+prefix_info
    return complete


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', default=str(CACHE))
    parser.add_argument('--output', default=str(Path(B.R.REPO)/'dataset/cache/model_osm/routes.json'))
    parser.add_argument('--fetch-only', action='store_true')
    parser.add_argument('--topology', choices=['shortest', 'inner', 'outer'], default='shortest')
    parser.add_argument('--extend-prefixes', action='store_true', help='Add all train-matched public return-track prefixes')
    args = parser.parse_args(argv)
    data, provenance = fetch_osm(args.cache)
    print(json.dumps(provenance, indent=2))
    if args.fetch_only:
        return
    # нужен готовый файл разбиения, как в build_route: разбиение здесь не строим
    # и эталоны val для выбора геометрии не загружаем
    import csv
    split_file = Path(B.refdata.CACHE)/'split.csv'
    with split_file.open(newline='') as f:
        rows = list(csv.reader(f))
    train = [b for b, label in rows if label == 'train']
    if not train or set(train)&{b for b, label in rows if label == 'val'}:
        parser.error('Invalid fixed train split')
    missing = [b for b in train if not (Path(B.refdata.CACHE)/(b+'.npz')).exists()]
    if missing:
        parser.error('Train caches missing: '+', '.join(missing))
    original = B.complete_west
    try:
        B.complete_west = osm_completion(data, provenance, topology=args.topology, extend_prefixes=args.extend_prefixes)
        routes, inferred = B.build(train, complete=True)
    finally:
        B.complete_west = original
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    B.save_routes(routes, path=args.output, extra=dict(source='train+public_osm', builder='build_osm_route',
                  train_bags=train, inferred_connectors=inferred, osm_provenance=provenance))
    print('Saved', args.output, flush=True)


if __name__ == '__main__':
    main()
