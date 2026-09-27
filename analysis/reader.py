"""Быстрое чтение sqlite-файлов rosbag2 и геодезические функции.

Фиксированные смещения CDR (little-endian, выравнивание от байта 4, длина строки
включает завершающий ноль):

  header (все типы):                stamp.sec int32 @ 4, stamp.nanosec uint32 @ 8
  VelocitySensor (36 B):            velocity float64 @ 28   (единицы: км/ч)
  DriverControllerCommand (20 B):   position int8    @ 17
  NavSatFix (128 B):                status int8 @ 20, lat f64 @ 28, lon @ 36, alt @ 44
  TwistStamped (68 B):              linear xyz @ 20/28/36, angular xyz @ 44/52/60

Данные: $MOS_TRAP_DATA, по умолчанию <repo>/dataset/data.
"""
import glob
import hashlib
import json
import os
import sqlite3
import struct

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get('MOS_TRAP_DATA') or os.path.join(REPO, 'dataset', 'data')
MAP_DIR = os.path.join(REPO, 'dataset', 'map')
FIG_DIR = os.path.join(REPO, 'analysis', 'figures')

# скорости колёс публикуются в км/ч (подтвердили организаторы)
KMH_TO_MS = 1.0 / 3.6

# WGS84
R_EARTH = 6378137.0
WGS84_F = 1.0 / 298.257223563

# система карты и чекера: MGRS-квадрат 37U, то есть UTM 37N минус это смещение
# проверено: точки остановок по GNSS лежат в 0.1-4 м от центрлинии JSON
UTM_ZONE = 37
MGRS_ORIGIN = (300000.0, 6100000.0)

# положения антенн GNSS в base_link (м); base_link - ось передней тележки, z - верх рельса
ANTENNA_IN_BASE_LINK = {
    'master': (-9.873, 0.0, 3.0),
    'rover': (2.563, 0.0, 3.0),
}
BOGIE_PIVOT_DISTANCE = 7.55  # м, между вертикальными осями передней и задней тележек

TOPIC_FRONT = '/vehicle/front_bogie_velocity'
TOPIC_REAR = '/vehicle/rear_bogie_velocity'
TOPIC_DRIVER = '/vehicle/driver_position_cmd'
TOPIC_MASTER_FIX = '/sensing/gnss/master/fix'
TOPIC_MASTER_VEL = '/sensing/gnss/master/vel'
TOPIC_ROVER_FIX = '/sensing/gnss/rover/fix'
TOPIC_ROVER_VEL = '/sensing/gnss/rover/vel'


def bag_path(bag_id):
    return glob.glob(os.path.join(DATA_DIR, bag_id, '*.db3'))[0]


def all_bags():
    return sorted(os.path.basename(b) for b in glob.glob(os.path.join(DATA_DIR, '*'))
                  if glob.glob(os.path.join(b, '*.db3')))


def unique_bags():
    """Bag без побайтных дубликатов .db3 (в датасете 25 пар дубликатов).

    Возвращает (unique_ids, [(dup_id, kept_id), ...]).
    """
    seen, out, dup = {}, [], []
    for bn in all_bags():
        h = hashlib.md5()
        with open(bag_path(bn), 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
        h = h.hexdigest()
        if h in seen:
            dup.append((bn, seen[h]))
        else:
            seen[h] = bn
            out.append(bn)
    return out, dup


def read_topics(bag_id, want, stamp='header'):
    """want: dict топик -> тип разбора ('wheel', 'drv', 'fix', 'twist').

    stamp: 'header' -> header.stamp (согласован между топиками, использовать его),
           'bag'    -> messages.timestamp (время приёма рекордером).
    Возвращает dict топик -> (время, с, np.array; значения np.array).
    'wheel' - как есть, км/ч. GNSS - только валидные фиксы (status == 2, конечные).
    """
    f = bag_path(bag_id)
    con = sqlite3.connect(f)
    tmap = dict((n, i) for i, n in con.execute('SELECT id,name FROM topics'))
    out = dict((t, ([], [])) for t in want)
    targets = {}
    for t, kind in want.items():
        if t in tmap:
            targets[tmap[t]] = (t, kind)
    for tid, ts, blob in con.execute('SELECT topic_id,timestamp,data FROM messages ORDER BY timestamp'):
        hit = targets.get(tid)
        if hit is None:
            continue
        name, kind = hit
        b = bytes(blob)
        if kind == 'wheel':
            v = struct.unpack_from('<d', b, 28)[0]
        elif kind == 'drv':
            v = struct.unpack_from('<b', b, 17)[0]
        elif kind == 'fix':
            if struct.unpack_from('<b', b, 20)[0] != 2:
                continue
            v = struct.unpack_from('<3d', b, 28)
            if not np.all(np.isfinite(v)):
                continue
        elif kind == 'twist':
            v = struct.unpack_from('<6d', b, 20)
        else:
            raise ValueError(kind)
        if stamp == 'header':
            sec, nsec = struct.unpack_from('<iI', b, 4)
            t = sec + nsec * 1e-9
        else:
            t = ts * 1e-9
        out[name][0].append(t)
        out[name][1].append(v)
    con.close()
    res = {}
    for t, (ts, vs) in out.items():
        res[t] = (np.array(ts, float), np.array(vs, float))
    return res


def latlon_to_enu(lat, lon, lat0, lon0):
    """Локальные East/North (м) вокруг (lat0, lon0), равнопромежуточная проекция; точность ~1e-4 на 5 км."""
    k = R_EARTH * np.pi / 180.0
    x = (np.asarray(lon) - lon0) * k * np.cos(np.radians(lat0))
    y = (np.asarray(lat) - lat0) * k
    return x, y


def latlon_to_utm(lat, lon, zone=UTM_ZONE):
    """WGS84 -> UTM (северное полушарие), ряды Снайдера; < 1 мм в пределах +-3 град от осевого меридиана."""
    a, f, k0 = R_EARTH, WGS84_F, 0.9996
    e2 = f * (2 - f)
    ep2 = e2 / (1 - e2)
    phi = np.radians(np.asarray(lat, float))
    lam = np.radians(np.asarray(lon, float))
    lam0 = np.radians(zone * 6.0 - 183.0)
    s, c = np.sin(phi), np.cos(phi)
    N = a / np.sqrt(1 - e2 * s * s)
    T = np.tan(phi) ** 2
    C = ep2 * c * c
    A = (lam - lam0) * c
    e4, e6 = e2 * e2, e2 * e2 * e2
    M = a * ((1 - e2 / 4 - 3 * e4 / 64 - 5 * e6 / 256) * phi
             - (3 * e2 / 8 + 3 * e4 / 32 + 45 * e6 / 1024) * np.sin(2 * phi)
             + (15 * e4 / 256 + 45 * e6 / 1024) * np.sin(4 * phi)
             - (35 * e6 / 3072) * np.sin(6 * phi))
    x = k0 * N * (A + (1 - T + C) * A ** 3 / 6
                  + (5 - 18 * T + T * T + 72 * C - 58 * ep2) * A ** 5 / 120) + 500000.0
    y = k0 * (M + N * np.tan(phi) * (A * A / 2 + (5 - T + 9 * C + 4 * C * C) * A ** 4 / 24
                                     + (61 - 58 * T + T * T + 600 * C - 330 * ep2) * A ** 6 / 720))
    return x, y


def latlon_to_mgrs_local(lat, lon):
    """WGS84 -> плоские координаты MGRS-квадрата, как в JSON карты и у чекера."""
    x, y = latlon_to_utm(lat, lon)
    return x - MGRS_ORIGIN[0], y - MGRS_ORIGIN[1]


def load_map(name):
    """name: 'tallinskaya_shchukinskaya' или 'shchukinskaya_tallinskaya'.

    Возвращает массив (N, 5): x, y, z, tang, curv (x/y в MGRS local, м, шаг ~1 м).
    """
    with open(os.path.join(MAP_DIR, name + '.json'), encoding='utf-8') as f:
        d = json.load(f)
    return np.array([[q['x'], q['y'], q['z'], q['tang'], q['curv']] for q in d['points']])


def map_names():
    return sorted(os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(os.path.join(MAP_DIR, '*.json')))


def detect_stops(wts, wv, gts, gll, v_thr=0.4, min_dur=8.0, max_gap=1.0):
    """Список (lat, lon, длительность, с) стоянок, где колёса < v_thr (км/ч)."""
    n = min(60000, max(1000, len(wts) * 2))
    t = np.linspace(wts[0], wts[-1], n)
    wi = np.interp(t, wts, wv)
    m = wi < v_thr
    d = np.diff(np.r_[0, m.astype(np.int8), 0])
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    out = []
    for a, b in zip(starts, ends):
        dur = t[b - 1] - t[a]
        if dur < min_dur:
            continue
        tc = (t[a] + t[b - 1]) / 2.0
        j = np.searchsorted(gts, tc)
        if j <= 0 or j >= len(gts) or abs(gts[j] - tc) > max_gap:
            continue
        out.append((gll[j, 0], gll[j, 1], dur))
    return out


def cluster_points(lats, lons, radius_m=30.0):
    """Жадная пространственная кластеризация. Возвращает список (count, lat, lon, idx_array)."""
    lats = np.asarray(lats)
    lons = np.asarray(lons)
    used = np.zeros(len(lats), bool)
    clusters = []
    for i in np.argsort(lats):
        if used[i]:
            continue
        dx, dy = latlon_to_enu(lats, lons, lats[i], lons[i])
        grp = np.where((np.hypot(dx, dy) < radius_m) & (~used))[0]
        used[grp] = True
        clusters.append((len(grp), lats[grp].mean(), lons[grp].mean(), grp))
    clusters.sort(key=lambda z: -z[0])
    return clusters
