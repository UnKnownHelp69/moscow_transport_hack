"""Проекция WGS84 -> UTM / MGRS local (numpy, отличие от pyproj < 1 мм в пределах зоны)."""
import numpy as np

R_EARTH = 6378137.0
WGS84_F = 1.0 / 298.257223563
UTM_ZONE = 37
# система карты и чекера: MGRS-квадрат 37U = UTM 37N минус это смещение
MGRS_ORIGIN = (300000.0, 6100000.0)


def latlon_to_utm(lat, lon, zone=UTM_ZONE):
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
    x, y = latlon_to_utm(lat, lon)
    return x - MGRS_ORIGIN[0], y - MGRS_ORIGIN[1]
