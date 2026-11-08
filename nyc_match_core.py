"""Полігон ядра Мідтауна: опукла оболонка ділянок EZ Pass сітки Мангеттена (без набережних, мостів і 72-ї вулиці), буфер 30 м."""
import numpy as np, pandas as pd
from shapely.geometry import MultiPoint

lat0 = 40.76; kx, ky = 111320 * np.cos(np.radians(lat0)), 110540


def _decode(s):
    pts, i, lat, lng = [], 0, 0, 0
    while i < len(s):
        for k in (0, 1):
            shift = res = 0
            while True:
                b = ord(s[i]) - 63; i += 1; res |= (b & 0x1f) << shift; shift += 5
                if b < 0x20: break
            d = ~(res >> 1) if res & 1 else res >> 1
            if k == 0: lat += d
            else: lng += d
        pts.append((lat / 1e5, lng / 1e5))
    return np.array(pts)


def core_polygon(path='ez_seg_cov.parquet'):
    ez = pd.read_parquet(path)
    pts = []
    for _, r in ez.iterrows():
        if r.borough != 'Manhattan' or not isinstance(r.polyline, str): continue
        if any(k in r.link_name for k in ('Henry Hudson', 'Bridge', 'Underpass', '72nd')): continue
        p = _decode(r.polyline)
        pts += list(zip((p[:, 1] + 73.97) * kx, (p[:, 0] - lat0) * ky))
    return MultiPoint(pts).convex_hull.buffer(30)
