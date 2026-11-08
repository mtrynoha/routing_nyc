"""Граф вулиць Мангеттена з OpenStreetMap (osmnx) з повторними спробами й резервними серверами Overpass.
Запуск (у папці data\\nyc):   python nyc_osm.py
Результат: osm\\manhattan_drive.graphml, а також osm\\manhattan_edges.csv / manhattan_nodes.csv (на випадок проблем з graphml).
"""
import time, traceback
from pathlib import Path

import osmnx as ox

OUT = Path(__file__).resolve().parent / 'osm'
OUT.mkdir(exist_ok=True)
ox.settings.cache_folder = str(Path(__file__).resolve().parent / 'cache')
ox.settings.use_cache = True
ox.settings.log_console = True
ox.settings.requests_timeout = 600       # osmnx >= 2.0
ox.settings.timeout = 600                # osmnx 1.x
ox.settings.overpass_rate_limit = False

ENDPOINTS = ['https://overpass-api.de/api', 'https://overpass.kumi.systems/api', 'https://overpass.private.coffee/api']
BBOX = (-74.0479, 40.6829, -73.9067, 40.8820)   # (west, south, east, north) — Мангеттен

import ssl, sys
print('python', sys.version.split()[0], '| osmnx', ox.__version__, '|', ssl.OPENSSL_VERSION)
if sys.version_info < (3, 9):
    print('УВАГА: Python < 3.9 зі старим OpenSSL не може встановити TLS-з\'єднання з серверами Overpass '
          '(помилка SSLV3_ALERT_HANDSHAKE_FAILURE). Запустіть скрипт новішим Python (3.10+).', flush=True)
G = None
for ep in ENDPOINTS:
    ox.settings.overpass_url = ep        # osmnx >= 2.0
    ox.settings.overpass_endpoint = ep   # osmnx 1.x (стара назва параметра)
    for how in ('place', 'bbox'):
        try:
            print(f'--- {ep}, спосіб: {how}', flush=True)
            if how == 'place':
                G = ox.graph_from_place('Manhattan, New York County, New York, USA', network_type='drive', simplify=True)
            else:
                try:
                    G = ox.graph_from_bbox(BBOX, network_type='drive', simplify=True)          # osmnx >= 2.0
                except TypeError:
                    G = ox.graph_from_bbox(BBOX[3], BBOX[1], BBOX[2], BBOX[0], network_type='drive', simplify=True)  # osmnx 1.x
            break
        except Exception:
            traceback.print_exc()
            time.sleep(15)
    if G is not None:
        break

if G is None:
    raise SystemExit('Не вдалося отримати граф з жодного сервера. Надішліть, будь ласка, текст помилок вище.')

print(f'вузлів {G.number_of_nodes():,}, ребер {G.number_of_edges():,}', flush=True)
ox.save_graphml(G, OUT / 'manhattan_drive.graphml')
nodes, edges = ox.graph_to_gdfs(G)
keep = [c for c in ['osmid', 'name', 'highway', 'oneway', 'lanes', 'maxspeed', 'length', 'geometry'] if c in edges.columns]
edges.reset_index()[['u', 'v', 'key'] + keep].to_csv(OUT / 'manhattan_edges.csv', index=False)
nodes.reset_index()[['osmid', 'y', 'x']].to_csv(OUT / 'manhattan_nodes.csv', index=False)
print('збережено в', OUT, flush=True)
