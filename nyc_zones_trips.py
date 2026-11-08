"""Зони таксі в ядрі Мідтауна, вузли графа в кожній зоні, поїздки yellow taxi між цими зонами (кві-тра 2024)."""
import json, glob
import numpy as np, pandas as pd, shapefile
from pyproj import Transformer
from shapely.geometry import Polygon, MultiPoint, Point, shape
from shapely.ops import unary_union

lat0 = 40.76; kx, ky = 111320 * np.cos(np.radians(lat0)), 110540
proj = lambda lon, lat: ((np.asarray(lon) + 73.97) * kx, (np.asarray(lat) - lat0) * ky)
tr = Transformer.from_crs('EPSG:2263', 'EPSG:4326', always_xy=True)

# ядро (як у nyc_match.py)
from nyc_match_core import core_polygon
core = core_polygon()

sf = shapefile.Reader('zones/taxi_zones/taxi_zones.shp')
zones = []
for sr in sf.shapeRecords():
    rec = sr.record.as_dict()
    if rec['borough'] != 'Manhattan':
        continue
    g = shape(sr.shape.__geo_interface__)
    polys = [g] if g.geom_type == 'Polygon' else list(g.geoms)
    pp = []
    for p in polys:
        lon, lat = tr.transform(*np.array(p.exterior.coords).T)
        x, y = proj(lon, lat); pp.append(Polygon(np.c_[x, y]))
    gz = unary_union(pp)
    zones.append(dict(zone=int(rec['LocationID']), name=rec['zone'], geom=gz, share=gz.intersection(core).area / gz.area))
Z = pd.DataFrame(zones)
Zc = Z[Z.share >= 0.6].copy()

Eg = pd.read_parquet('nyc_core_graph.parquet')
N = pd.read_csv('raw/osm/manhattan_nodes.csv').set_index('osmid')
nodes = np.unique(np.r_[Eg.u.values, Eg.v.values])
nx_, ny_ = proj(N.x.reindex(nodes).values, N.y.reindex(nodes).values)
zn = {}
for _, r in Zc.iterrows():
    inside = [int(n) for n, x, y in zip(nodes, nx_, ny_) if r.geom.contains(Point(x, y))]
    zn[int(r.zone)] = inside
Zc['n_nodes'] = Zc.zone.map(lambda z: len(zn[z]))
Zc = Zc[Zc.n_nodes >= 3]

# поїздки
cols = ['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'PULocationID', 'DOLocationID', 'trip_distance']
Y = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob('raw/tlc/yellow_*.parquet'))])
Y = Y[(Y.tpep_pickup_datetime >= '2024-04-01') & (Y.tpep_pickup_datetime < '2024-06-01')]
Y['dur_min'] = (Y.tpep_dropoff_datetime - Y.tpep_pickup_datetime).dt.total_seconds() / 60
zs = set(Zc.zone)
T = Y[Y.PULocationID.isin(zs) & Y.DOLocationID.isin(zs) & (Y.PULocationID != Y.DOLocationID)
      & Y.dur_min.between(1, 90) & Y.trip_distance.between(0.1, 15)].copy()
# фізично неможливі поїздки (середня швидкість > 60 км/год)
T = T[T.trip_distance * 1.609 / (T.dur_min / 60) < 60]
T = T.rename(columns={'tpep_pickup_datetime': 't0', 'PULocationID': 'o', 'DOLocationID': 'd'})[['t0', 'o', 'd', 'dur_min', 'trip_distance']]
T.to_parquet('core_trips.parquet')
json.dump({str(k): v for k, v in zn.items() if k in zs}, open('zone_nodes.json', 'w'))
Zc.drop(columns='geom').to_csv('core_zones.csv', index=False)
R = {'core_zones': int(len(Zc)), 'zones': Zc[['zone', 'name', 'n_nodes']].values.tolist(),
     'trips': int(len(T)), 'pairs_with_trips': int(T.groupby(['o', 'd']).ngroups),
     'trip_min_p10_p50_p90': np.percentile(T.dur_min, [10, 50, 90]).round(1).tolist(),
     'trip_miles_p50': float(T.trip_distance.median())}
json.dump(R, open('zones_trips.json', 'w'), indent=1, ensure_ascii=False)
print(json.dumps(R, indent=1, ensure_ascii=False))
