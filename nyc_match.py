"""Прив'язка ділянок EZ Pass і ланок NYC DOT до напрямлених ребер графа OSM Мангеттена.
Ребро u->v вважається спостережуваним датчиком s, якщо більшість контрольних точок ребра (25/50/75% довжини)
лежать не далі 15 м від полілінії s і напрямок руху збігається (різниця азимутів < 35°)."""
import json
import numpy as np, pandas as pd, networkx as nx
from shapely import wkt
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

LAT0 = 40.76
KX, KY = 111320 * np.cos(np.radians(LAT0)), 110540
proj = lambda lon, lat: ((np.asarray(lon) + 73.97) * KX, (np.asarray(lat) - LAT0) * KY)
MAXD, MAXANG = 15.0, 35.0


def decode(s):
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


def dot_points(s):
    out = []
    for xy in str(s).split():
        try:
            la, lo = map(float, xy.split(','))
            if 40.45 < la < 40.95 and -74.3 < lo < -73.65: out.append((la, lo))
        except ValueError:
            pass
    p = np.array(out)
    if len(p) > 1:                                      # прибрати точки, що «стрибають»
        jump = np.r_[0, np.hypot(np.diff(p[:, 0]) * 111, np.diff(p[:, 1]) * 84)]
        p = max(np.split(p, np.nonzero(jump > 1.5)[0]), key=len)
    return p


# --- датчики -> сегменти з азимутом
segs, meta = [], []
ez = pd.read_parquet('ez_seg_cov.parquet'); dot = pd.read_parquet('dot_seg_cov.parquet')
sensors = []
for _, r in ez.iterrows():
    if isinstance(r.polyline, str): sensors.append(('ez', int(r.sid), decode(r.polyline), r.link_length_ft * .3048, r['cov'], r.borough, r.link_name))
for _, r in dot.iterrows():
    if isinstance(r.link_points, str): sensors.append(('dot', int(r.link_id), dot_points(r.link_points), np.nan, r['cov'], r.borough, r.link_name))
for src, sid, p, L, cov, bor, name in sensors:
    if len(p) < 2: continue
    x, y = proj(p[:, 1], p[:, 0])
    for k in range(len(x) - 1):
        if (x[k + 1] - x[k]) ** 2 + (y[k + 1] - y[k]) ** 2 < 1: continue
        segs.append(LineString([(x[k], y[k]), (x[k + 1], y[k + 1])]))
        meta.append((src, sid, np.degrees(np.arctan2(y[k + 1] - y[k], x[k + 1] - x[k]))))
tree = STRtree(segs)
meta = np.array(meta, dtype=object)
print('sensor pieces:', len(segs))

# --- ребра OSM
E = pd.read_csv('raw/osm/manhattan_edges.csv')
N = pd.read_csv('raw/osm/manhattan_nodes.csv').set_index('osmid')
geoms = []
for _, r in E.iterrows():
    if isinstance(r.geometry, str):
        g = wkt.loads(r.geometry); c = np.array(g.coords)
    else:
        c = np.array([[N.at[r.u, 'x'], N.at[r.u, 'y']], [N.at[r.v, 'x'], N.at[r.v, 'y']]])
    x, y = proj(c[:, 0], c[:, 1]); geoms.append(LineString(np.c_[x, y]))
E['src'] = None; E['sid'] = -1
for i, g in enumerate(geoms):
    fr = [0.5] if g.length < 20 else [0.25, 0.5, 0.75]
    votes = {}
    for f in fr:
        a, b = g.interpolate(max(f - .05, 0), normalized=True), g.interpolate(min(f + .05, 1), normalized=True)
        ang = np.degrees(np.arctan2(b.y - a.y, b.x - a.x))
        pt = g.interpolate(f, normalized=True)
        for j in tree.query(pt.buffer(MAXD)):
            if segs[j].distance(pt) > MAXD: continue
            d = abs((meta[j][2] - ang + 180) % 360 - 180)
            if d < MAXANG:
                key = (meta[j][0], meta[j][1]); votes[key] = votes.get(key, 0) + 1
    if votes:
        (src, sid), n = max(votes.items(), key=lambda kv: kv[1])
        if n >= (1 if len(fr) == 1 else 2):
            E.at[i, 'src'] = src; E.at[i, 'sid'] = sid
E['len'] = E['length']

# --- статистика
R = {'edges': int(len(E)), 'km_total': float(E.len.sum() / 1000)}
for s in ('ez', 'dot'):
    m = E.src == s
    R[f'{s}_edges'] = int(m.sum()); R[f'{s}_km'] = float(E.len[m].sum() / 1000)
    R[f'{s}_sensors_matched'] = int(E.sid[m].nunique())
obs = E.src.notna()
R['observed_share_length_manhattan'] = float(E.len[obs].sum() / E.len.sum())
# узгодженість: сумарна довжина ребер, прив'язаних до ділянки EZ, проти довжини ділянки
ezL = ez.set_index('sid').link_length_ft * .3048
ml = E[E.src == 'ez'].groupby('sid').len.sum()
ratio = (ml / ezL.reindex(ml.index)).dropna()
R['ez_matched_length_ratio_median_p10_p90'] = [float(x) for x in np.percentile(ratio, [50, 10, 90])]
# ядро Мідтауна: опукла оболонка ділянок EZ Мангеттена
from shapely.geometry import MultiPoint
pts = []
for src, sid, p, L, cov, bor, name in sensors:
    if src == 'ez' and bor == 'Manhattan' and not any(k in name for k in ('Henry Hudson', 'Bridge', 'Underpass', '72nd')):
        x, y = proj(p[:, 1], p[:, 0]); pts += list(zip(x, y))
core = MultiPoint(pts).convex_hull.buffer(30)
mid = np.array([g.interpolate(.5, normalized=True).within(core) for g in geoms])
E['core'] = mid
R['core_edges'] = int(mid.sum()); R['core_km'] = float(E.len[mid].sum() / 1000)
R['core_observed_share_length'] = float(E.len[mid & obs].sum() / E.len[mid].sum())
R['core_observed_share_edges'] = float((mid & obs).sum() / mid.sum())
by_hw = E[mid].assign(o=obs[mid]).groupby(E.highway[mid].astype(str).str.strip("[]'").str.split("'").str[0]).apply(lambda d: pd.Series({'km': d.len.sum() / 1000, 'obs_share': (d.len * d.o).sum() / d.len.sum()}))
R['core_by_highway'] = by_hw.round(3).to_dict(orient='index')
G = nx.MultiDiGraph(); G.add_edges_from(zip(E.u[mid], E.v[mid]))
Go = nx.MultiDiGraph(); Go.add_edges_from(zip(E.u[mid & obs], E.v[mid & obs]))
R['core_nodes'] = G.number_of_nodes()
R['core_scc_largest'] = len(max(nx.strongly_connected_components(G), key=len))
R['core_observed_subgraph_scc_largest'] = len(max(nx.strongly_connected_components(Go), key=len)) if Go.number_of_nodes() else 0
# частка довжини найкоротших (за відстанню) маршрутів між вузлами ядра, що проходить спостережуваними ребрами
Gd = nx.DiGraph()
for u, v, L, o in zip(E.u[mid], E.v[mid], E.len[mid], obs[mid]):
    if not Gd.has_edge(u, v) or Gd[u][v]['length'] > L: Gd.add_edge(u, v, length=L, obs=bool(o))
scc = max(nx.strongly_connected_components(Gd), key=len); Gs = Gd.subgraph(scc)
rng = np.random.default_rng(0); nodes = np.array(list(scc)); shares = []; lens = []
for s_ in rng.choice(nodes, 60, replace=False):
    paths = nx.single_source_dijkstra_path(Gs, s_, weight='length')
    for t_ in rng.choice(nodes, 50, replace=False):
        if t_ == s_ or t_ not in paths: continue
        p = paths[t_]; L = [Gs[a][b]['length'] for a, b in zip(p[:-1], p[1:])]; O = [Gs[a][b]['obs'] for a, b in zip(p[:-1], p[1:])]
        shares.append(np.dot(L, O) / sum(L)); lens.append(sum(L))
shares = np.array(shares)
R['core_route_observed_share_median'] = float(np.median(shares)); R['core_route_share_ge50pct'] = float(np.mean(shares >= .5))
R['core_route_km_median'] = float(np.median(lens) / 1000)
json.dump(R, open('nyc_match.json', 'w'), indent=1)
print(json.dumps(R, indent=1))
E.drop(columns=['geometry']).to_csv('osm_edges_matched.csv', index=False)

# --- рисунок
fig, ax = plt.subplots(1, 2, figsize=(12, 9))
col = {None: '#cfcfcf', 'ez': '#2f855a', 'dot': '#c05621'}
for a, bb, title in [(ax[0], None, 'Мангеттен'), (ax[1], core.bounds, 'Ядро: сітка Мідтауна з вимірюваннями EZ Pass')]:
    for g, s in zip(geoms, E.src):
        x, y = g.xy
        a.plot(x, y, color=col[s], lw=0.5 if s is None else 1.6, zorder=1 if s is None else 2)
    if bb:
        a.set_xlim(bb[0] - 150, bb[2] + 150); a.set_ylim(bb[1] - 150, bb[3] + 150)
        cx, cy = core.exterior.xy; a.plot(cx, cy, '--', color='#555', lw=1)
    a.set_aspect('equal'); a.set_xticks([]); a.set_yticks([]); a.set_title(title)
ax[0].plot([], [], color=col['ez'], lw=2, label='виміряно EZ Pass'); ax[0].plot([], [], color=col['dot'], lw=2, label='виміряно NYC DOT')
ax[0].plot([], [], color=col[None], lw=2, label='без вимірювань'); ax[0].legend(loc='upper left', frameon=False)
plt.tight_layout(); plt.savefig('nyc_observed_edges.png', dpi=140)
print('saved')
