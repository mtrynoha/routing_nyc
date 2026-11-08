"""Модель часу маршруту з затримками на вузлах (граф поворотів: вершина = ребро вулиці).
Час маршруту = сума часу проїзду ребер (швидкості з nyc_td.speeds, виміряні масштабовано на beta)
             + p_left / p_right за кожен лівий / правий поворот
             + d_sig на кожному перехресті (вузол степеня >= 3) в кінці НЕвиміряного ребра (світлофор бічної вулиці;
               для виміряних ребер EZ Pass/DOT затримки вже входять у виміряний час).
Розвороти заборонено (крім тупиків). Для коротких поїздок ядра TD ~= миттєвій матриці (див. nyc_eval), тож тут миттєва модель."""
import json
import numpy as np, pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from nyc_td import NET, Eg, speeds

N, E = NET.N, NET.nE
ei, ej, L = NET.ei, NET.ej, NET.L
obs = Eg.observed.values[NET.order]
nodes = pd.read_csv('raw/osm/manhattan_nodes.csv').set_index('osmid')
lat0 = 40.76; kx, ky = 111320 * np.cos(np.radians(lat0)), 110540
X = (nodes.x.reindex(NET.osm).values + 73.97) * kx; Yc = (nodes.y.reindex(NET.osm).values - lat0) * ky
brg = np.degrees(np.arctan2(Yc[ej] - Yc[ei], X[ej] - X[ei]))
nbr = [set() for _ in range(N)]
for a, b in zip(ei, ej): nbr[a].add(b); nbr[b].add(a)
junction = np.array([len(s) >= 3 for s in nbr])
out_by = [[] for _ in range(N)]
for e in range(E): out_by[ei[e]].append(e)
# переходи e1 -> e2 через вузол v = ej[e1]
T1, T2, TURN = [], [], []
for e1 in range(E):
    v = ej[e1]; outs = out_by[v]
    for e2 in outs:
        if ej[e2] == ei[e1] and len(outs) > 1: continue        # розворот (крім тупика)
        d = (brg[e2] - brg[e1] + 180) % 360 - 180               # >0 — ліворуч (проти год. стрілки)
        TURN.append(0 if abs(d) < 35 else (1 if d > 0 else 2))  # 0 прямо, 1 ліво, 2 право
        T1.append(e1); T2.append(e2)
T1, T2, TURN = np.array(T1), np.array(T2), np.array(TURN)
SIG = (~obs[T1]) & junction[ej[T1]]
print(f'turn graph: {E} edge-nodes, {len(T1)} transitions; left {np.mean(TURN == 1):.2f}, right {np.mean(TURN == 2):.2f}, '
      f'junction-on-unobserved {SIG.mean():.2f}')
zs = json.load(open('zone_sources.json')); SRC = np.array(zs['SRC']); spos = {s: n for n, s in enumerate(SRC)}
src_of = {int(k): v for k, v in zs['src_of'].items()}
zn = {int(k): [NET.idx[n] for n in v] for k, v in json.load(open('zone_nodes.json')).items()}
zones = sorted(zn)
order_head = np.argsort(ej, kind='stable'); starts = np.r_[0, np.nonzero(np.diff(ej[order_head]))[0] + 1]
heads = ej[order_head][starts]


def node_times(v, p_left, p_right, d_sig, beta):
    """Час (с) від кожного джерела SRC до кожного вузла: [S, N]."""
    v = v.copy(); v[obs] *= beta
    te = L / v
    w = te[T2] + np.where(TURN == 1, p_left, np.where(TURN == 2, p_right, 0.0)) + np.where(SIG, d_sig, 0.0)
    # вершини: 0..E-1 — ребра; E..E+S-1 — джерела (стартові вершини)
    S = len(SRC)
    rows = np.r_[T1, np.concatenate([[E + n] * len(out_by[s]) for n, s in enumerate(SRC)])]
    cols = np.r_[T2, np.concatenate([out_by[s] for s in SRC])]
    wts = np.r_[w, np.concatenate([te[out_by[s]] for s in SRC])]
    G = csr_matrix((wts + 1e-9, (rows, cols)), shape=(E + S, E + S))
    D = dijkstra(G, directed=True, indices=np.arange(E, E + S))[:, :E]   # прибуття в кінець ребра
    Tn = np.full((S, N), np.inf)
    Tn[:, heads] = np.minimum.reduceat(D[:, order_head], starts, axis=1)
    Tn[np.arange(S), SRC] = 0.0
    return Tn


def zone_matrix(Tn):
    M = np.full((len(zones), len(zones)), np.nan)
    for a, o in enumerate(zones):
        rows = Tn[[spos[s] for s in src_of[o]]]
        for b, d in enumerate(zones):
            if o != d: M[a, b] = rows[:, zn[d]].mean() / 60
    return M


def alpha_best():
    b = json.load(open('calibration_best.json'))
    a = {c: b['a_side'] for c in ('residential', 'tertiary', 'unclassified', 'living_street')}
    a.update({c: b['a_main'] for c in ('primary', 'secondary', 'motorway', 'trunk')})
    return a
