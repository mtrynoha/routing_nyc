"""Маршрутизація на графі ядра Мідтауна: статична, миттєва (snapshot) і часо-залежна (TD, модель IGP з td_core).
15-хв швидкості ребер розгортаються на 5-хв сітку td_core (той самий кусково-сталий профіль)."""
import json, sys
import numpy as np, pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
sys.path.insert(0, '/home/claude/stepA')
from td_core import TDWindow

C = dict(np.load('nyc_core_components.npz', allow_pickle=True))
Eg = pd.read_parquet('nyc_core_graph.parquet')
SLOTS = pd.to_datetime(C['slots'].astype('datetime64[m]'))
K = len(SLOTS)


class Net:
    """Інтерфейс, який очікує td_core.TDWindow: N, nE, ei, ej, L, eid; ребра відсортовано за кінцевим вузлом."""
    def __init__(self):
        self.osm = np.unique(np.r_[Eg.u.values, Eg.v.values])
        idx = {n: i for i, n in enumerate(self.osm)}
        ei = Eg.u.map(idx).values; ej = Eg.v.map(idx).values
        self.order = np.lexsort((ei, ej))                  # перестановка ребер Eg -> внутрішній порядок
        self.ei, self.ej = ei[self.order].astype(np.int64), ej[self.order].astype(np.int64)
        self.L = Eg.len.values[self.order].astype(np.float64)
        self.N, self.nE = len(self.osm), len(self.L)
        self.eid = -np.ones((self.N, self.N), dtype=np.int64); self.eid[self.ei, self.ej] = np.arange(self.nE)
        self.idx = idx


NET = Net()
O, U = C['O'], C['U']
cls_u = C['cls_u']


def speeds(alpha=None, k0=0, k1=None):
    """Швидкості ребер (м/с) [k1-k0, nE] у внутрішньому порядку ребер."""
    k1 = K if k1 is None else k1
    V = np.empty((k1 - k0, len(Eg)), dtype=np.float64)
    V[:, O] = C['Vobs'][k0:k1]
    a = np.array([(alpha or {}).get(c, 1.0) for c in cls_u])
    V[:, U] = C['CIu'][k0:k1] * (C['vff_u'] * a)
    return (np.maximum(V, 2.0) / 3.6)[:, NET.order]


def vff_speeds(alpha=None):
    """Статична швидкість вільного руху ребер (м/с): 85-й перцентиль виміряних / vff класу для невиміряних."""
    v = np.empty(len(Eg))
    v[O] = np.percentile(C['Vobs'], 85, axis=0)
    a = np.array([(alpha or {}).get(c, 1.0) for c in cls_u]); v[U] = C['vff_u'] * a
    return (v / 3.6)[NET.order]


def dijkstra_times(v, sources):
    G = csr_matrix((NET.L / v, (NET.ei, NET.ej)), shape=(NET.N, NET.N))
    return dijkstra(G, directed=True, indices=sources)


def td_times(alpha, k_dep, sources, margin_slots=8):
    """TD-час (с) від джерел до всіх вузлів для моментів виїзду k_dep (15-хв слоти, один день-вікно)."""
    k0 = int(min(k_dep)); k1 = min(int(max(k_dep)) + margin_slots + 1, K)
    V15 = speeds(alpha, k0, k1)
    V5 = np.repeat(V15, 3, axis=0)                          # 15 хв -> 3 x 5 хв
    win = TDWindow(NET, V5)
    dep = (np.asarray(k_dep) - k0) * 900.0
    A = win.all_pairs(dep, sources=sources)
    return A - dep[:, None, None]
