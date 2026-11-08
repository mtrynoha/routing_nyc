"""Крок NYC-2б: швидкість (і час проїзду) кожного ребра ядра Мідтауна в кожному 15-хв слоті.
  * виміряні ребра: швидкість прив'язаного датчика EZ Pass / DOT (рівномірна вздовж ділянки);
  * невиміряні ребра: v_e(k) = alpha_c * vff_c * CI_e(k), де
        vff_c  — швидкість вільного руху класу дороги (85-й перцентиль виміряних ребер класу; для бічних вулиць — від secondary),
        alpha_c — поправка класу (за замовчуванням 1; калібрується на поїздках таксі квітня),
        CI_e(k) — локальний індекс заторів: середнє v/vff виміряних ребер у радіусі 500 м, вага 1/(d+50 м).
Результат: nyc_core_graph.parquet (ребра), nyc_core_components.npz (складові для швидкостей)."""
import json
import numpy as np, pandas as pd, networkx as nx

R_NEIGH = 500.0
E = pd.read_csv('osm_edges_matched.csv')
N = pd.read_csv('raw/osm/manhattan_nodes.csv').set_index('osmid')
S = pd.read_parquet('sensor_speed_15min.parquet')
K = len(S)
E['cls'] = E.highway.astype(str).str.replace(r"[\[\]' ]", '', regex=True).str.split(',').str[0].str.replace('_link', '')
E = E[E.core].copy()
# паралельні ребра -> одне (найкоротше), найбільша сильно зв'язна компонента
E = E.sort_values('len').drop_duplicates(['u', 'v'])
G = nx.DiGraph(); G.add_edges_from(zip(E.u, E.v))
scc = max(nx.strongly_connected_components(G), key=len)
E = E[E.u.isin(scc) & E.v.isin(scc)].reset_index(drop=True)
E['col'] = np.where(E.src.notna(), E.src.astype(str) + '_' + E.sid.astype(str), '')
E['observed'] = E.col.isin(S.columns)
lat0 = 40.76; kx, ky = 111320 * np.cos(np.radians(lat0)), 110540
mx = ((N.x.reindex(E.u).values + N.x.reindex(E.v).values) / 2 + 73.97) * kx
my = ((N.y.reindex(E.u).values + N.y.reindex(E.v).values) / 2 - lat0) * ky
E['mx'], E['my'] = mx, my

O = np.nonzero(E.observed.values)[0]; U = np.nonzero(~E.observed.values)[0]
Vobs = S[E.col.values[O]].values.astype(np.float32)          # [K, O] км/год
vff_e = np.percentile(Vobs, 85, axis=0)
CIobs = np.clip(Vobs / vff_e, 0.05, 1.5)
# vff класів — за всіма виміряними ребрами Мангеттена (не лише ядра)
Ea = pd.read_csv('osm_edges_matched.csv')
Ea['cls'] = Ea.highway.astype(str).str.replace(r"[\[\]' ]", '', regex=True).str.split(',').str[0].str.replace('_link', '')
Ea['col'] = np.where(Ea.src.notna(), Ea.src.astype(str) + '_' + Ea.sid.astype(str), '')
Ea = Ea[Ea.col.isin(S.columns)]
vff_all = np.percentile(S[Ea.col.values].values, 85, axis=0)
cls_obs = pd.Series(vff_all, index=Ea.cls.values).groupby(level=0)
vff_cls = cls_obs.median()[cls_obs.size() >= 5].to_dict()
FALLBACK = {'trunk': 'motorway', 'tertiary': 'residential', 'unclassified': 'residential', 'living_street': 'residential'}
for c in E.cls.unique():
    if c not in vff_cls:
        vff_cls[c] = vff_cls.get(FALLBACK.get(c, 'secondary'), vff_cls.get('secondary'))
# сусіди для локального індексу заторів
d = np.hypot(mx[U][:, None] - mx[O][None, :], my[U][:, None] - my[O][None, :])
Wt = np.where(d <= R_NEIGH, 1.0 / (d + 50.0), 0.0)
noN = Wt.sum(1) == 0
Wt[noN] = 1.0                                                 # без сусідів -> глобальний індекс
Wt /= Wt.sum(1, keepdims=True)
CIu = (CIobs @ Wt.T).astype(np.float32)                      # [K, U]
vff_u = np.array([vff_cls[c] for c in E.cls.values[U]], dtype=np.float32)
np.savez_compressed('nyc_core_components.npz', O=O, U=U, Vobs=Vobs, CIu=CIu, vff_u=vff_u,
                    cls_u=E.cls.values[U].astype(str), slots=S.index.values.astype('datetime64[m]').astype(np.int64))
E.drop(columns=['geometry'], errors='ignore').to_parquet('nyc_core_graph.parquet')

def edge_speeds(alpha=None, k0=0, k1=None):
    """Швидкості ребер ядра [k1-k0, E] у м/с; alpha — dict клас -> поправка для невиміряних ребер."""
    k1 = K if k1 is None else k1
    V = np.empty((k1 - k0, len(E)), dtype=np.float32)
    V[:, O] = Vobs[k0:k1]
    a = np.array([(alpha or {}).get(c, 1.0) for c in E.cls.values[U]], dtype=np.float32)
    V[:, U] = CIu[k0:k1] * (vff_u * a)
    return np.maximum(V, 2.0) / 3.6

R = {'core_edges_scc': int(len(E)), 'nodes': len(scc), 'observed_edges': int(len(O)), 'unobserved_edges': int(len(U)),
     'observed_km_share': float(E.len[E.observed].sum() / E.len.sum()),
     'unobserved_without_neighbours': int(noN.sum()),
     'vff_class_kmh': {k: round(float(v), 1) for k, v in vff_cls.items()},
     'unobserved_by_class_km': E[~E.observed].groupby('cls').len.sum().div(1000).round(1).to_dict()}
V = edge_speeds() * 3.6
slots = S.index; wk = slots.dayofweek < 5
w = E.len.values / E.len.sum()
R['network_mean_speed_kmh_weekday_by_hour'] = pd.Series(V[wk] @ w).groupby(slots[wk].hour).mean().round(1).tolist()
json.dump(R, open('nyc_edge_times.json', 'w'), indent=1)
print(json.dumps(R, indent=1))
