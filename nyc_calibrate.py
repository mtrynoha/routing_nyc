"""Калібрування на поїздках yellow taxi КВІТНЯ (травень не використовується):
  alpha_side — множник швидкості невиміряних бічних вулиць (residential, tertiary, unclassified, living_street),
  alpha_main — множник невиміряних основних доріг,
  c          — сталий додаток (хв): посадка/висадка, розташування точок усередині зони.
Прогноз пари зон = середній час між вибраними вузлами зони відправлення та всіма вузлами зони прибуття."""
import json
import numpy as np, pandas as pd
from nyc_td import NET, SLOTS, speeds, dijkstra_times

SIDE = ('residential', 'tertiary', 'unclassified', 'living_street')
rng = np.random.default_rng(0)
zn = {int(k): [NET.idx[n] for n in v] for k, v in json.load(open('zone_nodes.json')).items()}
zones = sorted(zn)
src_of = {z: list(rng.choice(v, min(6, len(v)), replace=False)) for z, v in zn.items()}
SRC = np.array(sorted({s for v in src_of.values() for s in v})); spos = {s: i for i, s in enumerate(SRC)}
json.dump({'SRC': SRC.tolist(), 'src_of': {str(z): [int(s) for s in v] for z, v in src_of.items()}}, open('zone_sources.json', 'w'))


def zone_matrix(Tnodes):
    """[n_src, N] (с) -> [Z, Z] (хв): середнє по джерелах зони o і всіх вузлах зони d."""
    M = np.full((len(zones), len(zones)), np.nan)
    for a, o in enumerate(zones):
        rows = Tnodes[[spos[s] for s in src_of[o]]]
        for b, d in enumerate(zones):
            if o != d: M[a, b] = rows[:, zn[d]].mean() / 60
    return M


T = pd.read_parquet('core_trips.parquet')
T['k'] = ((T.t0 - SLOTS[0]).dt.total_seconds() // 1800).astype(int) * 2   # 30-хв сітка, індекс у 15-хв слотах
zi = {z: i for i, z in enumerate(zones)}
T['a'] = T.o.map(zi); T['b'] = T.d.map(zi)
cell = T.groupby(['k', 'a', 'b']).dur_min.agg(['median', 'size']).reset_index()
cell = cell[cell['size'] >= 5]
cell.to_parquet('trip_cells.parquet')
apr = cell[SLOTS[cell.k].month == 4]
ks = np.sort(apr.k.unique())[::4]                                  # кожні 2 год квітня
A = apr[apr.k.isin(ks)]
print('April calibration cells:', len(A), 'departures:', len(ks))


def predict(alpha, ks):
    out = {}
    for k in ks:
        out[k] = zone_matrix(dijkstra_times(speeds(alpha, k, k + 1)[0], SRC))
    return out


res = []
for a_side in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
    for a_main in [0.6, 0.8, 1.0, 1.2]:
        alpha = {c: a_side for c in SIDE}
        for c in ('primary', 'secondary', 'motorway', 'trunk'):
            alpha[c] = a_main
        P = predict(alpha, ks)
        pred = np.array([P[k][a, b] for k, a, b in zip(A.k, A.a, A.b)])
        c = np.median(A['median'].values - pred)
        mape = np.mean(np.abs(pred + c - A['median'].values) / A['median'].values) * 100
        res.append(dict(a_side=a_side, a_main=a_main, c_min=c, mape=mape, bias=np.mean(pred + c - A['median'].values)))
        print(res[-1], flush=True)
R = pd.DataFrame(res).sort_values('mape')
R.to_csv('calibration_grid.csv', index=False)
best = R.iloc[0].to_dict()
json.dump(best, open('calibration_best.json', 'w'), indent=1)
print('BEST', best)
