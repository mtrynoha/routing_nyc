"""Перевірка матриць часу з прогнозу на реальних поїздках yellow taxi тестового періоду (19.05-31.05.2024).
Усі поправки — лише з КВІТНЯ: сталий додаток c (2.07 хв, калібрування), поправка пари зон c_pair, профіль поїздок HA і
профіль мережевої моделі для гібрида HA x (T(t) / T_профіль). Комірка = (30-хв момент виїзду, пара зон) з >= 5 поїздками."""
import json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

Z = [np.load(f'fc_mats_shard{s}.npz') for s in (0, 1)]
t = np.concatenate([z['t'] for z in Z]); o = np.argsort(t); t = t[o]
M = {p: np.concatenate([z[p] for z in Z])[o] for p in ('persist', 'dgcrn', 'oracle')}
zones = Z[0]['zones']; t0 = pd.Timestamp(str(Z[0]['t0'])); dep = t0 + pd.to_timedelta(5 * t, unit='min')
zi = {z: n for n, z in enumerate(zones)}
T = pd.read_parquet('core_trips.parquet')
T = T[(T.t0 >= dep[0]) & (T.t0 < dep[-1] + pd.Timedelta('30min'))].copy()
T['j'] = np.searchsorted(dep.values, T.t0.values, side='right') - 1
T['a'] = T.o.map(zi); T['b'] = T.d.map(zi)
cell = T.groupby(['j', 'a', 'b']).dur_min.agg(['median', 'size']).reset_index()
cell = cell[cell['size'] >= 5].copy()
for p in M:
    cell[p] = M[p][cell.j, cell.a, cell.b]
dd = dep[cell.j.values]; cell['hour'] = dd.hour; cell['wk'] = dd.dayofweek < 5
# поправки з квітня
best = json.load(open('calibration_best.json')); c0 = best['c_min']
ev = pd.read_parquet('eval_cells.parquet'); apr = ev[ev.month == 4]
cpair = (apr['median'] - apr.td).groupby([apr.a, apr.b]).median()
prof = apr.groupby(['a', 'b', 'wk', 'hour']).td.median()
Ta = pd.read_parquet('core_trips.parquet'); Ta = Ta[Ta.t0.dt.month == 4]
Ta['a'] = Ta.o.map(zi); Ta['b'] = Ta.d.map(zi); Ta['hour'] = Ta.t0.dt.hour; Ta['wk'] = Ta.t0.dt.dayofweek < 5
ha = Ta.groupby(['a', 'b', 'wk', 'hour']).dur_min.median()
key2 = pd.MultiIndex.from_arrays([cell.a, cell.b]); key4 = pd.MultiIndex.from_arrays([cell.a, cell.b, cell.wk, cell.hour])
cell['cpair'] = cpair.reindex(key2).fillna(c0).values
cell['ha'] = ha.reindex(key4).values; cell['prof'] = prof.reindex(key4).values
preds = {}
for p in M:
    preds[p] = cell[p] + c0
    preds[p + '+пара'] = cell[p] + cell.cpair
    preds['HA×' + p] = cell.ha * cell[p] / cell.prof
preds['HA'] = cell.ha


def score(mask):
    rows = {}
    y = cell['median'][mask]
    for name, p in preds.items():
        q = p[mask]; ok = q.notna()
        e = (q[ok] - y[ok]) / y[ok]
        rows[name] = dict(n=int(ok.sum()), MAPE=float(np.mean(np.abs(e)) * 100), bias=float(np.mean(e) * 100),
                          within20=float(np.mean(np.abs(e) < .2) * 100))
    return pd.DataFrame(rows).T.round(2)


R = {'cells': int(len(cell)), 'trips': int(cell['size'].sum()), 'departures': int(len(t))}
out = {}
for name, m in [('усі', np.ones(len(cell), bool)), ('будній пік 7-10, 16-19', cell.wk & cell.hour.isin([7, 8, 9, 16, 17, 18])),
                ('поїздки > 15 хв', cell['median'] > 15), ('поїздки > 20 хв', cell['median'] > 20)]:
    df = score(m.values if hasattr(m, 'values') else m); out[name] = df.to_dict(orient='index')
    print(f'\n=== {name} (комірок: {int(np.sum(m))}) ===\n{df.to_string()}')
R['results'] = out
# парне порівняння DGCRN vs persistence (бутстреп за днями)
y = cell['median']; ep = np.abs(preds['persist'] - y) / y; ed = np.abs(preds['dgcrn'] - y) / y; eo = np.abs(preds['oracle'] - y) / y
day = dep[cell.j.values].normalize(); days = np.unique(day); rng = np.random.default_rng(0)
dm = pd.DataFrame({'d': day, 'p': ep, 'g': ed, 'o': eo}).groupby('d').mean()
bs = [dm.iloc[rng.integers(0, len(dm), len(dm))].mean() for _ in range(2000)]
diff = np.array([(b.p - b.g) * 100 for b in bs]); diffo = np.array([(b.p - b.o) * 100 for b in bs])
R['dgcrn_vs_persist_MAPE_reduction_pp'] = [float((dm.p - dm.g).mean() * 100), float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]
R['oracle_vs_persist_MAPE_reduction_pp'] = [float((dm.p - dm.o).mean() * 100), float(np.percentile(diffo, 2.5)), float(np.percentile(diffo, 97.5))]
print('\nDGCRN vs persistence, MAPE reduction pp [95% CI over days]:', np.round(R['dgcrn_vs_persist_MAPE_reduction_pp'], 3))
print('oracle vs persistence:', np.round(R['oracle_vs_persist_MAPE_reduction_pp'], 3))
json.dump(R, open('nyc_forecast_eval.json', 'w'), indent=1, ensure_ascii=False, default=float)
cell.to_parquet('fc_eval_cells.parquet')
