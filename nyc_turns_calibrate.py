"""Калібрування затримок на вузлах на поїздках КВІТНЯ (кожні 4 год), перевірка на ТРАВНІ (усі 30-хв моменти).
Параметри: p_left, p_right (с за поворот), d_sig (с на перехресті в кінці невиміряного ребра), beta (масштаб виміряних швидкостей).
Сталий додаток c — медіанний залишок квітня для кожного набору параметрів."""
import itertools, json
import numpy as np, pandas as pd
from nyc_turns import node_times, zone_matrix, alpha_best
from nyc_td import speeds, SLOTS

A = alpha_best()
cells = pd.read_parquet('trip_cells.parquet')
mon = SLOTS[cells.k].month
apr, may = cells[mon == 4], cells[mon == 5]
ks_apr = np.sort(apr.k.unique())[::8]
Acal = apr[apr.k.isin(ks_apr)]
V = {k: speeds(A, k, k + 1)[0] for k in np.unique(np.r_[ks_apr, may.k.unique()])}


def predict(params, ks):
    return {k: zone_matrix(node_times(V[k], *params)) for k in ks}


def evaluate(P, df, c=None):
    pred = np.array([P[k][a, b] for k, a, b in zip(df.k, df.a, df.b)])
    y = df['median'].values
    if c is None: c = float(np.median(y - pred))
    e = (pred + c - y) / y
    long = y > 15
    return dict(c=c, MAPE=float(np.mean(np.abs(e)) * 100), bias=float(np.mean(e) * 100),
                bias_long15=float(np.mean(e[long]) * 100), MAPE_long15=float(np.mean(np.abs(e[long])) * 100)), pred


res = []
grid = list(itertools.product([0, 20, 40, 60], [0, 15], [0, 15, 30], [0.85, 1.0]))
for g in grid:
    r, _ = evaluate(predict(g, ks_apr), Acal)
    res.append(dict(p_left=g[0], p_right=g[1], d_sig=g[2], beta=g[3], **r)); print(res[-1], flush=True)
R = pd.DataFrame(res).sort_values('MAPE'); R.to_csv('turns_grid.csv', index=False)
best = R.iloc[0]; g_best = (best.p_left, best.p_right, best.d_sig, best.beta)
print('BEST', best.to_dict(), flush=True)
# травень: базова модель (без штрафів) проти відкаліброваної, c — з квітня
out = {}
for name, g in [('без затримок', (0, 0, 0, 1.0)), ('з затримками', g_best)]:
    c_apr, _ = evaluate(predict(g, ks_apr), Acal)
    r, pred = evaluate(predict(g, may.k.unique()), may, c=c_apr['c'])
    out[name] = dict(params=[float(x) for x in g], **r)
    print('MAY', name, out[name], flush=True)
    may = may.assign(**{f'pred_{"base" if g == (0, 0, 0, 1.0) else "turns"}': pred + c_apr['c']})
# зміщення за тривалістю (травень)
bins = pd.cut(may['median'], [0, 5, 10, 15, 20, 30, 90])
tab = pd.DataFrame({m: ((may[f'pred_{m}'] - may['median']) / may['median']).groupby(bins, observed=True).mean() * 100 for m in ('base', 'turns')}).round(2)
tab['cells'] = may.groupby(bins, observed=True).size()
print('\nMay bias by trip duration (%):\n', tab.to_string())
out['bias_by_duration'] = tab.set_axis(tab.index.astype(str)).to_dict()
json.dump(out, open('turns_result.json', 'w'), indent=1, ensure_ascii=False, default=float)
may.to_parquet('turns_may_cells.parquet')
