"""Перевірка матриць часу на реальних поїздках yellow taxi. Квітень — калібрування сталого додатку c для кожного методу;
ТРАВЕНЬ — незалежна перевірка. Комірка = (30-хв момент виїзду, пара зон) з >= 5 поїздками; еталон — медіана тривалості.
Методи: static (вільний рух), snapshot (миттєвий), TD (часо-залежний), HA (емпіричний профіль поїздок квітня:
медіана пари за типом дня (будній/вихідний) і годиною) — сильний «даних-орієнтований» орієнтир."""
import json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from nyc_td import SLOTS

Z = [np.load(f'zone_mats_shard{s}.npz') for s in (0, 1)]
k = np.concatenate([z['k'] for z in Z]); o = np.argsort(k); k = k[o]
TD = np.concatenate([z['td'] for z in Z])[o]; SN = np.concatenate([z['snap'] for z in Z])[o]; ST = Z[0]['static']
zones = Z[0]['zones']; kpos = {kk: i for i, kk in enumerate(k)}
cells = pd.read_parquet('trip_cells.parquet')
cells = cells[cells.k.isin(kpos)].copy()
i = cells.k.map(kpos).values; a = cells.a.values; b = cells.b.values
cells['td'] = TD[i, a, b]; cells['snap'] = SN[i, a, b]; cells['static'] = ST[a, b]
t = SLOTS[cells.k.values]
cells['month'] = t.month; cells['hour'] = t.hour; cells['wk'] = t.dayofweek < 5
# HA з квітневих поїздок
T = pd.read_parquet('core_trips.parquet'); zi = {z: n for n, z in enumerate(zones)}
T = T[T.t0.dt.month == 4]; T['a'] = T.o.map(zi); T['b'] = T.d.map(zi); T['hour'] = T.t0.dt.hour; T['wk'] = T.t0.dt.dayofweek < 5
ha = T.groupby(['a', 'b', 'wk', 'hour']).dur_min.median().rename('ha')
cells = cells.join(ha, on=['a', 'b', 'wk', 'hour'])
methods = ['static', 'snap', 'td', 'ha']
apr, may = cells[cells.month == 4], cells[cells.month == 5]
R = {'cells_april': int(len(apr)), 'cells_may': int(len(may)), 'trips_in_may_cells': int(may['size'].sum())}
rows = []
for m in methods:
    c = 0.0 if m == 'ha' else float(np.nanmedian(apr['median'] - apr[m]))
    p = may[m] + c; y = may['median']; ok = p.notna()
    e = (p[ok] - y[ok]) / y[ok]
    # чи відтворює метод зміни протягом дня: кореляція всередині пари (будні травня)
    wkm = may[ok & may.wk]
    r_within = wkm.groupby(['a', 'b']).apply(lambda d: np.corrcoef(d[m], d['median'])[0, 1] if len(d) >= 10 and d[m].std() > 0 else np.nan).median()
    rows.append(dict(method=m, c_min=round(c, 2), coverage=float(ok.mean()), MAPE=float(np.mean(np.abs(e)) * 100),
                     bias_pct=float(np.mean(e) * 100), within_20pct=float(np.mean(np.abs(e) < .2) * 100),
                     corr_within_pair_weekday=float(r_within)))
df = pd.DataFrame(rows).set_index('method').round(3)
print(df.to_string()); R['may'] = df.to_dict(orient='index')
# за годиною (будні травня)
wk = may[may.wk]
byh = pd.DataFrame({m: wk.groupby('hour').apply(lambda d: np.mean(np.abs(d[m] + (0 if m == 'ha' else R['may'][m]['c_min']) - d['median']) / d['median']) * 100)
                    for m in methods}).round(2)
print('\nMay weekdays, MAPE by hour:\n', byh.to_string()); R['may_weekday_mape_by_hour'] = byh.to_dict()
json.dump(R, open('nyc_eval.json', 'w'), indent=1, default=float)
cells.to_parquet('eval_cells.parquet')

# рисунки
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
col = {'static': '#9c6ade', 'snap': '#c05621', 'td': '#2f855a', 'ha': '#718096'}
lbl = {'static': 'вільний рух (статично)', 'snap': 'миттєва (snapshot)', 'td': 'часо-залежна (TD)', 'ha': 'профіль поїздок квітня (HA)'}
ls = {'static': '--', 'snap': '-', 'td': '-', 'ha': ':'}
for m in methods:
    ax[0].plot(byh.index, byh[m], ls[m], marker='o', ms=3, lw=2, color=col[m], label=lbl[m])
ax[0].set(xlabel='година виїзду', ylabel='MAPE, %', title='Травень, будні: похибка медіани поїздки'); ax[0].legend(fontsize=8, frameon=False)
# медіана по годинах для однієї популярної пари
top = may.groupby(['a', 'b'])['size'].sum().idxmax(); d = may[(may.a == top[0]) & (may.b == top[1]) & may.wk].groupby('hour')
ax[1].plot(d['median'].median().index, d['median'].median(), 'k-', lw=2.5, label='реальні поїздки (медіана)')
for m in ['td', 'snap', 'static']:
    ax[1].plot(d[m].median().index, d[m].median() + R['may'][m]['c_min'], ls[m], color=col[m], lw=2, label=lbl[m])
ax[1].set(xlabel='година виїзду', ylabel='хв', title=f'Пара зон {zones[top[0]]}→{zones[top[1]]}, будні травня'); ax[1].legend(fontsize=8, frameon=False)
s = may.sample(min(4000, len(may)), random_state=0)
ax[2].scatter(s['median'], s['td'] + R['may']['td']['c_min'], s=4, alpha=.3, color=col['td'])
lim = [0, np.percentile(may['median'], 99)]; ax[2].plot(lim, lim, 'k--', lw=1)
ax[2].set(xlim=lim, ylim=lim, xlabel='реальна медіана, хв', ylabel='TD-прогноз, хв', title='Травень: TD проти реальних поїздок')
for x in ax:
    x.grid(color='#e5e5e5', lw=.6); x.set_axisbelow(True)
    for sp in ('top', 'right'): x.spines[sp].set_visible(False)
plt.tight_layout(); plt.savefig('nyc_eval.png', dpi=150)
print('saved')
