"""Гібриди (параметри лише з квітня), перевірка на травні:
  td_pair  = TD + c_пари (медіанний залишок пари в квітні) — виправляє геометрію зон, посадку/висадку;
  ha_x_td  = HA * TD(t) / TD_профіль(пара, тип дня, година) — рівень з поїздок, динаміка з мережі."""
import json, numpy as np, pandas as pd
c = pd.read_parquet('eval_cells.parquet')
apr, may = c[c.month == 4].copy(), c[c.month == 5].copy()
cp = (apr['median'] - apr.td).groupby([apr.a, apr.b]).median().rename('cpair')
may = may.join(cp, on=['a', 'b'])
may['td_pair'] = may.td + may.cpair.fillna((apr['median'] - apr.td).median())
prof = apr.groupby(['a', 'b', 'wk', 'hour']).td.median().rename('tdprof')
may = may.join(prof, on=['a', 'b', 'wk', 'hour'])
may['ha_x_td'] = may.ha * may.td / may.tdprof
cg = (apr['median'] - apr.td).median()
rows = []
for m, off in [('static', None), ('snap', None), ('td', cg), ('ha', 0), ('td_pair', 0), ('ha_x_td', 0)]:
    if off is None: off = (apr['median'] - apr[m]).median()
    p = may[m] + off; y = may['median']; ok = p.notna()
    e = (p[ok] - y[ok]) / y[ok]
    wkm = may[ok & may.wk].assign(p=p)
    r = wkm.groupby(['a', 'b']).apply(lambda d: np.corrcoef(d.p, d['median'])[0, 1] if len(d) >= 10 and d.p.std() > 0 else np.nan).median()
    rows.append(dict(method=m, coverage=round(float(ok.mean()), 3), MAPE=round(float(np.mean(np.abs(e)) * 100), 2), bias_pct=round(float(np.mean(e) * 100), 2),
                     within_20pct=round(float(np.mean(np.abs(e) < .2) * 100), 1), corr_within_pair=round(float(r), 3)))
df = pd.DataFrame(rows).set_index('method'); print(df.to_string())
# пікові години будніх днів
pk = may[may.wk & may.hour.isin([8, 9, 16, 17, 18])]
print('\nweekday peak MAPE:', {m: round(float(np.mean(np.abs(pk[m] + (cg if m == 'td' else 0) - pk['median']) / pk['median']) * 100), 2) for m in ['td', 'ha', 'td_pair', 'ha_x_td']})
json.dump(df.to_dict(orient='index'), open('nyc_eval_hybrid.json', 'w'), indent=1)
