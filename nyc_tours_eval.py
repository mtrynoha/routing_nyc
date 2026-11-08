"""Аналіз турів доставки: похибка ETA (тур, зупинки), програш порядку відносно найкращого відомого плану.
Бутстреп-ДІ (2000 повторів) за днями виїзду (кластери), парні різниці політик відносно persistence."""
import json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

D = pd.concat([pd.read_csv(f'tours_shard{s}.csv') for s in (0, 1)], ignore_index=True)
D['t'] = pd.to_datetime(D.t); D['day'] = D.t.dt.normalize(); D['hour'] = D.t.dt.hour
ref = D.groupby('tour').real_min.min().clip(upper=D.groupby('tour').opt_min.first())   # найкращий відомий план
D['ref_min'] = D.tour.map(ref)
D['regret'] = (D.real_min - D.ref_min) / D.ref_min * 100
D['abs_eta'] = D.eta_err_pct.abs()
D['abs_last'] = D.stop_eta_last_err_min.abs()
POL = ['persist', 'ha', 'dgcrn', 'oracle']
NAME = {'persist': 'Persistence', 'ha': 'Профіль HA', 'dgcrn': 'DGCRN (1 год) + HA', 'oracle': 'Оракул'}
MET = {'abs_eta': '|похибка тривалості туру|, %', 'eta_err_pct': 'зміщення тривалості, %',
       'stop_eta_mae_min': 'MAE часу прибуття на зупинки, хв', 'abs_last': '|похибка ETA останньої зупинки|, хв',
       'regret': 'програш порядку, %'}
rng = np.random.default_rng(0)
days = np.sort(D.day.unique())


def table(sub):
    W = {m: sub.pivot(index='tour', columns='policy', values=m) for m in MET}
    tday = sub.groupby('tour').day.first()
    out = {}
    nd = tday.nunique(); B = [rng.integers(0, nd, nd) for _ in range(2000)]
    for m, Wm in W.items():
        g = Wm.groupby(tday).agg(['sum', 'count'])
        row = {}
        for p in POL:
            s, c = g[(p, 'sum')].values, g[(p, 'count')].values
            row[p] = float(s.sum() / c.sum())
            if p in ('ha', 'dgcrn'):
                d = (g[('persist', 'sum')] - g[(p, 'sum')]).values
                bs = [d[b].sum() / c[b].sum() for b in B]
                row[f'{p}_vs_persist'] = [float(d.sum() / c.sum()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        d = (g[('ha', 'sum')] - g[('dgcrn', 'sum')]).values; c = g[('ha', 'count')].values
        bs = [d[b].sum() / c[b].sum() for b in B]
        row['dgcrn_vs_ha'] = [float(d.sum() / c.sum()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        out[m] = row
    return out


R = {'tours': int(D.tour.nunique()), 'days': int(len(days)),
     'tour_min_median': float(D[D.policy == 'oracle'].real_min.median()),
     'tour_min_p10_p90': [float(x) for x in D[D.policy == 'oracle'].real_min.quantile([.1, .9])]}
R['all'] = table(D)
R['by_start'] = {}
for name, m in [('7-9 год (ранковий пік)', D.hour.isin([7, 8, 9])), ('10-12 год', D.hour.isin([10, 11, 12])),
                ('13-15 год (тур захоплює вечірній пік)', D.hour.isin([13, 14, 15]))]:
    if D[m].tour.nunique() < 2: continue
    R['by_start'][name] = dict(tours=int(D[m].tour.nunique()), **table(D[m]))
json.dump(R, open('nyc_tours_eval.json', 'w'), indent=1, ensure_ascii=False)


def show(T, title):
    print(f'\n=== {title} ===')
    rows = []
    for m, lab in MET.items():
        r = T[m]
        f = lambda v: f'{v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]'
        rows.append([lab] + [f'{r[p]:.2f}' for p in POL] + [f(r['dgcrn_vs_persist']), f(r['ha_vs_persist']), f(r['dgcrn_vs_ha'])])
    print(pd.DataFrame(rows, columns=['метрика'] + [NAME[p] for p in POL] + ['persist−DGCRN', 'persist−HA', 'HA−DGCRN']).to_string(index=False))


print(f"тури: {R['tours']}, днів: {R['days']}, медіана тривалості {R['tour_min_median']:.0f} хв, p10-p90 {R['tour_min_p10_p90']}")
show(R['all'], 'усі тури')
for k, v in R['by_start'].items():
    show(v, f"{k}, турів {v['tours']}")

# графік: похибка ETA на зупинках і програш порядку
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
col = {'persist': '#8c8c8c', 'ha': '#4c72b0', 'dgcrn': '#dd8452', 'oracle': '#55a868'}
for j, (m, lab) in enumerate([('abs_eta', '|похибка тривалості туру|, %'), ('stop_eta_mae_min', 'MAE ETA зупинок, хв'), ('regret', 'програш порядку, %')]):
    groups = ['усі'] + list(R['by_start'])
    x = np.arange(len(groups))
    for i, p in enumerate(POL[:3] if m != 'regret' else POL):
        vals = [R['all'][m][p]] + [R['by_start'][g][m][p] for g in R['by_start']]
        ax[j].bar(x + (i - 1.5) * 0.2, vals, 0.2, color=col[p], label=NAME[p])
    ax[j].set_xticks(x); ax[j].set_xticklabels(['усі'] + [g.split(' ')[0] for g in R['by_start']], fontsize=9)
    ax[j].set_title(lab, fontsize=10); ax[j].set_xlabel('година виїзду', fontsize=9); ax[j].grid(axis='y', alpha=.3)
ax[2].legend(fontsize=8, loc='lower right')
plt.tight_layout(); plt.savefig('nyc_tours_eval.png', dpi=150)
D.to_parquet('tours_all.parquet')
