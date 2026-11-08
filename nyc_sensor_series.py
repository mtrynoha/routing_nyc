"""Крок NYC-2а: 15-хв часові ряди швидкостей датчиків EZ Pass і NYC DOT (01.04-31.05.2024).
EZ Pass: кожен запис — медіана часу проїзду за попередні 900 с; відносимо його до слоту, що містить центр вікна (T - 7.5 хв),
і беремо медіану записів слоту. DOT: швидкість зі status=0, медіана в слоті.
Пропуски: до 1 год — лінійна інтерполяція; довші — профіль датчика «день тижня × слот»; маска obs зберігає справжні виміри."""
import glob, json
import numpy as np, pandas as pd

SLOT = '15min'
slots = pd.date_range('2024-04-01', '2024-06-01', freq=SLOT, inclusive='left')
K = len(slots)
VMIN, VMAX = 2.0, 110.0                          # км/год, поза межами — викид

ez_seg = pd.read_parquet('ez_seg_cov.parquet').set_index('sid')
cols = ['sid', 'median_calculation_timestamp', 'median_tt_sec', 'n_samples']
ez = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob('raw/ez/ez_2024-*.parquet'))])
ez = ez[(ez.n_samples > 0) & (ez.median_tt_sec > 0)]
ez['slot'] = (pd.to_datetime(ez.median_calculation_timestamp) - pd.Timedelta('450s')).dt.floor(SLOT)
ez['kmh'] = ez_seg.link_length_ft.reindex(ez.sid).values * 0.3048 / ez.median_tt_sec.values * 3.6
ez = ez[ez.kmh.between(VMIN, VMAX)]
ez_w = ez.groupby(['slot', 'sid']).kmh.median().unstack().reindex(slots)
ez_w.columns = [f'ez_{c}' for c in ez_w.columns]

dot = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob('raw/dot/dot_2024-*.parquet'))])
dot = dot[(dot.status == 0) & (dot.speed > 0)]
dot['slot'] = pd.to_datetime(dot.data_as_of).dt.floor(SLOT)
dot['kmh'] = dot.speed * 1.609344
dot = dot[dot.kmh.between(VMIN, VMAX)]
dot_w = dot.groupby(['slot', 'link_id']).kmh.median().unstack().reindex(slots)
dot_w.columns = [f'dot_{c}' for c in dot_w.columns]

W = pd.concat([ez_w, dot_w], axis=1)
obs = W.notna()
F = W.interpolate(limit=4, limit_area='inside')
key = slots.dayofweek * 96 + slots.hour * 4 + slots.minute // 15
prof = F.groupby(key).median()
F = F.fillna(prof.reindex(key).set_axis(F.index))
F = F.fillna(F.median())
F.to_parquet('sensor_speed_15min.parquet'); obs.to_parquet('sensor_obs_15min.parquet')
R = {'slots': K, 'sensors': W.shape[1], 'ez_sensors': ez_w.shape[1], 'dot_sensors': dot_w.shape[1],
     'observed_share_all': float(obs.values.mean()),
     'observed_share_ez': float(obs[ez_w.columns].values.mean()), 'observed_share_dot': float(obs[dot_w.columns].values.mean()),
     'sensors_all_nan': int(W.isna().all().sum())}
# профіль за годиною (будні) для правдоподібності
wk = slots.dayofweek < 5
R['ez_weekday_median_kmh_by_hour'] = F.loc[wk, ez_w.columns].median(axis=1).groupby(slots[wk].hour).median().round(1).tolist()
R['dot_weekday_median_kmh_by_hour'] = F.loc[wk, dot_w.columns].median(axis=1).groupby(slots[wk].hour).median().round(1).tolist()
json.dump(R, open('sensor_series.json', 'w'), indent=1)
print(json.dumps(R, indent=1))
