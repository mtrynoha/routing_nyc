"""Якість даних NYC: покриття EZ Pass / DOT і придатність поїздок TLC як еталону T_ij (Мангеттен)."""
import pandas as pd, numpy as np, glob, json
R = {}
# --- EZ Pass: 15-хв слоти, валідні = n_samples > 0
seg = pd.read_parquet('raw/ez/ez_segments.parquet')
ez = pd.concat([pd.read_parquet(f, columns=['sid', 'median_calculation_timestamp', 'median_tt_sec', 'n_samples']) for f in sorted(glob.glob('raw/ez/ez_2024-*.parquet'))])
ez['t'] = pd.to_datetime(ez.median_calculation_timestamp).dt.floor('15min')
ez = ez[ez.n_samples > 0]
slots = pd.date_range('2024-04-01', '2024-06-01', freq='15min', inclusive='left')
cov = ez.groupby('sid').t.nunique() / len(slots)
cov = cov.reindex(seg.sid, fill_value=0)
seg['cov'] = cov.values
R['ez_coverage_by_borough_median'] = seg.groupby('borough')['cov'].median().round(3).to_dict()
R['ez_segments_cov_ge_80pct'] = int((seg['cov'] >= .8).sum())
h = ez.assign(h=ez.t.dt.hour).groupby('h').sid.nunique()
ez['spd'] = seg.set_index('sid').link_length_ft.reindex(ez.sid).values * 0.3048 / ez.median_tt_sec * 3.6
R['ez_speed_kmh_p5_p50_p95'] = np.percentile(ez.spd[np.isfinite(ez.spd)], [5, 50, 95]).round(1).tolist()
# --- DOT: валідні = status 0 і speed > 0
dseg = pd.read_parquet('raw/dot/dot_segments.parquet')
dot = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob('raw/dot/dot_2024-*.parquet'))])
dot['t'] = pd.to_datetime(dot.data_as_of).dt.floor('5min')
v = dot[(dot.status == 0) & (dot.speed > 0)]
s5 = pd.date_range('2024-04-01', '2024-06-01', freq='5min', inclusive='left')
dcov = (v.groupby('link_id').t.nunique() / len(s5)).reindex(dseg.link_id, fill_value=0)
dseg['cov'] = dcov.values
R['dot_coverage_by_borough_median'] = dseg.groupby('borough')['cov'].median().round(3).to_dict()
R['dot_links_cov_ge_80pct'] = int((dseg['cov'] >= .8).sum())
R['dot_speed_mph_p5_p50_p95'] = np.percentile(v.speed, [5, 50, 95]).round(1).tolist()
# --- TLC yellow: поїздки всередині Мангеттена
z = pd.read_csv('raw/tlc/taxi_zone_lookup.csv'); man = set(z[z.Borough == 'Manhattan'].LocationID)
Y = pd.concat([pd.read_parquet(f, columns=['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'PULocationID', 'DOLocationID', 'trip_distance']) for f in sorted(glob.glob('raw/tlc/yellow_*.parquet'))])
Y['dur'] = (Y.tpep_dropoff_datetime - Y.tpep_pickup_datetime).dt.total_seconds() / 60
Y = Y[(Y.tpep_pickup_datetime >= '2024-04-01') & (Y.tpep_pickup_datetime < '2024-06-01')]
m = Y.PULocationID.isin(man) & Y.DOLocationID.isin(man) & (Y.PULocationID != Y.DOLocationID) & Y.dur.between(1, 120) & (Y.trip_distance > 0.1)
M = Y[m].copy()
R['yellow_trips_total'] = int(len(Y)); R['yellow_intra_manhattan_clean'] = int(len(M))
R['manhattan_zones'] = len(man)
M['slot'] = M.tpep_pickup_datetime.dt.floor('15min')
per = M.groupby(['slot', 'PULocationID', 'DOLocationID']).size()
R['od_slot_cells_with_ge5_trips'] = int((per >= 5).sum())
R['od_pairs_with_ge5_trips_per_15min_median_over_weekday_peak'] = None
wk = per.reset_index(name='n'); wk['h'] = wk.slot.dt.hour; wk['wd'] = wk.slot.dt.dayofweek < 5
pk = wk[wk.wd & wk.h.isin([8, 17])]
R['peak_slot_pairs_ge5_median'] = float(pk[pk.n >= 5].groupby('slot').size().median())
R['peak_slot_pairs_ge1_median'] = float(pk.groupby('slot').size().median())
R['trip_duration_min_p50_p90'] = np.percentile(M.dur, [50, 90]).round(1).tolist()
R['pairs_total_possible'] = len(man) * (len(man) - 1)
json.dump(R, open('nyc_quality.json', 'w'), indent=1, default=float)
print(json.dumps(R, indent=1, default=float))
seg.to_parquet('ez_seg_cov.parquet'); dseg.to_parquet('dot_seg_cov.parquet')
