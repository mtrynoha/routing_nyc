"""Набір NYC для DGCRN (формат DCRNN): 453 датчики EZ Pass + NYC DOT, крок 5 хв, 01.04-31.05.2024.
  * y — сирі виміри (0 = пропуск, маскується у функції втрат, як у METR-LA);
  * x — пропуски заповнено БЕЗ зазирання в майбутнє: останнє значення до 1 год, далі профіль «день тижня × слот»
        з навчального періоду (перші 70%), далі середнє навчального періоду. Так модель не отримує нулів на вхід;
  * канал 1 — час доби в [0, 1).
Вихід (компактний, для перенесення на ПК): nyc_series.npz, adj_mx_nyc.pkl, nyc_sensors.csv."""
import glob, pickle, json
import numpy as np, pandas as pd

STEP = '5min'
t_index = pd.date_range('2024-04-01', '2024-06-01', freq=STEP, inclusive='left')
VMIN, VMAX = 2.0, 110.0
ez_seg = pd.read_parquet('ez_seg_cov.parquet').set_index('sid')
ez = pd.concat([pd.read_parquet(f, columns=['sid', 'median_calculation_timestamp', 'median_tt_sec', 'n_samples'])
                for f in sorted(glob.glob('raw/ez/ez_2024-*.parquet'))])
ez = ez[(ez.n_samples > 0) & (ez.median_tt_sec > 0)]
ez['slot'] = (pd.to_datetime(ez.median_calculation_timestamp) - pd.Timedelta('450s')).dt.floor(STEP)
ez['kmh'] = ez_seg.link_length_ft.reindex(ez.sid).values * 0.3048 / ez.median_tt_sec.values * 3.6
ez = ez[ez.kmh.between(VMIN, VMAX)]
ez_w = ez.groupby(['slot', 'sid']).kmh.median().unstack().reindex(t_index); ez_w.columns = [f'ez_{c}' for c in ez_w.columns]
dot = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob('raw/dot/dot_2024-*.parquet'))])
dot = dot[(dot.status == 0) & (dot.speed > 0)]
dot['slot'] = pd.to_datetime(dot.data_as_of).dt.floor(STEP); dot['kmh'] = dot.speed * 1.609344
dot = dot[dot.kmh.between(VMIN, VMAX)]
dot_w = dot.groupby(['slot', 'link_id']).kmh.median().unstack().reindex(t_index); dot_w.columns = [f'dot_{c}' for c in dot_w.columns]
W = pd.concat([ez_w, dot_w], axis=1)
T = len(W)
n_samples = T - 23; n_train = round(n_samples * 0.7)
t_train_end = n_train + 11                          # останній індекс часу, що входить у навчальні входи
# прибрати датчики з < 5% вимірів у НАВЧАЛЬНОМУ періоді (інакше нема з чого будувати профіль) або загалом
keep = (W.iloc[:t_train_end + 1].notna().mean() >= 0.05) & (W.notna().mean() >= 0.05)
W = W.loc[:, keep]
T, N = W.shape
train = W.iloc[:t_train_end + 1]
key = W.index.dayofweek * 288 + W.index.hour * 12 + W.index.minute // 5
prof = train.groupby(key[:t_train_end + 1]).median()
X = W.ffill(limit=12)
X = X.fillna(prof.reindex(key).set_axis(X.index)).fillna(train.mean()).fillna(float(np.nanmean(train.values)))
assert not X.isna().values.any()
Y = W.fillna(0.0)
tod = ((W.index.hour * 60 + W.index.minute) / 1440.0).values.astype(np.float32)

# координати середини ділянки датчика (для графа)
def decode(s):
    pts, i, lat, lng = [], 0, 0, 0
    while i < len(s):
        for k in (0, 1):
            shift = res = 0
            while True:
                b = ord(s[i]) - 63; i += 1; res |= (b & 0x1f) << shift; shift += 5
                if b < 0x20: break
            d = ~(res >> 1) if res & 1 else res >> 1
            if k == 0: lat += d
            else: lng += d
        pts.append((lat / 1e5, lng / 1e5))
    return np.array(pts)
dseg = pd.read_parquet('dot_seg_cov.parquet').set_index('link_id')
meta = []
for c in W.columns:
    src, sid = c.split('_'); sid = int(sid)
    if src == 'ez':
        p = decode(ez_seg.at[sid, 'polyline']); bor = ez_seg.at[sid, 'borough']; name = ez_seg.at[sid, 'link_name']
    else:
        pts = []
        for xy in str(dseg.at[sid, 'link_points']).split():
            try:
                la, lo = map(float, xy.split(','))
                if 40.45 < la < 40.95 and -74.3 < lo < -73.65: pts.append((la, lo))
            except ValueError: pass
        p = np.array(pts); bor = dseg.at[sid, 'borough']; name = dseg.at[sid, 'link_name']
    m = p[len(p) // 2]
    meta.append(dict(sensor=c, source=src, id=sid, borough=bor, name=name, lat=m[0], lon=m[1],
                     observed_share=float(W[c].notna().mean())))
M = pd.DataFrame(meta)
# граф: гаусове ядро відстані між серединами ділянок (як для NE-BJ у DGCRN), поріг k=0.1
lat, lon = np.radians(M.lat.values), np.radians(M.lon.values)
D = 2 * 6371.0 * np.arcsin(np.sqrt(np.sin((lat[:, None] - lat[None]) / 2) ** 2 +
                                   np.cos(lat[:, None]) * np.cos(lat[None]) * np.sin((lon[:, None] - lon[None]) / 2) ** 2))
sigma = 1.5                                         # км; поріг 0.1 -> радіус ~2.3 км
A = np.exp(-(D / sigma) ** 2); A[A < 0.1] = 0.0
ids = list(M.sensor); pickle.dump([ids, {s: i for i, s in enumerate(ids)}, A.astype(np.float32)], open('adj_mx_nyc.pkl', 'wb'), protocol=2)
np.savez_compressed('nyc_series.npz', x=X.values.astype(np.float16), y=Y.values.astype(np.float16), tod=tod,
                    t0=np.array(str(W.index[0])), step_min=np.array(5))
M.to_csv('nyc_sensors.csv', index=False)
R = dict(T=int(T), N=int(N), samples=int(n_samples), train=int(n_train), val=int(round(n_samples * 0.1)),
         test=int(n_samples - n_train - round(n_samples * 0.1)), observed_share=float(W.notna().values.mean()),
         test_start=str(W.index[n_train + round(n_samples * 0.1) + 11]), adj_density=float((A > 0).mean()),
         neighbours_median=float(np.median((A > 0).sum(1))), sigma_km=float(sigma),
         speed_mean_train=float(train.stack().mean()))
json.dump(R, open('nyc_dgcrn_data.json', 'w'), indent=1); print(json.dumps(R, indent=1))
