"""Поля швидкостей (5 хв) для політик у момент планування t і перетворення «датчики -> ребра ядра Мідтауна».
  persist — x[t] заморожено;  ha — профіль «день тижня x 5 хв» навчального періоду (лише справжні виміри);
  dgcrn   — x[t], далі прогноз DGCRN на 5..60 хв, після 60 хв — профіль HA;
  oracle  — фактичні виміри (пропуски — x, заповнений без зазирання вперед)."""
import json
import numpy as np, pandas as pd
from nyc_td import NET, Eg

z = np.load('nyc_series.npz'); X = z['x'].astype(np.float64); Y = z['y'].astype(np.float64)
t0 = pd.Timestamp(str(z['t0'])); T5 = pd.date_range(t0, periods=len(X), freq='5min')
P = np.clip(np.load('dgcrn_nyc_pred.npy').astype(np.float64), 2, 110)
T_FIRST = 14047
TRUTH = np.where(Y > 0, Y, X)
n_train_end = round((len(X) - 23) * 0.7) + 11
key = (T5.dayofweek * 288 + T5.hour * 12 + T5.minute // 5).values
Ytr = np.where(Y[:n_train_end] > 0, Y[:n_train_end], np.nan)
HA = pd.DataFrame(Ytr).groupby(key[:n_train_end]).median().reindex(range(2016))
HA = HA.fillna(pd.DataFrame(Ytr).median()).values                       # [2016, 435]

sens = pd.read_csv('nyc_sensors.csv'); col = {s: j for j, s in enumerate(sens.sensor)}
best = json.load(open('calibration_best.json')); vffc = json.load(open('nyc_edge_times.json'))['vff_class_kmh']
alpha = {c: best['a_side'] for c in ('residential', 'tertiary', 'unclassified', 'living_street')}
alpha.update({c: best['a_main'] for c in ('primary', 'secondary', 'motorway', 'trunk')})
ecol = Eg.col.fillna('').values
obs = np.array([c in col for c in ecol]); O = np.nonzero(obs)[0]; U = np.nonzero(~obs)[0]
oj = np.array([col[c] for c in ecol[O]])
vff_s = np.percentile(X[:n_train_end], 85, axis=0)
d = np.hypot(Eg.mx.values[U][:, None] - Eg.mx.values[O][None], Eg.my.values[U][:, None] - Eg.my.values[O][None])
Wt = np.where(d <= 500, 1 / (d + 50), 0.0); Wt[Wt.sum(1) == 0] = 1.0; Wt /= Wt.sum(1, keepdims=True)
vff_u = np.array([vffc.get(c, vffc['secondary']) * alpha.get(c, 1.0) for c in Eg.cls.values[U]])


def edge_field(V):
    """V [w, 435] км/год -> швидкості ребер [w, nE] м/с (внутрішній порядок NET)."""
    Vo = V[:, oj]; Ve = np.empty((len(V), len(Eg)))
    Ve[:, O] = Vo; Ve[:, U] = (np.clip(Vo / vff_s[oj], 0.05, 1.5) @ Wt.T) * vff_u
    return (np.maximum(Ve, 2.0) / 3.6)[:, NET.order]


def sensor_field(policy, t, W):
    idx = np.minimum(np.arange(t, t + W), len(X) - 1)
    if policy == 'persist':
        return np.repeat(X[t][None], W, 0)
    if policy == 'oracle':
        return TRUTH[idx]
    if policy == 'ha':
        return HA[key[idx]]
    i = t - T_FIRST                                                    # dgcrn
    F = HA[key[idx]].copy(); F[0] = X[t]; F[1:13] = P[i].T
    return F
