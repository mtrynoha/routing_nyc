"""Матриці часу між зонами Мідтауна на ТЕСТОВОМУ періоді DGCRN (19.05-31.05.2024), крок полів 5 хв.
Політики (поле швидкостей датчиків від моменту виїзду t, слоти w = 0..W-1 по 5 хв):
  persist — x[t] для всіх w (миттєва матриця);
  dgcrn   — w=0: x[t]; w=1..12: прогноз DGCRN на горизонт w; w>12: прогноз на 60 хв;
  oracle  — фактичні виміри y[t+w] (пропуски — x, заповнений без зазирання вперед).
Датчики -> ребра: як у nyc_edge_times.py (виміряні ребра — швидкість датчика; інші — alpha*vff_класу*локальний індекс заторів).
Запуск: python3 nyc_forecast_routing.py <shard> <nshards>"""
import json, sys, time
import numpy as np, pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from nyc_td import NET, Eg
sys.path.insert(0, '/home/claude/stepA')
from td_batch import TDBatch

shard, nsh = int(sys.argv[1]), int(sys.argv[2])
W = 36
z = np.load('nyc_series.npz'); X = z['x'].astype(np.float64); Y = z['y'].astype(np.float64)
t0 = pd.Timestamp(str(z['t0'])); T5 = pd.date_range(t0, periods=len(X), freq='5min')
P = np.clip(np.load('dgcrn_nyc_pred.npy').astype(np.float64), 2, 110)       # [S, 435, 12]
T_FIRST = 14047                                                             # t першого тестового зразка
sens = pd.read_csv('nyc_sensors.csv'); col = {s: j for j, s in enumerate(sens.sensor)}
TRUTH = np.where(Y > 0, Y, X)
n_train_end = round((len(X) - 23) * 0.7) + 11

# --- датчики -> ребра (у порядку Eg), потім у внутрішній порядок NET
best = json.load(open('calibration_best.json')); vffc = json.load(open('nyc_edge_times.json'))['vff_class_kmh']
SIDE = ('residential', 'tertiary', 'unclassified', 'living_street')
alpha = {c: best['a_side'] for c in SIDE}; alpha.update({c: best['a_main'] for c in ('primary', 'secondary', 'motorway', 'trunk')})
ecol = Eg.col.fillna('').values
obs = np.array([c in col for c in ecol]); O = np.nonzero(obs)[0]; U = np.nonzero(~obs)[0]
oj = np.array([col[c] for c in ecol[O]])
vff_s = np.percentile(X[:n_train_end], 85, axis=0)                         # vff датчика (навчальний період)
d = np.hypot(Eg.mx.values[U][:, None] - Eg.mx.values[O][None], Eg.my.values[U][:, None] - Eg.my.values[O][None])
Wt = np.where(d <= 500, 1 / (d + 50), 0.0); Wt[Wt.sum(1) == 0] = 1.0; Wt /= Wt.sum(1, keepdims=True)
vff_u = np.array([vffc.get(c, vffc['secondary']) * alpha.get(c, 1.0) for c in Eg.cls.values[U]])


def edge_field(V):
    """V [w, 435] км/год (датчики) -> швидкості ребер [w, nE] м/с у внутрішньому порядку NET."""
    Vo = V[:, oj]
    Ve = np.empty((len(V), len(Eg)))
    Ve[:, O] = Vo
    Ve[:, U] = (np.clip(Vo / vff_s[oj], 0.05, 1.5) @ Wt.T) * vff_u
    return (np.maximum(Ve, 2.0) / 3.6)[:, NET.order]


def field(policy, t):
    i = t - T_FIRST
    if policy == 'persist':
        return np.repeat(X[t][None], W, 0)
    if policy == 'oracle':
        return TRUTH[np.minimum(np.arange(t, t + W), len(X) - 1)]
    F = np.empty((W, X.shape[1])); F[0] = X[t]; F[1:13] = P[i].T; F[13:] = P[i, :, -1]
    return F


zs = json.load(open('zone_sources.json')); SRC = np.array(zs['SRC']); spos = {s: n for n, s in enumerate(SRC)}
src_of = {int(k): v for k, v in zs['src_of'].items()}
zn = {int(k): [NET.idx[n] for n in v] for k, v in json.load(open('zone_nodes.json')).items()}
zones = sorted(zn)


def zone_matrix(Tn):
    M = np.full((len(zones), len(zones)), np.nan)
    for a, o in enumerate(zones):
        rows = Tn[[spos[s] for s in src_of[o]]]
        for b, dd in enumerate(zones):
            if o != dd: M[a, b] = rows[:, zn[dd]].mean() / 60
    return M


def td_sources(tb, src):
    """TDBatch.all_pairs, обмежений джерелами src: повертає [B, S, N] (с від виїзду)."""
    net = NET; B, N = tb.B, net.N
    A = np.full((B, len(src), N), np.inf); A[:, np.arange(len(src)), src] = 0.0
    ei, ej = net.ei, net.ej; bofs = (np.arange(B) * tb.E)[:, None, None]; active = np.ones(N, bool)
    for _ in range(1000):
        em = active[ei]
        if not em.any(): return A
        idx = np.nonzero(em)[0]; tau = A[:, :, ei[idx]]
        cand = tb.arrive(np.broadcast_to(bofs + idx[None, None, :], tau.shape), tau)
        cand = np.where(np.isnan(cand), np.inf, cand)
        tgt = ej[idx]; starts = np.r_[0, np.nonzero(np.diff(tgt))[0] + 1]; nodes = tgt[starts]
        best_ = np.minimum.reduceat(cand, starts, axis=2); cur = A[:, :, nodes]; imp = best_ < cur - 1e-9
        A[:, :, nodes] = np.where(imp, best_, cur)
        active = np.zeros(N, bool); active[nodes[imp.any(axis=(0, 1))]] = True
    raise RuntimeError('no convergence')


test_t = T_FIRST + np.arange(P.shape[0])
deps = test_t[(T5[test_t].minute % 30) == 0][shard::nsh]
out = {'t': [], 'persist': [], 'dgcrn': [], 'oracle': []}
tic = time.time()
for b0 in range(0, len(deps), 4):
    tb_ = deps[b0:b0 + 4]
    for pol in ('dgcrn', 'oracle'):
        tb = TDBatch(NET, np.stack([edge_field(field(pol, t)) for t in tb_]))
        A = td_sources(tb, SRC)
        out[pol] += [zone_matrix(A[n]) for n in range(len(tb_))]
    for t in tb_:
        v = edge_field(field('persist', t)[:1])[0]
        G = csr_matrix((NET.L / v, (NET.ei, NET.ej)), shape=(NET.N, NET.N))
        out['persist'].append(zone_matrix(dijkstra(G, directed=True, indices=SRC))); out['t'].append(int(t))
    if (b0 // 4) % 10 == 0:
        print(f'[{shard}] {len(out["t"])}/{len(deps)} {time.time() - tic:.0f}s', flush=True)
np.savez(f'fc_mats_shard{shard}.npz', t=np.array(out['t']), persist=np.array(out['persist']), dgcrn=np.array(out['dgcrn']),
         oracle=np.array(out['oracle']), zones=np.array(zones), t0=str(t0))
print(f'[{shard}] saved {time.time() - tic:.0f}s', flush=True)
