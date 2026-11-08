"""Тури доставки в Мідтауні: часо-залежна задача комівояжера (депо + 12 зупинок, 4 хв обслуговування).
Для кожної політики: матриці TD-часу між точками туру для моментів виїзду кожні 5 хв на 4 год уперед (за її полем швидкостей),
план = найближчий сусід + 2-opt за її матрицями; реалізація плану — за матрицями оракула.
Запуск: python3 nyc_tours.py <shard> <nshards> <n_tours_total>"""
import sys, time, json
import numpy as np, pandas as pd
from nyc_td import NET
from nyc_fields import T5, T_FIRST, P, sensor_field, edge_field
sys.path.insert(0, '/home/claude/stepA')
from td_core import TDWindow

shard, nsh, NT = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
W, NSTOP, SERV = 48, 12, 240.0                      # 4 год поля, 12 зупинок, 4 хв обслуговування
POL = ['persist', 'ha', 'dgcrn', 'oracle']
rng = np.random.default_rng(2024)
test_t = T_FIRST + np.arange(P.shape[0])
ok_t = test_t[(T5[test_t].hour >= 7) & (T5[test_t].hour <= 15) & (test_t + W < len(T5))]
tours = [(int(rng.choice(ok_t)), rng.choice(NET.N, NSTOP + 1, replace=False)) for _ in range(NT)]


def td_matrices(policy, t, pts):
    """M[m, a, b] — TD-час (с) з точки a до b при виїзді в t + 5m хв, m = 0..W-1 (за полем політики в момент t)."""
    win = TDWindow(NET, edge_field(sensor_field(policy, t, W + 12)))
    dep = np.arange(W) * 300.0
    A = win.all_pairs(dep, sources=pts)
    return (A[:, :, pts] - dep[:, None, None])


def leg(M, a, b, tau):
    """Час переїзду a->b при виїзді в момент tau (с від t), лінійна інтерполяція між 5-хв точками."""
    m = min(tau / 300.0, M.shape[0] - 1.001); k = int(m); f = m - k
    return (1 - f) * M[k, a, b] + f * M[k + 1, a, b]


def simulate(M, order):
    """order — послідовність індексів зупинок (0 = депо на початку й у кінці). Повертає (тривалість, час прибуття до кожної зупинки)."""
    tau, arr, prev = 0.0, [], 0
    for s in order:
        tau += leg(M, prev, s, tau); arr.append(tau); tau += SERV; prev = s
    tau += leg(M, prev, 0, tau)
    return tau, np.array(arr)


def plan(M):
    left, order, tau, prev = list(range(1, NSTOP + 1)), [], 0.0, 0
    while left:                                     # найближчий сусід (TD)
        nxt = min(left, key=lambda s: leg(M, prev, s, tau))
        tau += leg(M, prev, nxt, tau) + SERV; order.append(nxt); left.remove(nxt); prev = nxt
    best = simulate(M, order)[0]; improved = True
    while improved:                                 # 2-opt + or-opt (перенесення 1-3 зупинок) з повною TD-переоцінкою
        improved = False
        for i in range(NSTOP - 1):
            for j in range(i + 1, NSTOP):
                cand = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
                c = simulate(M, cand)[0]
                if c < best - 1e-6:
                    order, best, improved = cand, c, True
        for k in (1, 2, 3):
            for i in range(NSTOP - k + 1):
                seg, rest = order[i:i + k], order[:i] + order[i + k:]
                for j in range(len(rest) + 1):
                    if j == i: continue
                    cand = rest[:j] + seg + rest[j:]
                    c = simulate(M, cand)[0]
                    if c < best - 1e-6:
                        order, best, improved = cand, c, True
                        break
    return order, best


rows = []
tic = time.time()
for n, (t, pts) in list(enumerate(tours))[shard::nsh]:
    Ms = {p: td_matrices(p, t, pts) for p in POL}
    Mo = Ms['oracle']
    opt_order, opt_real = plan(Mo)
    for p in POL:
        order, pred = plan(Ms[p])
        real, arr_real = simulate(Mo, order)
        _, arr_pred = simulate(Ms[p], order)
        rows.append(dict(tour=n, t=str(T5[t]), policy=p, pred_min=pred / 60, real_min=real / 60, opt_min=opt_real / 60,
                         eta_err_pct=(pred - real) / real * 100, regret_pct=(real - opt_real) / opt_real * 100,
                         stop_eta_mae_min=float(np.mean(np.abs(arr_pred - arr_real)) / 60),
                         stop_eta_last_err_min=float((arr_pred[-1] - arr_real[-1]) / 60)))
    print(f'[{shard}] tour {n} {T5[t]} real {real / 60:.0f} min, {time.time() - tic:.0f}s', flush=True)
pd.DataFrame(rows).to_csv(f'tours_shard{shard}.csv', index=False)
print(f'[{shard}] saved', flush=True)
