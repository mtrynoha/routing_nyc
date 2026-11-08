"""Матриці часу між зонами ядра для всіх 30-хв моментів виїзду (кві-тра 2024), де є поїздки:
static (вільний рух), snapshot (швидкості заморожені на момент виїзду), TD (IGP, фактичні швидкості під час поїздки).
Параметри невиміряних вулиць — з calibration_best.json (підібрані лише на квітні).
Запуск: python3 nyc_td_run.py <shard> <nshards>"""
import json, sys, time
import numpy as np, pandas as pd
from nyc_td import NET, SLOTS, speeds, vff_speeds, dijkstra_times, td_times

shard, nsh = int(sys.argv[1]), int(sys.argv[2])
best = json.load(open('calibration_best.json'))
SIDE = ('residential', 'tertiary', 'unclassified', 'living_street')
alpha = {c: best['a_side'] for c in SIDE}
for c in ('primary', 'secondary', 'motorway', 'trunk'):
    alpha[c] = best['a_main']
zs = json.load(open('zone_sources.json')); SRC = np.array(zs['SRC']); spos = {s: i for i, s in enumerate(SRC)}
src_of = {int(z): v for z, v in zs['src_of'].items()}
zn = {int(k): [NET.idx[n] for n in v] for k, v in json.load(open('zone_nodes.json')).items()}
zones = sorted(zn)


def zone_matrix(Tn):
    M = np.full((len(zones), len(zones)), np.nan)
    for a, o in enumerate(zones):
        rows = Tn[[spos[s] for s in src_of[o]]]
        for b, d in enumerate(zones):
            if o != d: M[a, b] = rows[:, zn[d]].mean() / 60
    return M


cells = pd.read_parquet('trip_cells.parquet')
ks = np.sort(cells.k.unique())[shard::nsh]
day = SLOTS[ks].normalize()
out = {'k': [], 'td': [], 'snap': []}
static = zone_matrix(dijkstra_times(vff_speeds(alpha), SRC))
tic = time.time()
for d in np.unique(day):
    kd = ks[day == d]
    for b0 in range(0, len(kd), 4):
        kb = kd[b0:b0 + 4]
        A = td_times(alpha, kb, SRC)
        for n, k in enumerate(kb):
            out['k'].append(int(k)); out['td'].append(zone_matrix(A[n]))
            out['snap'].append(zone_matrix(dijkstra_times(speeds(alpha, k, k + 1)[0], SRC)))
    print(f'[{shard}] {pd.Timestamp(d).date()} {len(out["k"])}/{len(ks)} {time.time() - tic:.0f}s', flush=True)
np.savez(f'zone_mats_shard{shard}.npz', k=np.array(out['k']), td=np.array(out['td']), snap=np.array(out['snap']), static=static,
         zones=np.array(zones))
print(f'[{shard}] saved', flush=True)
