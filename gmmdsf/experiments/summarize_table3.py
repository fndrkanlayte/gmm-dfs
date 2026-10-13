"""Table 3 (benchmarks: Hangzhou 4x4 + Atlanta 1x5, heterogeneous layouts).

Whole-map headline G = (1/N) sum_z n_z U_z / ||w_z * FEATURE_SCALE|| — the same
scale-balanced objective as Table 2, but with ZONE_W_BENCH norms (benchmark
equity = waiting-only). U_z = mean_{i in z} z_i . w_i is read verbatim from
each row's zone_util_json (actual per-agent task weights of the run).

Output: ONE merged markdown table.
    Method | extra training per layout | <L> G (rank) | <L> delay | ...
rank = integer rank by mean G within each layout. G x100, mean +- std over
eval seeds. Also prints heterogeneity H = mean_i (1 - cos(w_i*s, w_T0*s)).

Usage: python summarize_table3.py [bench_results.csv] [--md table3_final.md]
"""
import os as _os, sys as _sys
_PKG = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _PKG)
_os.chdir(_PKG)  # all inputs/outputs are relative to gmmdsf/
import csv
import json
import sys
from collections import defaultdict

import numpy as np

from zones import ZONE_W_BENCH, bench_stats, bench_zone_of
from composite_reward import FEATURE_SCALE, DEFAULT_W

SRC = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else 'bench_results.csv'
MD = sys.argv[sys.argv.index('--md') + 1] if '--md' in sys.argv else None

LAYOUTS = {'HZA3': 'hangzhou_4x4_gudang_18041610_1h',
           'HZB3': 'hangzhou_4x4_gudang_18041610_1h',
           'ATL3': 'atlanta_1x5'}
NRM = {k: float(np.linalg.norm(np.asarray(v) * FEATURE_SCALE)) for k, v in ZONE_W_BENCH.items()}

# method key template -> (display name, extra-training-per-layout label)
ROWS = [('ft', 'FT', 'none'),
        ('mp', 'MaxPressure', 'none'),
        ('presslight', 'PressLight', 'shared (one net)'),
        ('mplight', 'MPLight', 'shared (one net)'),
        ('idqn_zone{L}', 'zone IDQN', 'per layout (500 ep)'),
        ('prlight_{L}', 'PRLight', 'per layout (200 ep adapt)'),
        ('mausfa', 'MA-USFA', 'none (zero-shot)'),
        ('ours', 'Ours zero-shot', 'none'),
        ('ours_gate2{L}', 'Ours + gated composer', 'per layout (composer)')]

S = np.asarray(FEATURE_SCALE, dtype=float)
W_T0 = np.asarray(DEFAULT_W, dtype=float)

# zone sizes + heterogeneity per layout
ZN, HET = {}, {}
for L, scn in LAYOUTS.items():
    zo = bench_zone_of(L)
    ZN[L] = {z: sum(v == z for v in zo.values()) for z in set(zo.values())}
    cos = []
    for ts, z in zo.items():
        w = np.asarray(ZONE_W_BENCH[z], dtype=float)
        cos.append(1 - float(np.dot(w * S, W_T0 * S) /
                             (np.linalg.norm(w * S) * np.linalg.norm(W_T0 * S))))
    HET[L] = float(np.mean(cos))

rows = [r for r in csv.DictReader(open(SRC)) if r['task'] in LAYOUTS]
G = defaultdict(lambda: defaultdict(list))          # method -> layout -> [G]
D = defaultdict(lambda: defaultdict(list))
for r in rows:
    L = r['task']
    U = json.loads(r['zone_util_json'])
    n = ZN[L]
    N = sum(n.values())
    G[r['method']][L].append(sum(n[z] * U[z] / NRM[z] for z in n) / N)
    D[r['method']][L].append(float(r['net_delay']))

for L in LAYOUTS:
    print(f'{L}: zones {ZN[L]} | N={sum(ZN[L].values())} | H={HET[L]:.3f}')

hdr = ['Method', 'extra training per layout']
for L in LAYOUTS:
    hdr += [f'{L} G x100 (rank)', f'{L} delay']
lines = ['| ' + ' | '.join(hdr) + ' |', '|' + '---|' * len(hdr)]

body = []
for key_t, name, extra in ROWS:
    cells = [name, extra]
    for L in LAYOUTS:
        key = key_t.format(L=L)
        gs = G[key].get(L, [])
        if gs:
            cells += [f'{100 * np.mean(gs):.2f} +- {100 * np.std(gs):.2f}',
                      f'{np.mean(D[key][L]):.1f}']
        else:
            cells += ['-', '-']
    body.append((key_t, cells))

# ranks per layout by mean G (desc)
for i, L in enumerate(LAYOUTS):
    scored = [(np.mean(G[k.format(L=L)][L]), j) for j, (k, c) in enumerate(body)
              if G[k.format(L=L)].get(L)]
    scored.sort(key=lambda x: -x[0])
    for rank, (_, j) in enumerate(scored, 1):
        body[j][1][2 + 2 * i] += f' ({rank})'

for _, cells in body:
    lines.append('| ' + ' | '.join(cells) + ' |')

out = '\n'.join(lines)
print(out)
if MD:
    with open(MD, 'w') as f:
        f.write(out + '\n')
    print(f'-> {MD}')
