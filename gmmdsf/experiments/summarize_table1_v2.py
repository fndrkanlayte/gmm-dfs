"""Aggregate table1_v2.csv -> mean +- std over seeds (Part A / Part B)."""
import os as _os, sys as _sys
_PKG = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _PKG)
_os.chdir(_PKG)  # all inputs/outputs are relative to gmmdsf/
import csv
import sys
from collections import defaultdict

import numpy as np

rows = list(csv.DictReader(open('table1_v2.csv')))
g = defaultdict(list)
for r in rows:
    g[(r['part'], r['task'], r['method'])].append(r)


def ms(rs, k, fmt):
    v = [float(r[k]) for r in rs if r[k] not in ('', 'None', 'nan')]
    if not v:
        return '-'
    return (fmt % np.mean(v)) + ('±' + fmt % np.std(v) if len(v) > 1 else '') + f'({len(v)})'


ORDER = ['ft', 'mp', 'presslight', 'mplight', 'idqn_mlp_T0', 'idqn_gat_T0', 'idqn_mlp_T1spec', 'idqn_gat_T2spec', 'plight_T0', 'plight_T1spec', 'plight_T2spec', 'prlight_T1', 'prlight_T2', 'prlight_A', 'prlight_B',
         'idqn_zoneA', 'idqn_zoneB', 'sf', 'tfh_blind', 'tfh', 'tfh_compT0', 'tfh_compA', 'tfh_compB', 'tfz_blind', 'tfz', 'tfz_ts1_blind', 'tfz_ts1', 'tfz_ts2_blind', 'tfz_ts2', 'tfc_blind', 'tfc', 'tfd_blind', 'tfd', 'tfz_compT0', 'tfc_compT0', 'tfc_normT0', 'tfc_compT1', 'tfc_normT1', 'tfc_compT2', 'tfc_normT2', 'tfz_compA', 'tfz_compB',
         'tfz100_blind', 'tfz100', 'tfz150_blind', 'tfz150', 'tfz200_blind', 'tfz200', 'tfz250_blind', 'tfz250', 'tfz300_blind', 'tfz300', 'tfz_comp']
methods = sorted({k[2] for k in g}, key=lambda m: ORDER.index(m) if m in ORDER else 99)

A_COLS = [('return', '%.1f'), ('net_delay', '%.1f'), ('throughput', '%.0f'),
          ('travel_time', '%.1f'), ('avg_queue', '%.2f'), ('avg_wait', '%.2f'),
          ('avg_speed', '%.2f')]
B_COLS = [('return', '%.1f'), ('net_delay', '%.1f'), ('corridor_speed_art', '%.2f'),
          ('corridor_wait_mean', '%.1f'), ('gateway_spill', '%.4f'),
          ('gateway_queue_lane', '%.3f'), ('equity_wait_p95', '%.1f'),
          ('equity_wait_mean', '%.2f'), ('throughput', '%.0f')]

for part, tasks, cols in (('A', ['T0', 'T1', 'T2'], A_COLS), ('B', ['A', 'B'], B_COLS)):
    for t in tasks:
        print(f'\n=== Part {part} task {t} ===')
        print('%-16s' % 'method' + ''.join('%22s' % c for c, _ in cols))
        for m in methods:
            rs = g.get((part, t, m))
            if not rs:
                continue
            print('%-16s' % m + ''.join('%22s' % ms(rs, c, f) for c, f in cols))
