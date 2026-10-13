"""Attainment (FT=0, MP=1, seed-matched) + zone physical metrics for bench_results.csv."""
import os as _os, sys as _sys
_PKG = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _PKG)
_os.chdir(_PKG)  # all inputs/outputs are relative to gmmdsf/
import csv, json, sys
from collections import defaultdict
import numpy as np
rows = list(csv.DictReader(open('bench_results.csv')))
U = defaultdict(dict); RW = defaultdict(dict)
for r in rows:
    if r['zone_util_json']:
        U[(r['bench'], r['task'], r['method'])][r['seed']] = json.loads(r['zone_util_json'])
    RW[(r['bench'], r['task'], r['method'])][r['seed']] = r
PHYS = ['corridor_speed_art', 'gateway_queue_lane', 'equity_wait_p95', 'equity_wait_mean', 'net_delay']
for (b, t) in sorted({(k[0], k[1]) for k in RW}):
    print(f'\n=== {b} / {t} ===')
    if t == 'T0':
        for (bb, tt, m), d in sorted(RW.items()):
            if (bb, tt) != (b, t): continue
            f = lambda k: np.mean([float(x[k]) for x in d.values() if x[k] not in ('', 'None')])
            print('%-16s return %8.1f  delay %7.1f  tput %6.0f  queue %5.2f  wait %6.2f  (n=%d)' % (m, f('return'), f('net_delay'), f('throughput'), f('avg_queue'), f('avg_wait'), len(d)))
        continue
    ft, mp = U.get((b, t, 'ft'), {}), U.get((b, t, 'mp'), {})
    zones = sorted(next(iter(ft.values())).keys()) if ft else []
    print('%-16s' % 'method' + ''.join('%11s' % z[:8] for z in zones) + '%9s' % 'HETERO' + ''.join('%14s' % k[:13] for k in PHYS))
    for (bb, tt, m), seeds in sorted(U.items()):
        if (bb, tt) != (b, t): continue
        per = []
        for s, u in seeds.items():
            if s in ft and s in mp:
                # amended (pre-results): 0 = worse of {FT, MP}, 1 = better of {FT, MP} per zone
                a = []
                for z in zones:
                    lo, hi = min(ft[s][z], mp[s][z]), max(ft[s][z], mp[s][z])
                    a.append((u[z] - lo) / (hi - lo) if hi - lo > 1e-9 else np.nan)
                per.append(a + [np.nanmean(a)])
        if not per: continue
        mu = np.nanmean(np.array(per), 0)
        ph = [np.mean([float(x[k]) for x in RW[(b, t, m)].values() if x.get(k) not in ('', None, 'None')]) for k in PHYS]
        print('%-16s' % m + ''.join('%11.2f' % v for v in mu[:-1]) + '%9.2f' % mu[-1] + ''.join('%14.3f' % v for v in ph) + '  (n=%d)' % len(per))
