"""Ablation summary on Manhattan: T0/T1/T2 reward (Part A) and A3/C3 G + delay (Part B)."""
import os as _os, sys as _sys
_PKG = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _PKG)
_os.chdir(_PKG)  # all inputs/outputs are relative to gmmdsf/
import csv, json, sys
import numpy as np
from zones import ZONE_W_V3, zone_assignment
from composite_reward import FEATURE_SCALE as S
NRM = {k: float(np.linalg.norm(np.asarray(v) * S)) for k, v in ZONE_W_V3.items()}
IDS = [f'intersection_{r}_{c}' for r in range(1, 8) for c in range(1, 29)]
R = list(csv.DictReader(open('table1_v2.csv')))
def rows(part, task, m): return [r for r in R if r['part'] == part and r['task'] == task and r['method'] == m]
def ret(task, m):
    rr = rows('A', task, m); return (np.mean([float(r['return']) for r in rr]), np.std([float(r['return']) for r in rr]), len(rr)) if rr else None
def G(lay, m):
    rr = rows('B', lay, m)
    if not rr: return None
    zo, _ = zone_assignment(lay, IDS); n = {z: sum(v == z for v in zo.values()) for z in set(zo.values())}
    g = [100 * sum(n[z] * u / NRM[z] for z, u in json.loads(r['zone_util_json']).items()) / 196 for r in rr]
    return np.mean(g), np.std(g), len(rr), np.mean([float(r['net_delay']) for r in rr])
ALL = sorted({r['method'] for r in R if r['method'].startswith('tf')})
for m in ALL:
    out = [m]
    for t in ('T0', 'T1', 'T2'):
        v = ret(t, m); out.append(f'{t}={v[0]:.1f}±{v[1]:.1f}(n{v[2]})' if v else f'{t}=-')
    for l in ('A3', 'C3'):
        v = G(l, m); out.append(f'{l}={v[0]:.2f}±{v[1]:.2f}(n{v[2]},d{v[3]:.1f})' if v else f'{l}=-')
    print('  '.join(out))
