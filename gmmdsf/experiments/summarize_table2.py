"""Table 2 (Manhattan heterogeneous, layout A3).

Whole-map headline G = (1/N) sum_i z_i . w_i / ||w_i * s||  (scale-balanced
heterogeneous objective: every zone's scoring weight is rescaled to unit
standardized norm, the same rule already used to define the gateway/equity
weights; s = FEATURE_SCALE). Computed exactly from the logged per-zone
utilities U_z = mean_{i in z} z_i . w_i:  G = sum_z n_z U_z / ||w_z * s|| / N.
Guardrails: full-vehicle delay, throughput. Per-zone: own normalized utility
U_z / ||w_z * s|| and the zone's physical metric.
"""
import os as _os, sys as _sys
_PKG = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _PKG)
_os.chdir(_PKG)  # all inputs/outputs are relative to gmmdsf/
import csv, json, sys
from collections import defaultdict
import numpy as np
from zones import ZONE_W_V3, zone_assignment
from composite_reward import FEATURE_SCALE

LAYOUT = sys.argv[1] if len(sys.argv) > 1 else 'A3'
SRC = sys.argv[2] if len(sys.argv) > 2 else 'table1_v2.csv'
NRM = {k: float(np.linalg.norm(np.asarray(v) * FEATURE_SCALE)) for k, v in ZONE_W_V3.items()}
PHYS = {'corridor': ('corridor_speed_art', 'art. speed'), 'gateway': ('gateway_queue_lane', 'queue/lane'),
        'equity': ('equity_wait_p95', 'wait p95'), 'base': ('base_speed', 'speed')}
HEADLINE = ['ft', 'mp', 'presslight', 'mplight', 'idqn_mlp_T0', 'idqn_gat_T0', 'plight_T0',
            f'idqn_zone{LAYOUT}', f'prlight_{LAYOUT}', 'sf', 'tfc_blind', 'tfc',
            'tfc_compA3', 'tfc_gateA3', 'tfc_gateC3', 'tfc_normA3',
            f'tfc_gate2{LAYOUT}', f'tfc_gate4{LAYOUT}']

ids = [f'intersection_{r}_{c}' for r in range(1, 8) for c in range(1, 29)]
zone_of, _ = zone_assignment(LAYOUT, ids)
n = {z: sum(v == z for v in zone_of.values()) for z in PHYS}
N = sum(n.values())
assert N == 196, n
ZONES = [z for z in PHYS if n[z] > 0]

rows = [r for r in csv.DictReader(open(SRC)) if r['part'] == 'B' and r['task'] == LAYOUT and r['method'] in HEADLINE]

g = defaultdict(list)
for r in rows:
    U = json.loads(r['zone_util_json'])
    zj = json.loads(r['zone_json'])
    rec = {'G': sum(n[z] * U[z] / NRM[z] for z in ZONES) / N,
           'net_delay': float(r['net_delay']), 'throughput': float(r['throughput'])}
    for z in ZONES:
        rec['u_' + z] = U[z] / NRM[z]
        rec['p_' + z] = float(r[PHYS[z][0]]) if r.get(PHYS[z][0]) not in (None, '') else zj[PHYS[z][0]]
    g[r['method']].append(rec)

mean = {m: {k: np.mean([x[k] for x in v]) for k in v[0]} for m, v in g.items()}
sd = {m: np.std([x['G'] for x in v]) for m, v in g.items()}
G_mp = mean['mp']['G']
print(f'zone sizes {n} | norms ' + ', '.join(f'{z}={NRM[z]:.3f}' for z in ZONES))
hdr = f"{'method':<13}{'G (x100)':>16}{'dG vs MP':>10}{'delay':>8}{'tput':>6}" + \
      ''.join(f'{"u_"+z[:4]:>9}{PHYS[z][1]:>12}' for z in ZONES)
print(hdr)
for m in sorted(mean, key=lambda m: -mean[m]['G']):
    x = mean[m]
    print(f"{m:<13}{100*x['G']:>9.2f}+-{100*sd[m]:<5.2f}{100*(x['G']-G_mp):>10.2f}{x['net_delay']:>8.1f}{x['throughput']:>6.0f}"
          + ''.join(f"{100*x['u_'+z]:>9.2f}{x['p_'+z]:>12.3f}" for z in ZONES) + f'  (n={len(g[m])})')

# ---- paper Table 2 view: no T0-trained RL baselines
# (they appear in Table 1); rank = integer rank by G; physical metrics = full-vehicle
# delay + one per zone (corridor art. speed, gateway queue/lane, equity wait p95).
MAIN = [('FixedTime', 'ft', '0'), ('MaxPressure', 'mp', '0'), ('PressLight', 'presslight', '0'),
        ('MPLight', 'mplight', '0'), ('Zone-specialised IDQN', f'idqn_zone{LAYOUT}', '500'),
        ('PRLight (adapted)', f'prlight_{LAYOUT}', '200'), ('MA-USFA', 'sf', '0'),
        ('Ours zero-shot', 'tfc', '0'), ('Ours + gated composer', f'tfc_gate2{LAYOUT}', '200')]
ZCOL = [z for z in ('corridor', 'gateway', 'equity') if n[z] > 0]
main = [(lab, m, c) for lab, m, c in MAIN if m in mean]
order = sorted(main, key=lambda t: -mean[t[1]]['G'])
rank = {m: i + 1 for i, (_, m, _) in enumerate(order)}
print()
print(f'| Method | extra training on layout | G (x100) | rank | delay | ' + ' | '.join(f'{z} {PHYS[z][1]}' for z in ZCOL) + ' |')
print('|' + '---|' * (5 + len(ZCOL)))
for lab, m, c in order:
    x = mean[m]
    print(f"| {lab} | {c} | {100*x['G']:.2f} +- {100*sd[m]:.2f} | {rank[m]} | {x['net_delay']:.1f} | "
          + ' | '.join(f"{x['p_'+z]:.3f}" if z == 'gateway' else f"{x['p_'+z]:.2f}" for z in ZCOL) + ' |')
missing = [lab for lab, m, _ in MAIN if m not in mean]
if missing:
    print('pending:', ', '.join(missing))
