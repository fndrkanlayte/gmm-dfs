"""Fig 3: heterogeneity sweep (G advantage vs H); Fig 4: behaviour probes per zone;
Fig 7: non-linear (distributional) objectives T4/T5 — same backbone, mean vs GMM."""
import os as _os, sys as _sys
_PKG = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _PKG)
_os.chdir(_PKG)  # all inputs/outputs are relative to gmmdsf/
_os.makedirs('figures', exist_ok=True)
import csv, glob, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from zones import ZONE_W_V3, zone_assignment
from composite_reward import FEATURE_SCALE as S, DEFAULT_W
def _save(fig, fn, **kw):
    try:
        fig.savefig(fn, **kw)
    except PermissionError:
        print('skip (file open):', fn)
plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans'})
NRM = {k: float(np.linalg.norm(np.asarray(v) * S)) for k, v in ZONE_W_V3.items()}
IDS = [f'intersection_{r}_{c}' for r in range(1, 8) for c in range(1, 29)]
t0 = DEFAULT_W * S; t0 /= np.linalg.norm(t0)
COL = {'tfc': '#2e7d32', 'tfc_blind': '#a5d6a7', 'idqn_mlp_T0': '#1565c0', 'mp': '#424242', 'mplight': '#ff8f00', 'sf': '#7e57c2'}
LAB = {'tfc': 'Ours zero-shot', 'tfc_blind': 'Ours task-blind (all T0)', 'idqn_mlp_T0': 'IDQN (trained on T0)',
       'mp': 'MaxPressure', 'mplight': 'MPLight', 'sf': 'MA-USFA'}

# ---------------- Fig 3 ----------------
rows = [r for r in csv.DictReader(open('table1_v2.csv')) if r['part'] == 'B']
LAYS = [('S0', 'S0'), ('S25', 'S25'), ('S50', 'S50'), ('S75', 'S75'), ('S100', 'C3')]
def Gs(lay_eval, lay_csv, m):
    zo, W = zone_assignment(lay_eval, IDS)
    n = {z: sum(v == z for v in zo.values()) for z in set(zo.values())}
    return {int(r['seed']): 100 * sum(n[z] * u / NRM[z] for z, u in json.loads(r['zone_util_json']).items()) / 196
            for r in rows if r['task'] == lay_csv and r['method'] == m}
def H_of(lay):
    _, W = zone_assignment(lay, IDS)
    return float(np.mean([1 - (w * S) @ t0 / np.linalg.norm(w * S) for w in W]))
fig3 = {}
fig, ax = plt.subplots(figsize=(5.2, 3.6))
for m in ['tfc', 'tfc_blind', 'idqn_mlp_T0', 'mplight']:
    xs, ys, es = [], [], []
    for le, lc in LAYS:
        g, gm = Gs(le, lc, m), Gs(le, lc, 'mp')
        common = sorted(set(g) & set(gm))
        if len(common) < 3: continue
        d = [g[s] - gm[s] for s in common]
        xs.append(H_of(le)); ys.append(np.mean(d)); es.append(np.std(d))
    fig3[m] = list(zip(xs, ys, es))
    if xs:
        ax.errorbar(xs, ys, yerr=es, marker='o', ms=5, capsize=2, color=COL[m], label=LAB[m],
                    lw=2.2 if m == 'tfc' else 1.3)
ax.axhline(0, color='#424242', lw=1, ls='--'); ax.text(0.01, 0.15, 'MaxPressure', color='#424242', fontsize=8)
ax.set_xlabel('Task heterogeneity H (0 = homogeneous T0, 0.83 = C3)')
ax.set_ylabel('ΔG vs MaxPressure (×100)')
ax.grid(alpha=0.3); ax.legend(fontsize=8, frameon=False)
fig.tight_layout()
for e in ('png', 'pdf'): _save(fig, f'figures/fig3_heterogeneity_sweep.{e}', dpi=300, bbox_inches='tight')
print('FIG3', json.dumps({m: [(round(a, 3), round(b, 2), round(c, 2)) for a, b, c in v] for m, v in fig3.items()}))

# ---------------- Fig 4 ----------------
if os.path.exists('behaviour_results.jsonl'):
    B = [json.loads(l) for l in open('behaviour_results.jsonl')]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 5.6))
    for j, L in enumerate(['C3', 'A3']):
        zones = ['corridor', 'gateway', 'equity'] + (['base'] if L == 'A3' else [])
        for i, (key, yl) in enumerate([('art_green', 'Green share to arterial lanes'), ('switch_rate', 'Phase-switch rate per decision')]):
            ax = axes[i, j]
            for k, m in enumerate(['mp', 'tfc_blind', 'tfc']):
                v = [[b[f'{z}_{key}'] for b in B if b['layout'] == L and b['method'] == m and f'{z}_{key}' in b] for z in zones]
                mu = [np.mean(x) if x else np.nan for x in v]; sd = [np.std(x) if x else 0 for x in v]
                ax.bar(np.arange(len(zones)) + (k - 1) * 0.27, mu, 0.27, yerr=sd, capsize=2, color=COL[m],
                       edgecolor='k', lw=0.4, label=LAB[m])
                print('FIG4', L, key, m, [round(x, 3) for x in mu])
            ax.set_xticks(range(len(zones))); ax.set_xticklabels(zones); ax.set_ylabel(yl, fontsize=8)
            ax.set_title(f'{L}: {yl.lower()}', fontsize=9); ax.grid(alpha=0.3, axis='y')
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    for e in ('png', 'pdf'): _save(fig, f'figures/fig4_behaviour.{e}', dpi=300, bbox_inches='tight')

# ---------------- Fig 7 ----------------
DELTA = np.array([0.1924, 0.3132, 0.2888, 999.0, 999.0])  # cost-only box (speed has no upper bound); GOAL = np.array([0.0899, 0.0957, 0.2116, 2.8178, 0.0070])
GOAL = np.array([0.0899, 0.0957, 0.2116, 2.8178, 0.0070])
def z_of(p):
    d = json.load(open(p)); return np.array(d['z_matrix'], float)
def viol(z): return 100 * (1 - (z <= DELTA).all(1).mean())
def gdev(z): return np.sqrt(((z - GOAL) ** 2).sum(1)).mean()
src = {
    'MaxPressure': ('mp', [f'eval_runs/table1_v2/mp_AT0_s{s}/maxpressure_homo_result.json' for s in range(3)]),
    'IDQN (T0)': ('idqn_mlp_T0', [f'composite_ev_idqn_mlp_T0_AT0_s{s}_s{s}/DQN_mlp/eval_only_s{s}.json' for s in range(3)]),
    'MA-USFA': ('sf', [f'sf_ev_s{s}/transfer/transfer_homo_sf_AT0_s{s}_result.json' for s in range(3)]),
    'Ours mean-only': ('tfc_blind', [f'gmm_ev_s{s}/transfer/transfer_homo_tfc_AT0_s{s}_result.json' for s in range(3)]),
}
gm4 = [f'gmm_ev_s{s}/transfer/transfer_homo_tfc_T4cost_s{s}_result.json' for s in range(3)]
gm5 = [f'gmm_ev_s{s}/transfer/transfer_homo_tfc_T5gauss_s{s}_result.json' for s in range(3)]
if all(os.path.exists(p) for p in gm4 + gm5):
    res = {}
    for lab, (c, ps) in src.items():
        zs = [z_of(p) for p in ps]; res[lab] = (c, [viol(z) for z in zs], [gdev(z) for z in zs])
    res['Ours GMM (distribution-aware)'] = ('tfc', [viol(z_of(p)) for p in gm4], [gdev(z_of(p)) for p in gm5])
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    labs = list(res)
    for ax, k, yl in [(axes[0], 1, 'T4 threshold task: violation rate (%) ↓'), (axes[1], 2, 'T5 precision task: goal deviation ↓')]:
        mu = [np.mean(res[l][k]) for l in labs]; sd = [np.std(res[l][k]) for l in labs]
        ax.bar(range(len(labs)), mu, yerr=sd, capsize=3, color=[COL[res[l][0]] for l in labs], edgecolor='k', lw=0.4)
        for x, v in enumerate(mu): ax.text(x, v, f'{v:.2f}' if k == 2 else f'{v:.1f}', ha='center', va='bottom', fontsize=8)
        ax.set_xticks(range(len(labs))); ax.set_xticklabels([l.replace(' (', '\n(') for l in labs], fontsize=7.5)
        ax.set_title(yl, fontsize=9); ax.grid(alpha=0.3, axis='y')
        lo = min(mu) * 0.85; ax.set_ylim(lo, max(mu) * 1.08)
        print('FIG7', yl, {l: round(float(v), 3) for l, v in zip(labs, mu)})
    fig.tight_layout()
    for e in ('png', 'pdf'): _save(fig, f'figures/fig7_distributional.{e}', dpi=300, bbox_inches='tight')
