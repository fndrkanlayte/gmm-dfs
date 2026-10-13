"""Fig 5: adaptation cost vs whole-map heterogeneous objective G (A3, C3).
Fig 6: per-intersection spatial map on Manhattan C3 (MP vs Ours zero-shot), 3 eval seeds."""
import os as _os, sys as _sys
_PKG = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _PKG)
_os.chdir(_PKG)  # all inputs/outputs are relative to gmmdsf/
_os.makedirs('figures', exist_ok=True)
import csv, glob, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from zones import ZONE_W_V3, zone_assignment
from composite_reward import FEATURE_SCALE

NRM = {k: float(np.linalg.norm(np.asarray(v) * FEATURE_SCALE)) for k, v in ZONE_W_V3.items()}
IDS = [f'intersection_{r}_{c}' for r in range(1, 8) for c in range(1, 29)]
plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans'})

# ---------------- Fig 5 ----------------
rows = [r for r in csv.DictReader(open('table1_v2.csv')) if r['part'] == 'B']
def G_of(layout, method):
    zo, _ = zone_assignment(layout, IDS)
    n = {z: sum(v == z for v in zo.values()) for z in set(zo.values())}
    vals = [sum(n[z] * u / NRM[z] for z, u in json.loads(r['zone_util_json']).items()) / 196
            for r in rows if r['task'] == layout and r['method'] == method]
    return (100 * np.mean(vals), 100 * np.std(vals)) if vals else None

M = [  # label, method key pattern, extra training episodes on the layout, style
    ('FixedTime', 'ft', 0, dict(marker='s', c='#9e9e9e')),
    ('MaxPressure', 'mp', 0, dict(marker='s', c='#424242')),
    ('PressLight', 'presslight', 0, dict(marker='^', c='#8d6e63')),
    ('MPLight', 'mplight', 0, dict(marker='^', c='#ff8f00')),
    ('MA-USFA', 'sf', 0, dict(marker='D', c='#7e57c2')),
    ('PRLight (adapted)', 'prlight_{L}', 200, dict(marker='v', c='#c62828')),
    ('Zone-specialised IDQN', 'idqn_zone{L}', 500, dict(marker='v', c='#1565c0')),
    ('Ours zero-shot', 'tfc', 0, dict(marker='*', c='#2e7d32', s=180)),
    ('Ours + gated composer', 'tfc_gate{L}', 200, dict(marker='P', c='#66bb6a', s=110)),
]
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
for ax, L, H in zip(axes, ['A3', 'C3'], [0.32, 0.83]):
    for lab, key, ep, st in M:
        g = G_of(L, key.format(L=L))
        if g is None:
            continue
        st = dict(st); s = st.pop('s', 60)
        ax.errorbar(ep, g[0], yerr=g[1], fmt='none', ecolor=st['c'], elinewidth=1, capsize=2)
        ax.scatter(ep, g[0], s=s, edgecolor='k', linewidth=0.5, zorder=3, label=lab, **st)
    ax.set_xlim(-40, 560); ax.set_xticks([0, 100, 200, 300, 400, 500])
    ax.set_xlabel('Layout-specific training episodes')
    ax.set_title(f'{L} (heterogeneity H = {H:.2f})')
    ax.grid(alpha=0.3)
axes[0].set_ylabel('Whole-map heterogeneous objective G (×100)')
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02), fontsize=8)
fig.tight_layout(rect=(0, 0.1, 1, 1))
for ext in ('png',):
    fig.savefig(f'figures/fig5_adaptation_cost.{ext}', dpi=300, bbox_inches='tight')

# ---------------- Fig 6 ----------------
def grid(vec, ids):
    g = np.full((7, 28), np.nan)
    for v, t in zip(vec, ids):
        _, r, c = t.rsplit('_', 2); g[int(r) - 1, int(c) - 1] = v
    return g
def load(paths, idkey):
    Z, U = [], []
    for p in paths:
        d = json.load(open(p)); ids = d[idkey]
        zo, W = zone_assignment('C3', ids)
        z = np.array(d['z_matrix'], float)
        nr = np.array([NRM[zo[t]] for t in ids])
        Z.append(grid(z[:, 1], ids)); U.append(grid((z * W).sum(1) / nr, ids))
    return np.mean(Z, 0), np.mean(U, 0)
mp_p = sorted([f'eval_runs/table1_v2/mp_BC3_s{s}/maxpressure_homo_result.json' for s in range(3)])
ou_p = sorted([f'gmm_ev_s{s}/transfer/transfer_hetero_tfc_BC3_s{s}_result.json' for s in range(3)])
assert len(mp_p) == 3 and len(ou_p) == 3, (mp_p, ou_p)
wz_mp, u_mp = load(mp_p, 'ids'); wz_ou, u_ou = load(ou_p, 'agent_ids')
zo, _ = zone_assignment('C3', IDS)
zcode = {'corridor': 0, 'gateway': 1, 'equity': 2}
zg = grid([zcode[zo[t]] for t in IDS], IDS)

fig, axes = plt.subplots(4, 1, figsize=(9.2, 7.0), sharex=True, gridspec_kw=dict(hspace=0.45))
ext = (0.5, 28.5, 7.5, 0.5)
im = axes[0].imshow(zg, cmap=ListedColormap(['#ef6c00', '#1565c0', '#2e7d32']), extent=ext, aspect='auto')
axes[0].set_title('(a) C3 zone layout:  corridor (speed)  |  gateway (queue)  |  equity (waiting + fairness)', fontsize=9)
from matplotlib.patches import Patch
axes[0].legend(handles=[Patch(color='#ef6c00', label='corridor'), Patch(color='#1565c0', label='gateway'),
                        Patch(color='#2e7d32', label='equity')], loc='upper right', bbox_to_anchor=(1.0, 1.55), ncol=3, fontsize=8, frameon=False)
vmax = np.nanpercentile(np.concatenate([wz_mp.ravel(), wz_ou.ravel()]), 98)
for ax, g, t in [(axes[1], wz_mp, '(b) MaxPressure: waiting feature per intersection'),
                 (axes[2], wz_ou, '(c) Ours zero-shot: waiting feature per intersection')]:
    imw = ax.imshow(g, cmap='Reds', vmin=0, vmax=vmax, extent=ext, aspect='auto'); ax.set_title(t, fontsize=9)
fig.colorbar(imw, ax=axes[1:3], fraction=0.02, pad=0.01, label='discounted normalised waiting')
d = u_ou - u_mp
lim = np.nanpercentile(np.abs(d), 98)
imd = axes[3].imshow(d, cmap='RdBu', norm=TwoSlopeNorm(0, -lim, lim), extent=ext, aspect='auto')
axes[3].set_title(f'(d) Own-zone objective gain, Ours − MaxPressure (blue = Ours better; '
                  f'{np.mean(d > 0) * 100:.0f}% of intersections improve)', fontsize=9)
fig.colorbar(imd, ax=axes[3], fraction=0.02, pad=0.01, label='Δ normalised utility')
for ax in axes:
    ax.set_yticks([1, 4, 7]); ax.set_ylabel('row'); ax.set_xticks([1, 7, 14, 21, 28])
    ax.tick_params(labelbottom=False, labeltop=False)
axes[-1].tick_params(labelbottom=True)
axes[-1].set_xlabel('column (east–west arterial axis)')
for ext_ in ('png',):
    fig.savefig(f'figures/fig6_spatial_C3.{ext_}', dpi=300, bbox_inches='tight')
print('improve share by zone:', {z: round(float(np.mean(d[zg == c] > 0)), 2) for z, c in zcode.items()})
print('mean gain by zone:', {z: round(float(np.nanmean(d[zg == c])), 4) for z, c in zcode.items()})
