"""Fig 3 (v2): (a) dG vs IDQN-T0 along the heterogeneity sweep, with gated-composer points;
(b) paired task-conditioning gain (zero-shot minus task-blind, same backbone)."""
import os as _os, sys as _sys
_PKG = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _PKG)
_os.chdir(_PKG)  # all inputs/outputs are relative to gmmdsf/
_os.makedirs('figures', exist_ok=True)
import csv, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from zones import ZONE_W_V3, zone_assignment
from composite_reward import FEATURE_SCALE as S, DEFAULT_W
plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans'})
NRM = {k: float(np.linalg.norm(np.asarray(v) * S)) for k, v in ZONE_W_V3.items()}
IDS = [f'intersection_{r}_{c}' for r in range(1, 8) for c in range(1, 29)]
t0 = DEFAULT_W * S; t0 /= np.linalg.norm(t0)
R = list(csv.DictReader(open('table1_v2.csv')))
def H_of(lay):
    _, W = zone_assignment(lay, IDS)
    return float(np.mean([1 - (w * S) @ t0 / np.linalg.norm(w * S) for w in W]))
def G(lay, task, m, part='B'):
    zo, _ = zone_assignment(lay, IDS)
    n = {z: sum(v == z for v in zo.values()) for z in set(zo.values())}
    out = {}
    for r in R:
        if r['part'] != part or r['task'] != task or r['method'] != m: continue
        zu = json.loads(r['zone_util_json']) if r['zone_util_json'] else {'base': float(r['realized_utility'])}
        out[int(r['seed'])] = 100 * sum(n[z] * u / NRM[z] for z, u in zu.items()) / 196
    assert len(out) == 3, (lay, task, m, len(out))
    return out
def diff(a, b):
    d = [a[s] - b[s] for s in sorted(a)]; return np.mean(d), np.std(d)
SW = [('S0', 'S0'), ('S25', 'S25'), ('S50', 'S50'), ('S75', 'S75'), ('S100', 'C3')]
Hs = [H_of(l) for l, _ in SW]
ref = {t: G(l, t, 'idqn_mlp_T0') for l, t in SW}
res = {}
for m in ['tfc', 'tfc_blind', 'mp']:
    res[m] = [diff(G(l, t, m), ref[t]) for l, t in SW]
comp = [(Hs[0], *diff(G('S0', 'T0', 'tfc_gate2T0', 'A'), G('S0', 'T0', 'idqn_mlp_T0', 'A'))),
        (Hs[-1], *diff(G('S100', 'C3', 'tfc_gate2C3'), ref['C3']))]
compA3 = (H_of('A3'), *diff(G('A3', 'A3', 'tfc_gate2A3'), G('A3', 'A3', 'idqn_mlp_T0')))
zsA3 = (H_of('A3'), *diff(G('A3', 'A3', 'tfc'), G('A3', 'A3', 'idqn_mlp_T0')))
gain = [diff(G(l, t, 'tfc'), G(l, t, 'tfc_blind')) for l, t in SW]

fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.4), gridspec_kw={'width_ratios': [1.35, 1]})
figz, zax = plt.subplots(1, 2, figsize=(9.2, 3.0), sharey=True)
a.axhspan(0, 3.2, color='#e8f5e9', zorder=0)
a.axhline(0, color='#1565c0', lw=1.2, ls='--')
a.text(0.005, 0.12, 'IDQN trained on T0 (reference)', color='#1565c0', fontsize=8)
sty = {'tfc': ('#2e7d32', 'o', '-', 2.2, 'Ours zero-shot'),
       'tfc_blind': ('#9e9e9e', 's', ':', 1.4, 'Ours task-blind (w = T0 everywhere)'),
       'mp': ('#424242', '^', '-.', 1.2, 'MaxPressure')}
for m, (c, mk, ls, lw, lab) in sty.items():
    y, e = zip(*res[m]); a.errorbar(Hs, y, yerr=e, color=c, marker=mk, ls=ls, lw=lw, ms=5, capsize=2, label=lab)
cx, cy, ce = zip(*comp)
a.errorbar(cx, cy, yerr=ce, color='#c62828', marker='*', ms=11, ls='none', capsize=2, label='Ours + gated composer (200 ep)')
a.errorbar([compA3[0]], [compA3[1]], yerr=[compA3[2]], color='#c62828', marker='*', mfc='white', ms=10, ls='none', capsize=2)
a.errorbar([zsA3[0]], [zsA3[1]], yerr=[zsA3[2]], color='#2e7d32', marker='o', mfc='white', ms=6, ls='none', capsize=2)
a.annotate('A3', (compA3[0], compA3[1]), xytext=(6, -2), textcoords='offset points', fontsize=7.5)
a.set_xlabel('Task heterogeneity $H$'); a.set_ylabel(r'$\Delta G$ vs IDQN-T0 (×100)')
a.set_ylim(-3.8, 3.2); a.set_xlim(-0.04, 0.88); a.grid(alpha=0.3)
a.set_title('(a) Advantage grows with heterogeneity', fontsize=9.5)
y, e = zip(*gain)
b.bar(range(5), y, yerr=e, color=['#c8e6c9', '#a5d6a7', '#81c784', '#4caf50', '#2e7d32'], edgecolor='#1b5e20', capsize=3)
for i, v in enumerate(y): b.text(i, v + e[i] + 0.06, f'+{v:.2f}' if v > 0.005 else '0', ha='center', fontsize=8)
b.set_xticks(range(5)); b.set_xticklabels([f'{abs(h):.2f}' for h in Hs]); b.set_xlabel('Task heterogeneity $H$')
b.set_ylabel('$G$(zero-shot) $-$ $G$(task-blind)'); b.set_ylim(0, 2.9); b.grid(alpha=0.3, axis='y')
b.set_title('(b) Gain from task conditioning (same model)', fontsize=9.5)
h_, l_ = a.get_legend_handles_labels()
fig.legend(h_, l_, loc='lower center', ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.07, 1, 1))
for e_ in ('png', 'pdf'):
    try: fig.savefig(f'figures/fig3_heterogeneity_sweep_v2.{e_}', dpi=300, bbox_inches='tight')
    except PermissionError: print('skip', e_)
NL = chr(10)
ZM = [('corridor_speed_art', +1, 'Corridor' + NL + 'speed'), ('gateway_queue_lane', -1, 'Gateway' + NL + 'queue'),
      ('equity_wait_p95', -1, 'Equity' + NL + 'wait p95'), ('equity_wait_mean', -1, 'Equity' + NL + 'mean wait')]
def zm(lay, m, c):
    return np.mean([float(r[c]) for r in R if r['part'] == 'B' and r['task'] == lay and r['method'] == m])
ZR = {}
for ax_, lay, cm in [(zax[0], 'A3', 'tfc_gate2A3'), (zax[1], 'C3', 'tfc_gate2C3')]:
    x = np.arange(len(ZM)); wd = 0.26
    for k, (m, c, lab) in enumerate([('mp', '#9e9e9e', 'MaxPressure'), ('tfc', '#2e7d32', 'Ours zero-shot'), (cm, '#c62828', 'Ours + gated composer')]):
        v = [sgn * 100 * (zm(lay, m, col) - zm(lay, 'idqn_mlp_T0', col)) / zm(lay, 'idqn_mlp_T0', col) for col, sgn, _ in ZM]
        ZR[(lay, lab)] = [round(t, 1) for t in v]
        bars = ax_.bar(x + (k - 1) * wd, v, wd, color=c, label=lab)
        if m != 'mp':
            for xi, t in zip(x + (k - 1) * wd, v): ax_.text(xi, t + (2 if t >= 0 else -2), f'{t:+.0f}%', ha='center', va='bottom' if t >= 0 else 'top', fontsize=6.8)
    ax_.axhline(0, color='#1565c0', lw=1.2, ls='--'); ax_.set_xticks(x); ax_.set_xticklabels([t for *_, t in ZM], fontsize=8)
    ax_.grid(alpha=0.3, axis='y'); ax_.set_title(f'Layout {lay} ($H$={H_of(lay):.2f})', fontsize=9.5)
zax[0].set_ylabel('Improvement over IDQN-T0 (%)'); zax[0].set_ylim(-32, 62)
zax[0].text(-0.45, 3, 'IDQN-T0', color='#1565c0', fontsize=7.5)
h_, l_ = zax[0].get_legend_handles_labels()
figz.legend(h_, l_, loc='lower center', ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
figz.tight_layout(rect=(0, 0.08, 1, 1))
for e_ in ('png', 'pdf'):
    try: figz.savefig(f'figures/fig3b_zone_vs_idqn.{e_}', dpi=300, bbox_inches='tight')
    except PermissionError: print('skip', e_)
print('ZONE', ZR)
print('H', [round(h, 3) for h in Hs])
for m in res: print(m, [(round(x, 2), round(s, 2)) for x, s in res[m]])
print('composer', [(round(h, 2), round(x, 2), round(s, 2)) for h, x, s in comp], 'A3', [round(v, 2) for v in compA3], 'zsA3', [round(v, 2) for v in zsA3])
print('gain', [(round(x, 2), round(s, 2)) for x, s in gain])

# ---- combined Fig 3: (a) task-conditioning gain, (b) zone-level improvement vs IDQN-T0 ----
plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.8})
fc = plt.figure(figsize=(12, 3.6))
gs = fc.add_gridspec(1, 3, width_ratios=[0.9, 1.2, 1.2], wspace=0.22)
ga = fc.add_subplot(gs[0]); gb1 = fc.add_subplot(gs[1]); gb2 = fc.add_subplot(gs[2], sharey=gb1)
y, e = zip(*gain)
ga.bar(range(5), y, width=0.62, color=['#e8f5e9', '#c8e6c9', '#a5d6a7', '#66bb6a', '#2e7d32'], edgecolor='#1b5e20', lw=0.8,
       yerr=e, error_kw=dict(ecolor='#555555', elinewidth=0.9, capsize=3, capthick=0.9), zorder=3)
for i, (v, s) in enumerate(zip(y, e)):
    ga.text(i, v + s + 0.08 if v > 0.005 else 0.08, f'+{v:.2f}' if v > 0.005 else '0', ha='center', va='bottom', fontsize=8, color='#1b5e20', fontweight='bold')
ga.set_xticks(range(5)); ga.set_xticklabels([f'{abs(h):.2f}' for h in Hs]); ga.set_xlabel('Task heterogeneity $H$')
ga.set_ylabel(r'$\Delta G$  (zero-shot $-$ task-blind)'); ga.set_ylim(0, 3.1); ga.grid(alpha=0.25, axis='y', zorder=0)
ga.text(-0.02, 1.13, '(a) Gain from task conditioning', transform=ga.transAxes, fontsize=10)
ga.set_title('same model, reads $w$ vs. ignores $w$', fontsize=9.5)
SER = [('mp', '#bdbdbd', 'MaxPressure'), ('tfc', '#43a047', 'Ours zero-shot'), (None, '#e53935', 'Ours + gated composer')]
for ax_, lay, cm in [(gb1, 'A3', 'tfc_gate2A3'), (gb2, 'C3', 'tfc_gate2C3')]:
    x = np.arange(len(ZM)) * 1.15; wd = 0.3
    ax_.axhspan(0, 62, color='#f1f8e9', zorder=0)
    for k, (m, c, lab) in enumerate(SER):
        m = m or cm
        v = [sgn * 100 * (zm(lay, m, col) - zm(lay, 'idqn_mlp_T0', col)) / zm(lay, 'idqn_mlp_T0', col) for col, sgn, _ in ZM]
        xs = x + (k - 1) * wd
        ax_.bar(xs, v, wd * 0.92, color=c, label=lab, zorder=3, edgecolor='white', lw=0.5)
        if m == 'mp': continue
        for xi, t in zip(xs, v):
            if abs(t) >= 12:
                ax_.text(xi, t - np.sign(t) * 1.5, f'{t:+.0f}', ha='center', va='top' if t > 0 else 'bottom',
                         fontsize=7, color='white', fontweight='bold', rotation=90, zorder=4)
            else:
                ax_.text(xi, t + (1.2 if t >= 0 else -1.2), f'{t:+.0f}', ha='center', va='bottom' if t >= 0 else 'top',
                         fontsize=7, color=c, fontweight='bold', zorder=4)
    ax_.axhline(0, color='#1565c0', lw=1.1, ls='--', zorder=2)
    ax_.set_xticks(x); ax_.set_xticklabels([t for *_, t in ZM], fontsize=8)
    ax_.grid(alpha=0.25, axis='y', zorder=0)
    ax_.set_title(f'{lay}  ($H$ = {H_of(lay):.2f})', fontsize=9.5)
gb1.set_ylabel('Improvement over IDQN-T0 (%)'); gb1.set_ylim(-30, 60)
gb1.text(-0.02, 1.13, '(b) Zone-level metrics relative to IDQN trained on T0', transform=gb1.transAxes, fontsize=10)
plt.setp(gb2.get_yticklabels(), visible=False); gb2.tick_params(axis='y', length=0)
h_, l_ = gb1.get_legend_handles_labels()
from matplotlib.lines import Line2D
h_ = h_ + [Line2D([], [], color='#1565c0', lw=1.1, ls='--')]; l_ = l_ + ['IDQN-T0 (0 %)']
fc.legend(h_, l_, fontsize=8.5, frameon=False, loc='lower center', ncol=4, bbox_to_anchor=(0.63, -0.07))
for e_ in ('png', 'pdf'):
    try: fc.savefig(f'figures/fig3_combined.{e_}', dpi=300, bbox_inches='tight')
    except PermissionError: print('skip', e_)
