"""Fig: the three evaluation networks + the Manhattan heterogeneous zone task (1x4)."""
import os as _os, sys as _sys
_PKG = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _PKG)
_os.chdir(_PKG)  # all inputs/outputs are relative to gmmdsf/
_os.makedirs('figures', exist_ok=True)
import os, sys, xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from zones import zone_assignment

RAW = os.path.join(_PKG, '..', 'data', 'raw_data')
LAYOUT = sys.argv[1] if len(sys.argv) > 1 else 'C3'
plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans'})
ZC = {'corridor': '#ef6c00', 'gateway': '#1565c0', 'equity': '#2e7d32', 'base': '#9e9e9e'}

def load(scn):
    root = ET.parse(os.path.join(RAW, scn, scn + '.net.xml')).getroot()
    tls = {t.get('id') for t in root.iter('tlLogic')}
    junc = {j.get('id'): (float(j.get('x')), float(j.get('y'))) for j in root.iter('junction') if j.get('type') != 'internal'}
    edges, inc = [], {}
    for e in root.iter('edge'):
        if e.get('function') == 'internal':
            continue
        for ln in e.findall('lane'):
            pts = np.array([[float(v) for v in p.split(',')] for p in ln.get('shape').split()])
            edges.append((e.get('to'), pts))
    pos = {t: junc[t] for t in tls if t in junc}
    return edges, pos

def draw(ax, scn, title, sub, zone_of=None):
    edges, pos = load(scn)
    for to, pts in edges:
        c, lw = '#b0b0b0', 0.5
        if zone_of is not None and to in zone_of:
            c, lw = ZC[zone_of[to]], 1.1
        ax.plot(pts[:, 0], pts[:, 1], color=c, lw=lw, solid_capstyle='round', zorder=1)
    xy = np.array(list(pos.values()))
    if zone_of is None:
        ax.scatter(xy[:, 0], xy[:, 1], s=14 if len(pos) > 50 else 40, c='#c62828', edgecolor='k', lw=0.3, zorder=3)
    else:
        cols = [ZC[zone_of[t]] for t in pos]
        ax.scatter(xy[:, 0], xy[:, 1], s=14, c=cols, edgecolor='k', lw=0.3, zorder=3)
    ax.set_aspect('equal'); ax.axis('off'); ax.set_anchor('N')
    ax.set_title(title, fontsize=9.5, fontweight='bold')

fig, axes = plt.subplots(1, 4, figsize=(10, 5.0), gridspec_kw=dict(width_ratios=[1, 1.25, 0.9, 1], wspace=0.15))
draw(axes[0], 'manhattan_28x7', '(a) Manhattan 7×28', '196 signals · 600 s')
draw(axes[1], 'hangzhou_4x4_gudang_18041610_1h', '(b) Hangzhou 4×4', '16 signals · 3600 s')
draw(axes[2], 'atlanta_1x5', '(c) Atlanta 1×5', '5 signals · 1200 s')
ids = [f'intersection_{r}_{c}' for r in range(1, 8) for c in range(1, 29)]
zo, _ = zone_assignment(LAYOUT, ids)
H = {'A3': 0.32, 'C3': 0.83}.get(LAYOUT)
draw(axes[3], 'manhattan_28x7', f'(d) Manhattan, zoned {LAYOUT}',
     f'3 zone objectives · H = {H:.2f}' if H else 'zone objectives', zone_of=zo)
present = [z for z in ['corridor', 'gateway', 'equity', 'base'] if z in set(zo.values())]
desc = {'corridor': 'corridor: speed', 'gateway': 'gateway: queue', 'equity': 'equity: waiting + fairness', 'base': 'base: T0'}
axes[3].legend(handles=[Line2D([0], [0], color=ZC[z], lw=3, label=desc[z]) for z in present],
               loc='upper center', bbox_to_anchor=(0.5, 0.0), fontsize=7.5, frameon=False, ncol=1)
out = f'figures/fig_envs_{LAYOUT}'
fig.savefig(out + '.png', dpi=300, bbox_inches='tight')
try:
    fig.savefig(out + '.pdf', bbox_inches='tight')
except PermissionError:
    print('pdf open, skipped')
print('saved', out)
