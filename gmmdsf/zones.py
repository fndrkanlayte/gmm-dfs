"""City-zone heterogeneous tasks + zone-level physical metrics (Table 1, Part B).

Grid facts (manhattan_28x7, verified from the route file):
  * signals are intersection_<r>_<c>, r=1..7, c=1..28; edge road_r_c_d with
    d=1/3 = +c/-c (the long axis), d=0/2 = +r/-r.
  * 76% of route-edge traversals run along the c axis; per-row c-axis load:
    r2 76.9k, r1 52.0k, r7 35.8k, r3 14.1k, r5 11.3k, r4 4.2k, r6 2.4k
    -> rows 2/1/7 are the arterials, rows 4-6 are low-volume side streets.
  * entries: west boundary (c=0) 4.4k, east (c=29) 2.3k, north/south 9.6k;
    cross-street load is highest at c=1 (9.1k) and c=2 (6.6k).

Each layout assigns every signal to one zone and every zone one in-family
objective (all four lie inside sample_task_family_v2's support):

  corridor  -- arterial progression : speed-emphasis  w=[-.3,-.1,-.5, 6, 0]
  gateway   -- spillback protection : queue-axis      w=[-2,-.03,-.1,.3, 0]
  equity    -- residential equity   : wait+fairness   w=[-.1,-.6,-.1,.3,-1.5]
  base      -- everything else      : DEFAULT_W (standard efficiency)

Zone metrics are measured identically for every controller (FT/MP/IDQN/ours)
from SUMO state, independent of the reward features:
  corridor : vehicle-weighted mean speed on the zone's ARTERIAL-direction
             incoming lanes (edge d in {1,3}) (m/s, up)
  gateway  : spillback rate = share of (lane,step) with halting >= 0.8*capacity
             (capacity = lane length / 7.5 m) on zone lanes (down)
  equity   : p95 / mean of per-vehicle current waiting time over vehicles on
             zone lanes, pooled over steps (s, down)
  network  : all-vehicle delay = mean(timeLoss + departDelay) over arrived +
             still-running vehicles (tripinfo write-unfinished) + vehicles never
             inserted (delay = horizon - scheduled depart) (s, down)
"""
import os
import xml.etree.ElementTree as ET

import numpy as np

from composite_reward import DEFAULT_W

ZONE_W = {
    'corridor': np.array([-0.3, -0.1, -0.5, 6.0, 0.0], dtype=np.float32),
    'gateway': np.array([-2.0, -0.03, -0.1, 0.3, 0.0], dtype=np.float32),
    'equity': np.array([-0.1, -0.6, -0.1, 0.3, -1.5], dtype=np.float32),
    'base': DEFAULT_W.astype(np.float32),
}


# ---- v3 zone objectives (speed-free non-mobility zones) -------------------
# Rationale (measured): any non-zero speed weight dominates a
# zone's utility (~90%) because good controllers drive cost features to a few
# % of their reference level while speed stays ~policy-invariant. v3 therefore
# gives the gateway/equity zones NO speed term; relative weights among cost
# features are standardized by the random-policy scale s (feature_scale.json),
# so each listed cost counts equally at the reference level.
_S = np.array([0.43771, 1.22025, 0.13925, 1.83454, 0.26254], dtype=np.float32)
ZONE_W_V3 = {
    'corridor': ZONE_W['corridor'],                                   # unchanged
    'gateway': np.array([-1.0, 0, 0, 0, 0], dtype=np.float32) / np.where(_S > 0, _S, 1) * np.array([1, 0, 0, 0, 0]),
    'equity': np.array([0, -1.0, 0, 0, -1.0], dtype=np.float32) / _S,
    'base': ZONE_W['base'],                                           # = T0
}
ZONE_W_V3 = {k: v.astype(np.float32) for k, v in ZONE_W_V3.items()}

# Benchmark layouts (HZA3/HZB3/ATL3): equity zone is waiting-ONLY. The v3
# equity vector also carried the fairness feature (squared deviation of an
# intersection's waiting from the network mean), which is two-sided: on the
# small benchmarks it rewards making already low-wait intersections wait
# LONGER (measured: HZ equity wait p95 = 96 s wait+fair vs 5 s wait-only vs
# 1936 s fair-only). Dropping the fairness term fixes the direction.
# Manhattan layouts keep ZONE_W_V3 unchanged.
ZONE_W_BENCH = {**ZONE_W_V3,
                'equity': np.array([0, -1.0, 0, 0, 0], dtype=np.float32) / np.where(_S > 0, _S, 1)}
ZONE_W_BENCH = {k: v.astype(np.float32) for k, v in ZONE_W_BENCH.items()}


def _rc(ts):
    _, r, c = ts.rsplit('_', 2)
    return int(r), int(c)


def _layout_A(r, c):
    # Probe (FT/MP, 600 s): load sits on rows 1/2/7 and the west entry column;
    # rows 3-6 interior are near-empty, so zones must sit on loaded streets.
    # west entry gateway (c<=2, incl. the row-2 entrance) | busiest arterial
    # row 2 | southern residential district rows 6-7 | rest standard.
    if c <= 2:
        return 'gateway'
    if r == 2:
        return 'corridor'
    if r >= 6:
        return 'equity'
    return 'base'


def _layout_B(r, c):
    # mirror: east entry gateway (c>=27) | arterial row 7 | northern row 1
    if c >= 27:
        return 'gateway'
    if r == 7:
        return 'corridor'
    if r == 1:
        return 'equity'
    return 'base'


def _layout_C(r, c):
    # C3: maximal heterogeneity, no base/T0 zone (designed post A3/B3, disclosed)
    if c <= 2 or c >= 27: return 'gateway'
    if r == 2: return 'corridor'
    if r in (1, 7): return 'equity'
    return 'equity' if c <= 14 else 'gateway'


LAYOUTS = {'A': _layout_A, 'B': _layout_B, 'C': _layout_C}


def zone_assignment(layout, ids):
    """-> (zone_of {ts: zone}, W [N,5] aligned with ``ids`` order).

    'A'/'B': v2 objectives (ZONE_W). 'A3'/'B3'/'C3': same geometry, v3
    objectives. 'HZA3'/'HZB3'/'ATL3': benchmark zones, ZONE_W_BENCH
    (equity = waiting-only, see note above)."""
    if layout.startswith(('HZ', 'ATL')):
        zone_of, table = bench_zone_of(layout), ZONE_W_BENCH
        zone_of = {ts: zone_of[ts] for ts in ids}
        W = np.stack([table[zone_of[ts]] for ts in ids]).astype(np.float32)
        return zone_of, W
    if layout.startswith('S') and layout[1:].isdigit():
        # Heterogeneity sweep (Fig. 3): a fixed random p% of intersections keep
        # their C3 zone objective, the rest get T0 ('base'). S0 = homogeneous T0,
        # S100 = C3. Fixed permutation (seed 2026) over the sorted grid ids, so
        # the S-layouts are nested (S25 subset of S50 subset of S75).
        p = int(layout[1:]) / 100.0
        grid = sorted(f'intersection_{r}_{c}' for r in range(1, 8) for c in range(1, 29))
        perm = np.random.RandomState(2026).permutation(len(grid))
        keep = {grid[j] for j in perm[:int(round(p * len(grid)))]}
        zone_of = {ts: (_layout_C(*_rc(ts)) if ts in keep else 'base') for ts in ids}
        W = np.stack([ZONE_W_V3[zone_of[ts]] for ts in ids]).astype(np.float32)
        return zone_of, W
    geo, table = (layout[:-1], ZONE_W_V3) if layout.endswith('3') else (layout, ZONE_W)
    fn = LAYOUTS[geo]
    zone_of = {ts: fn(*_rc(ts)) for ts in ids}
    W = np.stack([table[zone_of[ts]] for ts in ids]).astype(np.float32)
    return zone_of, W


_DEPART_CACHE = {}


def _scheduled_departs(route_file):
    if route_file not in _DEPART_CACHE:
        dep = {}
        for _, e in ET.iterparse(route_file):
            if e.tag in ('vehicle', 'trip'):
                dep[e.get('id')] = float(e.get('depart'))
            e.clear()
        _DEPART_CACHE[route_file] = dep
    return _DEPART_CACHE[route_file]


class ZoneMeter:
    """Per-step zone sampler. Call step(sumo) after every env.step, finish(sumo)
    once before env.close(), then summary(tripinfo_path)."""

    def __init__(self, core, ids, zone_of, horizon_s, route_file):
        self.zone_of = zone_of
        self.horizon = float(horizon_s)
        self.route_file = route_file
        sumo = core.sumo
        self.lanes = {}
        for ts in ids:
            z = zone_of[ts]
            self.lanes.setdefault(z, [])
            for ln in dict.fromkeys(core.traffic_signals[ts].lanes):
                self.lanes[z].append(ln)
        self.cap = {ln: max(1.0, sumo.lane.getLength(ln) / 7.5)
                    for lns in self.lanes.values() for ln in lns}
        # arterial-direction lanes. Manhattan: edge road_r_c_d with d in {1,3}
        # (c axis, unchanged). Other benchmarks: lanes of the two
        # highest-flow incoming edges of each signal (bench_stats major).
        scn = os.path.basename(os.path.dirname(os.path.abspath(route_file)))
        self.manhattan = scn == 'manhattan_28x7'
        if self.manhattan:
            self.art = {ln: ln.rsplit('_', 2)[-2] in ('1', '3')
                        for lns in self.lanes.values() for ln in lns}
            self.bst = None
        else:
            self.bst = bench_stats(scn)
            maj = {e for s in ids for e in self.bst['major'].get(s, [])}
            self.art = {ln: ln.rsplit('_', 1)[0] in maj
                        for lns in self.lanes.values() for ln in lns}
        self.backlog = {z: 0.0 for z in self.lanes}
        self.n_steps = 0
        # behaviour probes (Fig. 4, additive): per zone, share of green signal
        # links that serve arterial-direction lanes, and phase-switch rate.
        self.ts_zone = [(ts, zone_of[ts]) for ts in ids]
        self.links = {ts: [l[0][0] if l else None for l in sumo.trafficlight.getControlledLinks(ts)]
                      for ts in ids}
        self.prev_state = {}
        self.beh = {z: {'art_g': 0.0, 'art_n': 0, 'sw': 0, 'sw_n': 0} for z in self.lanes}
        self.acc = {z: {'spd_num': 0.0, 'spd_den': 0.0, 'halt': 0.0,
                        'art_num': 0.0, 'art_den': 0.0,
                        'lane_steps': 0, 'spill': 0, 'waits': []}
                    for z in self.lanes}
        self.pending = []

    def step(self, sumo):
        for z, lns in self.lanes.items():
            a = self.acc[z]
            for ln in lns:
                n = sumo.lane.getLastStepVehicleNumber(ln)
                h = sumo.lane.getLastStepHaltingNumber(ln)
                if n:
                    v_mean = sumo.lane.getLastStepMeanSpeed(ln)
                    a['spd_num'] += v_mean * n
                    a['spd_den'] += n
                    if self.art[ln]:
                        a['art_num'] += v_mean * n
                        a['art_den'] += n
                    for v in sumo.lane.getLastStepVehicleIDs(ln):
                        a['waits'].append(sumo.vehicle.getWaitingTime(v))
                a['halt'] += h
                a['lane_steps'] += 1
                a['spill'] += int(h >= 0.8 * self.cap[ln])

        for ts, z in self.ts_zone:
            st = sumo.trafficlight.getRedYellowGreenState(ts)
            b = self.beh[z]
            g = [i for i, ch in enumerate(st) if ch in 'Gg' and i < len(self.links[ts]) and self.links[ts][i]]
            if g:
                b['art_g'] += sum(self.art.get(self.links[ts][i], False) for i in g) / len(g)
                b['art_n'] += 1
            gs = frozenset(g)
            if ts in self.prev_state:
                b['sw'] += int(gs != self.prev_state[ts]); b['sw_n'] += 1
            self.prev_state[ts] = gs

        # entry backlog per zone: vehicles waiting outside the network whose
        # first edge feeds a signal of that zone (benchmarks only)
        self.n_steps += 1
        if self.bst is not None:
            for vid in sumo.simulation.getPendingVehicles():
                e = self.bst['first'].get(vid) or self.bst['first'].get(vid.split('.')[0])
                z = self.zone_of.get(self.bst['edges'].get(e, (None, None))[1])
                if z in self.backlog:
                    self.backlog[z] += 1

    def finish(self, sumo):
        self.pending = list(sumo.simulation.getPendingVehicles())

    def summary(self, tripinfo_path):
        out = {}
        for z, a in self.acc.items():
            w = np.asarray(a['waits'], dtype=np.float64)
            out[f'{z}_speed'] = a['spd_num'] / max(a['spd_den'], 1e-9)
            out[f'{z}_speed_art'] = a['art_num'] / max(a['art_den'], 1e-9)
            out[f'{z}_queue_lane'] = a['halt'] / max(a['lane_steps'], 1)
            out[f'{z}_spill'] = a['spill'] / max(a['lane_steps'], 1)
            out[f'{z}_wait_mean'] = float(w.mean()) if w.size else 0.0
            out[f'{z}_wait_p95'] = float(np.percentile(w, 95)) if w.size else 0.0
            b = self.beh.get(z)
            if b is not None:
                out[f'{z}_art_green'] = b['art_g'] / max(b['art_n'], 1)
                out[f'{z}_switch_rate'] = b['sw'] / max(b['sw_n'], 1)
            out[f'{z}_backlog'] = self.backlog[z] / max(self.n_steps, 1)
        out.update(network_delay(tripinfo_path, self.pending, self.horizon,
                                 self.route_file))
        return out


def network_delay(tripinfo_path, pending_ids, horizon_s, route_file):
    """All-vehicle delay incl. unfinished and never-inserted vehicles."""
    delays, arrived, running = [], 0, 0
    if tripinfo_path and os.path.exists(tripinfo_path):
        for _, e in ET.iterparse(tripinfo_path):
            if e.tag == 'tripinfo':
                delays.append(float(e.get('timeLoss')) + float(e.get('departDelay')))
                if float(e.get('arrival')) >= 0:
                    arrived += 1
                else:
                    running += 1
            e.clear()
    dep = _scheduled_departs(route_file)
    miss = 0
    for vid in pending_ids:
        d = dep.get(vid)
        if d is None:
            d = dep.get(vid.split('.')[0])
        if d is None:
            miss += 1
            continue
        delays.append(max(0.0, horizon_s - d))
    d = np.asarray(delays, dtype=np.float64)
    return {'net_delay': float(d.mean()) if d.size else float('nan'),
            'n_arrived': arrived, 'n_running': running,
            'n_pending': len(pending_ids), 'n_pending_unmatched': miss}


# =========================================================================== #
# Other benchmarks (Hangzhou 4x4, Atlanta 1x5): zones DERIVED from topology +
# route-file demand by fixed rules (frozen before any benchmark run).
#   edge_flow[e]  = #routes using e ; entry_flow[e] = #routes starting on e
#   major(s)      = the 2 incoming edges of signal s with the largest flow
#   minor_share(s)= 1 - flow(major(s)) / flow(all incoming edges of s)
# Hangzhou (grid): corridor = busiest straight line (row/col, axial flow);
#   gateway = boundary side with the largest entry volume (side != corridor
#   line; crossing signal goes to gateway); equity = top-3 minor_share among
#   the rest; base = rest. Layout B: corridor = busiest line of the OTHER axis,
#   gateway = 2nd-largest entry side. Atlanta (5-signal arterial chain):
#   gateway = the two chain ends; equity = middle signal with max minor_share;
#   corridor = the other two middle signals (conflicting corridor).
# All benchmark layouts use the v3 objectives (ZONE_W_V3).
# =========================================================================== #
BENCH_SCN = {'HZ': 'hangzhou_4x4_gudang_18041610_1h', 'ATL': 'atlanta_1x5'}
ATL_CHAIN = ['69421277', '69249210', '69387071', '69227168', '69515842']
_BSTAT = {}


def bench_stats(scn):
    if scn in _BSTAT:
        return _BSTAT[scn]
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'raw_data', scn, scn)
    edges, lanes, tls = {}, {}, set()
    for _, e in ET.iterparse(base + '.net.xml'):
        if e.tag == 'edge' and e.get('function') != 'internal' and e.get('from'):
            edges[e.get('id')] = (e.get('from'), e.get('to'))
            lanes[e.get('id')] = [l.get('id') for l in e.findall('lane')]
        elif e.tag == 'tlLogic':
            tls.add(e.get('id'))
        if e.tag in ('edge', 'tlLogic', 'junction'):
            e.clear()
    flow, entry, first = {}, {}, {}
    cur = None
    for ev, e in ET.iterparse(base + '.rou.xml', events=('start', 'end')):
        if ev == 'start' and e.tag == 'vehicle':
            cur = e.get('id')
        elif ev == 'end' and e.tag == 'route' and e.get('edges'):
            es = e.get('edges').split()
            for x in es:
                flow[x] = flow.get(x, 0) + 1
            entry[es[0]] = entry.get(es[0], 0) + 1
            if cur is not None:
                first[cur] = es[0]
        elif ev == 'end' and e.tag == 'vehicle':
            cur = None
            e.clear()
    inc = {s: [x for x, (fr, to) in edges.items() if to == s] for s in tls}
    major, minor_share, entry_vol = {}, {}, {}
    for s in tls:
        fl = sorted(inc[s], key=lambda x: -flow.get(x, 0))
        major[s] = fl[:2]
        tot = sum(flow.get(x, 0) for x in fl)
        minor_share[s] = 1 - sum(flow.get(x, 0) for x in fl[:2]) / tot if tot else 0.0
        entry_vol[s] = sum(entry.get(x, 0) for x in inc[s])
    st = dict(edges=edges, lanes=lanes, tls=sorted(tls), flow=flow, entry=entry, first=first,
              major=major, minor_share=minor_share, entry_vol=entry_vol, inc=inc)
    _BSTAT[scn] = st
    return st


def _hz_zones(variant):
    st = bench_stats(BENCH_SCN['HZ'])
    fl = st['flow']
    R = C = 4
    row = {r: sum(fl.get(f'road_{r}_{c}_{d}', 0) for c in range(0, C + 2) for d in (1, 3)) for r in range(1, R + 1)}
    col = {c: sum(fl.get(f'road_{r}_{c}_{d}', 0) for r in range(0, R + 2) for d in (0, 2)) for c in range(1, C + 1)}
    lines = sorted([(v, 'row', r) for r, v in row.items()] + [(v, 'col', c) for c, v in col.items()], reverse=True)
    best = lines[0]
    corr = best if variant == 'A' else next(l for l in lines if l[1] != best[1])
    cs = {f'intersection_{corr[2]}_{c}' for c in range(1, C + 1)} if corr[1] == 'row' else \
         {f'intersection_{r}_{corr[2]}' for r in range(1, R + 1)}
    sides = {'N': {f'intersection_1_{c}' for c in range(1, C + 1)},
             'S': {f'intersection_{R}_{c}' for c in range(1, C + 1)},
             'W': {f'intersection_{r}_1' for r in range(1, R + 1)},
             'E': {f'intersection_{r}_{C}' for r in range(1, R + 1)}}
    ev = st['entry_vol']
    ranked = sorted([k for k, v in sides.items() if v != cs], key=lambda k: -sum(ev[s] for s in sides[k]))
    gate = sides[ranked[0] if variant == 'A' else ranked[1]]
    zone = {}
    for s in st['tls']:
        zone[s] = 'gateway' if s in gate else ('corridor' if s in cs else None)
    rest = sorted([s for s in st['tls'] if zone[s] is None], key=lambda s: -st['minor_share'][s])
    for i, s in enumerate(rest):
        zone[s] = 'equity' if i < 3 else 'base'
    return zone, dict(corridor=corr, gateway_side=ranked[0] if variant == 'A' else ranked[1])


def _atl_zones():
    st = bench_stats(BENCH_SCN['ATL'])
    zone = {ATL_CHAIN[0]: 'gateway', ATL_CHAIN[-1]: 'gateway'}
    mid = ATL_CHAIN[1:-1]
    eq = max(mid, key=lambda s: st['minor_share'][s])
    for s in mid:
        zone[s] = 'equity' if s == eq else 'corridor'
    return zone, dict(equity=eq)


def bench_zone_of(layout):
    """layout in {'HZA3', 'HZB3', 'ATL3'} -> {tl_id: zone}."""
    if layout.startswith('HZ'):
        return _hz_zones(layout[2])[0]
    if layout.startswith('ATL'):
        return _atl_zones()[0]
    raise KeyError(layout)
