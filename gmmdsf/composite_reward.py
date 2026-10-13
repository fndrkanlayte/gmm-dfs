"""Composite traffic objective r = phi . w and task samplers.

The five normalized per-intersection outcome features phi (queue, waiting time,
pressure, speed, fairness) are the successor features; a task is a signed
weight vector w over them (homogeneous) or one w_i per intersection
(heterogeneous). This module also defines the training task families used to
pre-train the GMM-DSF backbone.

    r = - w_queue * queue_norm
        - w_wait  * wait_norm
        - w_press * pressure_norm
        + w_speed * speed_norm
        - w_fair  * fairness_norm

All terms are normalized to ~[0,1] per intersection (see the per-lane maxima
below) so the weights below are the true relative importances. Weights and
normalization constants are fixed across all experiments.

Usage
-----
    from composite_reward import make_composite_reward
    reward_fn = make_composite_reward()      # fresh, self-contained cache
    env = sumo_rl.parallel_env(..., reward_fn=reward_fn)

`make_composite_reward` returns a *callable* with its own closed-over component
and fairness caches, so building several environments never cross-contaminates
state (unlike the old module-global version). The returned callable also carries
a ``.component_cache`` attribute for callers that want the raw per-intersection
term values, but the current agent.py collects its physical metrics from the
env's ``system_*`` info dict and does NOT need it.
"""
import numpy as np

# ---- term weights (verbatim from signal/old/v4) ----
W_QUEUE = 1.0
W_WAIT = 0.3
W_PRESSURE = 1.5
W_SPEED = 3.0
W_FAIR = 0.0

# ---- per-lane normalization maxima (verbatim from signal/old/v4) ----
MAX_QUEUE_PER_LANE = 50       # max queued vehicles / lane
MAX_WAIT_PER_LANE = 1000      # max accumulated waiting time / lane (s)
MAX_PRESSURE_PER_LANE = 20    # max flow / lane / step (veh)
MAX_SPEED = 15.0             # free-flow speed (m/s)


def make_composite_reward():
    """Return a fresh composite reward_fn(ts) with its own cache state."""
    component_cache = {}
    fairness_cache = {'sim_step': -1, 'avg_wait': 0.0}

    def reward_fn(ts):
        lanes_count = len(ts.lanes)

        # queue length
        queue = ts.get_total_queued()
        queue_norm = queue / (lanes_count * MAX_QUEUE_PER_LANE)

        # total accumulated waiting time
        waiting = sum(ts.sumo.lane.getWaitingTime(l) for l in ts.lanes)
        wait_norm = waiting / (lanes_count * MAX_WAIT_PER_LANE)

        # pressure via inflow/outflow differencing (per-ts persistent counters)
        current_in = sum(ts.sumo.lane.getLastStepVehicleNumber(l) for l in ts.lanes)
        out_lanes = getattr(ts, 'out_lanes', [])
        current_out = sum(ts.sumo.lane.getLastStepVehicleNumber(l) for l in out_lanes)
        if not hasattr(ts, '_prev_in'):
            ts._prev_in = current_in
            ts._prev_out = current_out
        inflow = max(0, current_in - ts._prev_in)
        outflow = max(0, current_out - ts._prev_out)
        ts._prev_in, ts._prev_out = current_in, current_out
        pressure = abs(inflow - outflow)
        pressure_norm = pressure / (lanes_count * MAX_PRESSURE_PER_LANE)

        # average speed (vehicle-weighted over the intersection's lanes)
        if hasattr(ts, 'get_average_speed'):
            speed = ts.get_average_speed()
        else:
            total_speed = 0.0
            veh_cnt = 0
            for lane in ts.lanes:
                veh = ts.sumo.lane.getLastStepVehicleNumber(lane)
                if veh > 0:
                    total_speed += ts.sumo.lane.getLastStepMeanSpeed(lane) * veh
                    veh_cnt += veh
            speed = total_speed / veh_cnt if veh_cnt > 0 else 0.0
        speed_norm = speed / MAX_SPEED

        # fairness: squared deviation of this ts's waiting from the net-wide mean
        current_step = int(ts.sumo.simulation.getTime() / ts.delta_time)
        if current_step < fairness_cache.get('sim_step', -1):   # new episode -> reset
            fairness_cache.clear()
            fairness_cache['sim_step'] = -1
            fairness_cache['avg_wait'] = 0.0
        if fairness_cache.get('sim_step') != current_step:
            if hasattr(ts, 'env') and hasattr(ts.env, 'traffic_signals'):
                all_ts = list(ts.env.traffic_signals.values())
                if len(all_ts) > 1:
                    waitings = [sum(sig.sumo.lane.getWaitingTime(l) for l in sig.lanes)
                                for sig in all_ts]
                    fairness_cache['avg_wait'] = float(np.mean(waitings))
                else:
                    fairness_cache['avg_wait'] = waiting
            else:
                fairness_cache['avg_wait'] = waiting
            fairness_cache['sim_step'] = current_step

        deviation = waiting - fairness_cache['avg_wait']
        fairness_penalty = deviation ** 2
        fairness_denom = (lanes_count * MAX_WAIT_PER_LANE) ** 2
        fairness_norm = fairness_penalty / fairness_denom if fairness_denom > 0 else 0.0
        fairness_norm = min(fairness_norm, 1.0)

        reward = (
            - W_QUEUE * queue_norm
            - W_WAIT * wait_norm
            - W_PRESSURE * pressure_norm
            + W_SPEED * speed_norm
            - W_FAIR * fairness_norm
        )

        component_cache[ts.id] = {
            'queue': queue, 'waiting': waiting, 'pressure': pressure,
            'speed': speed, 'fairness': fairness_penalty,
        }
        return reward

    reward_fn.component_cache = component_cache
    return reward_fn


# ===================================================================== #
# Successor-feature decomposition of the composite reward
# ===================================================================== #
# The composite reward is ALREADY a linear combination of normalized
# congestion signals:
#     r = phi . w,   phi = [queue, wait, pressure, speed, fairness]_norm
#     w   = [-W_QUEUE, -W_WAIT, -W_PRESSURE, +W_SPEED, -W_FAIR]
# so the five normalized terms are exactly the successor features phi and the
# signed term weights are the task vector w. Exposing phi lets us (a) learn
# w-free successor features psi (USFA) whose value transfers to any weighting by
# the closed form Q = psi . w, and (b) retrain a per-task baseline on r = phi.w
# for any specific w. This is the SF hook the paper's TSC experiment needs; the
# scalar-reward IDDQN path above is untouched (it stays the per-task baseline).

FEATURE_NAMES = ("queue", "wait", "pressure", "speed", "fairness")
FEATURE_DIM = len(FEATURE_NAMES)

# The default task vector: signs + importances that reproduce the scalar
# composite reward EXACTLY (phi . DEFAULT_W == the reward above).
DEFAULT_W = np.array([-W_QUEUE, -W_WAIT, -W_PRESSURE, +W_SPEED, -W_FAIR],
                     dtype=np.float32)


def _phi_of_ts(ts, fairness_cache):
    """Compute the 5-dim normalized successor-feature vector for one signal.

    Identical term math to ``make_composite_reward`` (queue / wait / pressure /
    speed / fairness), returned as a raw feature vector phi in [~0, 1]^5 BEFORE
    any weighting -- the weighting is applied by the caller as phi . w."""
    lanes_count = len(ts.lanes)

    queue = ts.get_total_queued()
    queue_norm = queue / (lanes_count * MAX_QUEUE_PER_LANE)

    waiting = sum(ts.sumo.lane.getWaitingTime(l) for l in ts.lanes)
    wait_norm = waiting / (lanes_count * MAX_WAIT_PER_LANE)

    current_in = sum(ts.sumo.lane.getLastStepVehicleNumber(l) for l in ts.lanes)
    out_lanes = getattr(ts, 'out_lanes', [])
    current_out = sum(ts.sumo.lane.getLastStepVehicleNumber(l) for l in out_lanes)
    if not hasattr(ts, '_prev_in'):
        ts._prev_in = current_in
        ts._prev_out = current_out
    inflow = max(0, current_in - ts._prev_in)
    outflow = max(0, current_out - ts._prev_out)
    ts._prev_in, ts._prev_out = current_in, current_out
    pressure = abs(inflow - outflow)
    pressure_norm = pressure / (lanes_count * MAX_PRESSURE_PER_LANE)

    if hasattr(ts, 'get_average_speed'):
        speed = ts.get_average_speed()
    else:
        total_speed, veh_cnt = 0.0, 0
        for lane in ts.lanes:
            veh = ts.sumo.lane.getLastStepVehicleNumber(lane)
            if veh > 0:
                total_speed += ts.sumo.lane.getLastStepMeanSpeed(lane) * veh
                veh_cnt += veh
        speed = total_speed / veh_cnt if veh_cnt > 0 else 0.0
    speed_norm = speed / MAX_SPEED

    current_step = int(ts.sumo.simulation.getTime() / ts.delta_time)
    if current_step < fairness_cache.get('sim_step', -1):
        fairness_cache.clear()
        fairness_cache['sim_step'] = -1
        fairness_cache['avg_wait'] = 0.0
    if fairness_cache.get('sim_step') != current_step:
        if hasattr(ts, 'env') and hasattr(ts.env, 'traffic_signals'):
            all_ts = list(ts.env.traffic_signals.values())
            if len(all_ts) > 1:
                waitings = [sum(sig.sumo.lane.getWaitingTime(l) for l in sig.lanes)
                            for sig in all_ts]
                fairness_cache['avg_wait'] = float(np.mean(waitings))
            else:
                fairness_cache['avg_wait'] = waiting
        else:
            fairness_cache['avg_wait'] = waiting
        fairness_cache['sim_step'] = current_step
    deviation = waiting - fairness_cache['avg_wait']
    fairness_denom = (lanes_count * MAX_WAIT_PER_LANE) ** 2
    fairness_norm = (deviation ** 2) / fairness_denom if fairness_denom > 0 else 0.0
    fairness_norm = min(fairness_norm, 1.0)

    return np.array([queue_norm, wait_norm, pressure_norm, speed_norm,
                     fairness_norm], dtype=np.float32)


def make_sf_reward(w=None):
    """Return an SF reward_fn(ts) = phi(ts) . w that also caches phi per signal.

    ``w`` is the (signed) task vector over FEATURE_NAMES; ``None`` uses
    ``DEFAULT_W`` (which reproduces the scalar composite reward exactly). The
    returned callable carries a ``.feature_cache`` dict mapping ts.id -> the raw
    phi vector of the latest step, so the training agent can read phi for the
    SF-TD update after each env step. Pass a DIFFERENT ``w`` (or resample it per
    episode via ``.set_w``) to instantiate a new task in the same environment.

    For the w-HETEROGENEOUS scenario, call ``.set_w_map({ts_id: w_row})`` to give
    each intersection its OWN task vector; when a w-map is set the per-ts reward
    is ``phi(ts) . w_map[ts.id]`` (falling back to the shared w for any id not in
    the map). This keeps the environment's reward signal consistent with the
    per-agent w the zero-shot / composer rules optimize."""
    w = DEFAULT_W.copy() if w is None else np.asarray(w, dtype=np.float32)
    state = {'w': w, 'w_map': None}
    feature_cache = {}
    fairness_cache = {'sim_step': -1, 'avg_wait': 0.0}

    def reward_fn(ts):
        phi = _phi_of_ts(ts, fairness_cache)
        feature_cache[ts.id] = phi
        wmap = state['w_map']
        w_here = wmap[ts.id] if (wmap is not None and ts.id in wmap) else state['w']
        return float(phi @ w_here)

    def set_w(new_w):
        state['w'][:] = np.asarray(new_w, dtype=np.float32)

    def set_w_map(w_map):
        """Per-intersection weights: dict ts_id -> length-K vector (or None to
        revert to the shared homogeneous w)."""
        state['w_map'] = (None if w_map is None else
                          {k: np.asarray(v, dtype=np.float32) for k, v in w_map.items()})

    reward_fn.feature_cache = feature_cache
    reward_fn.set_w = set_w
    reward_fn.set_w_map = set_w_map
    reward_fn.w = state['w']
    return reward_fn


def sample_task(rng, base=None, sign=None, scale=(0.5, 1.5)):
    """Sample a random task vector for w-free (USFA) SF training.

    Keeps the FIXED signs of DEFAULT_W (queue/wait/pressure/fairness are costs,
    speed is a gain) and only jitters the per-term IMPORTANCE magnitudes within
    ``scale`` x their default, so every sampled task is a physically sensible
    congestion objective. Returns a length-FEATURE_DIM float32 vector."""
    base = np.abs(DEFAULT_W) if base is None else np.abs(base)
    sign = np.sign(DEFAULT_W) if sign is None else sign
    lo, hi = scale
    mag = base * rng.uniform(lo, hi, size=FEATURE_DIM)
    return (sign * mag).astype(np.float32)


def sample_task_family(rng, base=None, sign=None, mix=(0.30, 0.45, 0.15, 0.10)):
    """Mixture task sampler covering the TSC deployment task family.

    Every draw keeps DEFAULT_W signs, so each w is a physically meaningful
    congestion objective. Modes (all feature dims are queue/wait/pressure/
    speed/fairness):

      mix[0]  efficiency cone   -- original jitter, all dims base*U(0.5,1.5)
      mix[1]  axis-dominant     -- ONE congestion dim (queue/wait/pressure)
                                   gets base*U(0.7,2.0), others base*U(0,0.15).
                                   Spillback prevention / delay reduction /
                                   stability tasks (T1-family corners).
      mix[2]  speed emphasis    -- speed gets base*U(1.0,2.2) (covers
                                   speed=6.0), others base*U(0.3,1.0).
                                   Arterial progression / T2-family.
      mix[3]  wide cone         -- all dims base*U(0.2,2.0), mixed coverage.

    ``mix`` is a 4-tuple of mode probabilities (default the aggressive
    30/45/15/10 blend). Milder variant for fine-tuning a converged model:
    mix=(0.70, 0.15, 0.10, 0.05).
    """
    base = np.abs(DEFAULT_W) if base is None else np.abs(base)
    sign = np.sign(DEFAULT_W) if sign is None else sign
    p = np.cumsum(np.asarray(mix, dtype=np.float64))
    u = rng.uniform()
    if u < p[0]:
        mag = base * rng.uniform(0.5, 1.5, size=FEATURE_DIM)
    elif u < p[1]:
        mag = base * rng.uniform(0.0, 0.15, size=FEATURE_DIM)
        dom = rng.randint(0, 3)          # queue / wait / pressure axis
        mag[dom] = base[dom] * rng.uniform(0.7, 2.0)
    elif u < p[2]:
        mag = base * rng.uniform(0.3, 1.0, size=FEATURE_DIM)
        mag[3] = base[3] * rng.uniform(1.0, 2.2)
    else:
        mag = base * rng.uniform(0.2, 2.0, size=FEATURE_DIM)
    return (sign * mag).astype(np.float32)


# Fairness has weight 0 in DEFAULT_W, so base*U(.) would never activate it.
# FAIR_BASE gives the equity dimension its own magnitude (fairness_norm is
# heavy-tailed: median ~0.005, hotspots ~1-8 per discounted episode, i.e.
# comparable to the wait term exactly where an equity objective should act).
FAIR_BASE = 1.0


def sample_task_family_v2(rng, mix=(0.25, 0.30, 0.15, 0.20, 0.10)):
    """Task family v2 = v1 + an EQUITY mode (wait + fairness dominant).

    Modes: cone / axis(queue|wait|pressure) / speed / equity / wide. Signs
    fixed (costs negative, speed positive) -> every draw is a legitimate
    TSC objective. Fairness is only non-zero in the equity and wide modes.
    """
    base = np.abs(DEFAULT_W).astype(np.float64)
    base[4] = FAIR_BASE
    sign = np.sign(DEFAULT_W).astype(np.float64)
    sign[4] = -1.0
    p = np.cumsum(np.asarray(mix, dtype=np.float64))
    u = rng.uniform()
    if u < p[0]:                                   # efficiency cone
        mag = base * rng.uniform(0.5, 1.5, size=FEATURE_DIM)
        mag[4] = 0.0
    elif u < p[1]:                                 # axis-dominant
        mag = base * rng.uniform(0.0, 0.15, size=FEATURE_DIM)
        mag[4] = 0.0
        dom = rng.randint(0, 3)
        mag[dom] = base[dom] * rng.uniform(0.7, 2.0)
    elif u < p[2]:                                 # speed emphasis
        mag = base * rng.uniform(0.3, 1.0, size=FEATURE_DIM)
        mag[4] = 0.0
        mag[3] = base[3] * rng.uniform(1.0, 2.2)
    elif u < p[3]:                                 # equity
        mag = base * rng.uniform(0.0, 0.3, size=FEATURE_DIM)
        mag[1] = base[1] * rng.uniform(0.7, 2.0)
        mag[4] = base[4] * rng.uniform(0.5, 2.0)
    else:                                          # wide
        mag = base * rng.uniform(0.2, 2.0, size=FEATURE_DIM)
        mag[4] = base[4] * rng.uniform(0.0, 1.0)
    return (sign * mag).astype(np.float32)


def _grid_rc(ts_id):
    """'intersection_<r>_<c>' -> (r, c) ints."""
    _, r, c = ts_id.rsplit('_', 2)
    return int(r), int(c)


FEATURE_SCALE = np.array([0.43771, 1.22025, 0.13925, 1.83454, 0.26254])  # feature_scale.json


def sample_task_family_v3(rng, p_cost=0.25):
    """v2 + a SPEED-FREE cost-only mode (prob p_cost).

    Cost-only: 1-2 of {queue, wait, pressure, fairness} get standardized
    weight U(0.5,1.5) (raw w_k = -w~_k / s_k, s = random-policy scale), all
    other weights incl. speed are exactly 0. Needed because any non-zero speed
    weight dominates realized utility (~90%), so v2 never contains objectives
    that are cost-dominated in effect.
    """
    if rng.uniform() >= p_cost:
        return sample_task_family_v2(rng)
    dims = [0, 1, 2, 4]
    k = rng.randint(1, 3)
    chosen = rng.choice(dims, size=k, replace=False)
    w = np.zeros(FEATURE_DIM)
    for d in chosen:
        w[d] = -rng.uniform(0.5, 1.5) / FEATURE_SCALE[d]
    return w.astype(np.float32)


def sample_block_tasks(rng, ids, n_blocks=(3, 8), sampler=None):
    """Spatially STRUCTURED random task matrix for training.

    The grid is cut into k random axis-aligned rectangles (recursive random
    splits); every intersection in a rectangle shares one w ~ family v2.
    Random every episode -> never equals the fixed evaluation zone layouts,
    but teaches psi the 'neighbours share my objective' regime that zoned
    deployment exhibits (the i.i.d. per-agent sampler never shows it).
    """
    try:
        rc = np.array([_grid_rc(t) for t in ids])
    except ValueError:
        from zones import ATL_CHAIN
        order = ATL_CHAIN if set(ids) <= set(ATL_CHAIN) else sorted(ids)
        rc = np.array([(0, order.index(t)) for t in ids])
    rects = [(rc[:, 0].min(), rc[:, 0].max(), rc[:, 1].min(), rc[:, 1].max())]
    k = rng.randint(n_blocks[0], n_blocks[1] + 1)
    while len(rects) < k:
        i = int(np.argmax([(r1 - r0 + 1) * (c1 - c0 + 1) for r0, r1, c0, c1 in rects]))
        r0, r1, c0, c1 = rects.pop(i)
        can_c, can_r = c1 > c0, r1 > r0
        if can_c and (not can_r or rng.uniform() < 0.5):
            cut = rng.randint(c0, c1)
            rects += [(r0, r1, c0, cut), (r0, r1, cut + 1, c1)]
        elif can_r:
            cut = rng.randint(r0, r1)
            rects += [(r0, cut, c0, c1), (cut + 1, r1, c0, c1)]
        else:
            rects.append((r0, r1, c0, c1))
            break
    W = np.zeros((len(ids), FEATURE_DIM), dtype=np.float32)
    sampler = sampler or sample_task_family_v2
    for r0, r1, c0, c1 in rects:
        w = sampler(rng)
        m = (rc[:, 0] >= r0) & (rc[:, 0] <= r1) & (rc[:, 1] >= c0) & (rc[:, 1] <= c1)
        W[m] = w
    return W


def sample_hetero_task(rng, num_agents, scale=(0.5, 1.5)):
    """Random PER-INTERSECTION task matrix W of shape [num_agents, FEATURE_DIM].

    Each intersection draws its OWN importance magnitudes independently (fixed
    DEFAULT_W signs), so the network is asked to optimize a spatially
    heterogeneous objective -- different corners of the city care about
    different congestion terms. This is the w-heterogeneous TSC scenario: the
    closed-form transfer must serve every node's own w simultaneously, and the
    neighbour composer can additionally condition on neighbours' w."""
    return np.stack([sample_task(rng, scale=scale) for _ in range(num_agents)],
                    axis=0).astype(np.float32)
