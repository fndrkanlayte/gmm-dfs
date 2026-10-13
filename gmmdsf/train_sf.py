"""GMM-DSF training / deployment entry point.

Phases (``--phase``):

  * ``usfa``        -- pre-train the task-free outcome model. With
                       ``--sf_net gmm`` this is the Gaussian-mixture distributional
                       successor-feature (GMM-DSF) backbone; ``--sf_net sf`` trains
                       the point-estimate successor features used as warm start.
                       A fresh task weight w (or per-intersection w_i, see
                       ``--task_family``) is drawn every episode.
  * ``independent`` -- zero-shot deployment of a frozen backbone on a target task
                       (homogeneous ``--task``, or heterogeneous ``--zone_layout``):
                       every intersection picks argmax_a E[U_i(Z_i(s_i, a))].
                       ``--utility`` selects a linear or closed-form non-linear
                       utility (box threshold / Gaussian target).
  * ``ma-usfa``     -- frozen backbone + bounded one-hop neighbour composer,
                       fine-tuned on the target task (optionally gated with
                       ``--composer_gate 1`` or scale-normalised with
                       ``--composer_norm 1``).

``transfer`` / ``composer`` are accepted as aliases of independent / ma-usfa.
Full task-specific retraining baselines live in ``baselines/``.
"""
import warnings
warnings.filterwarnings('ignore')
import os
os.environ["LIBSUMO_AS_TRACI"] = "1"

import argparse
import json
import numpy as np
import torch
import tqdm
import sumo_rl

from agent import SFAgent
from utils import parse_tripinfo, set_seed
from composite_reward import (make_sf_reward, DEFAULT_W, FEATURE_NAMES,
                              sample_task, sample_hetero_task,
                              sample_task_family)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', 'data', 'raw_data')


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', type=str, default='usfa',
                   choices=['usfa', 'independent', 'ma-usfa',
                            'transfer', 'composer'],
                   help="usfa: pre-train the task-free outcome model. "
                        "independent: zero-shot deployment of the frozen "
                        "backbone on the target task. ma-usfa: frozen backbone "
                        "+ learned neighbour composer fine-tuned on the target "
                        "task. (transfer/composer are aliases.)")
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--max_step', type=int, default=60)
    p.add_argument('--episodes', type=int, default=500)
    p.add_argument('--eval_episode', type=int, default=5)
    p.add_argument('--eval_episodes', type=int, default=5,
                   help='rollouts averaged for transfer/composer evaluation. '
                        'Increase to e.g. 10 for multi-rollout mean/std reporting '
                        'without retraining (single-seed variance estimate).')
    p.add_argument('--cpu', action='store_true', default=False)
    p.add_argument('--cuda', type=str, default='0')
    p.add_argument('--hidden_dim', type=int, default=64)
    p.add_argument('--n_heads', type=int, default=4)
    p.add_argument('--use_agent_emb', type=int, default=1)
    p.add_argument('--buffer_size', type=int, default=100000)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--updates_per_step', type=int, default=1)
    p.add_argument('--ckpt_every', type=int, default=0,
                   help='save psi snapshot every N epochs to run_dir '
                        '(0=off). Used for W_val checkpoint selection.')
    p.add_argument('--target_update_freq', type=int, default=200)
    p.add_argument('--warmup', type=int, default=1000)
    p.add_argument('--eps_start', type=float, default=1.0)
    p.add_argument('--eps_end', type=float, default=0.05)
    p.add_argument('--eps_decay_episodes', type=int, default=300)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--task_family', type=int, default=0,
                   help='usfa training only: use sample_task_family mixture '
                        '(axis/wide/speed TSC task coverage) instead of the '
                        'DEFAULT_W cone sampler.')
    p.add_argument('--run_suffix', type=str, default='',
                   help='optional tag inserted into run_dir name (ablation arms '
                        'that would otherwise collide, e.g. lm5).')
    p.add_argument('--sf_ckpt', type=str, default=None,
                   help='trained psi checkpoint (required for transfer/composer).')
    p.add_argument('--sf_init', type=str, default=None,
                   help='gmm only: path to an SFQNet checkpoint to warm-start '
                        'the backbone + mean head from (mixture mean == psi '
                        'at step 0). Ignored for sf_net=sf.')
    p.add_argument('--freeze_mean', type=int, default=0,
                   help='gmm only: freeze backbone + every mu head row; train '
                        'only pi logits + log-sigma (V2 arm — the pure '
                        'distribution-learning variant; needs --sf_init).')
    p.add_argument('--task', type=str, default=None,
                   help="target task w as comma-separated floats over "
                        f"{FEATURE_NAMES}; default = DEFAULT_W. Ignored in usfa.")
    p.add_argument('--demand_scale', type=float, default=None,
                   help='SUMO --scale (default: 2.0 Manhattan, 1.0 others).')
    p.add_argument('--scenario', type=str, default='manhattan_28x7',
                   help='SUMO scenario under ../data/raw_data/<name>/ '
                        'with <name>.net.xml + <name>.rou.xml.')
    p.add_argument('--task_seed', type=int, default=None,
                   help='if set, sample the target task from this seed instead '
                        'of --task (for a batch of random target tasks).')
    p.add_argument('--hetero', type=int, default=0,
                   help='1: w-HETEROGENEOUS target task -- draw an independent '
                        'per-intersection w (random per node, fixed DEFAULT_W '
                        'signs). 0: w-homogeneous (one shared w). Uses --task_seed '
                        '(or --seed) as the generator seed. Ignored in usfa.')
    p.add_argument('--sf_net', type=str, default='sf', choices=['sf', 'gmm'],
                   help="outcome representation: 'sf' = original mean psi "
                        "(SFQNet); 'gmm' = Gaussian-mixture outcome "
                        "distribution (GMMQNet, distributional TD). "
                        "'sf' reproduces the original behaviour exactly.")
    p.add_argument('--n_comp', type=int, default=3,
                   help='GMM mixture components (sf_net=gmm only).')
    p.add_argument('--nll_samples', type=int, default=16,
                   help='target-mixture samples per transition for the GMM '
                        'likelihood TD update.')
    p.add_argument('--lambda_nll', type=float, default=1.0,
                   help='weight of the distributional NLL term (gmm).')
    p.add_argument('--lambda_mean', type=float, default=1.0,
                   help='weight of the mean-anchor SF-TD term (gmm).')
    p.add_argument('--ent_reg', type=float, default=0.0,
                   help='mixture-weight entropy bonus, anti-collapse (gmm).')
    p.add_argument('--sigma_min', type=float, default=1e-3,
                   help='floor for GMM component std (gmm).')
    p.add_argument('--utility', type=str, default='linear',
                   choices=['linear', 'box', 'box_mean', 'gauss'],
                   help='deployment-time utility over the outcome dist (gmm '
                        'eval only): linear=E[w.Z]; box=E[w.Z*1{Z<=delta}] '
                        '(threshold safety); gauss=E[exp(-||Z-goal||^2/beta)] '
                        '(precision target). Ignored for sf_net=sf.')
    p.add_argument('--util_delta', type=str, default=None,
                   help='box utility bound delta, scalar or K comma-floats.')
    p.add_argument('--util_goal', type=str, default=None,
                   help='gauss utility goal g, K comma-floats.')
    p.add_argument('--util_beta', type=float, default=1.0,
                   help='gauss utility sharpness beta.')
    p.add_argument('--tripinfo', type=int, default=1)
    p.add_argument('--obs', type=str, default='default', choices=['default', 'downstream'],
                   help="'downstream': default obs + outgoing-lane density "
                        "(obs_baselines.DownstreamObservation).")
    p.add_argument('--zone_layout', type=str, default=None,
                   help='city-zone task (zones.LAYOUTS): W from the layout + '
                        'zone metrics. Overrides --task/--hetero.')
    p.add_argument('--meter_layout', type=str, default=None,
                   help='record zone/all-vehicle metrics for this layout '
                        'without changing the task.')
    p.add_argument('--blind', type=int, default=0,
                   help='1: task-blind control (agent acts with DEFAULT_W) '
                        'while the reward is scored on the given task.')
    p.add_argument('--composer_norm', type=int, default=0,
                   help='1: objective-scale-normalised composer (per-agent TD '
                        'scale, residual bounded in per-agent units).')
    p.add_argument('--composer_gate', type=int, default=0,
                   help='1: speed-gated composer (neighbour residual only for '
                        'agents with positive speed weight w[3]).')
    p.add_argument('--composer_ckpt', type=str, default=None,
                   help='with --episodes 0: load this composer.pth instead of '
                        'run_dir/composer.pth (eval under other seeds).')
    p.add_argument('--out_tag', type=str, default='',
                   help='suffix for tripinfo/result files (concurrent evals).')
    args = p.parse_args()
    _phase_alias = {'independent': 'transfer', 'ma-usfa': 'composer'}
    args.phase = _phase_alias.get(args.phase, args.phase)
    return args


def parse_task(args, num_agents=None):
    """Target task for transfer/composer.

    w-homogeneous -> a single [K] vector (from --task, --task_seed, or DEFAULT_W).
    w-heterogeneous (--hetero 1) -> a [num_agents, K] matrix, one independent w
    per intersection (random magnitudes, fixed signs)."""
    if args.hetero:
        assert num_agents is not None, "hetero task needs num_agents"
        seed = args.task_seed if args.task_seed is not None else args.seed
        if args.hetero == 2:
            # multi-task COMBINATION: each intersection draws an independent
            # task-family w (axis/speed/cone mix) -- matches tfh training.
            rng = np.random.RandomState(seed)
            return np.stack([sample_task_family(rng)
                             for _ in range(num_agents)]).astype(np.float32)
        return sample_hetero_task(np.random.RandomState(seed), num_agents)
    if args.task_seed is not None:
        return sample_task(np.random.RandomState(args.task_seed))
    if args.task:
        return np.array([float(x) for x in args.task.split(',')], dtype=np.float32)
    return DEFAULT_W.copy()


def _sf_kwargs(args):
    """GMM-related kwargs splatted into every SFAgent construction."""
    return dict(sf_net=args.sf_net, n_comp=args.n_comp,
                composer_norm=bool(getattr(args, 'composer_norm', 0)),
                composer_gate=bool(getattr(args, 'composer_gate', 0)),
                nll_samples=args.nll_samples, lambda_nll=args.lambda_nll,
                lambda_mean=args.lambda_mean, ent_reg=args.ent_reg,
                sigma_min=args.sigma_min, sf_init=args.sf_init,
                freeze_mean=bool(args.freeze_mean),
                task_family=int(getattr(args, 'task_family', 0)))


def _parse_vec(s, K):
    v = [float(x) for x in s.split(',')]
    return v if len(v) > 1 else v * K


def _apply_utility(agent, args):
    """Push the deployment utility spec onto the agent (gmm only) and arm the
    realized-outcome tracker used by rollout_metrics for utility/violation
    metrics."""
    agent.track_outcome = True
    if not agent.is_gmm or args.utility == 'linear':
        agent.set_utility('linear')
        return
    delta = goal = None
    if args.utility in ('box', 'box_mean'):
        assert args.util_delta is not None, "box utility needs --util_delta"
        delta = _parse_vec(args.util_delta, agent.K)
    if args.utility == 'gauss':
        assert args.util_goal is not None, "gauss utility needs --util_goal"
        goal = _parse_vec(args.util_goal, agent.K)
    agent.set_utility(args.utility, delta=delta, goal=goal,
                      beta=args.util_beta)


def demand_scale(args):
    """SUMO --scale: 2.0 for the Manhattan protocol (unchanged), 1.0 (raw
    real-world demand) for the other benchmarks unless --demand_scale given."""
    d = getattr(args, 'demand_scale', None)
    if d is not None:
        return float(d)
    return 2.0 if getattr(args, 'scenario', 'manhattan_28x7') == 'manhattan_28x7' else 1.0


def build_env(reward_fn, args, tripinfo_path=None):
    delta_time = 10
    extra = f"--scale {demand_scale(args)} --no-warnings"
    if tripinfo_path:
        extra += (f" --tripinfo-output {tripinfo_path}"
                  " --tripinfo-output.write-unfinished true")
    scn = getattr(args, 'scenario', 'manhattan_28x7')
    return sumo_rl.parallel_env(
        net_file=f'{DATA_DIR}/{scn}/{scn}.net.xml',
        route_file=f'{DATA_DIR}/{scn}/{scn}.rou.xml',
        use_gui=False, num_seconds=args.max_step * delta_time, delta_time=delta_time,
        yellow_time=2, min_green=5, max_green=50,
        reward_fn=reward_fn, additional_sumo_cmd=extra, **_obs_kwargs(args))


def _obs_kwargs(args):
    if getattr(args, 'obs', 'default') == 'default':
        return {}
    from obs_baselines import OBS_CLASSES
    return {'observation_class': OBS_CLASSES[args.obs]}


def _task_for(args, agent):
    """parse_task, or the zone-layout task matrix aligned with agent.id."""
    if getattr(args, 'zone_layout', None):
        from zones import zone_assignment
        args.hetero = args.hetero or 1
        return zone_assignment(args.zone_layout, list(agent.id))[1]
    return parse_task(args, num_agents=agent.num)


def _apply_task(agent, reward_fn, w, args):
    """Push the target task onto BOTH the operator (agent.w) and the environment
    reward (reward_fn), handling homogeneous [K] and heterogeneous [N,K] w."""
    agent.w = w
    if getattr(args, 'blind', 0):
        agent.w = (np.tile(DEFAULT_W.astype(np.float32), (agent.num, 1))
                   if np.asarray(w).ndim == 2 else DEFAULT_W.astype(np.float32).copy())
    if args.hetero:
        # per-intersection reward weights, keyed by the agent's sorted ids
        w_map = {ts: w[i] for i, ts in enumerate(agent.id)}
        if hasattr(reward_fn, 'set_w_map'):
            reward_fn.set_w_map(w_map)
    else:
        if hasattr(reward_fn, 'set_w'):
            reward_fn.set_w(w)


def rollout_metrics(agent, args, tripinfo_path=None, deterministic=True):
    """One greedy rollout; returns (total phi.w return, full metric dict).

    The metric dict carries the physical TSC panel every method is compared on:
    avg_queue, avg_wait, avg_speed, fairness dispersion (from get_stats) and
    travel_time / throughput (from the SUMO tripinfo file)."""
    observe = agent.reset(train=False)
    meter = None
    lay = getattr(args, 'zone_layout', None) or getattr(args, 'meter_layout', None)
    if lay:
        from zones import zone_assignment, ZoneMeter
        scn = getattr(args, 'scenario', 'manhattan_28x7')
        meter = ZoneMeter(agent.env.unwrapped.env, list(agent.id),
                          zone_assignment(lay, list(agent.id))[0],
                          args.max_step * 10, f'{DATA_DIR}/{scn}/{scn}.rou.xml')
    total = 0.0
    for _ in range(args.max_step):
        action_dic = agent.act(observe, deterministic=deterministic)
        observe, reward, done = agent.step(action_dic)
        total += reward
        if meter is not None:
            meter.step(agent.env.unwrapped.env.sumo)
        if done:
            break
    if meter is not None:
        meter.finish(agent.env.unwrapped.env.sumo)
    if tripinfo_path:
        try:
            agent.env.close()
            tm = parse_tripinfo(tripinfo_path)
            agent.travel_time = tm['travel_time']
            agent.throughput = tm['throughput']
            agent.time_loss = tm['time_loss']
            agent.trip_wait = tm['wait_time']
        except Exception as e:
            print(f"[tripinfo] parse skipped: {e}")
    stats = agent.get_stats()
    if meter is not None:
        stats['zone_metrics'] = meter.summary(tripinfo_path)
    stats['time_loss'] = getattr(agent, 'time_loss', float('nan'))
    stats['trip_wait'] = getattr(agent, 'trip_wait', float('nan'))
    if getattr(agent, 'track_outcome', False) and agent.outcome_disc is not None:
        # realized discounted outcome z_i = sum_t gamma^t phi_{i,t+1}: evaluate
        # the SAME deployment utility the actions were scored under, plus the
        # constraint-violation rate that mean-centric baselines cannot see.
        z = agent.outcome_disc                                        # [N,K]
        stats['z_matrix'] = z.tolist()
        stats['agent_ids'] = list(agent.id)
        w = np.tile(agent.w, (agent.num, 1)) if agent.w.ndim == 1 else agent.w
        stats['w_matrix'] = w.tolist()
        u = agent.util
        stats['realized_utility'] = float(np.mean((z * w).sum(-1)))
        stats['violation_rate'] = float('nan')
        if u['type'] in ('box', 'box_mean'):
            delta = np.asarray(u['delta'], dtype=np.float64)
            ok = (z <= delta).all(axis=1)
            stats['violation_rate'] = float(1.0 - ok.mean())
            stats['realized_utility'] = float(np.mean((z * w).sum(-1) * ok))
        elif u['type'] == 'gauss':
            g = np.asarray(u['goal'], dtype=np.float64)
            stats['realized_utility'] = float(
                np.exp(-((z - g) ** 2).sum(-1) / u['beta']).mean())
            stats['goal_dev'] = float(np.sqrt(((z - g) ** 2).sum(-1)).mean())
    return total, stats


def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device("cuda:" + args.cuda
                          if torch.cuda.is_available() and not args.cpu else 'cpu')

    scn_tag = '' if args.scenario == 'manhattan_28x7' else f'{args.scenario}_'
    fz_tag = 'fz_' if args.freeze_mean else ''
    sfx_tag = f'{args.run_suffix}_' if getattr(args, 'run_suffix', '') else ''
    run_dir = os.path.abspath(
        f"./{'gmm' if args.sf_net == 'gmm' else 'sf'}_{fz_tag}{sfx_tag}{scn_tag}s{args.seed}/{args.phase}")
    os.makedirs(run_dir, exist_ok=True)

    if args.phase == 'usfa':
        # w-free training: reward fn's w is overwritten each episode by the agent.
        reward_fn = make_sf_reward(DEFAULT_W)
        env = build_env(reward_fn, args)
        agent = SFAgent(env, reward_fn, device, mode='usfa', hidden_dim=args.hidden_dim,
                        lr=args.lr, gamma=args.gamma, horizon_len=args.max_step,
                        training_episodes=max(1, args.episodes), buffer_size=args.buffer_size,
                        n_heads=args.n_heads, use_agent_emb=bool(args.use_agent_emb),
                        target_update_freq=args.target_update_freq, **_sf_kwargs(args))
        agent.set_base_seed(args.seed)
        _train_loop(agent, args, run_dir, learn=True)
        agent.save_sf(os.path.join(run_dir, 'sf_psi.pth'))
        print(f"[usfa] saved psi -> {os.path.join(run_dir, 'sf_psi.pth')}")

    elif args.phase == 'transfer':
        assert args.sf_ckpt, "transfer requires --sf_ckpt (trained psi)"
        # need num_agents for hetero w -> build a throwaway to read agent ids,
        # but simpler: build env+agent first with DEFAULT_W, then apply the task.
        reward_fn = make_sf_reward(DEFAULT_W)
        trip = (os.path.join(run_dir, f"tripinfo{('_' + args.out_tag) if args.out_tag else ''}.xml")
                if args.tripinfo else None)
        env = build_env(reward_fn, args, tripinfo_path=trip)
        agent = SFAgent(env, reward_fn, device, mode='usfa', hidden_dim=args.hidden_dim,
                        gamma=args.gamma, horizon_len=args.max_step,
                        use_agent_emb=bool(args.use_agent_emb), sf_ckpt=args.sf_ckpt,
                        **_sf_kwargs(args))
        agent.set_base_seed(args.seed)
        w = _task_for(args, agent)
        _apply_task(agent, reward_fn, w, args)
        _apply_utility(agent, args)
        results = [rollout_metrics(agent, args, tripinfo_path=trip)
                   for _ in range(args.eval_episodes)]
        _report(run_dir, 'transfer', w, results, args)

    else:  # composer
        assert args.sf_ckpt, "composer requires --sf_ckpt (trained psi)"
        reward_fn = make_sf_reward(DEFAULT_W)
        trip = (os.path.join(run_dir, f"tripinfo{('_' + args.out_tag) if args.out_tag else ''}.xml")
                if args.tripinfo else None)
        env = build_env(reward_fn, args, tripinfo_path=trip)
        agent = SFAgent(env, reward_fn, device, mode='composer', hidden_dim=args.hidden_dim,
                        lr=args.lr, gamma=args.gamma, horizon_len=args.max_step,
                        training_episodes=max(1, args.episodes), buffer_size=args.buffer_size,
                        n_heads=args.n_heads, use_agent_emb=bool(args.use_agent_emb),
                        target_update_freq=args.target_update_freq, sf_ckpt=args.sf_ckpt,
                        **_sf_kwargs(args))
        agent.set_base_seed(args.seed)
        w = _task_for(args, agent)
        _apply_task(agent, reward_fn, w, args)
        _apply_utility(agent, args)
        # --episodes 0: skip training and load the existing composer checkpoint.
        # This is used for multi-rollout variance estimation (EXP2) where the
        # trained composer.pth already exists and we only want repeated evals.
        composer_ckpt = os.path.join(run_dir, 'composer.pth')
        if args.episodes == 0 and args.composer_ckpt:
            composer_ckpt = args.composer_ckpt
        if args.episodes == 0:
            if os.path.exists(composer_ckpt):
                agent.load(composer_ckpt)
                print(f"[composer] --episodes 0: loaded existing checkpoint "
                      f"{composer_ckpt}, skipping training.")
            else:
                print(f"[composer] WARNING: --episodes 0 but no checkpoint found "
                      f"at {composer_ckpt}. Evaluating warm-started (untrained) composer.")
        else:
            _train_loop(agent, args, run_dir, learn=True, fixed_w=True)
            agent.save(composer_ckpt)
        results = [rollout_metrics(agent, args, tripinfo_path=trip)
                   for _ in range(args.eval_episodes)]
        _report(run_dir, 'composer', w, results, args)


def _train_loop(agent, args, run_dir, learn, fixed_w=False):
    log_path = os.path.join(run_dir, 'train.txt')
    global_env_steps = 0
    for epoch in range(1, args.episodes + 1):
        frac = min(1.0, (epoch - 1) / max(1, args.eps_decay_episodes))
        eps = args.eps_start + frac * (args.eps_end - args.eps_start)
        observe = agent.reset(train=True, episode=epoch)
        total, ep_losses = 0.0, []
        for t in range(args.max_step):
            step_eps = 1.0 if global_env_steps < args.warmup else eps
            action_dic = agent.act(observe, exploration_rate=step_eps)
            observe, reward, done = agent.step(action_dic)
            total += reward
            global_env_steps += 1
            if learn and global_env_steps >= args.warmup:
                l = agent.learn(batch_size=args.batch_size, updates=args.updates_per_step)
                if l:
                    ep_losses.append(l)
            if done:
                break
        agent.scheduler.step()
        loss = float(np.mean(ep_losses)) if ep_losses else 0.0
        log = f"Epoch {epoch} | Reward {total:.2f} | Loss {loss:.4f} | Eps {eps:.3f}"
        print(log)
        with open(log_path, 'a') as f:
            f.write(log + "\n")
        if learn and args.ckpt_every and epoch % args.ckpt_every == 0:
            agent.save_sf(os.path.join(run_dir, f'psi_ep{epoch}.pth'))


def _report(run_dir, phase, w, results, args):
    """Write the full physical-metric panel (not just reward) for one method.

    ``results`` is a list of (total_return, stats_dict) from rollout_metrics.
    Every method (retraining baselines, zero-shot transfer, composer) is
    reported on the SAME panel so the comparison is on real traffic metrics,
    not only the phi.w objective: reward, travel_time, throughput, avg_queue,
    avg_wait, avg_speed, fairness dispersion."""
    vals = [r for r, _ in results]
    stats = [s for _, s in results]

    def _mean(key):
        xs = [s.get(key, float('nan')) for s in stats]
        xs = [x for x in xs if x == x]                 # drop NaNs
        return float(np.mean(xs)) if xs else float('nan')

    panel = {k: _mean(k) for k in
             ('travel_time', 'time_loss', 'trip_wait', 'throughput',
              'avg_queue', 'avg_wait', 'avg_speed', 'avg_fairness',
              'realized_utility', 'violation_rate', 'goal_dev')}
    out = {
        'phase': phase,
        'hetero': bool(args.hetero),
        'w': (w.tolist() if hasattr(w, 'tolist') else [float(x) for x in w]),
        'mean_return': float(np.mean(vals)),
        'std_return': float(np.std(vals)),
        'n': len(vals),
        'returns': [float(v) for v in vals],
        'metrics': panel,
        # per-agent realized outcome/task matrices (episodic, last episode if n>1)
        'z_matrix': stats[-1].get('z_matrix'),
        'w_matrix': stats[-1].get('w_matrix'),
        'agent_ids': stats[-1].get('agent_ids'),
        'zone_layout': getattr(args, 'zone_layout', None),
        'meter_layout': getattr(args, 'meter_layout', None),
        'blind': int(getattr(args, 'blind', 0)),
        'zone_metrics': ({k: float(np.mean([s['zone_metrics'][k] for s in stats]))
                          for k in stats[0]['zone_metrics']}
                         if 'zone_metrics' in stats[0] else None),
    }
    tag = 'hetero' if args.hetero else 'homo'
    otag = f"_{args.out_tag}" if getattr(args, 'out_tag', '') else ''
    with open(os.path.join(run_dir, f'{phase}_{tag}{otag}_result.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print(f"[{phase}|{tag}] mean_return={out['mean_return']:.2f} +- {out['std_return']:.2f} "
          f"(n={out['n']}) | travel_time={panel['travel_time']:.2f} "
          f"throughput={panel['throughput']:.0f} queue={panel['avg_queue']:.2f} "
          f"wait={panel['avg_wait']:.2f} speed={panel['avg_speed']:.2f} "
          f"fairness={panel['avg_fairness']:.3f}")


if __name__ == '__main__':
    main()
