"""Manhattan evaluation battery (Tables 1-2, ablations, Fig. 3/5/6).

Part A (homogeneous): tasks T0/T1/T2, zone meter layout A only for the
all-vehicle delay. Part B (heterogeneous): zone layouts A/B/A3/B3/C3 + S-sweep (task = layout W).
Every job is an independent process with unique output paths; rows are
appended to gmmdsf/table1_v2.csv (resume-safe: done jobs are skipped; methods
whose checkpoint has not been trained yet are skipped with a notice).

Usage: python experiments/run_table1_v2.py [--parts A,B] [--methods m1,m2]
                                           [--seeds 0,1,2] [--workers 6]
"""
import os as _os, sys as _sys
_PKG = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _PKG)
_os.chdir(_PKG)  # all inputs/outputs are relative to gmmdsf/
import argparse
import csv
import glob
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = _PKG
PY = sys.executable
OUT_CSV = os.path.join(ROOT, 'table1_v2.csv')
ENV = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', LIBSUMO_AS_TRACI='1')

TASKS_A = {'T0': '-1,-0.3,-1.5,3,0', 'T1': '-1,0,0,0,0', 'T2': '-0.3,-0.1,-0.5,6,0'}
LAYOUTS_B = ['A', 'B', 'A3', 'B3', 'C3', 'S0', 'S25', 'S50', 'S75']

# Checkpoint paths are the run directories written by scripts/*.sh (relative
# to gmmdsf/). method -> spec. kind: rule | idqn | plight | ours.
# 'only' restricts a method to the listed tasks / layouts.
BACKBONE = 'gmm_tfc_s0/usfa/sf_psi.pth'          # main GMM-DSF backbone (stage 5)
HET = ['A', 'B', 'A3', 'B3', 'C3', 'S0', 'S25', 'S50', 'S75']


def _comp(run, gate=0, cnorm=0, only=None):
    """Composer row: frozen main backbone + composer trained by train_composers.sh."""
    return dict(kind='ours', net='gmm', ckpt=BACKBONE, phase='ma-usfa',
                composer=f'gmm_{run}_s0/composer/composer.pth', gate=gate,
                cnorm=cnorm, only=only)


METHODS = {
    # ---- rule-based ----
    'ft': dict(kind='rule', script='baselines/eval_fixedtime.py'),
    'mp': dict(kind='rule', script='baselines/eval_maxpressure.py'),
    # ---- learned baselines trained on T0 / with their own reward ----
    'idqn_mlp_T0': dict(kind='idqn', net='mlp', ckpt='composite_s0/DQN_mlp/pretrain_latest.pth'),
    'idqn_gat_T0': dict(kind='idqn', net='gat', ckpt='composite_s0/DQN_gat/pretrain_latest.pth'),
    'presslight': dict(kind='idqn', net='mlp', obs='presslight',
                       ckpt='abspressure_presslight_s0/DQN_mlp/pretrain_latest.pth'),
    'mplight': dict(kind='idqn', net='mlp', obs='mplight', emb=0,
                    ckpt='abspressure_mplight_s0/DQN_mlp/pretrain_latest.pth'),
    'plight_T0': dict(kind='plight', model='baseline_runs/plight_T0'),
    # ---- task-specific retraining / adaptation ----
    'idqn_mlp_T1spec': dict(kind='idqn', net='mlp', ckpt='composite_t1_s0/DQN_mlp/pretrain_latest.pth', only=['T1']),
    'idqn_gat_T2spec': dict(kind='idqn', net='gat', ckpt='composite_t2_s0/DQN_gat/pretrain_latest.pth', only=['T2']),
    'plight_T1spec': dict(kind='plight', model='baseline_runs/plight_T1', only=['T1']),
    'plight_T2spec': dict(kind='plight', model='baseline_runs/plight_T2', only=['T2']),
    'prlight_T1': dict(kind='plight', model='baseline_runs/prlight_T1', only=['T1']),
    'prlight_T2': dict(kind='plight', model='baseline_runs/prlight_T2', only=['T2']),
    **{f'idqn_zone{L}': dict(kind='idqn', net='mlp', ckpt=f'composite_zone{L}_s0/DQN_mlp/pretrain_latest.pth', only=[L])
       for L in ('A3', 'B3', 'C3')},
    **{f'prlight_{L}': dict(kind='plight', model=f'baseline_runs/prlight_{L}', only=[L])
       for L in ('A3', 'B3', 'C3')},
    # ---- point-estimate SF backbone (MA-USFA), independent zero-shot rule ----
    'sf': dict(kind='ours', net='sf', ckpt='sf_s0/usfa/sf_psi.pth'),
    # ---- GMM-DSF zero-shot: main model and training-stage ablations ----
    'tfc': dict(kind='ours', net='gmm', ckpt=BACKBONE),
    'tfc_blind': dict(kind='ours', net='gmm', ckpt=BACKBONE, blind=1, only=['T1', 'T2'] + HET),
    'tfh': dict(kind='ours', net='gmm', ckpt='gmm_tfh_s0/usfa/sf_psi.pth'),
    'tfh_blind': dict(kind='ours', net='gmm', ckpt='gmm_tfh_s0/usfa/sf_psi.pth', blind=1, only=HET),
    'tfz': dict(kind='ours', net='gmm', ckpt='gmm_tfz_s0/usfa/sf_psi.pth'),
    'tfz_blind': dict(kind='ours', net='gmm', ckpt='gmm_tfz_s0/usfa/sf_psi.pth', blind=1, only=HET),
    'tfd': dict(kind='ours', net='gmm', ckpt='gmm_tfd_s0/usfa/sf_psi.pth', obs='downstream'),
    'tfd_blind': dict(kind='ours', net='gmm', ckpt='gmm_tfd_s0/usfa/sf_psi.pth', obs='downstream',
                      blind=1, only=HET),
    # ---- composer (frozen backbone): main = speed-gated, 200 episodes ----
    **{f'tfc_gate2{t}': _comp(f'tfcg2{t}', gate=1, only=[t]) for t in ('T0', 'T2', 'A3', 'C3')},
    # ablations: 400 episodes / ungated / scale-normalised / old exploration schedule
    **{f'tfc_gate4{t}': _comp(f'tfcg4{t}', gate=1, only=[t]) for t in ('T0', 'T2', 'A3', 'C3')},
    **{f'tfc_u2{t}': _comp(f'tfcu2{t}', only=[t]) for t in ('T1', 'A3', 'C3')},
    **{f'tfc_n2{t}': _comp(f'tfcn2{t}', gate=1, cnorm=1, only=[t]) for t in ('T0', 'T2', 'A3', 'C3')},
    **{f'tfc_comp{t}': _comp(f'tfccomp{t}', only=[t]) for t in ('T0', 'T1', 'T2', 'A3')},
}

ZONE_KEYS = ['net_delay', 'n_arrived', 'n_running', 'n_pending',
             'corridor_speed_art', 'corridor_wait_mean', 'corridor_wait_p95',
             'gateway_spill', 'gateway_queue_lane', 'gateway_wait_p95',
             'equity_wait_p95', 'equity_wait_mean', 'equity_queue_lane',
             'base_wait_mean', 'base_speed']
FIELDS = (['part', 'method', 'task', 'seed', 'return', 'travel_time', 'throughput',
           'avg_queue', 'avg_wait', 'avg_speed', 'realized_utility'] + ZONE_KEYS
          + ['zone_json', 'zone_util_json'])


def job_cmd(part, method, task, seed):
    sp = METHODS[method]
    tag = f'{method}_{part}{task}_s{seed}'
    if part == 'A':
        task_args = [f'--task={TASKS_A[task]}']
        lay_args = ['--meter_layout', 'A']
    else:
        task_args = []
        lay_args = ['--zone_layout', task]
    if sp['kind'] == 'rule':
        out = os.path.join('eval_runs', 'table1_v2', tag)
        return ([PY, '-u', sp['script'], '--eval_episodes', '1', '--seed', str(seed),
                 '--output', out] + task_args + lay_args), ('rule', out)
    if sp['kind'] == 'idqn':
        rt = f'ev_{tag}'
        extra, pre = [], ''
        if sp.get('obs'):
            extra += ['--obs', sp['obs']]
            pre = f'_{sp["obs"]}'
        if sp.get('emb') == 0:
            extra += ['--use_agent_emb', '0']
        return ([PY, '-u', 'baselines/train.py', '--net', sp['net'], '--eval_only', '1',
                 '--eval_episode', '1', '--model_path', sp['ckpt'], '--run_tag', rt,
                 '--seed', str(seed)] + extra + task_args + lay_args), \
            ('idqn', os.path.join(f'composite{pre}_{rt}_s{seed}', f'DQN_{sp["net"]}',
                                  f'eval_only_s{seed}.json'))
    if sp['kind'] == 'plight':
        out = os.path.join('eval_runs', 'table1_v2', tag)
        return ([PY, '-u', 'baselines/prlight.py', '--mode', 'eval', '--model', sp['model'],
                 '--seed', str(seed), '--out', out] + task_args + lay_args), \
            ('idqn', os.path.join(out, f'eval_s{seed}.json'))
    phase = sp.get('phase', 'independent')
    cmd = [PY, '-u', 'train_sf.py', '--phase', phase, '--sf_net', sp['net'],
           '--sf_ckpt', sp['ckpt'], '--eval_episodes', '1', '--seed', str(seed),
           '--run_suffix', 'ev', '--out_tag', tag] + task_args + lay_args
    if sp.get('blind'):
        cmd += ['--blind', '1']
    if sp.get('obs'):
        cmd += ['--obs', sp['obs']]
    if sp.get('cnorm'):
        cmd += ['--composer_norm', '1']
    if sp.get('gate'):
        cmd += ['--composer_gate', '1']
    if sp.get('composer'):
        cmd += ['--episodes', '0', '--composer_ckpt', sp['composer']]
    pdir = {'independent': 'transfer', 'ma-usfa': 'composer'}[phase]
    htag = 'hetero' if part == 'B' else 'homo'
    pre = 'gmm' if sp['net'] == 'gmm' else 'sf'
    res = os.path.join(f'{pre}_ev_s{seed}', pdir, f'{pdir}_{htag}_{tag}_result.json')
    return cmd, ('ours', res)


def zone_utils(r, layout, ours):
    """Zone utility U_z = mean_{i in z} z_i . w_i (scoring W)."""
    from zones import zone_assignment
    import numpy as np
    z = r.get('z_matrix')
    ids = r.get('agent_ids') or r.get('ids')
    if z is None or ids is None:
        return {}
    W = np.array(r['w'] if ours else r['w_matrix'], dtype=float)
    if W.ndim == 1:
        W = np.tile(W, (len(ids), 1))
    z = np.array(z, dtype=float)
    zone_of, _ = zone_assignment(layout, list(ids))
    zn = np.array([zone_of[t] for t in ids])
    return {k: float((z[zn == k] * W[zn == k]).sum(-1).mean()) for k in sorted(set(zn))}


def read_row(part, method, task, seed, kind, path):
    full = os.path.join(ROOT, path)
    if kind == 'rule':
        cands = glob.glob(os.path.join(full, '*_result.json'))
        if not cands:
            return None
        r = json.load(open(cands[0]))
        m = r['metrics']
        zm = {k: v for k, v in m.items()}
    else:
        if not os.path.exists(full):
            return None
        r = json.load(open(full))
        m = r['metrics']
        zm = r.get('zone_metrics') or {}
    row = {'part': part, 'method': method, 'task': task, 'seed': seed,
           'return': r['mean_return'], 'travel_time': m.get('travel_time'),
           'throughput': m.get('throughput'), 'avg_queue': m.get('avg_queue'),
           'avg_wait': m.get('avg_wait'), 'avg_speed': m.get('avg_speed'),
           'realized_utility': m.get('realized_utility')}
    for k in ZONE_KEYS:
        row[k] = zm.get(k)
    row['zone_util_json'] = json.dumps(zone_utils(r, task, kind == 'ours')) if part == 'B' else ''
    row['zone_json'] = json.dumps({k: v for k, v in zm.items()
                                   if k.split('_')[0] in ('corridor', 'gateway', 'equity', 'base')})
    return row


def trained(method):
    sp = METHODS[method]
    if sp['kind'] == 'rule':
        return True
    if sp['kind'] == 'plight':
        return os.path.exists(os.path.join(ROOT, sp['model'], 'edqn.pt'))
    paths = [sp['ckpt']] + ([sp['composer']] if sp.get('composer') else [])
    return all(os.path.exists(os.path.join(ROOT, p)) for p in paths)


def done_set():
    if not os.path.exists(OUT_CSV):
        return set()
    return {(r['part'], r['method'], r['task'], r['seed'])
            for r in csv.DictReader(open(OUT_CSV))}


def run_job(job):
    part, method, task, seed = job
    cmd, (kind, path) = job_cmd(*job)
    p = subprocess.run(cmd, cwd=ROOT, env=ENV, capture_output=True, text=True)
    row = read_row(part, method, task, seed, kind, path) if p.returncode == 0 else None
    err = None if row else (p.stdout[-400:] + '\n' + p.stderr[-800:])
    return job, row, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parts', default='A,B')
    ap.add_argument('--methods', default=','.join(METHODS))
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(',')]
    done = done_set()
    jobs = []
    for part in a.parts.split(','):
        tasks = list(TASKS_A) if part == 'A' else LAYOUTS_B
        for method in a.methods.split(','):
            only = METHODS[method].get('only')
            for task in tasks:
                if only and task not in only:
                    continue
                for s in seeds:
                    if (part, method, task, str(s)) not in done:
                        jobs.append((part, method, task, s))
    ready = {m: trained(m) for m in {j[1] for j in jobs}}
    for m in sorted(k for k, v in ready.items() if not v):
        print(f'skip {m}: checkpoint not found (train it first, see scripts/)', flush=True)
    jobs = [j for j in jobs if ready[j[1]]]
    print(f'{len(jobs)} jobs', flush=True)
    new = not os.path.exists(OUT_CSV)
    with open(OUT_CSV, 'a', newline='') as f, ThreadPoolExecutor(a.workers) as ex:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for fut in as_completed([ex.submit(run_job, j) for j in jobs]):
            job, row, err = fut.result()
            if row:
                w.writerow(row)
                f.flush()
                print(f'OK {job} ret={row["return"]:.1f} delay={row["net_delay"]}', flush=True)
            else:
                print(f'FAIL {job}\n{err}', flush=True)


if __name__ == '__main__':
    main()
