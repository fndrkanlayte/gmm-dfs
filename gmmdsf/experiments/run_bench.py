"""Other-benchmark battery (Hangzhou 4x4, Atlanta 1x5) -> bench_results.csv.

Tasks: 'T0' (homogeneous standard, all-vehicle metrics) and the zone layouts
(HZA3/HZB3, ATL3; v3 objectives). Only methods whose training has finished
are evaluated (resume-safe), so this can be called repeatedly.
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

import run_table1_v2 as R

ROOT = _PKG
OUT = os.path.join(ROOT, 'bench_results.csv')
PY = sys.executable
T0 = '-1,-0.3,-1.5,3,0'
SCN = {
    'hz': dict(scenario='hangzhou_4x4_gudang_18041610_1h', city='hangzhou', steps=360, layouts=['HZA3', 'HZB3']),
    'atl': dict(scenario='atlanta_1x5', city='atlanta', steps=120, layouts=['ATL3']),
}


def methods(p):
    c, L = SCN[p]['city'], SCN[p]['layouts']
    m = {
        'ft': dict(kind='rule', script='baselines/eval_fixedtime.py'),
        'mp': dict(kind='rule', script='baselines/eval_maxpressure.py'),
        'presslight': dict(kind='idqn', net='mlp', obs='presslight',
                           ckpt=f'abspressure_{c}_presslight_s0/DQN_mlp/pretrain_latest.pth'),
        'mplight': dict(kind='idqn', net='mlp', obs='mplight', emb=0,
                        ckpt=f'abspressure_{c}_mplight_s0/DQN_mlp/pretrain_latest.pth'),
        'idqn_mlp_T0': dict(kind='idqn', net='mlp',
                            ckpt=f'composite_{c}_s0/DQN_mlp/pretrain_latest.pth'),
        'idqn_gat_T0': dict(kind='idqn', net='gat',
                            ckpt=f'composite_{c}_s0/DQN_gat/pretrain_latest.pth'),
        'plight_T0': dict(kind='plight', model=f'baseline_runs/{p}_plight_T0'),
    }
    scn = SCN[p]['scenario']
    m['ours'] = dict(kind='ours', ckpt=f'gmm_btfc_{scn}_s0/usfa/sf_psi.pth')
    m['ours_blind'] = dict(kind='ours', ckpt=f'gmm_btfc_{scn}_s0/usfa/sf_psi.pth', blind=1, only=L)
    m['mausfa'] = dict(kind='ours', snet='sf', ckpt=f'sf_bsf_{scn}_s0/usfa/sf_psi.pth')
    for lay in L:
        # layout-specific baselines trained on the benchmark zone objective
        # (ZONE_W_BENCH, waiting-only equity zone): run tag / dir suffix 'eq'.
        m[f'idqn_zone{lay}'] = dict(kind='idqn', net='mlp',
                                    ckpt=f'composite_{c}_zone{lay}eq_s0/DQN_mlp/pretrain_latest.pth', only=[lay])
        m[f'prlight_{lay}'] = dict(kind='plight', model=f'baseline_runs/{p}_prlight_{lay}eq', only=[lay])
        m[f'ours_gate2{lay}'] = dict(kind='ours', ckpt=f'gmm_btfc_{scn}_s0/usfa/sf_psi.pth', phase='ma-usfa',
                                     composer=f'gmm_bgate2{lay}_{scn}_s0/composer/composer.pth', gate=1, only=[lay])
    return m


def ready(sp):
    if sp['kind'] == 'rule':
        return True
    if sp['kind'] == 'ours':
        return os.path.exists(os.path.join(ROOT, sp.get('composer') or sp['ckpt']))
    if sp['kind'] == 'plight':
        return os.path.exists(os.path.join(ROOT, sp['model'], 'edqn.pt'))
    return os.path.exists(os.path.join(ROOT, sp['ckpt']))


def job(p, method, task, seed):
    sc = SCN[p]
    sp = methods(p)[method]
    tag = f'{p}_{method}_{task}_s{seed}'
    common = ['--scenario', sc['scenario'], '--max_step', str(sc['steps']), '--seed', str(seed)]
    ta = ([f'--task={T0}', '--meter_layout', sc['layouts'][0]] if task == 'T0' else ['--zone_layout', task])
    if sp['kind'] == 'rule':
        out = os.path.join('eval_runs', 'bench', tag)
        return [PY, '-u', sp['script'], '--eval_episodes', '1', '--output', out] + common + ta, ('rule', out)
    if sp['kind'] == 'plight':
        out = os.path.join('eval_runs', 'bench', tag)
        return [PY, '-u', 'baselines/prlight.py', '--mode', 'eval', '--model', sp['model'], '--out', out] + common + ta, \
            ('idqn', os.path.join(out, f'eval_s{seed}.json'))
    if sp['kind'] == 'ours':
        phase = sp.get('phase', 'independent')
        snet = sp.get('snet', 'gmm')
        cmd = [PY, '-u', 'train_sf.py', '--phase', phase, '--sf_net', snet, '--sf_ckpt', sp['ckpt'],
               '--eval_episodes', '1', '--run_suffix', 'bev', '--out_tag', tag] + common + ta
        if sp.get('blind'):
            cmd += ['--blind', '1']
        if sp.get('cnorm'):
            cmd += ['--composer_norm', '1']
        if sp.get('gate'):
            cmd += ['--composer_gate', '1']
        if sp.get('composer'):
            cmd += ['--episodes', '0', '--composer_ckpt', sp['composer']]
        pdir = {'independent': 'transfer', 'ma-usfa': 'composer'}[phase]
        htag = 'homo' if task == 'T0' else 'hetero'
        return cmd, ('ours', os.path.join(f'{snet}_bev_{sc["scenario"]}_s{seed}', pdir,
                                          f'{pdir}_{htag}_{tag}_result.json'))
    rt = f'ev_{tag}'
    extra, pre = [], f'_{sc["city"]}'
    if sp.get('obs'):
        extra += ['--obs', sp['obs']]
        pre += f'_{sp["obs"]}'
    if sp.get('emb') == 0:
        extra += ['--use_agent_emb', '0']
    cmd = [PY, '-u', 'baselines/train.py', '--net', sp['net'], '--eval_only', '1', '--eval_episode', '1',
           '--model_path', sp['ckpt'], '--run_tag', rt] + extra + common + ta
    # train.py run_dir: composite + _<city> + _<obs> + _<run_tag>
    pre = f'_{sc["city"]}' if not sp.get('obs') else f'_{sc["city"]}_{sp["obs"]}'
    return cmd, ('idqn', os.path.join(f'composite{pre}_{rt}_s{seed}', f'DQN_{sp["net"]}', f'eval_only_s{seed}.json'))


FIELDS = ['bench'] + R.FIELDS


def run(j):
    p, method, task, seed = j
    cmd, (kind, path) = job(*j)
    pr = subprocess.run(cmd, cwd=ROOT, env=R.ENV, capture_output=True, text=True)
    row = None
    if pr.returncode == 0:
        row = R.read_row('B' if task != 'T0' else 'A', method, task, seed, kind, path)
    return j, row, None if row else (pr.stdout[-300:] + pr.stderr[-600:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--benches', default='hz,atl')
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    done = set()
    if os.path.exists(OUT):
        done = {(r['bench'], r['method'], r['task'], r['seed']) for r in csv.DictReader(open(OUT))}
    jobs = []
    for p in a.benches.split(','):
        for m, sp in methods(p).items():
            if not ready(sp):
                continue
            for t in ['T0'] + SCN[p]['layouts']:
                if sp.get('only') and t not in sp['only']:
                    continue
                for s in a.seeds.split(','):
                    if (p, m, t, s) not in done:
                        jobs.append((p, m, t, int(s)))
    print(f'{len(jobs)} jobs', flush=True)
    new = not os.path.exists(OUT)
    with open(OUT, 'a', newline='') as f, ThreadPoolExecutor(a.workers) as ex:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for fut in as_completed([ex.submit(run, j) for j in jobs]):
            j, row, err = fut.result()
            if row:
                row['bench'] = j[0]
                w.writerow(row)
                f.flush()
                print('OK', j, f'ret={row["return"]:.1f}', flush=True)
            else:
                print('FAIL', j, err, flush=True)


if __name__ == '__main__':
    main()
