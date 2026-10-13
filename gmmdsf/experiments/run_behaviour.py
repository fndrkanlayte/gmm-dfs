"""Fig. 4 behaviour probes: re-run C3 (and A3) for mp / tfc / tfc_blind with separate
output tags so table rows/files are untouched; collects zone art_green / switch_rate."""
import os as _os, sys as _sys
_PKG = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _PKG)
_os.chdir(_PKG)  # all inputs/outputs are relative to gmmdsf/
import json, glob, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
import run_table1_v2 as R
OUT = 'behaviour_results.jsonl'
jobs = []
for L in ['C3', 'A3']:
    for m in ['mp', 'tfc', 'tfc_blind']:
        for s in range(3):
            cmd, (kind, path) = R.job_cmd('B', m, L, s)
            if kind == 'rule':
                i = cmd.index('--output'); cmd[i + 1] = cmd[i + 1] + '_beh'; path = cmd[i + 1]
            else:
                i = cmd.index('--out_tag'); old = cmd[i + 1]; cmd[i + 1] = old + '_beh'
                path = path.replace(old, old + '_beh')
            jobs.append((L, m, s, cmd, kind, path))
def run(j):
    L, m, s, cmd, kind, path = j
    r = subprocess.run(cmd, cwd=R.ROOT, capture_output=True, text=True)
    try:
        if kind == 'rule':
            d = json.load(open(glob.glob(os.path.join(R.ROOT, path, '*_result.json'))[0])); zm = d['metrics']
        else:
            d = json.load(open(os.path.join(R.ROOT, path))); zm = d['zone_metrics']
        rec = dict(layout=L, method=m, seed=s, **{k: v for k, v in zm.items() if isinstance(v, (int, float))})
        with open(OUT, 'a') as f: f.write(json.dumps(rec) + '\n')
        print('OK', L, m, s, flush=True)
    except Exception as e:
        print('FAIL', L, m, s, e, r.stderr[-500:], flush=True)
with ThreadPoolExecutor(4) as ex:
    list(ex.map(run, jobs))
print('BEHAVIOUR DONE')
