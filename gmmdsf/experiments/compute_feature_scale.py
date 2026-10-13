"""Feature scale constants s_k for the standardized task space.

s_k = mean over (intersection, episode) of the discounted cumulative feature
z_ik = sum_t gamma^t phi_ik(t+1) under a UNIFORM-RANDOM policy (the same
behaviour that fills the replay during warmup), i.e. a neutral reference that
is none of the compared controllers. SUMO seeds 20,000,000+k are disjoint from
training (seed+episode) and evaluation (seed+10,000,000) streams.

Output: feature_scale.json {"s": [...], "per_episode": [...], ...}
"""
import os as _os, sys as _sys
_PKG = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _PKG)
_os.chdir(_PKG)  # all inputs/outputs are relative to gmmdsf/
import json
import types

import numpy as np

import train_sf
from composite_reward import make_sf_reward, DEFAULT_W, FEATURE_NAMES

GAMMA, H, EPISODES = 0.99, 60, 3


def main():
    args = types.SimpleNamespace(max_step=H, scenario='manhattan_28x7')
    rng = np.random.RandomState(0)
    per_ep = []
    for k in range(EPISODES):
        rf = make_sf_reward(DEFAULT_W)
        env = train_sf.build_env(rf, args, None)
        obs, _ = env.reset(seed=20_000_000 + k)
        ids = sorted(obs)
        n_act = {ts: int(env.action_space(ts).n) for ts in ids}
        z = np.zeros((len(ids), len(FEATURE_NAMES)))
        for t in range(H):
            env.step({ts: int(rng.randint(n_act[ts])) for ts in ids})
            fc = rf.feature_cache
            z += GAMMA ** t * np.array([fc[ts] for ts in ids])
        env.close()
        per_ep.append(z.mean(0))
        print(f'episode {k}: ' + ' '.join(f'{n}={v:.4f}' for n, v in zip(FEATURE_NAMES, z.mean(0))),
              flush=True)
    s = np.mean(per_ep, 0)
    out = {'features': list(FEATURE_NAMES), 's': s.tolist(),
           'per_episode': [p.tolist() for p in per_ep],
           'reference': 'uniform-random policy, 3 episodes, gamma=0.99, H=60, '
                        'SUMO seeds 20000000-20000002'}
    json.dump(out, open('feature_scale.json', 'w'), indent=1)
    print('s =', dict(zip(FEATURE_NAMES, np.round(s, 5))))
    print('DEFAULT_W in standardized units (w_k * s_k):',
          dict(zip(FEATURE_NAMES, np.round(DEFAULT_W * s, 4))))


if __name__ == '__main__':
    main()
