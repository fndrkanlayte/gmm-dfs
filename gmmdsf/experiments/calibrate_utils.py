"""Calibrate the non-linear task parameters (T4 threshold, T5 target).

Runs a few independent-rule rollouts with the point-estimate SF checkpoint on
the default task and collects every agent's realized discounted outcome
z_i = sum_t gamma^t phi_{i,t+1}. Then prints the frozen parameters:

  * delta[k] = 70th percentile of z[:,k]        (box task T4 thresholds)
  * goal[k]  = median of z[:,k], except speed = 1.2 x median  (gauss task T5)
  * beta     = K (=5)

The constants used in the paper are fixed in scripts/eval_nonlinear.sh.

Usage (from gmmdsf/):
    python experiments/calibrate_utils.py --sf_ckpt sf_s0/usfa/sf_psi.pth --episodes 5
"""
import os as _os, sys as _sys
_PKG = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _PKG)
_os.chdir(_PKG)  # all inputs/outputs are relative to gmmdsf/
import argparse
import os
import sys

os.environ.setdefault('LIBSUMO_AS_TRACI', '1')
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sf_ckpt', required=True)
    p.add_argument('--episodes', type=int, default=5)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--box_q', type=float, default=70.0,
                   help='percentile of realized z used as box threshold')
    args = p.parse_args()

    import train_sf
    from composite_reward import make_sf_reward, DEFAULT_W
    from agent import SFAgent
    from utils import set_seed

    # get_args() parses sys.argv -- give it a clean one so our own flags
    # (--sf_ckpt/--episodes/...) don't collide.
    argv_bak = sys.argv
    sys.argv = [argv_bak[0]]
    ns = train_sf.get_args()
    sys.argv = argv_bak
    ns.task_seed, ns.hetero, ns.tripinfo = None, 0, 0
    ns.sf_net = 'sf'
    args_out = ns

    set_seed(args.seed)
    reward_fn = make_sf_reward(DEFAULT_W)
    env = train_sf.build_env(reward_fn, args_out)
    agent = SFAgent(env, reward_fn, 'cuda' if __import__('torch').cuda.is_available() else 'cpu',
                    mode='usfa', gamma=args_out.gamma,
                    horizon_len=args_out.max_step,
                    use_agent_emb=bool(args_out.use_agent_emb),
                    sf_ckpt=args.sf_ckpt)
    agent.set_base_seed(args.seed)
    agent.track_outcome = True

    zs = []
    for ep in range(args.episodes):
        observe = agent.reset(train=False)
        # mode='usfa' reset() samples a training task; calibration must run
        # under the frozen DEFAULT_W instead.
        agent.w = DEFAULT_W
        agent.reward_fn.set_w(DEFAULT_W)
        while True:
            action_dic = agent.act(observe)
            observe, reward, done = agent.step(action_dic)
            if done:
                break
        zs.append(agent.outcome_disc.copy())
        agent.env.close()
        print(f"ep{ep}: collected z [{agent.outcome_disc.shape}]")
    Z = np.concatenate(zs, 0)                                  # [eps*N,K]
    K = Z.shape[1]
    delta = np.percentile(Z, args.box_q, axis=0)
    med = np.median(Z, axis=0)
    goal = med.copy()
    goal[3] = 1.2 * med[3]                                     # speed dim target
    print("\n=== CALIBRATED PARAMETERS ===")
    print("n =", Z.shape[0], "agent-outcomes")
    names = ['queue', 'wait', 'pressure', 'speed', 'fairness']
    for i, n in enumerate(names):
        print(f"  {n:9s} median {med[i]:8.3f}  p70 {delta[i]:8.3f}  "
              f"goal {goal[i]:8.3f}")
    print("util_delta:", ','.join(f'{x:.4f}' for x in delta))
    print("util_goal :", ','.join(f'{x:.4f}' for x in goal))
    print("util_beta :", float(K))
    print("also report: per-dim z std", np.std(Z, 0).round(3).tolist())


if __name__ == '__main__':
    sys.exit(main())
