"""Function-preserving input expansion of a GMMQNet checkpoint for the
downstream observation (obs_baselines.DownstreamObservation).

Network input = [obs (n_old) | timestep (1) | agent embedding (E)].
New obs = [default obs (n_old) | out-lane density (n_add)], so the first layer
gets n_add new zero columns inserted after column n_old; the timestep and
embedding columns shift right. At initialization the expanded network is
EXACTLY the old one (verified numerically below) for any downstream values.

Usage: python expand_ckpt_downstream.py IN.pth OUT.pth --n_old 41 --n_add 12
"""
import os as _os, sys as _sys
_PKG = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _PKG)
_os.chdir(_PKG)  # all inputs/outputs are relative to gmmdsf/
import argparse

import torch

from model import GMMQNet


def expand(sd, n_old, n_add):
    W = sd['net.0.weight']                      # [H, n_old + 1 + E]
    H, din = W.shape
    E = din - n_old - 1
    new = torch.zeros(H, din + n_add, dtype=W.dtype)
    new[:, :n_old] = W[:, :n_old]
    new[:, n_old + n_add] = W[:, n_old]                      # timestep
    new[:, n_old + n_add + 1:] = W[:, n_old + 1:]            # embedding
    out = dict(sd)
    out['net.0.weight'] = new
    return out, E


def main():
    p = argparse.ArgumentParser()
    p.add_argument('src')
    p.add_argument('dst')
    p.add_argument('--n_old', type=int, default=41)
    p.add_argument('--n_add', type=int, default=12)
    a = p.parse_args()
    sd = torch.load(a.src, map_location='cpu')
    new, E = expand(sd, a.n_old, a.n_add)
    N, A, K = sd['agent_emb.weight'].shape[0], 16, 5
    C = sd['net.4.weight'].shape[0] // (A * (1 + 2 * K))
    old_net = GMMQNet(a.n_old + 1, A, K, N, hidden_dim=E, n_comp=C)
    new_net = GMMQNet(a.n_old + a.n_add + 1, A, K, N, hidden_dim=E, n_comp=C)
    old_net.load_state_dict(sd)
    new_net.load_state_dict(new)
    torch.manual_seed(0)
    s_old = torch.rand(4, N, a.n_old + 1)
    down = torch.rand(4, N, a.n_add)
    s_new = torch.cat([s_old[..., :a.n_old], down, s_old[..., a.n_old:]], -1)
    w = torch.randn(K)
    with torch.no_grad():
        d = (old_net.q_values(s_old, w) - new_net.q_values(s_new, w)).abs().max().item()
        lp0, mu0, sg0 = old_net.outcome_dist(s_old)
        lp1, mu1, sg1 = new_net.outcome_dist(s_new)
        d2 = max((lp0 - lp1).abs().max().item(), (mu0 - mu1).abs().max().item(),
                 (sg0 - sg1).abs().max().item())
    print(f'max |dQ| = {d:.3e}, max |d dist| = {d2:.3e}  (C={C}, E={E}, N={N})')
    assert d < 1e-5 and d2 < 1e-5, 'expansion is not function-preserving'
    torch.save(new, a.dst)
    print('saved', a.dst)


if __name__ == '__main__':
    main()
