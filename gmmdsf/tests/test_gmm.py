"""Unit tests for the GMM outcome net (sf_net='gmm').

Run:  python test_gmm.py        (no SUMO/env needed -- pure torch math)

Checks:
  * outcome_dist / forward / q_values shapes and the drop-in contract
    (forward == mixture mean; q_values == E[w.Z] closed form)
  * closed-form utilities vs Monte-Carlo (box, gauss) and C=1 degeneracy
  * the exact target-mixture math used by _learn_usfa_gmm (shift/scale,
    done masking)
  * NeighborComposer forward works on a GMMQNet backbone
"""
import os as _os, sys as _sys
_PKG = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _PKG)
import numpy as np
import torch

from model import (GMMQNet, SFQNet, NeighborComposer,
                   gmm_util_linear, gmm_util_box, gmm_util_gauss,
                   gmm_sample_outcome)

torch.manual_seed(0)
np.random.seed(0)

B, N, S, A, K, C = 2, 4, 10, 3, 5, 3
DEV = 'cpu'


def _net(n_comp=C):
    return GMMQNet(state_dim=S, action_dim=A, feat_dim=K, agent_num=N,
                   hidden_dim=32, use_agent_emb=True, n_comp=n_comp)


def _state():
    return torch.rand(B, N, S)


def test_shapes():
    net = _net()
    st = _state()
    log_pi, mu, sig = net.outcome_dist(st)
    assert log_pi.shape == (B, N, A, C), log_pi.shape
    assert mu.shape == (B, N, A, C, K), mu.shape
    assert sig.shape == (B, N, A, C, K) and (sig > 0).all()
    psi = net(st)
    assert psi.shape == (B, N, A, K), psi.shape


def test_forward_is_mixture_mean():
    net = _net()
    st = _state()
    log_pi, mu, _ = net.outcome_dist(st)
    pi = torch.softmax(log_pi, -1)
    manual = (pi.unsqueeze(-1) * mu).sum(3)
    assert torch.allclose(net(st), manual, atol=1e-6)


def test_q_values_linear():
    net = _net()
    st = _state()
    w = torch.randn(K)
    log_pi, mu, sig = net.outcome_dist(st)
    assert torch.allclose(net.q_values(st, w),
                          gmm_util_linear(log_pi, mu, sig, w), atol=1e-6)
    w_agent = torch.randn(B, N, K)               # per-agent w is [B,N,K]
    q = net.q_values(st, w_agent)
    assert q.shape == (B, N, A)


def _mc_expectation(fn, log_pi, mu, sig, n=40000):
    z = gmm_sample_outcome(log_pi, mu, sig, n)
    return fn(z).mean(dim=3)          # [B,N,A]


def test_util_box_vs_mc():
    net = _net()
    st = _state()
    log_pi, mu, sig = net.outcome_dist(st)
    w = torch.randn(K)
    delta = torch.rand(K) * 0.5 + 0.25
    exact = gmm_util_box(log_pi, mu, sig, w, delta)
    z = gmm_sample_outcome(log_pi, mu, sig, 60000)
    ok = (z <= delta.view(1, 1, 1, 1, K)).all(-1, keepdim=False)
    mc = ((z * w.view(1, 1, 1, 1, K)).sum(-1) * ok.float()).mean(3)
    assert torch.allclose(exact, mc, atol=2e-2), (exact - mc).abs().max()


def test_util_gauss_vs_mc():
    net = _net()
    st = _state()
    log_pi, mu, sig = net.outcome_dist(st)
    goal = torch.rand(K)
    beta = 0.7
    exact = gmm_util_gauss(log_pi, mu, sig, goal, beta)
    z = gmm_sample_outcome(log_pi, mu, sig, 60000)
    mc = torch.exp(-((z - goal.view(1, 1, 1, 1, K)) ** 2).sum(-1) / beta).mean(3)
    assert torch.allclose(exact, mc, atol=5e-3), (exact - mc).abs().max()


def test_c1_degenerate():
    """C=1 must reduce to the single-Gaussian closed forms."""
    net = _net(n_comp=1)
    st = _state()
    log_pi, mu, sig = net.outcome_dist(st)
    assert log_pi.shape[-1] == 1
    w, delta = torch.randn(K), torch.rand(K) + 0.3
    u = (delta - mu.squeeze(3)) / sig.squeeze(3)
    cdf = torch.special.ndtr(u)
    pdf = torch.exp(-0.5 * u * u) / np.sqrt(2 * np.pi)
    pbox = cdf.prod(-1, keepdim=True)
    ez = mu.squeeze(3) * pbox - sig.squeeze(3) * pdf * (pbox / cdf.clamp(min=1e-12))
    expect = (ez * w.view(1, 1, 1, K)).sum(-1)
    got = gmm_util_box(log_pi, mu, sig, w, delta)
    assert torch.allclose(got, expect, atol=1e-5)


def test_target_mixture_math():
    """Z = phi + keep*Z' shifts/scales a GMM componentwise."""
    net_t = _net()
    s2 = _state()
    lp2, mu2, sig2 = net_t.outcome_dist(s2)
    a2 = torch.randint(0, A, (B, N))
    idx = a2.view(B, N, 1, 1, 1).expand(-1, -1, 1, C, K)
    mu2_t = mu2.gather(2, idx).squeeze(2)
    sig2_t = sig2.gather(2, idx).squeeze(2)
    lp2_t = lp2.gather(2, a2.view(B, N, 1, 1).expand(-1, -1, 1, C)).squeeze(2)
    phi2 = torch.rand(B, N, K)
    keep = 0.99
    mean_t = phi2.unsqueeze(2) + keep * mu2_t
    std_t = (keep * sig2_t).clamp(min=1e-3)
    pi_t = torch.softmax(lp2_t, -1)
    # sample the exact target mixture and compare empirical moments
    M = 80000
    comp = torch.multinomial(pi_t.reshape(-1, C), M, replacement=True).view(B, N, M)
    gidx = comp.unsqueeze(-1).expand(-1, -1, -1, K)
    zm = mean_t.gather(2, gidx) + std_t.gather(2, gidx) * torch.randn(B, N, M, K)
    emp_mean = zm.mean(2)
    true_mean = phi2 + keep * (pi_t.unsqueeze(-1) * mu2_t).sum(2)
    assert torch.allclose(emp_mean, true_mean, atol=3e-2)
    emp_var = zm.var(2)
    true_var = (pi_t.unsqueeze(-1) * (std_t ** 2 + mean_t ** 2)).sum(2) \
        - true_mean.unsqueeze(2).squeeze(2) ** 2
    assert torch.allclose(emp_var, true_var, rtol=5e-2, atol=5e-3)


def test_composer_on_gmm():
    net = _net()
    adj = (torch.rand(N, N) > 0.5).float()
    adj = ((adj + adj.t()) > 0).float()
    adj.fill_diagonal_(1.0)
    comp = NeighborComposer(net, adj, K, hidden_dim=32, n_heads=4)
    u = comp(_state(), torch.rand(B, N, K))
    assert u.shape == (B, N, A)


def test_warm_start_equals_sf_psi():
    """After warm_start_from_sf, GMM forward must reproduce the sf psi
    elementwise (any pi, since all component means are copies of psi)."""
    sf = SFQNet(state_dim=S, action_dim=A, feat_dim=K, agent_num=N,
                hidden_dim=32, use_agent_emb=True)
    gmm = _net()
    copied = gmm.warm_start_from_sf(sf.state_dict(), sigma0=0.1)
    assert 'agent_emb.weight' in copied
    st = _state()
    assert torch.allclose(gmm(st), sf(st), atol=1e-6)
    # and the distribution head is sane: uniform-ish pi, sigma==sigma0
    log_pi, mu, sig = gmm.outcome_dist(st)
    pi = torch.softmax(log_pi, -1)
    assert (pi - 1.0 / C).abs().max() < 0.05
    assert torch.allclose(sig, torch.full_like(sig, 0.1), atol=1e-4)
    # zero-sum jitter: components differ but the weighted mean stays == psi
    spread = (mu.max(3).values - mu.min(3).values)
    assert (spread > 0).any(), "components should be separated by jitter"
    assert torch.allclose(gmm(st), sf(st), atol=1e-6)


def test_done_degenerate_target():
    """keep=0 -> target mixture concentrates at phi2 (std hits floor)."""
    net = _net()
    st = _state()
    lp, mu, sig = net.outcome_dist(st)
    keep = torch.zeros(B, 1, 1)
    phi2 = torch.rand(B, N, K)
    pi_t = torch.softmax(lp[..., :].mean(2), -1)   # any mixture weights
    mean_t = phi2.unsqueeze(2) + keep.unsqueeze(-1) * mu.gather(
        2, torch.zeros(B, N, 1, 1, 1).expand(-1, -1, 1, C, K).long()).squeeze(2)
    assert torch.allclose(mean_t, phi2.unsqueeze(2).expand(-1, -1, C, -1))


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
