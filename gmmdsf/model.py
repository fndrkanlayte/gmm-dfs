import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MLPQNet(nn.Module):
    """Independent per-agent Q-network (the IDDQN default).

    Every intersection is scored from its OWN observation only -- there is no
    cross-agent information flow, exactly the "independent learners" setting
    (IQL/IDQN). Parameters are shared across agents (one network applied to each
    agent's obs), and an optional per-agent identity embedding keeps
    heterogeneous intersections distinguishable while introducing zero coupling.

    Input  state : [B, N, state_dim]
    Output Q     : [B, N, action_dim]
    """

    def __init__(self, state_dim, action_dim, agent_num, hidden_dim=64,
                 use_agent_emb=True):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.agent_num = agent_num
        self.use_agent_emb = use_agent_emb

        in_dim = state_dim
        if use_agent_emb:
            self.agent_emb = nn.Embedding(agent_num, hidden_dim)
            in_dim = state_dim + hidden_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim))

    def _augment(self, state):
        """Append each agent's own identity embedding to its own obs.
        state: [B, N, state_dim] -> [B, N, in_dim]. No cross-agent mixing."""
        if self.use_agent_emb:
            emb = self.agent_emb.weight.unsqueeze(0).expand(state.shape[0], -1, -1)
            state = torch.cat([state, emb], dim=-1)
        return state

    def forward(self, state):
        x = self._augment(state)
        return self.net(x)   # [B, N, action_dim]


class GATLayer(nn.Module):
    """Single graph-attention layer (Velickovic et al. 2018), masked to a fixed
    1-hop adjacency.

    Each node aggregates a learned attention-weighted combination of its 1-hop
    neighbours' (and its own) transformed features. Attention logits for
    non-neighbour pairs are masked to -inf before the softmax, so a node only
    ever attends within its neighbourhood -- the "consider 1-hop neighbours"
    coupling the GAT variant is meant to add on top of the independent MLP.

    adj : [N, N] float {0,1} adjacency WITH self-loops (fixed, not learned).
    """

    def __init__(self, in_dim, out_dim, n_heads=4, concat=True, dropout=0.0,
                 leaky_slope=0.2):
        super().__init__()
        self.n_heads = n_heads
        self.out_dim = out_dim
        self.concat = concat
        self.W = nn.Linear(in_dim, out_dim * n_heads, bias=False)
        # Attention: separate source/target scoring vectors per head.
        self.a_src = nn.Parameter(torch.empty(n_heads, out_dim))
        self.a_dst = nn.Parameter(torch.empty(n_heads, out_dim))
        self.leaky = nn.LeakyReLU(leaky_slope)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, h, adj):
        # h: [B, N, in_dim]; adj: [N, N] with self-loops.
        B, N, _ = h.shape
        Wh = self.W(h).view(B, N, self.n_heads, self.out_dim)   # [B, N, H, d]
        Wh = Wh.permute(0, 2, 1, 3)                              # [B, H, N, d]
        # e_ij = LeakyReLU(a_src . Wh_i + a_dst . Wh_j)  (additive attention).
        src = (Wh * self.a_src.view(1, self.n_heads, 1, self.out_dim)).sum(-1)  # [B,H,N]
        dst = (Wh * self.a_dst.view(1, self.n_heads, 1, self.out_dim)).sum(-1)  # [B,H,N]
        e = self.leaky(src.unsqueeze(-1) + dst.unsqueeze(-2))    # [B, H, N, N]
        # Mask out non-neighbours before the softmax.
        mask = (adj.unsqueeze(0).unsqueeze(0) > 0)               # [1, 1, N, N]
        e = e.masked_fill(~mask, float('-inf'))
        attn = torch.softmax(e, dim=-1)                          # over neighbours j
        attn = self.dropout(attn)
        out = torch.matmul(attn, Wh)                            # [B, H, N, d]
        out = out.permute(0, 2, 1, 3)                           # [B, N, H, d]
        if self.concat:
            return out.reshape(B, N, self.n_heads * self.out_dim)
        return out.mean(dim=2)                                  # [B, N, d]


class GATQNet(nn.Module):
    """1-hop GAT Q-network variant.

    A per-agent encoder lifts each intersection's observation (+ optional
    identity embedding) into a hidden representation; ONE graph-attention layer
    then mixes each node with its 1-hop neighbours (masked by the fixed road
    adjacency); a per-agent head reads out Q-values. This is the only place
    cross-agent information enters -- exactly a 1-hop coupled Q-function, in
    contrast to the fully independent :class:`MLPQNet`.

    The adjacency is registered as a buffer so it moves with .to(device) and is
    saved/loaded with the network.

    Input  state : [B, N, state_dim]
    Output Q     : [B, N, action_dim]
    """

    def __init__(self, state_dim, action_dim, agent_num, adj, hidden_dim=64,
                 n_heads=4, use_agent_emb=True):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.agent_num = agent_num
        self.use_agent_emb = use_agent_emb

        in_dim = state_dim
        if use_agent_emb:
            self.agent_emb = nn.Embedding(agent_num, hidden_dim)
            in_dim = state_dim + hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU())
        # 1-hop attention mixing. Multi-head, concat -> hidden_dim, then a
        # residual add keeps each node's own encoding available even if a node
        # is isolated.
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"
        self.gat = GATLayer(hidden_dim, hidden_dim // n_heads, n_heads=n_heads,
                            concat=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim))

        adj = torch.as_tensor(adj, dtype=torch.float32)
        self.register_buffer('adj', adj)   # [N, N] with self-loops

    def _augment(self, state):
        if self.use_agent_emb:
            emb = self.agent_emb.weight.unsqueeze(0).expand(state.shape[0], -1, -1)
            state = torch.cat([state, emb], dim=-1)
        return state

    def forward(self, state):
        x = self._augment(state)
        h = self.encoder(x)                       # [B, N, hidden]
        h = h + self.gat(h, self.adj)             # 1-hop neighbour mixing (residual)
        h = F.relu(h)
        return self.head(h)                       # [B, N, action_dim]


class SFQNet(nn.Module):
    """Per-agent SUCCESSOR-FEATURE network: predicts psi(s, a) in R^K per action.

    Identical independent-learner backbone to :class:`MLPQNet` (parameters
    shared across agents, own-observation only, optional identity embedding) but
    the head emits ``action_dim * feat_dim`` values reshaped to
    ``[B, N, action_dim, feat_dim]`` -- the expected discounted feature vector
    psi_i(s_i, a). The task value is the closed form ``Q = psi . w`` computed by
    the caller, so ONE trained psi transfers to any task weighting w without
    retraining. Trained w-free (USFA): a fresh w is drawn per episode and the
    SF-TD target uses phi + gamma * psi(s2, a2) (see agent.py).

    Input  state : [B, N, state_dim]
    Output psi   : [B, N, action_dim, feat_dim]
    """

    def __init__(self, state_dim, action_dim, feat_dim, agent_num,
                 hidden_dim=64, use_agent_emb=True):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.feat_dim = feat_dim
        self.agent_num = agent_num
        self.use_agent_emb = use_agent_emb

        in_dim = state_dim
        if use_agent_emb:
            self.agent_emb = nn.Embedding(agent_num, hidden_dim)
            in_dim = state_dim + hidden_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * feat_dim))

    def _augment(self, state):
        if self.use_agent_emb:
            emb = self.agent_emb.weight.unsqueeze(0).expand(state.shape[0], -1, -1)
            state = torch.cat([state, emb], dim=-1)
        return state

    def forward(self, state):
        B, N, _ = state.shape
        x = self._augment(state)
        psi = self.net(x)                                  # [B, N, A*K]
        return psi.view(B, N, self.action_dim, self.feat_dim)

    def q_values(self, state, w):
        """Closed-form transfer value Q = psi . w. ``w`` is [K] or [B,N,K]."""
        psi = self.forward(state)                          # [B,N,A,K]
        w_t = torch.as_tensor(w, dtype=psi.dtype, device=psi.device)
        if w_t.dim() == 1:
            return torch.einsum('bnak,k->bna', psi, w_t)   # [B,N,A]
        return torch.einsum('bnak,bnk->bna', psi, w_t)     # per-agent w


class GMMQNet(nn.Module):
    """Per-agent GAUSSIAN-MIXTURE outcome network: models the full outcome
    distribution Z(s,a) ~ sum_c pi_c * N(mu_c, diag(sigma_c^2)) in R^K instead
    of only its mean psi(s,a).

    Same independent-learner backbone as :class:`SFQNet` (shared params across
    agents, own-observation + optional identity embedding) but the head emits
    ``action_dim * n_comp * (1 + 2*feat_dim)`` values: mixture logits, component
    means, and component log-stds per action.

    DROP-IN CONTRACT: ``forward`` and ``q_values`` deliberately return the
    mixture MEAN (the effective psi), so every existing consumer --
    ``NeighborComposer`` and the linear deployment rule -- works unchanged and the
    linear-task behaviour is identical by construction. Distributional
    information is only exposed through :meth:`outcome_dist` and the closed-form
    utility helpers below.

    Input  state : [B, N, state_dim]
    forward      : [B, N, action_dim, feat_dim]   (mixture mean, == effective psi)
    outcome_dist : (log_pi [B,N,A,C], mu [B,N,A,C,K], sigma [B,N,A,C,K])
    """

    def __init__(self, state_dim, action_dim, feat_dim, agent_num,
                 hidden_dim=64, use_agent_emb=True, n_comp=3, sigma_min=1e-3):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.feat_dim = feat_dim
        self.agent_num = agent_num
        self.use_agent_emb = use_agent_emb
        self.n_comp = n_comp
        self.sigma_min = sigma_min

        in_dim = state_dim
        if use_agent_emb:
            self.agent_emb = nn.Embedding(agent_num, hidden_dim)
            in_dim = state_dim + hidden_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * n_comp * (1 + 2 * feat_dim)))

    def _augment(self, state):
        if self.use_agent_emb:
            emb = self.agent_emb.weight.unsqueeze(0).expand(state.shape[0], -1, -1)
            state = torch.cat([state, emb], dim=-1)
        return state

    def outcome_dist(self, state):
        """Full per-action outcome GMM. Returns (log_pi, mu, sigma) with
        log_pi [B,N,A,C] (raw logits -- apply log_softmax for log-probs),
        mu/sigma [B,N,A,C,K] (sigma = softplus-free exp of logit, floored at
        sigma_min)."""
        B, N, _ = state.shape
        x = self._augment(state)
        out = self.net(x)                                    # [B,N,A*C*(1+2K)]
        out = out.view(B, N, self.action_dim, self.n_comp, 1 + 2 * self.feat_dim)
        log_pi = out[..., 0]                                 # [B,N,A,C]
        mu = out[..., 1:1 + self.feat_dim]                   # [B,N,A,C,K]
        log_sig = out[..., 1 + self.feat_dim:]               # [B,N,A,C,K]
        # bounded dispersion: sigma_min floor + cap keeps NLL stable (a
        # max-outcome magnitude of ~63 makes sigma > ~64 meaningless)
        sigma = log_sig.exp().clamp(min=self.sigma_min, max=64.0)  # [B,N,A,C,K]
        return log_pi, mu, sigma

    def forward(self, state):
        """Mixture mean psi_eff(s,a) = sum_c pi_c mu_c  -> [B,N,A,K].
        Identical semantics to SFQNet.forward so downstream code is unchanged."""
        log_pi, mu, _ = self.outcome_dist(state)
        pi = torch.softmax(log_pi, dim=-1)                   # [B,N,A,C]
        return (pi.unsqueeze(-1) * mu).sum(dim=3)            # [B,N,A,K]

    def q_values(self, state, w):
        """Closed-form linear utility E[w . Z] = sum_c pi_c (mu_c . w).
        ``w`` is [K] or [B,N,K]. Same signature as SFQNet.q_values."""
        log_pi, mu, _ = self.outcome_dist(state)
        pi = torch.softmax(log_pi, dim=-1)                   # [B,N,A,C]
        w_t = torch.as_tensor(w, dtype=mu.dtype, device=mu.device)
        if w_t.dim() == 1:
            return torch.einsum('bnack,bnac,k->bna', mu, pi, w_t)
        return torch.einsum('bnack,bnac,bnk->bna', mu, pi, w_t)

    @torch.no_grad()
    def warm_start_from_sf(self, sf_state_dict, sigma0=0.1, jitter=0.05):
        """Initialize this GMM net from a trained SFQNet state dict so that
        the mixture MEAN equals the original psi at step 0.

        Copies: agent_emb, the two shared MLP layers (identical shapes), and
        the last layer's mean rows -- every component's mu head starts from
        the sf psi, all C copies identical, so sum_c pi_c mu_c == psi for ANY
        pi. Mixture logits stay at their small random init (near-uniform pi);
        component sigmas start at sigma0 via the last-layer bias of the
        log-sigma rows. Returns the list of initialized parameter names."""
        C, K, A = self.n_comp, self.feat_dim, self.action_dim
        own = self.state_dict()
        copied = []
        for name in ('agent_emb.weight', 'net.0.weight', 'net.0.bias',
                     'net.2.weight', 'net.2.bias'):
            if name in sf_state_dict and sf_state_dict[name].shape == own[name].shape:
                own[name].copy_(sf_state_dict[name])
                copied.append(name)
        w_sf, b_sf = sf_state_dict['net.4.weight'], sf_state_dict['net.4.bias']
        assert w_sf.shape[0] == A * K, (w_sf.shape, A, K)
        gout = 1 + 2 * K                                   # per (a,c) block
        w_g, b_g = own['net.4.weight'], own['net.4.bias']
        w_mu = w_sf.view(A, K, -1)                          # [A,K,H]
        b_mu = b_sf.view(A, K)
        g = torch.Generator(device=w_g.device)
        g.manual_seed(0)
        for a in range(A):
            # zero-sum per-component jitter: sum_c mu_c == C*psi for each
            # (a,k) -> mixture mean is untouched, but components start
            # SEPARATED, breaking the exchangeability saddle of identical
            # copies under the NLL update.
            jit = [jitter * (0.5 + b_mu[a].abs()) * torch.randn(
                K, generator=g, device=w_g.device)
                for _ in range(max(0, C - 1))]
            for c in range(C):
                base = (a * C + c) * gout
                w_g[base].zero_()                           # log_pi -> 0 (uniform pi)
                b_g[base].zero_()
                w_g[base + 1:base + 1 + K].copy_(w_mu[a])   # mu rows
                if c < C - 1:
                    b_g[base + 1:base + 1 + K].copy_(b_mu[a] + jit[c])
                elif jit:
                    b_g[base + 1:base + 1 + K].copy_(
                        b_mu[a] - torch.stack(jit).sum(0))
                else:
                    b_g[base + 1:base + 1 + K].copy_(b_mu[a])
                w_g[base + 1 + K:base + 1 + 2 * K].zero_()  # log_sig rows
                b_g[base + 1 + K:base + 1 + 2 * K].fill_(
                    float(torch.log(torch.as_tensor(sigma0))))
        copied += ['net.4.weight(mu rows xC)', 'net.4.bias(mu rows, log-sig)']
        return copied


# ===================================================================== #
# Closed-form utilities E[f(Z)] under a diagonal-Gaussian-component GMM.
# All functions broadcast over leading dims [B,N,A] with per-dim K.
# ===================================================================== #

_SQRT2 = 1.4142135623730951


def _phi_norm(x):
    """Standard normal pdf."""
    return torch.exp(-0.5 * x * x) / 2.5066282746310002


def _kdim(x, ref):
    """Reshape a task tensor to broadcast against [B,N,A,C,K]:
    [K] -> [1,1,1,1,K]; [N,K]/[B,N,K]/[B,N,A,K] -> [..,1,1,K]."""
    t = torch.as_tensor(x, dtype=ref.dtype, device=ref.device)
    K = ref.shape[-1]
    if t.dim() == 1:
        return t.view(*([1] * (ref.dim() - 1)), K)
    while t.dim() < ref.dim() - 1:
        t = t.unsqueeze(-2)
    return t.unsqueeze(-2)


def gmm_util_linear(log_pi, mu, sigma, w):
    """E[w . Z] = sum_c pi_c (mu_c . w). w: [K] or [B,N,K] -> [B,N,A]."""
    pi = torch.softmax(log_pi, dim=-1)
    w_t = torch.as_tensor(w, dtype=mu.dtype, device=mu.device)
    if w_t.dim() == 1:
        return torch.einsum('bnack,bnac,k->bna', mu, pi, w_t)
    return torch.einsum('bnack,bnac,bnk->bna', mu, pi, w_t)


def gmm_util_box(log_pi, mu, sigma, w, delta):
    """E[(w . Z) * 1{Z_k <= delta_k for all k}].

    Threshold/safety objective (GDSF 'Const' family): reward accrues only when
    EVERY feature outcome stays under its bound delta_k. For a diagonal
    component the truncated first moment factorises:

        E[Z_k 1{box}] = mu_k * P(box) - sigma_k * phi(u_k) * prod_{j!=k} Phi(u_j)
        u_j = (delta_j - mu_j) / sigma_j ,   P(box) = prod_j Phi(u_j)

    w/delta: [K] or broadcastable [B,N,K] -> [B,N,A].
    """
    pi = torch.softmax(log_pi, dim=-1)                       # [B,N,A,C]
    w_t = _kdim(w, mu)                                       # [.,.,1,1,K]
    d_t = _kdim(delta, mu)
    u = (d_t - mu) / sigma                                   # [B,N,A,C,K]
    cdf = torch.special.ndtr(u)                              # Phi(u)
    pdf = _phi_norm(u)
    p_box = cdf.prod(dim=-1, keepdim=True)                   # [B,N,A,C,1]
    # prod_{j!=k} Phi(u_j) = P(box) / Phi(u_k), guarded against Phi->0.
    cdf_safe = cdf.clamp(min=1e-12)
    rest = p_box / cdf_safe                                  # [B,N,A,C,K]
    ez_box = mu * p_box - sigma * pdf * rest                 # [B,N,A,C,K]
    weighted = (ez_box * w_t).sum(-1)                        # [B,N,A,C]
    return (pi * weighted).sum(-1)                           # [B,N,A]


def gmm_util_gauss(log_pi, mu, sigma, goal, beta):
    """E[exp(-||Z - g||^2 / beta)] -- Gaussian-shaped precision objective.

    Per component and dim: E[exp(-(Z-g)^2/beta)]
        = sqrt(beta/(beta+2*sigma^2)) * exp(-(mu-g)^2/(beta+2*sigma^2)).
    goal: [K] or broadcastable; beta: scalar > 0 -> [B,N,A].
    """
    pi = torch.softmax(log_pi, dim=-1)
    g_t = _kdim(goal, mu)                                    # [.,.,1,1,K]
    var = sigma * sigma + 1e-12
    denom = beta + 2.0 * var                                 # [B,N,A,C,K]
    log_term = 0.5 * (torch.log(torch.as_tensor(beta, dtype=mu.dtype)) - denom.log()) \
        - ((mu - g_t) ** 2) / denom
    per_comp = torch.exp(log_term.sum(-1))                   # [B,N,A,C]
    return (pi * per_comp).sum(-1)                           # [B,N,A]


def gmm_sample_outcome(log_pi, mu, sigma, n_samples, generator=None):
    """Draw samples z ~ GMM(s,a). log_pi [B,N,A,C], mu/sigma [B,N,A,C,K]
    -> z [B,N,A,S,K]."""
    pi = torch.softmax(log_pi, dim=-1)
    B, N, A, C = pi.shape
    K = mu.shape[-1]
    comp = torch.multinomial(pi.reshape(-1, C), n_samples,
                             replacement=True, generator=generator)
    comp = comp.view(B, N, A, n_samples)
    idx = comp.unsqueeze(-1).expand(-1, -1, -1, -1, K)       # [B,N,A,S,K]
    mu_sel = mu.gather(3, idx)                               # [B,N,A,S,K]
    sg_sel = sigma.gather(3, idx)
    return mu_sel + sg_sel * torch.randn_like(mu_sel)


class NeighborComposer(nn.Module):
    """Neighbor-limited composer (frozen-backbone adaptation).

    Warm-starts from the closed-form SF transfer utilities ``u_i = psi_i . w_i``
    and adds a bounded residual computed from a SINGLE 1-hop graph-attention
    round over the real road adjacency -- so global task information reaches a
    node's decision ONLY through its immediate neighbours (never a global
    softmax over all intersections, which is infeasible at 196 agents).

        u_i(s, a) = psi_i(s_i, a) . w_i  +  rho * tanh( head( gat_i(context) ) )
        context_j = [ enc(s_j) , w_j ]     (own obs + own task weights)

    The residual head is zero-initialized so training begins exactly at the
    transfer policy. IGM is preserved trivially: the composer outputs per-agent
    utilities and the joint action is per-agent argmax (decentralized
    execution). Positive-mixing / monotone value is inherited from summing
    per-agent utilities (additive team reward).

    Input  state : [B, N, state_dim] ; w : [B, N, K] (per-agent task weights)
    Output u     : [B, N, action_dim]
    """

    RESID_SCALE = 4.0

    def __init__(self, sf_net, adj, feat_dim, hidden_dim=64, n_heads=4,
                 use_agent_emb=True, scale_norm=False):
        super().__init__()
        self.sf_net = sf_net                     # frozen or fine-tuned SF backbone
        # objective-scale normalisation (heterogeneous objectives differ by
        # orders of magnitude): per-agent utility scale, learned as a running
        # std of the agent's TD targets; bounds the residual in the agent's
        # own units and weights the TD loss by 1/scale^2 (see SFAgent).
        self.scale_norm = scale_norm
        # speed-gated composer: the neighbour residual is only applied for
        # agents whose own task has a positive speed weight (w[...,3] > 0);
        # pure cost-objective agents keep the closed-form transfer value.
        self.gate_speed = False
        self.register_buffer('agent_scale', torch.ones(int(torch.as_tensor(adj).shape[0])))
        self.feat_dim = feat_dim
        state_dim = sf_net.state_dim
        action_dim = sf_net.action_dim

        ctx_dim = state_dim + feat_dim           # [own obs, own w]
        self.encoder = nn.Sequential(
            nn.Linear(ctx_dim, hidden_dim), nn.ReLU())
        assert hidden_dim % n_heads == 0
        self.gat = GATLayer(hidden_dim, hidden_dim // n_heads, n_heads=n_heads,
                            concat=True)
        # zero-initialized final layer -> residual starts at 0 (transfer policy)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

        adj = torch.as_tensor(adj, dtype=torch.float32)
        self.register_buffer('adj', adj)

    def forward(self, state, w):
        # base transfer utilities u = psi . w  (per-agent w)
        u_base = self.sf_net.q_values(state, w)            # [B,N,A]
        # neighbour-limited residual
        ctx = torch.cat([state, w], dim=-1)                # [B,N,state+K]
        h = self.encoder(ctx)
        h = h + self.gat(h, self.adj)                      # 1-hop mixing
        h = F.relu(h)
        resid = self.RESID_SCALE * torch.tanh(self.head(h))  # [B,N,A]
        if self.scale_norm:
            resid = resid * self.agent_scale.view(1, -1, 1)
        if self.gate_speed:
            resid = resid * (w[..., 3:4] > 0).to(resid.dtype)
        return u_base + resid
