import numpy as np
import torch
import torch.nn as nn
from model import (SFQNet, GMMQNet, NeighborComposer,
                   gmm_util_linear, gmm_util_box, gmm_util_gauss)
from utils import ReplayBuffer
from graph import build_coupling_adjacency, describe_support
from composite_reward import (FEATURE_DIM, DEFAULT_W, sample_task,
                              sample_task_family, sample_task_family_v2,
                              sample_task_family_v3, sample_block_tasks)


class SFReplayBuffer(ReplayBuffer):
    """Replay buffer that additionally stores the per-agent feature vector phi
    observed at the NEXT state (needed for the SF-TD target phi + gamma*psi).

    Each transition: state [N,s], action [N], phi2 [N,K], next_state [N,s],
    done (scalar). The scalar reward is NOT stored -- under SF the reward is
    reconstructed as phi2 . w for whatever task w the update uses (w-free USFA)."""

    def append(self, state, action, phi2, next_state, done):
        super().append(state, action, phi2, next_state, done)

    def sample(self, batch_size):
        states, actions, phi2, next_states, dones = super().sample(batch_size)
        return states, actions, phi2, next_states, dones


class SFAgent:
    """GMM-DSF / successor-feature agent for multi-intersection signal control.

    Roles, selected by ``mode``:

    * ``mode='usfa'`` -- pre-train the task-free outcome model with a TD update.
      With ``sf_net='gmm'`` the model is an action-conditioned Gaussian mixture
      over discounted cumulative outcome features (distributional TD on an
      analytically constructed mixture target); with ``sf_net='sf'`` it is the
      point-estimate successor feature psi. A fresh task (homogeneous w or
      per-intersection w_i from the task family) is drawn every episode. The
      same mode, evaluated on a fixed target task, is zero-shot deployment:
      a_i = argmax_a E[U_i(Z_i(s_i, a))], fully decentralized.
    * ``mode='composer'`` -- freeze the trained backbone and fine-tune a bounded
      one-hop :class:`NeighborComposer` residual on a specific target task
      (optionally speed-gated or scale-normalised).

    Same reset/act/step/learn/save/load interface as :class:`Agent`.
    """

    def __init__(self, env, reward_fn, device, mode='usfa', hidden_dim=64,
                 lr=1e-3, gamma=0.99, horizon_len=60, training_episodes=500,
                 buffer_size=100000, n_heads=4, use_agent_emb=True,
                 target_update_freq=200, grad_clip=1.0, task_scale=(0.5, 1.5),
                 task_family=False,
                 sf_ckpt=None, sf_net='sf', n_comp=3,
                 nll_samples=16, lambda_nll=1.0, lambda_mean=1.0, ent_reg=0.0,
                 sigma_min=1e-3, sf_init=None, freeze_mean=False, **kwargs):
        assert mode in ('usfa', 'composer'), mode
        self.env = env
        self.reward_fn = reward_fn
        self.device = device
        self.mode = mode
        self.gamma = gamma
        self.horizon_len = horizon_len
        self.grad_clip = grad_clip
        self.target_update_freq = target_update_freq
        self.task_scale = task_scale
        self.task_family = task_family
        self.base_seed = None
        self._learn_steps = 0
        self.K = FEATURE_DIM

        agents_list = env.possible_agents if hasattr(env, 'possible_agents') else env.agents
        self.id = sorted(agents_list)
        self.num = len(self.id)
        self.state_size = max(
            env.observation_spaces[ts].shape[0] for ts in self.id) + 1  # +timestep
        # heterogeneous action spaces: pad to the max, mask invalid actions
        self._action_ns = np.array([env.action_spaces[ts].n for ts in self.id])
        self.action_size = int(self._action_ns.max())
        self.action_mask = torch.from_numpy(
            np.arange(self.action_size)[None, None, :]
            < self._action_ns[None, :, None]).to(self.device)     # [1,N,A]

        # 1-hop road adjacency (shared by the composer's neighbour mixing).
        C, graph_source = build_coupling_adjacency(self.id, env=self.env, method="auto")
        print(f"[SFAgent] coupling graph source={graph_source} -> {describe_support(C)}")
        self.adj = C

        # ---- successor-feature backbone (+ target copy for SF-TD) ----
        # sf_net='sf' : original SFQNet (point estimate psi).
        # sf_net='gmm': GMMQNet outcome mixture Z(s,a) ~ sum_c pi_c N(mu_c,sig_c).
        #   forward/q_values return the mixture MEAN, so composer/act are
        #   unchanged on linear tasks; distributional info only enters through
        #   the GMM TD loss (usfa) and the non-linear utility path (eval).
        self.is_gmm = (sf_net == 'gmm')
        net_cls = GMMQNet if self.is_gmm else SFQNet
        net_kw = dict(hidden_dim=hidden_dim, use_agent_emb=use_agent_emb)
        if self.is_gmm:
            net_kw.update(n_comp=n_comp, sigma_min=sigma_min)
        self.nll_samples = nll_samples
        self.lambda_nll = lambda_nll
        self.lambda_mean = lambda_mean
        self.ent_reg = ent_reg
        # deployment-time utility over the outcome distribution:
        # {'type': 'linear'|'box'|'gauss', 'delta', 'goal', 'beta'}
        self.util = {'type': 'linear'}
        # eval-time discounted per-agent outcome accumulator (for realized
        # utility / violation metrics); enabled by rollout_metrics via
        # track_outcome=True.
        self.track_outcome = False
        self.outcome_disc = None
        self.sf_net = net_cls(self.state_size, self.action_size, self.K, self.num,
                              **net_kw).to(device)
        self.sf_target = net_cls(self.state_size, self.action_size, self.K, self.num,
                                 **net_kw).to(device)
        if self.is_gmm and sf_init is not None:
            sd = torch.load(sf_init, map_location=device)
            if any(k.startswith('net.4') for k in sd) and \
                    sd['net.4.weight'].shape[0] == self.action_size * self.sf_net.n_comp * (1 + 2 * self.K):
                # gmm->gmm continuation init: a GMM state dict (pi+mu+sigma
                # head rows) loads directly -- e.g. task-coverage fine-tuning
                # from a converged GMM checkpoint.
                self.sf_net.load_state_dict(sd)
                print(f"[SFAgent] GMM full-init from {sf_init} (gmm ckpt)")
            else:
                # warm-start: backbone + mean head copied from a trained
                # SFQNet so the mixture mean == original psi at step 0
                # (policy immediately usable); only the distributional head
                # has to learn.
                copied = self.sf_net.warm_start_from_sf(sd)
                print(f"[SFAgent] GMM warm-started from {sf_init}: {copied}")
        if sf_ckpt is not None:
            self.sf_net.load_state_dict(torch.load(sf_ckpt, map_location=device))
        if self.is_gmm and freeze_mean:
            # V2 arm: learn ONLY the distributional head (pi logits + log-sigma
            # rows of the last layer). Backbone and every mu row are frozen, so
            # the mixture mean -- and therefore all linear-task behaviour --
            # stays EXACTLY the warm-started psi. This isolates what the
            # outcome distribution adds on top of the point-estimate SF.
            gout = 1 + 2 * self.K
            mu_rows = torch.zeros(
                self.action_size * self.sf_net.n_comp * gout, dtype=torch.bool,
                device=self.device)
            for a in range(self.action_size):
                for c in range(self.sf_net.n_comp):
                    base = (a * self.sf_net.n_comp + c) * gout
                    mu_rows[base + 1:base + 1 + self.K] = True
            keep = (~mu_rows).float()                      # [out_dim]
            keep_w = keep.unsqueeze(-1)                    # [out_dim,1] rows
            last = self.sf_net.net[-1]
            last.weight.requires_grad = True
            last.bias.requires_grad = True
            last.weight.register_hook(lambda g, k=keep_w: g * k)
            last.bias.register_hook(lambda g, k=keep: g * k)
            for n, p in self.sf_net.named_parameters():
                if not n.startswith('net.4.'):
                    p.requires_grad = False
            print("[SFAgent] freeze_mean: backbone + mu rows frozen; "
                  "learning pi/log-sigma only")
        self.sf_target.load_state_dict(self.sf_net.state_dict())
        for p in self.sf_target.parameters():
            p.requires_grad = False
        self.sf_target.eval()

        # current task vector w (per-agent identical unless heterogeneous tasks)
        self.w = DEFAULT_W.copy()

        if mode == 'composer':
            # Freeze the SF backbone: the composer only learns the
            # neighbour-limited residual on top of the fixed transfer value.
            for p in self.sf_net.parameters():
                p.requires_grad = False
            self.sf_net.eval()
            self.composer_norm = bool(kwargs.get('composer_norm', False))
            self.composer_gate = bool(kwargs.get('composer_gate', False))
            self.composer = NeighborComposer(
                self.sf_net, self.adj, self.K, hidden_dim=hidden_dim,
                n_heads=n_heads, use_agent_emb=use_agent_emb,
                scale_norm=self.composer_norm).to(device)
            self.composer_target = NeighborComposer(
                self.sf_net, self.adj, self.K, hidden_dim=hidden_dim,
                n_heads=n_heads, use_agent_emb=use_agent_emb,
                scale_norm=self.composer_norm).to(device)
            # gate is a behaviour flag, not a saved param: set it on both
            # composers so it survives load() unchanged.
            self.composer.gate_speed = self.composer_gate
            self.composer_target.gate_speed = self.composer_gate
            self.composer_target.load_state_dict(self.composer.state_dict())
            for p in self.composer_target.parameters():
                p.requires_grad = False
            self.composer_target.eval()
            # optimize ONLY the residual params (encoder / gat / head), never
            # the shared frozen sf_net backbone.
            self._resid_params = [p for n, p in self.composer.named_parameters()
                                  if not n.startswith('sf_net.') and p.requires_grad]
            params = self._resid_params
        else:
            params = self.sf_net.parameters()

        self.optim = torch.optim.Adam(params, lr=lr)
        lr_lambda = lambda step: max(0.0, 1.0 - step / training_episodes)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=lr_lambda)
        self.buffer = SFReplayBuffer(buffer_size)

        self.epoch_stats = {'queue_sum': 0.0, 'wait_sum': 0.0, 'speed_sum': 0.0,
                            'fairness_sum': 0.0, 'steps': 0}
        self.travel_time = float('nan')
        self.throughput = 0
        print(f"[SFAgent] MODE={mode} | agents={self.num} | state={self.state_size} "
              f"| action={self.action_size} | feat={self.K}")

    # ---- checkpoint I/O ----
    def save(self, path):
        if self.mode == 'composer':
            torch.save({'sf': self.sf_net.state_dict(),
                        'composer': self.composer.state_dict()}, path)
        else:
            torch.save(self.sf_net.state_dict(), path)

    def save_sf(self, path):
        torch.save(self.sf_net.state_dict(), path)

    def load(self, path):
        blob = torch.load(path, map_location=self.device)
        if isinstance(blob, dict) and 'sf' in blob:
            self.sf_net.load_state_dict(blob['sf'])
            if self.mode == 'composer':
                sd = dict(blob['composer'])
                # pre-normalisation checkpoints have no agent_scale buffer;
                # scale 1 reproduces their behaviour exactly.
                sd.setdefault('agent_scale', torch.ones_like(self.composer.agent_scale))
                self.composer.load_state_dict(sd)
        else:
            self.sf_net.load_state_dict(blob)
        self.sf_target.load_state_dict(self.sf_net.state_dict())

    def set_base_seed(self, seed):
        self.base_seed = seed

    def _w_tensor(self):
        """Per-agent task-weight tensor [1, N, K] for the current task."""
        w = np.tile(self.w, (self.num, 1)) if self.w.ndim == 1 else self.w
        return torch.tensor(w, dtype=torch.float32, device=self.device).unsqueeze(0)

    def reset(self, train=True, episode=0, **kwargs):
        self.is_train = train
        self.episode = episode
        self.current_step = 0
        self.epoch_stats = {k: 0.0 for k in self.epoch_stats}
        self.epoch_stats['steps'] = 0
        self.outcome_disc = np.zeros((self.num, self.K), dtype=np.float64)
        # w-free USFA: draw a fresh task each TRAIN episode and push it to the
        # reward fn so the environment reward matches the psi being learned.
        if self.mode == 'usfa' and train:
            rng = np.random.RandomState((self.base_seed or 0) + episode)
            if self.task_family in (4, 5):
                # family v2 (adds equity) ; 50% per-agent i.i.d., 50% random
                # spatial blocks. RNG offset keeps training draws disjoint
                # from any evaluation task_seed stream.
                rng = np.random.RandomState(
                    (self.base_seed or 0) + 7_000_000 + episode)
                fam = sample_task_family_v3 if self.task_family == 5                     else sample_task_family_v2
                if rng.uniform() < 0.5:
                    self.w = np.stack([fam(rng) for _ in range(self.num)])
                else:
                    self.w = sample_block_tasks(rng, self.id, sampler=fam)
                if hasattr(self.reward_fn, 'set_w_map'):
                    self.reward_fn.set_w_map(
                        {ts: self.w[i] for i, ts in enumerate(self.id)})
            elif self.task_family == 3:
                # per-intersection task family: every agent draws its own w
                # from the mixture each episode -> the WHOLE task family is
                # covered within a single episode (no per-episode regime
                # switch) and matches the w-heterogeneous deployment scenario.
                self.w = np.stack([sample_task_family(rng)
                                   for _ in range(self.num)])
                if hasattr(self.reward_fn, 'set_w_map'):
                    self.reward_fn.set_w_map(
                        {ts: self.w[i] for i, ts in enumerate(self.id)})
            else:
                if self.task_family == 1:
                    self.w = sample_task_family(rng)
                elif self.task_family == 2:
                    self.w = sample_task_family(rng, mix=(0.70, 0.15, 0.10, 0.05))
                else:
                    self.w = sample_task(rng, scale=self.task_scale)
                if hasattr(self.reward_fn, 'set_w'):
                    self.reward_fn.set_w(self.w)
        if self.base_seed is not None:
            env_seed = self.base_seed + episode if train else self.base_seed + 10_000_000
            observe, _ = self.env.reset(seed=env_seed)
        else:
            observe, _ = self.env.reset()
        return observe

    def _build_states(self, observe):
        # per-agent zero-padding to the max observation dim (heterogeneous
        # networks have different lane counts -> different obs lengths).
        dim = self.state_size - 1
        states = np.zeros((self.num, dim), dtype=np.float32)
        for i, ts in enumerate(self.id):
            o = np.asarray(observe[ts], dtype=np.float32)
            states[i, :min(dim, o.shape[0])] = o[:dim]
        timestep = np.zeros([states.shape[0], 1]) + self.current_step / self.horizon_len
        return np.concatenate([states, timestep], axis=-1)

    def set_utility(self, util_type='linear', delta=None, goal=None, beta=1.0):
        """Deployment utility over the outcome distribution (gmm only).
        'linear': E[w.Z] (default, identical to psi.w); 'box': threshold/safety
        E[w.Z * 1{Z<=delta}]; 'gauss': precision E[exp(-||Z-goal||^2/beta)]."""
        self.util = {'type': util_type, 'delta': delta, 'goal': goal,
                     'beta': float(beta)}

    def _utilities(self, states_t):
        """Per-agent action values used for control: closed-form transfer
        (usfa) or composed utilities (composer). states_t: [1,N,s].

        With sf_net='gmm' and a non-linear deployment utility, the usfa/
        independent rule scores each action by the closed-form E[f(Z(s,a))]
        instead of the mean-only w.psi -- this is where the distributional
        information changes decisions. (The composer stays on the mixture mean;
        its non-linear variant is not implemented.)"""
        w_t = self._w_tensor()
        if self.mode == 'composer':
            u = self.composer(states_t, w_t)               # [1,N,A]
        elif self.is_gmm and self.util['type'] != 'linear':
            log_pi, mu, sigma = self.sf_net.outcome_dist(states_t)
            if self.util['type'] == 'box':
                u = gmm_util_box(log_pi, mu, sigma, self.w, self.util['delta'])
            elif self.util['type'] == 'box_mean':
                m = (torch.softmax(log_pi, dim=-1).unsqueeze(-1) * mu).sum(-2)   # [B,N,A,K]
                w_t = self._w_tensor()
                lin = ((m * w_t.unsqueeze(-2)).sum(-1) if w_t.dim() == 3
                       else (m * w_t).sum(-1))
                d = torch.as_tensor(self.util['delta'], dtype=m.dtype, device=m.device)
                u = lin * (m <= d).all(-1).to(m.dtype) + 1e-6 * lin
            elif self.util['type'] == 'gauss':
                u = gmm_util_gauss(log_pi, mu, sigma, self.util['goal'],
                                   self.util['beta'])
            else:
                raise ValueError(f"unknown utility {self.util['type']}")
        else:
            u = self.sf_net.q_values(states_t, w_t)        # [1,N,A]
        return u.masked_fill(~self.action_mask, float('-inf'))

    def act(self, observe, exploration_rate=0, deterministic=False):
        states = self._build_states(observe)
        states_t = torch.tensor(states, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            u = self._utilities(states_t).squeeze(0)       # [N,A]
            greedy = u.argmax(dim=-1).cpu().numpy()
        eps = 0.0 if deterministic else float(exploration_rate)
        if eps > 0.0:
            explore = np.random.rand(self.num) < eps
            rand_actions = (np.random.rand(self.num)
                            * self._action_ns).astype(np.int64)
            actions = np.where(explore, rand_actions, greedy)
        else:
            actions = greedy
        self.last_states = states
        self.last_actions = actions.astype(np.int64)
        return {ts: int(a) for ts, a in zip(self.id, actions)}

    def _collect_phi2(self):
        """Read the per-agent NEXT-state feature vector phi from the reward fn's
        feature cache (populated during env.step by make_sf_reward)."""
        cache = getattr(self.reward_fn, 'feature_cache', {})
        return np.array([cache.get(ts, np.zeros(self.K, np.float32))
                         for ts in self.id], dtype=np.float32)  # [N,K]

    def step(self, action_dic):
        self.current_step += 1
        next_observe, rewards, dones_dict, _, info = self.env.step(action_dic)
        total_reward = sum(rewards.values())

        sysinfo = {}
        if isinstance(info, dict) and info:
            first = info.get(self.id[0])
            sysinfo = first if isinstance(first, dict) else info
        self.epoch_stats['queue_sum'] += sysinfo.get('system_total_stopped', 0.0)
        self.epoch_stats['wait_sum'] += sysinfo.get('system_mean_waiting_time', 0.0)
        self.epoch_stats['speed_sum'] += sysinfo.get('system_mean_speed', 0.0)
        self.epoch_stats['steps'] += 1

        done = (self.current_step >= self.horizon_len) or any(dones_dict.values())
        if self.is_train:
            next_states = self._build_states(next_observe)
            phi2 = self._collect_phi2()
            self.buffer.append(self.last_states, self.last_actions, phi2,
                               next_states, float(done))
        elif self.track_outcome:
            # realized discounted outcome z_i = sum_t gamma^t phi_{i,t+1} -- the
            # quantity the deployment utility f(z) is evaluated on.
            phi2 = self._collect_phi2()
            self.outcome_disc += (self.gamma ** (self.current_step - 1)) * phi2
        return next_observe, total_reward, done

    def get_stats(self):
        steps = self.epoch_stats['steps']
        if steps == 0:
            base = {'avg_queue': 0.0, 'avg_wait': 0.0, 'avg_speed': 0.0, 'avg_fairness': 0.0}
        else:
            base = {'avg_queue': self.epoch_stats['queue_sum'] / steps / self.num,
                    'avg_wait': self.epoch_stats['wait_sum'] / steps,
                    'avg_speed': self.epoch_stats['speed_sum'] / steps,
                    'avg_fairness': self.epoch_stats['fairness_sum'] / steps}
        base['travel_time'] = self.travel_time
        base['throughput'] = self.throughput
        return base

    def learn(self, batch_size=32, updates=1):
        if self.buffer.num < batch_size:
            return 0.0
        return (self._learn_composer(batch_size, updates) if self.mode == 'composer'
                else self._learn_usfa(batch_size, updates))

    def _learn_usfa(self, batch_size, updates):
        if self.is_gmm:
            return self._learn_usfa_gmm(batch_size, updates)
        """SF-TD on the w-free features: for the CURRENT episode's task w,
            psi(s,a).w  <-  phi2.w + gamma (1-done) psi_target(s2, a2*).w
        with a2* = argmax_a psi_target(s2,a).w. The vector SF regression target
        is phi2 + gamma psi_target(s2,a2*); we regress psi(s,a) to it per-feature
        (standard USFA), so the same psi serves every task."""
        losses = []
        w_t = self._w_tensor().squeeze(0)                  # [N,K]
        for _ in range(updates):
            S, A, PHI2, S2, D = self.buffer.sample(batch_size)
            S = torch.from_numpy(S).float().to(self.device)     # [B,N,s]
            A = torch.from_numpy(A).long().to(self.device)      # [B,N]
            PHI2 = torch.from_numpy(PHI2).float().to(self.device)  # [B,N,K]
            S2 = torch.from_numpy(S2).float().to(self.device)
            D = torch.from_numpy(D).float().to(self.device).view(-1, 1, 1)  # [B,1,1]

            psi = self.sf_net(S)                                 # [B,N,A,K]
            idx = A.view(*A.shape, 1, 1).expand(-1, -1, 1, self.K)
            psi_taken = psi.gather(2, idx).squeeze(2)            # [B,N,K]

            with torch.no_grad():
                psi2 = self.sf_target(S2)                        # [B,N,A,K]
                q2 = torch.einsum('bnak,nk->bna', psi2, w_t)     # [B,N,A]
                a2 = q2.masked_fill(~self.action_mask, float('-inf')).argmax(
                    dim=-1)                                          # [B,N]
                idx2 = a2.view(*a2.shape, 1, 1).expand(-1, -1, 1, self.K)
                psi2_sel = psi2.gather(2, idx2).squeeze(2)       # [B,N,K]
                target = PHI2 + self.gamma * (1.0 - D) * psi2_sel  # [B,N,K]

            loss = torch.mean((psi_taken - target) ** 2)
            losses.append(loss.item())
            self.optim.zero_grad()
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.sf_net.parameters(), self.grad_clip)
            self.optim.step()
            self._learn_steps += 1
            if self._learn_steps % self.target_update_freq == 0:
                self.sf_target.load_state_dict(self.sf_net.state_dict())
        return float(np.mean(losses)) if losses else 0.0

    def _learn_usfa_gmm(self, batch_size, updates):
        """Distributional TD for the GMM outcome net (replaces SF-TD when
        sf_net='gmm').

        The Bellman target is a CLOSED-FORM mixture: Z'(s2,a2) ~ GMM(pi',mu',sig')
        implies Z(s,a) ~ GMM(pi', phi2 + keep*mu', keep*sig') with
        keep = gamma*(1-done) -- no model fitting for the target. We draw
        ``nll_samples`` points from this exact target mixture and maximize the
        log-likelihood of the online taken-action GMM (projected distributional
        update). A mean-anchor MSE on the mixture mean (the ORIGINAL SF-TD
        vector target) keeps the linear path calibrated and stabilizes early
        training. a2 is chosen by the target mixture's linear utility
        E[w.Z'] -- the same greedy rule the original code applies to psi.
        """
        losses = []
        w_t = self._w_tensor().squeeze(0)                  # [N,K]
        C = self.sf_net.n_comp
        M = self.nll_samples
        LOG_SQRT_2PI = 0.5 * float(np.log(2 * np.pi))
        for _ in range(updates):
            S, A, PHI2, S2, D = self.buffer.sample(batch_size)
            S = torch.from_numpy(S).float().to(self.device)
            A = torch.from_numpy(A).long().to(self.device)     # [B,N]
            PHI2 = torch.from_numpy(PHI2).float().to(self.device)  # [B,N,K]
            S2 = torch.from_numpy(S2).float().to(self.device)
            D = torch.from_numpy(D).float().to(self.device).view(-1, 1, 1)  # [B,1,1]
            B, N = A.shape

            # online GMM at the TAKEN action
            log_pi, mu, sig = self.sf_net.outcome_dist(S)  # [B,N,A,C(,K)]
            lp_t = log_pi.gather(2, A.view(B, N, 1, 1).expand(-1, -1, 1, C)).squeeze(2)
            idx = A.view(B, N, 1, 1, 1).expand(-1, -1, 1, C, self.K)
            mu_t = mu.gather(2, idx).squeeze(2)            # [B,N,C,K]
            sig_t = sig.gather(2, idx).squeeze(2)          # [B,N,C,K]

            with torch.no_grad():
                lp2, mu2, sig2 = self.sf_target.outcome_dist(S2)
                pi2 = torch.softmax(lp2, dim=-1)           # [B,N,A,C]
                mu2_mix = (pi2.unsqueeze(-1) * mu2).sum(3)  # [B,N,A,K] = psi_eff
                q2 = torch.einsum('bnak,nk->bna', mu2_mix, w_t)
                a2 = q2.masked_fill(~self.action_mask, float('-inf')).argmax(
                    dim=-1)                                   # [B,N]
                lp2_t = lp2.gather(2, a2.view(B, N, 1, 1).expand(-1, -1, 1, C)).squeeze(2)
                idx2 = a2.view(B, N, 1, 1, 1).expand(-1, -1, 1, C, self.K)
                mu2_t = mu2.gather(2, idx2).squeeze(2)     # [B,N,C,K]
                sig2_t = sig2.gather(2, idx2).squeeze(2)   # [B,N,C,K]
                keep = self.gamma * (1.0 - D)              # [B,1,1]
                # exact target mixture components
                mean_t = PHI2.unsqueeze(2) + keep.unsqueeze(-1) * mu2_t   # [B,N,C,K]
                std_t = (keep.unsqueeze(-1) * sig2_t).clamp(min=self.sf_net.sigma_min)
                pi_t = torch.softmax(lp2_t, dim=-1)        # [B,N,C]
                # sample target outcomes z ~ GMM(pi_t, mean_t, std_t)
                comp = torch.multinomial(pi_t.reshape(-1, C), M,
                                         replacement=True).view(B, N, M)
                gidx = comp.unsqueeze(-1).expand(-1, -1, -1, self.K)
                zm = (mean_t.gather(2, gidx)
                      + std_t.gather(2, gidx)
                      * torch.randn(B, N, M, self.K, device=self.device))   # [B,N,M,K]
                # mean-anchor target = original SF-TD vector target on the mean
                mean_anchor = PHI2 + keep * (pi_t.unsqueeze(-1) * mu2_t).sum(2)  # [B,N,K]

            # NLL of target samples under the online taken-action GMM
            z_e = zm.unsqueeze(3)                          # [B,N,M,1,K]
            ll_c = (-0.5 * ((z_e - mu_t.unsqueeze(2)) / sig_t.unsqueeze(2)) ** 2
                    - sig_t.unsqueeze(2).log() - LOG_SQRT_2PI).sum(-1)        # [B,N,M,C]
            logw = torch.log_softmax(lp_t, dim=-1).unsqueeze(2)              # [B,N,1,C]
            nll = -torch.logsumexp(logw + ll_c, dim=-1).mean()               # scalar

            pi_taken = torch.softmax(lp_t, dim=-1)
            mu_mix = (pi_taken.unsqueeze(-1) * mu_t).sum(2)                  # [B,N,K]
            l_mean = torch.mean((mu_mix - mean_anchor) ** 2)
            ent = -(pi_taken * torch.log_softmax(lp_t, dim=-1)).sum(-1).mean()

            loss = (self.lambda_nll * nll + self.lambda_mean * l_mean
                    - self.ent_reg * ent)
            losses.append(loss.item())
            self.last_losses = {'nll': float(nll.item()),
                                'mean': float(l_mean.item()),
                                'ent': float(ent.item())}
            self.optim.zero_grad()
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.sf_net.parameters(), self.grad_clip)
            self.optim.step()
            self._learn_steps += 1
            if self._learn_steps % self.target_update_freq == 0:
                self.sf_target.load_state_dict(self.sf_net.state_dict())
        return float(np.mean(losses)) if losses else 0.0

    def _learn_composer(self, batch_size, updates):
        """Per-agent honest TD on the composer utilities for the fixed task w:
            delta_i = r_i + gamma (1-done) max_a u_target(s2,a) - u(s,a_i)
        with r_i = phi2_i . w_i (reconstructed from the stored feature vector).
        Per-agent (not joint-IGM) targets: the joint max overestimates under
        transition coupling. The composer backbone psi is frozen; only the
        neighbour-limited residual learns. Target uses a hard-synced copy."""
        losses = []
        w_t = self._w_tensor()                             # [1,N,K]
        w_bn = w_t.squeeze(0)                               # [N,K]
        for _ in range(updates):
            S, A, PHI2, S2, D = self.buffer.sample(batch_size)
            S = torch.from_numpy(S).float().to(self.device)
            A = torch.from_numpy(A).long().to(self.device)
            PHI2 = torch.from_numpy(PHI2).float().to(self.device)  # [B,N,K]
            S2 = torch.from_numpy(S2).float().to(self.device)
            D = torch.from_numpy(D).float().to(self.device).view(-1, 1)  # [B,1]

            B = S.shape[0]
            w_batch = w_t.expand(B, -1, -1)                 # [B,N,K]
            u = self.composer(S, w_batch)                  # [B,N,A]
            u_taken = u.gather(-1, A.unsqueeze(-1)).squeeze(-1)  # [B,N]
            r = torch.einsum('bnk,nk->bn', PHI2, w_bn)      # [B,N] per-agent reward

            with torch.no_grad():
                u2 = self.composer_target(S2, w_batch)      # [B,N,A]
                u2_max = u2.max(dim=-1)[0]                  # [B,N]
                target = r + self.gamma * (1.0 - D) * u2_max

            if getattr(self, 'composer_norm', False):
                # per-agent objective-scale normalisation: TD error measured in
                # the agent's running target-std units (snapshot; the buffer is
                # updated AFTER the optimiser step, never inside the graph).
                sc = self.composer.agent_scale.detach().clone()
                loss = torch.mean(((u_taken - target) / sc.view(1, -1)) ** 2)
            else:
                loss = torch.mean((u_taken - target) ** 2)
            losses.append(loss.item())
            self.optim.zero_grad()
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self._resid_params, self.grad_clip)
            self.optim.step()
            if getattr(self, 'composer_norm', False):
                with torch.no_grad():
                    sd = target.std(dim=0).clamp(min=1e-4)          # [N]
                    buf = self.composer.agent_scale
                    if getattr(self, '_scale_init', False):
                        buf.mul_(0.99).add_(0.01 * sd)
                    else:
                        buf.copy_(sd)
                        self._scale_init = True
            self._learn_steps += 1
            if self._learn_steps % self.target_update_freq == 0:
                self.composer_target.load_state_dict(self.composer.state_dict())
        return float(np.mean(losses)) if losses else 0.0
