"""PLight / PRLight baselines (Li et al., ESWA 2026, 'Enhancing traffic signal
control through model-based reinforcement learning and policy reuse';
official code: github.com/RL-DLMU/PRLight-and-PLight), ported to our sumo-rl
Manhattan setting.

Kept from the official code (PRLight/EDQN.py, tran-*.py):
  * EDQN = Encoder (MLP 2x32 + one neighbour-attention layer) + Decoder
    (h, action) -> next observation + dueling Q head (128 hidden);
  * loss = TD MSE (target net, synced every 50 updates) + next-obs MSE;
  * PLight = model-based pretraining of policy + environment model on a source
    domain; PRLight = target agent initialised from source model 1, a pool of
    frozen source PLight agents, and per-intersection behaviour-policy reuse:
    every 5 steps each intersection samples the agent whose predicted next
    observation is closest (in the parameter-averaged encoder's feature space)
    to the real one (softmax over -distance); epsilon = 0.05.
Adapted: obs = sumo-rl default + time step (42-d), 196 intersections, 16
phases, 1-hop road adjacency with self loops; source/target domains are
OBJECTIVES (reward r_i = phi_i . w_i), not traffic scenarios; batch 32,
buffer 20000, lr 1e-3, gamma 0.99 as in the official code.

Usage:
  python prlight.py --mode plight  --task=... | --zone_layout A  --out DIR
  python prlight.py --mode prlight --pool D1,D2 --init D1 (--task|--zone_layout) --out DIR
  python prlight.py --mode eval    --model DIR (--task|--zone_layout|--meter_layout) --seed S
"""
import os as _os, sys as _sys
_PKG = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _PKG)
import argparse
import json
import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import train_sf
from composite_reward import make_sf_reward, DEFAULT_W
from graph import build_coupling_adjacency
from utils import parse_tripinfo, set_seed


# ------------------------------------------------------------------ model --
class Embedding(nn.Module):
    def __init__(self, din, h):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(din, h), nn.Linear(h, h)

    def forward(self, x):
        return F.relu(self.fc2(F.relu(self.fc1(x))))


class AttModel(nn.Module):
    """Neighbour attention (official AttModel), neighbours given by a 0/1 mask."""

    def __init__(self, din, h, dout):
        super().__init__()
        self.fcv, self.fck, self.fcq = nn.Linear(din, h), nn.Linear(din, h), nn.Linear(din, h)
        self.fcout = nn.Linear(h, dout)

    def forward(self, x, adj):                       # x [B,N,d], adj [N,N]
        q = F.relu(self.fcq(x))                      # [B,N,h]
        k = F.relu(self.fck(x))
        v = F.relu(self.fcv(x))
        att = torch.matmul(q, k.transpose(1, 2))     # [B,N,N]
        att = att.masked_fill(adj.unsqueeze(0) <= 0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        return F.relu(self.fcout(torch.matmul(att, v)))


class Encoder(nn.Module):
    def __init__(self, din, h):
        super().__init__()
        self.mlp, self.att_1 = Embedding(din, h), AttModel(h, h, h)

    def forward(self, obs, adj):
        return self.att_1(self.mlp(obs), adj)


class Decoder(nn.Module):
    def __init__(self, din, h):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(h + 1, h), nn.Linear(h, din)

    def forward(self, h, a):                         # a [B,N,1] (action index)
        return F.relu(self.fc2(F.relu(self.fc1(torch.cat([h, a], -1)))))


class VAnet(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.f1, self.fV, self.fA = nn.Linear(din, 128), nn.Linear(128, 1), nn.Linear(128, dout)

    def forward(self, x):
        x = F.relu(self.f1(x))
        A, V = self.fA(x), self.fV(x)
        return V + A - A.mean(-1, keepdim=True)


class EDQN(nn.Module):
    def __init__(self, din, h, n_act):
        super().__init__()
        self.encoder, self.decoder, self.q_net = Encoder(din, h), Decoder(din, h), VAnet(h, n_act)

    def forward(self, obs, a, adj):
        h = self.encoder(obs, adj)
        return self.q_net(h), self.decoder(h, a)


# -------------------------------------------------------------------- env --
class Ctx:
    def __init__(self, args, tripinfo=None):
        self.args = args
        self.rf = make_sf_reward(DEFAULT_W)
        self.env = train_sf.build_env(self.rf, args, tripinfo)
        self.trip = tripinfo
        self.ids = sorted(self.env.possible_agents)
        self.N = len(self.ids)
        # heterogeneous intersections: pad obs to the max dim, mask actions
        self.n_acts = np.array([int(self.env.action_space(ts).n) for ts in self.ids])
        self.n_act = int(self.n_acts.max())
        self.din = max(int(self.env.observation_space(ts).shape[0]) for ts in self.ids) + 1
        self.mask = torch.tensor(np.arange(self.n_act)[None, :] < self.n_acts[:, None])
        A, _ = build_coupling_adjacency(self.ids, env=self.env, method='auto')
        self.adj = torch.tensor(A, dtype=torch.float32)
        self.W = None
        if args.zone_layout:
            from zones import zone_assignment
            self.zone_of, self.W = zone_assignment(args.zone_layout, self.ids)
            self.rf.set_w_map({ts: self.W[i] for i, ts in enumerate(self.ids)})
        else:
            w = (np.array([float(x) for x in args.task.split(',')], dtype=np.float32)
                 if args.task else DEFAULT_W.astype(np.float32))
            self.rf.set_w(w)
            self.W = np.tile(w, (self.N, 1))

    def states(self, obs, t):
        s = np.zeros((self.N, self.din), dtype=np.float32)
        for i, ts in enumerate(self.ids):
            o = np.asarray(obs[ts], dtype=np.float32)
            s[i, :min(len(o), self.din - 1)] = o[:self.din - 1]
        s[:, -1] = t / self.args.max_step
        return s

    def reset(self, seed):
        obs, _ = self.env.reset(seed=seed)
        return self.states(obs, 0)

    def step(self, actions, t):
        obs, rew, term, trunc, info = self.env.step({ts: int(a) for ts, a in zip(self.ids, actions)})
        r = np.array([rew[ts] for ts in self.ids], dtype=np.float32)
        done = any(term.values()) or all(trunc.values())
        first = info.get(self.ids[0]) if isinstance(info, dict) else None
        self.sysinfo = first if isinstance(first, dict) else info
        return self.states(obs, t), r, done


# ------------------------------------------------------------------ learn --
def td_update(agent, target, opt, buf, adj, gamma, batch=32, mask=None):
    b = random.sample(buf, batch)
    S = torch.tensor(np.stack([x[0] for x in b]))
    A = torch.tensor(np.stack([x[1] for x in b]), dtype=torch.long)
    R = torch.tensor(np.stack([x[2] for x in b]))
    S2 = torch.tensor(np.stack([x[3] for x in b]))
    D = torch.tensor([float(x[4]) for x in b]).view(-1, 1)
    a_in = A.float().unsqueeze(-1)
    q, pred = agent(S, a_in, adj)
    with torch.no_grad():
        q2, _ = target(S2, a_in, adj)
        if mask is not None:
            q2 = q2.masked_fill(~mask.unsqueeze(0), float('-inf'))
        y = R + gamma * (1 - D) * q2.max(-1)[0]
    loss = F.mse_loss(q.gather(-1, A.unsqueeze(-1)).squeeze(-1), y) + (pred - S2).pow(2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    return float(loss.item())


def greedy(model, s, adj, mask=None):
    with torch.no_grad():
        q, _ = model(torch.tensor(s).unsqueeze(0), torch.zeros(1, s.shape[0], 1), adj)
        q = q[0]
        if mask is not None:
            q = q.masked_fill(~mask, float('-inf'))
    return q.argmax(-1).numpy()


def save(model, d):
    os.makedirs(d, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(d, 'edqn.pt'))


def load(d, din, n_act, h=32):
    m = EDQN(din, h, n_act)
    m.load_state_dict(torch.load(os.path.join(d, 'edqn.pt'), map_location='cpu'))
    return m


def train(args):
    ctx = Ctx(args)
    agent = load(args.init, ctx.din, ctx.n_act) if args.init else EDQN(ctx.din, 32, ctx.n_act)
    target = EDQN(ctx.din, 32, ctx.n_act)
    target.load_state_dict(agent.state_dict())
    opt = torch.optim.Adam(agent.parameters(), lr=1e-3)
    buf = deque(maxlen=20000)
    pool, avg_enc = [], None
    if args.mode == 'prlight':
        pool = [load(d, ctx.din, ctx.n_act).eval() for d in args.pool.split(',')]
        avg_enc = Encoder(ctx.din, 32)
        sd = [p.encoder.state_dict() for p in pool]
        avg_enc.load_state_dict({k: sum(x[k] for x in sd) / len(sd) for k in sd[0]})
        avg_enc.eval()
    n_upd, log = 0, open(os.path.join(args.out, 'train.txt'), 'a')
    for ep in range(1, args.episodes + 1):
        if args.mode == 'plight':
            eps = max(0.05, 1.0 - (ep - 1) / 300 * 0.95)
        else:
            eps = 0.05
        s = ctx.reset(args.seed + ep)
        exp_id = np.random.randint(0, len(pool) + 1, ctx.N)
        acc = np.zeros((ctx.N, len(pool) + 1))
        tot, use = 0.0, np.zeros(len(pool) + 1)
        for t in range(1, args.max_step + 1):
            if pool:
                models = pool + [agent]
                acts = np.stack([greedy(m, s, ctx.adj, ctx.mask) for m in models])   # [M,N]
                a = acts[exp_id, np.arange(ctx.N)]
            else:
                a = greedy(agent, s, ctx.adj, ctx.mask)
            rnd = np.random.rand(ctx.N) < eps
            a = np.where(rnd, (np.random.rand(ctx.N) * ctx.n_acts).astype(int), a)
            s2, r, done = ctx.step(a, t)
            buf.append((s, a, r, s2, done))
            tot += r.sum()
            if pool:
                with torch.no_grad():
                    S = torch.tensor(s).unsqueeze(0)
                    a_in = torch.tensor(a, dtype=torch.float32).view(1, -1, 1)
                    real = avg_enc(torch.tensor(s2).unsqueeze(0), ctx.adj)[0]
                    dis = []
                    for m in pool + [agent]:
                        _, pre = m(S, a_in, ctx.adj)
                        dis.append((avg_enc(pre, ctx.adj)[0] - real).norm(dim=-1).numpy())
                dis = np.stack(dis, 1)                                     # [N,M]
                acc += pow(0.8, 5 - t) * dis + dis
                if t % 5 == 0:
                    p = torch.softmax(torch.tensor(-acc), dim=1).numpy()
                    exp_id = np.array([np.random.choice(p.shape[1], p=pi / pi.sum()) for pi in p])
                    np.add.at(use, exp_id, 1)
                    acc[:] = 0
            if len(buf) >= 1000:
                td_update(agent, target, opt, buf, ctx.adj, 0.99, mask=ctx.mask)
                n_upd += 1
                if n_upd % 50 == 0:
                    target.load_state_dict(agent.state_dict())
            s = s2
            if done:
                break
        log.write(f'Epoch {ep} | Reward {tot:.2f} | Eps {eps:.3f} | use {use.tolist()}\n')
        log.flush()
        print(f'Epoch {ep} | Reward {tot:.2f} | Eps {eps:.3f} | use {use.tolist()}', flush=True)
        if ep % 50 == 0:
            save(agent, os.path.join(args.out, f'ep{ep}'))
    save(agent, args.out)
    ctx.env.close()


def evaluate(args):
    os.makedirs(args.out, exist_ok=True)
    trip = os.path.join(args.out, f'tripinfo_s{args.seed}.xml')
    ctx = Ctx(args, trip)
    model = load(args.model, ctx.din, ctx.n_act).eval()
    s = ctx.reset(args.seed + 10_000_000)
    meter = None
    lay = args.zone_layout or args.meter_layout
    if lay:
        from zones import zone_assignment, ZoneMeter
        scn = getattr(args, 'scenario', 'manhattan_28x7')
        meter = ZoneMeter(ctx.env.unwrapped.env, ctx.ids, zone_assignment(lay, ctx.ids)[0],
                          args.max_step * 10, f'{train_sf.DATA_DIR}/{scn}/{scn}.rou.xml')
    tot, q, w, v, z = 0.0, 0.0, 0.0, 0.0, np.zeros((ctx.N, 5))
    for t in range(1, args.max_step + 1):
        s, r, done = ctx.step(greedy(model, s, ctx.adj, ctx.mask), t)
        tot += r.sum()
        sumo = ctx.env.unwrapped.env.sumo
        if meter is not None:
            meter.step(sumo)
        # same aggregation as SFAgent / Agent / eval_* (sumo-rl system info)
        q += ctx.sysinfo.get('system_total_stopped', 0.0)
        w += ctx.sysinfo.get('system_mean_waiting_time', 0.0)
        v += ctx.sysinfo.get('system_mean_speed', 0.0)
        fc = ctx.rf.feature_cache
        z += (0.99 ** (t - 1)) * np.array([fc.get(ts, np.zeros(5)) for ts in ctx.ids])
        if done:
            break
    if meter is not None:
        meter.finish(ctx.env.unwrapped.env.sumo)
    ctx.env.close()
    tm = parse_tripinfo(trip)
    out = {'mean_return': float(tot), 'metrics': {
        'travel_time': tm['travel_time'], 'throughput': tm['throughput'],
        'avg_queue': float(q / t / ctx.N), 'avg_wait': float(w / t), 'avg_speed': float(v / t),
        'realized_utility': float(np.mean((z * ctx.W).sum(-1)))},
        'zone_metrics': meter.summary(trip) if meter else None,
        'z_matrix': z.tolist(), 'w_matrix': np.asarray(ctx.W).tolist(), 'agent_ids': ctx.ids,
        'model': args.model, 'seed': args.seed}
    with open(os.path.join(args.out, f'eval_s{args.seed}.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in ('mean_return', 'metrics')}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['plight', 'prlight', 'eval'], required=True)
    p.add_argument('--task', type=str, default=None)
    p.add_argument('--zone_layout', type=str, default=None)
    p.add_argument('--meter_layout', type=str, default=None)
    p.add_argument('--pool', type=str, default=None)
    p.add_argument('--init', type=str, default=None)
    p.add_argument('--model', type=str, default=None)
    p.add_argument('--episodes', type=int, default=500)
    p.add_argument('--max_step', type=int, default=60)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', type=str, required=True)
    p.add_argument('--scenario', type=str, default='manhattan_28x7')
    p.add_argument('--demand_scale', type=float, default=None)
    args = p.parse_args()
    set_seed(args.seed)
    torch.set_num_threads(1)
    if args.mode == 'eval':
        evaluate(args)
    else:
        os.makedirs(args.out, exist_ok=True)
        train(args)


if __name__ == '__main__':
    main()
