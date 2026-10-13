"""FixedTime (pre-timed cyclic) baseline on the same SUMO evaluation
setup as train_sf.py.

Each traffic signal cycles through its green phases in order, holding each
phase for ``--phase_steps`` decision steps (default 1 = 10 s at
delta_time=10). All switching constraints (min/yellow) stay with sumo-rl,
identical to eval_maxpressure.py so the comparison is apples-to-apples.
"""
import os as _os, sys as _sys
_PKG = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _PKG)
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False,
                               allow_nan=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task_seed", type=int, default=11)
    parser.add_argument("--task", type=str, default=None,
                        help="explicit w: 'default' for DEFAULT_W or K comma-floats; "
                             "overrides --task_seed")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--delta", type=str, default=None,
                        help="box thresholds, K comma-floats (default: frozen S2.1)")
    parser.add_argument("--zone_layout", type=str, default=None,
                        help="city-zone task layout (zones.LAYOUTS): sets the "
                             "per-signal W and records zone metrics")
    parser.add_argument("--meter_layout", type=str, default=None,
                        help="record zone/all-vehicle metrics for this layout "
                             "without changing the task")
    parser.add_argument("--hetero", type=int, default=0,
                        help="per-intersection task family W (task_seed-driven)")
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--max_step", type=int, default=60)
    parser.add_argument("--scenario", type=str, default="manhattan_28x7")
    parser.add_argument("--demand_scale", type=float, default=None)
    parser.add_argument("--phase_steps", type=int, default=1,
                        help="decision steps each green phase is held for "
                             "(phase_steps * delta_time seconds of green)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.eval_episodes < 1 or args.max_step < 1 or args.phase_steps < 1:
        parser.error("Episode, step and phase-step counts must be positive")
    import train_sf
    from composite_reward import make_sf_reward, sample_task
    from utils import parse_tripinfo, set_seed

    started = time.perf_counter()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    set_seed(args.seed)
    if args.task:
        if args.task.strip().lower() == 'default':
            from composite_reward import DEFAULT_W
            w = DEFAULT_W.copy().astype(np.float64)
        else:
            w = np.array([float(x) for x in args.task.split(',')],
                         dtype=np.float64)
    else:
        w = sample_task(np.random.RandomState(args.task_seed))
    manifest = {
        "method": f"FixedTime (round-robin green phases, {args.phase_steps} step(s)/phase)",
        "argv": sys.argv, "cwd": str(Path.cwd()), "pid": os.getpid(),
        "seed": args.seed, "simulation_seed": args.seed + 10_000_000,
        "task_seed": args.task_seed, "hetero": bool(args.hetero), "zone_layout": args.zone_layout, "w": w.tolist(),
        "eval_episodes": args.eval_episodes, "max_step": args.max_step,
        "delta_time": 10, "scale": 2.0, "yellow_time": 2, "min_green": 5,
        "phase_steps": args.phase_steps,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    write_json(output / "manifest.json", manifest)
    records = []
    for episode in range(1, args.eval_episodes + 1):
        episode_start = time.perf_counter()
        trip_path = output / f"tripinfo_{episode}.xml"
        reward_fn = make_sf_reward(w)
        env = train_sf.build_env(reward_fn, args, str(trip_path))
        try:
            obs, _ = env.reset(seed=args.seed + 10_000_000)
            core = env.unwrapped.env
            ids = sorted(obs)
            meter = None
            if args.zone_layout:
                from zones import zone_assignment, ZoneMeter
                zone_of, w_eval = zone_assignment(args.zone_layout, ids)
                reward_fn.set_w_map({ts: w_eval[i] for i, ts in enumerate(ids)})
                scn = getattr(args, 'scenario', 'manhattan_28x7')
                meter = ZoneMeter(env.unwrapped.env, ids, zone_of,
                                  args.max_step * 10,
                                  f'{train_sf.DATA_DIR}/{scn}/{scn}.rou.xml')
            elif args.meter_layout:
                from zones import zone_assignment, ZoneMeter
                scn = getattr(args, 'scenario', 'manhattan_28x7')
                meter = ZoneMeter(env.unwrapped.env, ids,
                                  zone_assignment(args.meter_layout, ids)[0],
                                  args.max_step * 10,
                                  f'{train_sf.DATA_DIR}/{scn}/{scn}.rou.xml')
            if args.zone_layout:
                pass
            elif args.hetero:
                from composite_reward import sample_task_family
                _rng = np.random.RandomState(args.task_seed)
                w_eval = np.stack([sample_task_family(_rng)
                                   for _ in ids]).astype(np.float64)
                reward_fn.set_w_map(
                    {ts: w_eval[i] for i, ts in enumerate(ids)})
            else:
                w_eval = np.tile(w, (len(ids), 1))
            n_actions = {ts: int(env.action_space(ts).n) for ts in ids}
            if episode == 1:
                write_json(output / "phase_counts.json", n_actions)
                print(f"FixedTime: {len(ids)} signals, actions "
                      f"{min(n_actions.values())}-{max(n_actions.values())}",
                      flush=True)
            phase_idx = {ts: 0 for ts in ids}
            total = queue = wait = speed = 0.0
            z = np.zeros((len(ids), 5), dtype=np.float64)
            step = 0
            for step in range(1, args.max_step + 1):
                # advance to next phase every phase_steps decision steps
                if (step - 1) % args.phase_steps == 0:
                    for ts in ids:
                        phase_idx[ts] = (phase_idx[ts] + (1 if step > 1 else 0)) % n_actions[ts]
                actions = dict(phase_idx)
                obs, rewards, terminated, truncated, info = env.step(actions)
                if meter is not None:
                    meter.step(env.unwrapped.env.sumo)
                sysinfo = info[ids[0]] if isinstance(info.get(ids[0]), dict) else info
                total += sum(rewards.values())
                queue += sysinfo["system_total_stopped"]
                wait += sysinfo["system_mean_waiting_time"]
                speed += sysinfo["system_mean_speed"]
                # realized discounted outcome z_i = sum_t gamma^t phi_{i,t+1}
                # (same semantics as SFAgent.track_outcome in agent.py).
                fc = getattr(reward_fn, 'feature_cache', {})
                z += (args.gamma ** (step - 1)) * np.array(
                    [fc.get(ts, np.zeros(5)) for ts in ids])
                if step % 10 == 0:
                    print(f"Episode {episode} step {step}/{args.max_step} "
                          f"return={total:.4f}", flush=True)
                if any(terminated.values()) or (truncated and all(truncated.values())):
                    break
            if meter is not None:
                meter.finish(env.unwrapped.env.sumo)
        finally:
            env.close()
        trips = parse_tripinfo(str(trip_path))
        # Frozen non-linear task constants (T4 box delta, T5 gauss goal/beta).
        DELTA = (np.array([float(x) for x in args.delta.split(',')])
                 if args.delta else
                 np.array([0.1924, 0.3132, 0.2888, 2.7083, 999.0]))
        GOAL = np.array([0.0899, 0.0957, 0.2116, 2.8178, 0.0070])
        BETA = 5.0
        metrics = {"travel_time": trips["travel_time"], "throughput": trips["throughput"],
                   "avg_queue": queue / step / len(ids), "avg_wait": wait / step,
                   "avg_speed": speed / step,
                   "realized_utility": float(np.mean((z * w_eval).sum(-1))),
                   "violation_rate": float(1.0 - (z <= DELTA).all(axis=1).mean()),
                   "goal_dev": float(np.sqrt(((z - GOAL) ** 2).sum(-1)).mean()),
                   "utility_box": float(np.mean((z * w).sum(-1)
                                                * (z <= DELTA).all(axis=1))),
                   "utility_gauss": float(np.exp(-((z - GOAL) ** 2).sum(-1)
                                                 / BETA).mean())}
        if meter is not None:
            zm = meter.summary(str(trip_path))
            metrics.update({k: float(v) for k, v in zm.items()})
        if not np.isfinite([total, *metrics.values()]).all():
            raise ValueError(f"Non-finite metrics: {metrics}")
        row = {"episode": episode, "return": total, **metrics,
               "wall_seconds": time.perf_counter() - episode_start,
               "steps": step}
        records.append(row)
        globals().setdefault('_z_last', (z.tolist(), w_eval.tolist()))
        with (output / "episodes.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
    keys = ["travel_time", "throughput", "avg_queue", "avg_wait", "avg_speed",
            "realized_utility", "violation_rate", "goal_dev",
            "utility_box", "utility_gauss"]
    result = {"phase": "fixedtime", "hetero": False, "w": w.tolist(),
              "mean_return": float(np.mean([r["return"] for r in records])),
              "std_return": float(np.std([r["return"] for r in records])),
              "n": len(records), "returns": [r["return"] for r in records],
              "metrics": {k: float(np.mean([r[k] for r in records]))
                          for k in records[0] if k not in ("episode", "return", "steps")
                          and isinstance(records[0][k], (int, float))},
              "z_matrix": globals().get('_z_last', (None,))[0],
              "w_matrix": globals().get('_z_last', (None, None))[1],
              "ids": ids if 'ids' in dir() else None,
              "wall_seconds": time.perf_counter() - started}
    write_json(output / "fixedtime_homo_result.json", result)
    with (output / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
