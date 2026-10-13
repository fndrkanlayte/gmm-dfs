# GMM-DSF: Distributional Successor Features for Task-Conditioned Traffic Signal Control

Code for the experiments of the ESWA manuscript on **GMM-DSF**
(Gaussian-mixture distributional successor features for multi-intersection
traffic signal control).

The environment reward of a signal controller is already linear in five
normalized outcome features:

```
r = phi . w,   phi = [queue, wait, pressure, speed, fairness]_norm
```

so a *task* is simply a weight vector `w`. GMM-DSF learns, once, an
action-conditioned Gaussian-mixture distribution over the discounted
cumulative outcome `Z_i(s_i, a) = sum_t gamma^t phi_i(t+1)` at every
intersection. At deployment time a new task is executed **without retraining
the backbone**:

```
a_i = argmax_a  E[ U_i(Z_i(s_i, a)) ]
```

for homogeneous `w`, for spatially heterogeneous per-intersection objectives
`w_i` (city zones), and for *non-linear* utilities that mean-only successor
features cannot express (a threshold/safety rule and a precision/target rule,
both closed-form under the mixture). Optionally, a bounded one-hop neighbour
composer is fine-tuned on the target task with the backbone frozen
(`--composer_gate 1` activates the residual only where the local task has a
speed term), which needs ~200 episodes versus 500 for a full retrain.

## Provenance

This codebase started from the public **MA-USFA** repository (MIT license,
"Is Per-Agent Policy Composition Safe? ..."), which contributes the SF
backbone, the per-intersection `w` interface, and the neighbour-composer
mechanism. It also builds on the idea of the IJCAI paper **GDSF** (distributional
successor features for non-linear utilities), which GMM-DSF generalizes from
single-agent offline control to decentralized multi-intersection signal
control. What is new here relative to MA-USFA: the Gaussian-mixture outcome
model and its mixture-target TD training, the closed-form non-linear utilities
and plug-in-mean control rule, the heterogeneous/spatially-blocked task
families used to pre-train the backbone, the speed-gated and scale-normalized
composer variants, and the full task-conditioned evaluation protocol on three
networks. The MA-USFA toy experiments and the synchronized-anchor rule are not
part of this codebase. `baselines/prlight.py` is a port of the official
PLight/PRLight code (github.com/RL-DLMU/PRLight-and-PLight).

**Note on the `sf` / `mausfa` rows in the evaluation tables:** the
point-estimate SF baseline (`--sf_net sf`) is the MA-USFA backbone evaluated
with the independent zero-shot rule — the learned-composer baseline is
*itself* fine-tuned on the same backbone (`--phase ma-usfa`), not a separate
method.

## Layout

```
gmmdsf/
  train_sf.py          backbone pre-training + zero-shot/composer deployment
  agent.py             SFAgent: mixture TD learning, utilities, gated composer
  model.py             SFQNet (point SF) / GMMQNet (mixture) / NeighborComposer
  composite_reward.py  phi features, w samplers, task families, SF reward
  zones.py             zone layouts (A3/C3/S*/HZA3/...), ZoneMeter metrics
  graph.py             1-hop road adjacency
  utils.py             seeds, replay buffer, tripinfo parsing
  obs_baselines.py     PressLight/MPLight observation & reward functions
  baselines/           train.py (IDQN/CoLight-style/PressLight/MPLight),
                       prlight.py (PLight/PRLight), eval_fixedtime.py,
                       eval_maxpressure.py
  experiments/         evaluation batteries, table/figure scripts, calibration
  tests/               unit tests for the mixture head and MaxPressure
scripts/               end-to-end training/evaluation pipelines
data/raw_data/         SUMO networks (Manhattan 28x7, Hangzhou 4x4, Atlanta 1x5)
```

## Install

```bash
pip install -r requirements.txt
# SUMO is required by sumo-rl; either install eclipse-sumo>=1.26 (provides
# libsumo/traci/sumo-data) or a system SUMO build, then:
export SUMO_HOME=$(python -c "import sumo, os; print(os.path.dirname(sumo.__file__))")
```

All scripts assume they run **from `gmmdsf/`** (or through `scripts/*.sh`,
which cd there) because results are written into cwd-relative run directories.

## Quick check (~minutes, one scenario)

```bash
cd gmmdsf
python tests/test_gmm.py        # 9 unit tests: shapes, mixture TD target, utilities
python tests/test_maxpressure.py
# smoke-test the full pipeline on 2 episodes:
python train_sf.py --phase usfa --sf_net sf --episodes 2 --warmup 0 --seed 7
```

## Reproducing the paper

Every stage is idempotent (existing checkpoints are skipped), so the scripts
below can be re-run safely and interrupted at will.

```bash
# 1. Manhattan: five-stage backbone chain -> gmm_tfc_s0/usfa/sf_psi.pth (~days)
bash scripts/train_backbone.sh            # seeds: pass 1,2 for tfz ablation arms

# 2. Manhattan baselines (IDQN, CoLight-style, PressLight, MPLight,
#    task-specific IDQN, zone IDQN, PLight sources, PRLight)
bash scripts/train_baselines.sh

# 3. Composer on the frozen backbone; ABL=1 adds the ablation composers
bash scripts/train_composers.sh           # ABL=1 bash scripts/train_composers.sh

# 4. Hangzhou + Atlanta chains, baselines and composers
bash scripts/train_bench.sh hz
bash scripts/train_bench.sh atl

# 5. Evaluate everything (3 seeds) + tables + figures
bash scripts/evaluate.sh 6
```

Evaluation produces `gmmdsf/table1_v2.csv` / `bench_results.csv` /
`behaviour_results.jsonl` and regenerates `figures/`; the summarizers print
Tables 1-3 and the ablation table. To rerun only part of the battery:

```bash
python experiments/run_table1_v2.py --methods tfc,tfc_blind --parts B
python experiments/run_bench.py --benches hz
```

Non-linear tasks (T4 threshold, T5 target; distributional vs plug-in-mean
decision rules on the same backbone) are evaluated by
`scripts/eval_nonlinear.sh`. Their frozen parameters (delta at the 70th
percentile, goal at the median x1.2 speed, beta=5) can be recalibrated with
`experiments/calibrate_utils.py --sf_ckpt sf_s0/usfa/sf_psi.pth`.

## Method <-> CLI map

| method key (`--methods`) | command |
|---|---|
| `ft` / `mp` | `baselines/eval_fixedtime.py` / `eval_maxpressure.py` |
| `idqn_mlp_T0`, `idqn_gat_T0` | `baselines/train.py --net mlp\|gat` |
| `presslight` / `mplight` | `baselines/train.py --obs presslight\|mplight --reward abspressure` |
| `plight_*` / `prlight_*` | `baselines/prlight.py --mode plight\|prlight` |
| `sf` | `train_sf.py --phase independent --sf_net sf` (point-SF zero-shot) |
| `tfc` | `train_sf.py --phase independent --sf_net gmm` (GMM-DSF zero-shot) |
| `tfc_blind` | same + `--blind 1` (ignores w, ablation) |
| `tfh`/`tfz`/`tfd` | earlier training stages / downstream-obs ablation |
| `tfc_gate2*` | `--phase ma-usfa --composer_gate 1` (main composer) |
| `tfc_u2*` / `tfc_n2*` / `tfc_gate4*` / `tfc_comp*` | ablations: ungated / scale-normalized / 400 ep / old schedule |
| T4 / T5 evals | `--utility box` \| `box_mean` \| `gauss` |

## Tasks

* `T0 = -1,-0.3,-1.5,3,0` (standard), `T1 = -1,0,0,0,0` (queue-only),
  `T2 = -0.3,-0.1,-0.5,6,0` (speed).
* Zone layouts `A, B, A3, B3, C3` (Manhattan) and `HZA3, HZB3, ATL3` assign each
  intersection to a corridor / gateway / equity / base zone with its own
  objective (`zones.py`); `S0`..`S75` interpolate heterogeneity for Fig. 3.
* `T4` (box threshold on cumulative cost outcomes) and `T5` (Gaussian target)
  are evaluated with `--utility`.

## License

MIT (see LICENSE). SUMO scenario files under `data/raw_data/` are
redistributed for reproducibility; they originate from the LibSignal
benchmarks (github.com/DaRL-LibSignal/LibSignal).
