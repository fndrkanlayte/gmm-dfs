# GMM-DSF

Gaussian-mixture distributional successor features for task-conditioned
traffic signal control.

One learned model over the discounted cumulative outcomes
`Z_i(s_i, a) = sum_t gamma^t phi_i(t+1)` per intersection, where `phi` =
[queue, wait, pressure, speed, fairness] and `r = phi . w`. Deployed on new
tasks without retraining: `a_i = argmax_a E[U_i(Z_i(s_i, a))]`, for homogeneous
`w`, spatially heterogeneous zone objectives, and closed-form non-linear
utilities. A bounded one-hop neighbour composer gives cheap task adaptation
with the backbone frozen.

Derived from MA-USFA (MIT license): the SF backbone, per-agent `w` interface
and neighbour composer are inherited; the mixture outcome model, closed-form
non-linear utilities, task families and gated composer are new.
`baselines/prlight.py` is a port of PLight/PRLight
(github.com/RL-DLMU/PRLight-and-PLight). SUMO scenarios under `data/raw_data/`
are redistributed from LibSignal.

## Layout

```
gmmdsf/                 training, deployment, zones, models
  baselines/            IDQN / CoLight-style / PressLight / MPLight /
                        PLight / PRLight / FixedTime / MaxPressure
  experiments/          evaluation batteries, table and figure scripts
  tests/                unit tests
scripts/                end-to-end pipelines
data/raw_data/          SUMO networks
```

## Install

```bash
pip install -r requirements.txt
pip install "eclipse-sumo>=1.26"    # or a system SUMO install
export SUMO_HOME=$(python -c "import sumo, os; print(os.path.dirname(sumo.__file__))")
```

## Reproduce

Every stage skips existing checkpoints and can be interrupted/re-run.

```bash
bash scripts/train_backbone.sh      # 5-stage GMM-DSF chain (Manhattan)
bash scripts/train_baselines.sh     # learned baselines
bash scripts/train_composers.sh     # gated composer (ABL=1 adds ablations)
bash scripts/train_bench.sh hz      # Hangzhou
bash scripts/train_bench.sh atl     # Atlanta
bash scripts/evaluate.sh            # 3-seed evals -> tables + figures/
bash scripts/eval_nonlinear.sh      # T4/T5 non-linear tasks
```

Tasks: `T0/T1/T2` homogeneous weights; `A..C3` + `S*` Manhattan zone layouts;
`HZA3/HZB3/ATL3` benchmark layouts; `T4`/`T5` non-linear utilities
(`--utility box|box_mean|gauss`). Method keys are listed in
`experiments/run_table1_v2.py`.

## License

MIT (see LICENSE).
