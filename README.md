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

SUMO scenarios under `data/raw_data/` are redistributed from LibSignal.

## Layout

```
gmmdsf/
  train_sf.py          backbone pre-training + zero-shot/composer deployment
  agent.py             SFAgent: mixture TD learning, utilities, gated composer
  model.py             SFQNet (point SF) / GMMQNet (mixture) / NeighborComposer
  composite_reward.py  phi features, w samplers, task families, SF reward
  zones.py             zone layouts (A3/C3/S*/HZA3/...) and ZoneMeter metrics
  graph.py             1-hop road adjacency
  utils.py             seeds, replay buffer, tripinfo parsing
  obs_baselines.py     downstream-observation variant
data/raw_data/         SUMO networks
```

## Install

```bash
pip install -r requirements.txt
pip install "eclipse-sumo>=1.26"    # or a system SUMO install
export SUMO_HOME=$(python -c "import sumo, os; print(os.path.dirname(sumo.__file__))")
```

## Usage

Run from `gmmdsf/` (results are written to cwd-relative run directories).

```bash
# backbone pre-training (task weights re-sampled every episode)
python train_sf.py --phase usfa --sf_net gmm --episodes 500 --seed 0

# zero-shot deployment on a target task (homogeneous -task / zoned --zone_layout)
python train_sf.py --phase independent --sf_net gmm \
    --sf_ckpt gmm_s0/usfa/sf_psi.pth --task=-1,-0.3,-1.5,3,0 --eval_episodes 5

# composer adaptation: backbone frozen, bounded 1-hop residual, speed-gated
python train_sf.py --phase ma-usfa --sf_net gmm \
    --sf_ckpt gmm_s0/usfa/sf_psi.pth --zone_layout A3 --composer_gate 1 \
    --episodes 200 --eval_episodes 5

# non-linear utilities on the frozen backbone (no retraining)
python train_sf.py --phase independent --sf_net gmm \
    --sf_ckpt gmm_s0/usfa/sf_psi.pth --utility box --util_delta=0.19,0.31,0.29,999,999
```

`--task_family` selects the pre-training task distribution; `--utility` accepts
`linear` (default), `box`, `box_mean`, `gauss`; see `train_sf.py --help` and
`zones.py` for the available layouts.

## License

MIT (see LICENSE).
