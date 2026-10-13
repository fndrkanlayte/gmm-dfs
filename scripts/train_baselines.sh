#!/usr/bin/env bash
# Learned baselines on Manhattan 28x7 (all 500 ep unless noted; run in parallel).
#   IDQN (MLP) / CoLight-style IDQN (GAT) on T0, PressLight, MPLight,
#   task-specific IDQN (T1, T2) and zone-specialised IDQN (A3, B3, C3),
#   PLight on T0/T1/T2, then PRLight adapted from the PLight pool (200 ep).
# Usage: bash scripts/train_baselines.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
B=baseline_runs
bg() { local log=$1; shift; "$PY" -u "$@" > logs/$log.log 2>&1 & }

bg idqn_mlp_T0   baselines/train.py --net mlp
bg idqn_gat_T0   baselines/train.py --net gat
bg presslight    baselines/train.py --net mlp --obs presslight --reward abspressure
bg mplight       baselines/train.py --net mlp --obs mplight --reward abspressure --use_agent_emb 0
bg idqn_mlp_T1   baselines/train.py --net mlp --task=$T1 --run_tag t1
bg idqn_gat_T2   baselines/train.py --net gat --task=$T2 --run_tag t2
for L in A3 B3 C3; do
    bg idqn_zone$L baselines/train.py --net mlp --zone_layout $L --run_tag zone$L
done
for k in T0 T1 T2; do
    bg plight_$k baselines/prlight.py --mode plight --task=${!k} --episodes 500 --out $B/plight_$k
done
wait

# PRLight: init = PLight-T0; the source pool leaves the target task out.
P="--mode prlight --init $B/plight_T0 --episodes 200"
bg prlight_T1 baselines/prlight.py $P --pool $B/plight_T0,$B/plight_T2 --task=$T1 --out $B/prlight_T1
bg prlight_T2 baselines/prlight.py $P --pool $B/plight_T0,$B/plight_T1 --task=$T2 --out $B/prlight_T2
for L in A3 B3 C3; do
    bg prlight_$L baselines/prlight.py $P --pool $B/plight_T0,$B/plight_T1,$B/plight_T2 \
        --zone_layout $L --out $B/prlight_$L
done
wait
echo "baselines done"
