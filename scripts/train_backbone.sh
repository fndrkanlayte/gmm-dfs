#!/usr/bin/env bash
# GMM-DSF backbone on Manhattan 28x7: five-stage training chain.
#   1 sf   point-estimate SF warm start (narrow task cone)     500 ep -> sf_s{S}
#   2 gmm  Gaussian-mixture head, warm-started from 1          500 ep -> gmm_s{S}
#   3 tfh  heterogeneous task family (mode 3)                  500 ep -> gmm_tfh_s{S}
#   4 tfz  + fairness / spatial-block tasks (mode 4)           300 ep -> gmm_tfz_s{S}
#   5 tfc  + cost-only tasks (mode 5) = MAIN MODEL             300 ep -> gmm_tfc_s{S}
# Optional ablation (ABL_DOWNSTREAM=1): tfd = tfc + downstream observation, 400 ep.
# Stages whose checkpoint exists are skipped. Usage: bash scripts/train_backbone.sh [seed]
set -euo pipefail
source "$(dirname "$0")/env.sh"
S=${1:-0}
L=logs/backbone_s$S
run() { local out=$1 log=$2; shift 2
    [ -f "$out" ] && { echo "skip $out"; return; }
    echo "train -> $out"
    "$PY" -u train_sf.py --phase usfa --seed "$S" --ckpt_every 50 "$@" > "$log" 2>&1
}
run sf_s$S/usfa/sf_psi.pth       ${L}_1_sf.log  --sf_net sf  --episodes 500
run gmm_s$S/usfa/sf_psi.pth      ${L}_2_gmm.log --sf_net gmm --episodes 500 \
    --sf_init sf_s$S/usfa/sf_psi.pth
run gmm_tfh_s$S/usfa/sf_psi.pth  ${L}_3_tfh.log --sf_net gmm --episodes 500 --eps_start 0.3 \
    --task_family 3 --sf_init gmm_s$S/usfa/sf_psi.pth --run_suffix tfh
run gmm_tfz_s$S/usfa/sf_psi.pth  ${L}_4_tfz.log --sf_net gmm --episodes 300 --eps_start 0.3 \
    --eps_decay_episodes 200 --task_family 4 --sf_init gmm_tfh_s$S/usfa/sf_psi.pth --run_suffix tfz
run gmm_tfc_s$S/usfa/sf_psi.pth  ${L}_5_tfc.log --sf_net gmm --episodes 300 --eps_start 0.3 \
    --eps_decay_episodes 200 --task_family 5 --sf_init gmm_tfz_s$S/usfa/sf_psi.pth --run_suffix tfc

if [ "${ABL_DOWNSTREAM:-0}" = 1 ]; then
    INIT=gmm_tfc_s$S/usfa/sf_psi_down_init.pth
    [ -f "$INIT" ] || "$PY" experiments/expand_ckpt_downstream.py gmm_tfc_s$S/usfa/sf_psi.pth "$INIT" --n_old 41 --n_add 12
    run gmm_tfd_s$S/usfa/sf_psi.pth ${L}_6_tfd.log --sf_net gmm --episodes 400 --eps_start 0.3 \
        --eps_decay_episodes 200 --task_family 5 --obs downstream --sf_init "$INIT" --run_suffix tfd
fi
echo "backbone seed $S done"
