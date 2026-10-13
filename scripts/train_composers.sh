#!/usr/bin/env bash
# Neighbour composer on the frozen main backbone (Manhattan).
#   main:      speed-gated composer, 200 ep, eps 0.3 -> 0.05 over 150 ep (tfcg2*)
#   ABL=1 also trains the ablations:
#     400 episodes (tfcg4*), ungated (tfcu2*), gated + scale-normalised (tfcn2*),
#     ungated with the default schedule eps 1.0 -> 0.05 over 300 ep (tfccomp*).
# On T1 no intersection has a speed term, so the gated composer equals zero-shot
# deployment and is not trained. Usage: bash scripts/train_composers.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
CK=gmm_tfc_s0/usfa/sf_psi.pth
[ -f $CK ] || { echo "missing $CK: run scripts/train_backbone.sh first" >&2; exit 1; }
comp() { local tag=$1; shift
    [ -f gmm_${tag}_s0/composer/composer.pth ] && { echo "skip $tag"; return; }
    "$PY" -u train_sf.py --phase ma-usfa --sf_net gmm --sf_ckpt $CK \
        --eval_episodes 1 --seed 0 --run_suffix "$tag" "$@" > logs/composer_${tag}.log 2>&1 &
}
NEW="--eps_start 0.3 --eps_decay_episodes 150"
for t in T0 T2 A3 C3; do
    case $t in T0) TA="--task=$T0";; T2) TA="--task=$T2";; *) TA="--zone_layout $t";; esac
    comp tfcg2$t --composer_gate 1 $NEW --episodes 200 $TA
    if [ "${ABL:-0}" = 1 ]; then
        comp tfcg4$t --composer_gate 1 $NEW --episodes 400 $TA
        comp tfcn2$t --composer_gate 1 --composer_norm 1 $NEW --episodes 200 $TA
    fi
done
if [ "${ABL:-0}" = 1 ]; then
    comp tfcu2T1 --composer_gate 0 $NEW --episodes 200 --task=$T1
    comp tfcu2A3 --composer_gate 0 $NEW --episodes 200 --zone_layout A3
    comp tfcu2C3 --composer_gate 0 $NEW --episodes 200 --zone_layout C3
    comp tfccompT0 --episodes 200 --task=$T0
    comp tfccompT1 --episodes 200 --task=$T1
    comp tfccompT2 --episodes 200 --task=$T2
    comp tfccompA3 --episodes 200 --zone_layout A3
fi
wait
echo "composers done"
