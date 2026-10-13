#!/usr/bin/env bash
# Hangzhou 4x4 (360 steps = 3600 s) / Atlanta 1x5 (120 steps = 1200 s), raw demand.
#   1. GMM-DSF chain, step-matched to Manhattan (episodes x 60/H)
#   2. baselines (IDQN, GAT, PressLight, MPLight, PLight T0/T1/T2, zone IDQN, PRLight)
#   3. speed-gated composer on each zone layout
# Usage: bash scripts/train_bench.sh hz|atl
set -euo pipefail
source "$(dirname "$0")/env.sh"
case ${1:-} in
    hz)  SCN=hangzhou_4x4_gudang_18041610_1h; H=360; LAYS="HZA3 HZB3"; CE=33;  CD=25 ;;
    atl) SCN=atlanta_1x5;                    H=120; LAYS="ATL3";      CE=100; CD=75 ;;
    *) echo "usage: $0 hz|atl" >&2; exit 1 ;;
esac
P=$1; B=baseline_runs
sc() { "$PY" -c "print(max(1, round($1 * 60 / $H)))"; }
E500=$(sc 500); E300=$(sc 300); D300=$(sc 300); D200=$(sc 200); CK=$(sc 50)
C="--scenario $SCN --max_step $H --seed 0 --ckpt_every $CK"
bg() { local log=$1; shift; "$PY" -u "$@" > logs/${P}_$log.log 2>&1 & }

# ---- 1. GMM-DSF chain (sequential) ----
R1=sf_bsf_${SCN}_s0/usfa; R2=gmm_bv1_${SCN}_s0/usfa; R3=gmm_btfh_${SCN}_s0/usfa
R4=gmm_btfz_${SCN}_s0/usfa; R5=gmm_btfc_${SCN}_s0/usfa
st() { local out=$1 log=$2; shift 2
    [ -f "$out/sf_psi.pth" ] && { echo "skip $out"; return; }
    echo "train -> $out"
    "$PY" -u train_sf.py --phase usfa $C "$@" > logs/${P}_$log.log 2>&1
}
st $R1 chain1_sf  --sf_net sf  --episodes $E500 --eps_decay_episodes $D300 --run_suffix bsf
st $R2 chain2_gmm --sf_net gmm --episodes $E500 --eps_decay_episodes $D300 --sf_init $R1/sf_psi.pth --run_suffix bv1
st $R3 chain3_tfh --sf_net gmm --episodes $E500 --eps_decay_episodes $D300 --eps_start 0.3 \
    --task_family 3 --sf_init $R2/sf_psi.pth --run_suffix btfh
st $R4 chain4_tfz --sf_net gmm --episodes $E300 --eps_decay_episodes $D200 --eps_start 0.3 \
    --task_family 4 --sf_init $R3/sf_psi.pth --run_suffix btfz
st $R5 chain5_tfc --sf_net gmm --episodes $E300 --eps_decay_episodes $D200 --eps_start 0.3 \
    --task_family 5 --sf_init $R4/sf_psi.pth --run_suffix btfc

# ---- 2. baselines (parallel) ----
S="--scenario $SCN --max_step $H"
bg presslight baselines/train.py $S --net mlp --obs presslight --reward abspressure
bg mplight    baselines/train.py $S --net mlp --obs mplight --reward abspressure --use_agent_emb 0
bg idqn_mlp   baselines/train.py $S --net mlp
bg idqn_gat   baselines/train.py $S --net gat
for L in $LAYS; do
    bg idqn_zone$L baselines/train.py $S --net mlp --zone_layout $L --run_tag zone${L}eq
done
for k in T0 T1 T2; do
    bg plight_$k baselines/prlight.py --mode plight $S --task=${!k} --episodes 500 --out $B/${P}_plight_$k
done
# ---- 3. gated composer on the frozen backbone ----
for L in $LAYS; do
    bg gate2$L train_sf.py --phase ma-usfa --sf_net gmm --sf_ckpt $R5/sf_psi.pth $S --zone_layout $L \
        --composer_gate 1 --eps_start 0.3 --eps_decay_episodes $CD --episodes $CE \
        --eval_episodes 1 --seed 0 --run_suffix bgate2$L
done
wait
for L in $LAYS; do
    bg prlight_$L baselines/prlight.py --mode prlight $S --zone_layout $L --episodes 200 \
        --pool $B/${P}_plight_T0,$B/${P}_plight_T1,$B/${P}_plight_T2 --init $B/${P}_plight_T0 \
        --out $B/${P}_prlight_${L}eq
done
wait
echo "$P done"
