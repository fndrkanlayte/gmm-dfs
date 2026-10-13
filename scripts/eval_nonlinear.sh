#!/usr/bin/env bash
# Non-linear objectives on Manhattan with the frozen main backbone (no retraining).
#   T4 threshold: w = T0, cost outcomes (queue, wait, pressure) <= delta
#      box       closed-form GMM utility  E[w.Z 1{Z <= delta}]
#      box_mean  plug-in mean rule        w.mu 1{mu <= delta}
#   T5 precision: gauss utility E[exp(-||Z - g||^2 / beta)]
# delta / g are the 70th percentile / median outcomes of the point-estimate SF
# controller under T0 (experiments/calibrate_utils.py); beta = 5.
# Results: gmm_ev_s{seed}/transfer/transfer_homo_tfc_{T4cost,T4plug,T5gauss}_s{seed}_result.json
set -euo pipefail
source "$(dirname "$0")/env.sh"
CK=gmm_tfc_s0/usfa/sf_psi.pth
DELTA='0.1924,0.3132,0.2888,999,999'
GOAL='0.0899,0.0957,0.2116,2.8178,0.0070'
ev() { local tag=$1 s=$2; shift 2
    "$PY" -u train_sf.py --phase independent --sf_net gmm --sf_ckpt $CK --task=$T0 \
        --eval_episodes 1 --seed $s --run_suffix ev --out_tag ${tag}_s$s --meter_layout A "$@" \
        > logs/nonlinear_${tag}_s$s.log 2>&1 &
}
for s in 0 1 2; do
    ev tfc_T4cost  $s --utility box      --util_delta=$DELTA
    ev tfc_T4plug  $s --utility box_mean --util_delta=$DELTA
    ev tfc_T5gauss $s --utility gauss    --util_goal=$GOAL --util_beta 5
done
wait
"$PY" - <<'PY'
import json, numpy as np
for tag, key in [('T4cost', 'violation_rate'), ('T4plug', 'violation_rate'), ('T5gauss', 'goal_dev')]:
    rs = [json.load(open(f'gmm_ev_s{s}/transfer/transfer_homo_tfc_{tag}_s{s}_result.json')) for s in range(3)]
    v = [r['metrics'][key] for r in rs]
    tp = [r['metrics']['throughput'] for r in rs]
    print(f'{tag:8s} {key} {np.mean(v):.4f} +- {np.std(v):.4f} | throughput {np.mean(tp):.0f}')
PY
