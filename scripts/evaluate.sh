#!/usr/bin/env bash
# Evaluate every trained model (3 evaluation seeds), then print tables and draw figures.
# Methods whose checkpoint is missing are skipped. Usage: bash scripts/evaluate.sh [workers]
set -euo pipefail
source "$(dirname "$0")/env.sh"
W=${1:-6}
"$PY" experiments/run_table1_v2.py --parts A,B --workers $W     # -> table1_v2.csv
"$PY" experiments/run_bench.py --workers $W                     # -> bench_results.csv
"$PY" experiments/run_behaviour.py                              # -> behaviour_results.jsonl
bash ../scripts/eval_nonlinear.sh

"$PY" experiments/summarize_table1_v2.py                        # Table 1 (T0/T1/T2)
for L in A3 C3; do "$PY" experiments/summarize_table2.py $L; done   # Table 2
"$PY" experiments/summarize_table3.py                           # Table 3 (Hangzhou / Atlanta)
"$PY" experiments/summarize_ablation.py                         # ablation table
mkdir -p figures
"$PY" experiments/make_fig_envs.py C3
"$PY" experiments/make_fig3_v2.py
"$PY" experiments/make_fig3_fig4_fig7.py
"$PY" experiments/make_fig5_fig6.py
