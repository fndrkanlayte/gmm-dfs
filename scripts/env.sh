# Common environment for all pipelines. Source it, do not execute it.
#   PY         python interpreter (default: python)
#   SUMO_HOME  SUMO installation (required by sumo-rl)
: "${PY:=python}"
if [ -z "${SUMO_HOME:-}" ]; then
    echo "SUMO_HOME is not set (e.g. export SUMO_HOME=\$(python -c 'import sumo, os; print(os.path.dirname(sumo.__file__))'))" >&2
    exit 1
fi
export SUMO_HOME OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LIBSUMO_AS_TRACI=1
cd "$(dirname "${BASH_SOURCE[0]}")/../gmmdsf"
mkdir -p logs baseline_runs

T0='-1,-0.3,-1.5,3,0'
T1='-1,0,0,0,0'
T2='-0.3,-0.1,-0.5,6,0'
