#!/usr/bin/env bash
# Run inside screen: screen -dmS rl5034-ppo bash scripts/run_experiments.sh
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
PPO_PYTHON="${PPO_PYTHON:-$(conda info --base)/envs/ppo/bin/python}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$TASK_ROOT/.matplotlib"
mkdir -p runs
exec >>runs/experiment-session.log 2>&1
date -u '+Batch start: %Y-%m-%d %H:%M:%S UTC'
"$PPO_PYTHON" -m hvac_ppo ablate --jobs "${1:-2}"
"$PPO_PYTHON" -m hvac_ppo baseline --cases configs/scenarios/test.json --output results/baselines
"$PPO_PYTHON" -m hvac_ppo summarize
date -u '+Batch complete: %Y-%m-%d %H:%M:%S UTC'
