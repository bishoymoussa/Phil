#!/bin/bash
#SBATCH --job-name=phil-theta
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/galoaa.b/phillm/logs/theta_%j.out
#SBATCH --error=/scratch/galoaa.b/phillm/logs/theta_%j.err

set -euo pipefail
source /scratch/galoaa.b/phillm/Phil/scripts/explorer/common.sh
export PHIL_VENV="${PHIL_VENV:-/scratch/galoaa.b/egoexo360/env}"
activate_phil_venv
cd "$PHIL_CODE"

MODEL="${MODEL:-Qwen/Qwen3.5-2B}"
ADAPTER="${ADAPTER:-$PHIL_SCRATCH/checkpoints/qwen35-2b-selective}"
DATA="$PHIL_SCRATCH/data/test.jsonl"
PRED="$PHIL_SCRATCH/preds"

# If selective adapter missing, fall back to lora adapter
if [[ ! -d "$ADAPTER" ]]; then
  ADAPTER="$PHIL_SCRATCH/checkpoints/qwen35-2b-lora"
fi

for theta in 0.05 0.1 0.2 0.35 0.5; do
  out="$PRED/selective_theta${theta}_test.jsonl"
  python -m phil.models.agent infer --model "$MODEL" --adapter "$ADAPTER" --data "$DATA" --out "$out" --theta "$theta"
  python -m phil.eval.eval --gold "$DATA" --pred "$out" --out "${out%.jsonl}.metrics.json"
done

# Random caller matched to mid theta call rate if available
python - <<'PY'
import json
from pathlib import Path
from phil.eval.random_caller import run_random_baseline
pred = Path("/scratch/galoaa.b/phillm/preds")
# pick theta=0.2 metrics for call rate if present
mpath = pred / "selective_theta0.2_test.metrics.json"
call_rate = 0.3
if mpath.exists():
    call_rate = float(json.loads(mpath.read_text())["call_rate"])
run_random_baseline(
    "/scratch/galoaa.b/phillm/data/test.jsonl",
    str(pred / "llm_only_test.jsonl"),
    str(pred / "always_call_test.jsonl"),
    call_rate,
    str(pred / "random_caller"),
)
print("random baseline call_rate", call_rate)
PY
