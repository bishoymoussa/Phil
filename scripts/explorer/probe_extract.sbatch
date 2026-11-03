#!/bin/bash
#SBATCH --job-name=phil-probe
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/galoaa.b/phillm/logs/probe_%j.out
#SBATCH --error=/scratch/galoaa.b/phillm/logs/probe_%j.err

set -euo pipefail
source /scratch/galoaa.b/phillm/Phil/scripts/explorer/common.sh
export PHIL_VENV="${PHIL_VENV:-/scratch/galoaa.b/egoexo360/env}"
activate_phil_venv
cd "$PHIL_CODE"

MODEL="${MODEL:-Qwen/Qwen3.5-2B}"
ADAPTER="${ADAPTER:-}"
TAG="${TAG:-base}"
OUT="$PHIL_SCRATCH/probes/${TAG}"
EXTRA=()
if [[ -n "$ADAPTER" ]]; then EXTRA+=(--adapter "$ADAPTER"); fi

python -m phil.probing.extract --model "$MODEL" --data "$PHIL_SCRATCH/data/dev.jsonl" --out "$OUT" --limit 500 "${EXTRA[@]}"
python -m phil.probing.probes --hidden "$OUT/hidden_last_prompt.npz" --targets "$OUT/targets.npz" --out "$OUT/probe_results.json"
python -m phil.probing.controls --hidden "$OUT/hidden_last_prompt.npz" --targets "$OUT/targets.npz" --out "$OUT/controls.json"
