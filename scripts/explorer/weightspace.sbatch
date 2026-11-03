#!/bin/bash
#SBATCH --job-name=phil-ws
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/galoaa.b/phillm/logs/weightspace_%j.out
#SBATCH --error=/scratch/galoaa.b/phillm/logs/weightspace_%j.err

set -euo pipefail
source /scratch/galoaa.b/phillm/Phil/scripts/explorer/common.sh
export PHIL_VENV="${PHIL_VENV:-/scratch/galoaa.b/egoexo360/env}"
activate_phil_venv
cd "$PHIL_CODE"

OUT="$PHIL_SCRATCH/weightspace"
python -m phil.weightspace.train_tiny_sims --out "$OUT/tiny_sims"
python -m phil.weightspace.ask_llm \
  --catalog "$OUT/tiny_sims/catalog.json" \
  --model "${MODEL:-Qwen/Qwen3.5-9B}" \
  --out "$OUT/read_qwen35_9b.jsonl"
