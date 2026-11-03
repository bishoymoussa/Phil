#!/bin/bash
#SBATCH --job-name=phil-pg
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/galoaa.b/phillm/logs/probeguided_%j.out
#SBATCH --error=/scratch/galoaa.b/phillm/logs/probeguided_%j.err

set -euo pipefail
source /scratch/galoaa.b/phillm/Phil/scripts/explorer/common.sh
export PHIL_VENV="${PHIL_VENV:-/scratch/galoaa.b/egoexo360/env}"
activate_phil_venv
cd "$PHIL_CODE"

python -m phil.models.train_probeguided \
  --model "${MODEL:-Qwen/Qwen3.5-2B}" \
  --train "$PHIL_SCRATCH/data/train.jsonl" \
  --dev "$PHIL_SCRATCH/data/dev.jsonl" \
  --out "$PHIL_SCRATCH/checkpoints/qwen35-2b-probeguided" \
  --epochs 1 \
  --limit 5000 \
  --aux-weight 0.1
