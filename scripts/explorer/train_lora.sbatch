#!/bin/bash
#SBATCH --job-name=phil-lora
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/galoaa.b/phillm/logs/lora_%j.out
#SBATCH --error=/scratch/galoaa.b/phillm/logs/lora_%j.err

set -euo pipefail
source /scratch/galoaa.b/phillm/Phil/scripts/explorer/common.sh
export PHIL_VENV="${PHIL_VENV:-/scratch/galoaa.b/egoexo360/env}"
activate_phil_venv
cd "$PHIL_CODE"

MODEL="${MODEL:-Qwen/Qwen3.5-2B}"
OUT="$PHIL_SCRATCH/checkpoints/qwen35-2b-lora"
DATA="$PHIL_SCRATCH/data"

python -m phil.models.train_lora \
  --model "$MODEL" \
  --train "$DATA/train.jsonl" \
  --dev "$DATA/dev.jsonl" \
  --out "$OUT" \
  --epochs 1.0 \
  --batch-size 2 \
  --grad-accum 8 \
  --lora-r 16 \
  --seed 42

echo "LoRA done: $OUT"
