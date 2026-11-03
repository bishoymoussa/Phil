#!/bin/bash
#SBATCH --job-name=phil-inf-lora
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/galoaa.b/phillm/logs/infer_lora_%j.out
#SBATCH --error=/scratch/galoaa.b/phillm/logs/infer_lora_%j.err

set -euo pipefail
source /scratch/galoaa.b/phillm/Phil/scripts/explorer/common.sh
export PHIL_VENV="${PHIL_VENV:-/scratch/galoaa.b/egoexo360/env}"
activate_phil_venv
cd "$PHIL_CODE"

MODEL="${MODEL:-Qwen/Qwen3.5-2B}"
ADAPTER="${ADAPTER:-$PHIL_SCRATCH/checkpoints/qwen35-2b-lora}"
SPLIT="${SPLIT:-test}"
OUT="$PHIL_SCRATCH/preds/lora_${SPLIT}.jsonl"
DATA="$PHIL_SCRATCH/data/${SPLIT}.jsonl"

python -m phil.models.infer_vllm \
  --model "$MODEL" \
  --adapter "$ADAPTER" \
  --data "$DATA" \
  --out "$OUT" \
  --backend hf \
  --no-fewshot

python -m phil.eval.eval --gold "$DATA" --pred "$OUT" --out "${OUT%.jsonl}.metrics.json"
