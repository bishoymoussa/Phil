#!/bin/bash
#SBATCH --job-name=phil-zs
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/galoaa.b/phillm/logs/zeroshot_%j.out
#SBATCH --error=/scratch/galoaa.b/phillm/logs/zeroshot_%j.err

set -euo pipefail
source /scratch/galoaa.b/phillm/Phil/scripts/explorer/common.sh
export PHIL_VENV="${PHIL_VENV:-/scratch/galoaa.b/egoexo360/env}"
activate_phil_venv
cd "$PHIL_CODE"

MODEL="${MODEL:-Qwen/Qwen3.5-2B}"
SPLIT="${SPLIT:-test}"
BACKEND="${BACKEND:-hf}"
LIMIT="${LIMIT:-}"
TAG=$(echo "$MODEL" | tr '/.' '__')
OUT="$PHIL_SCRATCH/preds/zeroshot_${TAG}_${SPLIT}.jsonl"
DATA="$PHIL_SCRATCH/data/${SPLIT}.jsonl"

EXTRA=()
if [[ -n "$LIMIT" ]]; then EXTRA+=(--limit "$LIMIT"); fi

python -m phil.models.infer_vllm \
  --model "$MODEL" \
  --data "$DATA" \
  --out "$OUT" \
  --backend "$BACKEND" \
  --no-fewshot \
  "${EXTRA[@]}"

python -m phil.eval.eval --gold "$DATA" --pred "$OUT" --out "${OUT%.jsonl}.metrics.json"
echo "done $OUT"
