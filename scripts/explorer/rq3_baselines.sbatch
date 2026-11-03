#!/bin/bash
#SBATCH --job-name=phil-rq3
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/galoaa.b/phillm/logs/rq3_%j.out
#SBATCH --error=/scratch/galoaa.b/phillm/logs/rq3_%j.err

set -euo pipefail
source /scratch/galoaa.b/phillm/Phil/scripts/explorer/common.sh
export PHIL_VENV="${PHIL_VENV:-/scratch/galoaa.b/egoexo360/env}"
activate_phil_venv
cd "$PHIL_CODE"

DATA="$PHIL_SCRATCH/data"
PRED="$PHIL_SCRATCH/preds"
MODEL="${MODEL:-Qwen/Qwen3.5-2B}"
ADAPTER="${ADAPTER:-$PHIL_SCRATCH/checkpoints/qwen35-2b-lora}"

# Oracle (CPU)
python -m phil.models.agent oracle --data "$DATA/test.jsonl" --out "$PRED/oracle_test.jsonl"
python -m phil.eval.eval --gold "$DATA/test.jsonl" --pred "$PRED/oracle_test.jsonl" --out "$PRED/oracle_test.metrics.json"

# LLM-only (reuse if present)
if [[ ! -f "$PRED/lora_test.jsonl" ]]; then
  python -m phil.models.infer_vllm --model "$MODEL" --adapter "$ADAPTER" --data "$DATA/test.jsonl" --out "$PRED/lora_test.jsonl" --backend hf --no-fewshot
fi
cp -f "$PRED/lora_test.jsonl" "$PRED/llm_only_test.jsonl"
python -m phil.eval.eval --gold "$DATA/test.jsonl" --pred "$PRED/llm_only_test.jsonl" --out "$PRED/llm_only_test.metrics.json"

# Always-call
python -m phil.models.agent infer --model "$MODEL" --adapter "$ADAPTER" --data "$DATA/test.jsonl" --out "$PRED/always_call_test.jsonl" --force-call
python -m phil.eval.eval --gold "$DATA/test.jsonl" --pred "$PRED/always_call_test.jsonl" --out "$PRED/always_call_test.metrics.json"

# Labels on dlab from LoRA errors
python -m phil.models.infer_vllm --model "$MODEL" --adapter "$ADAPTER" --data "$DATA/dlab.jsonl" --out "$PRED/lora_dlab.jsonl" --backend hf --no-fewshot
python -m phil.models.agent label-calls --gold "$DATA/dlab.jsonl" --pred "$PRED/lora_dlab.jsonl" --out "$DATA/labels_dlab.jsonl"
