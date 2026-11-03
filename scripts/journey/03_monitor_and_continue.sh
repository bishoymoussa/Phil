#!/usr/bin/env bash
# Poll Journey LoRA/zeroshot; when done, eval + push + kick next stages.
set -euo pipefail
ROOT=/home/bi.ga/Phil
WORK=/mnt/work/galoaa/phillm
PY=/home/bi.ga/.conda/envs/grpo-motion/bin/python
export PYTHONPATH=$ROOT
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
export HF_HOME=$WORK/cache/hf

wait_file() {
  local f="$1"
  while [[ ! -f "$f" ]]; do sleep 60; done
}

echo "[monitor] waiting for zeroshot preds..."
while pgrep -f "phil.models.infer_vllm.*zeroshot_Qwen3_5_2B_test" >/dev/null; do sleep 60; done
if [[ -f $WORK/preds/zeroshot_Qwen3_5_2B_test.jsonl ]]; then
  $PY -m phil.eval.eval --gold $WORK/data/test.jsonl --pred $WORK/preds/zeroshot_Qwen3_5_2B_test.jsonl --out $WORK/preds/zeroshot_Qwen3_5_2B_test.metrics.json
fi

echo "[monitor] waiting for LoRA checkpoint..."
while pgrep -f "phil.models.train_lora" >/dev/null; do sleep 60; done
wait_file $WORK/checkpoints/qwen35-2b-lora/adapter_config.json || true
ls -la $WORK/checkpoints/qwen35-2b-lora | head

echo "[monitor] LoRA eval on test + OOD..."
for split in test ood_mag ood_hor ood_obj ood_col; do
  out=$WORK/preds/lora_${split}.jsonl
  if [[ ! -f $out ]]; then
    CUDA_VISIBLE_DEVICES=1 $PY -m phil.models.infer_vllm \
      --model Qwen/Qwen3.5-2B --adapter $WORK/checkpoints/qwen35-2b-lora \
      --data $WORK/data/${split}.jsonl --out $out --backend hf --no-fewshot --limit ${LIMIT:-500}
  fi
  $PY -m phil.eval.eval --gold $WORK/data/${split}.jsonl --pred $out --out ${out%.jsonl}.metrics.json
done

echo "[monitor] probing base + lora..."
CUDA_VISIBLE_DEVICES=1 $PY -m phil.probing.extract --model Qwen/Qwen3.5-2B --data $WORK/data/dev.jsonl --out $WORK/probes/base --limit 300
$PY -m phil.probing.probes --hidden $WORK/probes/base/hidden_last_prompt.npz --targets $WORK/probes/base/targets.npz --out $WORK/probes/base/probe_results.json
$PY -m phil.probing.controls --hidden $WORK/probes/base/hidden_last_prompt.npz --targets $WORK/probes/base/targets.npz --out $WORK/probes/base/controls.json

CUDA_VISIBLE_DEVICES=1 $PY -m phil.probing.extract --model Qwen/Qwen3.5-2B --adapter $WORK/checkpoints/qwen35-2b-lora --data $WORK/data/dev.jsonl --out $WORK/probes/lora --limit 300
$PY -m phil.probing.probes --hidden $WORK/probes/lora/hidden_last_prompt.npz --targets $WORK/probes/lora/targets.npz --out $WORK/probes/lora/probe_results.json
$PY -m phil.probing.controls --hidden $WORK/probes/lora/hidden_last_prompt.npz --targets $WORK/probes/lora/targets.npz --out $WORK/probes/lora/controls.json

echo "[monitor] probe-guided train..."
CUDA_VISIBLE_DEVICES=1 $PY -m phil.models.train_probeguided \
  --model Qwen/Qwen3.5-2B --train $WORK/data/train.jsonl --dev $WORK/data/dev.jsonl \
  --out $WORK/checkpoints/qwen35-2b-probeguided --epochs 1 --limit 3000

echo "[monitor] rq3 oracle already done; label calls + always-call sample..."
CUDA_VISIBLE_DEVICES=2 $PY -m phil.models.infer_vllm --model Qwen/Qwen3.5-2B --adapter $WORK/checkpoints/qwen35-2b-lora \
  --data $WORK/data/dlab.jsonl --out $WORK/preds/lora_dlab.jsonl --backend hf --no-fewshot --limit 1000
$PY -m phil.models.agent label-calls --gold $WORK/data/dlab.jsonl --pred $WORK/preds/lora_dlab.jsonl --out $WORK/data/labels_dlab.jsonl
CUDA_VISIBLE_DEVICES=2 $PY -m phil.models.agent infer --model Qwen/Qwen3.5-2B --adapter $WORK/checkpoints/qwen35-2b-lora \
  --data $WORK/data/test.jsonl --out $WORK/preds/always_call_test.jsonl --force-call --limit 200
$PY -m phil.eval.eval --gold $WORK/data/test.jsonl --pred $WORK/preds/always_call_test.jsonl --out $WORK/preds/always_call_test.metrics.json

echo "[monitor] weightspace read with 9B if VRAM allows else 2B..."
$PY -m phil.weightspace.train_tiny_sims --out $WORK/weightspace/tiny_sims
CUDA_VISIBLE_DEVICES=3 $PY -m phil.weightspace.ask_llm --catalog $WORK/weightspace/tiny_sims/catalog.json \
  --model Qwen/Qwen3.5-2B --out $WORK/weightspace/read_qwen35_2b.jsonl --no-thinking || true

echo "[monitor] push artifacts to HF..."
$PY - <<'PY'
from huggingface_hub import upload_folder, HfApi
api=HfApi()
api.upload_folder(folder_path='/mnt/work/galoaa/phillm/preds', repo_id='bishoygaloaa/phillm-results', repo_type='dataset', path_in_repo='preds', commit_message='Update predictions/metrics')
api.upload_folder(folder_path='/mnt/work/galoaa/phillm/probes', repo_id='bishoygaloaa/phillm-results', repo_type='dataset', path_in_repo='probes', commit_message='Probe results')
api.upload_folder(folder_path='/mnt/work/galoaa/phillm/checkpoints/qwen35-2b-lora', repo_id='bishoygaloaa/phillm-qwen3.5-2b-lora', repo_type='model', commit_message='LoRA adapter')
api.upload_folder(folder_path='/mnt/work/galoaa/phillm/checkpoints/qwen35-2b-probeguided', repo_id='bishoygaloaa/phillm-qwen3.5-2b-probeguided', repo_type='model', commit_message='Probe-guided adapter')
print('HF push done')
PY

echo "[monitor] complete"
