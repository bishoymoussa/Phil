#!/usr/bin/env bash
# Run after LoRA checkpoint exists: eval ID/OOD, probe, probe-guided, RQ3 baselines, push HF.
set -euo pipefail
ROOT=/home/bi.ga/Phil
WORK=/mnt/work/galoaa/phillm
PY=/home/bi.ga/.conda/envs/grpo-motion/bin/python
export PYTHONPATH=$ROOT
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
export HF_HOME=$WORK/cache/hf
ADAPTER=$WORK/checkpoints/qwen35-2b-lora
MODEL=Qwen/Qwen3.5-2B

echo "[after_lora] waiting for adapter..."
while [[ ! -f $ADAPTER/adapter_config.json ]]; do sleep 60; done
echo "[after_lora] found adapter"

# Eval LoRA on test + OOD (50 each for speed; bump later)
for split in test ood_mag ood_hor ood_obj ood_col; do
  out=$WORK/preds/lora_${split}.jsonl
  if [[ ! -f $out ]]; then
    CUDA_VISIBLE_DEVICES=3 $PY -u -m phil.models.infer_vllm \
      --model $MODEL --adapter $ADAPTER --data $WORK/data/${split}.jsonl \
      --out $out --backend hf --no-fewshot --limit 50 --max-tokens 256
  fi
  $PY -m phil.eval.eval --gold $WORK/data/${split}.jsonl --pred $out --out ${out%.jsonl}.metrics.json
done

# Probing
CUDA_VISIBLE_DEVICES=3 $PY -m phil.probing.extract --model $MODEL --data $WORK/data/dev.jsonl --out $WORK/probes/base --limit 200
$PY -m phil.probing.probes --hidden $WORK/probes/base/hidden_last_prompt.npz --targets $WORK/probes/base/targets.npz --out $WORK/probes/base/probe_results.json
$PY -m phil.probing.controls --hidden $WORK/probes/base/hidden_last_prompt.npz --targets $WORK/probes/base/targets.npz --out $WORK/probes/base/controls.json

CUDA_VISIBLE_DEVICES=3 $PY -m phil.probing.extract --model $MODEL --adapter $ADAPTER --data $WORK/data/dev.jsonl --out $WORK/probes/lora --limit 200
$PY -m phil.probing.probes --hidden $WORK/probes/lora/hidden_last_prompt.npz --targets $WORK/probes/lora/targets.npz --out $WORK/probes/lora/probe_results.json
$PY -m phil.probing.controls --hidden $WORK/probes/lora/hidden_last_prompt.npz --targets $WORK/probes/lora/targets.npz --out $WORK/probes/lora/controls.json

# Probe-guided
CUDA_VISIBLE_DEVICES=2 $PY -m phil.models.train_probeguided \
  --model $MODEL --train $WORK/data/train.jsonl --dev $WORK/data/dev.jsonl \
  --out $WORK/checkpoints/qwen35-2b-probeguided --epochs 1 --limit 2000

# RQ3
$PY -m phil.models.agent oracle --data $WORK/data/test.jsonl --out $WORK/preds/oracle_test.jsonl
cp -f $WORK/preds/lora_test.jsonl $WORK/preds/llm_only_test.jsonl
CUDA_VISIBLE_DEVICES=3 $PY -m phil.models.infer_vllm --model $MODEL --adapter $ADAPTER \
  --data $WORK/data/dlab.jsonl --out $WORK/preds/lora_dlab.jsonl --backend hf --no-fewshot --limit 200
$PY -m phil.models.agent label-calls --gold $WORK/data/dlab.jsonl --pred $WORK/preds/lora_dlab.jsonl --out $WORK/data/labels_dlab.jsonl
CUDA_VISIBLE_DEVICES=3 $PY -m phil.models.agent infer --model $MODEL --adapter $ADAPTER \
  --data $WORK/data/test.jsonl --out $WORK/preds/always_call_test.jsonl --force-call --limit 50
$PY -m phil.eval.eval --gold $WORK/data/test.jsonl --pred $WORK/preds/always_call_test.jsonl --out $WORK/preds/always_call_test.metrics.json

# Weight reading with 2B (GPU)
$PY -m phil.weightspace.train_tiny_sims --out $WORK/weightspace/tiny_sims
CUDA_VISIBLE_DEVICES=3 $PY -m phil.weightspace.ask_llm --catalog $WORK/weightspace/tiny_sims/catalog.json \
  --model $MODEL --out $WORK/weightspace/read_qwen35_2b.jsonl --no-thinking --no-controls || true

# Figures + HF push
bash $ROOT/scripts/journey/04_make_figures.sh || true
$PY - <<'PY'
from huggingface_hub import upload_folder, HfApi
api=HfApi()
for folder, repo, rtype, path in [
  ('/mnt/work/galoaa/phillm/preds','bishoygaloaa/phillm-results','dataset','preds'),
  ('/mnt/work/galoaa/phillm/probes','bishoygaloaa/phillm-results','dataset','probes'),
  ('/mnt/work/galoaa/phillm/figs','bishoygaloaa/phillm-results','dataset','figs'),
  ('/mnt/work/galoaa/phillm/weightspace','bishoygaloaa/phillm-results','dataset','weightspace'),
  ('/mnt/work/galoaa/phillm/checkpoints/qwen35-2b-lora','bishoygaloaa/phillm-qwen3.5-2b-lora','model',None),
  ('/mnt/work/galoaa/phillm/checkpoints/qwen35-2b-probeguided','bishoygaloaa/phillm-qwen3.5-2b-probeguided','model',None),
]:
  kwargs=dict(folder_path=folder, repo_id=repo, repo_type=rtype, commit_message=f'PhiLLM update {repo}')
  if path: kwargs['path_in_repo']=path
  try:
    upload_folder(**kwargs); print('ok', repo)
  except Exception as e:
    print('fail', repo, e)
PY
echo "[after_lora] DONE"
