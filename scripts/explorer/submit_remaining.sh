#!/usr/bin/env bash
# Submit remaining jobs once QOS slots free up. Safe to re-run.
set -euo pipefail
CODE="${PHIL_CODE:-/scratch/galoaa.b/phillm/Phil}"
cd "$CODE/scripts/explorer"

submit() {
  local name="$1"; shift
  echo "submitting $name..."
  if out=$(sbatch "$@" 2>&1); then
    echo "$out"
  else
    echo "FAILED: $out"
    return 1
  fi
}

# probing / rq3 / probe-guided / weightspace / larger zeroshot / theta
submit probe_base --export=ALL,TAG=base probe_extract.sbatch || true
submit probe_lora --export=ALL,TAG=lora,ADAPTER=/scratch/galoaa.b/phillm/checkpoints/qwen35-2b-lora probe_extract.sbatch || true
submit rq3 rq3_baselines.sbatch || true
submit pg train_probeguided.sbatch || true
submit ws weightspace.sbatch || true
submit theta selective_theta.sbatch || true

for m in Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B; do
  submit "zs_$m" --export=ALL,MODEL=$m,SPLIT=test,LIMIT=500 infer_zeroshot.sbatch || true
done

# OOD zeroshot for main model
for split in ood_mag ood_hor ood_obj ood_col; do
  submit "zs_$split" --export=ALL,MODEL=Qwen/Qwen3.5-2B,SPLIT=$split infer_zeroshot.sbatch || true
done

squeue -u "$USER"
