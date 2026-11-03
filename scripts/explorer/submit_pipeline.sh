#!/usr/bin/env bash
# Submit chained Explorer jobs on multigpu. Run on Explorer after sync.
set -euo pipefail
CODE="${PHIL_CODE:-/scratch/galoaa.b/phillm/Phil}"
cd "$CODE/scripts/explorer"

smoke=$(sbatch --parsable smoke_qwen.sbatch)
echo "smoke=$smoke"

# After smoke, kick zeroshot on test for main model + train
zs=$(sbatch --parsable --dependency=afterok:$smoke --export=ALL,MODEL=Qwen/Qwen3.5-2B,SPLIT=test infer_zeroshot.sbatch)
echo "zeroshot_2b=$zs"

lora=$(sbatch --parsable --dependency=afterok:$smoke train_lora.sbatch)
echo "lora=$lora"

for split in test ood_mag ood_hor ood_obj ood_col; do
  j=$(sbatch --parsable --dependency=afterok:$lora --export=ALL,SPLIT=$split infer_lora.sbatch)
  echo "infer_lora_$split=$j"
done

probe_base=$(sbatch --parsable --dependency=afterok:$smoke --export=ALL,TAG=base probe_extract.sbatch)
probe_lora=$(sbatch --parsable --dependency=afterok:$lora --export=ALL,TAG=lora,ADAPTER=/scratch/galoaa.b/phillm/checkpoints/qwen35-2b-lora probe_extract.sbatch)
echo "probe_base=$probe_base probe_lora=$probe_lora"

rq3=$(sbatch --parsable --dependency=afterok:$lora rq3_baselines.sbatch)
pg=$(sbatch --parsable --dependency=afterok:$lora train_probeguided.sbatch)
ws=$(sbatch --parsable --dependency=afterok:$smoke weightspace.sbatch)
echo "rq3=$rq3 probeguided=$pg weightspace=$ws"

# optional larger zeroshot sweep (can run in parallel after smoke)
for m in Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B; do
  sbatch --dependency=afterok:$smoke --export=ALL,MODEL=$m,SPLIT=test,LIMIT=500 infer_zeroshot.sbatch
done

squeue -u "$USER"
