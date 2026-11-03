#!/usr/bin/env bash
set -euo pipefail
DATA="${PHIL_DATA:-/mnt/work/galoaa/phillm/data}"
PY="${PHIL_PY:-/home/bi.ga/.conda/envs/grpo-motion/bin/python}"
# Upload dataset folder
"$PY" - <<PY
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="$DATA",
    repo_id="bishoygaloaa/phillm-data",
    repo_type="dataset",
    commit_message="Add PhiLLM ID/OOD splits and inspect_50",
)
print("uploaded phillm-data")
PY
