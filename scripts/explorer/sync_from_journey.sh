#!/usr/bin/env bash
# Run ON journey: rsync code + data to explorer scratch
set -euo pipefail
ROOT="${PHIL_ROOT:-/home/bi.ga/Phil}"
REMOTE_CODE="/scratch/galoaa.b/phillm/Phil"
REMOTE_DATA="/scratch/galoaa.b/phillm/data"
ssh explorer "mkdir -p $REMOTE_CODE /scratch/galoaa.b/phillm/{data,checkpoints,logs,preds,probes,weightspace,cache}"
rsync -az --delete --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' "$ROOT/" "explorer:$REMOTE_CODE/"
if [[ -d /mnt/work/galoaa/phillm/data ]]; then
  rsync -az /mnt/work/galoaa/phillm/data/ "explorer:$REMOTE_DATA/"
fi
echo "synced to explorer:$REMOTE_CODE"
