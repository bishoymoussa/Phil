#!/usr/bin/env bash
set -euo pipefail
ROOT="${PHIL_ROOT:-/home/bi.ga/Phil}"
DATA="${PHIL_DATA:-/mnt/work/galoaa/phillm/data}"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

python -m phil.data.generate --level 0 --n 50 --out "$DATA/inspect_50.jsonl"
python -m phil.data.generate --full --out-dir "$DATA" --seed 42
echo "Data ready under $DATA"
ls -lh "$DATA"
