#!/usr/bin/env bash
set -euo pipefail
ROOT="${PHIL_ROOT:-/home/bi.ga/Phil}"
WORK="${PHIL_WORK:-/mnt/work/galoaa/phillm}"
cd "$ROOT"
python3 -m pip install -e ".[dev]" --user
mkdir -p "$WORK"/{data,checkpoints,logs,preds,probes,weightspace,cache}
python -c "from phil.sim.simulator import run; print('phil import ok')"
python -m pytest tests/ -q
python scripts/hf_setup.py
echo "Journey setup done"
