#!/usr/bin/env bash
# Create a dedicated PhiLLM venv on Explorer scratch (no conda).
set -euo pipefail
SCRATCH=/scratch/galoaa.b/phillm
VENV=$SCRATCH/venv
CODE=$SCRATCH/Phil
mkdir -p "$SCRATCH"
# Prefer a modern python if available
PY=python3
if [[ -x /scratch/galoaa.b/egoexo360/env/bin/python ]]; then
  PY=/scratch/galoaa.b/egoexo360/env/bin/python
fi
if [[ ! -x $VENV/bin/python ]]; then
  "$PY" -m venv "$VENV"
fi
# shellcheck disable=SC1090
source "$VENV/bin/activate"
python -m pip install -U pip setuptools wheel
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124 || python -m pip install torch
python -m pip install "transformers>=4.51" accelerate peft datasets trl huggingface_hub \
  numpy scipy pyyaml tqdm jsonlines scikit-learn matplotlib safetensors
python -m pip install -e "$CODE" --no-deps || true
python -c "import torch,transformers; print(torch.__version__, transformers.__version__, torch.cuda.is_available())"
echo "PHIL_VENV=$VENV"
