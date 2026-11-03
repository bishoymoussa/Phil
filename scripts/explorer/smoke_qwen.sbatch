#!/bin/bash
#SBATCH --job-name=phil-smoke
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/galoaa.b/phillm/logs/smoke_%j.out
#SBATCH --error=/scratch/galoaa.b/phillm/logs/smoke_%j.err

set -euo pipefail
source /scratch/galoaa.b/phillm/Phil/scripts/explorer/common.sh
export PHIL_VENV="${PHIL_VENV:-/scratch/galoaa.b/egoexo360/env}"
activate_phil_venv

cd "$PHIL_CODE"
nvidia-smi
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
from transformers import AutoModelForCausalLM, AutoTokenizer
name = "Qwen/Qwen3.5-2B"
try:
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    messages = [{"role":"user","content":"Say hi in one word."}]
    try:
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=16)
    print(tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
    print("SMOKE_OK", name)
except Exception as e:
    print("Qwen3.5 failed:", repr(e))
    name = "Qwen/Qwen3-1.7B"
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    messages = [{"role":"user","content":"Say hi in one word."}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=16)
    print(tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
    print("SMOKE_OK_FALLBACK", name)
PY
