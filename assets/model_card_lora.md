---
license: mit
base_model: Qwen/Qwen3.5-2B
tags:
  - physics
  - lora
  - phillm
library_name: peft
---

# PhiLLM LoRA — Qwen3.5-2B

LoRA adapter fine-tuned for Newtonian next-state prediction (T1–T3) on the PhiLLM ID training split.

- Base: `Qwen/Qwen3.5-2B`
- Data: [`bishoygaloaa/phillm-data`](https://huggingface.co/datasets/bishoygaloaa/phillm-data)
- Code: https://github.com/bishoymoussa/Phil

Checkpoint selected on ID **dev** only. Evaluate with `python -m phil.eval.eval`.
