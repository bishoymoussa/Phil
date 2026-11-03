# PhiLLM results (numeric snapshot)

**Takeaway:** Zero-shot ID **8%** → LoRA ID **38%** (n=50). OOD zeroshot **0%** on all axes; LoRA-OOD eval in flight.

## ID

| System | Accuracy | Pos MAE | Vel MAE | n | Status |
|--------|----------|---------|---------|---|--------|
| Oracle | 100.0% | 0.00 | 0.00 | 2000 | final |
| Zero-shot Qwen3.5-2B | 8.0% | 13.50 | 42.93 | 50 | partial |
| LoRA SFT Qwen3.5-2B | **38.0%** | 0.82 | 1.51 | 50 | partial |

LoRA by task: T1 11/32 (34%) · T2 2/11 (18%) · T3 6/7 (86%).

## OOD (zeroshot)

| Axis | Accuracy | Pos MAE | Vel MAE | n |
|------|----------|---------|---------|---|
| ood_mag | 0.0% | 64.72 | 1.31e4 | 50 |
| ood_hor | 0.0% | 38.38 | 17.98 | 50 |
| ood_obj | 0.0% | 31.85 | 42.96 | 50 |
| ood_col | 0.0% | 29.76 | 57.15 | 50 |

LoRA-OOD: running (mag/hor/obj/col).

## RQ3 baselines

| System | ID acc | Notes |
|--------|--------|-------|
| LLM-only | 8% | = zeroshot |
| Always-call oracle | 100% | U_0.1=0.90 |
| Random@0.5 | 50%±1.3% | 5 seeds |
