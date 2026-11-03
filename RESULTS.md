# PhiLLM results (numeric snapshot)

**Status:** partial zeroshot complete (n=50) · LoRA training ~67% · RQ3 baselines in · probes/weight-read in flight.

**Takeaway:** Zero-shot Qwen3.5-2B mostly cannot predict Newtonian next states — **8% ID** vs **100% oracle**, and **0%** on all OOD axes (`ood_mag` / `ood_hor` / `ood_obj` / `ood_col`, n=50 each).

## ID

| System | Accuracy | Pos MAE | Vel MAE | n | Status |
|--------|----------|---------|---------|---|--------|
| Oracle | 100.0% | 0.00 | 0.00 | 2000 | final |
| Zero-shot Qwen3.5-2B | 8.0% | 13.50 | 42.93 | 50 | partial |
| LoRA SFT | — | — | — | — | training ~59% of 500 steps |

By task (zeroshot ID): T1 0/32 · T2 0/11 · T3 4/7 (57%).

## OOD (zeroshot)

| Axis | Accuracy | Pos MAE | Vel MAE | n | Status |
|------|----------|---------|---------|---|--------|
| ood_mag | 0.0% | 64.72 | 1.31e4 | 50 | partial |
| ood_hor | 0.0% | 38.38 | 17.98 | 50 | partial |
| ood_obj | 0.0% | 31.85 | 42.96 | 50 | partial |
| ood_col | 0.0% | 29.76 | 57.15 | 50 | partial |

## Call policy / probes / weight-space

| Experiment | Metric | Value | Status |
|------------|--------|-------|--------|
| RQ3 LLM-only | ID acc | 8.0% | partial |
| RQ3 always-call oracle | ID acc / U_0.1 | 100% / 0.90 | n=50 |
| RQ3 random (rate 0.5) | ID acc | 50.0% ± 1.3% | 5 seeds |
| RQ3 selective θ | U_λ / AUROC | — | after LoRA labels |
| H2 ridge probes (base) | best R² | x0 0.42 · F 0.40 · xk 0.34 · a 0.15 | n=200 |
| RQ5 tiny sims | train MSE | catalog written | 4 laws |
| RQ5 LLM weight-read | law ID | 0/4 law IDs parsed (pilot, no controls) | pilot |

Source: `/mnt/work/galoaa/phillm/preds/*.metrics.json` → HF `bishoygaloaa/phillm-results`.
