# PhiLLM results (numeric snapshot)

**Status:** partial zeroshot · LoRA training in flight · probes / call-policy not yet.

**Takeaway:** Zero-shot Qwen3.5-2B mostly cannot predict Newtonian next states — **8% ID** vs **100% oracle**, and **0%** on all OOD axes (`ood_mag` / `ood_hor` / `ood_obj` / `ood_col`, n=50 each).

## ID

| System | Accuracy | Pos MAE | Vel MAE | n | Status |
|--------|----------|---------|---------|---|--------|
| Oracle | 100.0% | 0.00 | 0.00 | 2000 | final |
| Zero-shot Qwen3.5-2B | 8.0% | 13.50 | 42.93 | 50 | partial |
| LoRA SFT | — | — | — | — | training ~41% of 500 steps |

By task (zeroshot ID): T1 0/32 · T2 0/11 · T3 4/7 (57%).

## OOD (zeroshot)

| Axis | Accuracy | Pos MAE | Vel MAE | n | Status |
|------|----------|---------|---------|---|--------|
| ood_mag | 0.0% | 64.72 | 1.31e4 | 50 | partial |
| ood_hor | 0.0% | 38.38 | 17.98 | 50 | partial |
| ood_obj | 0.0% | 31.85 | 42.96 | 50 | partial |
| ood_col | 0.0% | 29.76 | 57.15 | 50 | partial |

## Call policy / probes

| Experiment | Metric | Value | Status |
|------------|--------|-------|--------|
| RQ3 selective caller | U_λ / AUROC | — | queued |
| H2 ridge probes | R² / selectivity | — | needs hidden states |
| RQ5 weight-space | law ID acc | — | pilot pending |

Source files: `/mnt/work/galoaa/phillm/preds/*.metrics.json` → HF `bishoygaloaa/phillm-results`.
