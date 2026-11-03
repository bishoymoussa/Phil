---
license: mit
task_categories:
  - text-generation
  - question-answering
language:
  - en
tags:
  - physics
  - llm
  - simulation
  - probing
pretty_name: PhiLLM
size_categories:
  - 10K<n<100K
---

# PhiLLM data

Synthetic Newtonian point-mass problems with exact simulator ground truth.

## Splits

| File | N | Description |
|------|---|-------------|
| `train.jsonl` | 50,000 | ID training |
| `dev.jsonl` | 1,000 | ID development (checkpoint selection) |
| `dlab.jsonl` | 5,000 | ID call-labeling set |
| `test.jsonl` | 2,000 | ID test |
| `ood_mag.jsonl` | 2,000 | OOD magnitudes |
| `ood_hor.jsonl` | 2,000 | OOD horizons |
| `ood_obj.jsonl` | 2,000 | OOD #objects |
| `ood_col.jsonl` | 2,000 | OOD multi-collision |
| `inspect_50.jsonl` | 50 | Level-0 inspection set |

Each row: `id`, `split`, `task_type` (T1/T2/T3), `text`, `gold_spec`, `gold_answer`, `tags`, `param_hash`.

Code: https://github.com/bishoymoussa/Phil
