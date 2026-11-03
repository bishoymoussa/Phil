# PhiLLM (Phil)

**Physics + LLMs**: can language models predict the next physical state, and do their internals encode physics we can probe and teach?

This repo implements the course project [proposal](docs/proposal.pdf) with an updated framing inspired by [Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/).

## Hypotheses

- **H1 (prediction):** Can an LLM predict the next full state of a Newtonian system from natural language? Tested ID vs OOD (magnitude, horizon, objects, collisions) with base and LoRA models.
- **H2 (understanding):** Are derived quantities (acceleration, future position, momentum, energy) linearly decodable? Can probe-guided training teach that structure?
- **RQ3:** When should an agent call a physics simulator vs answer directly?
- **RQ5:** Can an LLM read which physical law a tiny neural simulator encodes from its weights?

## Quick start

```bash
# Install (Journey or Explorer)
pip install -e ".[dev]"

# Run simulator tests
pytest tests/ -q

# Generate a tiny inspection set
python -m phil.data.generate --level 0 --n 50 --out data/inspect_50.jsonl

# Full ID/OOD splits (writes under --out-dir)
python -m phil.data.generate --full --out-dir /mnt/work/galoaa/phillm/data
```

## Artifacts

| Kind | Location |
|------|----------|
| Dataset | [`bishoygaloaa/phillm-data`](https://huggingface.co/datasets/bishoygaloaa/phillm-data) |
| Results | [`bishoygaloaa/phillm-results`](https://huggingface.co/datasets/bishoygaloaa/phillm-results) |
| LoRA | [`bishoygaloaa/phillm-qwen3.5-2b-lora`](https://huggingface.co/bishoygaloaa/phillm-qwen3.5-2b-lora) |
| Probe-guided | [`bishoygaloaa/phillm-qwen3.5-2b-probeguided`](https://huggingface.co/bishoygaloaa/phillm-qwen3.5-2b-probeguided) |
| Selective agent | [`bishoygaloaa/phillm-qwen3.5-2b-selective`](https://huggingface.co/bishoygaloaa/phillm-qwen3.5-2b-selective) |
| Demo Space | [`bishoygaloaa/phillm-demo`](https://huggingface.co/spaces/bishoygaloaa/phillm-demo) |
| Project site | [bishoymoussa.github.io/Phil](https://bishoymoussa.github.io/Phil) |

## Layout

```
phil/sim/          Newtonian simulator + JSON tool interface
phil/data/         NL problem generator, ID/OOD splits
phil/eval/         Metrics (MAE, tolerance accuracy, U_λ, AUROC)
phil/models/       Prompting, vLLM infer, LoRA / probe-guided / agent
phil/probing/      Hidden-state extraction + controlled probes
phil/weightspace/  Tiny neural sims + weight-reading eval
scripts/explorer/  SLURM sbatch scripts
scripts/journey/   Local CPU/orchestration helpers
```

## Compute

- **Journey** (`/mnt/work/galoaa/phillm`): data gen, tests, probe fitting, light inference
- **Explorer** (`/scratch/galoaa.b/phillm`): training, vLLM, hidden-state dumps; mirrored to HF

## Citation

```bibtex
@misc{galoaa2026phillm,
  title        = {PhiLLM: Physically Grounded Language Models},
  author       = {Galoaa, Bishoy and Sullivan, Jonathan and Luo, Yunhua},
  year         = {2026},
  howpublished = {\url{https://github.com/bishoymoussa/Phil}}
}
```
