# PhiLLM Midterm Slides (outline)

## 1. Title
**PhiLLM** — Can LLMs predict and understand physical state?
Bishoy Galoaa · Jonathan Sullivan · Yunhua Luo

## 2. Framing
Weight-space programming: LLMs can read/write other networks' weights.
Our physics analogue: next-state prediction + probes + teaching + tool use.

## 3. Hypotheses
- H1: Predict full next state (ID vs OOD)
- H2: Derived quantities decodable / teachable
- RQ3: When to call the simulator
- RQ5: Read which law a tiny neural sim encodes

## 4. Environment
1D/2D Newtonian point masses · elastic collisions · NL templates · exact simulator

## 5. Headline table
Base vs LoRA on ID + 4 OOD axes (paste metrics)

## 6. Probing sketch
Stated vs derived · layer-0 / control-task selectivity

## 7. Agent baselines
LLM-only · always-call · oracle · cost $U_\lambda$

## 8. Weight-space pilot
Tiny force/spring/drag/gravity nets → LLM names the law

## 9. Next
Probe-guided training · selective policy · demo Space · final report

## 10. Links
github.com/bishoymoussa/Phil · huggingface.co/bishoygaloaa
