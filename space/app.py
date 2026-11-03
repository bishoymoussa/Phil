"""Gradio demo stub for PhiLLM: scene -> base / LoRA / simulator."""

from __future__ import annotations

import json

import gradio as gr

from phil.sim.simulator import run
from phil.sim.spec import SpecError, parse_spec


DEMO_SPEC = {
    "dim": 1,
    "dt": 0.1,
    "horizon": 5,
    "objects": [
        {"id": "o0", "mass": 2.0, "position": [0.0], "velocity": [1.0], "force": [0.0]}
    ],
    "collisions": False,
}


def simulate(spec_json: str) -> str:
    try:
        result = run(parse_spec(spec_json))
        return json.dumps(result, indent=2)
    except SpecError as e:
        return f"Spec error: {e}"
    except Exception as e:  # noqa: BLE001
        return f"Error: {e}"


def placeholder_llm(spec_json: str, which: str) -> str:
    return (
        f"[{which}] Model inference is wired in the full Space deploy. "
        "For now, compare against the exact simulator output on the right.\n"
        f"Input spec length={len(spec_json)} chars."
    )


def predict(spec_json: str):
    return (
        placeholder_llm(spec_json, "base"),
        placeholder_llm(spec_json, "LoRA"),
        simulate(spec_json),
    )


def build() -> gr.Blocks:
    with gr.Blocks(title="PhiLLM demo") as demo:
        gr.Markdown(
            """
            # PhiLLM — Physics + LLMs
            Describe / paste a JSON world spec. Compare **base**, **LoRA**, and the **exact simulator**.
            Dataset: [phillm-data](https://huggingface.co/datasets/bishoygaloaa/phillm-data) ·
            Code: [github.com/bishoymoussa/Phil](https://github.com/bishoymoussa/Phil)
            """
        )
        inp = gr.Textbox(label="WorldSpec JSON", value=json.dumps(DEMO_SPEC, indent=2), lines=16)
        btn = gr.Button("Run")
        with gr.Row():
            base = gr.Textbox(label="Base LLM", lines=12)
            lora = gr.Textbox(label="LoRA LLM", lines=12)
            sim = gr.Textbox(label="Simulator (ground truth)", lines=12)
        btn.click(predict, inputs=[inp], outputs=[base, lora, sim])
    return demo


if __name__ == "__main__":
    build().launch()
