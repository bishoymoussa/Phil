"""PhiLLM Gradio Space: WorldSpec editor, simulator trajectories, base/LoRA placeholders."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Self-contained: vendored phil.sim lives next to this file
SPACE_ROOT = Path(__file__).resolve().parent
if str(SPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(SPACE_ROOT))

from phil.sim.simulator import run  # noqa: E402
from phil.sim.spec import SpecError, parse_spec  # noqa: E402


DEMO_SPEC = {
    "dim": 1,
    "dt": 0.1,
    "horizon": 8,
    "objects": [
        {"id": "o0", "mass": 2.0, "position": [-1.0], "velocity": [1.5], "force": [0.0]},
        {"id": "o1", "mass": 1.0, "position": [1.5], "velocity": [-1.0], "force": [0.0]},
    ],
    "collisions": True,
    "walls": [-8.0, 8.0],
}

GALLERY_DIR = SPACE_ROOT / "gallery"
EXAMPLES_DIR = SPACE_ROOT / "examples"


def _load_example_specs() -> list[tuple[str, str]]:
    out = [("Demo collision (1D)", json.dumps(DEMO_SPEC, indent=2))]
    if EXAMPLES_DIR.exists():
        for p in sorted(EXAMPLES_DIR.glob("*.json"))[:12]:
            out.append((p.stem, p.read_text(encoding="utf-8")))
    return out


def simulate_and_plot(spec_json: str):
    try:
        spec = parse_spec(spec_json)
        result = run(spec, return_trajectory=True)
    except SpecError as e:
        return None, f"Spec error: {e}", "—", "—", "—"
    except Exception as e:  # noqa: BLE001
        return None, f"Error: {e}", "—", "—", "—"

    traj = result["trajectory"]
    t = np.array([s["t"] for s in traj])
    fig, ax = plt.subplots(figsize=(7, 3.8))
    if spec.dim == 1:
        for i, oid in enumerate(result["objects"]):
            xs = [s["positions"][i][0] for s in traj]
            ax.plot(t, xs, marker="o", ms=3, label=f"{oid['id']} position")
        ax.set_xlabel("t (s)")
        ax.set_ylabel("x (m)")
    else:
        for i, oid in enumerate(result["objects"]):
            xs = [s["positions"][i][0] for s in traj]
            ys = [s["positions"][i][1] for s in traj]
            ax.plot(xs, ys, marker="o", ms=3, label=oid["id"])
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("Exact simulator trajectory")
    fig.tight_layout()

    final = {
        "objects": [
            {"id": o["id"], "position": o["position"], "velocity": o["velocity"]}
            for o in result["objects"]
        ],
        "t": result["t"],
        "n_collisions": result["n_collisions"],
    }
    sim_txt = json.dumps(final, indent=2)
    base_txt = (
        "[base placeholder] Load Qwen3.5-2B on a GPU Space to compare. "
        "Until then, use the exact simulator as ground truth."
    )
    lora_txt = (
        "[LoRA placeholder] Adapter: bishoygaloaa/phillm-qwen3.5-2b-lora. "
        "Wire PEFT inference when GPU quota is available."
    )
    return fig, sim_txt, base_txt, lora_txt, f"OK · collisions={result['n_collisions']} · t={result['t']:.3f}"


def gallery_images():
    if not GALLERY_DIR.exists():
        return []
    return sorted(str(p) for p in GALLERY_DIR.glob("*.png"))[:24]


def build() -> gr.Blocks:
    examples = _load_example_specs()
    with gr.Blocks(title="PhiLLM demo") as demo:
        gr.Markdown(
            """
# PhiLLM — Physics + LLMs
Paste a **WorldSpec** JSON, run the exact Newtonian simulator, and inspect trajectories.
Compare placeholders for **base** vs **LoRA** until GPU model serving is enabled.

[GitHub](https://github.com/bishoymoussa/Phil) ·
[Dataset](https://huggingface.co/datasets/bishoygaloaa/phillm-data) ·
[Results](https://huggingface.co/datasets/bishoygaloaa/phillm-results)
"""
        )
        with gr.Row():
            with gr.Column(scale=1):
                example_dd = gr.Dropdown(
                    choices=[e[0] for e in examples],
                    value=examples[0][0],
                    label="Load example",
                )
                inp = gr.Textbox(label="WorldSpec JSON", value=examples[0][1], lines=18)
                btn = gr.Button("Run simulator", variant="primary")
                status = gr.Textbox(label="Status", interactive=False)
            with gr.Column(scale=1):
                plot = gr.Plot(label="Trajectory")
                sim_out = gr.Textbox(label="Simulator final state", lines=10)
        with gr.Row():
            base_out = gr.Textbox(label="Base LLM", lines=6)
            lora_out = gr.Textbox(label="LoRA LLM", lines=6)

        def on_example(name: str):
            for n, body in examples:
                if n == name:
                    return body
            return examples[0][1]

        example_dd.change(on_example, inputs=[example_dd], outputs=[inp])
        btn.click(simulate_and_plot, inputs=[inp], outputs=[plot, sim_out, base_out, lora_out, status])

        gr.Markdown("## Result snapshots")
        res = sorted(str(p) for p in (SPACE_ROOT / "results_figs").glob("*.png"))
        if res:
            gr.Gallery(value=res, label="Oracle / metrics figures", columns=3, height=280)
        gr.Markdown("## Dataset gallery")
        imgs = gallery_images()
        if imgs:
            gr.Gallery(value=imgs, label="ID / OOD / inspect trajectories", columns=3, height=420)
        else:
            gr.Markdown("_Gallery images will appear after `scripts/journey/04_make_figures.sh`._")
    return demo


if __name__ == "__main__":
    build().launch()
