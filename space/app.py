"""PhiLLM Gradio Space: numeric results + playable 2D/1D simulator (GIF animation)."""

from __future__ import annotations

try:
    import audioop  # noqa: F401
except ModuleNotFoundError:
    try:
        import audioop_lts as audioop  # type: ignore
        import sys as _sys

        _sys.modules["audioop"] = audioop
    except Exception:
        pass

import io
import json
import sys
import tempfile
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

SPACE_ROOT = Path(__file__).resolve().parent
if str(SPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(SPACE_ROOT))

from phil.sim.simulator import run  # noqa: E402
from phil.sim.spec import SpecError, parse_spec  # noqa: E402

DEMO_SPEC = {
    "dim": 2,
    "dt": 0.05,
    "horizon": 40,
    "objects": [
        {"id": "o0", "mass": 2.0, "position": [-2.0, 0.5], "velocity": [1.8, 0.2], "force": [0.0, -1.2]},
        {"id": "o1", "mass": 1.0, "position": [2.2, -0.3], "velocity": [-1.4, 0.6], "force": [0.0, -1.2]},
    ],
    "collisions": True,
    "walls": [-4.0, 4.0, -3.0, 3.0],
}

EXAMPLES_DIR = SPACE_ROOT / "examples"

RESULTS_MD = """
# PhiLLM — Physics + LLMs

**Takeaway:** Zero-shot **Qwen3.5-2B** mostly **cannot** predict Newtonian next states
(**8% ID** vs **100% oracle**, **0%** on four OOD axes). **LoRA SFT lifts ID to ~38%**
(n=50) while OOD eval is finishing — early evidence of ID interpolation, not a full world model.

| System | ID acc | Pos MAE | Status |
|--------|--------|---------|--------|
| Oracle | **100%** | 0.00 | final · n=2000 |
| Zero-shot 2B | **8%** | 13.50 | partial · n=50 |
| LoRA SFT | **~38%** | ~0.72 | partial · n=50 |

| OOD axis (zero-shot) | Acc | Pos MAE |
|----------------------|-----|---------|
| `ood_mag` | **0%** | 64.7 |
| `ood_hor` | **0%** | 38.4 |
| `ood_obj` | **0%** | 31.8 |
| `ood_col` | **0%** | 29.8 |

[Paper site](https://bishoymoussa.github.io/Phil/) ·
[GitHub](https://github.com/bishoymoussa/Phil) ·
[LoRA model](https://huggingface.co/bishoygaloaa/phillm-qwen3.5-2b-lora) ·
[Metrics](https://huggingface.co/datasets/bishoygaloaa/phillm-results)
"""


def _load_example_specs() -> list[tuple[str, str]]:
    out = [("Demo collision (2D)", json.dumps(DEMO_SPEC, indent=2))]
    if EXAMPLES_DIR.exists():
        for p in sorted(EXAMPLES_DIR.glob("*.json"))[:12]:
            out.append((p.stem, p.read_text(encoding="utf-8")))
    return out


def _plot_upto(result: dict, dim: int, walls, frame_idx: int):
    traj = result["trajectory"]
    frame_idx = int(np.clip(frame_idx, 0, len(traj) - 1))
    fig, ax = plt.subplots(figsize=(6.2, 3.8), facecolor="#0b1c24")
    ax.set_facecolor("#071218")
    for spine in ax.spines.values():
        spine.set_color("#3a5560")
    ax.tick_params(colors="#9db4bd")
    ax.xaxis.label.set_color("#9db4bd")
    ax.yaxis.label.set_color("#9db4bd")
    ax.title.set_color("#e8f1f4")
    colors = ["#d4a35c", "#5cb8d4", "#d45c7a", "#7ad45c"]
    n_obj = len(result["objects"])
    if dim == 1:
        ax.set_xlabel("t (s)")
        ax.set_ylabel("x (m)")
        ts = [s["t"] for s in traj[: frame_idx + 1]]
        for i in range(n_obj):
            xs = [s["positions"][i][0] for s in traj[: frame_idx + 1]]
            ax.plot(ts, xs, color=colors[i % len(colors)], lw=2, label=result["objects"][i]["id"])
            ax.plot(ts[-1], xs[-1], "o", color=colors[i % len(colors)], ms=8)
        if walls and len(walls) >= 2:
            ax.axhline(walls[0], color="#9db4bd", ls="--", lw=1)
            ax.axhline(walls[1], color="#9db4bd", ls="--", lw=1)
    else:
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal", adjustable="datalim")
        for i in range(n_obj):
            xs = [s["positions"][i][0] for s in traj[: frame_idx + 1]]
            ys = [s["positions"][i][1] for s in traj[: frame_idx + 1]]
            ax.plot(xs, ys, color=colors[i % len(colors)], lw=1.8, label=result["objects"][i]["id"])
            ax.plot(xs[-1], ys[-1], "o", color=colors[i % len(colors)], ms=9)
        if walls and len(walls) >= 4:
            xmin, xmax, ymin, ymax = walls
            ax.plot(
                [xmin, xmax, xmax, xmin, xmin],
                [ymin, ymin, ymax, ymax, ymin],
                color="#9db4bd",
                ls="--",
                lw=1,
            )
    ax.legend(facecolor="#123040", labelcolor="#e8f1f4", fontsize=8)
    ax.set_title(f"t={traj[frame_idx]['t']:.3f}s · frame {frame_idx}/{len(traj)-1}")
    fig.tight_layout()
    return fig


def _fig_to_pil(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _pack_traj(result: dict, dim: int, walls) -> str:
    payload = {
        "dim": int(dim),
        "walls": walls,
        "objects": [{"id": o["id"]} for o in result["objects"]],
        "trajectory": [
            {"t": float(s["t"]), "positions": [[float(x) for x in p] for p in s["positions"]]}
            for s in result["trajectory"]
        ],
        "n_collisions": int(result["n_collisions"]),
        "t_final": float(result["t"]),
        "final_objects": [
            {
                "id": o["id"],
                "position": [float(x) for x in o["position"]],
                "velocity": [float(v) for v in o["velocity"]],
            }
            for o in result["objects"]
        ],
    }
    return json.dumps(payload)


def _unpack(traj_json: str) -> dict | None:
    if not traj_json or not str(traj_json).strip():
        return None
    try:
        return json.loads(traj_json)
    except Exception:
        return None


def _result_view(data: dict) -> dict:
    return {
        "trajectory": data["trajectory"],
        "objects": data["objects"],
        "n_collisions": data["n_collisions"],
        "t": data["t_final"],
    }


def render_gif(traj_json: str, stride: int = 1) -> str | None:
    """Write an animated GIF and return its path (Gradio Image-friendly)."""
    data = _unpack(traj_json)
    if not data:
        return None
    result = _result_view(data)
    frames = []
    n = len(data["trajectory"])
    for i in range(0, n, max(1, stride)):
        fig = _plot_upto(result, data["dim"], data.get("walls"), i)
        frames.append(_fig_to_pil(fig))
    if not frames:
        return None
    tmp = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
    path = tmp.name
    tmp.close()
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=80,
        loop=0,
    )
    return path


def simulate(spec_json: str):
    try:
        spec = parse_spec(spec_json)
        result = run(spec, return_trajectory=True)
    except SpecError as e:
        return None, "", "—", f"Spec error: {e}", RESULTS_MD, gr.update(maximum=1, value=0), "t=—"
    except Exception as e:  # noqa: BLE001
        return None, "", "—", f"Error: {e}", RESULTS_MD, gr.update(maximum=1, value=0), "t=—"

    try:
        walls = json.loads(spec_json).get("walls")
    except Exception:
        walls = None

    packed = _pack_traj(result, spec.dim, walls)
    gif = render_gif(packed, stride=1)
    lines = [f"t = {result['t']:.4f} s · collisions = {result['n_collisions']}"]
    for o in result["objects"]:
        pos = ", ".join(f"{x:.4f}" for x in o["position"])
        vel = ", ".join(f"{v:.4f}" for v in o["velocity"])
        lines.append(f"{o['id']}:  x=[{pos}]   v=[{vel}]")
    lines.append("")
    lines.append("Zero-shot ID 8% → LoRA ~38% (n=50). OOD zero-shot still ~0%.")
    n_frames = len(result["trajectory"]) - 1
    status = f"OK · dim={spec.dim} · steps={n_frames} · collisions={result['n_collisions']}"
    scrub = gr.update(maximum=max(n_frames, 1), value=0)
    return gif, packed, "\n".join(lines), status, RESULTS_MD, scrub, f"t=0.000s · frame 0/{n_frames}"


def show_frame(traj_json: str, frame_idx: float):
    data = _unpack(traj_json)
    if not data:
        return None, "Run a simulation first."
    result = _result_view(data)
    idx = int(frame_idx)
    fig = _plot_upto(result, data["dim"], data.get("walls"), idx)
    img = _fig_to_pil(fig)
    label = f"t={data['trajectory'][idx]['t']:.3f}s · frame {idx}/{len(data['trajectory'])-1}"
    return img, label


def play(traj_json: str):
    """Regenerate / return the looping GIF — reliable Play on Spaces."""
    data = _unpack(traj_json)
    if not data:
        return None, "Run a simulation first.", gr.update()
    gif = render_gif(traj_json, stride=1)
    n = len(data["trajectory"]) - 1
    return gif, f"Playing 0→{n}", gr.update(value=0, maximum=max(n, 1))


def reset(traj_json: str):
    img, label = show_frame(traj_json, 0)
    return img, label, gr.update(value=0)


def build() -> gr.Blocks:
    examples = _load_example_specs()
    with gr.Blocks(title="PhiLLM — results + simulator") as demo:
        results_box = gr.Markdown(RESULTS_MD)
        gr.Markdown(
            "### Exact Newtonian simulator\n"
            "Pick a preset → **Run simulation** (builds an animated GIF) → **Play** restarts the loop. "
            "JSON is the advanced editor."
        )
        with gr.Row():
            with gr.Column(scale=1):
                example_dd = gr.Dropdown(
                    choices=[e[0] for e in examples],
                    value=examples[0][0],
                    label="Preset scene",
                )
                with gr.Accordion("WorldSpec JSON", open=False):
                    inp = gr.Textbox(value=examples[0][1], lines=14, label="JSON")
                btn = gr.Button("Run simulation", variant="primary")
                status = gr.Textbox(label="Status", interactive=False)
            with gr.Column(scale=1.2):
                anim = gr.Image(label="Trajectory animation", type="filepath")
                with gr.Row():
                    play_btn = gr.Button("Play / replay", variant="primary")
                    reset_btn = gr.Button("Reset to t=0")
                scrub = gr.Slider(0, 40, value=0, step=1, label="Scrub frame (static)")
                frame_lbl = gr.Textbox(label="Frame", value="t=—", interactive=False)
                nums = gr.Textbox(label="Final state (numbers)", lines=7)
                traj_store = gr.Textbox(visible=False)

        def on_example(name: str):
            for n, body in examples:
                if n == name:
                    return body
            return examples[0][1]

        example_dd.change(on_example, inputs=[example_dd], outputs=[inp])
        # also load preset JSON into editor when dropdown changes, then user hits Run
        btn.click(
            simulate,
            inputs=[inp],
            outputs=[anim, traj_store, nums, status, results_box, scrub, frame_lbl],
        )
        play_btn.click(play, inputs=[traj_store], outputs=[anim, frame_lbl, scrub])
        reset_btn.click(reset, inputs=[traj_store], outputs=[anim, frame_lbl, scrub])
        scrub.release(show_frame, inputs=[traj_store, scrub], outputs=[anim, frame_lbl])
    return demo


if __name__ == "__main__":
    build().launch()
