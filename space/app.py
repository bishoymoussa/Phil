"""PhiLLM Gradio Space: results-first summary + Gradio-driven 2D/1D animator."""

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

import json
import sys
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
from PIL import Image as PILImage

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

**Takeaway:** Zero-shot **Qwen3.5-2B** mostly **cannot** predict Newtonian next states —
**8% ID accuracy** vs a **100% oracle**, and **0%** on all four OOD axes
(partial eval, n=50; LoRA still training).

| System | ID acc | Pos MAE | Status |
|--------|--------|---------|--------|
| Oracle | **100%** | 0.00 | final · n=2000 |
| Zero-shot 2B | **8%** | 13.50 | partial · n=50 |
| LoRA SFT | *pending* | — | training in flight |

| OOD axis | Acc | Pos MAE | Status |
|----------|-----|---------|--------|
| `ood_mag` | **0%** | 64.7 | partial |
| `ood_hor` | **0%** | 38.4 | partial |
| `ood_obj` | **0%** | 31.8 | partial |
| `ood_col` | **0%** | 29.8 | partial |

**RQ3:** LLM-only **8%** · always-call oracle **100%** (U₀.₁=0.90) · random@0.5 **50%±1.3%**.

[Paper site](https://bishoymoussa.github.io/Phil/) ·
[GitHub](https://github.com/bishoymoussa/Phil) ·
[Dataset](https://huggingface.co/datasets/bishoygaloaa/phillm-data) ·
[Raw metrics](https://huggingface.co/datasets/bishoygaloaa/phillm-results)
"""

COLORS = [
    (212, 163, 92),
    (92, 184, 212),
    (212, 92, 122),
    (122, 212, 92),
    (201, 160, 255),
]


def _load_example_specs() -> list[tuple[str, str]]:
    out = [("Demo collision (2D)", json.dumps(DEMO_SPEC, indent=2))]
    if EXAMPLES_DIR.exists():
        for p in sorted(EXAMPLES_DIR.glob("*.json"))[:12]:
            out.append((p.stem, p.read_text(encoding="utf-8")))
    return out


def _bounds(frames: list, dim: int, walls) -> tuple[float, float, float, float]:
    xs, ys = [], []
    for f in frames:
        for p in f["p"]:
            xs.append(float(p[0]))
            ys.append(float(p[1]) if dim > 1 and len(p) > 1 else 0.0)
    if dim == 1:
        ys = [-1.0, 1.0]
    if walls and len(walls) >= 2:
        xs.extend([float(walls[0]), float(walls[1])])
        if dim == 2 and len(walls) >= 4:
            ys.extend([float(walls[2]), float(walls[3])])
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    padx = (xmax - xmin or 1.0) * 0.12
    pady = (ymax - ymin or 1.0) * 0.12
    return xmin - padx, xmax + padx, ymin - pady, ymax + pady


def _blank_image() -> PILImage.Image:
    return PILImage.fromarray(np.full((400, 640, 3), (7, 18, 24), dtype=np.uint8))


def _render_frame(state: dict[str, Any] | None, idx: int) -> PILImage.Image:
    """Draw one trajectory frame to an RGB image (no client JS)."""
    w, h = 640, 400
    img = np.full((h, w, 3), (7, 18, 24), dtype=np.uint8)
    if not state or not state.get("frames"):
        return PILImage.fromarray(img)
    frames = state["frames"]
    dim = int(state["dim"])
    walls = state.get("walls")
    ids = state["ids"]
    idx = int(np.clip(idx, 0, len(frames) - 1))
    xmin, xmax, ymin, ymax = state["bounds"]

    def toxy(p) -> tuple[int, int]:
        x = int((float(p[0]) - xmin) / (xmax - xmin) * (w - 1))
        if dim == 1:
            y = h // 2
        else:
            y = int((h - 1) - (float(p[1]) - ymin) / (ymax - ymin) * (h - 1))
        return x, y

    # walls
    if walls and len(walls) >= 2:
        if dim == 1:
            for wx in (walls[0], walls[1]):
                x, _ = toxy([wx, 0])
                img[20 : h - 20, max(0, min(w - 1, x)), :] = (58, 85, 96)
        elif len(walls) >= 4:
            x0, y0 = toxy([walls[0], walls[2]])
            x1, y1 = toxy([walls[1], walls[3]])
            x_lo, x_hi = sorted((x0, x1))
            y_lo, y_hi = sorted((y0, y1))
            img[y_lo, x_lo : x_hi + 1] = (58, 85, 96)
            img[y_hi, x_lo : x_hi + 1] = (58, 85, 96)
            img[y_lo : y_hi + 1, x_lo] = (58, 85, 96)
            img[y_lo : y_hi + 1, x_hi] = (58, 85, 96)

    for o, oid in enumerate(ids):
        color = COLORS[o % len(COLORS)]
        trail = [toxy(frames[k]["p"][o]) for k in range(idx + 1)]
        for a, b in zip(trail, trail[1:]):
            # crude line
            n = max(abs(b[0] - a[0]), abs(b[1] - a[1]), 1)
            for t in range(n + 1):
                x = int(a[0] + (b[0] - a[0]) * t / n)
                y = int(a[1] + (b[1] - a[1]) * t / n)
                if 0 <= x < w and 0 <= y < h:
                    img[max(0, y - 1) : min(h, y + 2), max(0, x - 1) : min(w, x + 2)] = color
        cx, cy = trail[-1]
        rr = 6
        for dy in range(-rr, rr + 1):
            for dx in range(-rr, rr + 1):
                if dx * dx + dy * dy <= rr * rr:
                    x, y = cx + dx, cy + dy
                    if 0 <= x < w and 0 <= y < h:
                        img[y, x] = color
    return PILImage.fromarray(img)


def _frame_label(state: dict[str, Any] | None, idx: int) -> str:
    if not state or not state.get("frames"):
        return "t=—"
    frames = state["frames"]
    idx = int(np.clip(idx, 0, len(frames) - 1))
    return f"t={frames[idx]['t']:.3f}s · frame {idx}/{len(frames) - 1}"


def _make_state(result: dict, dim: int, walls) -> dict[str, Any]:
    frames = [{"t": float(s["t"]), "p": s["positions"]} for s in result["trajectory"]]
    ids = [o["id"] for o in result["objects"]]
    st = {
        "frames": frames,
        "ids": ids,
        "dim": int(dim),
        "walls": walls,
        "i": 0,
        "playing": False,
    }
    st["bounds"] = _bounds(frames, int(dim), walls)
    return st


def simulate(spec_json: str):
    empty = _blank_image()
    try:
        spec = parse_spec(spec_json)
        result = run(spec, return_trajectory=True)
    except SpecError as e:
        return empty, None, "—", f"Spec error: {e}", RESULTS_MD, gr.update(value=0, maximum=0), gr.update(active=False), "t=—"
    except Exception as e:  # noqa: BLE001
        return empty, None, "—", f"Error: {e}", RESULTS_MD, gr.update(value=0, maximum=0), gr.update(active=False), "t=—"

    walls = None
    try:
        walls = json.loads(spec_json).get("walls")
    except Exception:
        walls = getattr(spec, "walls", None)

    state = _make_state(result, spec.dim, walls)
    lines = [f"t = {result['t']:.4f} s · collisions = {result['n_collisions']}"]
    for o in result["objects"]:
        pos = ", ".join(f"{x:.4f}" for x in o["position"])
        vel = ", ".join(f"{v:.4f}" for v in o["velocity"])
        lines.append(f"{o['id']}:  x=[{pos}]   v=[{vel}]")
    lines.append("")
    lines.append("Base / LoRA LLM compare: adapters still training — oracle above is ground truth.")
    lines.append("Zero-shot ID acc 8% · OOD axes 0% · LoRA eval pending.")
    status = f"OK · dim={spec.dim} · steps={len(result['trajectory']) - 1} · collisions={result['n_collisions']}"
    n = len(state["frames"])
    img = _render_frame(state, 0)
    return (
        img,
        state,
        "\n".join(lines),
        status,
        RESULTS_MD,
        gr.update(value=0, maximum=max(0, n - 1)),
        gr.update(active=False),
        _frame_label(state, 0),
    )


def toggle_play(state: dict[str, Any] | None):
    if not state or not state.get("frames"):
        return state, gr.update(active=False)
    if state.get("playing"):
        state["playing"] = False
        return state, gr.update(active=False)
    if state["i"] >= len(state["frames"]) - 1:
        state["i"] = 0
    state["playing"] = True
    return state, gr.update(active=True)


def on_reset(state: dict[str, Any] | None):
    if not state or not state.get("frames"):
        return _blank_image(), state, gr.update(active=False), 0, "t=—"
    state["i"] = 0
    state["playing"] = False
    return _render_frame(state, 0), state, gr.update(active=False), 0, _frame_label(state, 0)


def on_scrub(state: dict[str, Any] | None, idx: float):
    if not state or not state.get("frames"):
        return _blank_image(), state, gr.update(active=False), "t=—"
    state["i"] = int(idx)
    state["playing"] = False
    return _render_frame(state, state["i"]), state, gr.update(active=False), _frame_label(state, state["i"])


def on_tick(state: dict[str, Any] | None):
    if not state or not state.get("frames") or not state.get("playing"):
        i = state.get("i", 0) if state else 0
        img = _blank_image() if not state else _render_frame(state, i)
        return img, state, gr.update(active=False), i, _frame_label(state, i)
    n = len(state["frames"])
    state["i"] = min(state["i"] + 1, n - 1)
    done = state["i"] >= n - 1
    if done:
        state["playing"] = False
    return (
        _render_frame(state, state["i"]),
        state,
        gr.update(active=not done),
        state["i"],
        _frame_label(state, state["i"]),
    )


def build() -> gr.Blocks:
    examples = _load_example_specs()
    css = """
    .gradio-container { max-width: 980px !important; }
    """
    with gr.Blocks(title="PhiLLM — results + simulator", css=css) as demo:
        results_box = gr.Markdown(RESULTS_MD)
        gr.Markdown("## Exact simulator · 2D / 1D animator")
        anim_state = gr.State(None)
        with gr.Row():
            with gr.Column(scale=1):
                example_dd = gr.Dropdown(
                    choices=[e[0] for e in examples],
                    value=examples[0][0],
                    label="Preset",
                )
                inp = gr.Textbox(label="WorldSpec JSON", value=examples[0][1], lines=14)
                btn = gr.Button("Run simulation", variant="primary")
                status = gr.Textbox(label="Status", interactive=False)
            with gr.Column(scale=1):
                anim = gr.Image(label="Trajectory", interactive=False, type="pil", height=400)
                with gr.Row():
                    play_btn = gr.Button("Play", variant="primary")
                    reset_btn = gr.Button("Reset")
                scrub = gr.Slider(0, 40, value=0, step=1, label="Scrub frame")
                frame_lbl = gr.Markdown("t=—")
                nums = gr.Textbox(label="Final state (numbers)", lines=8)
                timer = gr.Timer(0.08, active=False)

        def on_example(name: str):
            for n, body in examples:
                if n == name:
                    return body
            return examples[0][1]

        example_dd.change(on_example, inputs=[example_dd], outputs=[inp])

        sim_outs = [anim, anim_state, nums, status, results_box, scrub, timer, frame_lbl]
        btn.click(simulate, inputs=[inp], outputs=sim_outs)
        demo.load(simulate, inputs=[inp], outputs=sim_outs)

        play_btn.click(toggle_play, inputs=[anim_state], outputs=[anim_state, timer])
        reset_btn.click(on_reset, inputs=[anim_state], outputs=[anim, anim_state, timer, scrub, frame_lbl])
        # release only — scrub.change would fight the timer while playing
        scrub.release(on_scrub, inputs=[anim_state, scrub], outputs=[anim, anim_state, timer, frame_lbl])
        timer.tick(on_tick, inputs=[anim_state], outputs=[anim, anim_state, timer, scrub, frame_lbl])

        with gr.Accordion("Raw JSON final state / notes", open=False):
            gr.Markdown(
                "Play/Pause/scrub are Gradio-native (no client JS) so they work on Spaces. "
                "Model predictions will appear once the LoRA adapter is pushed."
            )
    return demo


if __name__ == "__main__":
    build().launch()
