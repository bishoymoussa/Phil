"""PhiLLM Gradio Space: results-first summary + HTML5 2D/1D animator."""

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

import base64
import html
import json
import sys
from pathlib import Path

import gradio as gr

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

# Embedded snapshot — keep in sync with docs/RESULTS.md / preds/*.metrics.json
RESULTS_MD = """
# PhiLLM — Physics + LLMs

**Takeaway:** Zero-shot **Qwen3.5-2B** mostly **cannot** predict Newtonian next states —
**8% ID accuracy** vs a **100% oracle**, and **0%** on all four OOD axes
(partial eval, n=50; LoRA still training).

| System | ID acc | Pos MAE | Status |
|--------|--------|---------|--------|
| Oracle | **100%** | 0.00 | final · n=2000 |
| Zero-shot 2B | **8%** | 13.50 | partial · n=50 |
| LoRA SFT | *pending* | — | ~49% through training |

| OOD axis | Acc | Pos MAE | Status |
|----------|-----|---------|--------|
| `ood_mag` | **0%** | 64.7 | partial |
| `ood_hor` | **0%** | 38.4 | partial |
| `ood_obj` | **0%** | 31.8 | partial |
| `ood_col` | **0%** | 29.8 | partial |

**RQ3:** LLM-only **8%** · always-call oracle **100%** (U₀.₁=0.90) · random@0.5 **50%±1.3%**. Probes/weight-read in flight; LoRA ~59%.

[Paper site](https://bishoymoussa.github.io/Phil/) ·
[GitHub](https://github.com/bishoymoussa/Phil) ·
[Dataset](https://huggingface.co/datasets/bishoygaloaa/phillm-data) ·
[Raw metrics](https://huggingface.co/datasets/bishoygaloaa/phillm-results)
"""


def _load_example_specs() -> list[tuple[str, str]]:
    out = [("Demo collision (2D)", json.dumps(DEMO_SPEC, indent=2))]
    if EXAMPLES_DIR.exists():
        for p in sorted(EXAMPLES_DIR.glob("*.json"))[:12]:
            out.append((p.stem, p.read_text(encoding="utf-8")))
    return out


def _animator_html(traj: list, objects: list, dim: int, walls) -> str:
    """HTML5 canvas animator in an iframe.

    Gradio's ``gr.HTML`` strips ``<script>`` tags, so inline JS never runs and
    Play/Pause become no-ops. A data-URL iframe keeps the script executable.
    """
    payload = {
        "dim": int(dim),
        "walls": walls,
        "ids": [o["id"] for o in objects],
        "frames": [{"t": float(s["t"]), "p": s["positions"]} for s in traj],
    }
    data_js = json.dumps(payload)
    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<style>
  html,body {{ margin:0; background:#071218; color:#e8f1f4;
    font-family: system-ui, -apple-system, Segoe UI, sans-serif; }}
  canvas {{ width:100%; max-width:640px; display:block; background:#071218;
    border:1px solid rgba(232,241,244,0.12); }}
  .row {{ display:flex; gap:8px; align-items:center; margin-top:8px; flex-wrap:wrap; }}
  button {{ cursor:pointer; padding:6px 12px; font-weight:700; border:0; }}
  #play {{ background:#d4a35c; color:#0b1c24; }}
  #reset {{ background:transparent; color:#e8f1f4; border:1px solid rgba(232,241,244,0.2); }}
  #scrub {{ flex:1; min-width:120px; }}
  #label {{ color:#9db4bd; font-size:0.9rem; }}
</style></head><body>
<canvas id="c" width="640" height="400"></canvas>
<div class="row">
  <button type="button" id="play">Play</button>
  <button type="button" id="reset">Reset</button>
  <input type="range" id="scrub" min="0" max="0" value="0"/>
  <span id="label">t=0</span>
</div>
<script>
const data = {data_js};
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
const playBtn = document.getElementById('play');
const resetBtn = document.getElementById('reset');
const scrub = document.getElementById('scrub');
const label = document.getElementById('label');
const colors = ['#d4a35c','#5cb8d4','#d45c7a','#7ad45c','#c9a0ff'];
scrub.max = Math.max(0, data.frames.length - 1);
let i = 0, playing = false, timer = null;
function bounds() {{
  let xmin=Infinity,xmax=-Infinity,ymin=Infinity,ymax=-Infinity;
  for (const f of data.frames) {{
    for (const p of f.p) {{
      xmin=Math.min(xmin,p[0]); xmax=Math.max(xmax,p[0]);
      if (data.dim===1) {{ ymin=Math.min(ymin,-1); ymax=Math.max(ymax,1); }}
      else {{ ymin=Math.min(ymin,p[1]||0); ymax=Math.max(ymax,p[1]||0); }}
    }}
  }}
  if (data.walls && data.walls.length>=2) {{
    xmin=Math.min(xmin,data.walls[0]); xmax=Math.max(xmax,data.walls[1]);
    if (data.dim===2 && data.walls.length>=4) {{
      ymin=Math.min(ymin,data.walls[2]); ymax=Math.max(ymax,data.walls[3]);
    }}
  }}
  const padx=(xmax-xmin||1)*0.12, pady=(ymax-ymin||1)*0.12;
  return [xmin-padx,xmax+padx,ymin-pady,ymax+pady];
}}
const B = bounds();
function toxy(p) {{
  const x = (p[0]-B[0])/(B[1]-B[0]) * canvas.width;
  if (data.dim===1) return [x, canvas.height*0.5];
  const y = canvas.height - (p[1]-B[2])/(B[3]-B[2]) * canvas.height;
  return [x,y];
}}
function draw(frameIdx) {{
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.fillStyle='#071218'; ctx.fillRect(0,0,canvas.width,canvas.height);
  if (data.walls) {{
    ctx.strokeStyle='#3a5560'; ctx.setLineDash([6,4]); ctx.lineWidth=1;
    if (data.dim===1) {{
      const x0=toxy([data.walls[0],0])[0], x1=toxy([data.walls[1],0])[0];
      ctx.beginPath(); ctx.moveTo(x0,20); ctx.lineTo(x0,canvas.height-20); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(x1,20); ctx.lineTo(x1,canvas.height-20); ctx.stroke();
    }} else if (data.walls.length>=4) {{
      const a=toxy([data.walls[0],data.walls[2]]);
      const b=toxy([data.walls[1],data.walls[3]]);
      ctx.strokeRect(a[0],b[1],b[0]-a[0],a[1]-b[1]);
    }}
    ctx.setLineDash([]);
  }}
  for (let o=0; o<data.ids.length; o++) {{
    ctx.strokeStyle=colors[o%colors.length]; ctx.fillStyle=colors[o%colors.length];
    ctx.lineWidth=2; ctx.beginPath();
    for (let k=0; k<=frameIdx; k++) {{
      const xy=toxy(data.frames[k].p[o]);
      if (k===0) ctx.moveTo(xy[0],xy[1]); else ctx.lineTo(xy[0],xy[1]);
    }}
    ctx.stroke();
    const cxy=toxy(data.frames[frameIdx].p[o]);
    ctx.beginPath(); ctx.arc(cxy[0],cxy[1],7,0,Math.PI*2); ctx.fill();
    ctx.fillStyle='#9db4bd'; ctx.font='12px sans-serif';
    ctx.fillText(data.ids[o], cxy[0]+10, cxy[1]-8);
    ctx.fillStyle=colors[o%colors.length];
  }}
  label.textContent = 't=' + data.frames[frameIdx].t.toFixed(3) + 's · frame ' + frameIdx + '/' + (data.frames.length-1);
  scrub.value = String(frameIdx);
}}
function stop() {{
  playing=false; playBtn.textContent='Play';
  if (timer) {{ clearInterval(timer); timer=null; }}
}}
function start() {{
  if (playing) return;
  if (i >= data.frames.length-1) i = 0;
  playing=true; playBtn.textContent='Pause';
  timer=setInterval(function() {{
    i = Math.min(i+1, data.frames.length-1);
    draw(i);
    if (i >= data.frames.length-1) stop();
  }}, 80);
}}
playBtn.addEventListener('click', function() {{ playing ? stop() : start(); }});
resetBtn.addEventListener('click', function() {{ stop(); i=0; draw(0); }});
scrub.addEventListener('input', function() {{ stop(); i=+scrub.value; draw(i); }});
draw(0);
</script></body></html>"""
    b64 = base64.b64encode(page.encode("utf-8")).decode("ascii")
    return (
        f'<iframe title="PhiLLM trajectory animator" '
        f'src="data:text/html;base64,{b64}" '
        f'sandbox="allow-scripts" '
        f'style="width:100%;height:480px;border:0;background:#071218;border-radius:2px;"></iframe>'
    )


def simulate(spec_json: str):
    try:
        spec = parse_spec(spec_json)
        result = run(spec, return_trajectory=True)
    except SpecError as e:
        return (
            f"<p style='color:#d45c7a'>Spec error: {html.escape(str(e))}</p>",
            "—",
            "Spec error — see animator panel.",
            RESULTS_MD,
        )
    except Exception as e:  # noqa: BLE001
        return (
            f"<p style='color:#d45c7a'>Error: {html.escape(str(e))}</p>",
            "—",
            f"Error: {e}",
            RESULTS_MD,
        )

    anim = _animator_html(
        result["trajectory"],
        result["objects"],
        spec.dim,
        getattr(spec, "walls", None) or (result.get("walls")),
    )
    # Prefer walls from original JSON if present
    try:
        raw = json.loads(spec_json)
        walls = raw.get("walls")
        if walls is not None:
            anim = _animator_html(result["trajectory"], result["objects"], spec.dim, walls)
    except Exception:
        pass

    final = {
        "t": result["t"],
        "n_collisions": result["n_collisions"],
        "objects": [
            {"id": o["id"], "position": o["position"], "velocity": o["velocity"]}
            for o in result["objects"]
        ],
    }
    # Compact numeric readout (not a wall of JSON)
    lines = [f"t = {result['t']:.4f} s · collisions = {result['n_collisions']}"]
    for o in result["objects"]:
        pos = ", ".join(f"{x:.4f}" for x in o["position"])
        vel = ", ".join(f"{v:.4f}" for v in o["velocity"])
        lines.append(f"{o['id']}:  x=[{pos}]   v=[{vel}]")
    lines.append("")
    lines.append("Base / LoRA LLM compare: adapters still training — oracle above is ground truth.")
    lines.append("Zero-shot ID acc 8% · OOD mag/hor/obj 0% · LoRA eval pending.")
    status = f"OK · dim={spec.dim} · steps={len(result['trajectory'])-1} · collisions={result['n_collisions']}"
    return anim, "\n".join(lines), status, RESULTS_MD


def build() -> gr.Blocks:
    examples = _load_example_specs()
    css = """
    .gradio-container { max-width: 980px !important; }
    """
    with gr.Blocks(title="PhiLLM — results + simulator", css=css) as demo:
        results_box = gr.Markdown(RESULTS_MD)
        gr.Markdown("## Exact simulator · 2D / 1D animator")
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
                anim = gr.HTML(label="Animator")
                nums = gr.Textbox(label="Final state (numbers)", lines=8)

        def on_example(name: str):
            for n, body in examples:
                if n == name:
                    return body
            return examples[0][1]

        example_dd.change(on_example, inputs=[example_dd], outputs=[inp])
        btn.click(simulate, inputs=[inp], outputs=[anim, nums, status, results_box])
        # Auto-run demo on load
        demo.load(simulate, inputs=[inp], outputs=[anim, nums, status, results_box])

        with gr.Accordion("Raw JSON final state / notes", open=False):
            gr.Markdown(
                "Full trajectory JSON is intentionally secondary — "
                "use the animator and numeric readout above. "
                "Model predictions will appear here once the LoRA adapter is pushed."
            )
    return demo


if __name__ == "__main__":
    build().launch()
