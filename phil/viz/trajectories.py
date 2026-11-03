"""Trajectory and scene plots for PhiLLM examples."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from phil.sim.simulator import run
from phil.sim.spec import WorldSpec, parse_spec
from phil.utils.io import ensure_dir, read_jsonl


def trajectory_from_spec(spec: dict[str, Any] | WorldSpec) -> dict[str, Any]:
    if not isinstance(spec, WorldSpec):
        spec = parse_spec(spec)
    return run(spec, return_trajectory=True)


def plot_trajectory(
    spec: dict[str, Any],
    out_path: str | Path,
    *,
    title: str | None = None,
    pred_answer: dict[str, Any] | None = None,
    gold_answer: dict[str, Any] | None = None,
) -> Path:
    """Plot 1D time series or 2D paths for a gold (and optional predicted) trajectory."""
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    result = trajectory_from_spec(spec)
    traj = result["trajectory"]
    dim = int(spec["dim"])
    ids = [o["id"] for o in result["objects"]]
    t = np.array([s["t"] for s in traj])

    fig, axes = plt.subplots(1, 2 if dim == 1 else 1, figsize=(10 if dim == 1 else 6, 4))
    if dim == 1:
        ax_x, ax_v = axes
        for i, oid in enumerate(ids):
            xs = [s["positions"][i][0] for s in traj]
            vs = [s["velocities"][i][0] for s in traj]
            ax_x.plot(t, xs, marker="o", ms=3, label=f"{oid} x")
            ax_v.plot(t, vs, marker="o", ms=3, label=f"{oid} v")
        if pred_answer and "objects" in pred_answer:
            for o in pred_answer["objects"]:
                ax_x.scatter([t[-1]], [o["position"][0]], marker="x", s=80, label=f"pred {o['id']} x")
        if gold_answer and "objects" in gold_answer:
            for o in gold_answer["objects"]:
                ax_x.scatter([t[-1]], [o["position"][0]], marker="*", s=100, label=f"gold {o['id']} x")
        ax_x.set_xlabel("t (s)")
        ax_x.set_ylabel("position (m)")
        ax_v.set_xlabel("t (s)")
        ax_v.set_ylabel("velocity (m/s)")
        ax_x.legend(fontsize=8)
        ax_v.legend(fontsize=8)
        ax_x.grid(True, alpha=0.3)
        ax_v.grid(True, alpha=0.3)
    else:
        ax = axes
        for i, oid in enumerate(ids):
            xs = [s["positions"][i][0] for s in traj]
            ys = [s["positions"][i][1] for s in traj]
            ax.plot(xs, ys, marker="o", ms=3, label=oid)
            ax.scatter([xs[0]], [ys[0]], marker="s", s=40)
            ax.scatter([xs[-1]], [ys[-1]], marker="*", s=80)
        if spec.get("walls") and len(spec["walls"]) == 4:
            xmin, xmax, ymin, ymax = spec["walls"]
            ax.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], "k--", lw=1)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title or f"horizon={spec.get('horizon')} n_obj={len(ids)}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_rollout_errors(errors: list[float], out_path: str | Path, title: str = "Rollout error growth") -> Path:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(range(1, len(errors) + 1), errors, marker="o")
    ax.set_xlabel("step")
    ax.set_ylabel("position MAE")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def render_dataset_gallery(
    data_paths: dict[str, str],
    out_dir: str | Path,
    *,
    n_per_split: int = 3,
) -> list[str]:
    """Render a few T1 examples from each split into PNG gallery."""
    out_dir = Path(out_dir)
    ensure_dir(out_dir)
    written: list[str] = []
    for split, path in data_paths.items():
        rows = [r for r in read_jsonl(path) if r.get("task_type") == "T1"][:n_per_split]
        for j, ex in enumerate(rows):
            p = out_dir / f"{split}_{j}_{ex['id'][:16]}.png"
            plot_trajectory(
                ex["gold_spec"],
                p,
                title=f"{split} · {ex['id'][:24]} · k={ex['tags'].get('horizon')}",
                gold_answer=ex.get("gold_answer"),
            )
            written.append(str(p))
    return written
