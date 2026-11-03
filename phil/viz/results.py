"""Quantitative and qualitative result figures for PhiLLM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from phil.utils.io import ensure_dir, read_json, read_jsonl


def _load_metrics(path: str | Path) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        return None
    return read_json(p)


def plot_id_ood_bars(
    metrics_by_system_split: dict[str, dict[str, dict[str, Any]]],
    out_path: str | Path,
    metric_key: str = "accuracy",
) -> Path:
    """Bar chart: systems x splits for accuracy or MAE."""
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    splits = sorted({s for m in metrics_by_system_split.values() for s in m})
    systems = list(metrics_by_system_split.keys())
    x = np.arange(len(splits))
    width = 0.8 / max(len(systems), 1)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, sys in enumerate(systems):
        vals = []
        for sp in splits:
            m = metrics_by_system_split[sys].get(sp) or {}
            v = m.get(metric_key)
            vals.append(float(v) if v is not None else np.nan)
        ax.bar(x + i * width, vals, width, label=sys)
    ax.set_xticks(x + width * (len(systems) - 1) / 2)
    ax.set_xticklabels(splits, rotation=20)
    ax.set_ylabel(metric_key)
    ax.set_title(f"ID / OOD {metric_key}")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_probe_selectivity(probe_json: str | Path, out_path: str | Path, title: str = "Probe R²") -> Path:
    data = read_json(probe_json)
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    derived = data.get("derived") or data.get("layer0_vs_best") or {}
    if "derived" in data:
        for key, layers in data["derived"].items():
            xs = [d["layer"] for d in layers]
            ys = [d["r2_mean"] for d in layers]
            ax.plot(xs, ys, marker="o", label=key)
    elif isinstance(derived, dict):
        for key, v in derived.items():
            ax.scatter([v.get("best_layer", 0)], [v.get("best_r2", 0)], label=key)
    ax.set_xlabel("layer")
    ax.set_ylabel("R²")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_utility_curves(
    points: list[dict[str, float]],
    out_path: str | Path,
    title: str = "Accuracy vs call rate",
) -> Path:
    """points: [{call_rate, accuracy, label}]"""
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(6, 4))
    for p in points:
        ax.scatter([p["call_rate"]], [p["accuracy"]], s=60, label=p.get("label", ""))
    ax.set_xlabel("call rate")
    ax.set_ylabel("accuracy")
    ax.set_title(title)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def write_qualitative_examples(
    gold_path: str,
    pred_path: str,
    out_md: str | Path,
    *,
    n_ok: int = 3,
    n_fail: int = 3,
) -> Path:
    from phil.eval.eval import score_example

    gold = {r["id"]: r for r in read_jsonl(gold_path)}
    preds = read_jsonl(pred_path)
    scored = []
    for pr in preds:
        g = gold.get(pr["id"])
        if not g:
            continue
        sc = score_example(g, pr)
        scored.append((sc, g, pr))
    ok = [x for x in scored if x[0]["correct"]][:n_ok]
    fail = [x for x in scored if not x[0]["correct"]][:n_fail]
    lines = ["# Qualitative examples\n"]
    for label, items in [("Correct", ok), ("Failures", fail)]:
        lines.append(f"## {label}\n")
        for sc, g, pr in items:
            lines.append(f"### `{g['id']}` ({g['task_type']})\n")
            lines.append("**Prompt (truncated):**\n```\n" + g["text"][:600] + "\n```\n")
            lines.append("**Gold:**\n```\n" + json.dumps(g["gold_answer"])[:400] + "\n```\n")
            lines.append("**Pred:**\n```\n" + str(pr.get("final_answer"))[:400] + "\n```\n")
    out_md = Path(out_md)
    ensure_dir(out_md.parent)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    return out_md


def collect_metrics_tree(preds_dir: str | Path) -> dict[str, dict[str, dict[str, Any]]]:
    """Scan preds/*metrics.json into {system: {split: metrics}}."""
    preds_dir = Path(preds_dir)
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for p in preds_dir.glob("*.metrics.json"):
        name = p.name.replace(".metrics.json", "")
        # e.g. zeroshot_Qwen3_5_2B_test, lora_ood_mag, oracle_test
        parts = name.split("_")
        split = parts[-1]
        system = "_".join(parts[:-1]) or name
        out.setdefault(system, {})[split] = read_json(p)
    return out
