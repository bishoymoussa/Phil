"""Serialize tiny network weights as text for LLM weight-reading prompts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def format_array(name: str, arr: np.ndarray, precision: int = 4) -> str:
    flat = np.asarray(arr, dtype=float)
    if flat.ndim == 1:
        body = ", ".join(f"{x:.{precision}g}" for x in flat.tolist())
        return f"{name} = [{body}]  # shape {flat.shape}"
    lines = [f"{name} = [  # shape {flat.shape}"]
    for row in flat:
        body = ", ".join(f"{x:.{precision}g}" for x in row.tolist())
        lines.append(f"  [{body}],")
    lines.append("]")
    return "\n".join(lines)


def weights_to_text(path: str | Path, precision: int = 4) -> str:
    data = np.load(path)
    parts = ["# Neural one-step simulator weights (numpy)", f"# file: {Path(path).name}"]
    for k in sorted(data.files):
        parts.append(format_array(k, data[k], precision=precision))
    parts.append(f"# total_params = {sum(data[k].size for k in data.files)}")
    return "\n".join(parts)


def describe_io() -> str:
    return (
        "Architecture notes:\n"
        "- Input X = [x, v, p] (position, velocity, law parameter).\n"
        "- Output Y = [x_next, v_next] for one Euler step with dt=0.1, mass=1.\n"
        "- If keys are W,b: Y = X @ W + b (linear).\n"
        "- If keys are W1,b1,W2,b2: h=tanh(X@W1+b1); Y=h@W2+b2 (MLP).\n"
        "Possible laws: constant_force (p=F, a=F), hooke (p=k, a=-k*x), "
        "linear_drag (p=c, a=-c*v), inv_square (p=G, a=-G*sign(x)/(abs(x)+0.2)^2)."
    )


def build_read_prompt(weights_text: str) -> str:
    return (
        "You are analyzing neural network weights that implement a one-step physics update.\n"
        f"{describe_io()}\n\n"
        "Here are the raw weights:\n"
        "```\n"
        f"{weights_text}\n"
        "```\n\n"
        "Without running the network on data, answer:\n"
        "1) Which physical law does this network encode? "
        "Choose exactly one of: constant_force, hooke, linear_drag, inv_square, unknown.\n"
        "2) What is the approximate value of its parameter (F, k, c, or G)?\n"
        "3) Brief reasoning about which weight entries reveal the law.\n"
        "Final line format: LAW=<name> PARAM=<number>\n"
    )
