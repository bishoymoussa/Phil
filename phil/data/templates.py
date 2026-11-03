"""Natural-language templates with paraphrases and mixed units."""

from __future__ import annotations

import random
from typing import Any


MASS_UNITS = [
    ("kg", 1.0),
    ("g", 0.001),
]

VEL_UNITS = [
    ("m/s", 1.0),
    ("cm/s", 0.01),
]

POS_UNITS = [
    ("m", 1.0),
    ("cm", 0.01),
]

FORCE_UNITS = [
    ("N", 1.0),
]


def _fmt(x: float, digits: int = 4) -> str:
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.{digits}g}"


def with_unit(value_si: float, units: list[tuple[str, float]], rng: random.Random) -> tuple[str, float]:
    """Return (display_string_with_unit, si_value) choosing a random unit."""
    name, scale = rng.choice(units)
    display = value_si / scale
    return f"{_fmt(display)} {name}", value_si


T1_OPENERS = [
    "Consider the following physical system.",
    "Here is a Newtonian point-mass scene.",
    "You are given this physics setup.",
    "Simulate the evolution of the system below.",
]

T2_OPENERS = [
    "Infer the unknown quantity from the observed trajectory.",
    "An unknown parameter must be recovered from the data below.",
    "Use the observed motion to find the missing value.",
]

T3_OPENERS = [
    "Decide whether the proposed outcome is physically possible.",
    "Is the following claim consistent with Newtonian mechanics?",
    "Answer yes or no: is this outcome physically feasible?",
]


def describe_object(o: dict[str, Any], dim: int, rng: random.Random) -> str:
    mass_s, _ = with_unit(o["mass"], MASS_UNITS, rng)
    if dim == 1:
        x_s, _ = with_unit(o["position"][0], POS_UNITS, rng)
        v_s, _ = with_unit(o["velocity"][0], VEL_UNITS, rng)
        f_s, _ = with_unit(o["force"][0], FORCE_UNITS, rng)
        return (
            f"Object {o['id']} has mass {mass_s}, position {x_s}, "
            f"velocity {v_s}, and constant force {f_s}."
        )
    x_s = ", ".join(_fmt(x) for x in o["position"])
    v_s = ", ".join(_fmt(v) for v in o["velocity"])
    f_s = ", ".join(_fmt(f) for f in o["force"])
    # keep SI for 2D to avoid messy unit mixes in vectors
    return (
        f"Object {o['id']} has mass {mass_s}, position [{x_s}] m, "
        f"velocity [{v_s}] m/s, and force [{f_s}] N."
    )


def render_t1(example: dict[str, Any], rng: random.Random) -> str:
    spec = example["gold_spec"]
    dim = spec["dim"]
    dt = spec["dt"]
    k = spec["horizon"]
    lines = [rng.choice(T1_OPENERS), f"Time step Δt = {_fmt(dt)} s. Predict the state after {k} steps."]
    for o in spec["objects"]:
        lines.append(describe_object(o, dim, rng))
    if spec.get("walls"):
        lines.append(f"Walls at {spec['walls']}.")
    if len(spec["objects"]) > 1 and spec.get("collisions", True):
        lines.append("Objects collide elastically when they meet.")
    lines.append(
        "Return JSON with keys objects (list of {id, position, velocity}), "
        "using SI units (m, m/s)."
    )
    return "\n".join(lines)


def render_t2(example: dict[str, Any], rng: random.Random) -> str:
    """Mass-from-trajectory style inverse problem."""
    spec = example["gold_spec"]
    target = example["meta"]["unknown"]
    lines = [rng.choice(T2_OPENERS)]
    lines.append(f"Time step Δt = {_fmt(spec['dt'])} s, horizon = {spec['horizon']}.")
    for o in spec["objects"]:
        if o["id"] == target["object_id"] and target["field"] == "mass":
            # hide mass
            o_hide = dict(o)
            o_hide = {**o, "mass": None}
            dim = spec["dim"]
            if dim == 1:
                x_s, _ = with_unit(o["position"][0], POS_UNITS, rng)
                v_s, _ = with_unit(o["velocity"][0], VEL_UNITS, rng)
                f_s, _ = with_unit(o["force"][0], FORCE_UNITS, rng)
                lines.append(
                    f"Object {o['id']} has unknown mass, position {x_s}, "
                    f"velocity {v_s}, constant force {f_s}."
                )
            else:
                lines.append(describe_object({**o, "mass": float("nan")}, dim, rng).replace("nan kg", "unknown mass"))
        else:
            lines.append(describe_object(o, spec["dim"], rng))
    obs = example["meta"]["observation"]
    lines.append(f"After {spec['horizon']} steps the observed state is: {obs}.")
    lines.append(f"What is the mass of object {target['object_id']} in kg? Reply with a single number.")
    return "\n".join(lines)


def render_t3(example: dict[str, Any], rng: random.Random) -> str:
    spec = example["gold_spec"]
    claim = example["meta"]["claim"]
    lines = [rng.choice(T3_OPENERS)]
    for o in spec["objects"]:
        lines.append(describe_object(o, spec["dim"], rng))
    lines.append(f"Claim: {claim}")
    lines.append("Answer with exactly YES or NO.")
    return "\n".join(lines)


def render_text(example: dict[str, Any], rng: random.Random | None = None) -> str:
    rng = rng or random.Random(0)
    t = example["task_type"]
    if t == "T1":
        return render_t1(example, rng)
    if t == "T2":
        return render_t2(example, rng)
    if t == "T3":
        return render_t3(example, rng)
    raise ValueError(f"unknown task_type {t}")
