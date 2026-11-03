"""Metric primitives for PhiLLM evaluation."""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np


def relative_error(pred: float, gold: float, eps: float = 1e-8) -> float:
    denom = max(abs(gold), eps)
    return abs(pred - gold) / denom


def parse_numeric_answer(text: str) -> float | None:
    if text is None:
        return None
    s = str(text).strip()
    # YES/NO
    if s.upper() in ("YES", "NO"):
        return None
    # try JSON number or last number in text
    try:
        return float(s)
    except ValueError:
        pass
    nums = re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", s)
    if not nums:
        return None
    try:
        return float(nums[-1])
    except ValueError:
        return None


def parse_yes_no(text: str) -> str | None:
    if text is None:
        return None
    s = str(text).strip().upper()
    if s.startswith("YES"):
        return "YES"
    if s.startswith("NO"):
        return "NO"
    m = re.search(r"\b(YES|NO)\b", s)
    return m.group(1) if m else None


def parse_t1_answer(raw: Any) -> dict[str, Any] | None:
    """Extract {objects: [{id, position, velocity}]} from model output or dict."""
    import json

    if raw is None:
        return None
    if isinstance(raw, dict):
        if "objects" in raw:
            return raw
        return None
    s = str(raw)
    # find JSON object
    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(s[start : end + 1])
            if isinstance(obj, dict) and "objects" in obj:
                return obj
        except json.JSONDecodeError:
            pass
    return None


def _as_1d_list(x: Any) -> list[float]:
    """Normalize scalars / nested lists to a flat float list."""
    arr = np.asarray(x, dtype=float).reshape(-1)
    return arr.tolist()


def t1_mae(pred: dict[str, Any] | None, gold: dict[str, Any]) -> dict[str, float]:
    if pred is None or "objects" not in pred:
        return {"pos_mae": float("nan"), "vel_mae": float("nan"), "ok": 0.0}
    gold_map = {o["id"]: o for o in gold["objects"]}
    pos_errs: list[float] = []
    vel_errs: list[float] = []
    for o in pred["objects"]:
        g = gold_map.get(o["id"])
        if g is None:
            continue
        try:
            p = np.asarray(_as_1d_list(o["position"]), dtype=float)
            gp = np.asarray(_as_1d_list(g["position"]), dtype=float)
            v = np.asarray(_as_1d_list(o["velocity"]), dtype=float)
            gv = np.asarray(_as_1d_list(g["velocity"]), dtype=float)
        except (KeyError, TypeError, ValueError):
            continue
        if p.shape != gp.shape or v.shape != gv.shape:
            continue
        pos_errs.extend(np.abs(p - gp).tolist())
        vel_errs.extend(np.abs(v - gv).tolist())
    if not pos_errs:
        return {"pos_mae": float("nan"), "vel_mae": float("nan"), "ok": 0.0}
    return {
        "pos_mae": float(np.mean(pos_errs)),
        "vel_mae": float(np.mean(vel_errs)),
        "ok": 1.0,
    }


def t1_tolerance_correct(pred: dict[str, Any] | None, gold: dict[str, Any], tol: float = 0.05) -> bool:
    if pred is None or "objects" not in pred:
        return False
    gold_map = {o["id"]: o for o in gold["objects"]}
    if len(pred["objects"]) != len(gold["objects"]):
        return False
    for o in pred["objects"]:
        g = gold_map.get(o["id"])
        if g is None:
            return False
        try:
            p = _as_1d_list(o["position"])
            gp = _as_1d_list(g["position"])
            v = _as_1d_list(o["velocity"])
            gv = _as_1d_list(g["velocity"])
        except (KeyError, TypeError, ValueError):
            return False
        if len(p) != len(gp) or len(v) != len(gv):
            return False
        for a, b in zip(p, gp):
            if relative_error(a, b) > tol and abs(a - b) > 1e-3:
                return False
        for a, b in zip(v, gv):
            if relative_error(a, b) > tol and abs(a - b) > 1e-3:
                return False
    return True


def physical_violation_elastic(
    pred: dict[str, Any] | None,
    init_spec: dict[str, Any],
    energy_tol: float = 1e-2,
    mom_tol: float = 1e-2,
) -> bool:
    """Heuristic: for force-free multi-object elastic scenes, check KE/momentum roughly.

    Returns True if a violation is detected.
    """
    if pred is None or "objects" not in pred:
        return True
    # only meaningful when forces ~ 0 and no walls
    if init_spec.get("walls"):
        return False
    forces = [o.get("force") or [0.0] * init_spec["dim"] for o in init_spec["objects"]]
    if any(abs(fi) > 1e-8 for f in forces for fi in f):
        return False
    m0 = {o["id"]: float(o["mass"]) for o in init_spec["objects"]}
    v0 = {o["id"]: np.asarray(o["velocity"], dtype=float) for o in init_spec["objects"]}
    ke0 = 0.5 * sum(m0[i] * float(np.dot(v0[i], v0[i])) for i in m0)
    p0 = sum(m0[i] * v0[i] for i in m0)
    try:
        ke1 = 0.0
        p1 = np.zeros_like(p0)
        for o in pred["objects"]:
            m = m0[o["id"]]
            v = np.asarray(o["velocity"], dtype=float)
            ke1 += 0.5 * m * float(np.dot(v, v))
            p1 = p1 + m * v
    except Exception:
        return True
    if ke0 > 1e-8 and abs(ke1 - ke0) / ke0 > energy_tol:
        return True
    if np.linalg.norm(p0) > 1e-8 and np.linalg.norm(p1 - p0) / (np.linalg.norm(p0) + 1e-8) > mom_tol:
        return True
    return False


def bootstrap_ci(values: list[float], n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    means = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(float(np.mean(sample)))
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return float(np.mean(arr)), lo, hi


def auroc(scores: list[float], labels: list[int]) -> float:
    """Compute AUROC; labels 1 = positive (e.g. should CALL)."""
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # Mann-Whitney
    rank_sum = 0.0
    for i, (_, y) in enumerate(pairs, start=1):
        if y == 1:
            rank_sum += i
    u = rank_sum - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def utility(acc: float, call_rate: float, lam: float) -> float:
    return acc - lam * call_rate
