"""Control tasks (Hewitt & Liang) and decorrelated probing set."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from phil.probing.probes import DERIVED, STATED, fit_probe, r2_score
from phil.utils.io import write_json


def control_task_selectivity(
    hidden: np.ndarray,
    y: np.ndarray,
    keys: list[str],
    *,
    seeds: tuple[int, ...] = (0, 1, 2),
) -> dict[str, Any]:
    """For each target, train on shuffled labels (control) and compute selectivity."""
    key_to_idx = {k: i for i, k in enumerate(keys)}
    n, n_layers, _ = hidden.shape
    out: dict[str, Any] = {}
    for k in list(STATED | DERIVED):
        if k not in key_to_idx:
            continue
        yi = y[:, key_to_idx[k]]
        mask = np.isfinite(yi)
        # use middle layer
        mid = n_layers // 2
        X = hidden[mask, mid, :]
        yy = yi[mask]
        task_r2, ctrl_r2 = [], []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            task_r2.append(fit_probe(X, yy, seed=seed)["r2"])
            y_ctrl = yy.copy()
            rng.shuffle(y_ctrl)
            ctrl_r2.append(fit_probe(X, y_ctrl, seed=seed)["r2"])
        out[k] = {
            "layer": mid,
            "r2_task": float(np.nanmean(task_r2)),
            "r2_control": float(np.nanmean(ctrl_r2)),
            "selectivity": float(np.nanmean(task_r2) - np.nanmean(ctrl_r2)),
        }
    return out


def make_decorrelated_set(
    n: int = 2000,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Sample F, m so a=F/m is uncorrelated with F and with m (rejection / transform).

    Method: sample a and m independently, set F = a*m. Then corr(a,m)=0 by construction,
    corr(a,F) may be nonzero; we residualize / also sample log-uniform.
    """
    rng = np.random.default_rng(seed)
    # sample a and m independently
    m = rng.uniform(1.0, 10.0, size=n)
    a = rng.uniform(-5.0, 5.0, size=n)
    f = a * m
    # also provide x0,v0 independent
    x0 = rng.uniform(-5, 5, size=n)
    v0 = rng.uniform(-10, 10, size=n)
    return {"m": m, "f": f, "a": a, "x0": x0, "v0": v0}


def decorrelated_input_only_baseline(seed: int = 0) -> dict[str, float]:
    """Show that predicting a from (F,m) with linear regression on decorrelated set is hard
    if we only give F or only m; with both, nonlinear needed so linear R^2 on (F,m)->a is partial.
    """
    data = make_decorrelated_set(seed=seed)
    X = np.stack([data["m"], data["f"]], axis=1)
    y = data["a"]
    stats = fit_probe(X, y, seed=seed)
    stats_m = fit_probe(data["m"][:, None], y, seed=seed)
    stats_f = fit_probe(data["f"][:, None], y, seed=seed)
    return {
        "r2_from_m_f": stats["r2"],
        "r2_from_m": stats_m["r2"],
        "r2_from_f": stats_f["r2"],
        "corr_a_m": float(np.corrcoef(data["a"], data["m"])[0, 1]),
        "corr_a_f": float(np.corrcoef(data["a"], data["f"])[0, 1]),
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hidden", required=True)
    p.add_argument("--targets", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    h = np.load(args.hidden)
    t = np.load(args.targets, allow_pickle=True)
    sel = control_task_selectivity(h["hidden"], t["y"], [str(k) for k in t["keys"].tolist()])
    deco = decorrelated_input_only_baseline()
    write_json(args.out, {"selectivity": sel, "decorrelated_baseline": deco})
    for k, v in sel.items():
        print(f"{k}: selectivity={v['selectivity']:.3f}")


if __name__ == "__main__":
    main()
