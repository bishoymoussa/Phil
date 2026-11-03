"""Ridge probes with problem-level splits and multiple seeds."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from phil.utils.io import write_json


STATED = {"m", "x0", "v0", "f"}
DERIVED = {"a", "xk", "vk", "momentum", "ke"}


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def fit_probe(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = 0,
    alpha: float = 1.0,
) -> dict[str, float]:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed)
    model = Ridge(alpha=alpha, random_state=seed)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    return {
        "r2": r2_score(yte, pred),
        "mse": float(np.mean((yte - pred) ** 2)),
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
    }


def run_all_probes(
    hidden_path: str,
    targets_path: str,
    out_path: str,
    *,
    seeds: tuple[int, ...] = (0, 1, 2),
    alpha: float = 1.0,
) -> dict[str, Any]:
    h = np.load(hidden_path)
    t = np.load(targets_path, allow_pickle=True)
    hidden = h["hidden"]  # (N, L, H)
    y = t["y"]
    keys = [str(k) for k in t["keys"].tolist()]
    n, n_layers, _ = hidden.shape
    key_to_idx = {k: i for i, k in enumerate(keys)}

    results: dict[str, Any] = {"stated": {}, "derived": {}, "all": {}}
    for group_name, group_keys in [("stated", STATED), ("derived", DERIVED), ("all", set(keys))]:
        for k in group_keys:
            if k not in key_to_idx:
                continue
            yi = y[:, key_to_idx[k]]
            mask = np.isfinite(yi)
            layer_stats = []
            for layer in range(n_layers):
                seed_r2 = []
                for seed in seeds:
                    X = hidden[mask, layer, :]
                    yy = yi[mask]
                    if len(yy) < 20:
                        seed_r2.append(float("nan"))
                        continue
                    stats = fit_probe(X, yy, seed=seed, alpha=alpha)
                    seed_r2.append(stats["r2"])
                layer_stats.append(
                    {
                        "layer": layer,
                        "r2_mean": float(np.nanmean(seed_r2)),
                        "r2_std": float(np.nanstd(seed_r2)),
                        "seeds": seed_r2,
                    }
                )
            results[group_name][k] = layer_stats

    # input-only baseline: linear regression from stated values to derived
    stated_idx = [key_to_idx[k] for k in ("m", "x0", "v0", "f") if k in key_to_idx]
    X_in = y[:, stated_idx]
    input_only = {}
    for k in DERIVED:
        if k not in key_to_idx:
            continue
        yi = y[:, key_to_idx[k]]
        mask = np.isfinite(yi) & np.all(np.isfinite(X_in), axis=1)
        seed_r2 = []
        for seed in seeds:
            stats = fit_probe(X_in[mask], yi[mask], seed=seed, alpha=alpha)
            seed_r2.append(stats["r2"])
        input_only[k] = {"r2_mean": float(np.nanmean(seed_r2)), "r2_std": float(np.nanstd(seed_r2))}
    results["input_only_baseline"] = input_only

    # layer-0 baseline summary
    results["layer0_vs_best"] = {}
    for k, layer_stats in results["derived"].items():
        r0 = layer_stats[0]["r2_mean"]
        best = max(layer_stats, key=lambda d: d["r2_mean"] if np.isfinite(d["r2_mean"]) else -1e9)
        results["layer0_vs_best"][k] = {
            "layer0_r2": r0,
            "best_layer": best["layer"],
            "best_r2": best["r2_mean"],
            "delta": best["r2_mean"] - r0 if np.isfinite(best["r2_mean"]) and np.isfinite(r0) else None,
        }

    write_json(out_path, results)
    return results


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hidden", required=True)
    p.add_argument("--targets", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    res = run_all_probes(args.hidden, args.targets, args.out)
    # print brief summary
    for k, v in res.get("layer0_vs_best", {}).items():
        print(f"{k}: layer0={v['layer0_r2']:.3f} best@{v['best_layer']}={v['best_r2']:.3f}")


if __name__ == "__main__":
    main()
