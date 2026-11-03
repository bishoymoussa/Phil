"""Train tiny neural one-step physics simulators for weight-space reading (RQ5)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np

from phil.utils.io import ensure_dir, write_json


LAWS = ("constant_force", "hooke", "linear_drag", "inv_square")


def make_dataset(law: str, n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, dict]:
    """X = [x, v, p] -> Y = [x_next, v_next] for dt=0.1, m=1 unless p encodes param."""
    rng = np.random.default_rng(seed)
    dt = 0.1
    x = rng.uniform(-2, 2, size=(n, 1))
    v = rng.uniform(-2, 2, size=(n, 1))
    if law == "constant_force":
        # p = F (m=1) => a = F
        p = rng.uniform(-3, 3, size=(n, 1))
        a = p
        meta = {"param": "F", "dt": dt}
    elif law == "hooke":
        # F = -k x, a = -k x (m=1); p = k
        p = rng.uniform(0.5, 5.0, size=(n, 1))
        a = -p * x
        meta = {"param": "k", "dt": dt}
    elif law == "linear_drag":
        # F = -c v; p = c
        p = rng.uniform(0.1, 2.0, size=(n, 1))
        a = -p * v
        meta = {"param": "c", "dt": dt}
    elif law == "inv_square":
        # a = -G / (x+|eps|)^2 * sign(x); p = G; keep |x| away from 0
        x = rng.uniform(0.5, 3.0, size=(n, 1)) * rng.choice([-1, 1], size=(n, 1))
        p = rng.uniform(0.5, 3.0, size=(n, 1))
        a = -p / (np.abs(x) + 0.2) ** 2 * np.sign(x)
        meta = {"param": "G", "dt": dt}
    else:
        raise ValueError(law)
    # Euler as in proposal: x <- x + v dt; v <- v + a dt
    x_next = x + v * dt
    v_next = v + a * dt
    X = np.concatenate([x, v, p], axis=1).astype(np.float64)
    Y = np.concatenate([x_next, v_next], axis=1).astype(np.float64)
    meta["law"] = law
    return X, Y, meta


def fit_linear(X: np.ndarray, Y: np.ndarray) -> dict[str, np.ndarray]:
    """Y = X @ W + b  (closed form least squares)."""
    ones = np.ones((X.shape[0], 1))
    Xb = np.concatenate([X, ones], axis=1)
    # (4 x 2)
    Wb, _, _, _ = np.linalg.lstsq(Xb, Y, rcond=None)
    W = Wb[:-1]  # (3, 2)
    b = Wb[-1]  # (2,)
    return {"W": W, "b": b}


def fit_mlp(
    X: np.ndarray,
    Y: np.ndarray,
    hidden: int = 8,
    steps: int = 4000,
    lr: float = 0.05,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Tiny MLP 3->H->2 with tanh, trained by GD (numpy)."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    W1 = rng.normal(0, 0.3, size=(d, hidden))
    b1 = np.zeros(hidden)
    W2 = rng.normal(0, 0.3, size=(hidden, Y.shape[1]))
    b2 = np.zeros(Y.shape[1])
    for t in range(steps):
        h = np.tanh(X @ W1 + b1)
        pred = h @ W2 + b2
        err = pred - Y
        loss = np.mean(err**2)
        # grads
        dW2 = h.T @ err / n
        db2 = err.mean(axis=0)
        dh = err @ W2.T * (1 - h**2)
        dW1 = X.T @ dh / n
        db1 = dh.mean(axis=0)
        W2 -= lr * dW2
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1
        if t % 1000 == 0:
            lr *= 0.9
            _ = loss
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def predict_linear(weights: dict[str, np.ndarray], X: np.ndarray) -> np.ndarray:
    return X @ weights["W"] + weights["b"]


def predict_mlp(weights: dict[str, np.ndarray], X: np.ndarray) -> np.ndarray:
    h = np.tanh(X @ weights["W1"] + weights["b1"])
    return h @ weights["W2"] + weights["b2"]


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def train_all(out_dir: str, seed: int = 0) -> dict:
    ensure_dir(out_dir)
    catalog = {}
    for law in LAWS:
        X, Y, meta = make_dataset(law, n=4000, seed=seed)
        Xte, Yte, _ = make_dataset(law, n=500, seed=seed + 99)
        lin = fit_linear(X, Y)
        mlp = fit_mlp(X, Y, seed=seed)
        lin_mse = mse(predict_linear(lin, Xte), Yte)
        mlp_mse = mse(predict_mlp(mlp, Xte), Yte)
        # prefer linear when good enough
        use = "linear" if lin_mse < 1e-4 else "mlp"
        weights = lin if use == "linear" else mlp
        path = Path(out_dir) / f"{law}_{use}.npz"
        np.savez(path, **weights)
        # controls
        shuffled = {k: np.random.default_rng(seed + 1).permutation(v.ravel()).reshape(v.shape) for k, v in weights.items()}
        random_init = {k: np.random.default_rng(seed + 2).normal(0, 0.3, size=v.shape) for k, v in weights.items()}
        np.savez(Path(out_dir) / f"{law}_{use}_shuffled.npz", **shuffled)
        np.savez(Path(out_dir) / f"{law}_{use}_random.npz", **random_init)
        catalog[law] = {
            **meta,
            "arch": use,
            "path": str(path),
            "lin_mse": lin_mse,
            "mlp_mse": mlp_mse,
            "n_params": int(sum(v.size for v in weights.values())),
        }
        print(f"{law}: arch={use} lin_mse={lin_mse:.2e} mlp_mse={mlp_mse:.2e} params={catalog[law]['n_params']}")
    write_json(Path(out_dir) / "catalog.json", catalog)
    return catalog


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/mnt/work/galoaa/phillm/weightspace/tiny_sims")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    train_all(args.out, seed=args.seed)


if __name__ == "__main__":
    main()
