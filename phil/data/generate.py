"""Generate PhiLLM ID/OOD physics problem datasets."""

from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from typing import Any

import numpy as np

from phil.data.splits import ID, OOD, RangeBox, SPLIT_SIZES, assert_ood_disjoint_from_id
from phil.data.templates import render_text
from phil.sim.simulator import run
from phil.sim.spec import ObjectSpec, WorldSpec
from phil.utils.hashing import param_hash
from phil.utils.io import ensure_dir, write_json, write_jsonl


def _uniform(rng: random.Random, lo: float, hi: float) -> float:
    return rng.uniform(lo, hi)


def _sample_vec(rng: random.Random, dim: int, lo: float, hi: float, high_mag: bool = False) -> list[float]:
    if high_mag:
        # |v| in [20, 50]
        mag_lo, mag_hi = 20.0, 50.0
        out = []
        for _ in range(dim):
            mag = _uniform(rng, mag_lo, mag_hi)
            sign = -1.0 if rng.random() < 0.5 else 1.0
            out.append(sign * mag)
        return out
    return [_uniform(rng, lo, hi) for _ in range(dim)]


def sample_objects(
    rng: random.Random,
    box: RangeBox,
    dim: int,
    n_obj: int,
    high_mag: bool = False,
    force_collisions: bool = False,
) -> list[ObjectSpec]:
    objs: list[ObjectSpec] = []
    # place objects to optionally encourage collisions in 1D
    positions: list[list[float]] = []
    if dim == 1 and force_collisions and n_obj >= 2:
        # left object moving right, right object moving left
        gap = _uniform(rng, 0.5, 2.0)
        x0 = _uniform(rng, -5.0, 0.0)
        positions = [[x0], [x0 + gap]]
        for i in range(n_obj):
            m = _uniform(rng, box.m[0], box.m[1])
            if high_mag:
                m = _uniform(rng, 20.0, 100.0) if rng.random() < 0.5 else m
            if i == 0:
                v = [_uniform(rng, 2.0, abs(box.v[1]) if not high_mag else 40.0)]
            elif i == 1:
                v = [-_uniform(rng, 2.0, abs(box.v[1]) if not high_mag else 40.0)]
            else:
                v = _sample_vec(rng, 1, box.v[0], box.v[1], high_mag=high_mag)
            f = _sample_vec(rng, 1, box.f[0], box.f[1])
            # damp forces a bit when forcing collisions so they still meet
            f = [0.1 * fi for fi in f]
            objs.append(
                ObjectSpec(
                    id=f"o{i}",
                    mass=m,
                    position=positions[i] if i < len(positions) else [_uniform(rng, -5, 5)],
                    velocity=v,
                    force=f,
                )
            )
        return objs

    for i in range(n_obj):
        m = _uniform(rng, box.m[0], box.m[1])
        if high_mag and rng.random() < 0.7:
            m = _uniform(rng, 20.0, 100.0)
        pos = _sample_vec(rng, dim, -5.0, 5.0)
        # keep objects separated initially
        for _try in range(20):
            if all(np.linalg.norm(np.array(pos) - np.array(p)) > 0.3 for p in positions):
                break
            pos = _sample_vec(rng, dim, -5.0, 5.0)
        positions.append(pos)
        vel = _sample_vec(rng, dim, box.v[0], box.v[1], high_mag=high_mag)
        force = _sample_vec(rng, dim, box.f[0], box.f[1])
        objs.append(ObjectSpec(id=f"o{i}", mass=m, position=pos, velocity=vel, force=force))
    return objs


def make_spec(
    rng: random.Random,
    box: RangeBox,
    dim: int = 1,
    ood_axis: str | None = None,
) -> WorldSpec:
    high_mag = ood_axis == "ood_mag"
    force_col = ood_axis == "ood_col"
    n_obj = rng.randint(box.n_obj[0], box.n_obj[1])
    k = rng.randint(box.k[0], box.k[1])
    dt = rng.choice([0.05, 0.1, 0.2])
    objects = sample_objects(rng, box, dim, n_obj, high_mag=high_mag, force_collisions=force_col)
    walls = None
    if rng.random() < 0.35 or ood_axis == "ood_col":
        if dim == 1:
            walls = [-8.0, 8.0]
        else:
            walls = [-8.0, 8.0, -8.0, 8.0]
    collisions = n_obj > 1
    return WorldSpec(
        dim=dim,
        dt=dt,
        horizon=k,
        objects=objects,
        walls=walls,
        collisions=collisions,
        restitution=1.0,
        tags={"ood_axis": ood_axis or "id"},
    )


def gold_answer_from_result(result: dict[str, Any], task_type: str, meta: dict[str, Any] | None = None) -> Any:
    if task_type == "T1":
        return {
            "objects": [
                {"id": o["id"], "position": o["position"], "velocity": o["velocity"]}
                for o in result["objects"]
            ]
        }
    if task_type == "T2":
        assert meta is not None
        return float(meta["unknown"]["value"])
    if task_type == "T3":
        assert meta is not None
        return meta["label"]
    raise ValueError(task_type)


def build_t1(rng: random.Random, box: RangeBox, dim: int, ood_axis: str | None) -> dict[str, Any] | None:
    for _ in range(50):
        spec = make_spec(rng, box, dim=dim, ood_axis=ood_axis)
        result = run(spec)
        n_col = int(result["n_collisions"])
        if ood_axis == "ood_col" and n_col < 2:
            continue
        if ood_axis is None and n_col > box.n_col_max:
            continue
        if ood_axis is None and n_col > 1:
            continue
        ph = param_hash(
            {
                "dim": spec.dim,
                "dt": spec.dt,
                "horizon": spec.horizon,
                "objects": [o.__dict__ for o in spec.objects],
                "walls": spec.walls,
            }
        )
        ex = {
            "id": f"T1_{ph}",
            "split": "unset",
            "task_type": "T1",
            "gold_spec": spec.to_dict(),
            "gold_answer": gold_answer_from_result(result, "T1"),
            "tags": {
                "horizon": spec.horizon,
                "n_obj": len(spec.objects),
                "n_col": n_col,
                "ood_axis": ood_axis or "id",
                "dim": dim,
            },
            "meta": {},
            "param_hash": ph,
        }
        ex["text"] = render_text(ex, rng)
        return ex
    return None


def build_t2(rng: random.Random, box: RangeBox, dim: int = 1) -> dict[str, Any] | None:
    """Infer mass from final position under known force (1-object, no collision)."""
    for _ in range(40):
        spec = make_spec(rng, RangeBox(m=box.m, v=box.v, f=box.f, k=box.k, n_obj=(1, 1), n_col_max=0), dim=dim)
        spec.collisions = False
        spec.walls = None
        o = spec.objects[0]
        # need nonzero force
        if all(abs(fi) < 1e-6 for fi in (o.force or [0])):
            continue
        result = run(spec)
        obs = {
            "objects": [
                {"id": oo["id"], "position": oo["position"], "velocity": oo["velocity"]}
                for oo in result["objects"]
            ]
        }
        meta = {
            "unknown": {"object_id": o.id, "field": "mass", "value": float(o.mass)},
            "observation": obs,
        }
        ph = param_hash({"task": "T2", "spec": spec.to_dict()})
        ex = {
            "id": f"T2_{ph}",
            "split": "unset",
            "task_type": "T2",
            "gold_spec": spec.to_dict(),
            "gold_answer": float(o.mass),
            "tags": {
                "horizon": spec.horizon,
                "n_obj": 1,
                "n_col": 0,
                "ood_axis": "id",
                "dim": dim,
            },
            "meta": meta,
            "param_hash": ph,
        }
        ex["text"] = render_text(ex, rng)
        return ex
    return None


def build_t3(rng: random.Random, box: RangeBox, dim: int = 1) -> dict[str, Any] | None:
    """Feasibility: claim about final position sign / ordering."""
    for _ in range(40):
        spec = make_spec(rng, RangeBox(m=box.m, v=box.v, f=box.f, k=box.k, n_obj=(1, 1), n_col_max=0), dim=dim)
        spec.collisions = False
        result = run(spec)
        true_x = result["objects"][0]["position"][0]
        # 50% true claim, 50% false
        if rng.random() < 0.5:
            claim = f"After {spec.horizon} steps, object o0 has position[0] > 0."
            label = "YES" if true_x > 0 else "NO"
        else:
            # flip
            claim = f"After {spec.horizon} steps, object o0 has position[0] > 0."
            # force wrong label sometimes by using a false claim about magnitude
            if rng.random() < 0.5:
                claim = f"After {spec.horizon} steps, object o0 ends at position[0] = {_fmt_num(true_x + 5.0)}."
                label = "NO"
            else:
                claim = f"After {spec.horizon} steps, object o0 ends at position[0] ≈ {_fmt_num(true_x)} (within 1%)."
                label = "YES"
        meta = {"claim": claim, "label": label, "true_x0": true_x}
        ph = param_hash({"task": "T3", "spec": spec.to_dict(), "claim": claim})
        ex = {
            "id": f"T3_{ph}",
            "split": "unset",
            "task_type": "T3",
            "gold_spec": spec.to_dict(),
            "gold_answer": label,
            "tags": {
                "horizon": spec.horizon,
                "n_obj": 1,
                "n_col": 0,
                "ood_axis": "id",
                "dim": dim,
            },
            "meta": meta,
            "param_hash": ph,
        }
        ex["text"] = render_text(ex, rng)
        return ex
    return None


def _fmt_num(x: float) -> str:
    return f"{x:.4g}"


def generate_split(
    name: str,
    n: int,
    seed: int,
    ood_axis: str | None = None,
    task_mix: tuple[float, float, float] = (0.7, 0.15, 0.15),
    dim_prob_2d: float = 0.25,
    seen_hashes: set[str] | None = None,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    seen = seen_hashes if seen_hashes is not None else set()
    box = OOD[ood_axis] if ood_axis else ID
    out: list[dict[str, Any]] = []
    attempts = 0
    max_attempts = n * 40
    while len(out) < n and attempts < max_attempts:
        attempts += 1
        r = rng.random()
        dim = 2 if rng.random() < dim_prob_2d else 1
        # OOD-col / ood-obj prefer T1
        if ood_axis in ("ood_col", "ood_obj", "ood_hor", "ood_mag"):
            task = "T1"
        elif r < task_mix[0]:
            task = "T1"
        elif r < task_mix[0] + task_mix[1]:
            task = "T2"
        else:
            task = "T3"
        if task == "T1":
            ex = build_t1(rng, box, dim=dim, ood_axis=ood_axis)
        elif task == "T2":
            ex = build_t2(rng, box, dim=1)
        else:
            ex = build_t3(rng, box, dim=1)
        if ex is None:
            continue
        if ex["param_hash"] in seen:
            continue
        seen.add(ex["param_hash"])
        ex["split"] = name
        out.append(ex)
    if len(out) < n:
        raise RuntimeError(f"only generated {len(out)}/{n} for split {name}")
    return out


def generate_level0(n: int, seed: int = 0) -> list[dict[str, Any]]:
    """Simplest: 1D, 1 object, constant force, k=1."""
    rng = random.Random(seed)
    out = []
    seen: set[str] = set()
    while len(out) < n:
        m = _uniform(rng, 1, 10)
        x = [_uniform(rng, -5, 5)]
        v = [_uniform(rng, -10, 10)]
        f = [_uniform(rng, -20, 20)]
        dt = 0.1
        spec = WorldSpec(
            dim=1,
            dt=dt,
            horizon=1,
            objects=[ObjectSpec(id="o0", mass=m, position=x, velocity=v, force=f)],
            collisions=False,
        )
        result = run(spec)
        ph = param_hash(spec.to_dict())
        if ph in seen:
            continue
        seen.add(ph)
        ex = {
            "id": f"L0_{ph}",
            "split": "inspect",
            "task_type": "T1",
            "gold_spec": spec.to_dict(),
            "gold_answer": gold_answer_from_result(result, "T1"),
            "tags": {"horizon": 1, "n_obj": 1, "n_col": 0, "ood_axis": "id", "dim": 1, "level": 0},
            "meta": {},
            "param_hash": ph,
        }
        ex["text"] = render_text(ex, rng)
        out.append(ex)
    return out


def generate_full(out_dir: str, seed: int = 42, sizes: dict[str, int] | None = None) -> dict[str, str]:
    sizes = sizes or SPLIT_SIZES
    ensure_dir(out_dir)
    seen: set[str] = set()
    paths: dict[str, str] = {}
    # ID splits share hash space
    for i, (name, n) in enumerate(
        [("train", sizes["train"]), ("dev", sizes["dev"]), ("dlab", sizes["dlab"]), ("test", sizes["test"])]
    ):
        rows = generate_split(name, n, seed=seed + i, ood_axis=None, seen_hashes=seen)
        path = f"{out_dir}/{name}.jsonl"
        write_jsonl(path, rows)
        paths[name] = path
        print(f"wrote {len(rows)} -> {path}")
    for j, axis in enumerate(["ood_mag", "ood_hor", "ood_obj", "ood_col"]):
        rows = generate_split(axis, sizes[axis], seed=seed + 100 + j, ood_axis=axis, seen_hashes=seen)
        assert_ood_disjoint_from_id(rows, axis)
        path = f"{out_dir}/{axis}.jsonl"
        write_jsonl(path, rows)
        paths[axis] = path
        print(f"wrote {len(rows)} -> {path}")
    meta = {
        "seed": seed,
        "sizes": {k: sizes[k] for k in sizes},
        "n_unique_hashes": len(seen),
        "paths": paths,
    }
    write_json(f"{out_dir}/manifest.json", meta)
    return paths


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Generate PhiLLM datasets")
    p.add_argument("--level", type=int, default=None, help="0 = simplest inspection set")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--full", action="store_true")
    p.add_argument("--out-dir", type=str, default="/mnt/work/galoaa/phillm/data")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tiny", action="store_true", help="tiny sizes for smoke tests")
    args = p.parse_args(argv)

    if args.level == 0 or (args.out and not args.full):
        rows = generate_level0(args.n, seed=args.seed)
        out = args.out or "data/inspect_50.jsonl"
        write_jsonl(out, rows)
        print(f"wrote {len(rows)} level-0 examples -> {out}")
        return

    if args.full:
        sizes = SPLIT_SIZES
        if args.tiny:
            sizes = {k: (32 if k == "train" else 8) for k in SPLIT_SIZES}
        generate_full(args.out_dir, seed=args.seed, sizes=sizes)
        return

    p.print_help()


if __name__ == "__main__":
    main()
