"""Autoregressive multi-step rollouts: feed predicted state back as next prompt."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from typing import Any

import numpy as np

from phil.eval.metrics import parse_t1_answer, relative_error
from phil.sim.simulator import run
from phil.utils.io import read_jsonl, write_json, write_jsonl


def step_spec_from_state(spec: dict[str, Any], state_objects: list[dict[str, Any]]) -> dict[str, Any]:
    new = deepcopy(spec)
    new["horizon"] = 1
    by_id = {o["id"]: o for o in state_objects}
    for o in new["objects"]:
        s = by_id[o["id"]]
        o["position"] = s["position"]
        o["velocity"] = s["velocity"]
    return new


def gold_rollout_errors(example: dict[str, Any], steps: int) -> list[float]:
    """Oracle baseline: error should be ~0 when using gold one-step repeatedly."""
    spec = deepcopy(example["gold_spec"])
    errs = []
    cur = deepcopy(spec)
    # true trajectory from original long horizon
    long = deepcopy(spec)
    long["horizon"] = steps
    true = run(long)["objects"]
    for t in range(steps):
        cur["horizon"] = 1
        pred = run(cur)["objects"]
        # compare to stepping true? better compare cumulative to closed long run at each step
        long_t = deepcopy(spec)
        long_t["horizon"] = t + 1
        gold_t = run(long_t)["objects"]
        e = []
        gmap = {o["id"]: o for o in gold_t}
        for o in pred:
            g = gmap[o["id"]]
            for a, b in zip(o["position"], g["position"]):
                e.append(abs(a - b))
        errs.append(float(np.mean(e)))
        cur = step_spec_from_state(cur, pred)
    return errs


def evaluate_predicted_rollout(
    example: dict[str, Any],
    one_step_predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Given a sequence of predicted one-step answers, measure error growth vs gold."""
    spec = example["gold_spec"]
    errs = []
    for t, pred_raw in enumerate(one_step_predictions):
        parsed = parse_t1_answer(pred_raw)
        long_t = deepcopy(spec)
        long_t["horizon"] = t + 1
        gold_t = run(long_t)
        if parsed is None:
            errs.append(float("nan"))
            continue
        mae = []
        gmap = {o["id"]: o for o in gold_t["objects"]}
        for o in parsed.get("objects", []):
            g = gmap.get(o["id"])
            if not g:
                continue
            mae.extend(abs(np.asarray(o["position"]) - np.asarray(g["position"])).tolist())
        errs.append(float(np.mean(mae)) if mae else float("nan"))
    return {"errors": errs, "final_mae": errs[-1] if errs else None}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--limit", type=int, default=100)
    args = p.parse_args(argv)
    rows = [r for r in read_jsonl(args.data) if r["task_type"] == "T1"][: args.limit]
    out = []
    for ex in rows:
        errs = gold_rollout_errors(ex, args.steps)
        out.append({"id": ex["id"], "oracle_errors": errs, "oracle_final": errs[-1]})
    write_jsonl(args.out, out)
    write_json(
        args.out.replace(".jsonl", ".summary.json"),
        {"mean_final": float(np.nanmean([r["oracle_final"] for r in out])), "n": len(out)},
    )
    print(f"wrote {len(out)} -> {args.out}")


if __name__ == "__main__":
    main()
