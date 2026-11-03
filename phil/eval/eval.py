"""Evaluate prediction JSONL files against gold splits."""

from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any

import numpy as np

from phil.eval.metrics import (
    auroc,
    bootstrap_ci,
    parse_numeric_answer,
    parse_t1_answer,
    parse_yes_no,
    physical_violation_elastic,
    relative_error,
    t1_mae,
    t1_tolerance_correct,
    utility,
)
from phil.utils.io import read_jsonl, write_json


def score_example(gold: dict[str, Any], pred_row: dict[str, Any], tol: float = 0.05) -> dict[str, Any]:
    task = gold["task_type"]
    raw = pred_row.get("final_answer", pred_row.get("raw_output"))
    out: dict[str, Any] = {
        "id": gold["id"],
        "task_type": task,
        "correct": False,
        "pos_mae": None,
        "vel_mae": None,
        "rel_err": None,
        "phys_violation": False,
        "called": bool(pred_row.get("called", False)),
        "spec_valid": pred_row.get("spec_valid"),
        "n_tokens": pred_row.get("n_tokens"),
        "p_sim": pred_row.get("p_sim"),
    }
    if task == "T1":
        parsed = parse_t1_answer(raw)
        mae = t1_mae(parsed, gold["gold_answer"])
        out["pos_mae"] = mae["pos_mae"]
        out["vel_mae"] = mae["vel_mae"]
        out["correct"] = t1_tolerance_correct(parsed, gold["gold_answer"], tol=tol)
        out["phys_violation"] = physical_violation_elastic(parsed, gold["gold_spec"])
        # aggregate relative error on positions
        if parsed and mae["ok"]:
            errs = []
            gmap = {o["id"]: o for o in gold["gold_answer"]["objects"]}
            for o in parsed.get("objects", []):
                g = gmap.get(o["id"])
                if not g:
                    continue
                for a, b in zip(o["position"], g["position"]):
                    errs.append(relative_error(float(a), float(b)))
            out["rel_err"] = float(np.mean(errs)) if errs else None
    elif task == "T2":
        num = parse_numeric_answer(raw)
        gold_v = float(gold["gold_answer"])
        if num is None:
            out["correct"] = False
            out["rel_err"] = None
        else:
            re = relative_error(num, gold_v)
            out["rel_err"] = re
            out["correct"] = re <= tol
    elif task == "T3":
        yn = parse_yes_no(raw)
        out["correct"] = yn == str(gold["gold_answer"]).upper()
    else:
        raise ValueError(task)
    return out


def aggregate(scores: list[dict[str, Any]], lambdas: list[float] | None = None) -> dict[str, Any]:
    lambdas = lambdas or [0.0, 0.05, 0.1, 0.25]
    if not scores:
        return {"n": 0}
    acc = float(np.mean([s["correct"] for s in scores]))
    call_rate = float(np.mean([s["called"] for s in scores]))
    pos = [s["pos_mae"] for s in scores if s["pos_mae"] is not None and not (isinstance(s["pos_mae"], float) and np.isnan(s["pos_mae"]))]
    vel = [s["vel_mae"] for s in scores if s["vel_mae"] is not None and not (isinstance(s["vel_mae"], float) and np.isnan(s["vel_mae"]))]
    phys = float(np.mean([s["phys_violation"] for s in scores]))
    tokens = [s["n_tokens"] for s in scores if s.get("n_tokens") is not None]
    spec_vals = [s["spec_valid"] for s in scores if s.get("spec_valid") is not None]
    spec_err = None
    if spec_vals:
        # among called
        called_specs = [s for s in scores if s["called"] and s.get("spec_valid") is not None]
        if called_specs:
            spec_err = float(np.mean([not bool(s["spec_valid"]) for s in called_specs]))
    metrics: dict[str, Any] = {
        "n": len(scores),
        "accuracy": acc,
        "call_rate": call_rate,
        "pos_mae": float(np.mean(pos)) if pos else None,
        "vel_mae": float(np.mean(vel)) if vel else None,
        "phys_violation_rate": phys,
        "mean_tokens": float(np.mean(tokens)) if tokens else None,
        "spec_error_rate": spec_err,
        "U_lambda": {str(lam): utility(acc, call_rate, lam) for lam in lambdas},
    }
    # per task
    by_task: dict[str, list] = defaultdict(list)
    for s in scores:
        by_task[s["task_type"]].append(s)
    metrics["by_task"] = {
        t: {"n": len(v), "accuracy": float(np.mean([x["correct"] for x in v]))} for t, v in by_task.items()
    }
    return metrics


def evaluate(
    gold_path: str,
    pred_path: str,
    out_path: str | None = None,
    tol: float = 0.05,
) -> dict[str, Any]:
    gold_rows = {r["id"]: r for r in read_jsonl(gold_path)}
    pred_rows = read_jsonl(pred_path)
    scores = []
    missing = 0
    for pr in pred_rows:
        g = gold_rows.get(pr["id"])
        if g is None:
            missing += 1
            continue
        scores.append(score_example(g, pr, tol=tol))
    metrics = aggregate(scores)
    metrics["missing_gold"] = missing
    metrics["gold_path"] = gold_path
    metrics["pred_path"] = pred_path

    # AUROC of p_sim vs error if available
    pairs = [(s["p_sim"], 0 if s["correct"] else 1) for s in scores if s.get("p_sim") is not None]
    if len(pairs) >= 10:
        scores_p, labels = zip(*pairs)
        metrics["auroc_call_vs_error"] = auroc(list(scores_p), list(labels))
        # bootstrap on accuracy
        accs = [float(s["correct"]) for s in scores]
        mean, lo, hi = bootstrap_ci(accs)
        metrics["accuracy_ci95"] = {"mean": mean, "lo": lo, "hi": hi}

    # per-tag
    by_tag: dict[str, list] = defaultdict(list)
    for s in scores:
        g = gold_rows[s["id"]]
        axis = g.get("tags", {}).get("ood_axis", "id")
        by_tag[axis].append(s)
    metrics["by_tag"] = {k: aggregate(v) for k, v in by_tag.items()}

    if out_path:
        write_json(out_path, metrics)
    return metrics


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gold", required=True)
    p.add_argument("--pred", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--tol", type=float, default=0.05)
    args = p.parse_args(argv)
    metrics = evaluate(args.gold, args.pred, out_path=args.out, tol=args.tol)
    print(json_dumps(metrics))


def json_dumps(obj: Any) -> str:
    import json

    return json.dumps(obj, indent=2, default=str)


if __name__ == "__main__":
    main()
