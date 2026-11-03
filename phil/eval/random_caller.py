"""Matched-rate random caller baseline (5 seeds)."""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from phil.eval.eval import evaluate
from phil.utils.io import read_jsonl, write_json, write_jsonl


def make_random_caller_preds(
    llm_only_path: str,
    always_call_path: str,
    call_rate: float,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    llm = {r["id"]: r for r in read_jsonl(llm_only_path)}
    always = {r["id"]: r for r in read_jsonl(always_call_path)}
    ids = list(llm.keys())
    rows = []
    for i in ids:
        if rng.random() < call_rate and i in always:
            row = dict(always[i])
            row["system"] = f"random_seed_{seed}"
            row["called"] = True
        else:
            row = dict(llm[i])
            row["system"] = f"random_seed_{seed}"
            row["called"] = False
        rows.append(row)
    return rows


def run_random_baseline(
    gold_path: str,
    llm_only_path: str,
    always_call_path: str,
    call_rate: float,
    out_dir: str,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> dict[str, Any]:
    accs = []
    for seed in seeds:
        rows = make_random_caller_preds(llm_only_path, always_call_path, call_rate, seed)
        pred_path = f"{out_dir}/random_seed{seed}.jsonl"
        write_jsonl(pred_path, rows)
        metrics = evaluate(gold_path, pred_path, out_path=f"{out_dir}/random_seed{seed}.metrics.json")
        accs.append(metrics["accuracy"])
    summary = {
        "call_rate": call_rate,
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "seeds": list(seeds),
        "accuracies": accs,
    }
    write_json(f"{out_dir}/random_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gold", required=True)
    p.add_argument("--llm-only", required=True)
    p.add_argument("--always-call", required=True)
    p.add_argument("--call-rate", type=float, required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args(argv)
    s = run_random_baseline(args.gold, args.llm_only, args.always_call, args.call_rate, args.out_dir)
    print(s)


if __name__ == "__main__":
    main()
