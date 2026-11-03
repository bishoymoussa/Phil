"""Eval acceptance: gold-as-predictions => 100% accuracy, 0 MAE."""

from __future__ import annotations

import json
from pathlib import Path

from phil.data.generate import generate_level0
from phil.eval.eval import evaluate, score_example
from phil.utils.io import write_jsonl


def test_gold_predictions_perfect(tmp_path: Path):
    rows = generate_level0(20, seed=1)
    gold_path = tmp_path / "gold.jsonl"
    pred_path = tmp_path / "pred.jsonl"
    write_jsonl(gold_path, rows)
    preds = []
    for r in rows:
        preds.append(
            {
                "id": r["id"],
                "raw_output": json.dumps(r["gold_answer"]),
                "final_answer": json.dumps(r["gold_answer"]),
                "called": False,
                "n_tokens": 10,
            }
        )
    write_jsonl(pred_path, preds)
    metrics = evaluate(str(gold_path), str(pred_path))
    assert metrics["accuracy"] == 1.0
    assert metrics["pos_mae"] is not None
    assert metrics["pos_mae"] < 1e-12


def test_handmade_errors(tmp_path: Path):
    rows = generate_level0(4, seed=2)
    # corrupt two answers
    preds = []
    for i, r in enumerate(rows):
        ans = json.loads(json.dumps(r["gold_answer"]))
        if i < 2:
            ans["objects"][0]["position"][0] += 10.0
        preds.append({"id": r["id"], "final_answer": json.dumps(ans), "called": False})
    gold_path = tmp_path / "g.jsonl"
    pred_path = tmp_path / "p.jsonl"
    write_jsonl(gold_path, rows)
    write_jsonl(pred_path, preds)
    metrics = evaluate(str(gold_path), str(pred_path))
    assert abs(metrics["accuracy"] - 0.5) < 1e-9
