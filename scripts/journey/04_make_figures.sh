#!/usr/bin/env bash
set -euo pipefail
ROOT="${PHIL_ROOT:-/home/bi.ga/Phil}"
WORK="${PHIL_WORK:-/mnt/work/galoaa/phillm}"
export PYTHONPATH="$ROOT"
PY="${PHIL_PY:-python3}"
mkdir -p "$WORK/figs/dataset" "$WORK/figs/results" "$ROOT/assets/figs/dataset" "$ROOT/assets/figs/results"

"$PY" - <<PY
from pathlib import Path
from phil.viz.trajectories import render_dataset_gallery, plot_rollout_errors
from phil.viz.results import collect_metrics_tree, plot_id_ood_bars, plot_utility_curves, write_qualitative_examples
from phil.utils.io import read_jsonl
import json

data = Path("$WORK/data")
figs_ds = Path("$WORK/figs/dataset")
figs_res = Path("$WORK/figs/results")
repo_ds = Path("$ROOT/assets/figs/dataset")
repo_res = Path("$ROOT/assets/figs/results")

paths = {
    "inspect": str(data / "inspect_50.jsonl"),
    "test": str(data / "test.jsonl"),
    "ood_mag": str(data / "ood_mag.jsonl"),
    "ood_hor": str(data / "ood_hor.jsonl"),
    "ood_obj": str(data / "ood_obj.jsonl"),
    "ood_col": str(data / "ood_col.jsonl"),
}
# only existing
paths = {k: v for k, v in paths.items() if Path(v).exists()}
written = render_dataset_gallery(paths, figs_ds, n_per_split=4)
print("dataset figs", len(written))
for p in written:
    dest = repo_ds / Path(p).name
    dest.write_bytes(Path(p).read_bytes())

# oracle rollout mean curve if present
roll = data.parent / "preds" / "oracle_rollout.jsonl"
if roll.exists():
    rows = read_jsonl(roll)
    # mean error per step
    import numpy as np
    mat = np.array([r["oracle_errors"] for r in rows], dtype=float)
    mean = np.nanmean(mat, axis=0).tolist()
    plot_rollout_errors(mean, figs_res / "oracle_rollout_mae.png", title="Oracle multi-step position MAE")
    (repo_res / "oracle_rollout_mae.png").write_bytes((figs_res / "oracle_rollout_mae.png").read_bytes())

tree = collect_metrics_tree(data.parent / "preds")
if tree:
    plot_id_ood_bars(tree, figs_res / "id_ood_accuracy.png", metric_key="accuracy")
    (repo_res / "id_ood_accuracy.png").write_bytes((figs_res / "id_ood_accuracy.png").read_bytes())
    # utility from oracle if present
    points = []
    for sys, splits in tree.items():
        for sp, m in splits.items():
            if "accuracy" in m and "call_rate" in m:
                points.append({"call_rate": m["call_rate"], "accuracy": m["accuracy"], "label": f"{sys}/{sp}"})
    if points:
        plot_utility_curves(points[:12], figs_res / "utility_scatter.png")
        (repo_res / "utility_scatter.png").write_bytes((figs_res / "utility_scatter.png").read_bytes())

# qualitative if zeroshot or lora preds exist
for pred_name in ["zeroshot_Qwen3_5_2B_test.jsonl", "lora_test.jsonl", "oracle_test.jsonl"]:
    pred = data.parent / "preds" / pred_name
    gold = data / "test.jsonl"
    if pred.exists() and gold.exists():
        write_qualitative_examples(str(gold), str(pred), figs_res / f"qual_{pred_name}.md")
        print("qual", pred_name)
        break

print("done figures")
PY
