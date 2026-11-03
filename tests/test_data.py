"""Data generator + split disjointness checks."""

from __future__ import annotations

from phil.data.generate import generate_full, generate_level0
from phil.data.splits import assert_ood_disjoint_from_id
from phil.utils.io import read_jsonl


def test_level0_ids_unique():
    rows = generate_level0(30, seed=3)
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids))
    assert all(r["task_type"] == "T1" for r in rows)


def test_tiny_full_splits(tmp_path):
    sizes = {k: (24 if k == "train" else 6) for k in ["train", "dev", "dlab", "test", "ood_mag", "ood_hor", "ood_obj", "ood_col"]}
    paths = generate_full(str(tmp_path), seed=7, sizes=sizes)
    hashes = set()
    for name, path in paths.items():
        rows = read_jsonl(path)
        for r in rows:
            assert r["param_hash"] not in hashes
            hashes.add(r["param_hash"])
        if name.startswith("ood_"):
            assert_ood_disjoint_from_id(rows, name)
