"""ID / OOD range definitions and validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RangeBox:
    m: tuple[float, float]
    v: tuple[float, float]
    f: tuple[float, float]
    k: tuple[int, int]
    n_obj: tuple[int, int]
    n_col_max: int


ID = RangeBox(m=(1.0, 10.0), v=(-10.0, 10.0), f=(-20.0, 20.0), k=(1, 5), n_obj=(1, 2), n_col_max=1)

OOD = {
    "ood_mag": RangeBox(m=(20.0, 100.0), v=(-50.0, -20.0), f=(-20.0, 20.0), k=(1, 5), n_obj=(1, 2), n_col_max=1),
    # v also allow positive high magnitudes via sampling sign separately
    "ood_hor": RangeBox(m=(1.0, 10.0), v=(-10.0, 10.0), f=(-20.0, 20.0), k=(10, 50), n_obj=(1, 2), n_col_max=1),
    "ood_obj": RangeBox(m=(1.0, 10.0), v=(-10.0, 10.0), f=(-20.0, 20.0), k=(1, 5), n_obj=(3, 4), n_col_max=2),
    "ood_col": RangeBox(m=(1.0, 10.0), v=(-10.0, 10.0), f=(-20.0, 20.0), k=(3, 8), n_obj=(2, 3), n_col_max=5),
}


def in_range(x: float, lo: float, hi: float) -> bool:
    return lo - 1e-9 <= x <= hi + 1e-9


def example_matches_box(ex: dict[str, Any], box: RangeBox) -> bool:
    spec = ex["gold_spec"]
    tags = ex.get("tags", {})
    k = int(spec["horizon"])
    n_obj = len(spec["objects"])
    if not (box.k[0] <= k <= box.k[1]):
        return False
    if not (box.n_obj[0] <= n_obj <= box.n_obj[1]):
        return False
    if int(tags.get("n_col", 0)) > box.n_col_max and box is ID:
        return False
    for o in spec["objects"]:
        m = float(o["mass"])
        if not in_range(m, box.m[0], box.m[1]):
            return False
        for vi in o["velocity"]:
            # ood_mag uses high |v|; allow either side
            if box is OOD["ood_mag"]:
                if abs(float(vi)) < 20.0 - 1e-9 or abs(float(vi)) > 50.0 + 1e-9:
                    return False
            else:
                if not in_range(float(vi), box.v[0], box.v[1]):
                    return False
        for fi in o["force"]:
            if not in_range(float(fi), box.f[0], box.f[1]):
                return False
    return True


def assert_ood_disjoint_from_id(examples: list[dict[str, Any]], axis: str) -> None:
    """Verify OOD examples violate the ID box on the intended axis."""
    for ex in examples:
        spec = ex["gold_spec"]
        if axis == "ood_mag":
            ok = False
            for o in spec["objects"]:
                if o["mass"] >= 20.0 - 1e-9:
                    ok = True
                for vi in o["velocity"]:
                    if abs(vi) >= 20.0 - 1e-9:
                        ok = True
            if not ok:
                raise AssertionError(f"{ex['id']} not OOD-mag")
        elif axis == "ood_hor":
            if spec["horizon"] < 10:
                raise AssertionError(f"{ex['id']} not OOD-hor")
        elif axis == "ood_obj":
            if len(spec["objects"]) < 3:
                raise AssertionError(f"{ex['id']} not OOD-obj")
        elif axis == "ood_col":
            if int(ex["tags"].get("n_col", 0)) < 2:
                raise AssertionError(f"{ex['id']} not OOD-col")
        else:
            raise ValueError(axis)


SPLIT_SIZES = {
    "train": 50_000,
    "dev": 1_000,
    "dlab": 5_000,
    "test": 2_000,
    "ood_mag": 2_000,
    "ood_hor": 2_000,
    "ood_obj": 2_000,
    "ood_col": 2_000,
}
