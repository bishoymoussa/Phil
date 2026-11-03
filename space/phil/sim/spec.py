"""JSON specification schema and parsing for the PhiLLM simulator tool."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


class SpecError(ValueError):
    """Raised when a simulator specification is malformed or invalid."""


@dataclass
class ObjectSpec:
    id: str
    mass: float
    position: list[float]
    velocity: list[float]
    force: list[float] | None = None

    def dim(self) -> int:
        return len(self.position)


@dataclass
class WorldSpec:
    dim: int
    dt: float
    horizon: int
    objects: list[ObjectSpec]
    walls: list[float] | None = None  # [xmin, xmax] or [xmin,xmax,ymin,ymax]
    collisions: bool = True
    gravity: list[float] | None = None  # optional constant g added to force/m
    restitution: float = 1.0  # elastic by default
    tags: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_vec(v: Any, dim: int, name: str) -> list[float]:
    if not isinstance(v, (list, tuple)) or len(v) != dim:
        raise SpecError(f"{name} must be a length-{dim} list, got {v!r}")
    try:
        out = [float(x) for x in v]
    except (TypeError, ValueError) as e:
        raise SpecError(f"{name} must contain numbers") from e
    return out


def parse_spec(raw: str | dict[str, Any]) -> WorldSpec:
    """Parse a JSON string or dict into a validated WorldSpec."""
    if isinstance(raw, str):
        text = raw.strip()
        # tolerate optional <sim>...</sim> wrappers
        if "<sim>" in text.lower():
            lo = text.lower().find("<sim>") + 5
            hi = text.lower().find("</sim>")
            if hi < 0:
                raise SpecError("unclosed <sim> tag")
            text = text[lo:hi].strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise SpecError(f"invalid JSON: {e}") from e
    elif isinstance(raw, dict):
        data = raw
    else:
        raise SpecError(f"spec must be str or dict, got {type(raw)}")

    if not isinstance(data, dict):
        raise SpecError("spec root must be an object")

    try:
        dim = int(data["dim"])
        dt = float(data["dt"])
        horizon = int(data["horizon"])
    except (KeyError, TypeError, ValueError) as e:
        raise SpecError("dim, dt, horizon are required and must be numeric") from e

    if dim not in (1, 2):
        raise SpecError(f"dim must be 1 or 2, got {dim}")
    if dt <= 0:
        raise SpecError(f"dt must be > 0, got {dt}")
    if horizon < 0:
        raise SpecError(f"horizon must be >= 0, got {horizon}")

    objs_raw = data.get("objects")
    if not isinstance(objs_raw, list) or not objs_raw:
        raise SpecError("objects must be a non-empty list")

    objects: list[ObjectSpec] = []
    for i, o in enumerate(objs_raw):
        if not isinstance(o, dict):
            raise SpecError(f"object {i} must be a dict")
        try:
            oid = str(o["id"])
            mass = float(o["mass"])
        except (KeyError, TypeError, ValueError) as e:
            raise SpecError(f"object {i} needs id and mass") from e
        if mass <= 0:
            raise SpecError(f"object {oid} mass must be > 0")
        pos = _as_vec(o.get("position"), dim, f"object {oid} position")
        vel = _as_vec(o.get("velocity"), dim, f"object {oid} velocity")
        force = o.get("force")
        if force is None:
            force = [0.0] * dim
        else:
            force = _as_vec(force, dim, f"object {oid} force")
        objects.append(ObjectSpec(id=oid, mass=mass, position=pos, velocity=vel, force=force))

    walls = data.get("walls")
    if walls is not None:
        expected = 2 * dim
        if not isinstance(walls, (list, tuple)) or len(walls) != expected:
            raise SpecError(f"walls must have length {expected}")
        walls = [float(x) for x in walls]

    gravity = data.get("gravity")
    if gravity is not None:
        gravity = _as_vec(gravity, dim, "gravity")

    restitution = float(data.get("restitution", 1.0))
    collisions = bool(data.get("collisions", True))
    tags = data.get("tags") or {}
    if not isinstance(tags, dict):
        raise SpecError("tags must be a dict")

    return WorldSpec(
        dim=dim,
        dt=dt,
        horizon=horizon,
        objects=objects,
        walls=walls,
        collisions=collisions,
        gravity=gravity,
        restitution=restitution,
        tags=tags,
    )


def extract_sim_block(text: str) -> str | None:
    """Return JSON inside the first <sim>...</sim> block, or None."""
    low = text.lower()
    start = low.find("<sim>")
    if start < 0:
        return None
    end = low.find("</sim>", start)
    if end < 0:
        return None
    return text[start + 5 : end].strip()
