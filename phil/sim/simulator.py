"""Exact-ish Newtonian point-mass simulator (Euler + event-driven elastic collisions)."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from phil.sim.spec import ObjectSpec, SpecError, WorldSpec, parse_spec


EPS = 1e-12


@dataclass
class State:
    positions: np.ndarray  # (n, dim)
    velocities: np.ndarray  # (n, dim)
    masses: np.ndarray  # (n,)
    forces: np.ndarray  # (n, dim)
    ids: list[str]
    t: float = 0.0
    n_collisions: int = 0


def _state_from_spec(spec: WorldSpec) -> State:
    n = len(spec.objects)
    dim = spec.dim
    positions = np.array([o.position for o in spec.objects], dtype=np.float64)
    velocities = np.array([o.velocity for o in spec.objects], dtype=np.float64)
    masses = np.array([o.mass for o in spec.objects], dtype=np.float64)
    forces = np.array(
        [o.force if o.force is not None else [0.0] * dim for o in spec.objects],
        dtype=np.float64,
    )
    if spec.gravity is not None:
        g = np.asarray(spec.gravity, dtype=np.float64)
        forces = forces + masses[:, None] * g[None, :]
    return State(
        positions=positions,
        velocities=velocities,
        masses=masses,
        forces=forces,
        ids=[o.id for o in spec.objects],
        t=0.0,
        n_collisions=0,
    )


def acceleration(state: State) -> np.ndarray:
    return state.forces / state.masses[:, None]


def kinetic_energy(state: State) -> float:
    return float(0.5 * np.sum(state.masses * np.sum(state.velocities**2, axis=1)))


def momentum(state: State) -> np.ndarray:
    return np.sum(state.masses[:, None] * state.velocities, axis=0)


def _wall_collision_times(state: State, walls: list[float], dim: int) -> list[tuple[float, str, int, int]]:
    """Return candidate (time, 'wall', obj_idx, axis) events within (0, inf)."""
    events: list[tuple[float, str, int, int]] = []
    # walls layout: [xmin, xmax] or [xmin, xmax, ymin, ymax]
    for i in range(len(state.ids)):
        for axis in range(dim):
            lo = walls[2 * axis]
            hi = walls[2 * axis + 1]
            x = state.positions[i, axis]
            v = state.velocities[i, axis]
            if abs(v) < EPS:
                continue
            if v < 0:
                t = (lo - x) / v
                if t > EPS:
                    events.append((t, "wall", i, axis))
            else:
                t = (hi - x) / v
                if t > EPS:
                    events.append((t, "wall", i, axis))
    return events


def _pair_collision_times(state: State, radius: float = 0.05) -> list[tuple[float, str, int, int]]:
    """1D/2D point masses treated as disks of radius `radius` for collisions."""
    events: list[tuple[float, str, int, int]] = []
    n = len(state.ids)
    for i in range(n):
        for j in range(i + 1, n):
            dx = state.positions[i] - state.positions[j]
            dv = state.velocities[i] - state.velocities[j]
            # solve |dx + t dv|^2 = (2r)^2
            a = float(np.dot(dv, dv))
            if a < EPS:
                continue
            b = 2.0 * float(np.dot(dx, dv))
            c = float(np.dot(dx, dx)) - (2.0 * radius) ** 2
            disc = b * b - 4 * a * c
            if disc < 0:
                continue
            sqrt_disc = np.sqrt(disc)
            for sign in (-1.0, 1.0):
                t = (-b + sign * sqrt_disc) / (2.0 * a)
                if t > EPS:
                    # approaching?
                    if float(np.dot(dx, dv)) < 0:
                        events.append((t, "pair", i, j))
                        break
    return events


def _apply_wall(state: State, i: int, axis: int, restitution: float) -> None:
    state.velocities[i, axis] *= -restitution
    state.n_collisions += 1


def _apply_pair(state: State, i: int, j: int, restitution: float) -> None:
    """Elastic (or partially elastic) collision along contact normal."""
    mi, mj = state.masses[i], state.masses[j]
    vi, vj = state.velocities[i].copy(), state.velocities[j].copy()
    n = state.positions[j] - state.positions[i]
    norm = np.linalg.norm(n)
    if norm < EPS:
        # degenerate: use relative velocity direction
        n = vj - vi
        norm = np.linalg.norm(n)
        if norm < EPS:
            return
    n = n / norm
    vi_n = float(np.dot(vi, n))
    vj_n = float(np.dot(vj, n))
    # relative approach
    if vi_n - vj_n <= 0 and restitution == 1.0:
        # already separating in 1D sense along n; still apply for overlapping cases
        pass
    # 1D elastic formulas along normal
    vi_n_new = (vi_n * (mi - restitution * mj) + vj_n * mj * (1 + restitution)) / (mi + mj)
    vj_n_new = (vj_n * (mj - restitution * mi) + vi_n * mi * (1 + restitution)) / (mi + mj)
    state.velocities[i] = vi + (vi_n_new - vi_n) * n
    state.velocities[j] = vj + (vj_n_new - vj_n) * n
    state.n_collisions += 1


def _advance_free(state: State, dt: float) -> None:
    a = acceleration(state)
    state.positions = state.positions + state.velocities * dt
    state.velocities = state.velocities + a * dt
    state.t += dt


def step_event_driven(state: State, spec: WorldSpec, dt: float, radius: float = 0.05) -> State:
    """Advance up to `dt`, handling the earliest collision if any occurs sooner."""
    remaining = dt
    safety = 0
    while remaining > EPS and safety < 10_000:
        safety += 1
        events: list[tuple[float, str, int, int]] = []
        if spec.walls is not None:
            events.extend(_wall_collision_times(state, spec.walls, spec.dim))
        if spec.collisions and len(state.ids) > 1:
            events.extend(_pair_collision_times(state, radius=radius))
        events = [e for e in events if e[0] <= remaining + EPS]
        if not events:
            _advance_free(state, remaining)
            break
        events.sort(key=lambda e: e[0])
        t_hit, kind, a, b = events[0]
        # nudge slightly before contact to avoid sticking
        t_hit = max(0.0, min(t_hit, remaining))
        if t_hit > EPS:
            _advance_free(state, t_hit)
            remaining -= t_hit
        if kind == "wall":
            _apply_wall(state, a, b, spec.restitution)
        else:
            _apply_pair(state, a, b, spec.restitution)
        # tiny separation to prevent immediate re-collision
        if kind == "pair":
            n = state.positions[b] - state.positions[a]
            norm = np.linalg.norm(n)
            if norm > EPS:
                n = n / norm
                state.positions[a] -= 1e-9 * n
                state.positions[b] += 1e-9 * n
        elif kind == "wall" and spec.walls is not None:
            lo = spec.walls[2 * b]
            hi = spec.walls[2 * b + 1]
            x = state.positions[a, b]
            if abs(x - lo) < abs(x - hi):
                state.positions[a, b] = lo + 1e-9
            else:
                state.positions[a, b] = hi - 1e-9
    return state


def run(spec: WorldSpec | str | dict[str, Any], return_trajectory: bool = False) -> dict[str, Any]:
    """Run the simulator on a WorldSpec / JSON and return the final state dict.

    Matches the proposal interface: `sim.run(spec)`.
    """
    if not isinstance(spec, WorldSpec):
        spec = parse_spec(spec)
    state = _state_from_spec(spec)
    traj: list[dict[str, Any]] = []
    if return_trajectory:
        traj.append(_snapshot(state))
    for _ in range(spec.horizon):
        step_event_driven(state, spec, spec.dt)
        if return_trajectory:
            traj.append(_snapshot(state))
    result = {
        "objects": [
            {
                "id": state.ids[i],
                "position": state.positions[i].tolist(),
                "velocity": state.velocities[i].tolist(),
                "mass": float(state.masses[i]),
            }
            for i in range(len(state.ids))
        ],
        "t": float(state.t),
        "n_collisions": int(state.n_collisions),
        "kinetic_energy": kinetic_energy(state),
        "momentum": momentum(state).tolist(),
    }
    if return_trajectory:
        result["trajectory"] = traj
    return result


def _snapshot(state: State) -> dict[str, Any]:
    return {
        "t": float(state.t),
        "positions": state.positions.tolist(),
        "velocities": state.velocities.tolist(),
        "n_collisions": int(state.n_collisions),
    }


def closed_form_constant_force(
    mass: float,
    x0: list[float] | np.ndarray,
    v0: list[float] | np.ndarray,
    force: list[float] | np.ndarray,
    dt: float,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Semi-implicit? We use the proposal's Euler: v <- v+a dt; x <- x+v_old dt.

    The proposal says:
      a = F/m, v_{t+1} = v_t + a Δt, x_{t+1} = x_t + v_t Δt
    (explicit Euler using old velocity for position).
    """
    x = np.asarray(x0, dtype=np.float64).copy()
    v = np.asarray(v0, dtype=np.float64).copy()
    a = np.asarray(force, dtype=np.float64) / mass
    for _ in range(k):
        x = x + v * dt
        v = v + a * dt
    return x, v


def run_from_objects(
    objects: list[ObjectSpec],
    dim: int,
    dt: float,
    horizon: int,
    walls: list[float] | None = None,
    collisions: bool = True,
) -> dict[str, Any]:
    spec = WorldSpec(
        dim=dim,
        dt=dt,
        horizon=horizon,
        objects=objects,
        walls=walls,
        collisions=collisions,
    )
    return run(spec)


# Re-export for the frozen interface name in the proposal
def parse_and_run(raw: str | dict[str, Any]) -> dict[str, Any]:
    try:
        return run(raw)
    except SpecError:
        raise
