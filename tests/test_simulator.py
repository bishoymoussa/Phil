"""Acceptance tests for the Newtonian simulator (proposal A1/A3)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from phil.sim.simulator import closed_form_constant_force, kinetic_energy, momentum, run
from phil.sim.spec import ObjectSpec, SpecError, WorldSpec, parse_spec


def test_closed_form_match_1000():
    rng = np.random.default_rng(0)
    max_err = 0.0
    for _ in range(1000):
        m = float(rng.uniform(1, 10))
        x0 = rng.uniform(-5, 5, size=1)
        v0 = rng.uniform(-10, 10, size=1)
        f = rng.uniform(-20, 20, size=1)
        dt = 0.1
        k = int(rng.integers(1, 6))
        spec = WorldSpec(
            dim=1,
            dt=dt,
            horizon=k,
            objects=[ObjectSpec(id="o0", mass=m, position=x0.tolist(), velocity=v0.tolist(), force=f.tolist())],
            collisions=False,
        )
        result = run(spec)
        x_cf, v_cf = closed_form_constant_force(m, x0, v0, f, dt, k)
        err = max(
            abs(result["objects"][0]["position"][0] - float(x_cf[0])),
            abs(result["objects"][0]["velocity"][0] - float(v_cf[0])),
        )
        max_err = max(max_err, err)
    assert max_err < 1e-9, max_err


def test_elastic_collision_conserves_momentum_energy():
    spec = WorldSpec(
        dim=1,
        dt=0.01,
        horizon=200,
        objects=[
            ObjectSpec(id="a", mass=2.0, position=[-1.0], velocity=[1.0], force=[0.0]),
            ObjectSpec(id="b", mass=1.0, position=[1.0], velocity=[-0.5], force=[0.0]),
        ],
        collisions=True,
        walls=None,
        restitution=1.0,
    )
    s0 = {
        "masses": np.array([2.0, 1.0]),
        "vel": np.array([[1.0], [-0.5]]),
    }
    ke0 = 0.5 * (2.0 * 1.0**2 + 1.0 * 0.5**2)
    p0 = 2.0 * 1.0 + 1.0 * (-0.5)
    result = run(spec)
    assert result["n_collisions"] >= 1
    ke1 = result["kinetic_energy"]
    p1 = result["momentum"][0]
    assert abs(ke1 - ke0) < 1e-6
    assert abs(p1 - p0) < 1e-6


def test_gold_spec_roundtrip():
    raw = {
        "dim": 1,
        "dt": 0.1,
        "horizon": 3,
        "objects": [
            {"id": "o0", "mass": 2.0, "position": [0.0], "velocity": [1.0], "force": [0.0]}
        ],
        "collisions": False,
    }
    spec = parse_spec(json.dumps(raw))
    result = run(spec)
    result2 = run(spec.to_dict())
    assert result["objects"][0]["position"] == result2["objects"][0]["position"]


def test_parse_sim_wrapper():
    text = '<sim>{"dim":1,"dt":0.1,"horizon":1,"objects":[{"id":"o0","mass":1,"position":[0],"velocity":[0],"force":[1]}]}</sim>'
    spec = parse_spec(text)
    assert spec.horizon == 1
    out = run(spec)
    assert abs(out["objects"][0]["velocity"][0] - 0.1) < 1e-9


def test_invalid_spec_raises():
    with pytest.raises(SpecError):
        parse_spec('{"dim":1,"dt":0.1,"horizon":1,"objects":[]}')
