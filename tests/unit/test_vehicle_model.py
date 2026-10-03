"""Tests for ter_twin.models.vehicle (run: pytest tests/unit/test_vehicle_model.py -v)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from ter_twin.models import vehicle as V  # noqa: E402
from ter_twin.models.vehicle.load_transfer import steady_roll, wheel_loads  # noqa: E402
from ter_twin.models.vehicle.parameters import G  # noqa: E402
from ter_twin.models.vehicle.powertrain import tv_allocate  # noqa: E402
from ter_twin.models.vehicle.solvers import mmm_grid, solve_trim  # noqa: E402

vp = V.default_ter27()


def test_load_conservation():
    ls = wheel_loads(vp, 20.0, 3.0, 8.0, steady_roll(vp, 8.0), 0.0)
    expect = vp.chassis.mass * G + ls.aero.df_f + ls.aero.df_r
    assert float(jnp.sum(ls.fz)) == pytest.approx(float(expect), rel=1e-4)


def test_numpy_fallback_matches_jax():
    phi = steady_roll(vp, 6.0)
    a = wheel_loads(vp, 18.0, -2.0, 6.0, phi, 0.0, xp=np).fz
    b = wheel_loads(vp, 18.0, -2.0, 6.0, phi, 0.0).fz
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-10)


def test_lateral_transfer_to_outer_wheels():
    fz = np.asarray(wheel_loads(vp, 15.0, 0.0, 10.0, steady_roll(vp, 10.0), 0.0).fz)
    assert fz[1] > fz[0] and fz[3] > fz[2]


def test_trim_converges_and_matches_target():
    t = solve_trim(vp, 15.0, 6.0)
    assert bool(t.converged) and float(t.ay) == pytest.approx(6.0, abs=1e-3)
    assert float(t.delta) > 0.0
    drag = float(wheel_loads(vp, 15.0, 0.0, 6.0, steady_roll(vp, 6.0), 0.0).aero.drag)
    assert float(np.sum(np.asarray(t.fx_w))) == pytest.approx(drag, rel=0.05)  # ax = 0 -> tyres cancel drag


def test_trim_is_differentiable():
    g = jax.grad(lambda m: solve_trim(V.replace_in(vp, "chassis", mass=m), 15.0, 6.0).delta)(290.0)
    assert np.isfinite(float(g))


def test_tv_allocation_hits_request_and_power_cap():
    fz = jnp.array([700.0, 900.0, 800.0, 1000.0])
    re = jnp.full(4, 0.19)
    w = jnp.full(4, 60.0)
    out = tv_allocate(vp, 1200.0, 300.0, fz, w, re)
    assert float(out.fx_total) == pytest.approx(1200.0, rel=1e-3)
    assert float(out.mz_total) == pytest.approx(300.0, rel=1e-2)
    big = tv_allocate(vp, 8000.0, 0.0, fz, w, re)
    assert float(big.p_elec) <= vp.pt.p_acc_max * 1.001


def test_dynamic_agrees_with_trim():
    t = solve_trim(vp, 15.0, 4.0)
    x0 = V.initial_state(vp, 15.0)
    fn = lambda time, x: V.zero_controls(t.delta)
    _, X = V.simulate(vp, x0, fn, 1e-3, 1500)
    assert bool(jnp.all(jnp.isfinite(X)))
    assert float(X[-1, 2] * X[-1, 0]) == pytest.approx(4.0, rel=0.25)  # coasting loses a little speed


def test_mmm_grid_shapes():
    m = mmm_grid(vp, 15.0, np.linspace(-4, 4, 5), np.linspace(-8, 8, 5))
    assert m.ay.shape == (5, 5) and m.converged.mean() > 0.8