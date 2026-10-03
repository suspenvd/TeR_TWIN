import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from ter_twin.models.tires import pacejka_61 as mf  # noqa: E402

P = mf.MF61Params()
FZ = 1100.0


def test_zero_load_zero_forces():
    r = mf.mf61(P, 0.05, 0.05, 0.02, 0.0)
    for v in (r.fx, r.fy, r.mz):
        assert abs(float(v)) < 1e-6


def test_lateral_odd_symmetry_at_zero_camber():
    a = jnp.linspace(0.01, 0.3, 20)
    np.testing.assert_allclose(mf.lateral_pure_force(P, -a, FZ), -mf.lateral_pure_force(P, a, FZ), atol=1e-8)


def test_cornering_stiffness_matches_slope():
    d = 1e-6
    slope = (mf.lateral_pure_force(P, d, FZ) - mf.lateral_pure_force(P, -d, FZ)) / (2 * d)
    kya = mf.mf61(P, 0.0, 0.0, 0.0, FZ).kya
    np.testing.assert_allclose(slope, kya, rtol=1e-3)


def test_combined_reduces_to_pure():
    a = jnp.linspace(-0.2, 0.2, 9)
    r = mf.mf61(P, a, 0.0, 0.03, FZ)
    np.testing.assert_allclose(r.fy, r.fy0, rtol=1e-9, atol=1e-6)
    k = jnp.linspace(-0.2, 0.2, 9)
    r = mf.mf61(P, 0.0, k, 0.0, FZ)
    np.testing.assert_allclose(r.fx, r.fx0, rtol=1e-9, atol=1e-6)


def test_friction_ellipse_reduction():
    base = abs(float(mf.mf61(P, 0.12, 0.0, 0.0, FZ).fy))
    comb = abs(float(mf.mf61(P, 0.12, 0.12, 0.0, FZ).fy))
    assert comb < base


def test_mz_sign_and_pure_vs_combined():
    r = mf.mf61(P, 0.02, 0.0, 0.0, FZ)
    assert float(r.mz) < 0 < float(r.fy)
    np.testing.assert_allclose(mf.aligning_pure_moment(P, 0.02, FZ), r.mz, rtol=1e-9)


def test_gradients_finite_everywhere():
    g = jax.jacfwd(lambda x: mf.mf61(P, x[0], x[1], x[2], x[3]).mz)(jnp.array([0.0, 0.0, 0.0, FZ]))
    assert np.all(np.isfinite(np.asarray(g)))
    g = jax.grad(lambda a: mf.mf61(P, a, 0.0, 0.0, 0.0).fy)(0.0)
    assert np.isfinite(float(g))


def test_pytree_and_jit_roundtrip(tmp_path):
    leaves, tree = jax.tree_util.tree_flatten(P)
    P2 = jax.tree_util.tree_unflatten(tree, leaves)
    assert P2.to_dict() == P.to_dict()
    f = jax.jit(lambda pp, a: mf.lateral_pure_force(pp, a, FZ))
    np.testing.assert_allclose(f(P, 0.05), mf.lateral_pure_force(P, 0.05, FZ))
    path = tmp_path / "t.yaml"
    P.replace(pDy1=2.7).save(path, {"tire_name": "x"})
    P3, meta = mf.MF61Params.load_with_meta(path)
    assert P3.pDy1 == pytest.approx(2.7) and meta["tire_name"] == "x"


def test_pressure_and_camber_effects():
    Pp = P.replace(ppy3=0.3)
    hi = abs(float(mf.lateral_pure_force(Pp, 0.15, FZ, 0.0, P.NOMPRES * 1.2)))
    lo = abs(float(mf.lateral_pure_force(Pp, 0.15, FZ, 0.0, P.NOMPRES * 0.8)))
    assert hi > lo
    assert float(mf.mf61(P, 0.0, 0.0, math.radians(3), FZ).fy) != 0.0  # empuje de camber


def test_summary_and_compare():
    P8 = P.replace(pKy1=P.pKy1 * 1.1)
    out = mf.compare_params(P, P8, [600, 1100, 1600])
    assert out["mean_delta_pct"]["c_alpha_N_deg"] == pytest.approx(10.0, rel=1e-3)
    s = mf.summary(P, [600.0, 1100.0])
    assert s["mu_y_peak"].shape == (2,) and s["mu_y_peak"][0] > s["mu_y_peak"][1]