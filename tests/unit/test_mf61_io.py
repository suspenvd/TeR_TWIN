"""Tests for mf61_io.py – vectorized MF6.1 PyTree stacking and vmap evaluation.

Run with:
    pytest tests/unit/test_mf61_io.py -v
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from ter_twin.models.tires import pacejka_61 as mf
from ter_twin.models.tires import mf61_io as io_


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
N_WHEELS = 4


@pytest.fixture(scope="module")
def scalar_params() -> list[mf.MF61Params]:
    """4 wheel corners with slightly different peak friction to tell them apart."""
    base = mf.MF61Params()
    return [
        base.replace(pDy1=2.4 + 0.05 * i, pDx1=2.2 + 0.03 * i)
        for i in range(N_WHEELS)
    ]


@pytest.fixture(scope="module")
def stacked(scalar_params) -> mf.MF61Params:
    return io_.stack_params(scalar_params)


# ---------------------------------------------------------------------------
# stack_params / unstack_params structural tests
# ---------------------------------------------------------------------------
class TestStackUnstack:
    def test_stacked_leaves_are_arrays(self, stacked):
        """Every leaf in the stacked pytree must be a JAX array of shape (N,)."""
        leaves = jax.tree_util.tree_leaves(stacked)
        assert len(leaves) == len(mf.NAMES)
        for leaf in leaves:
            a = np.asarray(leaf)
            assert a.ndim == 1, f"Expected 1-D leaf, got shape {a.shape}"
            assert a.shape[0] == N_WHEELS

    def test_roundtrip(self, scalar_params, stacked):
        """unstack_params should recover each original scalar MF61Params."""
        recovered = io_.unstack_params(stacked)
        assert len(recovered) == N_WHEELS
        for orig, rec in zip(scalar_params, recovered):
            for k in mf.NAMES:
                assert float(getattr(orig, k)) == pytest.approx(float(getattr(rec, k)), rel=1e-9), \
                    f"Mismatch on {k}"

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            io_.stack_params([])

    def test_single_wheel(self):
        P = mf.MF61Params()
        stk = io_.stack_params([P])
        leaves = jax.tree_util.tree_leaves(stk)
        assert all(np.asarray(l).shape == (1,) for l in leaves)
        rec = io_.unstack_params(stk)
        assert len(rec) == 1
        assert rec[0].to_dict() == pytest.approx(P.to_dict(), rel=1e-9)


# ---------------------------------------------------------------------------
# vmap evaluation tests
# ---------------------------------------------------------------------------
class TestVmapEvaluation:
    """Validate that batch evaluation via vmap gives identical results to
    separate scalar evaluate() calls."""

    def test_vmap_shape(self, stacked):
        alpha = jnp.full((N_WHEELS,), 0.1)
        kappa = jnp.zeros((N_WHEELS,))
        gamma = jnp.zeros((N_WHEELS,))
        fz = jnp.full((N_WHEELS,), 1100.0)

        result = io_.batch_evaluate(stacked, alpha, kappa, gamma, fz)

        for field in mf.MF61Result._fields:
            arr = np.asarray(getattr(result, field))
            assert arr.shape == (N_WHEELS,), \
                f"Field {field}: expected shape ({N_WHEELS},), got {arr.shape}"

    def test_vmap_matches_scalar(self, scalar_params, stacked):
        """vmap results must match individual scalar evaluations to float64 precision."""
        alpha = jnp.array([0.05, 0.10, -0.05, 0.00])
        kappa = jnp.array([0.0, 0.05, 0.0, -0.10])
        gamma = jnp.array([0.0, math.radians(2), 0.0, 0.0])
        fz = jnp.array([900.0, 1100.0, 1300.0, 700.0])

        batch = io_.batch_evaluate(stacked, alpha, kappa, gamma, fz)

        for i, P in enumerate(scalar_params):
            ref = mf.evaluate(P, float(alpha[i]), float(kappa[i]), float(gamma[i]), float(fz[i]))
            for field in ("fx", "fy", "mz"):
                got = float(np.asarray(getattr(batch, field))[i])
                exp = float(np.asarray(getattr(ref, field)))
                assert got == pytest.approx(exp, rel=1e-6), \
                    f"Corner {i}, field {field}: got {got:.4f}, expected {exp:.4f}"

    def test_vmap_with_pressure(self, scalar_params, stacked):
        """Same test but passing an explicit pressure array."""
        alpha = jnp.full((N_WHEELS,), 0.08)
        kappa = jnp.zeros((N_WHEELS,))
        gamma = jnp.zeros((N_WHEELS,))
        fz = jnp.full((N_WHEELS,), 1100.0)
        p = jnp.array([80.0, 85.0, 90.0, 75.0])

        batch = io_.batch_evaluate(stacked, alpha, kappa, gamma, fz, p)

        for i, P in enumerate(scalar_params):
            ref = mf.evaluate(P, float(alpha[i]), 0.0, 0.0, float(fz[i]), float(p[i]))
            got = float(np.asarray(batch.fy)[i])
            exp = float(np.asarray(ref.fy))
            assert got == pytest.approx(exp, rel=1e-6), f"Corner {i} Fy mismatch (pressure)"

    def test_jit_compile(self, stacked):
        """Ensure the vmapped evaluate compiles under jax.jit without errors."""
        @jax.jit
        def _f(P, a, k, g, f):
            return io_.batch_evaluate(P, a, k, g, f)

        a = jnp.zeros((N_WHEELS,))
        r = _f(stacked, a + 0.1, a, a, a + 1100.0)
        assert np.all(np.isfinite(np.asarray(r.fy)))

    def test_grad_through_stack(self, stacked):
        """Gradient of sum(Fy) wrt stacked parameter leaf should be finite."""
        alpha = jnp.full((N_WHEELS,), 0.1)
        kappa = jnp.zeros((N_WHEELS,))
        gamma = jnp.zeros((N_WHEELS,))
        fz = jnp.full((N_WHEELS,), 1100.0)

        def loss(P):
            r = io_.batch_evaluate(P, alpha, kappa, gamma, fz)
            return jnp.sum(r.fy)

        # Use jacfwd since we have many parameters; just check it doesn't crash
        grads = jax.jacfwd(loss)(stacked)
        flat_grads = jax.tree_util.tree_leaves(grads)
        for g in flat_grads:
            assert np.all(np.isfinite(np.asarray(g))), "Non-finite gradient detected"


# ---------------------------------------------------------------------------
# load_json tests
# ---------------------------------------------------------------------------
class TestLoadJson:
    def test_load_json_ok_only(self, tmp_path):
        """load_json skips entries without status=='ok' or without YAML files."""
        P = mf.MF61Params(pDy1=2.5)
        yaml_path = tmp_path / "test_tire.yaml"
        P.save(yaml_path, {"tire_name": "TestTire"})

        index = {
            "version": 1,
            "tires": [
                {"slug": "ok_tire", "status": "ok", "params": "test_tire.yaml",
                 "tire_name": "TestTire", "rim_width_in": 7.0},
                {"slug": "bad_tire", "status": "error: something", "params": None,
                 "tire_name": "BadTire", "rim_width_in": 7.0},
                {"slug": "missing_yaml", "status": "ok", "params": "nonexistent.yaml",
                 "tire_name": "GhostTire", "rim_width_in": 7.0},
            ],
        }
        index_path = tmp_path / "index.json"
        index_path.write_text(json.dumps(index), encoding="utf-8")

        result = io_.load_json(index_path)
        assert len(result) == 1, "Should only return the 1 valid 'ok' entry with existing YAML"
        assert result[0].pDy1 == pytest.approx(2.5)

    def test_load_json_empty(self, tmp_path):
        index = {"version": 1, "tires": []}
        index_path = tmp_path / "index.json"
        index_path.write_text(json.dumps(index), encoding="utf-8")
        result = io_.load_json(index_path)
        assert result == []


# ---------------------------------------------------------------------------
# Dimension-compatibility checks (per task specification)
# ---------------------------------------------------------------------------
class TestVmapDimensionChecks:
    """Explicit vmap dimension checks as requested in the verification checklist."""

    @pytest.mark.parametrize("n_wheels", [1, 2, 4, 6])
    def test_arbitrary_batch_size(self, n_wheels):
        params = [mf.MF61Params() for _ in range(n_wheels)]
        stk = io_.stack_params(params)
        leaves = jax.tree_util.tree_leaves(stk)
        assert all(np.asarray(l).shape == (n_wheels,) for l in leaves)

        alpha = jnp.zeros((n_wheels,))
        fz = jnp.full((n_wheels,), 1100.0)
        result = io_.batch_evaluate(stk, alpha, alpha, alpha, fz)
        assert np.asarray(result.fy).shape == (n_wheels,)

    def test_pytree_treedef_preserved(self):
        """The treedef of the stacked params should match the treedef structure."""
        P = mf.MF61Params()
        _, treedef_scalar = jax.tree_util.tree_flatten(P)
        stk = io_.stack_params([P, P])
        _, treedef_stacked = jax.tree_util.tree_flatten(stk)
        # Both should have the same number of leaves (same param set)
        assert treedef_scalar.num_leaves == treedef_stacked.num_leaves
