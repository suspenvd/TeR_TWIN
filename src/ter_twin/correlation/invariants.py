"""Physical-invariant harness. Run BEFORE trusting any correlation number.
1. load conservation  sum Fz = m g + Fz_aero   (static and dynamic)
2. exact bilateral anti-symmetry under (delta, vy, r, phi, p, ay) -> -(...) with left/right wheel swap
3. passivity: coasting dual-track kinetic energy never increases; generic port-Hamiltonian check
   dH/dt = -grad(H)^T R grad(H) <= 0 and R symmetric PSD (for the 108-DOF H_net / R_net).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

G = 9.80665
__all__ = ["InvariantResult", "run_invariants", "check_ph_passivity"]


@dataclass(frozen=True)
class InvariantResult:
    name: str
    passed: bool
    error: float
    tol: float
    detail: str = ""


def _load_conservation(vp, tol: float) -> list[InvariantResult]:
    from ter_twin.models.suspension.load_transfer import steady_roll, wheel_loads

    m = vp.chassis.mass
    res = []
    ls = wheel_loads(vp, 0.0, 0.0, 0.0, 0.0, 0.0)
    e = abs(float(np.sum(ls.fz)) - m * G) / (m * G)
    res.append(InvariantResult("load_conservation_static", e < tol, e, tol))
    for v, ax, ay in ((20.0, 3.0, 8.0), (25.0, -9.0, 5.0), (12.0, 0.0, -10.0)):
        ls = wheel_loads(vp, v, ax, ay, steady_roll(vp, ay), 0.0)
        exp = m * G + float(ls.aero.df_f + ls.aero.df_r)
        e = abs(float(np.sum(ls.fz)) - exp) / exp
        res.append(InvariantResult(f"load_conservation_dyn(v={v:g},ax={ax:g},ay={ay:g})", e < tol, e, tol))
    return res


def _antisymmetry(vp, tol: float) -> list[InvariantResult]:
    import jax.numpy as jnp

    from ter_twin.models import vehicle as V

    sw = np.array([1, 0, 3, 2])
    x0 = np.asarray(V.initial_state(vp, 20.0))
    w = x0[5:9] * (1.0 + np.array([0.01, 0.02, -0.01, 0.0]))
    x = np.concatenate([[20.0, 0.4, 0.25, 0.02, 0.1], w, [1.0, 6.0]])
    xm = np.concatenate([[20.0, -0.4, -0.25, -0.02, -0.1], w[sw], [1.0, -6.0]])
    td = np.array([50.0, 30.0, 80.0, 20.0])
    u = V.Controls(jnp.asarray(0.05), jnp.asarray(td), jnp.zeros(4), jnp.asarray(0.0))
    um = V.Controls(jnp.asarray(-0.05), jnp.asarray(td[sw]), jnp.zeros(4), jnp.asarray(0.0))
    a, b = V.compute_forces(vp, jnp.asarray(x), u), V.compute_forces(vp, jnp.asarray(xm), um)
    pairs = {"fx": (float(a.fx), float(b.fx)), "fy": (float(a.fy), -float(b.fy)), "mz": (float(a.mz), -float(b.mz))}
    res = []
    for k, (p, q) in pairs.items():
        e = abs(p - q) / max(abs(p), 1.0)
        res.append(InvariantResult(f"antisymmetry_{k}", e < tol, e, tol,
                                   "non-zero => asymmetric tyre coefficients (pHy/pVy/pEy3) or alignment params" if e >= tol else ""))
    fz_a, fz_b = np.asarray(a.loads.fz), np.asarray(b.loads.fz)[sw]
    e = float(np.max(np.abs(fz_a - fz_b)) / np.max(fz_a))
    res.append(InvariantResult("antisymmetry_wheel_loads", e < tol, e, tol))
    return res


def _coast_passivity(vp, tol: float) -> InvariantResult:
    from ter_twin.models import vehicle as V

    x0 = V.initial_state(vp, 20.0)
    _, X = V.simulate(vp, x0, lambda t, x: V.zero_controls(0.0), 2e-3, 750)
    X = np.asarray(X)
    m, iw = vp.chassis.mass, vp.tire.inertia_w
    E = 0.5 * m * X[:, 0] ** 2 + 0.5 * iw * np.sum(X[:, 5:9] ** 2, axis=1)
    rise = float(np.max(np.diff(E))) / float(E[0])
    return InvariantResult("coasting_energy_nonincreasing", rise <= tol, rise, tol,
                           f"E0={E[0]:.1f} J -> E_end={E[-1]:.1f} J")


def run_invariants(vp, tol_load: float = 5e-4, tol_sym: float = 0.1, tol_energy: float = 1e-4) -> list[InvariantResult]:
    return [*_load_conservation(vp, tol_load), *_antisymmetry(vp, tol_sym), _coast_passivity(vp, tol_energy)]


def check_ph_passivity(H: Callable, R: Callable, q0: np.ndarray, p0: np.ndarray, q_scale: np.ndarray,
                       p_scale: np.ndarray, n: int = 64, tol: float = 1e-6, seed: int = 0) -> list[InvariantResult]:
    """Unforced PH system  x' = (J - R) grad H, J skew:  dH/dt = -g^T R g.  H(q,p)->scalar, R(q,p)->(d,d)."""
    import jax
    import jax.numpy as jnp

    rng = np.random.default_rng(seed)
    gH = jax.jit(jax.grad(H, argnums=(0, 1)))
    worst_dH, worst_eig, worst_sym = -np.inf, np.inf, 0.0
    for _ in range(n):
        q = jnp.asarray(q0 + q_scale * rng.standard_normal(q0.shape))
        p = jnp.asarray(p0 + p_scale * rng.standard_normal(p0.shape))
        gq, gp = gH(q, p)
        g = np.concatenate([np.asarray(gq), np.asarray(gp)])
        Rm = np.asarray(R(q, p))
        Rf = np.zeros((g.size, g.size))
        Rf[gq.size:, gq.size:] = Rm                       # dissipation acts on the momentum block
        worst_dH = max(worst_dH, float(-g @ Rf @ g))
        worst_eig = min(worst_eig, float(np.linalg.eigvalsh(0.5 * (Rm + Rm.T)).min()))
        worst_sym = max(worst_sym, float(np.max(np.abs(Rm - Rm.T))))
    sc = max(1.0, abs(worst_dH))
    return [InvariantResult("ph_dHdt_nonpositive", worst_dH <= tol * sc, worst_dH, tol),
            InvariantResult("ph_R_psd", worst_eig >= -tol, worst_eig, tol),
            InvariantResult("ph_R_symmetric", worst_sym <= tol, worst_sym, tol)]