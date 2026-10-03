"""Milliken Moment Method: yaw moment N vs lateral acceleration ay carpet over a (beta, delta) grid.

For every (beta, delta) at speed V the self-consistent lateral acceleration is found by a damped scalar
Newton on  R(ay) = sum(Fy)/m - ay  (yaw rate r = ay/vx couples back into the slip angles). Per iteration the
loads follow from (ax, ay), the slip ratios from the TV-allocated Fx demand (inner Newton). Output ``n`` is
the net yaw moment about the CG [N m]; ``converged`` flags points whose |R| < tol.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ..parameters import G, VehicleParams
from ..tire_interface import solve_kappa_for_fx, tire_forces
from .qss_solver import prepare, totals


class MMMResult(NamedTuple):
    v: float
    beta: np.ndarray     # (nb,) [rad]
    delta: np.ndarray    # (nd,) [rad]
    ay: np.ndarray       # (nb, nd) [m/s^2]
    ax: np.ndarray       # achieved ax [m/s^2]
    n: np.ndarray        # (nb, nd) yaw moment [N m]
    converged: np.ndarray


def _eval_ay(vp: VehicleParams, V, beta, delta, ay, ax, mz_tv, drs):
    pr = prepare(vp, V, beta, delta, ay, ax, mz_tv, drs)
    kappa = solve_kappa_for_fx(vp, pr.alpha, pr.gamma, pr.loads.fz, pr.tv.fx_cmd)
    t = tire_forces(vp, pr.alpha, kappa, pr.gamma, pr.loads.fz, pr.steer, pr.re)
    ay_c, ax_c, mz = totals(vp, pr, t, V)
    return ay_c, ax_c, mz


def mmm_point(vp: VehicleParams, V, beta, delta, ax=0.0, mz_tv=0.0, drs=0.0, n_iter: int = 20, tol: float = 5e-2):
    """Return (ay, ax_achieved, N, converged) for one (beta, delta)."""

    def R(ay):
        return _eval_ay(vp, V, beta, delta, ay, ax, mz_tv, drs)[0] - ay

    def body(_, ay):
        f, df = jax.jvp(R, (ay,), (jnp.ones_like(ay),))
        step = -f / jnp.where(jnp.abs(df) < 0.05, -0.05, df)       # df ~ -1 + (small coupling)
        return ay + jnp.clip(0.8 * step, -4.0, 4.0)

    ay = jax.lax.fori_loop(0, n_iter, body, jnp.asarray(0.0))
    ay_c, ax_c, mz = _eval_ay(vp, V, beta, delta, ay, ax, mz_tv, drs)
    return ay, ax_c, mz, jnp.abs(ay_c - ay) < tol


def mmm_grid(vp: VehicleParams, V, betas_deg=np.linspace(-8, 8, 17), deltas_deg=np.linspace(-14, 14, 15),
             ax=0.0, mz_tv=0.0, drs=0.0) -> MMMResult:
    b = jnp.radians(jnp.asarray(betas_deg, dtype=float))
    d = jnp.radians(jnp.asarray(deltas_deg, dtype=float))
    f = lambda bb, dd: mmm_point(vp, V, bb, dd, ax, mz_tv, drs)
    ay, axa, n, ok = jax.jit(jax.vmap(jax.vmap(f, (None, 0)), (0, None)))(b, d)
    return MMMResult(float(V), np.asarray(b), np.asarray(d), np.asarray(ay), np.asarray(axa), np.asarray(n),
                     np.asarray(ok))


def mmm_metrics(res: MMMResult) -> dict:
    """Limit lateral acceleration, stability and control derivatives from the carpet (central differences)."""
    ok = res.converged
    ay = np.where(ok, res.ay, np.nan)
    n = np.where(ok, res.n, np.nan)
    ib0 = int(np.argmin(np.abs(res.beta)))
    id0 = int(np.argmin(np.abs(res.delta)))
    out = {
        "ay_max_g": float(np.nanmax(np.abs(ay)) / G) if np.isfinite(ay).any() else float("nan"),
        # stability: dN/dbeta at delta=0, ay~0; control: dN/ddelta at beta=0
        "dN_dbeta_Nm_per_deg": float(np.gradient(n[:, id0], np.degrees(res.beta))[ib0]),
        "dN_ddelta_Nm_per_deg": float(np.gradient(n[ib0, :], np.degrees(res.delta))[id0]),
        "d_ay_ddelta_g_per_deg": float(np.gradient(ay[ib0, :], np.degrees(res.delta))[id0] / G),
    }
    return out


def plot_mmm(res: MMMResult, ax=None, show_unconverged: bool = False):
    """Carpet plot in (ay/g, N): constant-delta lines (solid) and constant-beta lines (dashed)."""
    import matplotlib.pyplot as plt

    ax = ax or plt.subplots(figsize=(7, 6))[1]
    g = res.ay / G
    ok = res.converged | show_unconverged
    for j, d in enumerate(np.degrees(res.delta)):
        m = ok[:, j]
        ax.plot(g[m, j], res.n[m, j], "-", lw=0.9, color="#58a6ff", alpha=0.8)
        if m.any():
            ax.annotate(f"{d:+.0f}°", (g[m, j][-1], res.n[m, j][-1]), fontsize=6, color="#58a6ff")
    for i, bdeg in enumerate(np.degrees(res.beta)):
        m = ok[i, :]
        ax.plot(g[i, m], res.n[i, m], "--", lw=0.9, color="#f78166", alpha=0.8)
        if m.any():
            ax.annotate(f"{bdeg:+.0f}°", (g[i, m][-1], res.n[i, m][-1]), fontsize=6, color="#f78166")
    ax.axhline(0, color="gray", lw=0.6)
    ax.axvline(0, color="gray", lw=0.6)
    ax.set(xlabel="ay [g]", ylabel="N [N m]", title=f"MMM — V = {res.v * 3.6:.0f} km/h  (solid: δ, dashed: β)")
    ax.grid(alpha=0.25)
    return ax