"""Quasi-steady-state trim: for given speed V, lateral acceleration ay and longitudinal acceleration ax
find body slip beta, mean steer delta and the four slip ratios kappa_i such that

    sum(Fy)/m = ay ,   sum(Mz) = 0 ,   Fx_w,i = Fx_cmd,i  (i = 1..4)

Wheel loads follow from (V, ax, ay) alone (steady roll + load transfer), so they are computed once.
Fx_cmd comes from the plant-side TV allocator (equal tyre utilisation; ``mz_tv`` adds a yaw request).
Newton with Levenberg damping and step limits, fixed iteration count -> jit/vmap/grad friendly.
Convergence is reported (``converged``); near the grip limit Newton can stall - use ``mmm_diagram``.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from ..kinematics import camber_toe, steer_angles
from ..load_transfer import LoadState, steady_roll, wheel_loads
from ..parameters import VehicleParams
from ..powertrain import TVOut, tv_allocate
from ..tire_interface import (TireOut, body_loads, tire_forces,
                              wheel_slips)


class Prep(NamedTuple):
    """Everything that does not depend on the slip unknowns' Newton state beyond (beta, delta)."""
    loads: LoadState
    phi: object
    steer: object
    gamma: object
    alpha: object
    vxw: object
    re: object
    vx: object
    vy: object
    r: object
    tv: TVOut


def prepare(vp: VehicleParams, V, beta, delta, ay, ax, mz_tv=0.0, drs=0.0) -> Prep:
    ch = vp.chassis
    vx, vy = V * jnp.cos(beta), V * jnp.sin(beta)
    r = ay / vx                                           # steady turn: ay = vx*r (vy' = 0)
    phi = steady_roll(vp, ay)
    ld = wheel_loads(vp, V, ax, ay, phi, 0.0, drs)
    gamma, st_toe = camber_toe(vp, phi, ld.dz)
    steer = steer_angles(vp, delta) + st_toe
    alpha, vxw, re = wheel_slips(vp, vx, vy, r, steer, ld.fz)
    fx_req = ch.mass * ax + ld.aero.drag * vx / V
    tv = tv_allocate(vp, fx_req, mz_tv, ld.fz, vxw / re, re)
    return Prep(ld, phi, steer, gamma, alpha, vxw, re, vx, vy, r, tv)


def totals(vp: VehicleParams, pr: Prep, t: TireOut, V):
    """(ay_c, ax_c, Mz) of the whole car including aero drag."""
    fx, fy, mz = body_loads(vp, t)
    D = pr.loads.aero.drag
    m = vp.chassis.mass
    return (fy - D * pr.vy / V) / m, (fx - D * pr.vx / V) / m, mz


class TrimResult(NamedTuple):
    beta: object
    delta: object
    kappa: object
    r: object
    ay: object
    ax: object
    fz: object
    alpha: object
    gamma: object
    fy_w: object
    fx_w: object
    mux: object
    muy: object
    phi: object
    theta: object
    rh_f: object
    rh_r: object
    residual: object
    converged: object


def solve_trim(vp: VehicleParams, V, ay, ax=0.0, mz_tv=0.0, drs=0.0, n_iter: int = 30,
               guess: tuple | None = None, tol: float = 1e-2) -> TrimResult:
    ch = vp.chassis
    beta0, delta0 = (0.0, ch.wheelbase * ay / (V * V)) if guess is None else guess
    s0 = jnp.concatenate([jnp.array([beta0, delta0]), jnp.zeros(4)])

    def residual(s):
        pr = prepare(vp, V, s[0], s[1], ay, ax, mz_tv, drs)
        t = tire_forces(vp, pr.alpha, s[2:], pr.gamma, pr.loads.fz, pr.steer, pr.re)
        ay_c, _, mz = totals(vp, pr, t, V)
        m = ch.mass
        return jnp.concatenate([jnp.array([ay_c - ay, mz / (m * ch.wheelbase)]),
                                (t.fx_w - pr.tv.fx_cmd) / m])

    lim = jnp.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05])

    def body(_, s):
        R = residual(s)
        J = jax.jacfwd(residual)(s)
        ds = jnp.linalg.solve(J + 1e-4 * jnp.eye(6), -R)
        return s + jnp.clip(ds, -lim, lim)

    s = jax.lax.fori_loop(0, n_iter, body, s0)
    R = residual(s)
    pr = prepare(vp, V, s[0], s[1], ay, ax, mz_tv, drs)
    t = tire_forces(vp, pr.alpha, s[2:], pr.gamma, pr.loads.fz, pr.steer, pr.re)
    ay_c, ax_c, _ = totals(vp, pr, t, V)
    nrm = jnp.linalg.norm(R)
    return TrimResult(s[0], s[1], s[2:], pr.r, ay_c, ax_c, pr.loads.fz, pr.alpha, pr.gamma, t.fy_w, t.fx_w,
                      t.mux, t.muy, pr.phi, pr.loads.theta, pr.loads.rh_f, pr.loads.rh_r, nrm, nrm < tol)


def trim_sweep(vp: VehicleParams, V, ay_grid, ax=0.0, **kw):
    """vmap of :func:`solve_trim` over a lateral-acceleration grid (cold start per point)."""
    return jax.vmap(lambda a: solve_trim(vp, V, a, ax, **kw))(jnp.asarray(ay_grid))


def max_lateral_acceleration(vp: VehicleParams, V, ax=0.0, ay_hi=25.0, n_bisect: int = 14, **kw):
    """Largest ay with a converged trim (bisection on the converged flag; coarse, use MMM for the limit curve)."""
    def body(_, lohi):
        lo, hi = lohi
        mid = 0.5 * (lo + hi)
        ok = solve_trim(vp, V, mid, ax, **kw).converged
        return jnp.where(ok, mid, lo), jnp.where(ok, hi, mid)

    lo, _ = jax.lax.fori_loop(0, n_bisect, body, (jnp.asarray(1.0), jnp.asarray(ay_hi)))
    return lo