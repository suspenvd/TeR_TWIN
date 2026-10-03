"""Body states -> wheel slips (alpha, kappa, gamma) -> MF6.1 forces -> body-frame loads.

The tyre model (``pacejka_61.mf61``) is fully broadcastable, so the four corners are evaluated in ONE
call by stacking front/rear ``MF61Params`` leaves into (4,) arrays (same result as
``mf61_io.stack_params([F, F, R, R])`` + ``vmap``, without the vmap).
Sign convention: alpha = -atan(vy_w/vx_w) (positive-slope model), kappa = (w*Re - vx_w)/|vx_w|.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from ter_twin.models.tires import pacejka_61 as mf

from .parameters import VehicleParams

V_EPS = 0.5  # low-speed regularisation of the slip denominators [m/s]


class TireOut(NamedTuple):
    fx_w: object
    fy_w: object
    mz: object
    fx_b: object     # body-frame forces [N]
    fy_b: object
    alpha: object
    kappa: object
    gamma: object
    fz: object
    re: object
    mux: object
    muy: object
    kya: object


def stacked_tire_params(vp: VehicleParams):
    t = vp.tire
    return jax.tree_util.tree_map(
        lambda f, r: jnp.stack([jnp.asarray(f), jnp.asarray(f), jnp.asarray(r), jnp.asarray(r)]),
        t.front, t.rear)


def rolling_radius(vp: VehicleParams, fz):
    return jnp.maximum(vp.tire.r0 - fz / vp.tire.kz, 0.8 * vp.tire.r0)


def wheel_slips(vp: VehicleParams, vx, vy, r, steer, fz):
    """Return (alpha, vx_w, re) for each wheel. ``steer``: total road-wheel angle per wheel [rad]."""
    ch = vp.chassis
    xs, ys = ch.x_wheels(), ch.y_wheels()
    vxh = vx - r * ys
    vyh = vy + r * xs
    c, s = jnp.cos(steer), jnp.sin(steer)
    vxw = c * vxh + s * vyh
    vyw = -s * vxh + c * vyh
    vxs = jnp.sqrt(vxw * vxw + V_EPS * V_EPS)
    alpha = -jnp.arctan2(vyw, vxs)
    return alpha, vxw, rolling_radius(vp, fz)


def kappa_from_wheel_speed(w, re, vxw):
    return (w * re - vxw) / jnp.sqrt(vxw * vxw + V_EPS * V_EPS)


def tire_forces(vp: VehicleParams, alpha, kappa, gamma, fz, steer, re=None, p=None) -> TireOut:
    """Evaluate the four corners and rotate the forces into the body frame."""
    P = stacked_tire_params(vp)
    r = mf.mf61(P, alpha, kappa, gamma, fz, p)
    c, s = jnp.cos(steer), jnp.sin(steer)
    fx_b = c * r.fx - s * r.fy
    fy_b = s * r.fx + c * r.fy
    re = rolling_radius(vp, fz) if re is None else re
    return TireOut(r.fx, r.fy, r.mz, fx_b, fy_b, alpha, kappa, gamma, fz, re, r.mux, r.muy, r.kya)


def solve_kappa_for_fx(vp: VehicleParams, alpha, gamma, fz, fx_target, n_iter: int = 6):
    """Per-wheel slip ratio that yields ``fx_target`` (combined slip with the given alpha). Newton, damped."""
    P = stacked_tire_params(vp)

    def fx_of(k):
        return mf.mf61(P, alpha, k, gamma, fz).fx

    k = jnp.zeros_like(fz)
    ones = jnp.ones_like(fz)
    for _ in range(n_iter):
        f, df = jax.jvp(fx_of, (k,), (ones,))
        step = (f - fx_target) / jnp.maximum(df, 5.0 * fz + 50.0)
        k = jnp.clip(k - jnp.clip(step, -0.1, 0.1), -0.4, 0.4)
    return k


def body_loads(vp: VehicleParams, t: TireOut):
    """Sum tyre forces about the CG: (Fx, Fy, Mz) in the body frame."""
    ch = vp.chassis
    xs, ys = ch.x_wheels(), ch.y_wheels()
    fx = jnp.sum(t.fx_b)
    fy = jnp.sum(t.fy_b)
    mz = jnp.sum(xs * t.fy_b - ys * t.fx_b + t.mz)
    return fx, fy, mz