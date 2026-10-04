"""Aero map solver: ride-height (and thereby pitch) / roll dependent downforce, drag and CoP.

Pitch enters through the two ride heights measured at the axle reference points
(theta ~ (rh_r - rh_f)/L - rake); roll through a symmetric loss factor (1 - roll_sens*phi^2).
Works with ``xp=jnp`` (jit/vmap/grad) or ``xp=numpy`` (GUI fallback, no tracing overhead).
"""
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from .parameters import AeroParams


class AeroOut(NamedTuple):
    df_f: object      # front downforce (positive = pushes car down) [N]
    df_r: object
    drag: object      # drag magnitude [N] (>0)
    cop_x: object     # centre of pressure measured from the FRONT axle towards the rear [m]
    balance_f: object  # front fraction of downforce [-]
    cla: object       # total ClA [m^2]
    cda: object
    q: object         # dynamic pressure [Pa]


def _bilinear(xg, yg, tab, x, y, xp):
    xg, yg, tab = xp.asarray(xg), xp.asarray(yg), xp.asarray(tab)
    x = xp.clip(x, xg[0], xg[-1])
    y = xp.clip(y, yg[0], yg[-1])
    i = xp.clip(xp.searchsorted(xg, x) - 1, 0, xg.shape[0] - 2)
    j = xp.clip(xp.searchsorted(yg, y) - 1, 0, yg.shape[0] - 2)
    tx = (x - xg[i]) / (xg[i + 1] - xg[i])
    ty = (y - yg[j]) / (yg[j + 1] - yg[j])
    return ((1 - tx) * (1 - ty) * tab[i, j] + tx * (1 - ty) * tab[i + 1, j]
            + (1 - tx) * ty * tab[i, j + 1] + tx * ty * tab[i + 1, j + 1])


def aero_forces(ap: AeroParams, v, rh_f, rh_r, phi=0.0, drs=0.0, wheelbase=1.55, xp=jnp) -> AeroOut:
    """Aero loads at airspeed ``v`` [m/s]; ``drs`` in [0, 1] (0 closed, 1 fully open)."""
    q = 0.5 * ap.rho * v * v
    roll = 1.0 - ap.roll_sens * phi * phi
    cf = _bilinear(ap.rh_f_grid, ap.rh_r_grid, ap.cla_f, rh_f, rh_r, xp) * roll + drs * ap.drs_dcla_f
    cr = _bilinear(ap.rh_f_grid, ap.rh_r_grid, ap.cla_r, rh_f, rh_r, xp) * roll + drs * ap.drs_dcla_r
    cd = _bilinear(ap.rh_f_grid, ap.rh_r_grid, ap.cda, rh_f, rh_r, xp) + drs * ap.drs_dcda
    df_f, df_r = q * cf, q * cr
    tot = cf + cr
    bal = cf / xp.where(xp.abs(tot) < 1e-9, 1e-9, tot)
    return AeroOut(df_f, df_r, q * cd, (1.0 - bal) * wheelbase, bal, tot, cd, q)