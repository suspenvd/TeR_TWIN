"""Understeer gradient K_us, handling balance and torque-vectoring yaw authority.

K_us convention: delta = L/R + K_us * ay  [rad per m/s^2]  (K_us > 0 understeer); also reported in deg/g.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from ter_twin.models.tires import pacejka_61 as mf
from ter_twin.models.suspension.load_transfer import steady_roll, wheel_loads
from ter_twin.models.powertrain.in_hub_4wd import torque_limits
from ter_twin.models.tires.interface import rolling_radius, stacked_tire_params
from ..parameters import G, VehicleParams
from .qss_solver import TrimResult, solve_trim


class UndersteerReport(NamedTuple):
    k_us_linear_deg_g: float     # from static axle cornering stiffnesses
    k_us_trim_deg_g: float       # slope of (delta - L ay/V^2) from two trim solves
    c_alpha_front: float         # axle cornering stiffness [N/rad]
    c_alpha_rear: float
    neutral_steer_point: float   # distance from front axle as fraction of L (static margin = 0.5-based)
    static_margin: float         # (x_NSP - x_cg)/L  (>0 stable)
    characteristic_speed: float  # [m/s] (nan if oversteer)


def axle_cornering_stiffness(vp: VehicleParams, V=15.0, ax=0.0, ay=0.0):
    """(C_f, C_r) [N/rad] from the tyre model at the loads of the given state (alpha = 0, gamma static)."""
    ld = wheel_loads(vp, V, ax, ay, steady_roll(vp, ay), 0.0)
    P = stacked_tire_params(vp)
    r = mf.mf61(P, jnp.zeros(4), jnp.zeros(4), jnp.zeros(4), ld.fz)
    kya = jnp.asarray(r.kya)
    return float(kya[0] + kya[1]), float(kya[2] + kya[3])


def understeer_gradient(vp: VehicleParams, V: float = 15.0, ay_lo: float = 2.0, ay_hi: float = 8.0) -> UndersteerReport:
    ch = vp.chassis
    cf, cr = axle_cornering_stiffness(vp, V)
    wf, wr = ch.mass * G * ch.b / ch.wheelbase, ch.mass * G * ch.a / ch.wheelbase
    k_lin = (wf / cf - wr / cr) / G                                   # rad / (m/s^2)
    t1 = solve_trim(vp, V, ay_lo)
    t2 = solve_trim(vp, V, ay_hi, guess=(float(t1.beta), float(t1.delta)))
    # slope of the Ackermann-corrected steer angle vs ay (includes load transfer, aero, camber, compliance)
    e1 = float(t1.delta) - ch.wheelbase * ay_lo / V ** 2
    e2 = float(t2.delta) - ch.wheelbase * ay_hi / V ** 2
    k_trim = (e2 - e1) / (ay_hi - ay_lo)
    to_deg_g = G * 180.0 / math.pi
    # neutral steer point: x_nsp/L from front axle = Cr / (Cf + Cr)... measured rear->front: x = L*Cr/(Cf+Cr)
    x_nsp = ch.wheelbase * cr / (cf + cr)
    x_cg = ch.a
    sm = (x_nsp - x_cg) / ch.wheelbase
    vch = math.sqrt(ch.wheelbase / k_lin) if k_lin > 0 else float("nan")
    return UndersteerReport(k_lin * to_deg_g, k_trim * to_deg_g, cf, cr, x_nsp / ch.wheelbase, sm, vch)


def balance_metrics(vp: VehicleParams, tr: TrimResult) -> dict:
    """Lateral-force utilisation per axle (|Fy|/(muy Fz)) and the front-minus-rear balance index (>0 understeer)."""
    fy, fz, muy = np.asarray(tr.fy_w), np.asarray(tr.fz), np.asarray(tr.muy)
    u = np.abs(fy) / np.maximum(muy * fz, 1e-6)
    uf = float(np.sum(np.abs(fy[:2])) / np.sum(muy[:2] * fz[:2]))
    ur = float(np.sum(np.abs(fy[2:])) / np.sum(muy[2:] * fz[2:]))
    return {"util_wheels": u, "util_front": uf, "util_rear": ur, "balance_index": uf - ur,
            "alpha_f_deg": float(np.degrees(np.mean(np.asarray(tr.alpha)[:2]))),
            "alpha_r_deg": float(np.degrees(np.mean(np.asarray(tr.alpha)[2:])))}


def tv_authority(vp: VehicleParams, tr: TrimResult, V: float) -> dict:
    """Maximum extra yaw moment [N m] (each sign) from torque vectoring at the trim state.

    Per axle a left/right Fx pair of equal magnitude (net force unchanged) is limited by the weaker of:
    the friction-ellipse residual  mux*Fz*sqrt(1-(Fy/(muy*Fz))^2), the drive envelope and the regen envelope.
    """
    ch = vp.chassis
    fz, fy = np.asarray(tr.fz), np.asarray(tr.fy_w)
    mux, muy = np.asarray(tr.mux), np.asarray(tr.muy)
    re = np.asarray(rolling_radius(vp, jnp.asarray(fz)))
    w = V / re
    td, trg = (np.asarray(x) for x in torque_limits(vp.pt, jnp.asarray(w)))
    cap_tyre = mux * fz * np.sqrt(np.clip(1.0 - (fy / (muy * fz + 1e-9)) ** 2, 0.0, 1.0))
    cur = np.asarray(tr.fx_w)
    up = np.minimum(td / re - cur, cap_tyre - cur)         # extra drive available
    dn = np.minimum(trg / re + cur, cap_tyre + cur)        # extra braking/regen available
    out = {}
    for name, (lo, hi), trk in (("front", (0, 1), ch.track_f), ("rear", (2, 3), ch.track_r)):
        # CCW (+Mz): left (y>0) brakes, right drives -> Mz = -y*Fx
        ccw = min(dn[lo], up[hi])
        cw = min(up[lo], dn[hi])
        out[name] = (float(max(ccw, 0.0) * trk), float(-max(cw, 0.0) * trk))
    return {"front_ccw_cw": out["front"], "rear_ccw_cw": out["rear"],
            "mz_max_ccw": out["front"][0] + out["rear"][0], "mz_max_cw": out["front"][1] + out["rear"][1]}