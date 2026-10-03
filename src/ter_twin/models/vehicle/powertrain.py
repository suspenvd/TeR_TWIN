"""4WD in-hub PMSM powertrain: torque/speed envelopes, regen limits, 80 kW cap, TV allocation.

All torques are WHEEL torques [N m] (positive = drive). ``w`` = wheel angular speed [rad/s].
This is the plant-side allocator (analytic weighted least squares + one saturation pass); the full
online QP allocator lives in ``control/torque_vectoring/qp_allocator.py``.
"""
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from .parameters import PowertrainParams, VehicleParams


class TVOut(NamedTuple):
    t_wheel: object    # (4,) final wheel torques after envelope + power cap [N m]
    fx_cmd: object     # (4,) corresponding wheel-frame longitudinal force demand [N]
    p_elec: object     # net accumulator power [W] (>0 discharge)
    cap_scale: object  # drive-torque scale applied by the 80 kW cap [-]
    fx_total: object   # achieved sum(Fx) [N]
    mz_total: object   # achieved yaw moment from Fx asymmetry [N m]


def torque_limits(pt: PowertrainParams, w, xp=jnp):
    """(t_drive_max, t_regen_max) at the wheel, both >= 0. Constant torque -> constant power -> speed taper."""
    wm = xp.abs(w) * pt.gear_ratio + 1e-3
    taper = xp.clip((pt.motor_w_max - wm) / (0.08 * pt.motor_w_max), 0.0, 1.0)
    td = xp.minimum(pt.motor_tq_peak, pt.motor_p_peak / wm) * taper
    ramp = xp.clip((wm - pt.regen_w_min) / (0.25 * pt.regen_w_min), 0.0, 1.0)
    tr = xp.minimum(pt.regen_tq_peak, pt.regen_p_peak / wm) * ramp * taper
    k = pt.gear_ratio
    return td * k * pt.eta_drive, tr * k * pt.eta_regen


def electrical_power(pt: PowertrainParams, t_wheel, w, xp=jnp):
    """Accumulator power per wheel [W]: drive = P_mech/eta, regen = P_mech*eta (negative)."""
    pm = t_wheel * w
    return xp.where(pm >= 0.0, pm / pt.eta_drive, pm * pt.eta_regen)


def apply_power_cap(pt: PowertrainParams, t_wheel, w, xp=jnp):
    """Scale drive torques so the summed discharge power <= p_acc_max; limit recuperation to p_regen_max."""
    pe = electrical_power(pt, t_wheel, w, xp)
    p_pos = xp.sum(xp.maximum(pe, 0.0))
    p_neg = xp.sum(xp.minimum(pe, 0.0))
    s_drv = xp.minimum(1.0, pt.p_acc_max / (p_pos + 1e-6))
    s_reg = xp.minimum(1.0, pt.p_regen_max / (-p_neg + 1e-6))
    t = xp.where(t_wheel > 0.0, t_wheel * s_drv, t_wheel * s_reg)
    pe = electrical_power(pt, t, w, xp)
    return t, s_drv, xp.sum(pe)


def _ls_alloc(fx_req, mz_req, y, weight, xp):
    """min sum(F_i^2/weight_i) s.t. sum F = fx_req, sum(-y_i F_i) = mz_req  (closed form)."""
    b1, b2 = xp.ones(4), -y
    m11 = xp.sum(b1 * b1 * weight)
    m12 = xp.sum(b1 * b2 * weight)
    m22 = xp.sum(b2 * b2 * weight)
    det = m11 * m22 - m12 * m12 + 1e-9
    l1 = (m22 * fx_req - m12 * mz_req) / det
    l2 = (-m12 * fx_req + m11 * mz_req) / det
    return weight * (b1 * l1 + b2 * l2)


def tv_allocate(vp: VehicleParams, fx_req, mz_req, fz, w, re, xp=jnp) -> TVOut:
    """Distribute a total longitudinal force and a yaw-moment request over the four hub motors.

    Objective: equal tyre utilisation (min sum (Fx_i/Fz_i)^2) -> weights Fz_i^2. Saturated wheels
    (motor envelope) are frozen and the residual is re-allocated once over the remaining wheels,
    then the accumulator cap is applied. ``re``: (4,) rolling radii.
    """
    pt, ch = vp.pt, vp.chassis
    y = ch.y_wheels() if xp is jnp else xp.asarray(ch.y_wheels())
    td, tr = torque_limits(pt, w, xp)
    t0 = _ls_alloc(fx_req, mz_req, y, fz * fz, xp) * re
    t1 = xp.clip(t0, -tr, td)
    f1 = t1 / re
    sat = (xp.abs(t1 - t0) > 1e-6).astype(t1.dtype)
    res_fx = fx_req - xp.sum(f1)
    res_mz = mz_req - xp.sum(-y * f1)
    wt = fz * fz * (1.0 - sat) + 1e-6 * fz * fz
    t2 = xp.clip(t1 + _ls_alloc(res_fx, res_mz, y, wt, xp) * re, -tr, td)
    t3, s, pel = apply_power_cap(pt, t2, w, xp)
    f3 = t3 / re
    return TVOut(t3, f3, pel, s, xp.sum(f3), xp.sum(-y * f3))