"""Layer 1 - deterministic quasi-steady macro-dynamics. Algebraic in the inputs => unconditionally stable,
zero state drift, so any residual is a parameter/sensor error and never an integrator artefact.

yaw rate : steady bicycle r = vx*delta / (L + Kus*vx^2), Kus from the full dual-track trim solver (load
           transfer, camber, compliance, aero included), then
           (a) a first-order yaw lag tau = Izz*V/(Cf a^2 + Cr b^2)  (~20 ms: any measured lag beyond it is hardware),
           (b) smooth friction saturation  ay = a_lim(V) tanh(ay_lin / a_lim(V)), a_lim(V) from MF6.1 + aero
               downforce, so mu_peak and Cl*A errors become observable.
ax       : (sum T_wheel/Re - D_aero - Crr N) / (m + 4 Iw/Re^2), driven by measured motor torque.
           (Not d(wheel speed)/dt: that would not exercise the model.)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import lfilter, lfilter_zi

from .dsp import fill_nan
from .types import SeriesResult

G = 9.80665
CORNERS = ("fl", "fr", "rl", "rr")


@dataclass
class BaselineResult:
    series: list[SeriesResult] = field(default_factory=list)
    kus_deg_g: float = float("nan")
    ay_lim_med_g: float = float("nan")
    tau_model_s: float = 0.0
    steer_road_rad: np.ndarray | None = None
    notes: list[str] = field(default_factory=list)


def _lag(x: np.ndarray, tau: float, dt: float) -> np.ndarray:
    a = dt / max(tau, 1e-6)
    if a >= 1.0:
        return x
    b, den = [a], [1.0, -(1.0 - a)]
    return lfilter(b, den, x, zi=lfilter_zi(b, den) * x[0])[0]


def run_baseline(t: np.ndarray, ch: dict[str, np.ndarray], vp, front=None, rear=None,
                 dead: frozenset[str] | set[str] = frozenset()) -> BaselineResult:
    from ter_twin.models.vehicle.aerodynamics import aero_forces
    from ter_twin.models.vehicle.powertrain import electrical_power
    from ter_twin.models.vehicle.solvers import understeer_gradient
    from ter_twin.telemetry import friction_ellipses_g

    t = np.asarray(t, float)
    dt = float(np.median(np.diff(t)))
    c = vp.chassis
    res = BaselineResult()
    if "vx" not in ch or "vx" in dead:
        res.notes.append("vx unavailable: Layer 1 skipped.")
        return res
    vx = np.maximum(fill_nan(ch["vx"]), 0.0) / 3.6
    vxk = vx * 3.6
    moving = vx > 4.2

    def push(name, unit, meas, pred, mask, mode="algebraic (no state)"):
        s = SeriesResult(name, unit, "L1", mode, t, np.asarray(meas, float), np.asarray(pred, float), mask, vxk)
        s.compute()
        res.series.append(s)

    # ---- lateral ------------------------------------------------------------------------------------
    steer_ok = "steer_angle" in ch and "steer_angle" not in dead
    if steer_ok and any(k in ch and k not in dead for k in ("yaw_rate", "ay")):
        v_ref = float(np.median(vx[moving])) if moving.any() else 15.0
        try:
            rep = understeer_gradient(vp, v_ref)
            kus_deg_g = rep.k_us_trim_deg_g if math.isfinite(rep.k_us_trim_deg_g) else rep.k_us_linear_deg_g
            cf, cr = rep.c_alpha_front, rep.c_alpha_rear
        except Exception as exc:  # noqa: BLE001
            res.notes.append(f"understeer_gradient failed ({exc}); Kus=0.")
            kus_deg_g, cf, cr = 0.0, 8e4, 9e4
        res.kus_deg_g = float(kus_deg_g)
        kus = kus_deg_g * math.pi / 180.0 / G
        delta = np.radians(fill_nan(ch["steer_angle"])) / vp.steer.steer_ratio   # steering-WHEEL angle by channel spec
        res.steer_road_rad = delta
        vs = np.maximum(vx, 1.5)
        r_lin = vs * delta / np.maximum(c.wheelbase + kus * vs * vs, 0.3)
        tau = float(c.izz * v_ref / max(cf * c.a ** 2 + cr * c.b ** 2, 1.0))
        res.tau_model_s = tau
        r_lin = _lag(r_lin, tau, dt)
        ay_lin = vs * r_lin / G
        grid = np.linspace(5.0, max(float(vx.max()), 8.0), 6)
        gg = friction_ellipses_g(front, rear, vp, tuple(float(v) for v in grid))
        if gg:
            lim = np.interp(vs, [g[0] for g in gg], [g[1] for g in gg])
            ay_p = lim * np.tanh(ay_lin / np.maximum(lim, 1e-3))
            res.ay_lim_med_g = float(np.interp(v_ref, [g[0] for g in gg], [g[1] for g in gg]))
        else:
            ay_p = ay_lin
            res.notes.append("friction envelope unavailable: no saturation => mu_peak unobservable.")
        r_p = ay_p * G / vs
        if "yaw_rate" in ch and "yaw_rate" not in dead:
            push("yaw_rate", "deg/s", ch["yaw_rate"], np.degrees(r_p), moving)
        if "ay" in ch and "ay" not in dead:
            push("ay", "g", ch["ay"], ay_p, moving)
    elif not steer_ok:
        res.notes.append("steering dead/missing: yaw_rate & ay NOT predicted (reconstructing delta from r would be circular).")

    # ---- longitudinal ---------------------------------------------------------------------------------
    tq = [ch.get(f"motor_torque_{k}") for k in CORNERS]
    if "ax" in ch and "ax" not in dead and all(q is not None for q in tq):
        re = vp.tire.r0 - c.mass * G / 4.0 / vp.tire.kz
        T = np.vstack([fill_nan(q) for q in tq])                     # wheel torque [N m]
        A = aero_forces(vp.aero, vx, c.rh0_f, c.rh0_r, 0.0, 0.0, c.wheelbase, np)
        fz_tot = c.mass * G + np.asarray(A.df_f + A.df_r)
        f_net = T.sum(0) / re - np.asarray(A.drag) - vp.tire.crr * fz_tot * np.tanh(vx / 0.5)
        m_eff = c.mass + 4.0 * vp.tire.inertia_w / re ** 2
        ax_p = f_net / m_eff / G
        mask = vx > 3.0
        for k in ("brake_press_front", "brake_press_rear"):
            if k in ch:
                mask &= np.nan_to_num(ch[k], nan=0.0) < 5.0          # friction brakes are not modelled here
        push("ax", "g", ch["ax"], ax_p, mask)
        ws = [ch.get(f"wheel_speed_{k}") for k in CORNERS]
        if "battery_voltage" in ch and "battery_current" in ch and all(w is not None for w in ws) \
                and np.nanmedian(ch["battery_voltage"]) > 100.0:
            w_rad = np.vstack([fill_nan(w) for w in ws]) / 3.6 / re
            p_mod = np.sum(electrical_power(vp.pt, T, w_rad, np), axis=0) / 1e3
            p_meas = fill_nan(ch["battery_voltage"]) * fill_nan(ch["battery_current"]) / 1e3
            push("battery_power_kw", "kW", p_meas, p_mod, vx > 3.0)
    elif "ax" in ch and "ax" not in dead:
        res.notes.append("motor_torque_* missing: ax not predicted (no wheel-speed-derivative surrogate).")
    return res