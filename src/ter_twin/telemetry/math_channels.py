"""Derived (virtual) telemetry channels.

Works on whole logs (post-processing) or on a fixed-size trailing window (live). Pure NumPy/SciPy; the
vehicle/tyre models (``ter_twin.models.vehicle``, ``ter_twin.models.tires.pacejka_61``, JAX) are imported
lazily. If they are unavailable the module degrades to an analytic fallback for Fz and skips the
tyre-model channels (utilisation); it never raises for missing inputs - channels whose inputs are absent
are simply not produced.

Conventions (ISO 8855, same as the vehicle model): ay>0 left turn, yaw_rate>0 CCW, wheel order FL FR RL RR
(``y`` > 0 = left), damper travel > 0 = compression, steer_angle>0 = left (steering-WHEEL angle).

Channels produced (when inputs exist)
-------------------------------------
battery_power_kw   P = V*I/1000                      power_limit_delta  P - 80 kW (alert if > 0)
damper_velocity_*  central difference of low-passed potentiometer:  v(k) = (z(k+1)-z(k-1)) / (2 dt)
distance           trapezoid integral of vx          vy_est             leaky integral of  ay - r*vx
track_x/track_y    GPS->ENU, else planar dead-reckoning      roll_angle, g_total
fz_*               dynamic wheel loads (vehicle model with MEASURED roll from the dampers)
slip_angle_* [deg] -atan2(vy_w, sqrt(vx_w^2+V_EPS^2))       slip_ratio_* [%]  (w Re - vx_w)/max(|vx_w|,0.5)
tire_util_*        sqrt((Fx/(mux Fz))^2 + (Fy/(muy Fz))^2)  (== |F|/(mu Fz) when mux = muy)
tv_yaw_moment      sum(-y_i Fx_i)

Notes / assumptions
-------------------
* ``wheel_speed_*`` is the rolling speed of the tyre surface (omega*Re) in km/h.
* ``motor_torque_*`` is WHEEL torque [N m]; Fx = T/Re (wheel inertia and rolling resistance ignored). If the
  channel is absent Fx comes from the tyre model. Fy always comes from the tyre model (alpha, Fz, camber).
* vy is not measured: ``vy_est`` integrates (ay - r*vx) with a 2 s leak, so it is an estimate that is
  biased in long steady states. Dead-reckoned track_x/track_y drift without GPS.
"""
from __future__ import annotations

import logging
import math
from types import SimpleNamespace
from typing import Any

import numpy as np
from scipy.signal import butter, filtfilt, lfilter

from .channel_definitions import CORNERS, POWER_LIMIT_KW

LOG = logging.getLogger("telemetry.math")
G = 9.80665
SIDE = np.array([1.0, -1.0, 1.0, -1.0])
V_EPS = 0.5

__all__ = ["compute_math_channels", "damper_velocity", "lowpass", "central_diff", "cumtrapz", "gps_to_local",
           "planar_odometry", "friction_ellipses_g"]


# --------------------------------------------------------------------------------------------------------
# Signal utilities
# --------------------------------------------------------------------------------------------------------
def fill_nan(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    bad = ~np.isfinite(y)
    if not bad.any():
        return y
    if bad.all():
        return np.zeros_like(y)
    x = np.arange(y.size)
    return np.interp(x, x[~bad], y[~bad])


def lowpass(y: np.ndarray, fs: float, fc: float, order: int = 2) -> np.ndarray:
    y = fill_nan(y)
    if fc >= 0.5 * fs or y.size < 30:
        return y
    b, a = butter(order, fc / (0.5 * fs))
    return filtfilt(b, a, y)


def central_diff(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    d = np.zeros(y.size)
    if y.size < 3:
        return d
    d[1:-1] = (y[2:] - y[:-2]) / (t[2:] - t[:-2])
    d[0] = (y[1] - y[0]) / (t[1] - t[0])
    d[-1] = (y[-1] - y[-2]) / (t[-1] - t[-2])
    return d


def cumtrapz(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    y = fill_nan(y)
    return np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(t))])


def damper_velocity(t: np.ndarray, z_mm: np.ndarray, fc: float = 30.0) -> np.ndarray:
    fs = 1.0 / float(np.median(np.diff(t)))
    return central_diff(t, lowpass(z_mm, fs, fc))


def gps_to_local(lat: np.ndarray, lon: np.ndarray):
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    ok = np.isfinite(lat) & np.isfinite(lon) & ((lat != 0) | (lon != 0))
    if not ok.any():
        return None
    la0, lo0 = lat[ok][0], lon[ok][0]
    R = 6371000.0
    x = np.radians(lon - lo0) * R * math.cos(math.radians(la0))
    y = np.radians(lat - la0) * R
    return fill_nan(np.where(ok, x, np.nan)), fill_nan(np.where(ok, y, np.nan))


def planar_odometry(t: np.ndarray, vx: np.ndarray, vy: np.ndarray, r: np.ndarray):
    psi = cumtrapz(r, t)
    c, s = np.cos(psi), np.sin(psi)
    return cumtrapz(vx * c - vy * s, t), cumtrapz(vx * s + vy * c, t), psi


def _stack(ch: dict[str, np.ndarray], fmt: str):
    rows = [ch.get(fmt.format(c)) for c in CORNERS]
    return None if any(r is None for r in rows) else np.vstack([fill_nan(r) for r in rows])


# --------------------------------------------------------------------------------------------------------
# Vehicle / tyre model access (lazy, optional)
# --------------------------------------------------------------------------------------------------------
def _fallback_vehicle() -> SimpleNamespace:
    chs = SimpleNamespace(mass=290.0, wheelbase=1.55, a=0.8525, b=1.55 - 0.8525, track_f=1.20, track_r=1.16,
                          h_cg=0.285)
    return SimpleNamespace(
        chassis=chs, steer=SimpleNamespace(steer_ratio=4.5, ackermann=0.6), aero=None,
        susp=SimpleNamespace(camber_static_f=math.radians(-2.2), camber_static_r=math.radians(-1.6),
                             camber_bump_f=1.2, camber_bump_r=1.0),
        tire=SimpleNamespace(r0=0.2032, kz=90e3))


def _vehicle(vp, front, rear):
    try:
        from ter_twin.models import vehicle as V

        return (vp if vp is not None else V.default_ter27(front, rear)), V
    except Exception as exc:  # noqa: BLE001
        LOG.warning("vehicle model unavailable (%s); using analytic fallback", exc)
        return (vp if vp is not None else _fallback_vehicle()), None


def _mf():
    try:
        from ter_twin.models.tires import pacejka_61 as mf

        return mf
    except Exception:  # noqa: BLE001
        return None


def _fz_fallback(vp, ax, ay):
    c = vp.chassis
    m, L = c.mass, c.wheelbase
    dN = m * ax * c.h_cg / L
    nf, nr = m * G * c.b / L - dN, m * G * c.a / L + dN
    dlf, dlr = 0.5 * m * ay * c.h_cg / c.track_f, 0.5 * m * ay * c.h_cg / c.track_r
    return np.maximum(np.vstack([0.5 * nf - dlf, 0.5 * nf + dlf, 0.5 * nr - dlr, 0.5 * nr + dlr]), 1.0)


# --------------------------------------------------------------------------------------------------------
def compute_math_channels(t: np.ndarray, ch: dict[str, np.ndarray], vp: Any = None, front: Any = None,
                          rear: Any = None, vy_tau: float = 2.0) -> dict[str, np.ndarray]:
    """Return ONLY the new channels (raw channels are never overwritten). Safe on missing inputs."""
    t = np.asarray(t, float)
    out: dict[str, np.ndarray] = {}
    n = t.size
    if n < 8:
        return out
    dt = float(np.median(np.diff(t)))
    fs = 1.0 / dt

    # ---- power ------------------------------------------------------------------------------------
    p_kw = None
    if "battery_voltage" in ch and "battery_current" in ch:
        p_kw = fill_nan(ch["battery_voltage"]) * fill_nan(ch["battery_current"]) / 1000.0
        out["battery_power_kw"] = p_kw
    elif "battery_power" in ch:
        p_kw = fill_nan(ch["battery_power"])
        out["battery_power_kw"] = p_kw
    if p_kw is not None:
        out["power_limit_delta"] = p_kw - POWER_LIMIT_KW

    # ---- dampers ----------------------------------------------------------------------------------
    for c in CORNERS:
        z = ch.get(f"damper_travel_{c}")
        if z is not None and f"damper_velocity_{c}" not in ch:
            out[f"damper_velocity_{c}"] = damper_velocity(t, z)

    # ---- kinematics -------------------------------------------------------------------------------
    if "vx" not in ch:
        return out
    vx = np.maximum(fill_nan(ch["vx"]) / 3.6, 0.0)
    out["distance"] = cumtrapz(vx, t)
    have_dyn = all(k in ch for k in ("ax", "ay", "yaw_rate"))
    if "ax" in ch and "ay" in ch:
        out["g_total"] = np.hypot(fill_nan(ch["ax"]), fill_nan(ch["ay"]))
    if not have_dyn:
        return out

    ax_ms, ay_ms = fill_nan(ch["ax"]) * G, fill_nan(ch["ay"]) * G
    r = np.radians(fill_nan(ch["yaw_rate"]))
    vy = lfilter([0.0, dt], [1.0, -(1.0 - dt / vy_tau)], ay_ms - r * vx)
    out["vy_est"] = vy

    gps = gps_to_local(ch["gps_lat"], ch["gps_lon"]) if "gps_lat" in ch and "gps_lon" in ch else None
    out["track_x"], out["track_y"] = gps if gps is not None else planar_odometry(t, vx, vy, r)[:2]

    vp, V = _vehicle(vp, front, rear)
    c = vp.chassis
    xs = np.array([c.a, c.a, -c.b, -c.b])
    ys = np.array([0.5 * c.track_f, -0.5 * c.track_f, 0.5 * c.track_r, -0.5 * c.track_r])

    # ---- roll from dampers (measured) ----------------------------------------------------------------
    zs = _stack(ch, "damper_travel_{}")
    vs = _stack(out, "damper_velocity_{}")
    if vs is None:
        vs = _stack(ch, "damper_velocity_{}")
    phi = np.zeros(n)
    p_roll = np.zeros(n)
    if zs is not None:
        phi = 0.5 * ((zs[1] - zs[0]) / c.track_f + (zs[3] - zs[2]) / c.track_r) / 1000.0
    elif V is not None:
        try:
            phi = np.asarray(V.load_transfer.steady_roll(vp, ay_ms), dtype=float)
        except Exception:  # noqa: BLE001
            pass
    if vs is not None:
        p_roll = 0.5 * ((vs[1] - vs[0]) / c.track_f + (vs[3] - vs[2]) / c.track_r) / 1000.0
    out["roll_angle"] = np.degrees(phi)

    # ---- dynamic Fz ----------------------------------------------------------------------------------
    fz, dz = None, None
    if V is not None:
        try:
            ld = V.load_transfer.wheel_loads(vp, vx, ax_ms, ay_ms, phi, p_roll, 0.0, 3, np)
            fz, dz = np.asarray(ld.fz, float), np.asarray(ld.dz, float)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("wheel_loads failed (%s); analytic Fz fallback", exc)
    if fz is None:
        fz = _fz_fallback(vp, ax_ms, ay_ms)
        dz = (zs - np.median(zs, axis=1, keepdims=True)) / 1000.0 if zs is not None else np.zeros((4, n))
    for i, cn in enumerate(CORNERS):
        out[f"fz_{cn}"] = fz[i]

    # ---- slips ---------------------------------------------------------------------------------------
    delta = np.radians(fill_nan(ch["steer_angle"])) / vp.steer.steer_ratio if "steer_angle" in ch else np.zeros(n)
    k = vp.steer.ackermann * c.track_f / (2.0 * c.wheelbase)
    dd = k * delta * np.sqrt(delta * delta + 1e-8)
    steer = np.vstack([delta + dd, delta - dd, np.zeros(n), np.zeros(n)])
    vxh = vx[None, :] - r[None, :] * ys[:, None]
    vyh = vy[None, :] + r[None, :] * xs[:, None]
    cs, sn = np.cos(steer), np.sin(steer)
    vxw, vyw = cs * vxh + sn * vyh, -sn * vxh + cs * vyh
    alpha = -np.arctan2(vyw, np.sqrt(vxw ** 2 + V_EPS ** 2))
    for i, cn in enumerate(CORNERS):
        out[f"slip_angle_{cn}"] = np.degrees(alpha[i])
    ws = _stack(ch, "wheel_speed_{}")
    kappa = None
    if ws is not None:
        kappa = (ws / 3.6 - vxw) / np.maximum(np.abs(vxw), V_EPS)
        for i, cn in enumerate(CORNERS):
            out[f"slip_ratio_{cn}"] = 100.0 * kappa[i]

    # ---- tyre model: utilisation, TV moment ------------------------------------------------------------
    mf = _mf()
    if mf is None:
        return out
    try:
        s = vp.susp
        cam0 = np.array([s.camber_static_f] * 2 + [s.camber_static_r] * 2)[:, None]
        kb = np.array([s.camber_bump_f] * 2 + [s.camber_bump_r] * 2)[:, None]
        gamma = SIDE[:, None] * (cam0 - kb * dz) - phi[None, :]
        Pf = front if front is not None else mf.MF61Params()
        Pr = rear if rear is not None else Pf
        kap = kappa if kappa is not None else np.zeros((4, n))
        rf = mf.evaluate(Pf, alpha[:2], kap[:2], gamma[:2], fz[:2])
        rr = mf.evaluate(Pr, alpha[2:], kap[2:], gamma[2:], fz[2:])
        cat = lambda a, b: np.vstack([np.asarray(a), np.asarray(b)])  # noqa: E731
        fx_m, fy_m = cat(rf.fx, rr.fx), cat(rf.fy, rr.fy)
        mux, muy = cat(rf.mux, rr.mux), cat(rf.muy, rr.muy)
        tq = _stack(ch, "motor_torque_{}")
        re = np.maximum(vp.tire.r0 - fz / vp.tire.kz, 0.8 * vp.tire.r0)
        fx = tq / re if tq is not None else fx_m
        fzs = np.maximum(fz, 1.0)
        util = np.sqrt((fx / (mux * fzs)) ** 2 + (fy_m / (muy * fzs)) ** 2)
        for i, cn in enumerate(CORNERS):
            out[f"tire_util_{cn}"] = util[i]
        out["tv_yaw_moment"] = np.sum(-ys[:, None] * fx, axis=0)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("tyre-model channels skipped: %s", exc)
    return out


def friction_ellipses_g(front: Any = None, rear: Any = None, vp: Any = None,
                        speeds_ms: tuple[float, ...] = (10.0, 25.0)) -> list[tuple[float, float, float]]:
    """[(v, ay_max_g, ax_max_g)] from MF6.1 peak mu at the corner load incl. aero downforce at speed v.

    Simplification: a single load-sensitive mu per axle-average corner load (no load transfer, no combined
    slip); good enough as a G-G reference boundary. Returns [] if the models are unavailable.
    """
    mf = _mf()
    if mf is None:
        return []
    vp, V = _vehicle(vp, front, rear)
    c = vp.chassis
    Pf = front if front is not None else mf.MF61Params()
    Pr = rear if rear is not None else Pf
    res = []
    for v in speeds_ms:
        df = 0.0
        if V is not None and getattr(vp, "aero", None) is not None:
            try:
                a = V.aerodynamics.aero_forces(vp.aero, v, c.rh0_f, c.rh0_r, 0.0, 0.0, c.wheelbase, np)
                df = float(a.df_f + a.df_r)
            except Exception:  # noqa: BLE001
                pass
        ftot = c.mass * G + df
        fz = [ftot / 4.0]
        sf, sr = mf.summary(Pf, fz), mf.summary(Pr, fz)
        muy = 0.5 * (float(sf["mu_y_peak"][0]) + float(sr["mu_y_peak"][0]))
        mux = 0.5 * (float(sf["mu_x_peak"][0]) + float(sr["mu_x_peak"][0]))
        res.append((float(v), muy * ftot / (c.mass * G), mux * ftot / (c.mass * G)))
    return res