"""Residual-signature engine. For a failing channel it tests physically-motivated hypotheses on
e(t) = y_model - y_meas, and for each computes a COUNTERFACTUAL R2 (what R2 would be if that cause were
fixed). Results are ranked by confidence x R2-gain, so the first line is the most valuable fix.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from .dsp import ols, r2_score, shift_signal, xcorr_lag
from .types import Diagnosis

LOG = logging.getLogger("correlation.diagnostics")
G = 9.80665
_FACTORS = (2.0, 0.5, 4.0, 0.25, 8.0, 0.125, 10.0, 0.1, 1000.0, 1e-3, G, 1.0 / G, 3.6, 1.0 / 3.6,
            180.0 / math.pi, math.pi / 180.0)

__all__ = ["DiagContext", "diagnose_channel", "format_diagnoses"]


@dataclass(frozen=True)
class DiagContext:
    fs: float
    unit: str = ""
    odd: bool = False                      # odd under left/right mirror (ay, yaw rate)
    kind: str = "lateral"                  # "lateral" | "longitudinal" | "other"
    zero_tol: float = 0.0                  # absolute floor for standstill offset significance
    mass: float = 290.0
    mu_eff: float = 1.6
    rho_air: float = 1.18
    steer_road_rad: np.ndarray | None = None
    steer_ratio: float = 4.5
    peer_lags_s: tuple[float, ...] = ()
    trigger_r2: float = 0.80
    max_lag_s: float = 0.5


def _si(ctx: DiagContext) -> float:
    return G if ctx.unit == "g" else 1.0


def _h_gain(name, t, y, p, M, vx, ctx, base):
    s, b, rho = ols(y[M], p[M])
    if not (np.isfinite(rho) and rho >= 0.85 and abs(s - 1.0) > 0.07):
        return None
    ya, pa = y[M], p[M]
    q = np.abs(ya) < 0.4 * np.percentile(np.abs(ya), 99)
    s_lo = ols(ya[q], pa[q])[0] if q.sum() > 100 else float("nan")
    s_hi = ols(ya[~q], pa[~q])[0] if (~q).sum() > 100 else float("nan")
    consistent = np.isfinite(s_lo) and np.isfinite(s_hi) and abs(s_lo - s_hi) < 0.12 * abs(s)
    after = r2_score(y * s + b, p, M)
    f = 1.0 / s
    cand = min(_FACTORS, key=lambda c: abs(math.log(abs(f) / c))) if f > 0 else float("nan")
    known = np.isfinite(cand) and abs(f / cand - 1.0) < 0.08
    conf = float(np.clip(0.5 + (rho - 0.85) / 0.3, 0, 1) * (1.0 if consistent else 0.55))
    return Diagnosis(
        "GAIN_SCALE", name, f"Sensor gain/range mismatch: model/measured slope = {s:.3f} at rho={rho:.2f}",
        f"Measured '{name}' reads x{f:.3f} of the model" + (f" (matches a x{cand:g} factor)" if known else "")
        + f". Check the DBC factor for this signal (new factor = old x {s:.3f}) and the firmware ADC range. "
        + ("Slope is load-independent => sensor, not tyre model."
           if consistent else f"Slope differs low/high load ({s_lo:.2f}/{s_hi:.2f}) => model grip is the likelier cause."),
        conf, after - base, {"slope": s, "intercept": b, "rho": rho, "slope_low": s_lo, "slope_high": s_hi})


def _h_lag(name, t, y, p, M, vx, ctx, base):
    lag, cc = xcorr_lag(np.where(M, y, np.nan), np.where(M, p, np.nan), ctx.fs, ctx.max_lag_s)
    if abs(lag) < max(2.0 / ctx.fs, 0.01):
        return None
    ys = shift_signal(t, y, lag)
    after = r2_score(ys, p, M & np.isfinite(ys))
    if after - base < 0.005:
        return None
    shared = [x for x in ctx.peer_lags_s if abs(x - lag) < 0.01]
    transport = len(shared) >= 2
    fc = 1.0 / (2.0 * math.pi * abs(lag))
    return Diagnosis(
        "PHASE_LAG", name, f"Pure phase shift: measurement {'lags' if lag > 0 else 'leads'} the model by {1e3 * abs(lag):.0f} ms",
        ("Same delay on several channels => CAN transport / timestamp offset. " if transport else
         f"Equivalent first-order hardware filter fc~{fc:.1f} Hz (IMU digital LPF / anti-alias). ")
        + "Compensate by advancing the channel by this constant (do not retune the model).",
        float(np.clip(cc, 0, 1) * min(1.0, 0.3 + (after - base) / 0.1)), after - base,
        {"lag_ms": 1e3 * lag, "xcorr": cc, "fc_equiv_hz": fc, "shared_channels": float(len(shared))})


def _h_offset(name, t, y, p, M, vx, ctx, base):
    still = np.isfinite(y) & (vx < 1.0)
    if still.sum() < 50:
        return None
    pm = p[still & np.isfinite(p)]
    off = float(np.mean(y[still]) - (np.mean(pm) if pm.size else 0.0))
    sem = float(np.std(y[still]) / math.sqrt(max(still.sum() / (0.1 * ctx.fs), 1.0)))
    if abs(off) < max(ctx.zero_tol, 3.0 * sem):
        return None
    after = r2_score(y - off, p, M)
    tilt = math.degrees(math.asin(max(-1.0, min(1.0, off)))) if ctx.unit == "g" else float("nan")
    what = {"ay": "roll inclination of the IMU mount", "ax": "pitch inclination of the IMU mount",
            "yaw_rate": "gyro bias"}.get(name, "sensor tare")
    return Diagnosis(
        "STATIC_OFFSET", name, f"Static offset at standstill: {off:+.4g} {ctx.unit}",
        f"Offset measured with vx<1 m/s => uncalibrated tare or {what}"
        + (f" (= {tilt:+.2f} deg)" if np.isfinite(tilt) else "") + ". Re-zero on a levelled surface or subtract the standstill median.",
        float(np.clip(abs(off) / (3 * max(ctx.zero_tol, sem, 1e-12)), 0, 1)), after - base,
        {"offset": off, "tilt_deg": tilt, "sem": sem})


def _h_v2(name, t, y, p, M, vx, ctx, base):
    v = vx[M]
    if v.size < 200 or v.max() < 5.0:
        return None
    vm = float(v.max())
    e = (p - y)[M]
    bn, a, rho = ols((v / vm) ** 2, e)
    if not (np.isfinite(rho) and abs(rho) > 0.35):
        return None
    b = bn / vm ** 2
    after = r2_score(y, p - b * vx ** 2, M)
    if after - base < 0.03:
        return None
    b_si = b * _si(ctx)
    half_rho = 0.5 * ctx.rho_air
    ev = {"b_per_v2": b, "rho_e_v2": rho}
    if ctx.kind == "lateral":
        txt = ("model over-predicts grip growth with speed" if b > 0 else "model under-predicts grip growth with speed") \
              + ": aerodynamic downforce (Cl*A), its ride-height/ground-effect-stall map, or the aero balance"
        if ctx.unit == "g":
            d = b_si * ctx.mass / (ctx.mu_eff * half_rho)
            ev["delta_ClA_m2"] = d
            txt += f". Equivalent Cl*A {'too high' if d > 0 else 'too low'} by ~{abs(d):.2f} m2 (mu_eff={ctx.mu_eff})"
    elif ctx.kind == "longitudinal":
        d = b_si * ctx.mass / half_rho
        ev["delta_CdA_m2"] = d
        txt = (f"model {'accelerates more / brakes less' if b > 0 else 'accelerates less'} than measured with speed: "
               f"Cd*A {'too low' if d > 0 else 'too high'} by ~{abs(d):.2f} m2 (or rolling-resistance/drivetrain loss)")
    else:
        txt = "error grows with v^2: aerodynamic force or ride-height discrepancy"
    return Diagnosis("AERO_V2", name, f"Residual proportional to vx^2 (rho(e,v^2)={rho:+.2f})", txt + ".",
                     float(np.clip(abs(rho), 0, 1)), after - base, ev)


def _h_sat(name, t, y, p, M, vx, ctx, base):
    ya, pa = np.abs(y[M]), np.abs(p[M])
    p99y, p99p = float(np.percentile(ya, 99)), float(np.percentile(pa, 99))
    if p99p <= 0 or ya.size < 200:
        return None
    top = float(ya.max())
    nclip = int(np.sum(ya >= 0.9995 * top))
    if nclip >= 8 and nclip / ya.size > 0.002:
        return Diagnosis("SENSOR_CLIP", name, f"Measurement clipped at {top:.4g} {ctx.unit} ({nclip} identical peak samples)",
                         "Flat-top with exactly repeated values is a sensor/ADC range limit, not the tyre limit. "
                         "Raise the sensor range before tuning mu.", 0.8, 0.0, {"clip_value": top, "n": float(nclip)})
    lam = p99y / p99p
    if abs(lam - 1.0) < 0.07:
        return None
    w = np.clip((np.abs(p) / p99p - 0.6) / 0.4, 0.0, 1.0)
    after = r2_score(y, p * (1.0 + (lam - 1.0) * w), M)
    if after - base < 0.01:
        return None
    bulk = np.abs(y[M]) < 0.5 * p99y
    s_bulk = ols(y[M][bulk], p[M][bulk])[0] if bulk.sum() > 100 else float("nan")
    conf = 0.85 if (np.isfinite(s_bulk) and abs(s_bulk - 1.0) < 0.1) else 0.35
    scale = "lMuy" if ctx.kind == "lateral" else "lMux"
    return Diagnosis(
        "MU_PEAK", name, f"Peak acceleration mismatch: p99|meas|/p99|model| = {lam:.3f}",
        f"Model {'over' if lam < 1 else 'under'}-estimates peak grip while the linear region is consistent "
        f"(bulk slope {s_bulk:.2f}). Scale MF6.1 '{scale}' by ~{lam:.3f} (or re-check load sensitivity pDy2 / thermal window).",
        conf, after - base, {"lambda_mu": lam, "bulk_slope": s_bulk})


def _h_asym(name, t, y, p, M, vx, ctx, base):
    if not ctx.odd:
        return None
    sc = float(np.percentile(np.abs(y[M]), 99))
    thr = 0.2 * sc
    L, R = M & (y > thr), M & (y < -thr)
    if L.sum() < 100 or R.sum() < 100:
        return None
    sL, sR = ols(y[L], p[L])[0], ols(y[R], p[R])[0]
    eL, eR = float(np.mean((p - y)[L])), float(np.mean((p - y)[R]))
    g_asym = (sL - sR) / (0.5 * (sL + sR))
    common = 0.5 * (eL + eR) / sc
    if abs(g_asym) > 0.10:
        p2 = np.where(p >= 0, p / sL, p / sR)
        after = r2_score(y, p2, M)
        return Diagnosis(
            "ASYM_GAIN", name, f"Left/right asymmetry: slope L={sL:.2f} vs R={sR:.2f}",
            "Direction-dependent gain: asymmetric corner weights / cross-weight, left-right camber or toe misalignment, "
            "or tyre-model asymmetry (conicity/plysteer). Check scales and static alignment.",
            float(np.clip(abs(g_asym) / 0.3, 0, 1)), after - base, {"slope_L": sL, "slope_R": sR, "asym": g_asym})
    if abs(common) > 0.03:
        a, c, _ = ols(p[M], y[M])
        after = r2_score(y, a * p + c, M)
        ev = {"common_mode": common, "intercept": c}
        txt = "Same-sign residual in left AND right turns (common mode) is a zero offset, not a gain error."
        if ctx.steer_road_rad is not None:
            d = ctx.steer_road_rad
            k = float(np.sum((d * p)[M]) / max(np.sum((d * d)[M]), 1e-12))
            if abs(a * k) > 1e-9:
                d0 = -c / (a * k)
                sw = math.degrees(d0) * ctx.steer_ratio
                ev["steer_zero_sw_deg"] = sw
                txt += f" Estimated steering-wheel zero offset: {sw:+.2f} deg (rack/sensor zero)."
        return Diagnosis("ASYM_OFFSET", name, f"Common-mode left/right residual ({common:+.3f} of range)", txt,
                         float(np.clip(abs(common) / 0.1, 0, 1)), after - base, ev)
    return None


def _h_noise(name, t, y, p, M, vx, ctx, base):
    f = y[np.isfinite(y)]
    if f.size < 200:
        return None
    d2 = f[2:] - 2.0 * f[1:-1] + f[:-2]
    s2 = float(np.var(d2) / 6.0)                                # white-noise variance estimate
    ceil = 1.0 - s2 / float(np.var(y[M]))
    if ceil > 0.95:
        return None
    return Diagnosis("NOISE_CEILING", name, f"Broadband noise caps achievable R2 at ~{ceil:.3f}",
                     "Measurement noise (not the model) bounds the metric. Evaluate on a lower-bandwidth version "
                     "(zero-phase LPF) or increase sensor resolution.", 0.6, 0.0, {"r2_ceiling": ceil, "noise_var": s2})


def diagnose_channel(name: str, t: np.ndarray, meas: np.ndarray, pred: np.ndarray, mask: np.ndarray,
                     vx_ms: np.ndarray, ctx: DiagContext, always: bool = False) -> list[Diagnosis]:
    y, p = np.asarray(meas, float), np.asarray(pred, float)
    M = np.asarray(mask, bool) & np.isfinite(y) & np.isfinite(p)
    if M.sum() < 200:
        return []
    base = r2_score(y, p, M)
    if not always and base >= ctx.trigger_r2:
        return []
    out: list[Diagnosis] = []
    for fn in (_h_gain, _h_lag, _h_offset, _h_v2, _h_sat, _h_asym, _h_noise):
        try:
            d = fn(name, t, y, p, M, vx_ms, ctx, base)
        except Exception:  # noqa: BLE001 - one broken hypothesis must not hide the others
            LOG.exception("hypothesis %s failed on %s", fn.__name__, name)
            continue
        if d is not None:
            out.append(d)
    out.sort(key=lambda d: d.score, reverse=True)
    if base < ctx.trigger_r2 and not any(d.r2_gain >= 0.03 for d in out):
        out.append(Diagnosis("UNEXPLAINED", name, "No single-parameter signature explains the residual",
                             "Likely coupled transient dynamics (roll/pitch/aero attitude, tyre relaxation). "
                             "Escalate to Layer 2 and inspect the held-out posterior.", 0.3, 0.0, {"r2": base}))
    return out


def format_diagnoses(ds: list[Diagnosis]) -> str:
    lines = []
    for i, d in enumerate(ds, 1):
        lines.append(f"{i}. [{d.code}] {d.channel}: {d.title}\n   conf={d.confidence:.2f}  R2 gain if fixed={d.r2_gain:+.3f}\n   -> {d.action}")
    return "\n".join(lines)