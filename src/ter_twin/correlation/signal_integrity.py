"""Layer 0 - Signal integrity & kinematic sanity gatekeeper.

Runs BEFORE any dynamics. Detects, reports and (only where configured) corrects: NaN/dropouts, frozen or
dead sensors, ZOH-hold frames, ISO 8855 polarity contradictions, standstill tare offsets and, via the
model-independent identity  ay ~= vx * r  (quasi-steady), DBC gain errors and relative IMU lag.
Nothing is corrected silently: every action lands in ``IntegrityReport.applied`` and ``findings``.
Corrections that cannot be attributed (gain/lag: ay or yaw may be the faulty one) are OFF by default.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .dsp import fill_nan, hampel, lowpass, ols, xcorr_lag
from .types import Finding, Severity

G = 9.80665
_MIN_ACTIVITY = {"vx": 0.5, "ay": 0.02, "ax": 0.01, "yaw_rate": 0.5, "steer_angle": 0.5}  # std on moving samples
_HOLD_CHECK = ("ay", "ax", "yaw_rate", "steer_angle")   # high-resolution channels: identical floats for >0.25 s are not physical
_TARE_TOL = {"ay": 0.01, "ax": 0.01, "yaw_rate": 0.15}   # g, g, deg/s
_LP_HZ = {"ay": 15.0, "ax": 15.0, "yaw_rate": 15.0, "steer_angle": 20.0}
_FACTORS = (2.0, 0.5, 4.0, 0.25, 8.0, 0.125, 10.0, 0.1, 1000.0, 1e-3, G, 1.0 / G, 3.6, 1.0 / 3.6,
            180.0 / math.pi, math.pi / 180.0)


@dataclass(frozen=True)
class IntegrityConfig:
    moving_speed: float = 4.2          # m/s
    still_speed: float = 1.0           # m/s
    nan_frac_error: float = 0.30
    hold_run_s: float = 0.25
    polarity_corr: float = 0.30
    max_lag_s: float = 0.5
    auto_flip: bool = True
    auto_tare: bool = True
    auto_gain: bool = False
    auto_lag: bool = False
    despike: bool = True
    lowpass: bool = True


@dataclass
class IntegrityReport:
    channels: dict[str, np.ndarray]
    findings: list[Finding] = field(default_factory=list)
    applied: dict[str, float] = field(default_factory=dict)
    dead: set[str] = field(default_factory=set)
    kin_gain: float = float("nan")      # slope of ay on vx*r/g
    kin_lag_s: float = float("nan")     # lag of ay w.r.t. vx*r

    def has(self, code: str, channel: str | None = None) -> bool:
        return any(f.code == code and (channel is None or f.channel == channel) for f in self.findings)

    def summary(self) -> str:
        if not self.findings:
            return "Layer 0: all channels nominal."
        return "\n".join(f"[{f.severity.value:5}] {f.code:<16} {f.channel:<12} {f.message}" for f in self.findings)


def _const_runs(y: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """[start, end) runs of exactly-identical consecutive samples with length >= min_len."""
    d = (np.diff(y) == 0).astype(np.int8)               # NaN compares False
    edges = np.flatnonzero(np.diff(np.concatenate(([0], d, [0]))))
    s, e = edges[::2], edges[1::2]
    keep = (e - s + 1) >= min_len
    return [(int(a), int(b) + 1) for a, b in zip(s[keep], e[keep])]


def _corr(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> float:
    m = m & np.isfinite(a) & np.isfinite(b)
    if m.sum() < 200 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def condition_channels(t: np.ndarray, ch: dict[str, np.ndarray], cfg: IntegrityConfig = IntegrityConfig()
                       ) -> IntegrityReport:
    t = np.asarray(t, float)
    n = t.size
    fs = 1.0 / float(np.median(np.diff(t)))
    out = {k: np.asarray(v, float).copy() for k, v in ch.items() if np.ndim(v) == 1 and len(v) == n}
    rep = IntegrityReport(out)

    def add(code: str, chn: str, sev: Severity, msg: str, **data: float) -> None:
        rep.findings.append(Finding(code, chn, sev, msg, {k: float(v) for k, v in data.items()}))

    vx = np.maximum(fill_nan(out["vx"]), 0.0) / 3.6 if "vx" in out else None
    moving = (vx > cfg.moving_speed) if vx is not None else np.ones(n, bool)

    # ---- 1. NaN / dropout ----------------------------------------------------------------------
    for name, y in out.items():
        frac = float(np.mean(~np.isfinite(y)))
        if frac > cfg.nan_frac_error:
            rep.dead.add(name)
            add("DROPOUT_NAN", name, Severity.ERROR, f"{100 * frac:.0f}% of samples missing; channel excluded.", frac=frac)
        elif frac > 0.02:
            add("DROPOUT_NAN", name, Severity.WARN, f"{100 * frac:.1f}% of samples missing (masked, not interpolated).", frac=frac)

    # ---- 2. frozen / dead sensor ---------------------------------------------------------------
    for name, floor in _MIN_ACTIVITY.items():
        if name not in out or name in rep.dead:
            continue
        sel = np.isfinite(out[name]) & (moving if name != "vx" else True)
        if sel.sum() > 100 and float(np.std(out[name][sel])) < floor:
            rep.dead.add(name)
            add("FROZEN_SENSOR", name, Severity.ERROR,
                f"std={np.std(out[name][sel]):.4g} < {floor} while the car moves: stuck/dead sensor. "
                "Dependent predictions are SKIPPED (no synthetic reconstruction, it would be circular).",
                std=np.std(out[name][sel]))

    # ---- 3. ZOH hold runs (CAN frame loss re-sampled by FrameAssembler) ------------------------
    for name in _HOLD_CHECK:
        if name not in out or name in rep.dead:
            continue
        held = 0
        for a, b in _const_runs(out[name], max(3, int(cfg.hold_run_s * fs))):
            if moving[a:b].mean() > 0.9:
                out[name][a:b] = np.nan
                held += b - a
        if held:
            add("DROPOUT_HOLD", name, Severity.WARN,
                f"{held} samples ({held / fs:.1f} s) were identical-valued holds while moving: masked.", samples=held)

    # ---- 4. ISO 8855 polarity ---------------------------------------------------------------------
    steer_ok = "steer_angle" in out and "steer_angle" not in rep.dead
    if vx is not None and all(k in out and k not in rep.dead for k in ("ay", "yaw_rate")):
        kin = vx * np.radians(out["yaw_rate"]) / G
        m = moving & (np.abs(kin) > 0.05)
        rho = _corr(out["ay"], kin, m)
        sr = _corr(out["steer_angle"], out["yaw_rate"], moving) if steer_ok else float("nan")
        sa = _corr(out["steer_angle"], out["ay"], moving) if steer_ok else float("nan")
        if np.isfinite(rho) and rho < -cfg.polarity_corr:
            # Contradiction between ay and yaw rate. The channel anti-correlated with steering is the inverted one.
            target = "yaw_rate" if (np.isfinite(sr) and np.isfinite(sa) and sr < sa) else "ay"
            msg = (f"corr(ay, vx*r)={rho:+.2f}: sign contradiction (ISO 8855: left turn => ay>0, r>0). "
                   f"Inverted channel attributed to '{target}'"
                   + ("" if steer_ok else " (steering unavailable: default assumption)") + ".")
            if cfg.auto_flip:
                out[target] = -out[target]
                rep.applied[f"sign_{target}"] = -1.0
                msg += " Sign flipped."
            add("POLARITY", target, Severity.ERROR, msg, corr=rho, steer_corr_r=sr, steer_corr_ay=sa)
        elif steer_ok and np.isfinite(sr) and np.isfinite(sa) and sr < -cfg.polarity_corr and sa < -cfg.polarity_corr:
            msg = f"steer vs (r, ay) correlations {sr:+.2f}/{sa:+.2f}: steering angle sign is inverted (positive must be LEFT)."
            if cfg.auto_flip:
                out["steer_angle"] = -out["steer_angle"]
                rep.applied["sign_steer_angle"] = -1.0
                msg += " Sign flipped."
            add("POLARITY", "steer_angle", Severity.ERROR, msg, corr_r=sr, corr_ay=sa)

    # ---- 5. standstill tare ------------------------------------------------------------------------
    if vx is not None:
        still = vx < cfg.still_speed
        for name, tol in _TARE_TOL.items():
            if name not in out or name in rep.dead:
                continue
            s = still & np.isfinite(out[name])
            if s.sum() < 100:
                continue
            off = float(np.median(out[name][s]))
            if abs(off) > tol:
                tilt = math.degrees(math.asin(max(-1.0, min(1.0, off)))) if name in ("ay", "ax") else float("nan")
                msg = f"standstill offset {off:+.4f}" + (f" (= {tilt:+.2f} deg mounting inclination)" if np.isfinite(tilt) else " (gyro bias)")
                if cfg.auto_tare:
                    out[name] = out[name] - off
                    rep.applied[f"tare_{name}"] = off
                    msg += "; tared."
                add("TARE_OFFSET", name, Severity.WARN, msg, offset=off, tilt_deg=tilt)

    # ---- 6. kinematic gain / relative-lag cross-check: ay ~= vx * r (model independent) --------------
    if vx is not None and all(k in out and k not in rep.dead for k in ("ay", "yaw_rate")):
        kin = vx * np.radians(out["yaw_rate"]) / G
        m = moving & (np.abs(kin) > 0.05) & np.isfinite(out["ay"])
        if m.sum() > 200:
            s, _, rho = ols(kin[m], out["ay"][m])
            lag, cc = xcorr_lag(np.where(moving, out["ay"], np.nan), np.where(moving, kin, np.nan), fs, cfg.max_lag_s)
            rep.kin_gain, rep.kin_lag_s = s, lag
            if np.isfinite(s) and rho >= 0.85 and abs(s - 1.0) > 0.12:
                f = min(_FACTORS, key=lambda c: abs(math.log(s / c)))
                known = abs(s / f - 1.0) < 0.07
                msg = (f"ay = {s:.3f} x (vx*r/g) at rho={rho:.2f}: scale mismatch between ay and yaw rate"
                       + (f", matches a x{f:g} factor" if known else "")
                       + ". Which channel owns the error is undecidable here (see L1 residual vs steer).")
                if cfg.auto_gain and known:
                    out["ay"] = out["ay"] / f
                    rep.applied["gain_ay"] = 1.0 / f
                    msg += f" ay divided by {f:g}."
                add("KIN_GAIN", "ay", Severity.ERROR, msg, slope=s, rho=rho)
            if np.isfinite(lag) and abs(lag) > max(2.0 / fs, 0.02) and np.isfinite(cc) and cc > 0.8:
                msg = f"ay lags vx*r by {1e3 * lag:+.0f} ms (IMU filter group delay or CAN transport; relative to the gyro)."
                if cfg.auto_lag:
                    k = int(round(lag * fs))
                    out["ay"] = np.concatenate([out["ay"][k:], np.full(k, np.nan)]) if k > 0 else out["ay"]
                    rep.applied["lag_ay_s"] = lag
                    msg += " Advanced."
                add("KIN_LAG", "ay", Severity.WARN, msg, lag_ms=1e3 * lag, corr=cc)

    # ---- 7. pre-conditioning ----------------------------------------------------------------------------
    for name, fc in _LP_HZ.items():
        if name not in out or name in rep.dead:
            continue
        if cfg.despike:
            out[name], k = hampel(out[name])
            if k:
                add("SPIKES", name, Severity.INFO, f"{k} impulsive outliers replaced (Hampel).", n=k)
        if cfg.lowpass:
            out[name] = lowpass(out[name], fs, fc)
    return rep