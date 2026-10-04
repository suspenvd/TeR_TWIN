"""Ladder orchestrator: invariants -> L0 -> L1 -> (L2) -> diagnostics -> report."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .baseline import run_baseline
from .diagnostics import DiagContext, diagnose_channel, format_diagnoses
from .dsp import xcorr_lag
from .invariants import InvariantResult, run_invariants
from .observer import ObserverConfig, observer_series
from .signal_integrity import IntegrityConfig, IntegrityReport, condition_channels
from .types import SeriesResult

G = 9.80665
_ZERO_TOL = {"ay": 0.01, "ax": 0.01, "yaw_rate": 0.15}
_KIND = {"ay": "lateral", "yaw_rate": "lateral", "ax": "longitudinal"}


class InvariantViolation(RuntimeError):
    pass


@dataclass(frozen=True)
class LadderConfig:
    integrity: IntegrityConfig = IntegrityConfig()
    observer: ObserverConfig = ObserverConfig()
    run_layer2: bool = False
    run_invariants: bool = True
    strict_invariants: bool = False
    trigger_r2: float = 0.80


@dataclass
class CorrelationReport:
    fs: float
    integrity: IntegrityReport
    invariants: list[InvariantResult]
    series: list[SeriesResult]
    kus_deg_g: float
    ay_lim_med_g: float
    notes: list[str] = field(default_factory=list)
    l2_resets: dict[str, float] = field(default_factory=dict)

    @property
    def score(self) -> float:
        r = [s.metrics.r2 for s in self.series if s.metrics and math.isfinite(s.metrics.r2) and s.layer != "L2-OL"]
        return float(np.mean([max(0.0, min(1.0, x)) for x in r]) * 100.0) if r else float("nan")

    def explain(self) -> str:
        out = ["== INVARIANTS =="]
        out += [f"{'OK  ' if i.passed else 'FAIL'} {i.name}: err={i.error:.3g} (tol {i.tol:g}) {i.detail}" for i in self.invariants]
        out += ["== LAYER 0 ==", self.integrity.summary(), "== CORRELATION =="]
        for s in self.series:
            m = s.metrics
            out.append(f"{s.name:<18}[{s.mode}] " + (f"R2={m.r2:.3f} rho={m.rho:.3f} NMAE={m.nmae:.1f}% -> {m.grade}" if m else "insufficient data"))
            if s.diagnoses:
                out.append(format_diagnoses(s.diagnoses))
        out += [f"note: {n}" for n in self.notes]
        return "\n".join(out)


def run_ladder(t, ch, vp=None, front=None, rear=None, cfg: LadderConfig = LadderConfig()) -> CorrelationReport:
    from ter_twin.models import vehicle as V

    vp = vp if vp is not None else V.default_ter27(front, rear)
    t = np.asarray(t, float)
    fs = 1.0 / float(np.median(np.diff(t)))
    inv = run_invariants(vp) if cfg.run_invariants else []
    if cfg.strict_invariants and any(not i.passed for i in inv):
        raise InvariantViolation("; ".join(f"{i.name}={i.error:.3g}" for i in inv if not i.passed))

    l0 = condition_channels(t, ch, cfg.integrity)
    base = run_baseline(t, l0.channels, vp, front, rear, l0.dead)
    series = list(base.series)
    notes = list(base.notes)
    resets: dict[str, float] = {}
    if cfg.run_layer2:
        if "steer_angle" in l0.channels and "steer_angle" not in l0.dead and "vx" in l0.channels:
            s2, resets = observer_series(t, l0.channels, vp, cfg.observer)
            series += s2
        else:
            notes.append("Layer 2 skipped: steering/vx unavailable.")

    vx_ms = np.maximum(np.nan_to_num(l0.channels.get("vx", np.zeros(t.size))), 0.0) / 3.6
    lags = []
    for s in series:
        if s.layer == "L1":
            lg, _ = xcorr_lag(np.where(s.mask, s.meas, np.nan), np.where(s.mask, s.pred, np.nan), fs)
            s.lag_ms = 1e3 * lg
            lags.append(lg)
    for s in series:
        base_name = s.name.split("@")[0]
        ctx = DiagContext(
            fs=fs, unit=s.unit, odd=base_name in ("ay", "yaw_rate"), kind=_KIND.get(base_name, "other"),
            zero_tol=_ZERO_TOL.get(base_name, 0.0), mass=vp.chassis.mass, steer_road_rad=base.steer_road_rad,
            steer_ratio=vp.steer.steer_ratio, peer_lags_s=tuple(lags), trigger_r2=cfg.trigger_r2)
        if s.layer == "L2-OL" or base_name in ("vx",):
            continue
        s.diagnoses = diagnose_channel(base_name, t, s.meas, s.pred, s.mask, vx_ms, ctx)
    return CorrelationReport(fs, l0, inv, series, base.kus_deg_g, base.ay_lim_med_g, notes, resets)


def correlate_for_view(t, ch, vp, front, rear, run_layer2: bool = False) -> dict:
    """Drop-in for ``correlation_view.correlate``: same dict contract + ``diagnostics`` text."""
    rep = run_ladder(t, ch, vp, front, rear, LadderConfig(run_layer2=run_layer2, run_invariants=False))
    rows, series = [], {}
    for s in rep.series:
        m = s.metrics
        rows.append({"name": s.name, "unit": s.unit, **({"n": m.n, "rmse": m.rmse, "nmae": m.nmae, "r2": m.r2,
                     "rho": m.rho, "bias": m.bias, "emax": m.emax} if m else {}),
                     "lag_ms": s.lag_ms, "grade": m.grade if m else "N/A"})
        series[s.name] = {"unit": s.unit, "t": s.t, "meas": s.meas, "pred": s.pred, "mask": s.mask, "vx": s.vx_kmh}
    ay = rep.integrity.channels.get("ay")
    lags = [abs(r["lag_ms"]) for r in rows if np.isfinite(r.get("lag_ms", np.nan))]
    return {"rows": rows, "series": series, "kus_deg_g": rep.kus_deg_g,
            "ay_meas": float(np.nanpercentile(np.abs(ay), 99)) if ay is not None else float("nan"),
            "ay_model": rep.ay_lim_med_g, "score": rep.score, "lag_mean": float(np.mean(lags)) if lags else float("nan"),
            "fs": rep.fs, "diagnostics": rep.explain()}