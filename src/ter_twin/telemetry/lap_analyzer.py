"""Lap detection (beacon or start/finish gate), sector splits, lap statistics and delta-t vs a reference lap.

Distance is the trapezoid integral of vx; sector boundaries are fractions of the lap distance (default
1/3, 2/3). Delta-t:  dt(s) = t_cur(s) - t_ref(s), with the current lap distance rescaled to the reference
lap length (integration error of vx would otherwise accumulate into the delta).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Lap", "LapAnalyzer", "format_lap_time", "default_gate"]


def format_lap_time(sec: float) -> str:
    if not np.isfinite(sec):
        return "-"
    m = int(sec // 60)
    return f"{m}:{sec - 60 * m:06.3f}"


@dataclass(frozen=True)
class Lap:
    index: int
    t_start: float
    t_end: float
    i0: int
    i1: int                       # exclusive
    lap_time: float
    sectors: tuple[float, float, float]
    v_max_kmh: float
    distance_m: float
    valid: bool


def default_gate(x: np.ndarray, y: np.ndarray, v_ms: np.ndarray, half_width: float = 15.0):
    """Start/finish gate through the first moving point, perpendicular to the initial heading."""
    mov = np.flatnonzero(v_ms > 3.0)
    k = int(mov[0]) if mov.size else 0
    k2 = min(k + 50, x.size - 1)
    d = np.array([x[k2] - x[k], y[k2] - y[k]])
    nd = np.linalg.norm(d)
    d = d / nd if nd > 1e-6 else np.array([1.0, 0.0])
    nrm = np.array([-d[1], d[0]])
    p0 = np.array([x[k], y[k]])
    return p0 + half_width * nrm, p0 - half_width * nrm, k


def _gate_crossings(t, x, y, a, b) -> np.ndarray:
    px, py = x[:-1], y[:-1]
    rx, ry = x[1:] - px, y[1:] - py
    sx, sy = b[0] - a[0], b[1] - a[1]
    den = rx * sy - ry * sx
    qx, qy = a[0] - px, a[1] - py
    with np.errstate(divide="ignore", invalid="ignore"):
        tt = (qx * sy - qy * sx) / den
        uu = (qx * ry - qy * rx) / den
    hit = (np.abs(den) > 1e-12) & (tt >= 0) & (tt < 1) & (uu >= 0) & (uu <= 1)
    if not hit.any():
        return np.zeros(0)
    k = np.flatnonzero(hit)
    sign = np.sign(den[k])
    k = k[sign == np.sign(den[k[0]])]            # one crossing direction only
    return t[k] + tt[k] * (t[k + 1] - t[k])


class LapAnalyzer:
    def __init__(self, t: np.ndarray, vx_kmh: np.ndarray, sector_fracs: tuple[float, float] = (1 / 3, 2 / 3)) -> None:
        self.t = np.asarray(t, float)
        self.vx_kmh = np.nan_to_num(np.asarray(vx_kmh, float))
        v = self.vx_kmh / 3.6
        d = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(self.t))])
        self.dist = np.maximum.accumulate(d)
        self.sector_fracs = sector_fracs
        self.laps: list[Lap] = []

    # ---------------------------------------------------------------- detection
    def detect_from_beacon(self, beacon: np.ndarray, threshold: float = 0.5, min_lap_time: float = 10.0) -> list[Lap]:
        b = np.nan_to_num(np.asarray(beacon, float)) > threshold
        rising = np.flatnonzero(b[1:] & ~b[:-1]) + 1
        return self._build(self.t[rising], min_lap_time)

    def detect_from_gate(self, x: np.ndarray, y: np.ndarray, gate=None, min_lap_time: float = 10.0) -> list[Lap]:
        x, y = np.asarray(x, float), np.asarray(y, float)
        if gate is None:
            a, b, k = default_gate(x, y, self.vx_kmh / 3.6)
        else:
            a, b = np.asarray(gate[0], float), np.asarray(gate[1], float)
            k = 0
        times = np.concatenate([[self.t[k]], _gate_crossings(self.t, x, y, a, b)])
        return self._build(times[np.argsort(times)], min_lap_time)

    def _build(self, cross_times: np.ndarray, min_lap_time: float) -> list[Lap]:
        kept: list[float] = []
        for tc in cross_times:
            if not kept or tc - kept[-1] >= min_lap_time:
                kept.append(float(tc))
        raw = []
        for k in range(len(kept) - 1):
            t0, t1 = kept[k], kept[k + 1]
            i0, i1 = int(np.searchsorted(self.t, t0)), int(np.searchsorted(self.t, t1))
            if i1 - i0 < 3:
                continue
            d0, d1 = np.interp([t0, t1], self.t, self.dist)
            sb = d0 + np.asarray(self.sector_fracs) * (d1 - d0)
            ts = np.interp(sb, self.dist, self.t)           # dist is non-decreasing
            marks = [t0, float(ts[0]), float(ts[1]), t1]
            sec = tuple(float(marks[j + 1] - marks[j]) for j in range(3))
            raw.append((t0, t1, i0, i1, sec, float(np.max(self.vx_kmh[i0:i1])), float(d1 - d0)))
        med = float(np.median([r[1] - r[0] for r in raw])) if raw else 0.0
        self.laps = [Lap(i + 1, r[0], r[1], r[2], r[3], r[1] - r[0], r[4], r[5], r[6],
                         (r[1] - r[0]) <= 1.5 * med) for i, r in enumerate(raw)]
        return self.laps

    # ---------------------------------------------------------------- queries
    def best_lap(self) -> Lap | None:
        v = [l for l in self.laps if l.valid]
        return min(v, key=lambda l: l.lap_time) if v else None

    def get(self, index: int) -> Lap | None:
        return next((l for l in self.laps if l.index == index), None)

    def lap_distance(self, lap: Lap) -> np.ndarray:
        return self.dist[lap.i0:lap.i1] - self.dist[lap.i0]

    def delta_time(self, lap: Lap, ref: Lap) -> tuple[np.ndarray, np.ndarray]:
        """(distance_into_lap [m], delta_t [s]) sampled at the current lap's native samples."""
        d_cur = self.lap_distance(lap)
        d_ref = self.lap_distance(ref)
        t_cur = self.t[lap.i0:lap.i1] - lap.t_start
        t_ref = self.t[ref.i0:ref.i1] - ref.t_start
        scale = d_ref[-1] / d_cur[-1] if d_cur[-1] > 1e-6 else 1.0
        return d_cur, t_cur - np.interp(d_cur * scale, d_ref, t_ref)