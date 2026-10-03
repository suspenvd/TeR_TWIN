"""Offline log ingestion (.mf4 / .csv / .npz / .mat) into a unified :class:`LogData`.

All formats: names go through ``resolve_channel_name`` (=> tyre-temp harness remap), non-numeric / non-1-D
columns are dropped, time is shifted to start at 0 and, when needed, resampled onto a uniform grid.
``.mf4`` is always resampled to 200 Hz (linear; zero-order hold for ``lap_beacon``). Other formats keep
their native rate when it is already uniform.

Assumption: channel values are already in the units of ``channel_definitions`` (km/h, g, deg, bar, kW ...).
"""
from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.signal import filtfilt, butter, lfilter

from .channel_definitions import CORNERS, TIRE_TEMP_LOGGED_NAME, normalize_name, resolve_channel_name

LOG = logging.getLogger("telemetry.loader")
FS_DEFAULT = 200.0
G = 9.80665
_TIME_KEYS = {"time", "timestamp", "timestamps", "et", "t", "time_s", "timestamp_s", "timestamp_ms", "time_ms"}
_ZOH = {"lap_beacon"}


@dataclass
class LogData:
    t: np.ndarray
    channels: dict[str, np.ndarray]
    fs: float
    path: Path | None = None
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.t.size)

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0]) if self.n > 1 else 0.0

    @property
    def names(self) -> list[str]:
        return list(self.channels)

    def to_dataframe(self):  # optional convenience, pandas not required elsewhere
        import pandas as pd

        return pd.DataFrame({"t": self.t, **self.channels})


def _zoh(grid: np.ndarray, ts: np.ndarray, ys: np.ndarray) -> np.ndarray:
    i = np.clip(np.searchsorted(ts, grid, side="right") - 1, 0, ts.size - 1)
    return ys[i]


def _interp(grid: np.ndarray, ts: np.ndarray, ys: np.ndarray, name: str) -> np.ndarray:
    if name in _ZOH:
        return _zoh(grid, ts, ys)
    ok = np.isfinite(ys) & np.isfinite(ts)
    if ok.sum() < 2:
        return np.full(grid.size, np.nan)
    return np.interp(grid, ts[ok], ys[ok], left=np.nan, right=np.nan)


def _build(t: np.ndarray, raw: Mapping[str, np.ndarray], fs: float | None, path: Path | None,
           meta: dict, force_fs: bool = False) -> LogData:
    t = np.asarray(t, dtype=float)
    ok = np.isfinite(t)
    order = np.argsort(t[ok], kind="stable")
    t = t[ok][order]
    t, uniq = np.unique(t, return_index=True)
    if t.size < 3:
        raise ValueError("log has fewer than 3 valid time samples")
    chans: dict[str, np.ndarray] = {}
    for k, v in raw.items():
        v = np.asarray(v)
        if v.ndim != 1 or v.size != ok.size or v.dtype.kind not in "fiub":
            continue
        name = resolve_channel_name(k)
        if name in _TIME_KEYS:
            continue
        if name in chans:
            LOG.warning("duplicate channel after name resolution: %s (keeping first)", name)
            continue
        chans[name] = v[ok][order][uniq].astype(float)
    t = t - t[0]
    dt = np.diff(t)
    med = float(np.median(dt))
    uniform = np.max(np.abs(dt - med)) < 0.25 * med
    if force_fs or not uniform:
        out_fs = float(fs or FS_DEFAULT)
        grid = np.arange(0.0, t[-1], 1.0 / out_fs)
        chans = {k: _interp(grid, t, v, k) for k, v in chans.items()}
        t = grid
    else:
        out_fs = 1.0 / med
    return LogData(t, chans, out_fs, path, meta)


# --------------------------------------------------------------------------------------------------------
def load_mf4(path: Path, fs: float = FS_DEFAULT) -> LogData:
    from asammdf import MDF

    sigs = []
    with MDF(str(path)) as mdf:
        for sig in mdf.iter_channels():
            s = np.asarray(sig.samples)
            ts = np.asarray(sig.timestamps, dtype=float)
            if s.ndim != 1 or s.dtype.kind not in "fiub" or ts.size < 2:
                continue
            o = np.argsort(ts, kind="stable")
            sigs.append((sig.name, ts[o], s[o].astype(float)))
    if not sigs:
        raise ValueError(f"{path.name}: no numeric channels")
    t_lo = min(s[1][0] for s in sigs)
    t_hi = max(s[1][-1] for s in sigs)
    grid = np.arange(t_lo, t_hi, 1.0 / fs)
    raw = {name: _interp(grid, ts, ys, resolve_channel_name(name)) for name, ts, ys in sigs}
    return _build(grid, raw, fs, path, {"format": "mf4"}, force_fs=True)


def _is_num(s: str) -> bool:
    try:
        float(s.replace(",", "."))
        return True
    except ValueError:
        return False


def load_csv(path: Path, fs: float = FS_DEFAULT) -> LogData:
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as fh:
        sample = fh.read(8192)
        fh.seek(0)
        try:
            delim = csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
        except csv.Error:
            delim = ","
        rows = list(csv.reader(fh, delimiter=delim))
    hdr_i = next((i for i, r in enumerate(rows[:200])
                  if len(r) >= 3 and sum(not _is_num(c) for c in r if c.strip()) >= 0.7 * len([c for c in r if c.strip()])), None)
    if hdr_i is None:
        raise ValueError(f"{path.name}: header row not found")
    names = [c.strip() for c in rows[hdr_i]]
    data = []
    for r in rows[hdr_i + 1:]:
        if len(r) < len(names):
            continue
        try:
            data.append([float(c.replace(",", ".")) if c.strip() else np.nan for c in r[:len(names)]])
        except ValueError:
            continue  # units row / junk
    if not data:
        raise ValueError(f"{path.name}: no numeric rows")
    arr = np.asarray(data)
    raw = {n: arr[:, i] for i, n in enumerate(names) if n}
    tkey = next((n for n in raw if normalize_name(n) in _TIME_KEYS), None)
    if tkey is None:
        LOG.warning("%s: no time column, assuming %.0f Hz", path.name, fs)
        t = np.arange(arr.shape[0]) / fs
    else:
        t = raw[tkey] * (1e-3 if normalize_name(tkey).endswith("ms") else 1.0)
    return _build(t, raw, fs, path, {"format": "csv", "time_column": tkey})


def load_npz(path: Path, fs: float = FS_DEFAULT) -> LogData:
    z = np.load(path, allow_pickle=False)
    raw = {k: z[k] for k in z.files}
    tkey = next((k for k in raw if normalize_name(k) in _TIME_KEYS), None)
    f = float(raw["fs"]) if "fs" in raw and np.ndim(raw["fs"]) == 0 else fs
    if tkey is None:
        n = max(v.size for v in raw.values() if v.ndim == 1)
        t = np.arange(n) / f
    else:
        t = raw[tkey].astype(float) * (1e-3 if normalize_name(tkey).endswith("ms") else 1.0)
    raw = {k: v for k, v in raw.items() if v.ndim == 1 and v.size == t.size}
    return _build(t, raw, f, path, {"format": "npz"})


def _flatten_mat(obj, prefix: str = "", depth: int = 0) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if isinstance(obj, np.ndarray) and obj.dtype.kind in "fiub" and obj.ndim <= 1 and obj.size > 1:
        out[prefix] = np.atleast_1d(obj).astype(float)
    elif hasattr(obj, "_fieldnames") and depth < 2:
        for f in obj._fieldnames:
            out.update(_flatten_mat(getattr(obj, f), f, depth + 1))
    return out


def load_mat(path: Path, fs: float = FS_DEFAULT) -> LogData:
    from scipy.io import loadmat

    mat = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    raw: dict[str, np.ndarray] = {}
    for k, v in mat.items():
        if not k.startswith("__"):
            raw.update(_flatten_mat(v, k))
    if not raw:
        raise ValueError(f"{path.name}: no 1-D numeric arrays found")
    tkey = next((k for k in raw if normalize_name(k) in _TIME_KEYS), None)
    if tkey is None:
        t = np.arange(max(v.size for v in raw.values())) / fs
    else:
        t = raw[tkey] * (1e-3 if normalize_name(tkey).endswith("ms") else 1.0)
    raw = {k: v for k, v in raw.items() if v.size == t.size}
    return _build(t, raw, fs, path, {"format": "mat"})


_LOADERS = {".mf4": load_mf4, ".mdf": load_mf4, ".csv": load_csv, ".npz": load_npz, ".mat": load_mat}


def load_log(path: str | Path, fs: float = FS_DEFAULT) -> LogData:
    p = Path(path)
    fn = _LOADERS.get(p.suffix.lower())
    if fn is None:
        raise ValueError(f"unsupported log format: {p.suffix}")
    return fn(p, fs)


# --------------------------------------------------------------------------------------------------------
# Synthetic session (demo / tests). Written with LEGACY logger names so the tyre-temp remap is exercised.
# --------------------------------------------------------------------------------------------------------
def make_demo_log(n_laps: int = 6, fs: float = FS_DEFAULT, seed: int = 1, legacy_names: bool = True) -> LogData:
    rng = np.random.default_rng(seed)
    N = 2400
    th = np.linspace(0, 2 * np.pi, N, endpoint=False)
    dth = th[1] - th[0]
    X = 110 * np.cos(th) + 24 * np.cos(3 * th)
    Y = 64 * np.sin(th) + 18 * np.sin(2 * th)

    def d1(a):
        return (np.roll(a, -1) - np.roll(a, 1)) / (2 * dth)

    dx, dy = d1(X), d1(Y)
    ddx, ddy = d1(dx), d1(dy)
    sp = np.hypot(dx, dy)
    kap = (dx * ddy - dy * ddx) / sp ** 3          # + = left turn (CCW)
    ds = sp * dth
    vc = np.clip(np.sqrt(1.5 * G / np.maximum(np.abs(kap), 1e-3)), 9.0, 28.0)
    v = vc.copy()
    for _ in range(2):
        for i in range(N):
            j = (i + 1) % N
            v[j] = min(v[j], math.sqrt(v[i] ** 2 + 2 * 8.0 * ds[i]))
        for i in range(N - 1, -1, -1):
            j = (i - 1) % N
            v[j] = min(v[j], math.sqrt(v[i] ** 2 + 2 * 13.0 * ds[j]))
    seg_t = ds / (0.5 * (v + np.roll(v, -1)))
    scales = [0.90] + [float(1.0 - 0.025 * rng.random()) for _ in range(n_laps - 1)]
    T_nodes, V_nodes, off = [], [], 0.0
    for sc in scales:
        T_nodes.append(off + np.concatenate([[0.0], np.cumsum(seg_t / sc)[:-1]]))
        V_nodes.append(v * sc)
        off += float(np.sum(seg_t / sc))
    T_nodes, V_nodes = np.concatenate(T_nodes), np.concatenate(V_nodes)
    tg = np.arange(0.0, off - 1.0 / fs, 1.0 / fs)
    dt = 1.0 / fs
    node = np.interp(tg, T_nodes, np.arange(T_nodes.size))
    u = node % N
    ring = np.arange(N + 1)
    Xt = np.interp(u, ring, np.append(X, X[0]))
    Yt = np.interp(u, ring, np.append(Y, Y[0]))
    kt = np.interp(u, ring, np.append(kap, kap[0]))
    vt = np.interp(tg, T_nodes, V_nodes)
    b, a = butter(2, 5.0 / (0.5 * fs))
    vt = filtfilt(b, a, vt)
    ax_g = filtfilt(b, a, np.gradient(vt, dt)) / G
    r = vt * kt
    ay_g = vt * r / G

    ys = np.array([0.60, -0.60, 0.58, -0.58])      # left +
    n = tg.size
    noise = lambda s: rng.normal(0, s, n)  # noqa: E731
    ch: dict[str, np.ndarray] = {}
    ch["vx"] = vt * 3.6 + noise(0.05)
    ch["ax"] = ax_g + noise(0.015)
    ch["ay"] = ay_g + noise(0.015)
    ch["yaw_rate"] = np.degrees(r) + noise(0.3)
    ch["steer_angle"] = np.degrees(1.55 * kt + 0.0035 * ay_g * G) * 4.5 + noise(0.3)
    ch["throttle_pct"] = np.where(ax_g > -0.05, np.clip(ax_g * 1.1 + 0.12, 0, 1), 0.0) * 100
    bf = np.clip(-ax_g - 0.03, 0, None) * 60
    ch["brake_press_front"], ch["brake_press_rear"] = bf, 0.55 * bf
    kslip = 0.05 * np.tanh(ax_g / 0.8)
    base_f = 290.0 * ax_g * G / 4.0
    heat = 0.0
    p_mech = np.zeros(n)
    for i, c in enumerate(CORNERS):
        vhub = vt - r * ys[i]
        ch[f"wheel_speed_{c}"] = vhub * (1 + kslip + noise(0.003)) * 3.6
        tq = np.clip((base_f - ys[i] * np.sign(1.0) * 120.0 * ay_g) * 0.19, -190, 270)
        ch[f"motor_torque_{c}"] = tq + noise(1.0)
        p_mech += tq * (vhub / 0.19)
        pl = np.abs(tq) / 270.0
        tau = 40.0
        filt = lambda x, tau=tau: lfilter([0, dt / tau], [1, -(1 - dt / tau)], x)  # noqa: E731
        ch[f"motor_temp_{c}"] = 35 + 70 * filt(pl ** 2) + noise(0.1)
        ch[f"inverter_temp_{c}"] = 32 + 45 * filt(pl ** 2) + noise(0.1)
        load = np.clip(1 - np.sign(ys[i]) * 0.35 * ay_g, 0.3, 2.0)
        th_i = np.abs(ay_g) * load + 0.3 * np.abs(ax_g)
        ch[f"_tire_{c}"] = 38 + 32 * lfilter([0, dt / 25.0], [1, -(1 - dt / 25.0)], th_i) + noise(0.1)
        roll = 9.0 if i < 2 else 7.0
        pitch = -5.0 * ax_g if i < 2 else 5.0 * ax_g
        bump = filtfilt(*butter(2, 12.0 / (0.5 * fs)), noise(3.0))
        ch[f"damper_travel_{c}"] = 25 - np.sign(ys[i]) * roll * ay_g + pitch + 0.002 * vt ** 2 + bump
    p_el = np.where(p_mech > 0, p_mech / 0.92, p_mech * 0.88) * 1.12
    vpack = 400.0 - 12.0 * np.clip(p_el / 80e3, -1, 1.5) + noise(0.1)
    ch["battery_voltage"] = vpack
    ch["battery_current"] = p_el / vpack
    ch["soc"] = 100.0 - np.cumsum(np.maximum(p_el, 0)) * dt / 3600.0 / 7000.0 * 100.0
    ch["min_cell_voltage"] = vpack / 110.0 - 0.02 + noise(0.001)
    lat0, lon0 = 43.3183, -1.9812
    ch["gps_lat"] = lat0 + (Yt + noise(0.05)) / 111320.0
    ch["gps_lon"] = lon0 + (Xt + noise(0.05)) / (111320.0 * math.cos(math.radians(lat0)))
    lap_no = np.floor(node / N).astype(int)
    bc = np.zeros(n)
    bc[1:] = (lap_no[1:] > lap_no[:-1]).astype(float)
    ch["lap_beacon"] = (np.convolve(bc, np.ones(3), "same") > 0).astype(float)

    logged = {k: v for k, v in ch.items() if not k.startswith("_tire_")}
    for c in CORNERS:   # store physical temps under the (transposed) names the logger would use
        logged[TIRE_TEMP_LOGGED_NAME[c] if legacy_names else f"tire_temp_{c}"] = ch[f"_tire_{c}"]
    return _build(tg, logged, fs, None, {"format": "demo", "n_laps": n_laps})