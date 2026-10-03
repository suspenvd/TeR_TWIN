"""Thread-safe circular telemetry buffer + plot decimators.

Storage is a pre-allocated contiguous ``(n_channels, capacity)`` float64 array plus a time vector, so a
writer thread (CAN/UDP/serial ingest at 200 Hz) never allocates. A lightweight ``threading.Lock`` guards
index bookkeeping; readers copy only the requested window. NaN means "no value yet".
"""
from __future__ import annotations

import threading
from typing import Mapping, Sequence

import numpy as np

__all__ = ["RingBuffer", "minmax_indices", "lttb_indices", "decimate_indices"]


class RingBuffer:
    def __init__(self, channels: Sequence[str], capacity: int = 120_000) -> None:
        self.names: tuple[str, ...] = tuple(dict.fromkeys(channels))
        self._idx = {n: i for i, n in enumerate(self.names)}
        self.capacity = int(capacity)
        self._data = np.full((len(self.names), self.capacity), np.nan)
        self._t = np.full(self.capacity, np.nan)
        self._lap = np.full(self.capacity, -1, dtype=np.int32)
        self._head = 0
        self._count = 0
        self._total = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ writing
    def append_block(self, t: Sequence[float] | np.ndarray, block: Mapping[str, Sequence[float] | np.ndarray]) -> None:
        t = np.atleast_1d(np.asarray(t, dtype=float))
        m = t.size
        if m == 0:
            return
        cap = self.capacity
        with self._lock:
            if m > cap:
                t = t[-cap:]
                block = {k: np.asarray(v, dtype=float)[-cap:] for k, v in block.items()}
                m = cap
            idx = (self._head + np.arange(m)) % cap
            self._t[idx] = t
            self._data[:, idx] = np.nan
            self._lap[idx] = -1
            for k, v in block.items():
                i = self._idx.get(k)
                if i is not None:
                    self._data[i, idx] = np.asarray(v, dtype=float)
            self._head = (self._head + m) % cap
            self._count = min(cap, self._count + m)
            self._total += m

    def append(self, t: float, values: Mapping[str, float]) -> None:
        self.append_block([t], {k: [v] for k, v in values.items()})

    def set_lap(self, lap_index: int, t0: float, t1: float) -> None:
        """Label the samples with t0 <= t < t1 as belonging to ``lap_index``."""
        with self._lock:
            idx, tl = self._logical()
            a, b = np.searchsorted(tl, t0, "left"), np.searchsorted(tl, t1, "left")
            self._lap[idx[a:b]] = lap_index

    # ------------------------------------------------------------------ reading
    def _logical(self) -> tuple[np.ndarray, np.ndarray]:
        start = (self._head - self._count) % self.capacity
        idx = (start + np.arange(self._count)) % self.capacity
        return idx, self._t[idx]

    def _select(self, sel: np.ndarray, channels: Sequence[str] | None) -> dict[str, np.ndarray]:
        names = self.names if channels is None else [c for c in channels if c in self._idx]
        return {n: self._data[self._idx[n], sel].copy() for n in names}

    def get_range(self, t0: float, t1: float, channels: Sequence[str] | None = None):
        with self._lock:
            idx, tl = self._logical()
            a, b = np.searchsorted(tl, t0, "left"), np.searchsorted(tl, t1, "right")
            return tl[a:b].copy(), self._select(idx[a:b], channels)

    def get_time_window(self, t_center: float, window_sec: float, channels: Sequence[str] | None = None):
        h = 0.5 * float(window_sec)
        return self.get_range(t_center - h, t_center + h, channels)

    def get_latest(self, n_samples: int, channels: Sequence[str] | None = None):
        with self._lock:
            idx, tl = self._logical()
            n = min(int(n_samples), idx.size)
            sel = idx[idx.size - n:]
            return self._t[sel].copy(), self._select(sel, channels)

    def get_channel(self, channel_name: str, lap_index: int | None = None):
        i = self._idx.get(channel_name)
        if i is None:
            return np.zeros(0), np.zeros(0)
        with self._lock:
            idx, tl = self._logical()
            if lap_index is not None:
                keep = self._lap[idx] == lap_index
                idx, tl = idx[keep], tl[keep]
            return tl.copy(), self._data[i, idx].copy()

    # ------------------------------------------------------------------ status
    @property
    def count(self) -> int:
        return self._count

    @property
    def total_written(self) -> int:
        return self._total

    @property
    def fill_fraction(self) -> float:
        return self._count / self.capacity

    @property
    def latest_time(self) -> float:
        with self._lock:
            return float(self._t[(self._head - 1) % self.capacity]) if self._count else float("nan")

    def __len__(self) -> int:
        return self._count


# --------------------------------------------------------------------------------------------------------
# Decimation (index selectors; apply the same indices to x and y)
# --------------------------------------------------------------------------------------------------------
def minmax_indices(y: np.ndarray, max_points: int) -> np.ndarray:
    """Per-bucket min & max (keeps every peak). NaN-tolerant. Returns sorted unique indices."""
    n = y.size
    if n <= max_points:
        return np.arange(n)
    nb = max(max_points // 2, 1)
    m = (n // nb) * nb
    yb = y[:m].reshape(nb, -1)
    fin = np.isfinite(yb)
    lo = np.where(fin, yb, np.inf).argmin(axis=1)
    hi = np.where(fin, yb, -np.inf).argmax(axis=1)
    base = np.arange(nb) * (m // nb)
    return np.unique(np.concatenate([base + lo, base + hi, [0, n - 1]]))


def lttb_indices(x: np.ndarray, y: np.ndarray, n_out: int) -> np.ndarray:
    """Largest-Triangle-Three-Buckets (Steinarsson 2013). Requires finite data."""
    n = x.size
    if n_out >= n or n_out < 3:
        return np.arange(n)
    every = (n - 2) / (n_out - 2)
    sel = np.empty(n_out, dtype=int)
    sel[0] = 0
    a = 0
    for i in range(n_out - 2):
        s, e = int((i + 1) * every) + 1, min(int((i + 2) * every) + 1, n)
        ax_, ay_ = x[s:e].mean(), y[s:e].mean()
        rs, re_ = int(i * every) + 1, int((i + 1) * every) + 1
        areas = np.abs((x[a] - ax_) * (y[rs:re_] - y[a]) - (x[a] - x[rs:re_]) * (ay_ - y[a]))
        a = rs + int(np.argmax(areas))
        sel[i + 1] = a
    sel[-1] = n - 1
    return sel


def decimate_indices(x: np.ndarray, y: np.ndarray, max_points: int, method: str = "minmax") -> np.ndarray:
    if y.size <= max_points:
        return np.arange(y.size)
    if method == "lttb" and np.isfinite(y).all():
        return lttb_indices(x, y, max_points)
    return minmax_indices(y, max_points)