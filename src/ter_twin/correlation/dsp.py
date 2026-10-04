"""Signal primitives shared by every correlation layer (NumPy/SciPy, deterministic, thread-safe)."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, medfilt

__all__ = ["fill_nan", "lowpass", "hampel", "ols", "r2_score", "xcorr_lag", "shift_signal"]


def fill_nan(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, float)
    bad = ~np.isfinite(y)
    if not bad.any():
        return y.copy()
    if bad.all():
        return np.zeros_like(y)
    x = np.arange(y.size)
    return np.interp(x, x[~bad], y[~bad])


def lowpass(y: np.ndarray, fs: float, fc: float, order: int = 2) -> np.ndarray:
    """Zero-phase Butterworth (adds NO lag, so it cannot mask a hardware-lag diagnosis). NaNs are preserved."""
    y = np.asarray(y, float)
    bad = ~np.isfinite(y)
    if bad.all() or fc >= 0.5 * fs or y.size < 30:
        return y.copy()
    b, a = butter(order, fc / (0.5 * fs))
    out = filtfilt(b, a, fill_nan(y))
    out[bad] = np.nan
    return out


def hampel(y: np.ndarray, win: int = 5, k: float = 7.0) -> tuple[np.ndarray, int]:
    """Median/MAD despike. Returns (cleaned, n_replaced)."""
    y = np.asarray(y, float)
    bad = ~np.isfinite(y)
    f = fill_nan(y)
    kern = 2 * win + 1
    med = medfilt(f, kern)
    mad = 1.4826 * medfilt(np.abs(f - med), kern)
    out = np.abs(f - med) > k * np.maximum(mad, 1e-12)
    g = np.where(out, med, f)
    g[bad] = np.nan
    return g, int(out[~bad].sum())


def ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """y = s*x + c. Returns (slope, intercept, pearson_rho); NaNs if degenerate."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan"), float("nan"), float("nan")
    x, y = x[m], y[m]
    vx = float(np.var(x))
    if vx <= 0.0:
        return float("nan"), float("nan"), float("nan")
    s = float(np.cov(x, y, bias=True)[0, 1] / vx)
    c = float(y.mean() - s * x.mean())
    rho = float(np.corrcoef(x, y)[0, 1]) if np.std(y) > 0 else float("nan")
    return s, c, rho


def r2_score(y: np.ndarray, p: np.ndarray, mask: np.ndarray | None = None) -> float:
    m = np.isfinite(y) & np.isfinite(p)
    if mask is not None:
        m &= mask
    if m.sum() < 3:
        return float("nan")
    ss = float(np.sum((y[m] - y[m].mean()) ** 2))
    return float(1.0 - np.sum((p[m] - y[m]) ** 2) / ss) if ss > 0 else float("nan")


def xcorr_lag(meas: np.ndarray, ref: np.ndarray, fs: float, max_s: float = 0.5) -> tuple[float, float]:
    """Delay of ``meas`` w.r.t. ``ref`` [s] (>0: meas lags) and the normalised peak correlation.

    FFT cross-correlation on the jointly-finite samples, sub-sample parabolic refinement.
    """
    m = np.isfinite(meas) & np.isfinite(ref)
    if m.sum() < 200:
        return 0.0, float("nan")
    a = np.where(m, meas - np.mean(meas[m]), 0.0)
    b = np.where(m, ref - np.mean(ref[m]), 0.0)
    n = a.size
    nfft = 1 << int(2 * n - 1).bit_length()
    cc = np.fft.irfft(np.fft.rfft(a, nfft) * np.conj(np.fft.rfft(b, nfft)), nfft)  # cc[k] = sum a[i+k] b[i]
    kmax = max(1, int(max_s * fs))
    lags = np.arange(-kmax, kmax + 1)
    vals = cc[lags % nfft]
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den <= 0:
        return 0.0, float("nan")
    k = int(np.argmax(vals))
    delta = 0.0
    if 0 < k < vals.size - 1:
        y0, y1, y2 = vals[k - 1], vals[k], vals[k + 1]
        d = y0 - 2.0 * y1 + y2
        delta = 0.5 * (y0 - y2) / d if abs(d) > 1e-18 else 0.0
    return float((lags[k] + delta) / fs), float(vals[k] / den)


def shift_signal(t: np.ndarray, y: np.ndarray, lag_s: float) -> np.ndarray:
    """y(t + lag): advance a signal that lags by ``lag_s``."""
    return np.interp(t + lag_s, t, y, left=np.nan, right=np.nan)