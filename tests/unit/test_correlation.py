"""Fault-injection tests for the diagnostic engine + physical invariants."""
from __future__ import annotations

import numpy as np
import pytest

from ter_twin.correlation.diagnostics import DiagContext, diagnose_channel
from ter_twin.correlation.signal_integrity import IntegrityConfig, condition_channels

FS = 200.0
rng = np.random.default_rng(3)
t = np.arange(0, 120, 1 / FS)
vx = np.where(t < 5, 0.0, 15.0 + 8.0 * np.sin(2 * np.pi * t / 53))
base = 1.2 * np.sin(2 * np.pi * t / 6) * np.sin(2 * np.pi * t / 37) * (vx > 1)
M = vx > 4.2
ctx = DiagContext(fs=FS, unit="g", odd=True, kind="lateral", zero_tol=0.01)


def codes(meas, pred):
    return {d.code: d for d in diagnose_channel("ay", t, meas, pred, M, vx, ctx, always=True)}


def noisy(x):
    return x + rng.normal(0, 0.005, x.size)


def test_gain():
    d = codes(noisy(2.0 * base), base)["GAIN_SCALE"]
    assert d.evidence["slope"] == pytest.approx(0.5, rel=0.05) and d.r2_gain > 0.2


def test_lag():
    k = int(0.12 * FS)
    d = codes(noisy(np.concatenate([np.zeros(k), base[:-k]])), base)["PHASE_LAG"]
    assert d.evidence["lag_ms"] == pytest.approx(120.0, abs=15.0)


def test_offset():
    d = codes(noisy(base + 0.03), base)["STATIC_OFFSET"]
    assert d.evidence["offset"] == pytest.approx(0.03, abs=0.005)


def test_v2():
    assert "AERO_V2" in codes(noisy(base), base + 0.0012 * vx ** 2 * (vx > 1))


def test_saturation():
    d = codes(noisy(0.8 * np.tanh(base / 0.8)), base)["MU_PEAK"]
    assert d.evidence["lambda_mu"] < 0.93


def test_asymmetry():
    assert "ASYM_GAIN" in codes(noisy(np.where(base > 0, base, 0.8 * base)), base)


def test_layer0_polarity_and_tare():
    ay = noisy(vx * 0.0 + 0.3 * np.sin(2 * np.pi * t / 6)) 
    yaw = np.degrees(np.sin(2 * np.pi * t / 6) * 0.3 * 9.80665 / np.maximum(vx, 4.0))
    rep = condition_channels(t, {"vx": vx * 3.6, "ay": -(ay * (vx > 4.2) + 0.05), "yaw_rate": yaw * (vx > 4.2)},
                             IntegrityConfig(lowpass=False, despike=False))
    assert rep.has("POLARITY") and rep.has("TARE_OFFSET", "ay")


def test_physical_invariants():
    from ter_twin.correlation.invariants import run_invariants
    from ter_twin.models import vehicle as V

    bad = [r for r in run_invariants(V.default_ter27()) if not r.passed]
    assert not bad, bad