"""Metrics, grading and result containers."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

GRADE_PASS_R2, GRADE_PASS_NMAE, GRADE_FAIL_R2 = 0.90, 8.0, 0.80


class Severity(str, Enum):
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"


@dataclass(frozen=True)
class ChannelMetrics:
    n: int
    rmse: float
    nmae: float      # % of p1-p99 range
    r2: float
    rho: float
    bias: float      # mean(pred - meas)
    emax: float
    slope: float     # OLS slope of pred on meas (1.0 = perfect gain)

    @property
    def grade(self) -> str:
        return grade(self.r2, self.nmae)


def grade(r2: float, nmae: float) -> str:
    if not (math.isfinite(r2) and math.isfinite(nmae)):
        return "N/A"
    if r2 < GRADE_FAIL_R2:
        return "FAIL"
    return "PASS" if (r2 >= GRADE_PASS_R2 and nmae <= GRADE_PASS_NMAE) else "WARN"


def compute_metrics(meas: np.ndarray, pred: np.ndarray, mask: np.ndarray | None = None) -> ChannelMetrics | None:
    m = np.isfinite(meas) & np.isfinite(pred)
    if mask is not None:
        m &= mask
    if int(m.sum()) < 50:
        return None
    y, p = meas[m], pred[m]
    e = p - y
    rng = float(np.percentile(y, 99) - np.percentile(y, 1)) or 1.0
    ss = float(np.sum((y - y.mean()) ** 2))
    vy = float(np.var(y))
    return ChannelMetrics(
        n=int(y.size), rmse=float(np.sqrt(np.mean(e ** 2))), nmae=float(np.mean(np.abs(e)) / rng * 100.0),
        r2=float(1.0 - np.sum(e ** 2) / ss) if ss > 0 else float("nan"),
        rho=float(np.corrcoef(y, p)[0, 1]) if y.std() > 0 and p.std() > 0 else float("nan"),
        bias=float(e.mean()), emax=float(np.max(np.abs(e))),
        slope=float(np.cov(y, p, bias=True)[0, 1] / vy) if vy > 0 else float("nan"))


@dataclass(frozen=True)
class Finding:
    """Layer-0 observation about a raw channel."""
    code: str
    channel: str
    severity: Severity
    message: str
    data: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Diagnosis:
    """Layer-1/2 root-cause hypothesis with counterfactual evidence."""
    code: str
    channel: str
    title: str
    action: str
    confidence: float            # 0..1
    r2_gain: float               # counterfactual: R2 after applying the hypothesised fix minus current R2
    evidence: dict[str, float] = field(default_factory=dict)

    @property
    def score(self) -> float:
        return self.confidence * max(self.r2_gain, 0.02)


@dataclass
class SeriesResult:
    name: str
    unit: str
    layer: str                   # "L1" | "L2" | "L2-OL"
    mode: str                    # how pred was produced (open-loop, held-out, one-step-ahead...)
    t: np.ndarray
    meas: np.ndarray
    pred: np.ndarray
    mask: np.ndarray
    vx_kmh: np.ndarray
    metrics: ChannelMetrics | None = None
    diagnoses: list[Diagnosis] = field(default_factory=list)
    lag_ms: float = float("nan")

    def compute(self) -> None:
        self.metrics = compute_metrics(self.meas, self.pred, self.mask)