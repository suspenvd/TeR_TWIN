"""Magic Formula 6.1 (Besselink-Schmeitz-Pacejka, con sensibilidad a presion) en JAX.

Suposiciones documentadas
-------------------------
* Convencion "pendiente positiva": Fy > 0 para alpha > 0, Fx > 0 para kappa > 0, Mz < 0 para
  alpha > 0 (Mz = -t*Fy'). Los datos TTC (SAE) se transforman en el ajuste, no aqui.
* Unidades SI: alpha, gamma [rad], kappa [-], Fz [N], p [kPa], R0 [m], Mz [N m].
* Rodadura hacia delante (Vcx > 0), estado estacionario (sin relajacion ni zeta-factors: zeta_i = 1,
  sin dependencia de velocidad: lambda_mu_V = 0). Relajacion transitoria vive en relaxation.py.
* alpha* = tan(alpha), gamma* = sin(gamma), cos(alpha') = cos(alpha).
* Fy' (para Mz) = Fy - SVyk - SVyg (se retira el desplazamiento vertical que ya modela Mzr).
* sgn() y |.| son suaves (eps ~1e-6) para conservar derivadas finitas en 0 (jax.grad/jacfwd).
* Todas las funciones son puras, broadcastables y jit/vmap/grad-compatibles. Para ajuste usar x64.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import yaml

EPS = 1e-6
KYA_FLOOR = 1.0  # [N/rad] evita 0/0 en SHy cuando Fz -> 0
_A_MU = 10.0  # constante A_mu de lambda'_mu (MF6.1, sin velocidad)
CONVENTION = "positive-slope: Fy(+alpha)>0, Fx(+kappa)>0, Mz(+alpha)<0; SI; alpha/gamma in rad"


# --------------------------------------------------------------------------------------------
# Especificacion de parametros
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ParamSpec:
    name: str
    default: float
    lo: float
    hi: float
    group: str  # static | fx | fy | mz | fxc | fyc | mzc | scale
    kind: str = "coef"  # coef | pressure | scale | static
    needs: tuple[str, ...] = ()  # fz2 | fz3 | gamma | p  (condiciones de identificabilidad)


def _c(name, d, lo, hi, group, *needs, kind="coef"):
    return ParamSpec(name, float(d), float(lo), float(hi), group, kind, tuple(needs))


def _pp(name, group, lo=-2.0, hi=2.0, d=0.0):
    return ParamSpec(name, d, lo, hi, group, "pressure", ("p",))


_SPECS: tuple[ParamSpec, ...] = (
    # ---- estaticos / nominales
    ParamSpec("FNOMIN", 1100.0, 100.0, 6000.0, "static", "static"),
    ParamSpec("R0", 0.2032, 0.10, 0.45, "static", "static"),
    ParamSpec("NOMPRES", 83.0, 20.0, 400.0, "static", "static"),
    # ---- Fx puro
    _c("pCx1", 1.6, 1.0, 2.5, "fx"),
    _c("pDx1", 2.2, 0.5, 4.0, "fx"),
    _c("pDx2", -0.15, -1.0, 1.0, "fx", "fz2"),
    _c("pDx3", 0.0, -30.0, 30.0, "fx", "gamma"),
    _c("pEx1", 0.1, -5.0, 1.0, "fx"),
    _c("pEx2", 0.0, -5.0, 5.0, "fx", "fz2"),
    _c("pEx3", 0.0, -5.0, 5.0, "fx", "fz3"),
    _c("pEx4", 0.0, -1.0, 1.0, "fx"),
    _c("pKx1", 30.0, 5.0, 200.0, "fx"),
    _c("pKx2", 0.0, -50.0, 50.0, "fx", "fz2"),
    _c("pKx3", -0.1, -3.0, 3.0, "fx", "fz2"),
    _c("pHx1", 0.0, -0.05, 0.05, "fx"),
    _c("pHx2", 0.0, -0.05, 0.05, "fx", "fz2"),
    _c("pVx1", 0.0, -0.3, 0.3, "fx"),
    _c("pVx2", 0.0, -0.3, 0.3, "fx", "fz2"),
    _pp("ppx1", "fx"), _pp("ppx2", "fx"), _pp("ppx3", "fx", -1.0, 1.0), _pp("ppx4", "fx", -1.0, 1.0),
    # ---- Fy puro
    _c("pCy1", 1.4, 1.0, 2.0, "fy"),
    _c("pDy1", 2.4, 0.5, 4.0, "fy"),
    _c("pDy2", -0.25, -1.0, 1.0, "fy", "fz2"),
    _c("pDy3", 5.0, -30.0, 30.0, "fy", "gamma"),
    _c("pEy1", -0.5, -5.0, 1.0, "fy"),
    _c("pEy2", 0.0, -5.0, 5.0, "fy", "fz2"),
    _c("pEy3", 0.0, -1.0, 1.0, "fy"),
    _c("pEy4", 0.0, -10.0, 10.0, "fy", "gamma"),
    _c("pEy5", 0.0, -50.0, 50.0, "fy", "gamma"),
    _c("pKy1", 45.0, 5.0, 200.0, "fy"),
    _c("pKy2", 1.5, 0.3, 6.0, "fy", "fz2"),
    _c("pKy3", 0.5, -2.0, 2.0, "fy", "gamma"),
    _c("pKy4", 2.0, 1.0, 4.0, "fy", "fz2"),
    _c("pKy5", 0.0, -10.0, 10.0, "fy", "gamma"),
    _c("pKy6", 3.0, -10.0, 10.0, "fy", "gamma"),
    _c("pKy7", 0.0, -5.0, 5.0, "fy", "gamma", "fz2"),
    _c("pHy1", 0.0, -0.05, 0.05, "fy"),
    _c("pHy2", 0.0, -0.05, 0.05, "fy", "fz2"),
    _c("pVy1", 0.0, -0.3, 0.3, "fy"),
    _c("pVy2", 0.0, -0.3, 0.3, "fy", "fz2"),
    _c("pVy3", 0.0, -3.0, 3.0, "fy", "gamma"),
    _c("pVy4", 0.0, -3.0, 3.0, "fy", "gamma", "fz2"),
    _pp("ppy1", "fy"), _pp("ppy2", "fy"), _pp("ppy3", "fy", -1.0, 1.0), _pp("ppy4", "fy", -1.0, 1.0),
    _pp("ppy5", "fy"),
    # ---- Mz puro (traza neumatica t + par residual Mzr)
    _c("qBz1", 10.0, 1.0, 40.0, "mz"),
    _c("qBz2", -1.0, -20.0, 20.0, "mz", "fz2"),
    _c("qBz3", 0.0, -20.0, 20.0, "mz", "fz3"),
    _c("qBz4", 0.0, -10.0, 10.0, "mz", "gamma"),
    _c("qBz5", 0.0, -10.0, 10.0, "mz", "gamma"),
    _c("qBz9", 0.0, -50.0, 50.0, "mz"),
    _c("qBz10", 0.0, -5.0, 5.0, "mz"),
    _c("qCz1", 1.2, 0.8, 2.5, "mz"),
    _c("qDz1", 0.15, 0.01, 0.5, "mz"),
    _c("qDz2", 0.0, -0.5, 0.5, "mz", "fz2"),
    _c("qDz3", 0.0, -5.0, 5.0, "mz", "gamma"),
    _c("qDz4", 0.0, -50.0, 50.0, "mz", "gamma"),
    _c("qDz6", 0.0, -0.1, 0.1, "mz"),
    _c("qDz7", 0.0, -0.1, 0.1, "mz", "fz2"),
    _c("qDz8", 0.0, -2.0, 2.0, "mz", "gamma"),
    _c("qDz9", 0.0, -2.0, 2.0, "mz", "gamma", "fz2"),
    _c("qDz10", 0.0, -2.0, 2.0, "mz", "gamma"),
    _c("qDz11", 0.0, -2.0, 2.0, "mz", "gamma", "fz2"),
    _c("qEz1", -1.0, -5.0, 1.0, "mz"),
    _c("qEz2", 0.0, -5.0, 5.0, "mz", "fz2"),
    _c("qEz3", 0.0, -5.0, 5.0, "mz", "fz3"),
    _c("qEz4", 0.0, -2.0, 2.0, "mz", "gamma"),
    _c("qEz5", 0.0, -10.0, 10.0, "mz", "gamma"),
    _c("qHz1", 0.0, -0.02, 0.02, "mz"),
    _c("qHz2", 0.0, -0.02, 0.02, "mz", "fz2"),
    _c("qHz3", 0.0, -0.5, 0.5, "mz", "gamma"),
    _c("qHz4", 0.0, -0.5, 0.5, "mz", "gamma", "fz2"),
    _pp("ppz1", "mz"), _pp("ppz2", "mz"),
    # ---- combinada Fx
    _c("rBx1", 12.0, 2.0, 40.0, "fxc"),
    _c("rBx2", 10.0, 1.0, 40.0, "fxc"),
    _c("rBx3", 0.0, -20.0, 20.0, "fxc", "gamma"),
    _c("rCx1", 1.0, 0.5, 1.5, "fxc"),
    _c("rEx1", 0.0, -1.0, 1.0, "fxc"),
    _c("rEx2", 0.0, -1.0, 1.0, "fxc", "fz2"),
    _c("rHx1", 0.0, -0.1, 0.1, "fxc"),
    # ---- combinada Fy
    _c("rBy1", 10.0, 2.0, 40.0, "fyc"),
    _c("rBy2", 8.0, 1.0, 40.0, "fyc"),
    _c("rBy3", 0.0, -0.3, 0.3, "fyc"),
    _c("rBy4", 0.0, -20.0, 20.0, "fyc", "gamma"),
    _c("rCy1", 1.0, 0.5, 1.5, "fyc"),
    _c("rEy1", 0.0, -2.0, 1.0, "fyc"),
    _c("rEy2", 0.0, -2.0, 2.0, "fyc", "fz2"),
    _c("rHy1", 0.0, -0.1, 0.1, "fyc"),
    _c("rHy2", 0.0, -0.1, 0.1, "fyc", "fz2"),
    _c("rVy1", 0.0, -0.5, 0.5, "fyc"),
    _c("rVy2", 0.0, -0.5, 0.5, "fyc", "fz2"),
    _c("rVy3", 0.0, -5.0, 5.0, "fyc", "gamma"),
    _c("rVy4", 10.0, 0.0, 50.0, "fyc"),
    _c("rVy5", 1.9, 0.5, 5.0, "fyc"),
    _c("rVy6", 10.0, 0.0, 50.0, "fyc"),
    # ---- combinada Mz (brazo de palanca s de Fx)
    _c("sSz1", 0.01, -0.1, 0.1, "mzc"),
    _c("sSz2", 0.0, -0.1, 0.1, "mzc"),
    _c("sSz3", 0.0, -0.1, 0.1, "mzc", "gamma"),
    _c("sSz4", 0.0, -0.1, 0.1, "mzc", "gamma", "fz2"),
) + tuple(
    ParamSpec(n, 1.0, 0.1, 3.0, "scale", "scale")
    for n in (
        "lFz0 lCx lMux lEx lKxk lHx lVx lCy lMuy lEy lKya lKyg lHy lVy lt lr lxa lyk lVyk ls lgz".split()
    )
)

SPECS: dict[str, ParamSpec] = {s.name: s for s in _SPECS}
NAMES: tuple[str, ...] = tuple(SPECS)
DEFAULTS: dict[str, float] = {s.name: s.default for s in _SPECS}
GROUPS: dict[str, tuple[ParamSpec, ...]] = {}
for _s in _SPECS:
    GROUPS.setdefault(_s.group, ())
    GROUPS[_s.group] += (_s,)


@jax.tree_util.register_pytree_node_class
class MF61Params:
    """Contenedor inmutable-por-convencion de coeficientes MF6.1, registrado como pytree."""

    __slots__ = ("_d",)

    def __init__(self, values: Mapping[str, Any] | None = None, **kw: Any):
        merged = dict(DEFAULTS)
        for src in (values or {}, kw):
            for k, v in src.items():
                if k not in SPECS:
                    raise KeyError(f"Coeficiente MF6.1 desconocido: {k!r}")
                merged[k] = v
        object.__setattr__(self, "_d", {k: merged[k] for k in NAMES})

    def __getattr__(self, name: str):
        try:
            return object.__getattribute__(self, "_d")[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        raise AttributeError("MF61Params es inmutable; usa replace()/with_vector()")

    def __reduce__(self):
        return (_rebuild_params, (self.to_dict(),))

    def __repr__(self) -> str:
        return f"MF61Params(FNOMIN={self.FNOMIN}, R0={self.R0}, NOMPRES={self.NOMPRES}, n={len(NAMES)})"

    # pytree
    def tree_flatten(self):
        return tuple(self._d[k] for k in NAMES), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        object.__setattr__(obj, "_d", dict(zip(NAMES, children)))
        return obj

    # utilidades
    def replace(self, **kw: Any) -> "MF61Params":
        return MF61Params({**self._d, **kw})

    def to_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in self._d.items()}

    def to_vector(self, names: Sequence[str]) -> np.ndarray:
        return np.array([float(self._d[n]) for n in names], dtype=np.float64)

    def with_vector(self, names: Sequence[str], vec) -> "MF61Params":
        """Sustituye coeficientes por `vec` (admite tracers de JAX; sin validacion para ir a jit)."""
        d = dict(self._d)
        for i, n in enumerate(names):
            d[n] = vec[i]
        obj = object.__new__(MF61Params)
        object.__setattr__(obj, "_d", d)
        return obj

    # YAML
    def save(self, path: str | Path, meta: Mapping[str, Any] | None = None) -> None:
        doc = {"model": "MF6.1", "convention": CONVENTION, "meta": dict(meta or {}), "parameters": self.to_dict()}
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")

    @classmethod
    def load_with_meta(cls, path: str | Path) -> tuple["MF61Params", dict]:
        doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if doc.get("model") != "MF6.1":
            raise ValueError(f"{path}: no es un fichero MF6.1")
        return cls(doc["parameters"]), dict(doc.get("meta") or {})

    @classmethod
    def load(cls, path: str | Path) -> "MF61Params":
        return cls.load_with_meta(path)[0]


def _rebuild_params(d: dict) -> MF61Params:
    return MF61Params(d)


# --------------------------------------------------------------------------------------------
# Primitivas
# --------------------------------------------------------------------------------------------
def _sgn(x):
    return x * jax.lax.rsqrt(x * x + 1e-12)


def _mf_arg(b, x, e):
    bx = b * x
    return jnp.arctan(bx - e * (bx - jnp.arctan(bx)))


def _lmu_prime(lmu):
    return _A_MU * lmu / (1.0 + (_A_MU - 1.0) * lmu)


class _Ctx(NamedTuple):
    fz: Any
    fz0p: Any
    dfz: Any
    dpi: Any
    gs: Any  # sin(gamma)
    g2: Any
    ag: Any  # |sin(gamma)| suave
    lmux_p: Any
    lmuy_p: Any


def _ctx(P: MF61Params, fz, gamma, p) -> _Ctx:
    fz = jnp.maximum(jnp.asarray(fz), 0.0)
    fz0p = P.FNOMIN * P.lFz0
    dpi = 0.0 if p is None else (jnp.asarray(p) - P.NOMPRES) / P.NOMPRES
    gs = jnp.sin(jnp.asarray(gamma))
    return _Ctx(fz, fz0p, (fz - fz0p) / fz0p, dpi, gs, gs * gs, jnp.sqrt(gs * gs + 1e-12),
                _lmu_prime(P.lMux), _lmu_prime(P.lMuy))


class _FxPure(NamedTuple):
    fx0: Any
    mux: Any
    kxk: Any


class _FyPure(NamedTuple):
    fy0: Any
    muy: Any
    kya: Any
    kyas: Any
    shy: Any
    svy: Any
    svyg: Any
    by: Any
    cy: Any


def _fx0(P, c: _Ctx, kappa) -> _FxPure:
    cx = P.pCx1 * P.lCx
    mux = (P.pDx1 + P.pDx2 * c.dfz) * (1 + P.ppx3 * c.dpi + P.ppx4 * c.dpi**2) * (1 - P.pDx3 * c.g2) * P.lMux
    dx = mux * c.fz
    kxk = c.fz * (P.pKx1 + P.pKx2 * c.dfz) * jnp.exp(P.pKx3 * c.dfz) * (1 + P.ppx1 * c.dpi + P.ppx2 * c.dpi**2) * P.lKxk
    bx = kxk / (cx * dx + EPS)
    shx = (P.pHx1 + P.pHx2 * c.dfz) * P.lHx
    svx = c.fz * (P.pVx1 + P.pVx2 * c.dfz) * P.lVx * c.lmux_p
    kx = kappa + shx
    ex = jnp.minimum((P.pEx1 + P.pEx2 * c.dfz + P.pEx3 * c.dfz**2) * (1 - P.pEx4 * _sgn(kx)) * P.lEx, 1.0)
    return _FxPure(dx * jnp.sin(cx * _mf_arg(bx, kx, ex)) + svx, mux, kxk)


def _fy0(P, c: _Ctx, alpha) -> _FyPure:
    a = jnp.tan(alpha)
    cy = P.pCy1 * P.lCy
    muy = (P.pDy1 + P.pDy2 * c.dfz) * (1 + P.ppy3 * c.dpi + P.ppy4 * c.dpi**2) * (1 - P.pDy3 * c.g2) * P.lMuy
    dy = muy * c.fz
    kya = (P.pKy1 * c.fz0p * (1 + P.ppy1 * c.dpi) * (1 - P.pKy3 * c.ag)
           * jnp.sin(P.pKy4 * jnp.arctan((c.fz / c.fz0p) / ((P.pKy2 + P.pKy5 * c.g2) * (1 + P.ppy2 * c.dpi))))
           * P.lKya)
    kyas = jnp.maximum(kya, KYA_FLOOR)
    kyg0 = c.fz * (P.pKy6 + P.pKy7 * c.dfz) * (1 + P.ppy5 * c.dpi) * P.lKyg
    svyg = c.fz * (P.pVy3 + P.pVy4 * c.dfz) * c.gs * P.lKyg * c.lmuy_p
    shy = (P.pHy1 + P.pHy2 * c.dfz) * P.lHy + (kyg0 * c.gs - svyg) / kyas
    svy = c.fz * (P.pVy1 + P.pVy2 * c.dfz) * P.lVy * c.lmuy_p + svyg
    ay = a + shy
    ey = jnp.minimum((P.pEy1 + P.pEy2 * c.dfz)
                     * (1 + P.pEy5 * c.g2 - (P.pEy3 + P.pEy4 * c.gs) * _sgn(ay)) * P.lEy, 1.0)
    by = kya / (cy * dy + EPS)
    return _FyPure(dy * jnp.sin(cy * _mf_arg(by, ay, ey)) + svy, muy, kya, kyas, shy, svy, svyg, by, cy)


def _trail_and_residual(P, c: _Ctx, fy: _FyPure, kxk, alpha, kappa):
    """t(alpha_t,eq) [m] y Mzr(alpha_r,eq) [N m]; con kappa=0 reduce a MF pura."""
    a = jnp.tan(alpha)
    ca = jnp.cos(alpha)
    bt = (P.qBz1 + P.qBz2 * c.dfz + P.qBz3 * c.dfz**2) * (1 + P.qBz4 * c.gs + P.qBz5 * c.ag) * P.lKya / c.lmuy_p
    ct = P.qCz1
    dt = (c.fz * (P.R0 / c.fz0p) * (P.qDz1 + P.qDz2 * c.dfz) * (1 + P.ppz1 * c.dpi) * P.lt
          * (1 + P.qDz3 * c.gs + P.qDz4 * c.g2))
    sht = P.qHz1 + P.qHz2 * c.dfz + (P.qHz3 + P.qHz4 * c.dfz) * c.gs
    shf = fy.shy + fy.svy / fy.kyas
    at, ar = a + sht, a + shf
    kx = (kxk / fy.kyas) * kappa
    at_eq = _sgn(at) * jnp.sqrt(at * at + kx * kx + 1e-12)
    ar_eq = _sgn(ar) * jnp.sqrt(ar * ar + kx * kx + 1e-12)
    et = jnp.minimum((P.qEz1 + P.qEz2 * c.dfz + P.qEz3 * c.dfz**2)
                     * (1 + (P.qEz4 + P.qEz5 * c.gs) * (2.0 / math.pi) * jnp.arctan(bt * ct * at_eq)), 1.0)
    t = dt * jnp.cos(ct * _mf_arg(bt, at_eq, et)) * ca
    br = P.qBz9 * P.lKya / c.lmuy_p + P.qBz10 * fy.by * fy.cy
    dr = c.fz * P.R0 * ((P.qDz6 + P.qDz7 * c.dfz) * P.lr
                        + ((P.qDz8 + P.qDz9 * c.dfz) * (1 + P.ppz2 * c.dpi)
                           + (P.qDz10 + P.qDz11 * c.dfz) * c.ag) * c.gs * P.lgz) * c.lmuy_p
    mzr = dr * jnp.cos(jnp.arctan(br * ar_eq)) * ca
    return t, mzr


class MF61Result(NamedTuple):
    fx: Any
    fy: Any
    mz: Any
    fx0: Any
    fy0: Any
    mz0: Any
    t: Any  # traza neumatica [m]
    mzr: Any  # par residual [N m]
    kya: Any  # rigidez de deriva [N/rad]
    kxk: Any  # rigidez longitudinal [N]
    mux: Any
    muy: Any


# --------------------------------------------------------------------------------------------
# API publica: puras
# --------------------------------------------------------------------------------------------
def lateral_pure_force(P, alpha, fz, gamma=0.0, p=None):
    return _fy0(P, _ctx(P, fz, gamma, p), alpha).fy0


def longitudinal_pure_force(P, kappa, fz, gamma=0.0, p=None):
    return _fx0(P, _ctx(P, fz, gamma, p), kappa).fx0


def aligning_pure_moment(P, alpha, fz, gamma=0.0, p=None):
    c = _ctx(P, fz, gamma, p)
    fy = _fy0(P, c, alpha)
    t, mzr = _trail_and_residual(P, c, fy, 0.0, alpha, 0.0)
    return -t * (fy.fy0 - fy.svyg) + mzr


def mf61(P: MF61Params, alpha, kappa, gamma, fz, p=None) -> MF61Result:
    """Fuerzas/momento combinados. Todas las entradas broadcastables; p [kPa] opcional (=NOMPRES)."""
    alpha, kappa, gamma, fz = jnp.broadcast_arrays(*(jnp.asarray(x) for x in (alpha, kappa, gamma, fz)))
    c = _ctx(P, fz, gamma, p)
    fxp = _fx0(P, c, kappa)
    fyp = _fy0(P, c, alpha)
    a = jnp.tan(alpha)

    # --- Fx combinada
    cxa = P.rCx1
    exa = jnp.minimum(P.rEx1 + P.rEx2 * c.dfz, 1.0)
    bxa = (P.rBx1 + P.rBx3 * c.g2) * jnp.cos(jnp.arctan(P.rBx2 * kappa)) * P.lxa
    shxa = P.rHx1
    gxa = jnp.cos(cxa * _mf_arg(bxa, a + shxa, exa)) / jnp.cos(cxa * _mf_arg(bxa, shxa, exa))
    fx = gxa * fxp.fx0

    # --- Fy combinada
    byk = (P.rBy1 + P.rBy4 * c.g2) * jnp.cos(jnp.arctan(P.rBy2 * (a - P.rBy3))) * P.lyk
    cyk = P.rCy1
    eyk = jnp.minimum(P.rEy1 + P.rEy2 * c.dfz, 1.0)
    shyk = P.rHy1 + P.rHy2 * c.dfz
    dvyk = fyp.muy * c.fz * (P.rVy1 + P.rVy2 * c.dfz + P.rVy3 * c.gs) * jnp.cos(jnp.arctan(P.rVy4 * a))
    svyk = dvyk * jnp.sin(P.rVy5 * jnp.arctan(P.rVy6 * kappa)) * P.lVyk
    gyk = jnp.cos(cyk * _mf_arg(byk, kappa + shyk, eyk)) / jnp.cos(cyk * _mf_arg(byk, shyk, eyk))
    fy = gyk * fyp.fy0 + svyk

    # --- Mz combinado y puro
    t, mzr = _trail_and_residual(P, c, fyp, fxp.kxk, alpha, kappa)
    s = P.R0 * (P.sSz1 + P.sSz2 * fy / c.fz0p + (P.sSz3 + P.sSz4 * c.dfz) * c.gs) * P.ls
    mz = -t * (fy - svyk - fyp.svyg) + mzr + s * fx
    t0, mzr0 = _trail_and_residual(P, c, fyp, 0.0, alpha, 0.0)
    mz0 = -t0 * (fyp.fy0 - fyp.svyg) + mzr0
    return MF61Result(fx, fy, mz, fxp.fx0, fyp.fy0, mz0, t, mzr, fyp.kya, fxp.kxk, fxp.mux, fyp.muy)


evaluate = jax.jit(mf61)


class Pacejka61:
    """Fachada orientada a objeto sobre las funciones puras (misma convencion de signos que mf61)."""

    def __init__(self, params: MF61Params | None = None):
        self.params = params or MF61Params()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Pacejka61":
        return cls(MF61Params.load(path))

    def __call__(self, alpha, kappa, gamma, fz, p=None) -> MF61Result:
        return evaluate(self.params, alpha, kappa, gamma, fz, p)

    def forces(self, alpha, kappa, gamma, fz, p=None):
        r = self(alpha, kappa, gamma, fz, p)
        return r.fx, r.fy, r.mz

    def summary(self, fz, gamma=0.0, p=None):
        return summary(self.params, fz, gamma, p)


# --------------------------------------------------------------------------------------------
# Metricas derivadas, envolvente y comparativa de neumaticos
# --------------------------------------------------------------------------------------------
def summary(P: MF61Params, fz, gamma: float = 0.0, p: float | None = None, *,
            alpha_max: float = 0.30, n: int = 601) -> dict[str, np.ndarray]:
    """Cα [N/deg], μy_peak, μx_peak, Fy_peak [N], Mz_peak [N m], t0 [mm] para un vector de Fz [N]."""
    fz = np.atleast_1d(np.asarray(fz, dtype=float))
    fzj = jnp.asarray(fz)[None, :]
    al = jnp.linspace(-alpha_max, alpha_max, n)[:, None]
    ka = jnp.linspace(-0.5, 0.5, n)[:, None]
    ra = evaluate(P, al, 0.0, gamma, fzj, p)
    rk = evaluate(P, 0.0, ka, gamma, fzj, p)
    r0 = evaluate(P, 0.0, 0.0, gamma, fzj, p)
    safe = np.maximum(fz, 1e-9)
    out = {
        "fz": fz,
        "c_alpha_N_deg": np.asarray(r0.kya) * math.pi / 180.0,
        "mu_y_peak": np.asarray(jnp.max(jnp.abs(ra.fy0), axis=0)) / safe,
        "mu_x_peak": np.asarray(jnp.max(jnp.abs(rk.fx0), axis=0)) / safe,
        "fy_peak_N": np.asarray(jnp.max(jnp.abs(ra.fy0), axis=0)),
        "mz_peak_Nm": np.asarray(jnp.max(jnp.abs(ra.mz0), axis=0)),
        "trail0_mm": np.asarray(r0.t)[0] * 1e3 if np.ndim(r0.t) > 1 else np.asarray(r0.t) * 1e3,
        "c_kappa_N": np.asarray(r0.kxk),
    }
    return {k: np.atleast_1d(np.asarray(v, dtype=float)).reshape(-1) for k, v in out.items()}


METRIC_LABELS = {
    "c_alpha_N_deg": "Cα [N/°]",
    "mu_y_peak": "μy pico [-]",
    "mu_x_peak": "μx pico [-]",
    "fy_peak_N": "Fy pico [N]",
    "mz_peak_Nm": "Mz pico [N·m]",
    "trail0_mm": "t0 traza [mm]",
    "c_kappa_N": "Cκ [N]",
}


def compare_params(P_ref: MF61Params, P_cmp: MF61Params, fz, gamma: float = 0.0,
                   p_ref: float | None = None, p_cmp: float | None = None) -> dict[str, Any]:
    """Δ% = 100 (cmp - ref)/ref para cada metrica y carga. Util para 7" (ref) vs 8" (cmp)."""
    a = summary(P_ref, fz, gamma, p_ref)
    b = summary(P_cmp, fz, gamma, p_cmp)
    rows = []
    for i, f in enumerate(a["fz"]):
        for k in METRIC_LABELS:
            ref, cmp_ = float(a[k][i]), float(b[k][i])
            rows.append({"fz": float(f), "metric": k, "label": METRIC_LABELS[k], "ref": ref, "cmp": cmp_,
                         "delta_pct": 100.0 * (cmp_ - ref) / ref if abs(ref) > 1e-12 else float("nan")})
    mean = {k: float(np.nanmean([r["delta_pct"] for r in rows if r["metric"] == k])) for k in METRIC_LABELS}
    return {"fz": a["fz"].tolist(), "gamma_deg": math.degrees(gamma), "rows": rows, "mean_delta_pct": mean}


def friction_envelope(P: MF61Params, fz: float, gamma: float = 0.0, p: float | None = None,
                      n_alpha: int = 61, n_kappa: int = 61, alpha_max: float = 0.25, kappa_max: float = 0.4):
    """Mallas Fx,Fy sobre (alpha x kappa) para dibujar elipse de friccion / diagrama G-G de neumatico."""
    al = jnp.linspace(-alpha_max, alpha_max, n_alpha)[:, None]
    ka = jnp.linspace(-kappa_max, kappa_max, n_kappa)[None, :]
    r = mf61(P, al, ka, gamma, fz, p)
    return np.asarray(r.fx), np.asarray(r.fy), np.asarray(al[:, 0]), np.asarray(ka[0])


__all__ = [
    "MF61Params", "MF61Result", "Pacejka61", "ParamSpec", "SPECS", "NAMES", "DEFAULTS", "GROUPS",
    "mf61", "evaluate", "lateral_pure_force", "longitudinal_pure_force", "aligning_pure_moment",
    "summary", "compare_params", "friction_envelope", "METRIC_LABELS", "CONVENTION",
]