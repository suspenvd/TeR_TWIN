#!/usr/bin/env python3
"""
TeR-Twin · Terminal Visual de Neumáticos  (apps/desktop/main.py)
=================================================================
Aplicación de escritorio autocontenida (PyQt6 + PyQtGraph + NumPy) para ver y
comparar cualquier número de neumáticos / anchos de llanta (7" vs 8" y más),
con métricas, ajuste contra datos TTC y exportación a JSON/PDF.

Ejecutar:
    pip install PyQt6 pyqtgraph numpy scipy matplotlib
    python apps/desktop/main.py
    python apps/desktop/main.py --params data/processed/mi_neumatico.json --ttc data/raw/ttc/B1654run12.dat
    python apps/desktop/main.py --selftest        # comprobación numérica sin GUI

IMPORTANTE (léelo):
 * El modelo incluido es una Magic Formula estilo 6.1 SIMPLIFICADA (lateral, longitudinal,
   combinado por funciones de peso, momento autoalineante con traza neumática, presión y
   temperatura). Es NumPy puro para que la app funcione sola, sin depender de los
   módulos vacíos del repo (pacejka_52.py / pacejka_61.py).
 * Los neumáticos "demo" son PLACEHOLDERS con valores plausibles; NO son ajustes TTC.
   Carga tus ajustes reales con "Cargar parámetros…" (JSON/NPZ) o déjalos en
   data/processed/pacejka_*.json|npz y se cargan al arrancar.
 * La temperatura NO forma parte de la Magic Formula: se aplica un factor empírico
   parabólico sobre el coeficiente de fricción (parámetros Topt y pT_k).
 * Convención de signos de la app: Fy > 0 para alpha > 0, Mz = -t * Fy, y camber
   negativo ayuda cuando Fy > 0 (empuje de camber = -Fz*pVy3*gamma). Al cargar TTC (SAE)
   se invierten Fy y Mz si la casilla está marcada.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt6")

import numpy as np

DEG = math.pi / 180.0
PSI_PER_KPA = 0.1450377
APP_VERSION = "1.0"
MODEL_NOTE = "Magic Formula estilo 6.1 simplificada (NumPy), temperatura empirica"

# =============================================================================
#  1. MODELO DE NEUMATICO
# =============================================================================
BASE_PARAMS: dict[str, float] = dict(
    Fz0=1000.0, P0=12.0, R0=0.2032, Topt=70.0, pT_k=0.12,
    # lateral puro
    pCy1=1.45, pDy1=2.55, pDy2=-0.35, pDy3=9.0,
    pEy1=-0.6, pEy2=-0.2, pEy3=0.2,
    pKy1=52.0, pKy2=1.7, pKy3=0.3,
    pHy1=0.0, pHy2=0.0, pVy1=0.0, pVy2=0.0, pVy3=2.3,
    ppy1=0.15, ppm3=0.0, ppm4=-0.6,
    # longitudinal puro
    pCx1=1.6, pDx1=2.7, pDx2=-0.35, pDx3=2.0,
    pEx1=-0.1, pEx2=0.0, pEx3=0.0,
    pKx1=65.0, pKx2=5.0, pKx3=-0.3,
    # combinado (funciones de peso)
    rBx1=13.0, rBx2=10.0, rCx1=1.0, rBy1=10.0, rBy2=6.0, rBy3=0.0, rCy1=1.0,
    # momento autoalineante
    qBz1=14.0, qBz2=-1.0, qBz5=0.5, qCz1=1.3, qDz1=0.17, qDz2=-0.03,
    qDz3=0.0, qEz1=-1.0, qEz2=0.0, qDz8=0.006, qBz9=8.0,
)


def _fz(p, fz):
    return np.maximum(np.asarray(fz, dtype=float), 1.0)


def _factors(p, fz, P, T):
    dfz = (fz - p["Fz0"]) / p["Fz0"]
    dpi = (np.asarray(P, dtype=float) - p["P0"]) / p["P0"]
    fP = 1.0 + p["ppm3"] * dpi + p["ppm4"] * dpi ** 2
    x = (np.asarray(T, dtype=float) - p["Topt"]) / 50.0
    fT = np.clip(1.0 - p["pT_k"] * x ** 2, 0.5, 1.2)
    return dfz, dpi, fP, fT


def pure_fy(p, alpha, fz, gamma, P, T):
    """Fy puro [N]. alpha, gamma en rad; P en psi; T en C. Broadcast NumPy."""
    fz = _fz(p, fz)
    gamma = np.asarray(gamma, dtype=float)
    dfz, dpi, fP, fT = _factors(p, fz, P, T)
    mu = (p["pDy1"] + p["pDy2"] * dfz) * (1.0 - p["pDy3"] * gamma ** 2) * fP * fT
    D = mu * fz
    Ky = (p["pKy1"] * p["Fz0"] * np.sin(2.0 * np.arctan(fz / (p["pKy2"] * p["Fz0"])))
          * (1.0 - p["pKy3"] * np.abs(gamma)) * (1.0 + p["ppy1"] * dpi) * (0.5 + 0.5 * fT))
    Cy = p["pCy1"]
    By = Ky / (Cy * D)
    ay = np.asarray(alpha, dtype=float) + p["pHy1"] + p["pHy2"] * dfz
    SVy = fz * (p["pVy1"] + p["pVy2"] * dfz) - fz * p["pVy3"] * gamma
    E = np.minimum((p["pEy1"] + p["pEy2"] * dfz) * (1.0 - p["pEy3"] * np.sign(ay)), 1.0)
    x = By * ay
    return D * np.sin(Cy * np.arctan(x - E * (x - np.arctan(x)))) + SVy


def pure_fx(p, kappa, fz, gamma, P, T):
    """Fx puro [N]."""
    fz = _fz(p, fz)
    gamma = np.asarray(gamma, dtype=float)
    dfz, dpi, fP, fT = _factors(p, fz, P, T)
    mu = (p["pDx1"] + p["pDx2"] * dfz) * (1.0 - p["pDx3"] * gamma ** 2) * fP * fT
    D = mu * fz
    Cx = p["pCx1"]
    Kx = fz * (p["pKx1"] + p["pKx2"] * dfz) * np.exp(p["pKx3"] * dfz) * (0.5 + 0.5 * fT)
    Bx = Kx / (Cx * D)
    k = np.asarray(kappa, dtype=float)
    E = np.minimum((p["pEx1"] + p["pEx2"] * dfz) * (1.0 - p["pEx3"] * np.sign(k)), 1.0)
    x = Bx * k
    return D * np.sin(Cx * np.arctan(x - E * (x - np.arctan(x))))


def trail(p, alpha, fz, gamma):
    """Traza neumatica t [m]."""
    fz = _fz(p, fz)
    gamma = np.asarray(gamma, dtype=float)
    dfz = (fz - p["Fz0"]) / p["Fz0"]
    Bt = (p["qBz1"] + p["qBz2"] * dfz) * (1.0 + p["qBz5"] * np.abs(gamma))
    Ct = p["qCz1"]
    Dt = fz * (p["R0"] / p["Fz0"]) * (p["qDz1"] + p["qDz2"] * dfz) * (1.0 + p["qDz3"] * gamma)
    Et = np.minimum(p["qEz1"] + p["qEz2"] * dfz, 1.0)
    a = np.asarray(alpha, dtype=float)
    x = Bt * a
    return Dt * np.cos(Ct * np.arctan(x - Et * (x - np.arctan(x)))) * np.cos(a)


def mz_total(p, alpha, fy, fz, gamma):
    """Mz [N m] = -t*Fy + Mz residual por camber."""
    fz = _fz(p, fz)
    a = np.asarray(alpha, dtype=float)
    mzr = fz * p["R0"] * p["qDz8"] * np.asarray(gamma, dtype=float) \
        * np.cos(np.arctan(p["qBz9"] * a)) * np.cos(a)
    return -trail(p, a, fz, gamma) * fy + mzr


def mf_combined(p, alpha, kappa, fz, gamma, P, T):
    """(Fx, Fy, Mz) con deslizamiento combinado (funciones de peso estilo Pacejka)."""
    a = np.asarray(alpha, dtype=float)
    k = np.asarray(kappa, dtype=float)
    fx0 = pure_fx(p, k, fz, gamma, P, T)
    fy0 = pure_fy(p, a, fz, gamma, P, T)
    Bxa = p["rBx1"] * np.cos(np.arctan(p["rBx2"] * k))
    Gxa = np.cos(p["rCx1"] * np.arctan(Bxa * a))
    Byk = p["rBy1"] * np.cos(np.arctan(p["rBy2"] * (a - p["rBy3"])))
    Gyk = np.cos(p["rCy1"] * np.arctan(Byk * k))
    fx, fy = Gxa * fx0, Gyk * fy0
    return fx, fy, mz_total(p, a, fy, fz, gamma)


def peak_fy(p, fz, gamma, P, T, n=361, amax_deg=25.0):
    """(Fy pico lado +, alpha de pico [rad]). Vectorizado sobre fz,gamma,P,T."""
    al = np.linspace(0.0, amax_deg * DEG, n)
    args = [np.asarray(v, dtype=float)[..., None] for v in (fz, gamma, P, T)]
    fy = pure_fy(p, al, args[0], args[1], args[2], args[3])
    return fy.max(-1), al[fy.argmax(-1)]


def peak_fx(p, fz, gamma, P, T, n=361, kmax=0.5):
    kk = np.linspace(0.0, kmax, n)
    args = [np.asarray(v, dtype=float)[..., None] for v in (fz, gamma, P, T)]
    fx = pure_fx(p, kk, args[0], args[1], args[2], args[3])
    return fx.max(-1), kk[fx.argmax(-1)]


# --- biblioteca demo --------------------------------------------------------
def _variant(base: dict, **kw) -> dict:
    d = dict(base)
    d.update(kw)
    return d


DEMO_LIBRARY: dict[str, dict] = {
    "Demo R20 16.0x7.5-10": _variant(BASE_PARAMS),
    "Demo C2000 16.0x7.5-10": _variant(BASE_PARAMS, pDy1=2.40, pDx1=2.55, pKy1=47.0,
                                       pKx1=60.0, Topt=62.0, pT_k=0.16),
    "Demo LC0 18.0x7.5-10": _variant(BASE_PARAMS, R0=0.2286, pDy1=2.70, pDx1=2.85, pKy1=56.0,
                                     pKx1=70.0, Topt=78.0, pT_k=0.10, Fz0=1100.0),
}


def rim_variant(p: dict, rim_in: float) -> dict:
    """Efecto ORIENTATIVO del ancho de llanta respecto a 7": placeholder, no es dato TTC."""
    s = rim_in - 7.0
    q = dict(p)
    q["pKy1"] *= 1 + 0.06 * s
    q["pKx1"] *= 1 + 0.03 * s
    q["pDy1"] *= 1 + 0.012 * s
    q["pDx1"] *= 1 + 0.008 * s
    q["pDy3"] *= 1 - 0.15 * s
    q["qDz1"] *= 1 + 0.04 * s
    q["qBz1"] *= 1 - 0.03 * s
    return q


@dataclass
class TireEntry:
    name: str
    rim: float
    params: dict
    source: str = "demo"

    @property
    def label(self) -> str:
        return f'{self.name} · {self.rim:.1f}"'


@dataclass
class OpState:
    fz: float = 1000.0
    alpha_deg: float = 4.0
    gamma_deg: float = 0.0
    p: float = 12.0
    t: float = 70.0
    kappa: float = 0.0


def demo_entries() -> list[TireEntry]:
    out = []
    for name, base in DEMO_LIBRARY.items():
        for rim in (7.0, 8.0):
            out.append(TireEntry(name, rim, rim_variant(base, rim), "demo"))
    return out


# --- carga de parametros ajustados ------------------------------------------
def _merge_params(over: dict) -> tuple[dict, int]:
    p = dict(BASE_PARAMS)
    n = 0
    for k, v in over.items():
        if k in p:
            p[k] = float(v)
            n += 1
    return p, n


def load_param_file(path: str) -> list[TireEntry]:
    path_l = str(path).lower()
    entries: list[TireEntry] = []
    if path_l.endswith(".npz"):
        z = np.load(path, allow_pickle=True)
        name = str(z["name"]) if "name" in z else Path(path).stem
        rim = float(z["rim_in"]) if "rim_in" in z else 7.0
        p, n = _merge_params({k: z[k] for k in z.files if k in BASE_PARAMS})
        if n == 0:
            raise ValueError("El NPZ no contiene ningun parametro reconocido")
        return [TireEntry(name, rim, p, "file")]
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = data["tires"] if isinstance(data, dict) and "tires" in data else \
        (data if isinstance(data, list) else [data])
    for it in items:
        name = it.get("name", Path(path).stem)
        if "rims" in it:  # {"rims": {"7.0": {...}, "8.0": {...}}}
            for rim_s, over in it["rims"].items():
                p, n = _merge_params(over)
                if n:
                    entries.append(TireEntry(name, float(rim_s), p, "file"))
        else:
            p, n = _merge_params(it.get("params", it))
            if n:
                entries.append(TireEntry(name, float(it.get("rim_in", 7.0)), p, "file"))
    if not entries:
        raise ValueError("No se encontro ningun parametro reconocido en el JSON")
    return entries


# =============================================================================
#  2. METRICAS
# =============================================================================
METRIC_DEFS = [
    ("c_alpha_deg", "Rigidez de deriva Cα (N/°)"),
    ("c_alpha_norm", "Cα / Fz (1/rad)"),
    ("fy_peak", "Fy pico (N)"),
    ("alpha_peak", "α de pico (°)"),
    ("mu_y", "μy pico"),
    ("fy0", "Fy en α=0 (empuje camber) (N)"),
    ("camber_stiff", "Rigidez de camber dFy/dγ (N/°)"),
    ("gamma_opt", "Camber óptimo para Fy pico (°)"),
    ("gain", "Ganancia de Fy pico con camber óptimo (%)"),
    ("dpeak_dgamma", "Gradiente dFy_pico/dγ en γ actual (N/°)"),
    ("trail0", "Traza neumática t0 (mm)"),
    ("mz_peak", "|Mz| pico (N·m)"),
    ("alpha_t0", "α donde la traza se anula (°)"),
    ("fx_peak", "Fx pico (N)"),
    ("kappa_peak", "κ de pico"),
    ("mu_x", "μx pico"),
    ("c_kappa", "Rigidez longitudinal Cκ (N/unid. de slip)"),
    ("dmu_dfz", "Sensibilidad dμy/dFz (1/kN, 300→1500 N)"),
    ("loss", "Pérdida por deslizamiento |Fx·κ|+|Fy·tanα| (J/m)"),
]


def compute_metrics(p: dict, st: OpState) -> dict[str, float]:
    fz, g, P, T = st.fz, st.gamma_deg * DEG, st.p, st.t
    m: dict[str, float] = {}
    pk, apk = peak_fy(p, fz, g, P, T)
    m["fy_peak"], m["alpha_peak"], m["mu_y"] = float(pk), float(apk / DEG), float(pk / fz)
    h = 1e-4
    ca = (pure_fy(p, h, fz, g, P, T) - pure_fy(p, -h, fz, g, P, T)) / (2 * h)
    m["c_alpha_deg"], m["c_alpha_norm"] = float(ca * DEG), float(ca / fz)
    dg = 0.1 * DEG
    m["camber_stiff"] = float((pure_fy(p, 0.0, fz, g + dg, P, T)
                               - pure_fy(p, 0.0, fz, g - dg, P, T)) / (2 * dg) * DEG)
    m["fy0"] = float(pure_fy(p, 0.0, fz, g, P, T))
    gs = np.linspace(-5, 5, 101) * DEG
    pks, _ = peak_fy(p, fz, gs, P, T)
    pk0, _ = peak_fy(p, fz, 0.0, P, T)
    i = int(np.argmax(pks))
    m["gamma_opt"], m["gain"] = float(gs[i] / DEG), float((pks[i] / pk0 - 1) * 100)
    pp, _ = peak_fy(p, fz, g + dg, P, T)
    pm, _ = peak_fy(p, fz, g - dg, P, T)
    m["dpeak_dgamma"] = float((pp - pm) / (2 * dg) * DEG)
    m["trail0"] = float(trail(p, 0.0, fz, g) * 1000)
    al = np.linspace(0, 25 * DEG, 2501)
    fys = pure_fy(p, al, fz, g, P, T)
    m["mz_peak"] = float(np.max(np.abs(mz_total(p, al, fys, fz, g))))
    idx = np.where(trail(p, al, fz, g) <= 0)[0]
    m["alpha_t0"] = float(al[idx[0]] / DEG) if len(idx) else float("nan")
    fxp, kp = peak_fx(p, fz, g, P, T)
    m["fx_peak"], m["kappa_peak"], m["mu_x"] = float(fxp), float(kp), float(fxp / fz)
    m["c_kappa"] = float((pure_fx(p, h, fz, g, P, T) - pure_fx(p, -h, fz, g, P, T)) / (2 * h))
    lo, _ = peak_fy(p, 300.0, g, P, T)
    hi, _ = peak_fy(p, 1500.0, g, P, T)
    m["dmu_dfz"] = float((hi / 1500.0 - lo / 300.0) / 1.2)
    fx, fy, _ = mf_combined(p, st.alpha_deg * DEG, st.kappa, fz, g, P, T)
    m["loss"] = float(abs(fx * st.kappa) + abs(fy * math.tan(st.alpha_deg * DEG)))
    return m


# =============================================================================
#  3. DATOS TTC
# =============================================================================
ALIASES = {
    "alpha": ["SA", "ALPHA", "SLIPANGLE", "SLIP_ANGLE"],
    "kappa": ["SL", "KAPPA", "SLIPRATIO", "SLIP_RATIO"],
    "gamma": ["IA", "GAMMA", "CAMBER", "INCLINATION"],
    "fz": ["FZ"], "fx": ["FX"], "fy": ["FY"], "mz": ["MZ"],
    "p": ["P", "PRESSURE"],
}
T_CHANNELS = ["TSTC", "TSTI", "TSTO"]


@dataclass
class TTCData:
    name: str
    alpha_deg: np.ndarray
    kappa: np.ndarray
    gamma_deg: np.ndarray
    fz: np.ndarray
    p: np.ndarray
    T: np.ndarray
    fx: np.ndarray
    fy: np.ndarray
    mz: np.ndarray
    has_kappa: bool
    has_p: bool
    has_T: bool
    channels: list

    def signed(self, flip: bool):
        s = -1.0 if flip else 1.0
        return self.fx, s * self.fy, s * self.mz


def _read_table(path: str) -> dict:
    lines = Path(path).read_text(errors="ignore").splitlines()
    sep = re.compile(r"[,\t; ]+")
    known = {a for v in ALIASES.values() for a in v} | set(T_CHANNELS)
    hdr = None
    for i, ln in enumerate(lines[:300]):
        toks = [t for t in sep.split(ln.strip()) if t]
        if sum(t.upper() in known for t in toks) >= 3:
            hdr = i
            break
    if hdr is None:
        raise ValueError("No se encontro la linea de cabecera con nombres de canal (FZ, SA, FY...)")
    names = [t.upper() for t in sep.split(lines[hdr].strip()) if t]
    rows = []
    for ln in lines[hdr + 1:]:
        toks = [t for t in sep.split(ln.strip()) if t]
        if len(toks) != len(names):
            continue
        try:
            rows.append([float(t) for t in toks])
        except ValueError:
            continue
    if not rows:
        raise ValueError("Cabecera encontrada pero sin filas numericas validas")
    arr = np.array(rows)
    return {n: arr[:, i] for i, n in enumerate(names)}


def load_ttc(path: str) -> TTCData:
    if str(path).lower().endswith(".mat"):
        from scipy.io import loadmat
        cols = {}
        for k, v in loadmat(path).items():
            if k.startswith("__"):
                continue
            try:
                a = np.asarray(v, dtype=float).ravel()
            except Exception:
                continue
            if a.size > 10:
                cols[k.upper()] = a
    else:
        cols = _read_table(path)

    def pick(key):
        for a in ALIASES[key]:
            if a in cols:
                return cols[a]
        return None

    fz = pick("fz")
    if fz is None:
        raise ValueError("El fichero no tiene canal FZ")
    N = len(fz)

    def arr(key, default=np.nan):
        a = pick(key)
        return a[:N].astype(float) if a is not None and len(a) >= N else np.full(N, default)

    tt = [cols[c][:N] for c in T_CHANNELS if c in cols and len(cols[c]) >= N]
    T = np.mean(tt, axis=0) if tt else np.full(N, np.nan)
    p = arr("p")
    if np.isfinite(p).any() and np.nanmedian(p) > 50:  # kPa -> psi
        p = p * PSI_PER_KPA
    kappa = arr("kappa")
    has_k = bool(np.isfinite(kappa).any())
    return TTCData(
        name=Path(path).name, alpha_deg=arr("alpha"),
        kappa=kappa if has_k else np.zeros(N), gamma_deg=arr("gamma", 0.0),
        fz=np.abs(fz.astype(float)), p=p, T=T, fx=arr("fx"), fy=arr("fy"), mz=arr("mz"),
        has_kappa=has_k, has_p=bool(np.isfinite(p).any()), has_T=bool(np.isfinite(T).any()),
        channels=sorted(cols.keys()))


@dataclass
class Tol:
    fz: float = 150.0
    gamma: float = 0.75
    p: float = 1.0


def ttc_mask(t: TTCData, st: OpState, tol: Tol, mode: str) -> np.ndarray:
    m = np.abs(t.fz - st.fz) <= tol.fz
    m &= np.abs(t.gamma_deg - st.gamma_deg) <= tol.gamma
    if t.has_p:
        m &= np.abs(t.p - st.p) <= tol.p
    if mode == "lat" and t.has_kappa:
        m &= np.abs(t.kappa) <= 0.03
    if mode == "long":
        m &= np.abs(t.alpha_deg) <= 1.0
    return m


def _stats(y, yh):
    err = yh - y
    rng = float(np.max(y) - np.min(y)) or 1.0
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1 - np.sum(err ** 2) / ss_tot) if ss_tot > 0 else float("nan")
    return dict(rmse=float(np.sqrt(np.mean(err ** 2))), nmae=float(np.mean(np.abs(err)) / rng * 100),
                r2=r2, n=int(len(y)))


def fit_metrics(entry: TireEntry, t: TTCData, st: OpState, tol: Tol, flip: bool):
    """RMSE, NMAE (MAE / rango medido, %) y R2 evaluando el modelo en las condiciones reales de cada punto."""
    fx_m, fy_m, mz_m = t.signed(flip)
    out = {}
    for key, mode, meas in (("Fy", "lat", fy_m), ("Mz", "lat", mz_m), ("Fx", "long", fx_m)):
        m = ttc_mask(t, st, tol, mode) & np.isfinite(meas) & np.isfinite(t.alpha_deg)
        if m.sum() < 5:
            out[key] = None
            continue
        P = np.where(np.isfinite(t.p[m]), t.p[m], st.p)
        T = np.where(np.isfinite(t.T[m]), t.T[m], st.t)
        fx, fy, mz = mf_combined(entry.params, t.alpha_deg[m] * DEG, t.kappa[m], t.fz[m],
                                 t.gamma_deg[m] * DEG, P, T)
        out[key] = _stats(meas[m], {"Fy": fy, "Mz": mz, "Fx": fx}[key])
    return out


# =============================================================================
#  4. EXPORTACION (JSON / PDF)
# =============================================================================
def export_json(path, entries, st, ttc, tol, flip):
    doc = {
        "app": f"TeR-Twin Tire Terminal {APP_VERSION}", "fecha": datetime.datetime.now().isoformat(timespec="seconds"),
        "modelo": MODEL_NOTE,
        "estado_operacional": dict(Fz_N=st.fz, alpha_deg=st.alpha_deg, gamma_deg=st.gamma_deg,
                                   P_psi=st.p, T_tread_C=st.t, kappa=st.kappa),
        "definiciones": {"NMAE": "MAE / (max-min) medido, en %", "perdida_deslizamiento": "|Fx*kappa|+|Fy*tan(alpha)| en J/m"},
        "neumaticos": [],
    }
    for e in entries:
        item = {"nombre": e.name, "llanta_in": e.rim, "origen": e.source,
                "metricas": compute_metrics(e.params, st), "parametros": e.params}
        if ttc is not None:
            item["ajuste_ttc"] = {"fichero": ttc.name, "ventana": vars(tol), **fit_metrics(e, ttc, st, tol, flip)}
        doc["neumaticos"].append(item)
    Path(path).write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")


def export_pdf(path, styled, st, ttc, tol, flip, active=0):
    """styled: lista de (TireEntry, color_hex, linestyle_mpl)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    A = np.linspace(-15, 15, 361)
    ents = [s[0] for s in styled]
    with PdfPages(path) as pdf:
        # portada + tabla
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis("off")
        ax.set_title(f"TeR-Twin · Informe de neumáticos  ({datetime.date.today()})\n"
                     f"Fz={st.fz:.0f} N  α={st.alpha_deg:.1f}°  γ={st.gamma_deg:.1f}°  P={st.p:.1f} psi  "
                     f"T={st.t:.0f} °C  κ={st.kappa:.2f}", fontsize=11)
        if ents:
            ms = [compute_metrics(e.params, st) for e in ents]
            cells = [[f"{m[k]:.3g}" for m in ms] for k, _ in METRIC_DEFS]
            tb = ax.table(cellText=cells, rowLabels=[l for _, l in METRIC_DEFS],
                          colLabels=[e.label for e in ents], loc="center")
            tb.auto_set_font_size(False)
            tb.set_fontsize(6.5)
            tb.scale(1, 1.25)
        fig.text(0.02, 0.02, f"Modelo: {MODEL_NOTE}. Parámetros 'demo' no son ajustes TTC.", fontsize=7)
        pdf.savefig(fig)
        plt.close(fig)

        def page(title, nrows, ncols, fn):
            fig, axs = plt.subplots(nrows, ncols, figsize=(11.69, 8.27), squeeze=False)
            fig.suptitle(title)
            fn(axs.ravel())
            for a in axs.ravel():
                a.grid(alpha=0.3)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        def p_fy_mz(axs):
            for e, c, ls in styled:
                a = A * DEG
                fy = pure_fy(e.params, a, st.fz, st.gamma_deg * DEG, st.p, st.t)
                axs[0].plot(A, fy, c=c, ls=ls, label=e.label)
                axs[1].plot(A, mz_total(e.params, a, fy, st.fz, st.gamma_deg * DEG), c=c, ls=ls)
                axs[2].plot(A, trail(e.params, a, st.fz, st.gamma_deg * DEG) * 1000, c=c, ls=ls)
            if ttc is not None:
                m = ttc_mask(ttc, st, tol, "lat")
                _, fyt, mzt = ttc.signed(flip)
                axs[0].scatter(ttc.alpha_deg[m], fyt[m], s=3, c="gray", alpha=0.4, label="TTC")
                axs[1].scatter(ttc.alpha_deg[m], mzt[m], s=3, c="gray", alpha=0.4)
            axs[0].set(xlabel="α (°)", ylabel="Fy (N)", title="Fy puro")
            axs[1].set(xlabel="α (°)", ylabel="Mz (N·m)", title="Mz")
            axs[2].set(xlabel="α (°)", ylabel="t (mm)", title="Traza neumática")
            axs[0].legend(fontsize=6)
            axs[3].axis("off")
        page("Fuerza lateral, momento autoalineante y traza", 2, 2, p_fy_mz)

        def p_ell(axs):
            for e, c, ls in styled:
                a = np.linspace(-15, 15, 121) * DEG
                k = np.linspace(-0.4, 0.4, 161)
                AA, KK = np.meshgrid(a, k, indexing="ij")
                fx, fy, _ = mf_combined(e.params, AA, KK, st.fz, st.gamma_deg * DEG, st.p, st.t)
                ang = np.arctan2(fx, fy).ravel()
                r = np.hypot(fx, fy).ravel()
                bins = np.linspace(-np.pi, np.pi, 73)
                ix = np.clip(np.digitize(ang, bins) - 1, 0, 71)
                env = np.zeros(72)
                np.maximum.at(env, ix, r)
                cen = 0.5 * (bins[:-1] + bins[1:])
                ok = env > 0
                axs[0].plot(np.append(env[ok] * np.cos(cen[ok]), (env[ok] * np.cos(cen[ok]))[0]),
                            np.append(env[ok] * np.sin(cen[ok]), (env[ok] * np.sin(cen[ok]))[0]),
                            c=c, ls=ls, label=e.label)
                kk = np.linspace(-0.4, 0.4, 321)
                axs[1].plot(kk, pure_fx(e.params, kk, st.fz, st.gamma_deg * DEG, st.p, st.t), c=c, ls=ls)
            axs[0].set(xlabel="Fy (N)", ylabel="Fx (N)", title="Envolvente de fricción combinada", aspect="equal")
            axs[1].set(xlabel="κ", ylabel="Fx (N)", title="Fx puro")
            axs[0].legend(fontsize=6)
        page("Elipse de fricción y Fx–κ", 1, 2, p_ell)

        def p_load(axs):
            fzg = np.linspace(100, 2000, 77)
            for e, c, ls in styled:
                g = st.gamma_deg * DEG
                pk, _ = peak_fy(e.params, fzg, g, st.p, st.t)
                px, _ = peak_fx(e.params, fzg, g, st.p, st.t)
                h = 1e-4
                ca = (pure_fy(e.params, h, fzg, g, st.p, st.t) - pure_fy(e.params, -h, fzg, g, st.p, st.t)) / (2 * h) * DEG
                axs[0].plot(fzg, pk / fzg, c=c, ls=ls, label=e.label)
                axs[1].plot(fzg, px / fzg, c=c, ls=ls)
                axs[2].plot(fzg, ca, c=c, ls=ls)
            axs[0].set(xlabel="Fz (N)", ylabel="μy pico", title="Sensibilidad a la carga (lateral)")
            axs[1].set(xlabel="Fz (N)", ylabel="μx pico", title="Sensibilidad a la carga (longitudinal)")
            axs[2].set(xlabel="Fz (N)", ylabel="Cα (N/°)", title="Rigidez de deriva")
            axs[0].legend(fontsize=6)
            axs[3].axis("off")
        page("Sensibilidad a la carga", 2, 2, p_load)

        def p_th(axs):
            Tg = np.linspace(20, 110, 46)
            Pg = np.linspace(8, 16, 33)
            g = st.gamma_deg * DEG
            for e, c, ls in styled:
                pk, _ = peak_fy(e.params, st.fz, g, st.p, Tg)
                axs[0].plot(Tg, pk / st.fz, c=c, ls=ls, label=e.label)
                pk, _ = peak_fy(e.params, st.fz, g, Pg, st.t)
                axs[1].plot(Pg, pk / st.fz, c=c, ls=ls)
            axs[0].set(xlabel="T banda (°C)", ylabel="μy pico", title=f"μ vs T (P={st.p:.1f} psi)")
            axs[1].set(xlabel="P (psi)", ylabel="μy pico", title=f"μ vs P (T={st.t:.0f} °C)")
            axs[0].legend(fontsize=6)
            if ents:
                e = ents[min(active, len(ents) - 1)]
                TT, PP = np.meshgrid(Tg, Pg, indexing="ij")
                pk, _ = peak_fy(e.params, st.fz, g, PP, TT)
                im = axs[2].imshow((pk / st.fz).T, origin="lower", aspect="auto",
                                   extent=[Tg[0], Tg[-1], Pg[0], Pg[-1]], cmap="viridis")
                axs[2].set(xlabel="T (°C)", ylabel="P (psi)", title=f"μy pico: {e.label}")
                plt.colorbar(im, ax=axs[2])
            axs[3].axis("off")
        page("Efectos térmicos y de presión", 2, 2, p_th)


# =============================================================================
#  5. INTERFAZ
# =============================================================================
def build_gui():
    from PyQt6 import QtCore, QtGui, QtWidgets
    import pyqtgraph as pg

    Qt = QtCore.Qt
    pg.setConfigOptions(antialias=True, background="#1b1d23", foreground="#d0d3da")
    PALETTE = ["#ff6b6b", "#4dabf7", "#51cf66", "#fcc419", "#cc5de8", "#ff922b",
               "#22b8cf", "#f06595", "#94d82d", "#868e96"]
    STYLES = [Qt.PenStyle.SolidLine, Qt.PenStyle.DashLine, Qt.PenStyle.DotLine,
              Qt.PenStyle.DashDotLine, Qt.PenStyle.DashDotDotLine]
    MPL_LS = {Qt.PenStyle.SolidLine: "-", Qt.PenStyle.DashLine: "--", Qt.PenStyle.DotLine: ":",
              Qt.PenStyle.DashDotLine: "-.", Qt.PenStyle.DashDotDotLine: "-."}
    DISCRETE_FZ = (300, 600, 1000, 1500)

    def mk(color, style, width, alpha=255):
        c = QtGui.QColor(color)
        c.setAlpha(alpha)
        return pg.mkPen(color=c, width=width, style=style)

    def rim_color(rim):
        return "#ff4d4d" if abs(rim - 7) < 0.05 else "#4da6ff" if abs(rim - 8) < 0.05 else "#5fd35f"

    def rim_style(rim):
        return Qt.PenStyle.SolidLine if abs(rim - 7) < 0.05 else \
            Qt.PenStyle.DashLine if abs(rim - 8) < 0.05 else Qt.PenStyle.DotLine

    class ParamSlider(QtWidgets.QWidget):
        changed = QtCore.pyqtSignal()

        def __init__(self, label, lo, hi, val, step, dec=1, suffix=""):
            super().__init__()
            self.lo, self.step = lo, step
            lay = QtWidgets.QGridLayout(self)
            lay.setContentsMargins(0, 0, 0, 0)
            self.slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
            self.slider.setRange(0, int(round((hi - lo) / step)))
            self.spin = QtWidgets.QDoubleSpinBox()
            self.spin.setRange(lo, hi)
            self.spin.setDecimals(dec)
            self.spin.setSingleStep(step)
            self.spin.setSuffix(suffix)
            lay.addWidget(QtWidgets.QLabel(label), 0, 0)
            lay.addWidget(self.spin, 0, 1)
            lay.addWidget(self.slider, 1, 0, 1, 2)
            self.slider.valueChanged.connect(self._from_slider)
            self.spin.valueChanged.connect(self._from_spin)
            self.setValue(val)

        def _from_slider(self, i):
            self.spin.blockSignals(True)
            self.spin.setValue(self.lo + i * self.step)
            self.spin.blockSignals(False)
            self.changed.emit()

        def _from_spin(self, v):
            self.slider.blockSignals(True)
            self.slider.setValue(int(round((v - self.lo) / self.step)))
            self.slider.blockSignals(False)
            self.changed.emit()

        def value(self):
            return self.spin.value()

        def setValue(self, v):
            self.spin.setValue(v)

    def new_plots(n_cols, n):
        glw = pg.GraphicsLayoutWidget()
        plots = []
        for i in range(n):
            pl = glw.addPlot(row=i // n_cols, col=i % n_cols)
            pl.showGrid(x=True, y=True, alpha=0.25)
            plots.append(pl)
        return glw, plots

    class Window(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("TeR-Twin · Terminal Visual de Neumáticos")
            self.resize(1620, 960)
            self.entries: list[TireEntry] = demo_entries()
            self.ttc: TTCData | None = None
            self.timer = QtCore.QTimer(self)
            self.timer.setSingleShot(True)
            self.timer.setInterval(25)
            self.timer.timeout.connect(self.redraw)
            self._build_ui()
            self.load_default_files()
            self.refresh_list()
            self.schedule()

        # ---------- UI ----------
        def _build_ui(self):
            split = QtWidgets.QSplitter()
            self.setCentralWidget(split)
            # --- panel izquierdo
            left = QtWidgets.QWidget()
            lv = QtWidgets.QVBoxLayout(left)
            warn = QtWidgets.QLabel("⚠ Los neumáticos «Demo» son parámetros de ejemplo, NO ajustes TTC. "
                                    "Carga los tuyos con «Cargar parámetros…».")
            warn.setWordWrap(True)
            warn.setStyleSheet("color:#ffb347;")
            lv.addWidget(warn)

            g1 = QtWidgets.QGroupBox("Neumáticos / llantas a comparar")
            v1 = QtWidgets.QVBoxLayout(g1)
            self.list = QtWidgets.QListWidget()
            self.list.setMinimumHeight(190)
            self.list.itemChanged.connect(lambda *_: self.schedule())
            self.list.currentRowChanged.connect(lambda *_: self.schedule())
            v1.addWidget(self.list)
            row = QtWidgets.QHBoxLayout()
            for txt, fn in (("Todos", lambda: self.set_all(True)), ("Ninguno", lambda: self.set_all(False)),
                            ('Solo 7"', lambda: self.set_rim(7.0)), ('Solo 8"', lambda: self.set_rim(8.0))):
                b = QtWidgets.QPushButton(txt)
                b.clicked.connect(fn)
                row.addWidget(b)
            v1.addLayout(row)
            b = QtWidgets.QPushButton("Cargar parámetros… (JSON / NPZ)")
            b.clicked.connect(self.on_load_params)
            v1.addWidget(b)
            b = QtWidgets.QPushButton("Quitar neumáticos cargados de fichero")
            b.clicked.connect(self.on_remove_files)
            v1.addWidget(b)
            self.color_mode = QtWidgets.QComboBox()
            self.color_mode.addItems(['Color por llanta (7" rojo / 8" azul), estilo por neumático',
                                      'Color por neumático, estilo por llanta',
                                      'Color distinto por cada curva'])
            self.color_mode.currentIndexChanged.connect(lambda *_: (self.refresh_list(), self.schedule()))
            v1.addWidget(self.color_mode)
            self.chk_disc = QtWidgets.QCheckBox("Mostrar cargas discretas (300/600/1000/1500 N)")
            self.chk_disc.setChecked(True)
            self.chk_fam = QtWidgets.QCheckBox("Mostrar familias α/κ en la elipse")
            self.chk_fam.setChecked(True)
            for c in (self.chk_disc, self.chk_fam):
                c.toggled.connect(lambda *_: self.schedule())
                v1.addWidget(c)
            lv.addWidget(g1)

            g2 = QtWidgets.QGroupBox("Estado operacional")
            v2 = QtWidgets.QVBoxLayout(g2)
            self.s_fz = ParamSlider("Carga vertical Fz", 100, 2000, 1000, 10, 0, " N")
            self.s_al = ParamSlider("Ángulo de deriva α", -15, 15, 4, 0.1, 1, " °")
            self.s_ga = ParamSlider("Camber γ", -5, 5, 0, 0.1, 1, " °")
            self.s_p = ParamSlider("Presión P", 8, 16, 12, 0.1, 1, " psi")
            self.s_t = ParamSlider("Temp. banda T", 20, 110, 70, 1, 0, " °C")
            self.s_k = ParamSlider("Slip longitudinal κ", -0.3, 0.3, 0.0, 0.005, 3, "")
            for s in (self.s_fz, self.s_al, self.s_ga, self.s_p, self.s_t, self.s_k):
                s.changed.connect(self.schedule)
                v2.addWidget(s)
            lv.addWidget(g2)

            g3 = QtWidgets.QGroupBox("Datos reales TTC")
            v3 = QtWidgets.QVBoxLayout(g3)
            b = QtWidgets.QPushButton("Cargar fichero TTC… (.dat / .mat / .csv)")
            b.clicked.connect(self.on_load_ttc)
            v3.addWidget(b)
            self.lbl_ttc = QtWidgets.QLabel("Sin fichero cargado")
            self.lbl_ttc.setWordWrap(True)
            v3.addWidget(self.lbl_ttc)
            self.chk_flip = QtWidgets.QCheckBox("TTC en convención SAE (invertir Fy y Mz)")
            self.chk_flip.setChecked(True)
            self.chk_flip.toggled.connect(lambda *_: self.schedule())
            v3.addWidget(self.chk_flip)
            form = QtWidgets.QFormLayout()
            self.tol_fz = QtWidgets.QDoubleSpinBox(); self.tol_fz.setRange(10, 1000); self.tol_fz.setValue(150); self.tol_fz.setSuffix(" N")
            self.tol_ga = QtWidgets.QDoubleSpinBox(); self.tol_ga.setRange(0.1, 5); self.tol_ga.setValue(0.75); self.tol_ga.setSuffix(" °")
            self.tol_p = QtWidgets.QDoubleSpinBox(); self.tol_p.setRange(0.1, 8); self.tol_p.setValue(1.0); self.tol_p.setSuffix(" psi")
            for w, lab in ((self.tol_fz, "Ventana ± Fz"), (self.tol_ga, "Ventana ± γ"), (self.tol_p, "Ventana ± P")):
                w.valueChanged.connect(lambda *_: self.schedule())
                form.addRow(lab, w)
            v3.addLayout(form)
            b = QtWidgets.QPushButton("Quitar datos TTC")
            b.clicked.connect(self.on_clear_ttc)
            v3.addWidget(b)
            lv.addWidget(g3)

            g4 = QtWidgets.QGroupBox("Informe")
            v4 = QtWidgets.QHBoxLayout(g4)
            for txt, fn in (("Exportar JSON…", self.on_export_json), ("Exportar PDF…", self.on_export_pdf)):
                b = QtWidgets.QPushButton(txt)
                b.clicked.connect(fn)
                v4.addWidget(b)
            lv.addWidget(g4)
            lv.addStretch(1)
            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(left)
            scroll.setMinimumWidth(380)
            split.addWidget(scroll)

            # --- pestañas
            self.tabs = QtWidgets.QTabWidget()
            split.addWidget(self.tabs)
            split.setStretchFactor(1, 1)
            self.w_fy, (self.pl_fy,) = new_plots(1, 1)
            self.w_mz, (self.pl_mz, self.pl_tr) = new_plots(1, 2)
            self.w_el, (self.pl_el,) = new_plots(1, 1)
            self.w_fx, (self.pl_fx,) = new_plots(1, 1)
            self.w_ld, (self.pl_ld1, self.pl_ld2, self.pl_ld3) = new_plots(2, 3)
            self.w_th, (self.pl_th1, self.pl_th2, self.pl_th3) = new_plots(2, 3)
            self.table = QtWidgets.QTableWidget()
            self.table.setAlternatingRowColors(True)
            self.draw_fns = [self.draw_fy, self.draw_mz, self.draw_ellipse, self.draw_fx,
                             self.draw_load, self.draw_thermal, self.draw_metrics]
            for w, name in ((self.w_fy, "Fy – α"), (self.w_mz, "Mz – α / traza"), (self.w_el, "Elipse Fx–Fy"),
                            (self.w_fx, "Fx – κ"), (self.w_ld, "Sensibilidad a la carga"),
                            (self.w_th, "Térmico / Presión"), (self.table, "Métricas")):
                self.tabs.addTab(w, name)
            self.tabs.currentChanged.connect(lambda *_: self.redraw())
            self.dirty = [True] * len(self.draw_fns)
            self.statusBar().showMessage("Listo")

        # ---------- estado ----------
        def state(self) -> OpState:
            return OpState(self.s_fz.value(), self.s_al.value(), self.s_ga.value(),
                           self.s_p.value(), self.s_t.value(), self.s_k.value())

        def tol(self) -> Tol:
            return Tol(self.tol_fz.value(), self.tol_ga.value(), self.tol_p.value())

        def checked(self):
            out = []
            for i in range(self.list.count()):
                if self.list.item(i).checkState() == Qt.CheckState.Checked:
                    out.append((i, self.entries[i]))
            return out

        def active_idx(self):
            r = self.list.currentRow()
            ch = [i for i, _ in self.checked()]
            if r in ch:
                return r
            return ch[0] if ch else None

        def compound_index(self, name):
            names = list(dict.fromkeys(e.name for e in self.entries))
            return names.index(name)

        def style(self, idx):
            e = self.entries[idx]
            mode = self.color_mode.currentIndex()
            ci = self.compound_index(e.name)
            if mode == 0:
                return rim_color(e.rim), STYLES[ci % len(STYLES)]
            if mode == 1:
                return PALETTE[ci % len(PALETTE)], rim_style(e.rim)
            return PALETTE[idx % len(PALETTE)], Qt.PenStyle.SolidLine

        def refresh_list(self):
            checks = {self.list.item(i).text(): self.list.item(i).checkState()
                      for i in range(self.list.count())}
            self.list.blockSignals(True)
            self.list.clear()
            for i, e in enumerate(self.entries):
                it = QtWidgets.QListWidgetItem(e.label + ("  [fichero]" if e.source == "file" else ""))
                it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                default = Qt.CheckState.Checked if (e.name == next(iter(DEMO_LIBRARY)) and e.source == "demo") \
                    else Qt.CheckState.Unchecked
                it.setCheckState(checks.get(it.text(), default if not checks else Qt.CheckState.Checked
                                            if e.source == "file" and it.text() not in checks else default))
                it.setForeground(QtGui.QBrush(QtGui.QColor(self.style(i)[0])))
                self.list.addItem(it)
            self.list.blockSignals(False)

        def set_all(self, on):
            self.list.blockSignals(True)
            for i in range(self.list.count()):
                self.list.item(i).setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
            self.list.blockSignals(False)
            self.schedule()

        def set_rim(self, rim):
            self.list.blockSignals(True)
            for i, e in enumerate(self.entries):
                self.list.item(i).setCheckState(Qt.CheckState.Checked if abs(e.rim - rim) < 0.05
                                                else Qt.CheckState.Unchecked)
            self.list.blockSignals(False)
            self.schedule()

        # ---------- ficheros ----------
        def load_default_files(self):
            d = Path(__file__).resolve().parents[2] / "data" / "processed" if len(Path(__file__).resolve().parents) > 2 else None
            if d and d.is_dir():
                for f in sorted(list(d.glob("pacejka*.json")) + list(d.glob("pacejka*.npz"))):
                    try:
                        self.entries += load_param_file(str(f))
                    except Exception as ex:
                        print(f"[aviso] {f.name}: {ex}", file=sys.stderr)

        def add_param_file(self, path):
            new = load_param_file(path)
            self.entries += new
            self.refresh_list()
            for i in range(len(self.entries) - len(new), len(self.entries)):
                self.list.item(i).setCheckState(Qt.CheckState.Checked)
            self.schedule()
            self.statusBar().showMessage(f"Cargados {len(new)} neumático(s) de {Path(path).name}")

        def on_load_params(self):
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Cargar parámetros", "", "Parámetros (*.json *.npz)")
            if path:
                try:
                    self.add_param_file(path)
                except Exception as ex:
                    QtWidgets.QMessageBox.warning(self, "Error", str(ex))

        def on_remove_files(self):
            self.entries = [e for e in self.entries if e.source != "file"]
            self.refresh_list()
            self.schedule()

        def set_ttc(self, path):
            self.ttc = load_ttc(path)
            t = self.ttc
            self.lbl_ttc.setText(f"{t.name}: {len(t.fz)} puntos. Canales: {', '.join(t.channels[:14])}…")
            self.schedule()

        def on_load_ttc(self):
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Cargar TTC", "", "TTC (*.dat *.mat *.csv *.txt)")
            if path:
                try:
                    self.set_ttc(path)
                except Exception as ex:
                    QtWidgets.QMessageBox.warning(self, "Error leyendo TTC", str(ex))

        def on_clear_ttc(self):
            self.ttc = None
            self.lbl_ttc.setText("Sin fichero cargado")
            self.schedule()

        def styled(self):
            return [(e, self.style(i)[0], MPL_LS[self.style(i)[1]]) for i, e in self.checked()]

        def on_export_json(self):
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Exportar JSON", "reports/resumen_neumaticos.json", "JSON (*.json)")
            if path:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                export_json(path, [e for _, e in self.checked()], self.state(), self.ttc, self.tol(), self.chk_flip.isChecked())
                self.statusBar().showMessage(f"Exportado {path}")

        def on_export_pdf(self):
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Exportar PDF", "reports/informe_neumaticos.pdf", "PDF (*.pdf)")
            if path:
                try:
                    Path(path).parent.mkdir(parents=True, exist_ok=True)
                    st = self.styled()
                    ch = [i for i, _ in self.checked()]
                    a = self.active_idx()
                    export_pdf(path, st, self.state(), self.ttc, self.tol(), self.chk_flip.isChecked(),
                               ch.index(a) if a in ch else 0)
                    self.statusBar().showMessage(f"Exportado {path}")
                except Exception as ex:
                    QtWidgets.QMessageBox.warning(self, "Error exportando PDF", str(ex))

        # ---------- dibujo ----------
        def schedule(self):
            self.dirty = [True] * len(self.draw_fns)
            self.timer.start()

        def redraw(self):
            i = self.tabs.currentIndex()
            if self.dirty[i]:
                self.draw_fns[i]()
                self.dirty[i] = False

        def _ttc_scatter(self, pl, x, y, name="TTC"):
            pl.plot(x, y, pen=None, symbol="o", symbolSize=4, symbolBrush=(200, 200, 200, 90),
                    symbolPen=None, name=name)

        def _prep(self, pl, xl, yl, legend=True):
            pl.clear()
            if legend and pl.legend is None:
                pl.addLegend(offset=(8, 8))
            pl.setLabel("bottom", xl)
            pl.setLabel("left", yl)

        def draw_fy(self):
            pl, st = self.pl_fy, self.state()
            self._prep(pl, "α (°)", "Fy (N)")
            A = np.linspace(-15, 15, 361)
            g = st.gamma_deg * DEG
            for idx, e in self.checked():
                col, sty = self.style(idx)
                if self.chk_disc.isChecked():
                    for j, fz in enumerate(DISCRETE_FZ):
                        pl.plot(A, pure_fy(e.params, A * DEG, fz, g, st.p, st.t), pen=mk(col, sty, 1.2, 70 + 45 * j))
                pl.plot(A, pure_fy(e.params, A * DEG, st.fz, g, st.p, st.t), pen=mk(col, sty, 2.8), name=e.label)
                pl.plot([st.alpha_deg], [float(pure_fy(e.params, st.alpha_deg * DEG, st.fz, g, st.p, st.t))],
                        pen=None, symbol="o", symbolSize=9, symbolBrush=col)
            if self.ttc is not None:
                m = ttc_mask(self.ttc, st, self.tol(), "lat")
                _, fy, _ = self.ttc.signed(self.chk_flip.isChecked())
                self._ttc_scatter(pl, self.ttc.alpha_deg[m], fy[m], f"TTC (n={int(m.sum())})")
            pl.setTitle(f"Fy puro · curva gruesa: Fz={st.fz:.0f} N · finas: 300/600/1000/1500 N "
                        f"· γ={st.gamma_deg:.1f}° P={st.p:.1f} psi T={st.t:.0f} °C")

        def draw_mz(self):
            st = self.state()
            self._prep(self.pl_mz, "α (°)", "Mz (N·m)")
            self._prep(self.pl_tr, "α (°)", "Traza neumática t (mm)")
            A = np.linspace(-15, 15, 361)
            g = st.gamma_deg * DEG
            for idx, e in self.checked():
                col, sty = self.style(idx)
                for j, fz in enumerate(DISCRETE_FZ if self.chk_disc.isChecked() else ()):
                    fy = pure_fy(e.params, A * DEG, fz, g, st.p, st.t)
                    self.pl_mz.plot(A, mz_total(e.params, A * DEG, fy, fz, g), pen=mk(col, sty, 1.2, 70 + 45 * j))
                    self.pl_tr.plot(A, trail(e.params, A * DEG, fz, g) * 1000, pen=mk(col, sty, 1.2, 70 + 45 * j))
                fy = pure_fy(e.params, A * DEG, st.fz, g, st.p, st.t)
                self.pl_mz.plot(A, mz_total(e.params, A * DEG, fy, st.fz, g), pen=mk(col, sty, 2.8), name=e.label)
                self.pl_tr.plot(A, trail(e.params, A * DEG, st.fz, g) * 1000, pen=mk(col, sty, 2.8), name=e.label)
            if self.ttc is not None:
                m = ttc_mask(self.ttc, st, self.tol(), "lat")
                _, _, mz = self.ttc.signed(self.chk_flip.isChecked())
                self._ttc_scatter(self.pl_mz, self.ttc.alpha_deg[m], mz[m])
            self.pl_mz.setTitle("Momento autoalineante")
            self.pl_tr.setTitle("Traza neumática (par restaurador = t·Fy)")

        def draw_ellipse(self):
            pl, st = self.pl_el, self.state()
            self._prep(pl, "Fy (N)", "Fx (N)")
            pl.setAspectLocked(True)
            g = st.gamma_deg * DEG
            a = np.linspace(-15, 15, 121) * DEG
            k = np.linspace(-0.4, 0.4, 161)
            AA, KK = np.meshgrid(a, k, indexing="ij")
            for idx, e in self.checked():
                col, sty = self.style(idx)
                fx, fy, _ = mf_combined(e.params, AA, KK, st.fz, g, st.p, st.t)
                ang = np.arctan2(fx, fy).ravel()
                r = np.hypot(fx, fy).ravel()
                bins = np.linspace(-np.pi, np.pi, 73)
                ix = np.clip(np.digitize(ang, bins) - 1, 0, 71)
                env = np.zeros(72)
                np.maximum.at(env, ix, r)
                cen = 0.5 * (bins[:-1] + bins[1:])
                ok = env > 0
                ex, ey = env[ok] * np.cos(cen[ok]), env[ok] * np.sin(cen[ok])
                pl.plot(np.append(ex, ex[0]), np.append(ey, ey[0]), pen=mk(col, sty, 2.8), name=e.label)
                if self.chk_fam.isChecked():
                    kk = np.linspace(-0.4, 0.4, 161)
                    for al in (-12, -8, -5, -2, 2, 5, 8, 12):
                        fx, fy, _ = mf_combined(e.params, al * DEG, kk, st.fz, g, st.p, st.t)
                        pl.plot(fy, fx, pen=mk(col, sty, 1.0, 70))
                    aa = np.linspace(-15, 15, 161) * DEG
                    for kap in (-0.2, -0.1, -0.05, 0.05, 0.1, 0.2):
                        fx, fy, _ = mf_combined(e.params, aa, kap, st.fz, g, st.p, st.t)
                        pl.plot(fy, fx, pen=mk(col, sty, 1.0, 70))
                fx, fy, _ = mf_combined(e.params, st.alpha_deg * DEG, st.kappa, st.fz, g, st.p, st.t)
                pl.plot([float(fy)], [float(fx)], pen=None, symbol="o", symbolSize=10, symbolBrush=col)
            if self.ttc is not None:
                m = ttc_mask(self.ttc, st, self.tol(), "free") if False else \
                    (np.abs(self.ttc.fz - st.fz) <= self.tol().fz) & (np.abs(self.ttc.gamma_deg - st.gamma_deg) <= self.tol().gamma)
                if self.ttc.has_p:
                    m &= np.abs(self.ttc.p - st.p) <= self.tol().p
                fx, fy, _ = self.ttc.signed(self.chk_flip.isChecked())
                m &= np.isfinite(fx) & np.isfinite(fy)
                self._ttc_scatter(pl, fy[m], fx[m], f"TTC (n={int(m.sum())})")
            pl.setTitle("Envolvente de fricción combinada (punto = α, κ actuales) · familias finas: α y κ constantes")

        def draw_fx(self):
            pl, st = self.pl_fx, self.state()
            self._prep(pl, "κ", "Fx (N)")
            kk = np.linspace(-0.4, 0.4, 321)
            g = st.gamma_deg * DEG
            for idx, e in self.checked():
                col, sty = self.style(idx)
                if self.chk_disc.isChecked():
                    for j, fz in enumerate(DISCRETE_FZ):
                        pl.plot(kk, pure_fx(e.params, kk, fz, g, st.p, st.t), pen=mk(col, sty, 1.2, 70 + 45 * j))
                pl.plot(kk, pure_fx(e.params, kk, st.fz, g, st.p, st.t), pen=mk(col, sty, 2.8), name=e.label)
            if self.ttc is not None:
                m = ttc_mask(self.ttc, st, self.tol(), "long")
                fx, _, _ = self.ttc.signed(self.chk_flip.isChecked())
                m &= np.isfinite(fx)
                self._ttc_scatter(pl, self.ttc.kappa[m], fx[m], f"TTC (n={int(m.sum())})")
            pl.setTitle("Fx puro vs κ")

        def draw_load(self):
            st = self.state()
            for pl, yl in ((self.pl_ld1, "μy pico"), (self.pl_ld2, "μx pico"), (self.pl_ld3, "Cα (N/°)")):
                self._prep(pl, "Fz (N)", yl)
            fzg = np.linspace(100, 2000, 77)
            g = st.gamma_deg * DEG
            h = 1e-4
            for idx, e in self.checked():
                col, sty = self.style(idx)
                p = e.params
                pk, _ = peak_fy(p, fzg, g, st.p, st.t)
                px, _ = peak_fx(p, fzg, g, st.p, st.t)
                ca = (pure_fy(p, h, fzg, g, st.p, st.t) - pure_fy(p, -h, fzg, g, st.p, st.t)) / (2 * h) * DEG
                self.pl_ld1.plot(fzg, pk / fzg, pen=mk(col, sty, 2.6), name=e.label)
                self.pl_ld2.plot(fzg, px / fzg, pen=mk(col, sty, 2.6), name=e.label)
                self.pl_ld3.plot(fzg, ca, pen=mk(col, sty, 2.6), name=e.label)
            for pl in (self.pl_ld1, self.pl_ld2, self.pl_ld3):
                pl.addItem(pg.InfiniteLine(st.fz, angle=90, pen=mk("#aaaaaa", Qt.PenStyle.DotLine, 1)))
            self.pl_ld1.setTitle("Sensibilidad a la carga lateral (μy = Fy_pico/Fz)")
            self.pl_ld2.setTitle("Sensibilidad a la carga longitudinal (μx)")
            self.pl_ld3.setTitle("Rigidez de deriva vs Fz")

        def draw_thermal(self):
            st = self.state()
            self._prep(self.pl_th1, "T banda (°C)", "μy pico")
            self._prep(self.pl_th2, "P (psi)", "μy pico")
            self.pl_th3.clear()
            Tg = np.linspace(20, 110, 46)
            Pg = np.linspace(8, 16, 33)
            g = st.gamma_deg * DEG
            act = self.active_idx()
            for idx, e in self.checked():
                col, sty = self.style(idx)
                pk, _ = peak_fy(e.params, st.fz, g, st.p, Tg)
                self.pl_th1.plot(Tg, pk / st.fz, pen=mk(col, sty, 2.8), name=e.label)
                pk, _ = peak_fy(e.params, st.fz, g, Pg, st.t)
                self.pl_th2.plot(Pg, pk / st.fz, pen=mk(col, sty, 2.8), name=e.label)
                if idx == act:
                    for j, pp in enumerate((8, 10, 12, 14, 16)):
                        pk, _ = peak_fy(e.params, st.fz, g, pp, Tg)
                        self.pl_th1.plot(Tg, pk / st.fz, pen=mk(col, sty, 1.0, 60 + 35 * j))
            self.pl_th1.addItem(pg.InfiniteLine(st.t, angle=90, pen=mk("#aaaaaa", Qt.PenStyle.DotLine, 1)))
            self.pl_th2.addItem(pg.InfiniteLine(st.p, angle=90, pen=mk("#aaaaaa", Qt.PenStyle.DotLine, 1)))
            self.pl_th1.setTitle(f"μy pico vs T (P={st.p:.1f} psi; finas: 8–16 psi del neumático activo)")
            self.pl_th2.setTitle(f"μy pico vs P (T={st.t:.0f} °C)")
            if act is not None:
                e = self.entries[act]
                TT, PP = np.meshgrid(Tg, Pg, indexing="ij")
                pk, _ = peak_fy(e.params, st.fz, g, PP, TT)
                Z = pk / st.fz
                img = pg.ImageItem(Z)
                img.setLookupTable(pg.colormap.get("viridis").getLookupTable(nPts=256))
                img.setLevels((float(Z.min()), float(Z.max())))
                img.setRect(QtCore.QRectF(Tg[0], Pg[0], Tg[-1] - Tg[0], Pg[-1] - Pg[0]))
                self.pl_th3.addItem(img)
                i, j = np.unravel_index(int(np.argmax(Z)), Z.shape)
                self.pl_th3.plot([Tg[i]], [Pg[j]], pen=None, symbol="star", symbolSize=14, symbolBrush="w")
                self.pl_th3.plot([st.t], [st.p], pen=None, symbol="o", symbolSize=9, symbolBrush="r")
                self.pl_th3.setLabel("bottom", "T banda (°C)")
                self.pl_th3.setLabel("left", "P (psi)")
                self.pl_th3.setTitle(f"μy pico (T×P) · {e.label} · máx={Z.max():.3f} en {Tg[i]:.0f} °C, "
                                     f"{Pg[j]:.1f} psi (★)")

        def draw_metrics(self):
            st = self.state()
            ch = self.checked()
            cols = [e.label for _, e in ch]
            has_delta = len(ch) >= 2
            if has_delta:
                cols.append("Δ (último − primero)")
            ms = [compute_metrics(e.params, st) for _, e in ch]
            rows = [(lab, [m[k] for m in ms]) for k, lab in METRIC_DEFS]
            if self.ttc is not None and ch:
                fits = [fit_metrics(e, self.ttc, st, self.tol(), self.chk_flip.isChecked()) for _, e in ch]
                for key in ("Fy", "Mz", "Fx"):
                    for sub, lab in (("rmse", "RMSE"), ("nmae", "NMAE (%)"), ("r2", "R²"), ("n", "puntos")):
                        rows.append((f"TTC {key}: {lab}", [f[key][sub] if f[key] else None for f in fits]))

            def fmt(v):
                if v is None or (isinstance(v, float) and not np.isfinite(v)):
                    return "—"
                return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.1f}" if abs(v) >= 100 else f"{v:.3f}"
            self.table.clear()
            self.table.setRowCount(len(rows))
            self.table.setColumnCount(len(cols))
            self.table.setHorizontalHeaderLabels(cols)
            self.table.setVerticalHeaderLabels([r[0] for r in rows])
            for i, (_, vals) in enumerate(rows):
                for j, v in enumerate(vals):
                    self.table.setItem(i, j, QtWidgets.QTableWidgetItem(fmt(v)))
                if has_delta:
                    a, b = vals[0], vals[-1]
                    ok = a is not None and b is not None and np.isfinite(a) and np.isfinite(b)
                    self.table.setItem(i, len(vals), QtWidgets.QTableWidgetItem(fmt(b - a) if ok else "—"))
            self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
            self.table.verticalHeader().setMinimumWidth(330)

    return Window


# =============================================================================
#  6. MAIN
# =============================================================================
def selftest():
    st = OpState()
    ents = demo_entries()
    print(f"{'métrica':58s}" + "".join(f"{e.label[-14:]:>16s}" for e in ents[:2]))
    ms = [compute_metrics(e.params, st) for e in ents[:2]]
    for k, lab in METRIC_DEFS:
        print(f"{lab:58s}" + "".join(f"{m[k]:16.4g}" for m in ms))
    # simetria y gradientes
    p = dict(ents[0].params, pEy3=0.0)  # con pEy3=0 la curva pura debe ser impar en alpha
    a = np.linspace(-0.2, 0.2, 9)
    assert np.allclose(pure_fy(p, a, 800, 0.0, 12, 70), -pure_fy(p, -a, 800, 0.0, 12, 70), atol=1e-6)
    assert ms[1]["c_alpha_deg"] > ms[0]["c_alpha_deg"], "8\" debe tener mayor Cα en el placeholder"
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser(description="TeR-Twin · Terminal Visual de Neumáticos")
    ap.add_argument("--params", nargs="*", help="ficheros JSON/NPZ de parámetros ajustados")
    ap.add_argument("--ttc", help="fichero TTC (.dat/.mat/.csv)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return 0
    from PyQt6 import QtWidgets, QtGui, QtCore
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    pal = QtGui.QPalette()
    for role, col in ((QtGui.QPalette.ColorRole.Window, "#23262e"), (QtGui.QPalette.ColorRole.WindowText, "#d0d3da"),
                      (QtGui.QPalette.ColorRole.Base, "#1b1d23"), (QtGui.QPalette.ColorRole.AlternateBase, "#23262e"),
                      (QtGui.QPalette.ColorRole.Text, "#d0d3da"), (QtGui.QPalette.ColorRole.Button, "#2b2f38"),
                      (QtGui.QPalette.ColorRole.ButtonText, "#d0d3da"), (QtGui.QPalette.ColorRole.Highlight, "#3b82f6"),
                      (QtGui.QPalette.ColorRole.HighlightedText, "#ffffff")):
        pal.setColor(role, QtGui.QColor(col))
    app.setPalette(pal)
    win = build_gui()()
    for f in args.params or []:
        win.add_param_file(f)
    if args.ttc:
        win.set_ttc(args.ttc)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())