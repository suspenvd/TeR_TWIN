#!/usr/bin/env python3
"""Ajuste masivo MF6.1 de ficheros TTC/Calspan .dat (SI).

Flujo: escaneo de cabeceras (Tire_Name, Rim_Width) -> agrupacion -> parseo + conversion de unidades
-> clasificacion de barridos (warmup / transitorio / lateral / longitudinal / combinado / presion)
-> ajuste por etapas con scipy.optimize.least_squares (perdida robusta Huber/Soft-L1, jacobiano JAX)
-> YAML de parametros, NPZ de datos normalizados, JSON de informe e index.json (lo lee tire_terminal).

Suposiciones documentadas
-------------------------
* Cabecera: lineas "Tire_Name <valor>" y "Rim_Width <valor>" (inch por defecto; mm si valor>30 o unidad mm).
  Si faltan: Tire_Name='UNKNOWN', ancho de llanta inferido del nombre de fichero o NaN (grupo propio).
* Columnas canonicas TTC: ET V SA IA SL/SR RL RE P FX FY FZ MX MZ ... Unidades de la linea de unidades
  (deg, km/h, kPa, cm, N, N·m). Sin linea de unidades se asume el estandar TTC SI.
* FZ en TTC es negativa (SAE): se usa |FZ|. Signos de Fy, Mz, Fx se autodetectan por pendiente a
  pequeno deslizamiento para pasar a la convencion del modelo (Fy>0 con +alpha, Mz<0, Fx>0 con +kappa).
  El signo de IA se controla con --ia-sign (por defecto +1, convencion TTC SAE).
* Clasificacion por umbrales de senal (configurables, ver Thresholds) o por manifest YAML:
      files:  {B2356run12.dat: {kind: transient}}
      tires:  {"Hoosier 16.0x7.5-10 R20": {rim_width_in: 7.0}}
  El nombre de fichero con 'warm'/'trans' fuerza warmup/transient.
* Barridos lateral con SL~0, longitudinal con SA~0 (se admite SL ausente = 0).
* Los barridos de presion (>=2 niveles de P) activan los coeficientes pp**; sin variacion de P, de
  carga (<2 niveles) o de caida (<2 niveles) los coeficientes dependientes se congelan en su defecto.
* R0 se infiere del nombre ("16.0x7.5-10" -> OD 16 in -> R0=0.2032 m); FNOMIN = nivel de carga mas
  cercano a la mediana (o --fz-nominal); NOMPRES = mediana de P.
* La validacion usa runs completos reservados (cada 4º run por tipo, si hay >=4); sin ellos, sin val.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import dataclasses
import io
import json
import logging
import math
import multiprocessing as mp
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import yaml

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from ter_twin.models.tires import pacejka_61 as mf  # noqa: E402

LOG = logging.getLogger("fit_all_ttc")
KIND_CODE = {"lateral": 0, "longitudinal": 1, "combined": 2}
STEADY_KINDS = tuple(KIND_CODE)


# --------------------------------------------------------------------------------------------
# Configuracion
# --------------------------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class Thresholds:
    sa_rate_degs: float = 10.0  # p95 |dSA/dt| por encima -> transitorio
    sl_rate: float = 1.0  # p95 |dSL/dt| [1/s]
    warmup_sa_deg: float = 1.5  # amplitud SA p99 por debajo (y SL ~0) -> calentamiento
    warmup_sl: float = 0.02
    lat_sa_deg: float = 2.5
    long_sl: float = 0.03
    long_max_sa_deg: float = 1.0
    pressure_span_kpa: float = 10.0


@dataclasses.dataclass(frozen=True)
class FitConfig:
    input: Path = Path("data/raw/ttc")
    output: Path = Path("data/processed/mf61_fits")
    pattern: str = "*.dat"
    manifest: Path | None = None
    loss: str = "soft_l1"
    f_scale: float = 0.03  # umbral Huber/soft-l1 en unidades de residuo normalizado (3% del pico)
    max_nfev: int = 200
    max_points: int = 40000
    restarts: int = 1
    reg: float = 1e-4
    jobs: int = 1
    min_fz: float = 50.0
    min_speed: float = 5.0
    trim_s: float = 0.0
    ia_sign: float = 1.0
    fz_nominal: float | None = None
    no_combined: bool = False
    tire_filter: str | None = None
    rim_filter: float | None = None
    seed: int = 0
    thresholds: Thresholds = Thresholds()


# --------------------------------------------------------------------------------------------
# Parser TTC
# --------------------------------------------------------------------------------------------
_CANON = {"ET": "et", "V": "v", "SA": "sa", "IA": "ia", "SL": "sl", "SR": "sl", "RL": "rl", "RE": "re",
          "P": "p", "FX": "fx", "FY": "fy", "FZ": "fz", "MX": "mx", "MZ": "mz", "MU": "mu", "N": "n",
          "RST": "rst", "AMBTMP": "amb", "NFX": "nfx", "NFY": "nfy"}

_NUM = re.compile(r"[-+]?\d+(?:[.,]\d+)?")


@dataclasses.dataclass(slots=True)
class TTCRun:
    path: Path
    tire_name: str
    rim_width_in: float
    meta: dict[str, str]
    ch: dict[str, np.ndarray]
    kind: str = "unknown"
    pressure_sweep: bool = False
    desc: dict[str, float] = dataclasses.field(default_factory=dict)


def _parse_rim_width(text: str | None, fallback: str = "") -> float:
    for src in (text, fallback):
        if not src:
            continue
        m = _NUM.search(src)
        if not m:
            continue
        v = float(m.group(0).replace(",", "."))
        low = src.lower()
        if "mm" in low or v > 30:
            return v / 25.4
        if "cm" in low:
            return v / 2.54
        return v
    return float("nan")


def _unit_scale(canon: str, unit: str | None) -> float:
    u = (unit or "").strip().lower().replace("°", "deg")
    if canon in ("sa", "ia"):
        return 1.0 if u.startswith("rad") else math.pi / 180.0
    if canon == "v":
        return {"m/s": 1.0, "mph": 0.44704}.get(u, 1.0 / 3.6)
    if canon == "p":
        return {"psi": 6.894757, "bar": 100.0, "pa": 1e-3}.get(u, 1.0)
    if canon in ("rl", "re"):
        return {"m": 1.0, "mm": 1e-3, "in": 0.0254}.get(u, 0.01)
    if canon == "sl":
        return 0.01 if u == "%" else 1.0
    if canon in ("fx", "fy", "fz"):
        return 4.448222 if u.startswith("lb") else 1.0
    if canon in ("mx", "mz"):
        return 1.355818 if ("ft" in u or "lb" in u) else 1.0
    return 1.0

def parse_ttc(path: Path, *, meta_only: bool = False) -> TTCRun:
    if meta_only:
        with open(path, "r", encoding="latin-1") as fh:
            lines = [fh.readline() for _ in range(50)]
    else:
        lines = path.read_text(encoding="latin-1").splitlines()
    
    meta: dict[str, str] = {}
    hidx = None
    for i, line in enumerate(lines[:40]):
        toks = line.split()
        if len(toks) >= 6 and {"SA", "FY", "FZ"} <= {t.upper() for t in toks}:
            hidx = i
            break
        # Parsear metadatos separados por ';' en la línea 0
        for part in line.split(";"):
            part = part.strip()
            if "=" in part:
                k, v = part.split("=", 1)
                meta[k.strip().lower()] = v.strip()
            elif ":" in part:
                k, v = part.split(":", 1)
                meta[k.strip().lower()] = v.strip()
                
    if hidx is None:
        raise ValueError(f"{path.name}: sin cabecera de columnas (SA/FY/FZ)")
    
    pre = lines[:hidx]
    # Búsqueda robusta de Tire_Name y Rim_Width
    tire = meta.get("tire_name") or meta.get("tire")
    if not tire:
        for ln in pre:
            m = re.search(r"tire[_ ]?name\s*[:=]\s*([^;\r\n]+)", ln, re.I)
            if m:
                tire = m.group(1).strip()
                break
                
    rim_txt = meta.get("rim_width") or meta.get("rim")
    if not rim_txt:
        for ln in pre:
            m = re.search(r"rim[_ ]?width\s*[:=]\s*([^;\r\n]+)", ln, re.I)
            if m:
                rim_txt = m.group(1).strip()
                break

    stem_rim = re.search(r"(\d+(?:\.\d+)?)\s*(?:in|inch|\")", path.stem, re.I)
    rim_val = _parse_rim_width(rim_txt, stem_rim.group(0) if stem_rim else "")
    
    run = TTCRun(path, (tire or "UNKNOWN").strip().strip('"'), rim_val, meta, {})
    if meta_only:
        return run

    # Separación tolerante a tabuladores para nombres y unidades
    hdr = [t.strip() for t in lines[hidx].split("\t") if t.strip()]
    if len(hdr) < 6:
        hdr = lines[hidx].split()
        
    k = hidx + 1
    units = None
    if k < len(lines) and lines[k].split() and not _NUM.fullmatch(lines[k].split()[0]):
        units = [t.strip() for t in lines[k].split("\t") if t.strip()]
        if len(units) != len(hdr):
            units = lines[k].split()
        k += 1
        
    body = [ln for ln in lines[k:] if ln.strip()]
    try:
        arr = np.loadtxt(io.StringIO("\n".join(body)), ndmin=2, comments=None)
    except ValueError:
        rows = []
        for ln in body:
            s = ln.split()
            if len(s) >= len(hdr):
                try:
                    rows.append([float(x) for x in s[: len(hdr)]])
                except ValueError:
                    continue
        arr = np.asarray(rows, dtype=float)
        
    if arr.size == 0:
        raise ValueError(f"{path.name}: sin datos numericos")
        
    ncol = min(arr.shape[1], len(hdr))
    for j in range(ncol):
        canon = _CANON.get(hdr[j].upper())
        if canon:
            unit = units[j] if units and len(units) == len(hdr) else None
            run.ch[canon] = arr[:, j].astype(np.float64) * _unit_scale(canon, unit)
            
    for req in ("sa", "fy", "fz"):
        if req not in run.ch:
            raise ValueError(f"{path.name}: falta canal {req.upper()}")
            
    n = len(run.ch["sa"])
    run.ch["fz"] = np.abs(run.ch["fz"])
    for opt, fill in (("ia", 0.0), ("sl", 0.0), ("fx", np.nan), ("mz", np.nan), ("p", np.nan), ("v", np.nan)):
        run.ch.setdefault(opt, np.full(n, fill))
    return run


# --------------------------------------------------------------------------------------------
# Descripcion y clasificacion
# --------------------------------------------------------------------------------------------
def _levels(x: np.ndarray, step: float, min_count: int = 50) -> np.ndarray:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([])
    q = np.round(x / step) * step
    u, c = np.unique(q, return_counts=True)
    return u[c >= min_count]


def describe(run: TTCRun) -> dict[str, float]:
    ch = run.ch
    # Filtrar solo datos con rodillo en marcha (> 15 km/h ≈ 4.2 m/s) para evitar artefactos en parado
    v = ch.get("v")
    moving = np.isfinite(ch["sa"])
    if v is not None and np.isfinite(v).any():
        moving &= (np.nan_to_num(v, nan=0.0) >= 4.2)
    if moving.sum() < 50:
        moving = np.ones(len(ch["sa"]), dtype=bool)

    sa = ch["sa"][moving]
    sl = ch["sl"][moving]
    et = ch.get("et")
    if et is not None and len(et) == len(ch["sa"]):
        et = et[moving]

    dt = float(np.median(np.diff(et))) if et is not None and len(et) > 2 and np.median(np.diff(et)) > 0 else 0.01
    w = max(3, int(round(0.2 / dt)))
    ker = np.ones(w) / w
    sa_s = np.convolve(np.nan_to_num(sa), ker, mode="same")
    sl_s = np.convolve(np.nan_to_num(sl), ker, mode="same")
    p = ch["p"][moving]
    fz = ch["fz"][moving]

    return {
        "n": float(len(sa)), "dt": dt, "duration_s": dt * len(sa),
        "sa_amp_deg": float(np.degrees(np.nanpercentile(np.abs(sa), 99))),
        "sl_amp": float(np.nanpercentile(np.abs(sl), 99)),
        "sa_rate_degs": float(np.degrees(np.percentile(np.abs(np.gradient(sa_s, dt)), 95))),
        "sl_rate": float(np.percentile(np.abs(np.gradient(sl_s, dt)), 95)),
        "n_fz_levels": float(len(_levels(fz[fz > 50], 75.0))),
        "n_p_levels": float(len(_levels(p, 5.0))),
        "p_span_kpa": float(np.nanmax(p) - np.nanmin(p)) if np.isfinite(p).any() else 0.0,
        "n_ia_levels": float(len(_levels(np.degrees(ch["ia"][moving]), 0.5))),
        "fz_med": float(np.nanmedian(fz)),
        "p_med": float(np.nanmedian(p)) if np.isfinite(p).any() else float("nan"),
        "ia_med_deg": float(np.degrees(np.nanmedian(ch["ia"][moving]))),
        "v_med": float(np.nanmedian(v[moving])) if v is not None and np.isfinite(v).any() else float("nan"),
    }


def classify(run: TTCRun, th: Thresholds, manifest: dict | None = None) -> None:
    run.desc = d = describe(run)
    forced = ((manifest or {}).get("files") or {}).get(run.path.name, {})
    name = run.path.stem.lower()

    if "kind" in forced:
        run.kind = forced["kind"]
    elif "warm" in name:
        run.kind = "warmup"
    # En la Ronda 9, los step-steers transitorios solo barren hasta 6.0 deg de SA
    elif "trans" in name or d["sa_amp_deg"] <= 8.0:
        run.kind = "transient"
    # Ensayos cuasi-estáticos de esquina (barridos de SA hasta ~12 deg)
    elif d["sa_amp_deg"] > th.lat_sa_deg:
        run.kind = "lateral"
    elif d["sl_amp"] >= 0.10:
        run.kind = "longitudinal"
    else:
        run.kind = "lateral"

    run.pressure_sweep = bool(forced.get("pressure_sweep", d["p_span_kpa"] >= th.pressure_span_kpa))


def slugify(tire: str, rim: float) -> str:
    rim_s = "NA" if not np.isfinite(rim) else f"{rim:g}".replace(".", "p")
    return re.sub(r"[^A-Za-z0-9]+", "_", f"{tire}_{rim_s}in").strip("_")


def scan_groups(cfg: FitConfig) -> dict[tuple[str, float], list[Path]]:
    manifest = yaml.safe_load(cfg.manifest.read_text()) if cfg.manifest else {}
    over = (manifest or {}).get("tires") or {}
    groups: dict[tuple[str, float], list[Path]] = defaultdict(list)
    for path in sorted(cfg.input.rglob(cfg.pattern)):
        try:
            r = parse_ttc(path, meta_only=True)
        except Exception as exc:
            LOG.warning("Omitido %s: %s", path.name, exc)
            continue
        rim = float(over.get(r.tire_name, {}).get("rim_width_in", r.rim_width_in))
        if cfg.tire_filter and not re.search(cfg.tire_filter, r.tire_name, re.I):
            continue
        if cfg.rim_filter is not None and not (np.isfinite(rim) and abs(rim - cfg.rim_filter) < 0.05):
            continue
        # CLAVE CORREGIDA: No usar float('nan') para que los ficheros se agrupen entre sí
        rim_key = round(rim, 2) if np.isfinite(rim) else -1.0
        groups[(r.tire_name, rim_key)].append(path)
    return dict(groups)


def load_group(paths: Sequence[Path], th: Thresholds, manifest: dict | None) -> list[TTCRun]:
    runs = []
    for p in paths:
        try:
            r = parse_ttc(p)
            classify(r, th, manifest)
            runs.append(r)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("Fallo parseando %s: %s", p.name, exc)
    return runs


# --------------------------------------------------------------------------------------------
# Dataset de ajuste
# --------------------------------------------------------------------------------------------
def _slope(x, y, m) -> float:
    m = m & np.isfinite(x) & np.isfinite(y)
    return float(np.polyfit(x[m], y[m], 1)[0]) if m.sum() >= 20 else 0.0


def assemble(runs: Sequence[TTCRun], cfg: FitConfig) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    rng = np.random.default_rng(cfg.seed)
    per_kind: dict[str, list[dict[str, np.ndarray]]] = defaultdict(list)
    run_idx: dict[str, int] = defaultdict(int)
    for r in runs:
        if r.kind not in STEADY_KINDS:
            continue
        c = r.ch
        m = np.isfinite(c["sa"]) & np.isfinite(c["fy"]) & (c["fz"] >= cfg.min_fz)
        if np.isfinite(c["v"]).any():
            m &= np.nan_to_num(c["v"], nan=1e9) >= cfg.min_speed
        if cfg.trim_s > 0 and "et" in c:
            m &= (c["et"] - c["et"][0]) >= cfg.trim_s
        if m.sum() < 50:
            continue
        rid = run_idx[r.kind]
        run_idx[r.kind] += 1
        per_kind[r.kind].append({
            "alpha": c["sa"][m], "kappa": c["sl"][m], "gamma": c["ia"][m] * cfg.ia_sign,
            "fz": c["fz"][m], "p": c["p"][m], "fx": c["fx"][m], "fy": c["fy"][m], "mz": c["mz"][m],
            "kind": np.full(m.sum(), KIND_CODE[r.kind], np.int8), "run": np.full(m.sum(), rid, np.int32),
        })
    cols = ("alpha", "kappa", "gamma", "fz", "p", "fx", "fy", "mz", "kind", "run")
    chunks: dict[str, list[np.ndarray]] = {k: [] for k in cols}
    for kind, lst in per_kind.items():
        cat = {k: np.concatenate([x[k] for x in lst]) for k in cols}
        n = len(cat["fz"])
        idx = np.sort(rng.choice(n, cfg.max_points, replace=False)) if n > cfg.max_points else np.arange(n)
        for k in cols:
            chunks[k].append(cat[k][idx])
    if not chunks["fz"]:
        return {}, {"n_runs": {}}
    d = {k: np.concatenate(v) for k, v in chunks.items()}
    p_med = np.nanmedian(d["p"]) if np.isfinite(d["p"]).any() else 83.0
    d["p"] = np.where(np.isfinite(d["p"]), d["p"], p_med)

    lat, lon = d["kind"] == 0, d["kind"] == 1
    near = np.abs(d["alpha"]) < np.radians(2.0)
    s_fy = 1.0 if _slope(d["alpha"], d["fy"], lat & near) >= 0 else -1.0
    s_mz = -1.0 if _slope(d["alpha"], d["mz"], lat & near) > 0 else 1.0
    s_fx = 1.0 if _slope(d["kappa"], d["fx"], lon & (np.abs(d["kappa"]) < 0.03)) >= 0 else -1.0
    d["fy"], d["mz"], d["fx"] = d["fy"] * s_fy, d["mz"] * s_mz, d["fx"] * s_fx
    n_runs = {k: int(v) for k, v in run_idx.items()}
    d["val"] = np.zeros(len(d["fz"]), bool)
    for kind, code in KIND_CODE.items():
        if n_runs.get(kind, 0) >= 4:
            d["val"] |= (d["kind"] == code) & (d["run"] % 4 == 3)
    info = {"signs": {"fy": s_fy, "mz": s_mz, "fx": s_fx, "ia": cfg.ia_sign}, "n_runs": n_runs,
            "n_points": int(len(d["fz"])), "n_val": int(d["val"].sum())}
    return d, info


def identifiability(d: dict[str, np.ndarray]) -> dict[str, bool]:
    nfz = len(_levels(d["fz"], 75.0, 200))
    ng = len(_levels(np.degrees(d["gamma"]), 0.5, 200))
    npr = len(_levels(d["p"], 5.0, 200))
    return {"fz2": nfz >= 2, "fz3": nfz >= 3, "gamma": ng >= 2, "p": npr >= 2,
            "_levels": {"fz": nfz, "gamma": ng, "p": npr}}


# --------------------------------------------------------------------------------------------
# Condiciones nominales y semilla
# --------------------------------------------------------------------------------------------
def nominal_params(d: dict[str, np.ndarray], tire_name: str, cfg: FitConfig) -> mf.MF61Params:
    lv = _levels(d["fz"], 75.0, 100)
    med = float(np.median(d["fz"]))
    fz0 = cfg.fz_nominal or (float(lv[np.argmin(np.abs(lv - med))]) if len(lv) else med)
    m = re.search(r"(\d+(?:\.\d+)?)\s*[xX]\s*(\d+(?:\.\d+)?)\s*-\s*(\d+)", tire_name)
    r0 = float(m.group(1)) * 0.0254 / 2.0 if m else 0.2032
    P = mf.MF61Params(FNOMIN=fz0, R0=r0, NOMPRES=float(np.median(d["p"])))
    band = np.abs(d["fz"] - fz0) < 0.15 * fz0
    upd: dict[str, float] = {}

    def clip(name, v):
        s = mf.SPECS[name]
        return float(np.clip(v, s.lo, s.hi))

    lat = band & ((d["kind"] == 0) | (d["kind"] == 2)) & (np.abs(d["kappa"]) < 0.01)
    if lat.sum() > 100:
        mu = np.percentile(np.abs(d["fy"][lat]) / d["fz"][lat], 95)
        upd["pDy1"] = clip("pDy1", mu)
        sm = lat & (np.abs(d["alpha"]) < np.radians(1.0))
        kya = _slope(d["alpha"], d["fy"], sm)
        if kya > 0:
            upd["pKy1"] = clip("pKy1", kya / (fz0 * math.sin(2.0 * math.atan(1.0 / 1.5))))
            kmz = _slope(d["alpha"], d["mz"], sm)
            if kmz < 0:
                upd["qDz1"] = clip("qDz1", (-kmz / kya) / r0)
    lon = band & (d["kind"] == 1) & np.isfinite(d["fx"])
    if lon.sum() > 100:
        upd["pDx1"] = clip("pDx1", np.percentile(np.abs(d["fx"][lon]) / d["fz"][lon], 95))
        kxk = _slope(d["kappa"], d["fx"], lon & (np.abs(d["kappa"]) < 0.02))
        if kxk > 0:
            upd["pKx1"] = clip("pKx1", kxk / fz0)
    return P.replace(**upd)


# --------------------------------------------------------------------------------------------
# Etapas de ajuste
# --------------------------------------------------------------------------------------------
Predict = Callable[[mf.MF61Params, dict], Any]


def _pred_fy_pure(P, d): return mf.lateral_pure_force(P, d["alpha"], d["fz"], d["gamma"], d["p"])
def _pred_fx_pure(P, d): return mf.longitudinal_pure_force(P, d["kappa"], d["fz"], d["gamma"], d["p"])
def _pred_mz_pure(P, d): return mf.aligning_pure_moment(P, d["alpha"], d["fz"], d["gamma"], d["p"])
def _pred_fx_comb(P, d): return mf.mf61(P, d["alpha"], d["kappa"], d["gamma"], d["fz"], d["p"]).fx
def _pred_fy_comb(P, d): return mf.mf61(P, d["alpha"], d["kappa"], d["gamma"], d["fz"], d["p"]).fy
def _pred_mz_comb(P, d): return mf.mf61(P, d["alpha"], d["kappa"], d["gamma"], d["fz"], d["p"]).mz


# etapa -> (grupo de parametros, canal objetivo, predictor)
STAGES: dict[str, tuple[str, str, Predict]] = {
    "fy_pure": ("fy", "fy", _pred_fy_pure),
    "fx_pure": ("fx", "fx", _pred_fx_pure),
    "mz_pure": ("mz", "mz", _pred_mz_pure),
    "fx_comb": ("fxc", "fx", _pred_fx_comb),
    "fy_comb": ("fyc", "fy", _pred_fy_comb),
    "mz_comb": ("mzc", "mz", _pred_mz_comb),
}


def stage_mask(d: dict[str, np.ndarray], stage: str) -> np.ndarray:
    k, ka, a = d["kind"], np.abs(d["kappa"]), np.abs(d["alpha"])
    target = STAGES[stage][1]
    fin = np.isfinite(d[target])
    if stage in ("fy_pure", "mz_pure"):
        return fin & ((k == 0) | ((k == 2) & (ka < 0.01)))
    if stage == "fx_pure":
        return fin & ((k == 1) | ((k == 2) & (a < np.radians(0.3))))
    return fin & (k == 2)


def metrics(pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    e = pred - y
    rng = np.percentile(y, 99) - np.percentile(y, 1)
    ss = float(np.sum((y - y.mean()) ** 2))
    return {"n": int(len(y)), "rmse": float(np.sqrt(np.mean(e**2))),
            "nmae": float(np.mean(np.abs(e)) / rng) if rng > 0 else float("nan"),
            "r2": float(1 - np.sum(e**2) / ss) if ss > 0 else float("nan")}


def fit_stage(stage: str, base: mf.MF61Params, d: dict[str, np.ndarray], ident: dict, cfg: FitConfig,
              rng: np.random.Generator) -> tuple[mf.MF61Params, dict[str, Any]]:
    group, target, predict = STAGES[stage]
    m = stage_mask(d, stage)
    mt = m & ~d["val"]
    if mt.sum() < 100:
        return base, {"status": "skipped", "reason": f"datos insuficientes ({int(mt.sum())})"}
    free = [s for s in mf.GROUPS[group] if all(ident.get(n, False) for n in s.needs)]
    if not free:
        return base, {"status": "skipped", "reason": "sin coeficientes identificables"}
    names = tuple(s.name for s in free)
    lo = np.array([s.lo for s in free])
    hi = np.array([s.hi for s in free])
    width = hi - lo
    x0 = np.clip(base.to_vector(names), lo + 1e-9 * width, hi - 1e-9 * width)
    data = {k: jnp.asarray(d[k][mt]) for k in ("alpha", "kappa", "gamma", "fz", "p")}
    y = jnp.asarray(d[target][mt])
    scale = max(float(np.percentile(np.abs(d[target][mt]), 95)), 1e-9)
    reg, x_ref, w_ = cfg.reg, jnp.asarray(x0), jnp.asarray(width)

    @jax.jit
    def res(x, dd, yy):
        P = base.with_vector(names, x)
        r = (predict(P, dd) - yy) / scale
        return jnp.concatenate([r, math.sqrt(reg) * (x - x_ref) / w_]) if reg > 0 else r

    fun = lambda x: np.array(res(jnp.asarray(x), data, y), copy=True, dtype=np.float64)  # noqa: E731
    jfun = lambda x: np.array(jac(jnp.asarray(x), data, y), copy=True, dtype=np.float64)  # noqa: E731
    best = None
    t0 = time.perf_counter()
    for i in range(max(1, cfg.restarts)):
        start = x0 if i == 0 else np.clip(x0 + rng.uniform(-0.1, 0.1, x0.shape) * width, lo + 1e-9 * width, hi - 1e-9 * width)
        try:
            sol = least_squares(fun, start, jac="2-point", bounds=(lo, hi), loss=cfg.loss, f_scale=cfg.f_scale,
                                x_scale="jac", max_nfev=cfg.max_nfev, method="trf")
        except (FloatingPointError, ValueError) as exc:
            LOG.warning("  %s: reinicio %d fallo (%s)", stage, i, exc)
            continue
        if best is None or sol.cost < best.cost:
            best = sol
    if best is None:
        return base, {"status": "failed", "reason": "least_squares no convergio"}
    P = base.with_vector(names, best.x)
    P = mf.MF61Params({k: float(v) for k, v in P._d.items()})
    out = {"status": "ok", "free": list(names), "cost": float(best.cost), "nfev": int(best.nfev),
           "message": best.message, "seconds": round(time.perf_counter() - t0, 2)}
    full = {k: jnp.asarray(d[k][m]) for k in ("alpha", "kappa", "gamma", "fz", "p")}
    pred = np.asarray(predict(P, full))
    yy = d[target][m]
    vm = d["val"][m]
    out["train"] = metrics(pred[~vm], yy[~vm])
    out["val"] = metrics(pred[vm], yy[vm]) if vm.sum() > 50 else None
    at_bound = [n for n, v, l, h in zip(names, best.x, lo, hi) if min(v - l, h - v) < 1e-3 * (h - l)]
    out["at_bounds"] = at_bound
    return P, out


STAGE_ORDER = ("fy_pure", "fx_pure", "mz_pure", "fx_comb", "fy_comb", "mz_comb")


def fit_dataset(tire_name: str, d: dict[str, np.ndarray], cfg: FitConfig):
    rng = np.random.default_rng(cfg.seed)
    ident = identifiability(d)
    P = nominal_params(d, tire_name, cfg)
    report: dict[str, Any] = {}
    for stage in STAGE_ORDER:
        if cfg.no_combined and stage.endswith("comb"):
            report[stage] = {"status": "skipped", "reason": "--no-combined"}
            continue
        P, rep = fit_stage(stage, P, d, ident, cfg, rng)
        report[stage] = rep
        if rep["status"] == "ok":
            LOG.info("  %-8s ok  R2=%.4f NMAE=%.4f (%d coef, %.1fs)", stage, rep["train"]["r2"],
                     rep["train"]["nmae"], len(rep["free"]), rep["seconds"])
        else:
            LOG.info("  %-8s %s: %s", stage, rep["status"], rep.get("reason", ""))
    return P, report, ident


# --------------------------------------------------------------------------------------------
# Un grupo (tire, rim) completo
# --------------------------------------------------------------------------------------------
def fit_group(key: tuple[str, float], paths: list[str], cfg: FitConfig) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    tire, rim = key
    slug = slugify(tire, rim)
    LOG.info("== %s (%d ficheros)", slug, len(paths))
    manifest = yaml.safe_load(cfg.manifest.read_text()) if cfg.manifest else {}
    runs = load_group([Path(p) for p in paths], cfg.thresholds, manifest)
    table = [{"file": r.path.name, "kind": r.kind, "pressure_sweep": r.pressure_sweep,
              **{k: round(v, 4) for k, v in r.desc.items()}} for r in runs]
    entry: dict[str, Any] = {"slug": slug, "tire_name": tire, "rim_width_in": rim, "files": [r.path.name for r in runs],
                             "kinds": {k: sum(r.kind == k for r in runs) for k in sorted({r.kind for r in runs})}}
    d, info = assemble(runs, cfg)
    if not d:
        return {**entry, "status": "no_steady_data"}
    P, rep, ident = fit_dataset(tire, d, cfg)
    out = cfg.output
    out.mkdir(parents=True, exist_ok=True)
    nom = mf.summary(P, [P.FNOMIN], 0.0, P.NOMPRES)
    summary = {k: float(v[0]) for k, v in nom.items()}
    meta = {"tire_name": tire, "rim_width_in": rim, "fitted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "signs": info["signs"], "ident": ident, "fit_loss": cfg.loss, "f_scale": cfg.f_scale, "summary_nominal": summary}
    P.save(out / f"{slug}.yaml", meta)
    keep = ("alpha", "kappa", "gamma", "fz", "p", "fx", "fy", "mz", "kind")
    np.savez_compressed(out / f"{slug}_data.npz", **{k: d[k] for k in keep})
    report = {"slug": slug, "info": info, "ident": ident, "stages": rep, "runs": table}
    (out / f"{slug}_report.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    return {**entry, "status": "ok", "params": f"{slug}.yaml", "data": f"{slug}_data.npz",
            "report": f"{slug}_report.json", "fnomin": P.FNOMIN, "nompres": P.NOMPRES, "summary": summary,
            "r2": {s: (r.get("train") or {}).get("r2") for s, r in rep.items()}}


def run_batch(cfg: FitConfig) -> list[dict[str, Any]]:
    groups = scan_groups(cfg)
    if not groups:
        raise SystemExit(f"Sin .dat en {cfg.input} (patron {cfg.pattern})")
    LOG.info("%d grupos (Tire_Name, Rim_Width): %s", len(groups), ", ".join(slugify(*k) for k in groups))
    jobs = [(k, [str(p) for p in v], cfg) for k, v in groups.items()]
    results: list[dict[str, Any]] = []
    if cfg.jobs <= 1:
        results = [fit_group(*j) for j in jobs]
    else:
        with cf.ProcessPoolExecutor(cfg.jobs, mp_context=mp.get_context("spawn")) as ex:
            futs = {ex.submit(fit_group, *j): j[0] for j in jobs}
            for f in cf.as_completed(futs):
                try:
                    results.append(f.result())
                except Exception as exc:  # noqa: BLE001
                    LOG.error("Grupo %s fallo: %s", futs[f], exc)
                    results.append({"slug": slugify(*futs[f]), "tire_name": futs[f][0],
                                    "rim_width_in": futs[f][1], "status": f"error: {exc}"})
    results.sort(key=lambda r: (r["tire_name"], r.get("rim_width_in", 0)))
    cfg.output.mkdir(parents=True, exist_ok=True)
    (cfg.output / "index.json").write_text(json.dumps({"version": 1, "tires": results}, indent=2, default=float),
                                           encoding="utf-8")
    return results


def dry_run(cfg: FitConfig) -> None:
    manifest = yaml.safe_load(cfg.manifest.read_text()) if cfg.manifest else {}
    for (tire, rim), paths in scan_groups(cfg).items():
        print(f"\n[{slugify(tire, rim)}]  {tire}  rim={rim:g}\"")
        for r in load_group(paths, cfg.thresholds, manifest):
            d = r.desc
            print(f"  {r.path.name:<28} {r.kind:<13} P={'*' if r.pressure_sweep else ' '} "
                  f"SA={d['sa_amp_deg']:5.1f}° SL={d['sl_amp']:.3f} dSA/dt={d['sa_rate_degs']:5.1f}°/s "
                  f"Fz×{int(d['n_fz_levels'])} IA×{int(d['n_ia_levels'])} P×{int(d['n_p_levels'])}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    D = FitConfig()
    ap.add_argument("--input", type=Path, default=D.input)
    ap.add_argument("--output", type=Path, default=D.output)
    ap.add_argument("--pattern", default=D.pattern)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--loss", choices=("soft_l1", "huber", "cauchy", "linear"), default=D.loss)
    ap.add_argument("--f-scale", type=float, default=D.f_scale)
    ap.add_argument("--max-nfev", type=int, default=D.max_nfev)
    ap.add_argument("--max-points", type=int, default=D.max_points)
    ap.add_argument("--restarts", type=int, default=D.restarts)
    ap.add_argument("--reg", type=float, default=D.reg)
    ap.add_argument("--jobs", type=int, default=D.jobs)
    ap.add_argument("--min-fz", type=float, default=D.min_fz)
    ap.add_argument("--min-speed", type=float, default=D.min_speed)
    ap.add_argument("--trim-s", type=float, default=D.trim_s)
    ap.add_argument("--ia-sign", type=float, choices=(-1.0, 1.0), default=D.ia_sign)
    ap.add_argument("--fz-nominal", type=float)
    ap.add_argument("--no-combined", action="store_true")
    ap.add_argument("--tire-filter")
    ap.add_argument("--rim-filter", type=float)
    ap.add_argument("--steady-rate", type=float, default=Thresholds().sa_rate_degs, help="deg/s")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="solo indexa y clasifica")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap


def config_from_args(a: argparse.Namespace) -> FitConfig:
    return FitConfig(input=a.input, output=a.output, pattern=a.pattern, manifest=a.manifest, loss=a.loss,
                     f_scale=a.f_scale, max_nfev=a.max_nfev, max_points=a.max_points, restarts=a.restarts,
                     reg=a.reg, jobs=a.jobs, min_fz=a.min_fz, min_speed=a.min_speed, trim_s=a.trim_s,
                     ia_sign=a.ia_sign, fz_nominal=a.fz_nominal, no_combined=a.no_combined,
                     tire_filter=a.tire_filter, rim_filter=a.rim_filter, seed=a.seed,
                     thresholds=Thresholds(sa_rate_degs=a.steady_rate))


def main(argv: Sequence[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    cfg = config_from_args(a)
    if a.dry_run:
        dry_run(cfg)
        return 0
    res = run_batch(cfg)
    for r in res:
        s = r.get("summary", {})
        print(f"{r['slug']:<40} {r['status']:<10} "
              + (f"Cα={s['c_alpha_N_deg']:.0f} N/°  μy={s['mu_y_peak']:.2f}  μx={s['mu_x_peak']:.2f}" if s else ""))
    return 0 if all(r["status"] == "ok" for r in res) else 1


if __name__ == "__main__":
    raise SystemExit(main())