#!/usr/bin/env python3
"""Terminal visual de analisis y comparacion de neumaticos MF6.1 (TeR-Twin).

Uso:
    python apps/desktop/tire_terminal.py --index data/processed/mf61_fits/index.json
    python apps/desktop/tire_terminal.py --demo                 # 3 neumaticos sinteticos
    python apps/desktop/tire_terminal.py --demo --selftest out/ # renderiza todas las pestanas a PNG (headless)

Caracteristicas
* Seleccion multiple (checkboxes) con filtro por ancho de llanta; curvas superpuestas al instante.
* Pestanas: Fy-α, Mz-α, Fx-κ, traza t(α), envolvente combinada, sensibilidad a carga, camber y
  "Comparativa 7" vs 8"" (Δ% de Cα, μ pico, Fy/Mz pico, traza, Cκ; tabla + graficos + export CSV).
* Superposicion opcional de datos TTC medidos (NPZ del ajuste, ya en convencion del modelo).
* Controles: lista de cargas Fz, caida γ, presion p (o nominal por neumatico), κ (para Fy/Mz), α (para Fx).

Suposiciones: lee index.json/YAML/NPZ generados por fit_all_ttc_dat.py; independiente del main del coche.
El motor de dibujo (`Engine`) no depende de Tk; la GUI (Tk) se importa solo al ejecutar `run_gui`.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
from matplotlib import colormaps  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from ter_twin.models.tires import pacejka_61 as mf  # noqa: E402

LOG = logging.getLogger("tire_terminal")
TABS = ("Fy–α", "Mz–α", "Fx–κ", "Traza t(α)", "Envolvente", "Sens. carga", "Camber", 'Comparativa 7" vs 8"')
LINESTYLES = ("-", "--", "-.", ":")
CMP_KEYS = ("c_alpha_N_deg", "mu_y_peak", "mu_x_peak", "fy_peak_N", "mz_peak_Nm", "trail0_mm", "c_kappa_N")


# --------------------------------------------------------------------------------------------
# Almacen de neumaticos
# --------------------------------------------------------------------------------------------
@dataclass
class Entry:
    slug: str
    tire_name: str
    rim: float
    params_path: Path | None = None
    data_path: Path | None = None
    params_obj: mf.MF61Params | None = None
    summary: dict[str, float] = field(default_factory=dict)
    r2: dict[str, float | None] = field(default_factory=dict)
    _data: dict[str, np.ndarray] | None = None

    @property
    def label(self) -> str:
        rim = f'{self.rim:g}"' if np.isfinite(self.rim) else "?"
        return f"{self.tire_name} · {rim}"

    @property
    def params(self) -> mf.MF61Params:
        if self.params_obj is None:
            self.params_obj = mf.MF61Params.load(self.params_path)
        return self.params_obj
    
    def fz_levels(self) -> list[float]:
        """Detecta los escalones reales de carga ensayados en Calspan."""
        d = self.data()
        if d is not None and "fz" in d:
            fz = d["fz"]
            valid = fz[np.isfinite(fz) & (fz > 80.0)]
            if valid.size > 0:
                # Agrupación por múltiplos de 75 N para aislar los plateaus
                q = np.round(valid / 75.0) * 75.0
                u, counts = np.unique(q, return_counts=True)
                min_pts = max(80, int(0.025 * valid.size))
                cand = u[counts >= min_pts]
                levels = []
                for c in cand:
                    sub = valid[np.abs(valid - c) <= 40.0]
                    if sub.size > 0:
                        levels.append(float(np.round(np.median(sub))))
                if levels:
                    return sorted(levels)
        # Fallback analítico si no hay datos .npz
        f0 = self.params.FNOMIN
        return [float(np.round(f0 * k)) for k in (0.4, 0.7, 1.0, 1.4)]
    
    def data(self) -> dict[str, np.ndarray] | None:
        if self._data is None and self.data_path and self.data_path.exists():
            with np.load(self.data_path) as z:
                self._data = {k: z[k] for k in z.files}
        return self._data


class Store:
    def __init__(self, entries: list[Entry]):
        self.entries = entries
        self.by_slug = {e.slug: e for e in entries}
        self.rims = sorted({e.rim for e in entries if np.isfinite(e.rim)})
        names = sorted({e.tire_name for e in entries})
        cmap = colormaps["tab10"]
        self.color = {n: to_hex(cmap(i % 10)) for i, n in enumerate(names)}

    @classmethod
    def from_index(cls, index_path: Path) -> "Store":
        doc = json.loads(index_path.read_text(encoding="utf-8"))
        base = index_path.parent
        out = []
        for t in doc["tires"]:
            if t.get("status") != "ok":
                LOG.warning("Omitido %s (%s)", t.get("slug"), t.get("status"))
                continue
            out.append(Entry(t["slug"], t["tire_name"], float(t["rim_width_in"]), base / t["params"],
                             base / t["data"] if t.get("data") else None, None, t.get("summary", {}), t.get("r2", {})))
        if not out:
            raise SystemExit(f"{index_path}: sin neumaticos ajustados")
        return cls(out)

    @classmethod
    def demo(cls) -> "Store":
        def mk(name, rim, **kw):
            P = mf.MF61Params(ppy3=0.3, ppy1=0.2, **kw)
            return Entry(re.sub(r"\W+", "_", f"{name}_{rim:g}"), name, rim, params_obj=P)
        h = "Hoosier 16.0x7.5-10 R20"
        return cls([
            mk(h, 7.0),
            mk(h, 8.0, pKy1=45 * 1.08, pDy1=2.4 * 1.02, qDz1=0.15 * 0.94),
            mk("Hoosier 18.0x6.0-13", 7.0, R0=0.2286, pKy1=40.0, pDy1=2.25, pDx1=2.1, pKy2=1.7),
        ])

    def pair_7_8(self) -> tuple[str | None, str | None]:
        for name in sorted({e.tire_name for e in self.entries}):
            lst = [e for e in self.entries if e.tire_name == name and np.isfinite(e.rim)]
            a = min(lst, key=lambda e: abs(e.rim - 7.0), default=None)
            b = min(lst, key=lambda e: abs(e.rim - 8.0), default=None)
            if a and b and a is not b and abs(a.rim - 7) < 0.6 and abs(b.rim - 8) < 0.6:
                return a.slug, b.slug
        return None, None


@dataclass
class Controls:
    fz: list[float] = field(default_factory=lambda: [600.0, 1100.0, 1600.0])
    gamma_deg: float = 0.0
    p_kpa: float | None = None
    kappa: float = 0.0
    alpha_deg: float = 0.0
    overlay: bool = True
    ref: str | None = None
    cmp: str | None = None


# --------------------------------------------------------------------------------------------
# Motor de dibujo (sin Tk)
# --------------------------------------------------------------------------------------------
class Engine:
    def __init__(self, store: Store):
        self.store = store

    # -- utilidades
    def style(self, e: Entry) -> tuple[str, str]:
        idx = self.store.rims.index(e.rim) if e.rim in self.store.rims else 0
        return self.store.color[e.tire_name], LINESTYLES[idx % len(LINESTYLES)]

    @staticmethod
    def _p(e: Entry, ctl: Controls) -> float:
        return ctl.p_kpa if ctl.p_kpa is not None else e.params.NOMPRES

    def _eval(self, e: Entry, ctl: Controls, alpha, kappa, fz, gamma=None) -> mf.MF61Result:
        g = math.radians(ctl.gamma_deg) if gamma is None else gamma
        r = mf.evaluate(e.params, alpha, kappa, g, fz, self._p(e, ctl))
        return mf.MF61Result(*(np.asarray(x) for x in r))

    def _meas(self, e: Entry, ctl: Controls, fz: float, kinds, xk, yk, extra=None):
        d = e.data()
        if d is None or not ctl.overlay:
            return None
        # Tolerancia ampliada a 120 N para que 600 N capture los 667 N reales, y 1100 N capture los 1112 N
        m = (np.isin(d["kind"], kinds) & (np.abs(d["fz"] - fz) < 85.0)
             & (np.abs(np.degrees(d["gamma"]) - ctl.gamma_deg) < 1.0)
             & (np.abs(d["p"] - self._p(e, ctl)) < 15.0))
        # En curva lateral pura (kind == 0), no filtramos por kappa para evitar descartar datos
        if extra is not None and 0 not in kinds:
            m &= extra(d)
        idx = np.flatnonzero(m & np.isfinite(d[yk]))
        if idx.size > 3000:
            idx = idx[np.linspace(0, idx.size - 1, 3000).astype(int)]
        return (d[xk][idx], d[yk][idx]) if idx.size else None

    @staticmethod
    def _decorate(ax, xl, yl, title=None, legend=True):
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="k", lw=0.5)
        ax.axvline(0, color="k", lw=0.5)
        if title:
            ax.set_title(title, fontsize=10)
        if legend and ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7, ncol=2, loc="best")

    def _empty(self, fig: Figure, msg="Selecciona al menos un neumático"):
        fig.clear()
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=12, color="gray")

    # -- pestanas simples: curva vs deslizamiento
    def _slip_plot(self, fig, slugs, ctl, *, xs, xkind, ykey, scale, xl, yl, title, meas_kinds, meas_x, meas_y,
                   meas_scale_x, meas_scale_y, meas_extra):
        fig.clear()
        ax = fig.add_subplot(111)
        n = len(ctl.fz)
        for s in slugs:
            e = self.store.by_slug[s]
            col, ls = self.style(e)
            for j, fz in enumerate(ctl.fz):
                alpha_a, kappa_a = (xs, ctl.kappa) if xkind == "alpha" else (math.radians(ctl.alpha_deg), xs)
                r = self._eval(e, ctl, alpha_a, kappa_a, fz)
                x = np.degrees(xs) if xkind == "alpha" else xs * 100.0
                ax.plot(x, getattr(r, ykey) * scale, color=col, ls=ls, alpha=0.4 + 0.6 * (j + 1) / n,
                        label=f"{e.label} @ {fz:.0f} N")
                m = self._meas(e, ctl, fz, meas_kinds, meas_x, meas_y, meas_extra)
                if m is not None:
                    # s=15 y alpha=0.45 para que se vean claramente definidos
                    ax.scatter(m[0] * meas_scale_x, m[1] * meas_scale_y, s=15, color=col, alpha=0.45, rasterized=True)
        self._decorate(ax, xl, yl, title)

    def fy_alpha(self, fig, slugs, ctl):
        a = np.radians(np.linspace(-14, 14, 281))
        self._slip_plot(fig, slugs, ctl, xs=a, xkind="alpha", ykey="fy", scale=1.0, xl="α [°]", yl="Fy [N]",
                        title=f"Fuerza lateral — γ={ctl.gamma_deg:.1f}°, κ={ctl.kappa:.3f}",
                        meas_kinds=(0, 2), meas_x="alpha", meas_y="fy", meas_scale_x=180 / math.pi, meas_scale_y=1.0,
                        meas_extra=lambda d: np.abs(d["kappa"] - ctl.kappa) < 0.01)

    def mz_alpha(self, fig, slugs, ctl):
        a = np.radians(np.linspace(-14, 14, 281))
        self._slip_plot(fig, slugs, ctl, xs=a, xkind="alpha", ykey="mz", scale=1.0, xl="α [°]", yl="Mz [N·m]",
                        title="Momento autoalineante  Mz = −t·Fy′ + Mzr (+ s·Fx)",
                        meas_kinds=(0, 2), meas_x="alpha", meas_y="mz", meas_scale_x=180 / math.pi, meas_scale_y=1.0,
                        meas_extra=lambda d: np.abs(d["kappa"] - ctl.kappa) < 0.01)

    def trail(self, fig, slugs, ctl):
        a = np.radians(np.linspace(-14, 14, 281))
        self._slip_plot(fig, slugs, ctl, xs=a, xkind="alpha", ykey="t", scale=1e3, xl="α [°]",
                        yl="Traza neumática t [mm]", title="Traza neumática", meas_kinds=(), meas_x="alpha",
                        meas_y="fy", meas_scale_x=1.0, meas_scale_y=1.0, meas_extra=None)

    def fx_kappa(self, fig, slugs, ctl):
        k = np.linspace(-0.3, 0.3, 301)
        a0 = math.radians(ctl.alpha_deg)
        self._slip_plot(fig, slugs, ctl, xs=k, xkind="kappa", ykey="fx", scale=1.0, xl="κ [%]", yl="Fx [N]",
                        title=f"Fuerza longitudinal — α={ctl.alpha_deg:.1f}°, γ={ctl.gamma_deg:.1f}°",
                        meas_kinds=(1, 2), meas_x="kappa", meas_y="fx", meas_scale_x=100.0, meas_scale_y=1.0,
                        meas_extra=lambda d: np.abs(d["alpha"] - a0) < math.radians(0.5))

    # -- envolvente
    def envelope(self, fig, slugs, ctl):
        fig.clear()
        ax = fig.add_subplot(111)
        fz = ctl.fz[0]
        for s in slugs:
            e = self.store.by_slug[s]
            col, ls = self.style(e)
            fx, fy, al, ka = mf.friction_envelope(e.params, fz, math.radians(ctl.gamma_deg), self._p(e, ctl))
            for i in range(0, len(al), 6):
                ax.plot(fy[i], fx[i], color=col, lw=0.9, ls=ls, alpha=0.8, label=e.label if i == 0 else None)
            for j in range(0, len(ka), 6):
                ax.plot(fy[:, j], fx[:, j], color=col, lw=0.6, ls=":", alpha=0.5)
        ax.set_aspect("equal", adjustable="datalim")
        self._decorate(ax, "Fy [N]", "Fx [N]", f"Elipse de fricción combinada @ Fz={fz:.0f} N (líneas: α=cte; puntos: κ=cte)")

    # -- sensibilidad a carga y camber
    def load_sens(self, fig, slugs, ctl):
        fig.clear()
        fzs = np.linspace(100, 2200, 36)
        axs = fig.subplots(2, 2).ravel()
        spec = (("c_alpha_N_deg", "Cα [N/°]"), ("mu_y_peak", "μy pico [-]"), ("mu_x_peak", "μx pico [-]"),
                ("trail0_mm", "t0 [mm]"))
        for s in slugs:
            e = self.store.by_slug[s]
            col, ls = self.style(e)
            out = mf.summary(e.params, fzs, math.radians(ctl.gamma_deg), self._p(e, ctl))
            for ax, (k, yl) in zip(axs, spec):
                ax.plot(fzs, out[k], color=col, ls=ls, label=e.label)
                ax.set_ylabel(yl)
        for ax in axs:
            ax.set_xlabel("Fz [N]")
            ax.grid(True, alpha=0.3)
        for fz in ctl.fz:
            for ax in axs:
                ax.axvline(fz, color="gray", lw=0.5, ls=":")
        axs[0].legend(fontsize=7)
        fig.suptitle(f"Sensibilidad a la carga — γ={ctl.gamma_deg:.1f}°", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.95))

    def camber(self, fig, slugs, ctl):
        fig.clear()
        gs = np.radians(np.linspace(-6, 6, 13))
        axs = fig.subplots(1, 3)
        fz = ctl.fz[0]
        for s in slugs:
            e = self.store.by_slug[s]
            col, ls = self.style(e)
            rows = [mf.summary(e.params, [fz], g, self._p(e, ctl)) for g in gs]
            g_deg = np.degrees(gs)
            axs[0].plot(g_deg, [r["fy_peak_N"][0] for r in rows], color=col, ls=ls, label=e.label)
            axs[1].plot(g_deg, [r["c_alpha_N_deg"][0] for r in rows], color=col, ls=ls)
            axs[2].plot(g_deg, [float(self._eval(e, ctl, 0.0, 0.0, fz, g).fy) for g in gs], color=col, ls=ls)
        for ax, yl in zip(axs, ("Fy pico [N]", "Cα [N/°]", "Empuje de camber Fy(α=0) [N]")):
            ax.set_xlabel("γ [°]")
            ax.set_ylabel(yl)
            ax.grid(True, alpha=0.3)
        axs[0].legend(fontsize=7)
        fig.suptitle(f"Efecto de la caída @ Fz={fz:.0f} N", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.93))

    # -- comparativa directa de llantas
    def compare(self, ctl: Controls) -> dict[str, Any] | None:
        if not ctl.ref or not ctl.cmp or ctl.ref == ctl.cmp:
            return None
        a, b = self.store.by_slug[ctl.ref], self.store.by_slug[ctl.cmp]
        return mf.compare_params(a.params, b.params, ctl.fz, math.radians(ctl.gamma_deg),
                                 self._p(a, ctl), self._p(b, ctl))

    def draw_compare(self, fig, ctl):
        res = self.compare(ctl)
        if res is None:
            return self._empty(fig, "Elige dos neumáticos distintos (base y comparado)")
        a, b = self.store.by_slug[ctl.ref], self.store.by_slug[ctl.cmp]
        fig.clear()
        ax_bar, ax_ca, ax_mu, ax_fy = fig.subplots(2, 2).ravel()
        keys = list(CMP_KEYS)
        vals = [res["mean_delta_pct"][k] for k in keys]
        y = np.arange(len(keys))
        ax_bar.barh(y, vals, color=["#2a9d8f" if v >= 0 else "#e76f51" for v in vals])
        ax_bar.set_yticks(y, [mf.METRIC_LABELS[k] for k in keys], fontsize=8)
        ax_bar.invert_yaxis()
        for yi, v in zip(y, vals):
            ax_bar.text(v, yi, f" {v:+.1f}%", va="center", ha="left" if v >= 0 else "right", fontsize=8)
        ax_bar.axvline(0, color="k", lw=0.6)
        ax_bar.set_title(f"Δ% medio ({b.label}) vs ({a.label})", fontsize=9)
        ax_bar.grid(True, axis="x", alpha=0.3)
        fzs = np.linspace(100, 2200, 36)
        g = math.radians(ctl.gamma_deg)
        for e, ls in ((a, "-"), (b, "--")):
            out = mf.summary(e.params, fzs, g, self._p(e, ctl))
            col = self.style(e)[0]
            ax_ca.plot(fzs, out["c_alpha_N_deg"], ls=ls, color=col if e is a else "#264653", label=e.label)
            ax_mu.plot(fzs, out["mu_y_peak"], ls=ls, color=col if e is a else "#264653", label=e.label)
        ax_ca.set(xlabel="Fz [N]", ylabel="Cα [N/°]")
        ax_mu.set(xlabel="Fz [N]", ylabel="μy pico [-]")
        for ax in (ax_ca, ax_mu):
            ax.grid(True, alpha=0.3)
        ax_ca.legend(fontsize=7)
        al = np.radians(np.linspace(-14, 14, 281))
        fz0 = ctl.fz[0]
        for e, ls in ((a, "-"), (b, "--")):
            r = self._eval(e, ctl, al, 0.0, fz0)
            col = self.style(e)[0] if e is a else "#264653"
            ax_fy.plot(np.degrees(al), r.fy, ls=ls, color=col, label=e.label)
            # SUPERPOSICIÓN DE PUNTOS EXPERIMENTALES
            m = self._meas(e, ctl, fz0, (0, 2), "alpha", "fy")
            if m is not None:
                ax_fy.scatter(np.degrees(m[0]), m[1], s=12, color=col, alpha=0.35, rasterized=True)
        ax_fy.set(xlabel="α [°]", ylabel="Fy [N]", title=f"Fy(α) @ {fz0:.0f} N")
        ax_fy.grid(True, alpha=0.3)
        fig.suptitle("Impacto de la rigidez de flanco: base vs comparado", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.95))

    # -- despachador
    def draw(self, fig: Figure, tab: str, slugs: list[str], ctl: Controls) -> None:
        if tab == TABS[7]:
            return self.draw_compare(fig, ctl)
        if not slugs:
            return self._empty(fig)
        {TABS[0]: self.fy_alpha, TABS[1]: self.mz_alpha, TABS[2]: self.fx_kappa, TABS[3]: self.trail,
         TABS[4]: self.envelope, TABS[5]: self.load_sens, TABS[6]: self.camber}[tab](fig, slugs, ctl)
        if tab not in (TABS[5], TABS[6]):
            fig.tight_layout()


def rows_to_csv(path: Path, res: dict[str, Any], ref: Entry, cmp_: Entry) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["ref", "cmp", "gamma_deg", "fz_N", "metric", "ref_value", "cmp_value", "delta_pct"])
        for r in res["rows"]:
            w.writerow([ref.label, cmp_.label, res["gamma_deg"], r["fz"], r["metric"], r["ref"], r["cmp"], r["delta_pct"]])


# --------------------------------------------------------------------------------------------
# GUI Tk
# --------------------------------------------------------------------------------------------
def _safe(name: str) -> str:
    return re.sub(r"\W+", "_", name).strip("_")


def merge_close_levels(levels: list[float], tol: float = 60.0) -> list[float]:
    if not levels:
        return []
    s = sorted(levels)
    merged = []
    curr = [s[0]]
    for x in s[1:]:
        if x - curr[-1] <= tol:
            curr.append(x)
        else:
            merged.append(float(np.round(np.mean(curr))))
            curr = [x]
    if curr:
        merged.append(float(np.round(np.mean(curr))))
    return merged


def parse_fz(text: str, default: list[float]) -> list[float]:
    try:
        v = [float(t) for t in re.split(r"[,\s;]+", text.strip()) if t]
    except ValueError:
        return default
    # Aumentado a 10 cargas para no recortar escalones
    return [x for x in v if x > 0][:10] or default


def run_gui(store: Store, index_path: Path | None) -> None:
    engine = Engine(store)
    root = tk.Tk()
    root.title("TeR-Twin · Terminal de neumáticos MF6.1")
    root.geometry("1500x900")

    checks: dict[str, tk.BooleanVar] = {}
    for i, e in enumerate(store.entries):
        checks[e.slug] = tk.BooleanVar(value=i < 2)
    v_fz = tk.StringVar(value="445, 667, 1112")
    v_fz_auto = tk.BooleanVar(value=True)  # Activo por defecto
    v_gamma, v_kappa, v_alpha, v_p = tk.DoubleVar(value=0.0), tk.DoubleVar(value=0.0), tk.DoubleVar(value=0.0), tk.DoubleVar(value=83.0)
    v_pauto, v_overlay = tk.BooleanVar(value=True), tk.BooleanVar(value=True)
    v_ref, v_cmp = tk.StringVar(), tk.StringVar()
    v_status = tk.StringVar(value="Listo")
    labels = {e.label: e.slug for e in store.entries}
    ref0, cmp0 = store.pair_7_8()
    if ref0:
        v_ref.set(store.by_slug[ref0].label)
        v_cmp.set(store.by_slug[cmp0].label)
        checks[ref0].set(True)
        checks[cmp0].set(True)
    state: dict[str, Any] = {"after": None, "last_cmp": None}

    # --- layout
    left = ttk.Frame(root, padding=6)
    left.pack(side="left", fill="y")
    right = ttk.Frame(root)
    right.pack(side="left", fill="both", expand=True)
    ttk.Label(left, text="Neumáticos", font=("TkDefaultFont", 11, "bold")).pack(anchor="w")
    cb_rim = ttk.Combobox(left, state="readonly", width=14, values=["Todas las llantas"] + [f'{r:g}"' for r in store.rims])
    cb_rim.current(0)
    cb_rim.pack(anchor="w", pady=2)
    box = ttk.Frame(left)
    box.pack(fill="y", expand=True)
    canvas_l = tk.Canvas(box, width=300, highlightthickness=0)
    sb = ttk.Scrollbar(box, orient="vertical", command=canvas_l.yview)
    inner = ttk.Frame(canvas_l)
    inner.bind("<Configure>", lambda _e: canvas_l.configure(scrollregion=canvas_l.bbox("all")))
    canvas_l.create_window((0, 0), window=inner, anchor="nw")
    canvas_l.configure(yscrollcommand=sb.set)
    canvas_l.pack(side="left", fill="y", expand=True)
    sb.pack(side="right", fill="y")

    def schedule(*_a):
        if state["after"] is not None:
            root.after_cancel(state["after"])
        state["after"] = root.after(40, redraw)

    def rebuild_list(*_a):
        for w in inner.winfo_children():
            w.destroy()
        sel = cb_rim.get()
        for e in store.entries:
            if sel != "Todas las llantas" and not (np.isfinite(e.rim) and f'{e.rim:g}"' == sel):
                continue
            row = ttk.Frame(inner)
            row.pack(anchor="w", fill="x")
            tk.Label(row, bg=store.color[e.tire_name], width=2).pack(side="left", padx=(0, 4))
            ttk.Checkbutton(row, text=f"{e.label}  ({engine.style(e)[1]})", variable=checks[e.slug],
                            command=schedule).pack(side="left")

    cb_rim.bind("<<ComboboxSelected>>", rebuild_list)
    rebuild_list()
    bf = ttk.Frame(left)
    bf.pack(fill="x", pady=4)

    def set_all(val):
        for v in checks.values():
            v.set(val)
        schedule()

    def pick_pair():
        a, b = store.pair_7_8()
        if not a:
            return messagebox.showinfo("Par 7\"/8\"", "No hay un mismo Tire_Name con llantas de 7\" y 8\".")
        set_all(False)
        checks[a].set(True)
        checks[b].set(True)
        v_ref.set(store.by_slug[a].label)
        v_cmp.set(store.by_slug[b].label)
        schedule()

    ttk.Button(bf, text="Todos", command=lambda: set_all(True)).pack(side="left")
    ttk.Button(bf, text="Ninguno", command=lambda: set_all(False)).pack(side="left", padx=2)
    ttk.Button(bf, text='Par 7"/8"', command=pick_pair).pack(side="left")

    def sync_auto_fz(force: bool = False):
        if not v_fz_auto.get() and not force:
            return
        # En la pestaña comparativa toma base y comparado; en las demás toma los seleccionados
        tab = nb.tab(nb.select(), "text") if nb.tabs() else ""
        if tab == TABS[7]:
            slugs = [labels.get(v_ref.get()), labels.get(v_cmp.get())]
        else:
            slugs = [s for s, v in checks.items() if v.get()]
        slugs = [s for s in slugs if s]
        if not slugs:
            return
        raw = []
        for s in slugs:
            e = store.by_slug.get(s)
            if e:
                raw.extend(e.fz_levels())
        merged = merge_close_levels(raw)
        if merged:
            txt = ", ".join(f"{int(round(x))}" for x in merged)
            if v_fz.get() != txt:
                v_fz.set(txt)

    # controles
    ctrl = ttk.LabelFrame(right, text="Condiciones", padding=6)
    ctrl.pack(fill="x", padx=6, pady=4)
    lab: dict[str, tk.StringVar] = {k: tk.StringVar() for k in "gkap"}

    def slider(col, text, var, lo, hi, key, fmt):
        ttk.Label(ctrl, text=text).grid(row=0, column=col * 3, sticky="e", padx=2)
        ttk.Scale(ctrl, from_=lo, to=hi, variable=var, length=140,
                  command=lambda _v: (lab[key].set(fmt.format(var.get())), schedule())).grid(row=0, column=col * 3 + 1)
        lab[key].set(fmt.format(var.get()))
        ttk.Label(ctrl, textvariable=lab[key], width=8).grid(row=0, column=col * 3 + 2, sticky="w")

    ttk.Label(ctrl, text="Fz [N]").grid(row=1, column=0, sticky="e")
    e_fz = ttk.Entry(ctrl, textvariable=v_fz, width=22)
    e_fz.grid(row=1, column=1, columnspan=2, sticky="w")
    e_fz.bind("<Return>", schedule)
    e_fz.bind("<FocusOut>", schedule)
    # Si el usuario escribe manualmente, se desmarca el modo automático
    e_fz.bind("<Key>", lambda _e: v_fz_auto.set(False))

    def on_click_auto_fz():
        v_fz_auto.set(True)
        sync_auto_fz(force=True)
        schedule()

    ttk.Button(ctrl, text="Auto Fz", width=8, command=on_click_auto_fz).grid(row=1, column=3, padx=(4, 2), sticky="w")
    ttk.Checkbutton(ctrl, text="Fz auto (TTC)", variable=v_fz_auto,
                    command=lambda: (sync_auto_fz(force=True), schedule())).grid(row=1, column=4, columnspan=2, sticky="w")
    slider(0, "γ [°]", v_gamma, -6, 6, "g", "{:+.1f}°")
    slider(1, "κ (Fy/Mz)", v_kappa, -0.3, 0.3, "k", "{:+.3f}")
    slider(2, "α (Fx) [°]", v_alpha, -12, 12, "a", "{:+.1f}°")
    slider(3, "p [kPa]", v_p, 50, 130, "p", "{:.0f}")
    ttk.Checkbutton(ctrl, text="p nominal de cada neumático", variable=v_pauto, command=schedule).grid(row=1, column=9, columnspan=3, sticky="w")
    ttk.Checkbutton(ctrl, text="Superponer datos TTC", variable=v_overlay, command=schedule).grid(row=1, column=6, columnspan=3, sticky="w")

    nb = ttk.Notebook(right)
    nb.pack(fill="both", expand=True, padx=6, pady=4)
    figs: dict[str, Figure] = {}
    canv: dict[str, FigureCanvasTkAgg] = {}
    for tab in TABS:
        fr = ttk.Frame(nb)
        nb.add(fr, text=tab)
        if tab == TABS[7]:
            top = ttk.Frame(fr, padding=4)
            top.pack(fill="x")
            ttk.Label(top, text="Base:").pack(side="left")
            cb_ref = ttk.Combobox(top, textvariable=v_ref, values=list(labels), state="readonly", width=34)
            cb_ref.pack(side="left", padx=4)
            ttk.Label(top, text="Comparado:").pack(side="left")
            cb_cmp = ttk.Combobox(top, textvariable=v_cmp, values=list(labels), state="readonly", width=34)
            cb_cmp.pack(side="left", padx=4)
            cb_ref.bind("<<ComboboxSelected>>", schedule)
            cb_cmp.bind("<<ComboboxSelected>>", schedule)
            ttk.Button(top, text='Auto 7" vs 8"', command=lambda: (pick_pair())).pack(side="left", padx=4)

            def export_csv():
                res = state["last_cmp"]
                if not res:
                    return messagebox.showinfo("Exportar", "No hay comparativa activa.")
                p = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="comparativa_llantas.csv")
                if p:
                    rows_to_csv(Path(p), res, store.by_slug[labels[v_ref.get()]], store.by_slug[labels[v_cmp.get()]])
                    v_status.set(f"Exportado {p}")

            ttk.Button(top, text="Exportar CSV", command=export_csv).pack(side="left")
            pw = ttk.PanedWindow(fr, orient="horizontal")
            pw.pack(fill="both", expand=True)
            tf = ttk.Frame(pw)
            pw.add(tf, weight=2)
            cols = ("fz", "metric", "ref", "cmp", "delta")
            tree = ttk.Treeview(tf, columns=cols, show="headings", height=22)
            for c, t, w in (("fz", "Fz [N]", 60), ("metric", "Métrica", 110), ("ref", "Base", 80), ("cmp", "Comparado", 80), ("delta", "Δ%", 70)):
                tree.heading(c, text=t)
                tree.column(c, width=w, anchor="e" if c != "metric" else "w")
            tree.tag_configure("pos", foreground="#1b7f6b")
            tree.tag_configure("neg", foreground="#c0392b")
            tree.pack(fill="both", expand=True)
            gf = ttk.Frame(pw)
            pw.add(gf, weight=4)
            figs[tab] = Figure(figsize=(7, 5), dpi=100)
            canv[tab] = FigureCanvasTkAgg(figs[tab], master=gf)
            NavigationToolbar2Tk(canv[tab], gf).update()
            canv[tab].get_tk_widget().pack(fill="both", expand=True)
            state["tree"] = tree
        else:
            figs[tab] = Figure(figsize=(8, 5), dpi=100)
            canv[tab] = FigureCanvasTkAgg(figs[tab], master=fr)
            NavigationToolbar2Tk(canv[tab], fr).update()
            canv[tab].get_tk_widget().pack(fill="both", expand=True)
    ttk.Label(right, textvariable=v_status, relief="sunken", anchor="w").pack(fill="x", side="bottom")

    def controls() -> Controls:
        return Controls(fz=parse_fz(v_fz.get(), [600.0, 1100.0, 1600.0]), gamma_deg=round(v_gamma.get(), 1),
                        p_kpa=None if v_pauto.get() else round(v_p.get()), kappa=round(v_kappa.get(), 3),
                        alpha_deg=round(v_alpha.get(), 1), overlay=v_overlay.get(),
                        ref=labels.get(v_ref.get()), cmp=labels.get(v_cmp.get()))

    def redraw():
        state["after"] = None
        sync_auto_fz()  # Actualiza Fz si v_fz_auto está activo
        tab = nb.tab(nb.select(), "text")
        ctl = controls()
        slugs = [s for s, v in checks.items() if v.get()]
        t0 = time.perf_counter()
        v_status.set("Calculando…")
        root.update_idletasks()
        try:
            engine.draw(figs[tab], tab, slugs, ctl)
            if tab == TABS[7]:
                res = engine.compare(ctl)
                state["last_cmp"] = res
                tree = state["tree"]
                tree.delete(*tree.get_children())
                for r in (res or {}).get("rows", []):
                    tree.insert("", "end", values=(f"{r['fz']:.0f}", r["label"], f"{r['ref']:.4g}", f"{r['cmp']:.4g}",
                                                   f"{r['delta_pct']:+.2f}"), tags=("pos" if r["delta_pct"] >= 0 else "neg",))
            canv[tab].draw_idle()
            v_status.set(f"{tab}: {len(slugs)} neumático(s), {len(ctl.fz)} carga(s) — {1e3 * (time.perf_counter() - t0):.0f} ms")
        except Exception as exc:  # noqa: BLE001 - la GUI no debe morir por un fallo de dibujo
            LOG.exception("Fallo al dibujar %s", tab)
            v_status.set(f"Error en {tab}: {exc}")

    def save_png():
        tab = nb.tab(nb.select(), "text")
        p = filedialog.asksaveasfilename(defaultextension=".png", initialfile=f"{_safe(tab)}.png")
        if p:
            figs[tab].savefig(p, dpi=200)
            v_status.set(f"Figura guardada en {p}")

    def open_index():
        p = filedialog.askopenfilename(filetypes=[("index.json", "*.json")])
        if p:
            root.destroy()
            run_gui(Store.from_index(Path(p)), Path(p))

    menu = tk.Menu(root)
    fm = tk.Menu(menu, tearoff=0)
    fm.add_command(label="Abrir index.json…", accelerator="Ctrl+O", command=open_index)
    fm.add_command(label="Guardar figura PNG…", accelerator="Ctrl+S", command=save_png)
    fm.add_separator()
    fm.add_command(label="Salir", command=root.destroy)
    menu.add_cascade(label="Archivo", menu=fm)
    root.config(menu=menu)
    root.bind("<Control-s>", lambda _e: save_png())
    root.bind("<Control-o>", lambda _e: open_index())
    nb.bind("<<NotebookTabChanged>>", schedule)
    sync_auto_fz(force=True)
    schedule()
    root.mainloop()


# --------------------------------------------------------------------------------------------
def selftest(store: Store, out: Path) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    out.mkdir(parents=True, exist_ok=True)
    eng = Engine(store)
    ref, cmp_ = store.pair_7_8()
    ctl = Controls(ref=ref, cmp=cmp_, kappa=0.0)
    slugs = [e.slug for e in store.entries]
    for tab in TABS:
        fig = Figure(figsize=(10, 6))
        FigureCanvasAgg(fig)
        eng.draw(fig, tab, slugs, ctl)
        fig.savefig(out / f"{_safe(tab)}.png", dpi=90)
        print("OK", tab)
    res = eng.compare(ctl)
    if res:
        rows_to_csv(out / "comparativa.csv", res, store.by_slug[ref], store.by_slug[cmp_])
        print({k: round(v, 2) for k, v in res["mean_delta_pct"].items()})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Terminal visual de neumáticos MF6.1")
    ap.add_argument("--index", type=Path, default=Path("data/processed/mf61_fits/index.json"))
    ap.add_argument("--demo", action="store_true", help="neumáticos sintéticos (sin datos TTC)")
    ap.add_argument("--selftest", type=Path, metavar="DIR", help="renderiza todas las pestañas a PNG y sale")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    store = Store.demo() if a.demo else Store.from_index(a.index)
    if a.selftest:
        selftest(store, a.selftest)
        return 0
    run_gui(store, None if a.demo else a.index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())