#!/usr/bin/env python3
"""Terminal visual de análisis y comparación de neumáticos MF6.1 (TeR-Twin) — dark edition.

Pestañas:
  1. Fy–α                  (Fuerza lateral pura y combinada con scatter TTC)
  2. Mz–α                  (Momento autoalineante con scatter TTC)
  3. Fx–κ                  (Fuerza longitudinal pura vs combinada con scatter TTC)
  4. Traza t(α)            (Brazo neumático t en mm)
  5. Envolvente            (Elipse Fy-Fx y Elipse normalizada μy-μx con contracción por carga)
  6. Diagrama Gough        (Carpet plot Mz vs Fy con iso-líneas de α)
  7. Interacción comb.     (Degradación Fy(κ) y Fx(α) para control y TV)
  8. Sens. carga           (Cα, μy, μx y t0 vs Fz)
  9. Sens. presión         (Sensibilidad a presión p: rigideces y μ pico)
 10. Rigideces y degres.   (Cα, Cα/Fz, Cκ y ratio Cκ/Cα)
 11. Camber                (Efecto de caída γ: Fy pico, Cα y camber thrust)
 12. Comparativa 7" vs 8"  (Tabla comparativa Δ% y gráficos base vs comparada)
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

import matplotlib  # noqa: E402
matplotlib.use("TkAgg")
matplotlib.rcParams.update({
    "figure.facecolor":  "#0d1117",
    "axes.facecolor":    "#161b22",
    "axes.edgecolor":    "#30363d",
    "axes.labelcolor":   "#c9d1d9",
    "axes.titlecolor":   "#e6edf3",
    "xtick.color":       "#8b949e",
    "ytick.color":       "#8b949e",
    "text.color":        "#c9d1d9",
    "grid.color":        "#21262d",
    "grid.linewidth":    0.6,
    "legend.facecolor":  "#161b22",
    "legend.edgecolor":  "#30363d",
    "legend.fontsize":   7,
    "lines.linewidth":   1.6,
    "font.family":       "sans-serif",
    "font.size":         9,
})

from matplotlib.figure import Figure  # noqa: E402
import tkinter as tk  # noqa: E402
from tkinter import filedialog, messagebox, ttk  # noqa: E402
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk  # noqa: E402

from ter_twin.models.tires import pacejka_61 as mf  # noqa: E402

LOG = logging.getLogger("tire_terminal")
TABS = (
    "Fy–α",
    "Mz–α",
    "Fx–κ",
    "Traza t(α)",
    "Envolvente",
    "Diagrama Gough",
    "Interacción comb.",
    "Sens. carga",
    "Sens. presión",
    "Rigideces y degres.",
    "Camber",
    'Comparativa 7" vs 8"',
)
LINESTYLES = ("-", "--", "-.", ":")
CMP_KEYS = ("c_alpha_N_deg", "mu_y_peak", "mu_x_peak", "fy_peak_N", "mz_peak_Nm", "trail0_mm", "c_kappa_N")

_ACCENT  = "#58a6ff"   # blue
_SUCCESS = "#3fb950"   # green
_WARN    = "#d29922"   # amber
_DANGER  = "#f85149"   # red
_PURPLE  = "#bc8cff"   # violet
_BG      = "#0d1117"
_BG2     = "#161b22"
_BG3     = "#21262d"
_BORDER  = "#30363d"
_FG      = "#c9d1d9"
_FG2     = "#8b949e"


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
        d = self.data()
        if d is not None and "fz" in d:
            fz = d["fz"]
            valid = fz[np.isfinite(fz) & (fz > 80.0)]
            if valid.size > 0:
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
        # Paleta extendida de alto contraste asignada por cada combinación neumático + llanta (slug)
        _palette = [
            "#58a6ff",  # Azul eléctrico
            "#f78166",  # Naranja coral
            "#3fb950",  # Verde lima
            "#bc8cff",  # Púrpura brillante
            "#d29922",  # Ámbar / Dorado
            "#79c0ff",  # Celeste cielo
            "#ff7b72",  # Rojo salmón
            "#39d353",  # Verde menta
            "#f0883e",  # Naranja intenso
            "#d2a8ff",  # Lavanda
            "#56d364",  # Verde esmeralda
            "#db61a2",  # Rosa magenta
            "#e3b341",  # Amarillo cálido
            "#a5d6ff",  # Azul hielo
        ]
        self.color = {e.slug: _palette[i % len(_palette)] for i, e in enumerate(entries)}

    @classmethod
    def from_index(cls, index_path: Path) -> "Store":
        doc = json.loads(index_path.read_text(encoding="utf-8"))
        base = index_path.parent
        out = []
        for t in doc["tires"]:
            if t.get("status") != "ok":
                continue
            out.append(Entry(
                t["slug"], t["tire_name"], float(t["rim_width_in"]),
                base / t["params"],
                base / t["data"] if t.get("data") else None,
                None, t.get("summary", {}), t.get("r2", {}),
            ))
        if not out:
            raise SystemExit(f"{index_path}: sin neumáticos ajustados")
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


class Engine:
    def __init__(self, store: Store):
        self.store = store

    def style(self, e: Entry) -> tuple[str, str]:
        idx = self.store.rims.index(e.rim) if e.rim in self.store.rims else 0
        return self.store.color[e.slug], LINESTYLES[idx % len(LINESTYLES)]

    @staticmethod
    def _p(e: Entry, ctl: Controls) -> float:
        return ctl.p_kpa if ctl.p_kpa is not None else e.params.NOMPRES

    def _eval(self, e: Entry, ctl: Controls, alpha, kappa, fz, gamma=None) -> mf.MF61Result:
        g = math.radians(ctl.gamma_deg) if gamma is None else gamma
        r = mf.evaluate(e.params, alpha, kappa, g, fz, self._p(e, ctl))
        return mf.MF61Result(*(np.asarray(x) for x in r))

    def _meas(self, e: Entry, ctl: Controls, fz: float, kinds,
              xk: str, yk: str, extra=None, max_pts: int = 3000):
        d = e.data()
        if d is None or not ctl.overlay:
            return None
        m = (np.isin(d["kind"], kinds)
             & (np.abs(d["fz"] - fz) < 85.0)
             & (np.abs(np.degrees(d["gamma"]) - ctl.gamma_deg) < 1.0)
             & (np.abs(d["p"] - self._p(e, ctl)) < 15.0))
        if extra is not None and 0 not in kinds:
            m &= extra(d)
        idx = np.flatnonzero(m & np.isfinite(d[yk]))
        if idx.size > max_pts:
            idx = idx[np.linspace(0, idx.size - 1, max_pts).astype(int)]
        return (d[xk][idx], d[yk][idx]) if idx.size else None

    def _meas_split(self, e, ctl, fz, xk, yk, extra=None, max_pts=3000):
        d = e.data()
        if d is None or not ctl.overlay:
            return None, None
        base = (np.isfinite(d[yk])
                & (np.abs(d["fz"] - fz) < 85.0)
                & (np.abs(np.degrees(d["gamma"]) - ctl.gamma_deg) < 1.0)
                & (np.abs(d["p"] - self._p(e, ctl)) < 15.0))
        if extra is not None:
            base &= extra(d)
        def _pick(kind_val):
            idx = np.flatnonzero(base & (d["kind"] == kind_val))
            if idx.size > max_pts:
                idx = idx[np.linspace(0, idx.size - 1, max_pts).astype(int)]
            return (d[xk][idx], d[yk][idx]) if idx.size else None
        return _pick(1), _pick(2)

    @staticmethod
    def _decorate(ax, xl, yl, title=None, legend=True):
        ax.set_xlabel(xl, color=_FG2)
        ax.set_ylabel(yl, color=_FG2)
        ax.grid(True, alpha=0.25, color=_BG3)
        ax.axhline(0, color=_BORDER, lw=0.7)
        ax.axvline(0, color=_BORDER, lw=0.7)
        if title:
            ax.set_title(title, fontsize=9, color=_FG, pad=6)
        if legend and ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7, ncol=2, loc="best",
                      facecolor=_BG2, edgecolor=_BORDER, labelcolor=_FG)

    def _empty(self, fig: Figure, msg="Selecciona al menos un neumático"):
        fig.clear()
        ax = fig.add_subplot(111)
        ax.set_facecolor(_BG2)
        ax.axis("off")
        ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=12, color=_FG2)

    # 1. Fy-α
    def fy_alpha(self, fig, slugs, ctl):
        fig.clear(); ax = fig.add_subplot(111)
        a = np.radians(np.linspace(-14, 14, 281))
        n = len(ctl.fz)
        for s in slugs:
            e = self.store.by_slug[s]; col, ls = self.style(e)
            for j, fz in enumerate(ctl.fz):
                r = self._eval(e, ctl, a, ctl.kappa, fz)
                ax.plot(np.degrees(a), r.fy, color=col, ls=ls,
                        alpha=0.4 + 0.6 * (j + 1) / n, lw=1.6,
                        label=f"{e.label} @ {fz:.0f} N")
                m = self._meas(e, ctl, fz, (0, 2), "alpha", "fy",
                               extra=lambda d: np.abs(d["kappa"] - ctl.kappa) < 0.01)
                if m is not None:
                    ax.scatter(np.degrees(m[0]), m[1], s=10, color=col,
                               alpha=0.35, rasterized=True, zorder=2)
        self._decorate(ax, "α [°]", "Fy [N]",
                       f"Fuerza lateral — γ={ctl.gamma_deg:.1f}°, κ={ctl.kappa:.3f}")

    # 2. Mz-α
    def mz_alpha(self, fig, slugs, ctl):
        fig.clear(); ax = fig.add_subplot(111)
        a = np.radians(np.linspace(-14, 14, 281))
        n = len(ctl.fz)
        for s in slugs:
            e = self.store.by_slug[s]; col, ls = self.style(e)
            for j, fz in enumerate(ctl.fz):
                r = self._eval(e, ctl, a, ctl.kappa, fz)
                ax.plot(np.degrees(a), r.mz, color=col, ls=ls,
                        alpha=0.4 + 0.6 * (j + 1) / n, lw=1.6,
                        label=f"{e.label} @ {fz:.0f} N")
                m = self._meas(e, ctl, fz, (0, 2), "alpha", "mz",
                               extra=lambda d: np.abs(d["kappa"] - ctl.kappa) < 0.01)
                if m is not None:
                    ax.scatter(np.degrees(m[0]), m[1], s=10, color=col,
                               alpha=0.35, rasterized=True, zorder=2)
        self._decorate(ax, "α [°]", "Mz [N·m]", "Momento autoalineante")

    # 3. Fx-κ
    def fx_kappa(self, fig, slugs, ctl):
        fig.clear(); ax = fig.add_subplot(111)
        k_arr = np.linspace(-0.3, 0.3, 301)
        a0 = math.radians(ctl.alpha_deg)
        n = len(ctl.fz)
        _first_legend = True
        for s in slugs:
            e = self.store.by_slug[s]; col, ls = self.style(e)
            for j, fz in enumerate(ctl.fz):
                r = self._eval(e, ctl, a0, k_arr, fz)
                ax.plot(k_arr * 100, r.fx, color=col, ls=ls,
                        alpha=0.4 + 0.6 * (j + 1) / n, lw=1.6,
                        label=f"{e.label} @ {fz:.0f} N")
                extra_fn = lambda d: np.abs(d["alpha"] - a0) < math.radians(0.5)
                pure, comb = self._meas_split(e, ctl, fz, "kappa", "fx", extra=extra_fn)
                if pure is not None:
                    ax.scatter(pure[0] * 100, pure[1], s=16, color=col,
                               marker="o", alpha=0.50, rasterized=True, zorder=3,
                               label="medido (puro)" if _first_legend else None)
                    _first_legend = False
                if comb is not None:
                    ax.scatter(comb[0] * 100, comb[1], s=20, color=col,
                               marker="D", alpha=0.40, rasterized=True, zorder=3,
                               label="medido (comb.)" if _first_legend else None)
                    _first_legend = False
        from matplotlib.lines import Line2D
        proxies = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=_FG2,
                   markersize=6, ls="none", label="puro (kind=1)"),
            Line2D([0], [0], marker="D", color="w", markerfacecolor=_FG2,
                   markersize=5, ls="none", label="combinado (kind=2)"),
        ]
        self._decorate(ax, "κ [%]", "Fx [N]",
                       f"Fuerza longitudinal — α={ctl.alpha_deg:.1f}°, γ={ctl.gamma_deg:.1f}°")
        h, l = ax.get_legend_handles_labels()
        ax.legend(handles=h + proxies, labels=l + ["puro (kind=1)", "combinado (kind=2)"],
                  fontsize=7, loc="best", facecolor=_BG2, edgecolor=_BORDER, labelcolor=_FG, ncol=2)

    # 4. Traza t(α)
    def trail(self, fig, slugs, ctl):
        fig.clear(); ax = fig.add_subplot(111)
        a = np.radians(np.linspace(-14, 14, 281))
        n = len(ctl.fz)
        for s in slugs:
            e = self.store.by_slug[s]; col, ls = self.style(e)
            for j, fz in enumerate(ctl.fz):
                r = self._eval(e, ctl, a, 0.0, fz)
                ax.plot(np.degrees(a), np.asarray(r.t) * 1e3, color=col, ls=ls,
                        alpha=0.4 + 0.6 * (j + 1) / n, lw=1.6,
                        label=f"{e.label} @ {fz:.0f} N")
        self._decorate(ax, "α [°]", "Traza t [mm]", "Traza neumática")

    # 5. Envolvente (Fuerzas absolutas + Elipse normalizada μ)
    def envelope(self, fig, slugs, ctl):
        fig.clear()
        ax_f, ax_mu = fig.subplots(1, 2)
        fz0 = ctl.fz[0]
        n_fz = len(ctl.fz)

        for s in slugs:
            e = self.store.by_slug[s]; col, ls = self.style(e)
            fx_g, fy_g, al, ka = mf.friction_envelope(e.params, fz0,
                                                      math.radians(ctl.gamma_deg),
                                                      self._p(e, ctl))
            for i in range(0, len(al), 6):
                ax_f.plot(fy_g[i], fx_g[i], color=col, lw=0.9, ls=ls, alpha=0.8,
                          label=e.label if i == 0 else None)
            for j in range(0, len(ka), 6):
                ax_f.plot(fy_g[:, j], fx_g[:, j], color=col, lw=0.5, ls=":", alpha=0.45)

            d = e.data()
            if d is not None and ctl.overlay and "kind" in d:
                mask = ((d["kind"] == 2)
                        & (np.abs(d["fz"] - fz0) < 85.0)
                        & (np.abs(np.degrees(d["gamma"]) - ctl.gamma_deg) < 1.0)
                        & (np.abs(d["p"] - self._p(e, ctl)) < 15.0)
                        & np.isfinite(d["fx"]) & np.isfinite(d["fy"]))
                idx = np.flatnonzero(mask)
                if idx.size > 2000:
                    idx = idx[np.linspace(0, idx.size - 1, 2000).astype(int)]
                if idx.size:
                    ax_f.scatter(d["fy"][idx], d["fx"][idx], s=8, color=col,
                                 marker="D", alpha=0.35, rasterized=True, zorder=4,
                                 label=f"{e.label} (medido)")

            for k, fz_k in enumerate(ctl.fz):
                fx_k, fy_k, _, _ = mf.friction_envelope(e.params, fz_k,
                                                        math.radians(ctl.gamma_deg),
                                                        self._p(e, ctl))
                bx = np.concatenate([fx_k[-1, :], fx_k[:, -1], fx_k[0, ::-1], fx_k[::-1, 0]]) / fz_k
                by = np.concatenate([fy_k[-1, :], fy_k[:, -1], fy_k[0, ::-1], fy_k[::-1, 0]]) / fz_k
                alpha_k = 0.4 + 0.6 * (k + 1) / n_fz
                ax_mu.plot(by, bx, color=col, ls=LINESTYLES[k % len(LINESTYLES)], lw=1.5,
                           alpha=alpha_k, label=f"{e.label} @ {fz_k:.0f} N")

        ax_f.set_aspect("equal", adjustable="datalim")
        ax_mu.set_aspect("equal", adjustable="datalim")
        self._decorate(ax_f, "Fy [N]", "Fx [N]", f"Fuerzas combinadas (Fy, Fx) @ Fz={fz0:.0f} N")
        self._decorate(ax_mu, "μy = Fy/Fz [-]", "μx = Fx/Fz [-]", "Elipse normalizada μ (Sensibilidad a carga)")
        fig.suptitle(f"Envolvente de Adherencia — γ={ctl.gamma_deg:.1f}°", fontsize=10, color=_FG)
        fig.tight_layout(rect=(0, 0, 1, 0.94))

    # 6. Diagrama de Gough (Carpet plot Mz vs Fy)
    def gough(self, fig, slugs, ctl):
        fig.clear(); ax = fig.add_subplot(111)
        a_deg = np.linspace(-14, 14, 281)
        a_rad = np.radians(a_deg)
        iso_alphas = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
        n = len(ctl.fz)

        for s in slugs:
            e = self.store.by_slug[s]; col, ls = self.style(e)
            carpet_pts = {al: [] for al in iso_alphas}
            for j, fz in enumerate(ctl.fz):
                r = self._eval(e, ctl, a_rad, ctl.kappa, fz)
                alpha_line = 0.4 + 0.6 * (j + 1) / n
                ax.plot(r.fy, r.mz, color=col, ls=ls, lw=1.6, alpha=alpha_line,
                        label=f"{e.label} @ {fz:.0f} N")
                for al in iso_alphas:
                    idx = np.argmin(np.abs(a_deg - al))
                    carpet_pts[al].append((r.fy[idx], r.mz[idx]))

                m = self._meas(e, ctl, fz, (0, 2), "fy", "mz",
                               extra=lambda d: np.abs(d["kappa"] - ctl.kappa) < 0.01)
                if m is not None:
                    ax.scatter(m[0], m[1], s=10, color=col, alpha=0.30, rasterized=True, zorder=2)

            if len(ctl.fz) > 1:
                for al in iso_alphas:
                    pts = carpet_pts[al]
                    fys = [p[0] for p in pts]
                    mzs = [p[1] for p in pts]
                    ax.plot(fys, mzs, color=_FG2, ls=":", lw=0.8, alpha=0.5)
                    ax.text(fys[-1], mzs[-1], f" {al:g}°", fontsize=7, color=_FG2, va="center")

        self._decorate(ax, "Fy [N]", "Mz [N·m]",
                       f"Diagrama de Gough — Mz(Fy) · γ={ctl.gamma_deg:.1f}°, κ={ctl.kappa:.3f}")

    # 7. Interacción combinada: Fy(κ) y Fx(α)
    def combined_interaction(self, fig, slugs, ctl):
        fig.clear()
        ax1, ax2 = fig.subplots(1, 2)
        fz0 = ctl.fz[0]
        k_arr = np.linspace(-0.3, 0.3, 241)
        a_deg_arr = np.linspace(0.0, 14.0, 241)
        a_rad_arr = np.radians(a_deg_arr)
        iso_alphas = [2.0, 4.0, 6.0, 8.0, 12.0]
        iso_kappas = [0.03, 0.07, 0.12, 0.20, 0.28]

        for s in slugs:
            e = self.store.by_slug[s]; col, ls = self.style(e)
            for i, al in enumerate(iso_alphas):
                al_rad = math.radians(al)
                r = self._eval(e, ctl, al_rad, k_arr, fz0)
                alpha_val = 0.35 + 0.65 * (i + 1) / len(iso_alphas)
                ax1.plot(k_arr * 100, r.fy, color=col, ls=ls, lw=1.5, alpha=alpha_val,
                         label=f"{e.label} α={al:g}°" if (len(slugs) > 1 and i == 2) or len(slugs) == 1 else None)

            for i, kp in enumerate(iso_kappas):
                r = self._eval(e, ctl, a_rad_arr, kp, fz0)
                alpha_val = 0.35 + 0.65 * (i + 1) / len(iso_kappas)
                ax2.plot(a_deg_arr, r.fx, color=col, ls=ls, lw=1.5, alpha=alpha_val,
                         label=f"{e.label} κ={kp:.2f}" if (len(slugs) > 1 and i == 2) or len(slugs) == 1 else None)

            d = e.data()
            if d is not None and ctl.overlay and "kind" in d:
                mask = ((d["kind"] == 2)
                        & (np.abs(d["fz"] - fz0) < 85.0)
                        & (np.abs(np.degrees(d["gamma"]) - ctl.gamma_deg) < 1.0)
                        & (np.abs(d["p"] - self._p(e, ctl)) < 15.0)
                        & np.isfinite(d["fx"]) & np.isfinite(d["fy"]))
                idx = np.flatnonzero(mask)
                if idx.size > 1500:
                    idx = idx[np.linspace(0, idx.size - 1, 1500).astype(int)]
                if idx.size:
                    ax1.scatter(d["kappa"][idx] * 100, d["fy"][idx], s=9, color=col, marker="D",
                                alpha=0.25, rasterized=True)
                    ax2.scatter(np.degrees(np.abs(d["alpha"][idx])), np.abs(d["fx"][idx]), s=9, color=col, marker="D",
                                alpha=0.25, rasterized=True)

        self._decorate(ax1, "κ [%]", "Fy [N]", "Degradación lateral por tracción/frenada Fy(κ)")
        self._decorate(ax2, "α [°]", "Fx [N]", "Degradación longitudinal por viraje Fx(α)")
        fig.suptitle(f"Interacción de Deslizamiento Combinado @ Fz={fz0:.0f} N, γ={ctl.gamma_deg:.1f}°",
                     fontsize=10, color=_FG)
        fig.tight_layout(rect=(0, 0, 1, 0.94))

    # 8. Sensibilidad a carga
    def load_sens(self, fig, slugs, ctl):
        fig.clear()
        fzs = np.linspace(100, 2200, 36)
        axs = fig.subplots(2, 2).ravel()
        spec = (("c_alpha_N_deg", "Cα [N/°]"), ("mu_y_peak", "μy pico [-]"),
                ("mu_x_peak", "μx pico [-]"), ("trail0_mm", "t₀ [mm]"))
        for s in slugs:
            e = self.store.by_slug[s]; col, ls = self.style(e)
            out = mf.summary(e.params, fzs, math.radians(ctl.gamma_deg), self._p(e, ctl))
            for ax, (k, yl) in zip(axs, spec):
                ax.plot(fzs, out[k], color=col, ls=ls, lw=1.6, label=e.label)
                ax.set_ylabel(yl, color=_FG2)
        for ax in axs:
            ax.set_xlabel("Fz [N]", color=_FG2)
            ax.grid(True, alpha=0.25)
            for fz in ctl.fz:
                ax.axvline(fz, color=_BORDER, lw=0.6, ls=":")
        axs[0].legend(fontsize=7, facecolor=_BG2, edgecolor=_BORDER, labelcolor=_FG)
        fig.suptitle(f"Sensibilidad a la carga — γ={ctl.gamma_deg:.1f}°", fontsize=10, color=_FG)
        fig.tight_layout(rect=(0, 0, 1, 0.95))

    # 9. Sensibilidad a presión
    def pressure_sens(self, fig, slugs, ctl):
        fig.clear()
        axs = fig.subplots(2, 2).ravel()
        fz0 = ctl.fz[0]
        ps = np.linspace(50.0, 130.0, 21)
        g_rad = math.radians(ctl.gamma_deg)
        a_deg = np.linspace(-14, 14, 241)
        a_rad = np.radians(a_deg)
        test_pressures = [60.0, 83.0, 110.0]

        for s in slugs:
            e = self.store.by_slug[s]; col, ls = self.style(e)
            ca_list, muy_list, mux_list = [], [], []
            for p_val in ps:
                res_p = mf.summary(e.params, [fz0], g_rad, float(p_val))
                ca_list.append(float(res_p["c_alpha_N_deg"][0]))
                muy_list.append(float(res_p["mu_y_peak"][0]))
                mux_list.append(float(res_p["mu_x_peak"][0]))

            axs[0].plot(ps, ca_list, color=col, ls=ls, lw=1.6, label=e.label)
            axs[1].plot(ps, muy_list, color=col, ls=ls, lw=1.6, label=e.label)
            axs[2].plot(ps, mux_list, color=col, ls=ls, lw=1.6, label=e.label)

            for k, p_val in enumerate(test_pressures):
                r = mf.evaluate(e.params, a_rad, 0.0, g_rad, fz0, float(p_val))
                p_alpha = 0.4 + 0.6 * (k + 1) / len(test_pressures)
                axs[3].plot(a_deg, np.asarray(r.fy), color=col, ls=LINESTYLES[k % len(LINESTYLES)],
                            lw=1.5, alpha=p_alpha,
                            label=f"{e.label} @ {p_val:.0f} kPa" if len(slugs) == 1 else (e.label if k == 1 else None))

        spec = (
            ("Cα [N/°]", "Rigidez de deriva vs Presión"),
            ("μy pico [-]", "Coeficiente lateral pico vs Presión"),
            ("μx pico [-]", "Coeficiente longitudinal pico vs Presión"),
        )
        for ax, (yl, tit) in zip(axs[:3], spec):
            ax.set_xlabel("Presión [kPa]", color=_FG2); ax.set_ylabel(yl, color=_FG2)
            ax.set_title(tit, fontsize=8, color=_FG); ax.grid(True, alpha=0.25)
            ax.axvline(83.0, color=_BORDER, ls="--", lw=0.8, alpha=0.7)

        axs[3].set_xlabel("α [°]", color=_FG2); axs[3].set_ylabel("Fy [N]", color=_FG2)
        axs[3].set_title("Fy(α) @ 60, 83, 110 kPa", fontsize=8, color=_FG); axs[3].grid(True, alpha=0.25)
        axs[3].axhline(0, color=_BORDER, lw=0.7); axs[3].axvline(0, color=_BORDER, lw=0.7)

        axs[0].legend(fontsize=7, facecolor=_BG2, edgecolor=_BORDER, labelcolor=_FG)
        if axs[3].get_legend_handles_labels()[0]:
            axs[3].legend(fontsize=6.5, facecolor=_BG2, edgecolor=_BORDER, labelcolor=_FG)

        fig.suptitle(f"Sensibilidad a la Presión de Inflado @ Fz={fz0:.0f} N, γ={ctl.gamma_deg:.1f}°",
                     fontsize=10, color=_FG)
        fig.tight_layout(rect=(0, 0, 1, 0.94))

    # 10. Rigideces y degresividad
    def stiffness_degressivity(self, fig, slugs, ctl):
        fig.clear()
        axs = fig.subplots(2, 2).ravel()
        fzs = np.linspace(150, 2200, 42)
        g_rad = math.radians(ctl.gamma_deg)

        for s in slugs:
            e = self.store.by_slug[s]; col, ls = self.style(e)
            out = mf.summary(e.params, fzs, g_rad, self._p(e, ctl))
            ca = out["c_alpha_N_deg"]
            ca_norm = ca / fzs
            ck = out["c_kappa_N"]
            ck_ca = ck / np.maximum(ca * (180.0 / math.pi), 1e-3)

            axs[0].plot(fzs, ca, color=col, ls=ls, lw=1.6, label=e.label)
            axs[1].plot(fzs, ca_norm, color=col, ls=ls, lw=1.6, label=e.label)
            axs[2].plot(fzs, ck, color=col, ls=ls, lw=1.6, label=e.label)
            axs[3].plot(fzs, ck_ca, color=col, ls=ls, lw=1.6, label=e.label)

        titulos = (
            ("Cα [N/°]", "Rigidez de deriva absoluta Cα"),
            ("Cα / Fz [1/°]", "Rigidez normalizada Cα/Fz (Degresividad)"),
            ("Cκ [N]", "Rigidez de tracción Cκ"),
            ("Cκ / Cα [-]", "Ratio Cκ / Cα (Longitudinal/Lateral)"),
        )
        for ax, (yl, tit) in zip(axs, titulos):
            ax.set_xlabel("Fz [N]", color=_FG2); ax.set_ylabel(yl, color=_FG2)
            ax.set_title(tit, fontsize=8, color=_FG); ax.grid(True, alpha=0.25)
            for fz_val in ctl.fz:
                ax.axvline(fz_val, color=_BORDER, lw=0.6, ls=":")

        axs[0].legend(fontsize=7, facecolor=_BG2, edgecolor=_BORDER, labelcolor=_FG)
        fig.suptitle(f"Rigideces y Degresividad por Carga Normal — γ={ctl.gamma_deg:.1f}°",
                     fontsize=10, color=_FG)
        fig.tight_layout(rect=(0, 0, 1, 0.94))

    # 11. Camber
    def camber(self, fig, slugs, ctl):
        fig.clear()
        gs = np.radians(np.linspace(-6, 6, 13))
        axs = fig.subplots(1, 3)
        fz0 = ctl.fz[0]
        for s in slugs:
            e = self.store.by_slug[s]; col, ls = self.style(e)
            rows = [mf.summary(e.params, [fz0], g, self._p(e, ctl)) for g in gs]
            g_deg = np.degrees(gs)
            axs[0].plot(g_deg, [r["fy_peak_N"][0] for r in rows], color=col, ls=ls, lw=1.6, label=e.label)
            axs[1].plot(g_deg, [r["c_alpha_N_deg"][0] for r in rows], color=col, ls=ls, lw=1.6)
            axs[2].plot(g_deg, [float(self._eval(e, ctl, 0.0, 0.0, fz0, g).fy) for g in gs],
                        color=col, ls=ls, lw=1.6)
        for ax, yl in zip(axs, ("Fy pico [N]", "Cα [N/°]", "Empuje camber [N]")):
            ax.set_xlabel("γ [°]", color=_FG2); ax.set_ylabel(yl, color=_FG2); ax.grid(True, alpha=0.25)
        axs[0].legend(fontsize=7, facecolor=_BG2, edgecolor=_BORDER, labelcolor=_FG)
        fig.suptitle(f"Efecto de caída @ Fz={fz0:.0f} N", fontsize=10, color=_FG)
        fig.tight_layout(rect=(0, 0, 1, 0.93))

    # 12. Comparativa
    def compare(self, ctl: Controls) -> dict[str, Any] | None:
        if not ctl.ref or not ctl.cmp or ctl.ref == ctl.cmp:
            return None
        a, b = self.store.by_slug[ctl.ref], self.store.by_slug[ctl.cmp]
        return mf.compare_params(a.params, b.params, ctl.fz,
                                 math.radians(ctl.gamma_deg),
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
        ax_bar.barh(y, vals, color=[_SUCCESS if v >= 0 else _DANGER for v in vals])
        ax_bar.set_yticks(y, [mf.METRIC_LABELS[k] for k in keys], fontsize=8)
        ax_bar.invert_yaxis()
        for yi, v in zip(y, vals):
            ax_bar.text(v, yi, f" {v:+.1f}%", va="center",
                        ha="left" if v >= 0 else "right", fontsize=8, color=_FG)
        ax_bar.axvline(0, color=_BORDER, lw=0.8)
        ax_bar.set_title(f"Δ% ({b.label}) vs ({a.label})", fontsize=9, color=_FG)
        ax_bar.grid(True, axis="x", alpha=0.25)
        fzs = np.linspace(100, 2200, 36)
        g = math.radians(ctl.gamma_deg)
        for e, ls_ in ((a, "-"), (b, "--")):
            out = mf.summary(e.params, fzs, g, self._p(e, ctl))
            col = self.style(e)[0]
            ax_ca.plot(fzs, out["c_alpha_N_deg"], ls=ls_, color=col, lw=1.6, label=e.label)
            ax_mu.plot(fzs, out["mu_y_peak"], ls=ls_, color=col, lw=1.6, label=e.label)
        for ax in (ax_ca, ax_mu):
            ax.set_xlabel("Fz [N]", color=_FG2); ax.grid(True, alpha=0.25)
        ax_ca.set_ylabel("Cα [N/°]", color=_FG2); ax_mu.set_ylabel("μy pico [-]", color=_FG2)
        ax_ca.legend(fontsize=7, facecolor=_BG2, edgecolor=_BORDER, labelcolor=_FG)
        al = np.radians(np.linspace(-14, 14, 281)); fz0 = ctl.fz[0]
        for e, ls_ in ((a, "-"), (b, "--")):
            r = self._eval(e, ctl, al, 0.0, fz0)
            col = self.style(e)[0]
            ax_fy.plot(np.degrees(al), r.fy, ls=ls_, color=col, lw=1.6, label=e.label)
            m = self._meas(e, ctl, fz0, (0, 2), "alpha", "fy")
            if m is not None:
                ax_fy.scatter(np.degrees(m[0]), m[1], s=8, color=col, alpha=0.3, rasterized=True)
        ax_fy.set(xlabel="α [°]", ylabel="Fy [N]")
        ax_fy.set_title(f"Fy(α) @ {fz0:.0f} N", color=_FG); ax_fy.grid(True, alpha=0.25)
        fig.suptitle("Base vs Comparado", fontsize=10, color=_FG)
        fig.tight_layout(rect=(0, 0, 1, 0.95))

    def draw(self, fig: Figure, tab: str, slugs: list[str], ctl: Controls) -> None:
        if tab == TABS[11]:
            return self.draw_compare(fig, ctl)
        if not slugs:
            return self._empty(fig)
        dispatch = {
            TABS[0]: self.fy_alpha,
            TABS[1]: self.mz_alpha,
            TABS[2]: self.fx_kappa,
            TABS[3]: self.trail,
            TABS[4]: self.envelope,
            TABS[5]: self.gough,
            TABS[6]: self.combined_interaction,
            TABS[7]: self.load_sens,
            TABS[8]: self.pressure_sens,
            TABS[9]: self.stiffness_degressivity,
            TABS[10]: self.camber,
        }
        if tab in dispatch:
            dispatch[tab](fig, slugs, ctl)
        if tab in (TABS[0], TABS[1], TABS[2], TABS[3], TABS[5]):
            fig.tight_layout()


def rows_to_csv(path: Path, res: dict[str, Any], ref: Entry, cmp_: Entry) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["ref", "cmp", "gamma_deg", "fz_N", "metric", "ref_value", "cmp_value", "delta_pct"])
        for r in res["rows"]:
            w.writerow([ref.label, cmp_.label, res["gamma_deg"], r["fz"],
                        r["metric"], r["ref"], r["cmp"], r["delta_pct"]])


def _safe(name: str) -> str:
    return re.sub(r"\W+", "_", name).strip("_")


def merge_close_levels(levels: list[float], tol: float = 60.0) -> list[float]:
    if not levels:
        return []
    s = sorted(levels); merged = []; curr = [s[0]]
    for x in s[1:]:
        if x - curr[-1] <= tol:
            curr.append(x)
        else:
            merged.append(float(np.round(np.mean(curr)))); curr = [x]
    if curr:
        merged.append(float(np.round(np.mean(curr))))
    return merged


def parse_fz(text: str, default: list[float]) -> list[float]:
    try:
        v = [float(t) for t in re.split(r"[,\s;]+", text.strip()) if t]
    except ValueError:
        return default
    return [x for x in v if x > 0][:10] or default


def _apply_dark_theme(root: tk.Tk) -> ttk.Style:
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(".", background=_BG, foreground=_FG, fieldbackground=_BG2,
                    bordercolor=_BORDER, troughcolor=_BG3, selectbackground=_ACCENT,
                    selectforeground=_BG, font=("Segoe UI", 9))
    style.configure("TFrame", background=_BG)
    style.configure("TLabel", background=_BG, foreground=_FG)
    style.configure("TButton", background=_BG3, foreground=_FG,
                    bordercolor=_BORDER, focuscolor=_ACCENT, padding=(8, 4))
    style.map("TButton",
              background=[("active", _ACCENT), ("pressed", "#1f6feb")],
              foreground=[("active", _BG)])
    style.configure("TCheckbutton", background=_BG, foreground=_FG, indicatorbackground=_BG2,
                    indicatorcolor=_ACCENT)
    style.map("TCheckbutton", background=[("active", _BG3)])
    style.configure("TCombobox", fieldbackground=_BG2, background=_BG2, foreground=_FG,
                    arrowcolor=_FG2, selectbackground=_ACCENT, selectforeground=_BG)
    style.map("TCombobox", fieldbackground=[("readonly", _BG2)])
    style.configure("TEntry", fieldbackground=_BG2, foreground=_FG,
                    insertcolor=_FG, bordercolor=_BORDER)
    style.configure("TLabelframe", background=_BG, foreground=_FG2, bordercolor=_BORDER)
    style.configure("TLabelframe.Label", background=_BG, foreground=_ACCENT,
                    font=("Segoe UI", 9, "bold"))
    style.configure("TNotebook", background=_BG, tabmargins=0)
    style.configure("TNotebook.Tab", background=_BG3, foreground=_FG2, padding=(8, 4),
                    bordercolor=_BORDER)
    style.map("TNotebook.Tab",
              background=[("selected", _BG2)],
              foreground=[("selected", _ACCENT)])
    style.configure("TScale", background=_BG, troughcolor=_BG3, sliderthickness=14,
                    slidercolor=_ACCENT)
    style.configure("Treeview", background=_BG2, foreground=_FG, fieldbackground=_BG2,
                    rowheight=22, bordercolor=_BORDER)
    style.configure("Treeview.Heading", background=_BG3, foreground=_FG2, bordercolor=_BORDER)
    style.map("Treeview", background=[("selected", _ACCENT)], foreground=[("selected", _BG)])
    style.configure("TScrollbar", background=_BG3, troughcolor=_BG,
                    arrowcolor=_FG2, bordercolor=_BORDER)
    style.configure("Status.TLabel", background=_BG3, foreground=_FG2,
                    relief="flat", padding=(6, 2))
    return style


def run_gui(store: Store, index_path: Path | None) -> None:
    engine = Engine(store)
    root = tk.Tk()
    root.title("TeR-Twin · Terminal de neumáticos MF6.1 (12 Pestañas)")
    root.geometry("1600x950")
    root.configure(bg=_BG)
    root.option_add("*Background", _BG)
    root.option_add("*Foreground", _FG)

    _apply_dark_theme(root)

    checks: dict[str, tk.BooleanVar] = {}
    for i, e in enumerate(store.entries):
        checks[e.slug] = tk.BooleanVar(value=i < 2)

    v_fz      = tk.StringVar(value="445, 667, 1112")
    v_fz_auto = tk.BooleanVar(value=True)
    v_gamma   = tk.DoubleVar(value=0.0)
    v_kappa   = tk.DoubleVar(value=0.0)
    v_alpha   = tk.DoubleVar(value=0.0)
    v_p       = tk.DoubleVar(value=83.0)
    v_pauto   = tk.BooleanVar(value=True)
    v_overlay = tk.BooleanVar(value=True)
    v_ref     = tk.StringVar()
    v_cmp     = tk.StringVar()
    v_status  = tk.StringVar(value="Listo")

    labels = {e.label: e.slug for e in store.entries}
    ref0, cmp0 = store.pair_7_8()
    if ref0:
        v_ref.set(store.by_slug[ref0].label)
        v_cmp.set(store.by_slug[cmp0].label)
        checks[ref0].set(True)
        checks[cmp0].set(True)

    state: dict[str, Any] = {"after": None, "last_cmp": None}

    pane = tk.PanedWindow(root, orient=tk.HORIZONTAL, bg=_BG, sashwidth=4, sashpad=0, handlesize=0)
    pane.pack(fill="both", expand=True)

    left = ttk.Frame(pane, padding=8)
    right = ttk.Frame(pane)
    pane.add(left, minsize=260, width=300)
    pane.add(right, minsize=800)

    ttk.Label(left, text="NEUMÁTICOS", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 6))

    cb_rim = ttk.Combobox(left, state="readonly", width=18,
                          values=["Todas las llantas"] + [f'{r:g}"' for r in store.rims])
    cb_rim.current(0)
    cb_rim.pack(anchor="w", pady=(0, 4))

    list_frame = ttk.Frame(left)
    list_frame.pack(fill="both", expand=True)
    canvas_l = tk.Canvas(list_frame, width=270, highlightthickness=0, bg=_BG)
    sb_l = ttk.Scrollbar(list_frame, orient="vertical", command=canvas_l.yview)
    inner = ttk.Frame(canvas_l)
    inner.bind("<Configure>", lambda _e: canvas_l.configure(scrollregion=canvas_l.bbox("all")))
    canvas_l.create_window((0, 0), window=inner, anchor="nw")
    canvas_l.configure(yscrollcommand=sb_l.set)
    canvas_l.pack(side="left", fill="both", expand=True)
    sb_l.pack(side="right", fill="y")

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
            row = tk.Frame(inner, bg=_BG)
            row.pack(anchor="w", fill="x", pady=2)
            # Cuadro de color con el tono único de este neumático + llanta
            swatch = tk.Label(row, bg=store.color[e.slug], width=2, height=1, relief="flat")
            swatch.pack(side="left", padx=(2, 6))
            # Checkbox nativo con casilla que se ilumina en azul eléctrico al marcarse
            cb = tk.Checkbutton(
                row,
                text=f"{e.label}  ({engine.style(e)[1]})",
                variable=checks[e.slug],
                command=schedule,
                bg=_BG,
                fg=_FG,
                selectcolor="#1f6feb",      # Relleno azul brillante cuando está marcado
                activebackground=_BG3,
                activeforeground="#ffffff",
                font=("Segoe UI", 9),
                cursor="hand2",
                highlightthickness=0,
                bd=0,
            )
            cb.pack(side="left")

    cb_rim.bind("<<ComboboxSelected>>", rebuild_list)
    rebuild_list()

    bf = ttk.Frame(left)
    bf.pack(fill="x", pady=(6, 0))

    def set_all(val):
        for v in checks.values():
            v.set(val)
        schedule()

    def pick_pair():
        a, b = store.pair_7_8()
        if not a:
            return messagebox.showinfo('Par 7"/8"', 'No hay un mismo Tire_Name con 7" y 8".')
        set_all(False)
        checks[a].set(True); checks[b].set(True)
        v_ref.set(store.by_slug[a].label); v_cmp.set(store.by_slug[b].label)
        schedule()

    ttk.Button(bf, text="Todos", command=lambda: set_all(True)).pack(side="left")
    ttk.Button(bf, text="Ninguno", command=lambda: set_all(False)).pack(side="left", padx=2)
    ttk.Button(bf, text='Par 7"/8"', command=pick_pair).pack(side="left")

    def sync_auto_fz(force: bool = False):
        if not v_fz_auto.get() and not force:
            return
        tab = nb.tab(nb.select(), "text") if nb.tabs() else ""
        if tab == TABS[11]:
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

    ctrl = ttk.LabelFrame(right, text="  Condiciones de ensayo  ", padding=(8, 6))
    ctrl.pack(fill="x", padx=6, pady=(6, 0))

    lab: dict[str, tk.StringVar] = {k: tk.StringVar() for k in "gkap"}

    def slider(col, row_n, text, var, lo, hi, key, fmt):
        ttk.Label(ctrl, text=text).grid(row=row_n, column=col * 3, sticky="e", padx=(8, 2))
        ttk.Scale(ctrl, from_=lo, to=hi, variable=var, length=130,
                  command=lambda _v: (lab[key].set(fmt.format(var.get())), schedule())
                  ).grid(row=row_n, column=col * 3 + 1, sticky="ew")
        lab[key].set(fmt.format(var.get()))
        ttk.Label(ctrl, textvariable=lab[key], width=8).grid(row=row_n, column=col * 3 + 2, sticky="w")

    slider(0, 0, "γ [°]",     v_gamma, -6,   6,   "g", "{:+.1f}°")
    slider(1, 0, "κ (Fy/Mz)", v_kappa, -0.3, 0.3, "k", "{:+.3f}")
    slider(2, 0, "α (Fx) [°]", v_alpha, -12, 12,  "a", "{:+.1f}°")
    slider(3, 0, "p [kPa]",   v_p,     50,  130,  "p", "{:.0f}")

    ttk.Label(ctrl, text="Fz [N]").grid(row=1, column=0, sticky="e", padx=(8, 2))
    e_fz = ttk.Entry(ctrl, textvariable=v_fz, width=22)
    e_fz.grid(row=1, column=1, columnspan=2, sticky="w")
    e_fz.bind("<Return>",   schedule)
    e_fz.bind("<FocusOut>", schedule)
    e_fz.bind("<Key>",      lambda _e: v_fz_auto.set(False))
    ttk.Button(ctrl, text="⟳ Auto Fz", width=10,
               command=lambda: (v_fz_auto.set(True), sync_auto_fz(True), schedule())
               ).grid(row=1, column=3, padx=(6, 2), sticky="w")
    ttk.Checkbutton(ctrl, text="Fz auto (TTC)", variable=v_fz_auto,
                    command=lambda: (sync_auto_fz(True), schedule())
                    ).grid(row=1, column=4, sticky="w")
    ttk.Checkbutton(ctrl, text="p nominal/neumático", variable=v_pauto,
                    command=schedule).grid(row=1, column=5, sticky="w", padx=(8, 0))
    ttk.Checkbutton(ctrl, text="Superponer datos TTC", variable=v_overlay,
                    command=schedule).grid(row=1, column=6, sticky="w", padx=(8, 0))

    nb = ttk.Notebook(right)
    nb.pack(fill="both", expand=True, padx=6, pady=4)
    figs: dict[str, Figure] = {}
    canv: dict[str, FigureCanvasTkAgg] = {}

    for tab in TABS:
        fr = ttk.Frame(nb)
        nb.add(fr, text=tab)
        if tab == TABS[11]:
            top = ttk.Frame(fr, padding=4)
            top.pack(fill="x")
            ttk.Label(top, text="Base:").pack(side="left")
            cb_ref = ttk.Combobox(top, textvariable=v_ref, values=list(labels), state="readonly", width=36)
            cb_ref.pack(side="left", padx=4)
            ttk.Label(top, text="Comparado:").pack(side="left")
            cb_cmp = ttk.Combobox(top, textvariable=v_cmp, values=list(labels), state="readonly", width=36)
            cb_cmp.pack(side="left", padx=4)
            cb_ref.bind("<<ComboboxSelected>>", schedule)
            cb_cmp.bind("<<ComboboxSelected>>", schedule)
            ttk.Button(top, text='Auto 7" vs 8"', command=pick_pair).pack(side="left", padx=4)

            def export_csv():
                res = state["last_cmp"]
                if not res:
                    return messagebox.showinfo("Exportar", "No hay comparativa activa.")
                p = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="comparativa_llantas.csv")
                if p:
                    rows_to_csv(Path(p), res,
                                store.by_slug[labels[v_ref.get()]],
                                store.by_slug[labels[v_cmp.get()]])
                    v_status.set(f"Exportado {p}")

            ttk.Button(top, text="Exportar CSV", command=export_csv).pack(side="left")
            pw = ttk.PanedWindow(fr, orient="horizontal")
            pw.pack(fill="both", expand=True)
            tf = ttk.Frame(pw); pw.add(tf, weight=2)
            cols = ("fz", "metric", "ref", "cmp", "delta")
            tree = ttk.Treeview(tf, columns=cols, show="headings", height=22)
            for c, t_h, w in (("fz", "Fz [N]", 60), ("metric", "Métrica", 110),
                               ("ref", "Base", 80), ("cmp", "Comparado", 80), ("delta", "Δ%", 70)):
                tree.heading(c, text=t_h)
                tree.column(c, width=w, anchor="e" if c != "metric" else "w")
            tree.tag_configure("pos", foreground=_SUCCESS)
            tree.tag_configure("neg", foreground=_DANGER)
            tree.pack(fill="both", expand=True)
            gf = ttk.Frame(pw); pw.add(gf, weight=4)
            figs[tab] = Figure(figsize=(7, 5), dpi=100, facecolor=_BG)
            canv[tab] = FigureCanvasTkAgg(figs[tab], master=gf)
            nav = NavigationToolbar2Tk(canv[tab], gf)
            nav.config(background=_BG3); nav.update()
            canv[tab].get_tk_widget().configure(bg=_BG)
            canv[tab].get_tk_widget().pack(fill="both", expand=True)
            state["tree"] = tree
        else:
            figs[tab] = Figure(figsize=(8, 5), dpi=100, facecolor=_BG)
            canv[tab] = FigureCanvasTkAgg(figs[tab], master=fr)
            nav = NavigationToolbar2Tk(canv[tab], fr)
            nav.config(background=_BG3); nav.update()
            canv[tab].get_tk_widget().configure(bg=_BG)
            canv[tab].get_tk_widget().pack(fill="both", expand=True)

    ttk.Label(right, textvariable=v_status, style="Status.TLabel").pack(
        fill="x", side="bottom", padx=6, pady=(0, 4))

    def controls() -> Controls:
        return Controls(
            fz=parse_fz(v_fz.get(), [600.0, 1100.0, 1600.0]),
            gamma_deg=round(v_gamma.get(), 1),
            p_kpa=None if v_pauto.get() else round(v_p.get()),
            kappa=round(v_kappa.get(), 3),
            alpha_deg=round(v_alpha.get(), 1),
            overlay=v_overlay.get(),
            ref=labels.get(v_ref.get()),
            cmp=labels.get(v_cmp.get()),
        )

    def redraw():
        state["after"] = None
        sync_auto_fz()
        tab = nb.tab(nb.select(), "text")
        ctl = controls()
        slugs = [s for s, v in checks.items() if v.get()]
        t0 = time.perf_counter()
        v_status.set("Calculando…")
        root.update_idletasks()
        try:
            engine.draw(figs[tab], tab, slugs, ctl)
            if tab == TABS[11]:
                res = engine.compare(ctl)
                state["last_cmp"] = res
                tree = state["tree"]
                tree.delete(*tree.get_children())
                for r in (res or {}).get("rows", []):
                    tree.insert("", "end",
                                values=(f"{r['fz']:.0f}", r["label"],
                                        f"{r['ref']:.4g}", f"{r['cmp']:.4g}",
                                        f"{r['delta_pct']:+.2f}"),
                                tags=("pos" if r["delta_pct"] >= 0 else "neg",))
            canv[tab].draw_idle()
            ms = 1e3 * (time.perf_counter() - t0)
            v_status.set(f"{tab} · {len(slugs)} neumático(s) · {len(ctl.fz)} carga(s) — {ms:.0f} ms")
        except Exception as exc:  # noqa: BLE001
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

    menu = tk.Menu(root, bg=_BG3, fg=_FG, activebackground=_ACCENT, activeforeground=_BG, tearoff=0, bd=0)
    fm = tk.Menu(menu, bg=_BG3, fg=_FG, activebackground=_ACCENT, activeforeground=_BG, tearoff=0)
    fm.add_command(label="Abrir index.json…", accelerator="Ctrl+O", command=open_index)
    fm.add_command(label="Guardar figura PNG…", accelerator="Ctrl+S", command=save_png)
    fm.add_separator()
    fm.add_command(label="Salir", command=root.destroy)
    menu.add_cascade(label="Archivo", menu=fm)

    vm = tk.Menu(menu, bg=_BG3, fg=_FG, activebackground=_ACCENT, activeforeground=_BG, tearoff=0)
    vm.add_checkbutton(label="Superponer datos TTC", variable=v_overlay, command=schedule)
    vm.add_checkbutton(label="Fz automático (TTC)", variable=v_fz_auto,
                       command=lambda: (sync_auto_fz(True), schedule()))
    menu.add_cascade(label="Vista", menu=vm)

    root.config(menu=menu)
    root.bind("<Control-s>", lambda _e: save_png())
    root.bind("<Control-o>", lambda _e: open_index())
    nb.bind("<<NotebookTabChanged>>", schedule)
    sync_auto_fz(force=True)
    schedule()
    root.mainloop()


def selftest(store: Store, out: Path) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    out.mkdir(parents=True, exist_ok=True)
    eng = Engine(store)
    ref, cmp_ = store.pair_7_8()
    ctl = Controls(ref=ref, cmp=cmp_, kappa=0.0)
    slugs = [e.slug for e in store.entries]
    for tab in TABS:
        fig = Figure(figsize=(10, 6), facecolor=_BG)
        FigureCanvasAgg(fig)
        eng.draw(fig, tab, slugs, ctl)
        fig.savefig(out / f"{_safe(tab)}.png", dpi=90)
        print("OK", tab)
    res = eng.compare(ctl)
    if res:
        rows_to_csv(out / "comparativa.csv", res, store.by_slug[ref], store.by_slug[cmp_])
        print({k: round(v, 2) for k, v in res["mean_delta_pct"].items()})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Terminal visual de neumáticos MF6.1 (dark mode)")
    ap.add_argument("--index", type=Path, default=Path("data/processed/mf61_fits/index.json"))
    ap.add_argument("--demo", action="store_true", help="neumáticos sintéticos (sin datos TTC)")
    ap.add_argument("--selftest", type=Path, metavar="DIR", help="renderiza tabs a PNG y sale")
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