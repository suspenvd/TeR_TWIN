#!/usr/bin/env python3
"""apps/desktop/views/tires_view.py
MF6.1 tyre terminal as an embeddable TeR-Twin Studio view.

Tabs: Fy-a, Mz-a, Fx-k, t(a), Envelope, Gough, Combined interaction, Load sens.,
Pressure sens., Stiffness/degressivity, Camber, Rim comparison.

Assumptions (documented, not inferred silently)
-----------------------------------------------
* Numerics are carried over verbatim from apps/desktop/tire_terminal.py (Store/Entry/Controls/Engine).
* Front tyre = first checked entry, rear tyre = second checked entry (falls back to the first).
  Published to AppState as front_tyre_params / rear_tyre_params / active_tyre_slugs.
* "nominal_fz" in AppState seeds the initial Fz plateaus; Fz auto-detect from TTC overrides it
  and is NOT written back to AppState (it is a plotting aid, not a setup change).
* Shell must call ``_do_mount()``; ``on_activate`` also guards against a missed mount.
* Standalone: ``python -m apps.desktop.views.tires_view [--demo] [--index PATH] [--selftest DIR]``.
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
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import colorchooser, filedialog, messagebox, ttk
from typing import Any, Callable

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
for _p in (_ROOT / "src", _ROOT):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import matplotlib  # noqa: E402

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from apps.desktop.state import AppState  # noqa: E402
from apps.desktop.theme import (  # noqa: E402
    ACCENT_BLUE, ACCENT_GREEN, ACCENT_RED, BG_CARD, BG_DARK, BG_HOVER, BORDER,
    TEXT_MUTED, TEXT_PRIMARY, TYRE_PALETTE, apply_global_theme, apply_matplotlib_theme,
)
from apps.desktop.views.base_view import BaseView  # noqa: E402
from ter_twin.models.tires import pacejka_61 as mf  # noqa: E402

apply_matplotlib_theme()

LOG = logging.getLogger("tires_view")

TABS: tuple[str, ...] = (
    "Fy–α", "Mz–α", "Fx–κ", "Traza t(α)", "Envolvente", "Diagrama Gough",
    "Interacción comb.", "Sens. carga", "Sens. presión", "Rigideces y degres.",
    "Camber", 'Comparativa 7" vs 8"',
)
LINESTYLES: tuple[str, ...] = ("-", "-", "-", "-")
CMP_KEYS: tuple[str, ...] = ("c_alpha_N_deg", "mu_y_peak", "mu_x_peak", "fy_peak_N",
                             "mz_peak_Nm", "trail0_mm", "c_kappa_N")
ALL_RIMS = "Todas las llantas"
DEFAULT_INDEX = _ROOT / "data" / "processed" / "mf61_fits" / "index.json"

_BG, _BG2, _BG3 = BG_DARK, BG_CARD, BG_HOVER
_BORDER, _FG, _FG2 = BORDER, TEXT_PRIMARY, TEXT_MUTED
_SUCCESS, _DANGER = ACCENT_GREEN, ACCENT_RED


# =====================================================================================
# Data layer
# =====================================================================================
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
            if self.params_path is None:
                raise ValueError(f"{self.slug}: sin parámetros")
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
                levels = []
                for c in u[counts >= min_pts]:
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


def get_semantic_tire_color(tire_name: str, rim: float) -> str:
    name = tire_name.lower()

    # Goodyear -> Gama amarilla / ámbar / dorada
    if "goodyear" in name:
        if rim >= 7.5:
            return "#f0883e"  # Naranja ámbar (8")
        elif rim >= 6.5:
            return "#e3b341"  # Amarillo competición (7")
        return "#d29922"      # Mostaza / dorado (6")

    # MRF -> Gama roja / coral
    if "mrf" in name:
        return "#ff7b72" if rim >= 6.5 else "#f85149"

    # Hoosier 16" -> Gama violeta / púrpura
    if "hoosier" in name and "16" in name:
        if rim >= 7.5:
            return "#d2a8ff"  # Lavanda claro (8")
        elif rim >= 6.5:
            return "#bc8cff"  # Violeta puro (7")
        return "#8a63d2"      # Púrpura profundo (6")

    # Hoosier 18" y 20.5" -> Gama azul eléctrico / celeste
    if "hoosier" in name:
        if rim >= 7.5:
            return "#79c0ff"  # Celeste cielo (8")
        elif rim >= 6.5:
            return "#58a6ff"  # Azul eléctrico (7")
        return "#1f6feb"      # Azul marino saturado (6")

    return "#3fb950"  # Verde esmeralda por defecto


class Store:
    def __init__(self, entries: list[Entry]):
        self.entries = entries
        self.by_slug = {e.slug: e for e in entries}
        self.rims = sorted({e.rim for e in entries if np.isfinite(e.rim)})
        # Asignación semántica directa
        self.color = {e.slug: get_semantic_tire_color(e.tire_name, e.rim) for e in entries}

    @classmethod
    def from_index(cls, index_path: Path) -> "Store":
        doc = json.loads(index_path.read_text(encoding="utf-8"))
        base = index_path.parent
        out = [
            Entry(t["slug"], t["tire_name"], float(t["rim_width_in"]), base / t["params"],
                  base / t["data"] if t.get("data") else None, None,
                  t.get("summary", {}), t.get("r2", {}))
            for t in doc["tires"] if t.get("status") == "ok"
        ]
        if not out:
            raise ValueError(f"{index_path}: sin neumáticos ajustados")
        return cls(out)

    @classmethod
    def demo(cls) -> "Store":
        def mk(name: str, rim: float, **kw: float) -> Entry:
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


# =====================================================================================
# Engine (plot logic, GUI-agnostic)
# =====================================================================================
class Engine:
    def __init__(self, store: Store) -> None:
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

    def _meas(self, e: Entry, ctl: Controls, fz: float, kinds, xk: str, yk: str,
              extra: Callable[[dict], np.ndarray] | None = None, max_pts: int = 3000):
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

    def _meas_split(self, e: Entry, ctl: Controls, fz: float, xk: str, yk: str,
                    extra: Callable[[dict], np.ndarray] | None = None, max_pts: int = 3000):
        d = e.data()
        if d is None or not ctl.overlay:
            return None, None
        base = (np.isfinite(d[yk])
                & (np.abs(d["fz"] - fz) < 85.0)
                & (np.abs(np.degrees(d["gamma"]) - ctl.gamma_deg) < 1.0)
                & (np.abs(d["p"] - self._p(e, ctl)) < 15.0))
        if extra is not None:
            base &= extra(d)

        def _pick(kind_val: int):
            idx = np.flatnonzero(base & (d["kind"] == kind_val))
            if idx.size > max_pts:
                idx = idx[np.linspace(0, idx.size - 1, max_pts).astype(int)]
            return (d[xk][idx], d[yk][idx]) if idx.size else None

        return _pick(1), _pick(2)

    @staticmethod
    def _decorate(ax, xl: str, yl: str, title: str | None = None, legend: bool = True) -> None:
        ax.set_xlabel(xl, color=_FG2)
        ax.set_ylabel(yl, color=_FG2)
        ax.grid(True, alpha=0.25, color=_BG3)
        ax.axhline(0, color=_BORDER, lw=0.7)
        ax.axvline(0, color=_BORDER, lw=0.7)
        if title:
            ax.set_title(title, fontsize=9, color=_FG, pad=6)
        if legend and ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7, ncol=2, loc="best", facecolor=_BG2, edgecolor=_BORDER, labelcolor=_FG)

    @staticmethod
    def _empty(fig: Figure, msg: str = "Selecciona al menos un neumático") -> None:
        fig.clear()
        ax = fig.add_subplot(111)
        ax.set_facecolor(_BG2)
        ax.axis("off")
        ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=12, color=_FG2)

    # 1
    def fy_alpha(self, fig, slugs, ctl):
        fig.clear(); ax = fig.add_subplot(111)
        a = np.radians(np.linspace(-14, 14, 281))
        n = len(ctl.fz)
        for s in slugs:
            e = self.store.by_slug[s]; col, ls = self.style(e)
            for j, fz in enumerate(ctl.fz):
                r = self._eval(e, ctl, a, ctl.kappa, fz)
                ax.plot(np.degrees(a), r.fy, color=col, ls=ls, alpha=0.4 + 0.6 * (j + 1) / n, lw=1.6,
                        label=f"{e.label} @ {fz:.0f} N")
                m = self._meas(e, ctl, fz, (0, 2), "alpha", "fy",
                               extra=lambda d: np.abs(d["kappa"] - ctl.kappa) < 0.01)
                if m is not None:
                    ax.scatter(np.degrees(m[0]), m[1], s=10, color=col, alpha=0.35, rasterized=True, zorder=2)
        self._decorate(ax, "α [°]", "Fy [N]", f"Fuerza lateral — γ={ctl.gamma_deg:.1f}°, κ={ctl.kappa:.3f}")

    # 2
    def mz_alpha(self, fig, slugs, ctl):
        fig.clear(); ax = fig.add_subplot(111)
        a = np.radians(np.linspace(-14, 14, 281))
        n = len(ctl.fz)
        for s in slugs:
            e = self.store.by_slug[s]; col, ls = self.style(e)
            for j, fz in enumerate(ctl.fz):
                r = self._eval(e, ctl, a, ctl.kappa, fz)
                ax.plot(np.degrees(a), r.mz, color=col, ls=ls, alpha=0.4 + 0.6 * (j + 1) / n, lw=1.6,
                        label=f"{e.label} @ {fz:.0f} N")
                m = self._meas(e, ctl, fz, (0, 2), "alpha", "mz",
                               extra=lambda d: np.abs(d["kappa"] - ctl.kappa) < 0.01)
                if m is not None:
                    ax.scatter(np.degrees(m[0]), m[1], s=10, color=col, alpha=0.35, rasterized=True, zorder=2)
        self._decorate(ax, "α [°]", "Mz [N·m]", "Momento autoalineante")

    # 3
    def fx_kappa(self, fig, slugs, ctl):
        fig.clear(); ax = fig.add_subplot(111)
        k_arr = np.linspace(-0.3, 0.3, 301)
        a0 = math.radians(ctl.alpha_deg)
        n = len(ctl.fz)
        for s in slugs:
            e = self.store.by_slug[s]; col, ls = self.style(e)
            for j, fz in enumerate(ctl.fz):
                r = self._eval(e, ctl, a0, k_arr, fz)
                ax.plot(k_arr * 100, r.fx, color=col, ls=ls, alpha=0.4 + 0.6 * (j + 1) / n, lw=1.6,
                        label=f"{e.label} @ {fz:.0f} N")
                pure, comb = self._meas_split(e, ctl, fz, "kappa", "fx",
                                              extra=lambda d: np.abs(d["alpha"] - a0) < math.radians(0.5))
                if pure is not None:
                    ax.scatter(pure[0] * 100, pure[1], s=16, color=col, marker="o", alpha=0.50,
                               rasterized=True, zorder=3)
                if comb is not None:
                    ax.scatter(comb[0] * 100, comb[1], s=20, color=col, marker="D", alpha=0.40,
                               rasterized=True, zorder=3)
        self._decorate(ax, "κ [%]", "Fx [N]",
                       f"Fuerza longitudinal — α={ctl.alpha_deg:.1f}°, γ={ctl.gamma_deg:.1f}°", legend=False)
        h, _ = ax.get_legend_handles_labels()
        proxies = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=_FG2, markersize=6, ls="none",
                   label="puro (kind=1)"),
            Line2D([0], [0], marker="D", color="w", markerfacecolor=_FG2, markersize=5, ls="none",
                   label="combinado (kind=2)"),
        ]
        ax.legend(handles=h + proxies, fontsize=7, loc="best", facecolor=_BG2, edgecolor=_BORDER,
                  labelcolor=_FG, ncol=2)

    # 4
    def trail(self, fig, slugs, ctl):
        fig.clear(); ax = fig.add_subplot(111)
        a = np.radians(np.linspace(-14, 14, 281))
        n = len(ctl.fz)
        for s in slugs:
            e = self.store.by_slug[s]; col, ls = self.style(e)
            for j, fz in enumerate(ctl.fz):
                r = self._eval(e, ctl, a, 0.0, fz)
                ax.plot(np.degrees(a), np.asarray(r.t) * 1e3, color=col, ls=ls,
                        alpha=0.4 + 0.6 * (j + 1) / n, lw=1.6, label=f"{e.label} @ {fz:.0f} N")
        self._decorate(ax, "α [°]", "Traza t [mm]", "Traza neumática")

    # 5
    def envelope(self, fig, slugs, ctl):
        fig.clear()
        ax_f, ax_mu = fig.subplots(1, 2)
        fz0 = ctl.fz[0]
        n_fz = len(ctl.fz)
        for s in slugs:
            e = self.store.by_slug[s]; col, ls = self.style(e)
            fx_g, fy_g, al, ka = mf.friction_envelope(e.params, fz0, math.radians(ctl.gamma_deg), self._p(e, ctl))
            for i in range(0, len(al), 6):
                ax_f.plot(fy_g[i], fx_g[i], color=col, lw=0.9, ls=ls, alpha=0.8, label=e.label if i == 0 else None)
            for j in range(0, len(ka), 6):
                ax_f.plot(fy_g[:, j], fx_g[:, j], color=col, lw=0.5, ls=":", alpha=0.45)
            d = e.data()
            if d is not None and ctl.overlay and "kind" in d:
                mask = ((d["kind"] == 2) & (np.abs(d["fz"] - fz0) < 85.0)
                        & (np.abs(np.degrees(d["gamma"]) - ctl.gamma_deg) < 1.0)
                        & (np.abs(d["p"] - self._p(e, ctl)) < 15.0)
                        & np.isfinite(d["fx"]) & np.isfinite(d["fy"]))
                idx = np.flatnonzero(mask)
                if idx.size > 2000:
                    idx = idx[np.linspace(0, idx.size - 1, 2000).astype(int)]
                if idx.size:
                    ax_f.scatter(d["fy"][idx], d["fx"][idx], s=8, color=col, marker="D", alpha=0.35,
                                 rasterized=True, zorder=4, label=f"{e.label} (medido)")
            for k, fz_k in enumerate(ctl.fz):
                fx_k, fy_k, _, _ = mf.friction_envelope(e.params, fz_k, math.radians(ctl.gamma_deg), self._p(e, ctl))
                bx = np.concatenate([fx_k[-1, :], fx_k[:, -1], fx_k[0, ::-1], fx_k[::-1, 0]]) / fz_k
                by = np.concatenate([fy_k[-1, :], fy_k[:, -1], fy_k[0, ::-1], fy_k[::-1, 0]]) / fz_k
                ax_mu.plot(by, bx, color=col, ls=LINESTYLES[k % len(LINESTYLES)], lw=1.5,
                           alpha=0.4 + 0.6 * (k + 1) / n_fz, label=f"{e.label} @ {fz_k:.0f} N")
        ax_f.set_aspect("equal", adjustable="datalim")
        ax_mu.set_aspect("equal", adjustable="datalim")
        self._decorate(ax_f, "Fy [N]", "Fx [N]", f"Fuerzas combinadas (Fy, Fx) @ Fz={fz0:.0f} N")
        self._decorate(ax_mu, "μy = Fy/Fz [-]", "μx = Fx/Fz [-]", "Elipse normalizada μ (Sensibilidad a carga)")
        fig.suptitle(f"Envolvente de Adherencia — γ={ctl.gamma_deg:.1f}°", fontsize=10, color=_FG)
        fig.tight_layout(rect=(0, 0, 1, 0.94))

    # 6
    def gough(self, fig, slugs, ctl):
        fig.clear(); ax = fig.add_subplot(111)
        a_deg = np.linspace(-14, 14, 281)
        a_rad = np.radians(a_deg)
        iso_alphas = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
        n = len(ctl.fz)
        for s in slugs:
            e = self.store.by_slug[s]; col, ls = self.style(e)
            carpet: dict[float, list[tuple[float, float]]] = {al: [] for al in iso_alphas}
            for j, fz in enumerate(ctl.fz):
                r = self._eval(e, ctl, a_rad, ctl.kappa, fz)
                ax.plot(r.fy, r.mz, color=col, ls=ls, lw=1.6, alpha=0.4 + 0.6 * (j + 1) / n,
                        label=f"{e.label} @ {fz:.0f} N")
                for al in iso_alphas:
                    idx = int(np.argmin(np.abs(a_deg - al)))
                    carpet[al].append((float(r.fy[idx]), float(r.mz[idx])))
                m = self._meas(e, ctl, fz, (0, 2), "fy", "mz",
                               extra=lambda d: np.abs(d["kappa"] - ctl.kappa) < 0.01)
                if m is not None:
                    ax.scatter(m[0], m[1], s=10, color=col, alpha=0.30, rasterized=True, zorder=2)
            if len(ctl.fz) > 1:
                for al in iso_alphas:
                    fys, mzs = zip(*carpet[al])
                    ax.plot(fys, mzs, color=_FG2, ls=":", lw=0.8, alpha=0.5)
                    ax.text(fys[-1], mzs[-1], f" {al:g}°", fontsize=7, color=_FG2, va="center")
        self._decorate(ax, "Fy [N]", "Mz [N·m]",
                       f"Diagrama de Gough — Mz(Fy) · γ={ctl.gamma_deg:.1f}°, κ={ctl.kappa:.3f}")

    # 7
    def combined_interaction(self, fig, slugs, ctl):
        fig.clear()
        ax1, ax2 = fig.subplots(1, 2)
        fz0 = ctl.fz[0]
        k_arr = np.linspace(-0.3, 0.3, 241)
        a_deg_arr = np.linspace(0.0, 14.0, 241)
        a_rad_arr = np.radians(a_deg_arr)
        iso_alphas = [2.0, 4.0, 6.0, 8.0, 12.0]
        iso_kappas = [0.03, 0.07, 0.12, 0.20, 0.28]
        multi = len(slugs) > 1
        for s in slugs:
            e = self.store.by_slug[s]; col, ls = self.style(e)
            for i, al in enumerate(iso_alphas):
                r = self._eval(e, ctl, math.radians(al), k_arr, fz0)
                ax1.plot(k_arr * 100, r.fy, color=col, ls=ls, lw=1.5, alpha=0.35 + 0.65 * (i + 1) / len(iso_alphas),
                         label=f"{e.label} α={al:g}°" if (multi and i == 2) or not multi else None)
            for i, kp in enumerate(iso_kappas):
                r = self._eval(e, ctl, a_rad_arr, kp, fz0)
                ax2.plot(a_deg_arr, r.fx, color=col, ls=ls, lw=1.5, alpha=0.35 + 0.65 * (i + 1) / len(iso_kappas),
                         label=f"{e.label} κ={kp:.2f}" if (multi and i == 2) or not multi else None)
            d = e.data()
            if d is not None and ctl.overlay and "kind" in d:
                mask = ((d["kind"] == 2) & (np.abs(d["fz"] - fz0) < 85.0)
                        & (np.abs(np.degrees(d["gamma"]) - ctl.gamma_deg) < 1.0)
                        & (np.abs(d["p"] - self._p(e, ctl)) < 15.0)
                        & np.isfinite(d["fx"]) & np.isfinite(d["fy"]))
                idx = np.flatnonzero(mask)
                if idx.size > 1500:
                    idx = idx[np.linspace(0, idx.size - 1, 1500).astype(int)]
                if idx.size:
                    ax1.scatter(d["kappa"][idx] * 100, d["fy"][idx], s=9, color=col, marker="D", alpha=0.25,
                                rasterized=True)
                    ax2.scatter(np.degrees(np.abs(d["alpha"][idx])), np.abs(d["fx"][idx]), s=9, color=col,
                                marker="D", alpha=0.25, rasterized=True)
        self._decorate(ax1, "κ [%]", "Fy [N]", "Degradación lateral por tracción/frenada Fy(κ)")
        self._decorate(ax2, "α [°]", "Fx [N]", "Degradación longitudinal por viraje Fx(α)")
        fig.suptitle(f"Interacción de Deslizamiento Combinado @ Fz={fz0:.0f} N, γ={ctl.gamma_deg:.1f}°",
                     fontsize=10, color=_FG)
        fig.tight_layout(rect=(0, 0, 1, 0.94))

    # 8
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

    # 9
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
            ca, muy, mux = [], [], []
            for p_val in ps:
                r = mf.summary(e.params, [fz0], g_rad, float(p_val))
                ca.append(float(r["c_alpha_N_deg"][0]))
                muy.append(float(r["mu_y_peak"][0]))
                mux.append(float(r["mu_x_peak"][0]))
            axs[0].plot(ps, ca, color=col, ls=ls, lw=1.6, label=e.label)
            axs[1].plot(ps, muy, color=col, ls=ls, lw=1.6, label=e.label)
            axs[2].plot(ps, mux, color=col, ls=ls, lw=1.6, label=e.label)
            for k, p_val in enumerate(test_pressures):
                r = mf.evaluate(e.params, a_rad, 0.0, g_rad, fz0, float(p_val))
                axs[3].plot(a_deg, np.asarray(r.fy), color=col, ls=LINESTYLES[k % len(LINESTYLES)], lw=1.5,
                            alpha=0.4 + 0.6 * (k + 1) / len(test_pressures),
                            label=f"{e.label} @ {p_val:.0f} kPa" if len(slugs) == 1 else (e.label if k == 1 else None))
        spec = (("Cα [N/°]", "Rigidez de deriva vs Presión"),
                ("μy pico [-]", "Coeficiente lateral pico vs Presión"),
                ("μx pico [-]", "Coeficiente longitudinal pico vs Presión"))
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

    # 10
    def stiffness_degressivity(self, fig, slugs, ctl):
        fig.clear()
        axs = fig.subplots(2, 2).ravel()
        fzs = np.linspace(150, 2200, 42)
        g_rad = math.radians(ctl.gamma_deg)
        for s in slugs:
            e = self.store.by_slug[s]; col, ls = self.style(e)
            out = mf.summary(e.params, fzs, g_rad, self._p(e, ctl))
            ca, ck = out["c_alpha_N_deg"], out["c_kappa_N"]
            for ax, y in zip(axs, (ca, ca / fzs, ck, ck / np.maximum(ca * (180.0 / math.pi), 1e-3))):
                ax.plot(fzs, y, color=col, ls=ls, lw=1.6, label=e.label)
        titulos = (("Cα [N/°]", "Rigidez de deriva absoluta Cα"),
                   ("Cα / Fz [1/°]", "Rigidez normalizada Cα/Fz (Degresividad)"),
                   ("Cκ [N]", "Rigidez de tracción Cκ"),
                   ("Cκ / Cα [-]", "Ratio Cκ / Cα (Longitudinal/Lateral)"))
        for ax, (yl, tit) in zip(axs, titulos):
            ax.set_xlabel("Fz [N]", color=_FG2); ax.set_ylabel(yl, color=_FG2)
            ax.set_title(tit, fontsize=8, color=_FG); ax.grid(True, alpha=0.25)
            for fz_val in ctl.fz:
                ax.axvline(fz_val, color=_BORDER, lw=0.6, ls=":")
        axs[0].legend(fontsize=7, facecolor=_BG2, edgecolor=_BORDER, labelcolor=_FG)
        fig.suptitle(f"Rigideces y Degresividad por Carga Normal — γ={ctl.gamma_deg:.1f}°", fontsize=10, color=_FG)
        fig.tight_layout(rect=(0, 0, 1, 0.94))

    # 11
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

    # 12
    def compare(self, ctl: Controls) -> dict[str, Any] | None:
        if not ctl.ref or not ctl.cmp or ctl.ref == ctl.cmp:
            return None
        a, b = self.store.by_slug[ctl.ref], self.store.by_slug[ctl.cmp]
        return mf.compare_params(a.params, b.params, ctl.fz, math.radians(ctl.gamma_deg),
                                 self._p(a, ctl), self._p(b, ctl))

    def draw_compare(self, fig: Figure, ctl: Controls) -> None:
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
            ax_bar.text(v, yi, f" {v:+.1f}%", va="center", ha="left" if v >= 0 else "right", fontsize=8, color=_FG)
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
            TABS[0]: self.fy_alpha, TABS[1]: self.mz_alpha, TABS[2]: self.fx_kappa, TABS[3]: self.trail,
            TABS[4]: self.envelope, TABS[5]: self.gough, TABS[6]: self.combined_interaction,
            TABS[7]: self.load_sens, TABS[8]: self.pressure_sens, TABS[9]: self.stiffness_degressivity,
            TABS[10]: self.camber,
        }
        dispatch[tab](fig, slugs, ctl)
        if tab in (TABS[0], TABS[1], TABS[2], TABS[3], TABS[5]):
            fig.tight_layout()


# =====================================================================================
# Helpers
# =====================================================================================
def rows_to_csv(path: Path, res: dict[str, Any], ref: Entry, cmp_: Entry) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["ref", "cmp", "gamma_deg", "fz_N", "metric", "ref_value", "cmp_value", "delta_pct"])
        for r in res["rows"]:
            w.writerow([ref.label, cmp_.label, res["gamma_deg"], r["fz"], r["metric"], r["ref"], r["cmp"],
                        r["delta_pct"]])


def _safe(name: str) -> str:
    return re.sub(r"\W+", "_", name).strip("_")


def merge_close_levels(levels: list[float], tol: float = 60.0) -> list[float]:
    if not levels:
        return []
    s = sorted(levels)
    merged: list[float] = []
    curr = [s[0]]
    for x in s[1:]:
        if x - curr[-1] <= tol:
            curr.append(x)
        else:
            merged.append(float(np.round(np.mean(curr)))); curr = [x]
    merged.append(float(np.round(np.mean(curr))))
    return merged


def parse_fz(text: str, default: list[float]) -> list[float]:
    try:
        v = [float(t) for t in re.split(r"[,\s;]+", text.strip()) if t]
    except ValueError:
        return default
    return [x for x in v if x > 0][:10] or default


def resolve_store(index_path: Path | None, demo: bool = False) -> tuple[Store, Path | None]:
    """index.json -> Store; falls back to the synthetic demo store when absent/invalid."""
    if demo:
        return Store.demo(), None
    path = index_path or DEFAULT_INDEX
    try:
        return Store.from_index(path), path
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        LOG.warning("index %s no disponible (%s); usando demo", path, exc)
        return Store.demo(), None


# =====================================================================================
# View
# =====================================================================================
class TiresView(BaseView):
    """Embeddable 12-tab MF6.1 tyre terminal."""

    def __init__(self, parent: tk.Widget, app_state: AppState, *, store: Store | None = None,
                 index_path: Path | None = None, **kwargs: Any) -> None:
        super().__init__(parent, app_state, **kwargs)
        self._init_store = store
        self._init_index = index_path
        self.store: Store = Store([])
        self.engine: Engine = Engine(self.store)
        self._after_id: str | None = None
        self._last_pub: tuple[str, ...] | None = None
        self._last_cmp: dict[str, Any] | None = None
        self._labels: dict[str, str] = {}
        self._checks: dict[str, tk.BooleanVar] = {}
        self._figs: dict[str, Figure] = {}
        self._canv: dict[str, FigureCanvasTkAgg] = {}
        self._active = False

    # ---------------------------------------------------------------- lifecycle
    def on_mount(self) -> None:
        if self._init_store is not None:
            store, path = self._init_store, self._init_index
        else:
            idx = self._init_index or (Path(p) if (p := self._app_state.get("index_path")) else None)
            store, path = resolve_store(idx)
        self._set_store(store, path, rebuild=False)

        nominal = self._app_state.get("nominal_fz") or [445.0, 667.0, 1112.0]
        self.v_fz = tk.StringVar(value=", ".join(f"{x:g}" for x in nominal))
        self.v_fz_auto = tk.BooleanVar(value=True)
        self.v_gamma = tk.DoubleVar(value=0.0)
        self.v_kappa = tk.DoubleVar(value=0.0)
        self.v_alpha = tk.DoubleVar(value=0.0)
        self.v_p = tk.DoubleVar(value=83.0)
        self.v_pauto = tk.BooleanVar(value=True)
        self.v_overlay = tk.BooleanVar(value=True)
        self.v_ref = tk.StringVar()
        self.v_cmp = tk.StringVar()
        self.v_status = tk.StringVar(value="Listo")

        self._build_ui()
        self._reset_selection()
        self._rebuild_list()
        self._app_state.subscribe("nominal_fz", self._on_nominal_fz)
        self.bind("<Destroy>", self._on_destroy, add="+")

    def on_activate(self) -> None:
        self._do_mount()
        self._active = True
        self._app_state.set("index_path", self._index_path)
        self._schedule_redraw()

    def on_deactivate(self) -> None:
        self._active = False
        self._after_id = None
        super().on_deactivate()

    def on_state_change(self, key: str, value: Any) -> None:
        if key == "nominal_fz" and self._mounted and not self.v_fz_auto.get() and value:
            self.v_fz.set(", ".join(f"{x:g}" for x in value))
            self._schedule_redraw()

    def _on_nominal_fz(self, key: str, value: Any) -> None:
        try:
            self.after(0, self.on_state_change, key, value)  # marshal onto Tk thread
        except tk.TclError:
            pass

    def _on_destroy(self, event: tk.Event) -> None:
        if event.widget is self:
            self._app_state.unsubscribe("nominal_fz", self._on_nominal_fz)

    # ---------------------------------------------------------------- store
    def _set_store(self, store: Store, path: Path | None, rebuild: bool = True) -> None:
        self.store = store
        self.engine = Engine(store)
        self._index_path = path
        self._labels = {e.label: e.slug for e in store.entries}
        self._checks = {e.slug: tk.BooleanVar(value=i < 2) for i, e in enumerate(store.entries)}
        self._last_pub = None
        if rebuild:
            self.cb_rim.configure(values=[ALL_RIMS] + [f'{r:g}"' for r in store.rims])
            self.cb_rim.current(0)
            self.cb_ref.configure(values=list(self._labels))
            self.cb_cmp.configure(values=list(self._labels))
            self._reset_selection()
            self._rebuild_list()
            self._app_state.set("index_path", path)
            self._schedule_redraw()

    def _reset_selection(self) -> None:
        a, b = self.store.pair_7_8()
        self.v_ref.set(""); self.v_cmp.set("")
        if a and b:
            self.v_ref.set(self.store.by_slug[a].label)
            self.v_cmp.set(self.store.by_slug[b].label)
            self._checks[a].set(True); self._checks[b].set(True)

    # ---------------------------------------------------------------- UI
    def _build_ui(self) -> None:
        pane = tk.PanedWindow(self, orient=tk.HORIZONTAL, bg=_BG, sashwidth=4, sashpad=0, handlesize=0)
        pane.pack(fill="both", expand=True)
        left = ttk.Frame(pane, padding=8)
        right = ttk.Frame(pane)
        pane.add(left, minsize=260, width=300)
        pane.add(right, minsize=800)

        ttk.Label(left, text="NEUMÁTICOS", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 6))
        self.cb_rim = ttk.Combobox(left, state="readonly", width=18,
                                   values=[ALL_RIMS] + [f'{r:g}"' for r in self.store.rims])
        self.cb_rim.current(0)
        self.cb_rim.pack(anchor="w", pady=(0, 4))

        list_frame = ttk.Frame(left)
        list_frame.pack(fill="both", expand=True)
        self._canvas_l = tk.Canvas(list_frame, width=270, highlightthickness=0, bg=_BG)
        sb = ttk.Scrollbar(list_frame, orient="vertical", command=self._canvas_l.yview)
        self._inner = ttk.Frame(self._canvas_l)
        self._inner.bind("<Configure>", lambda _e: self._canvas_l.configure(scrollregion=self._canvas_l.bbox("all")))
        self._canvas_l.create_window((0, 0), window=self._inner, anchor="nw")
        self._canvas_l.configure(yscrollcommand=sb.set)
        self._canvas_l.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        # Wheel scrolling only while the pointer is over the list, so plots keep their own bindings.
        self._canvas_l.bind("<Enter>", lambda _e: self._bind_wheel(True))
        self._canvas_l.bind("<Leave>", lambda _e: self._bind_wheel(False))
        self.cb_rim.bind("<<ComboboxSelected>>", lambda _e: self._rebuild_list())

        bf = ttk.Frame(left)
        bf.pack(fill="x", pady=(6, 0))
        ttk.Button(bf, text="Todos", command=lambda: self._set_all(True)).pack(side="left")
        ttk.Button(bf, text="Ninguno", command=lambda: self._set_all(False)).pack(side="left", padx=2)
        ttk.Button(bf, text='Par 7"/8"', command=self._pick_pair).pack(side="left")
        bf2 = ttk.Frame(left)
        bf2.pack(fill="x", pady=(4, 0))
        ttk.Button(bf2, text="Abrir index.json…", command=self._open_index).pack(side="left")
        ttk.Button(bf2, text="Guardar PNG…", command=self._save_png).pack(side="left", padx=2)

        ctrl = ttk.LabelFrame(right, text="  Condiciones de ensayo  ", padding=(8, 6))
        ctrl.pack(fill="x", padx=6, pady=(6, 0))
        self._lab: dict[str, tk.StringVar] = {k: tk.StringVar() for k in "gkap"}

        def slider(col: int, text: str, var: tk.DoubleVar, lo: float, hi: float, key: str, fmt: str) -> None:
            ttk.Label(ctrl, text=text).grid(row=0, column=col * 3, sticky="e", padx=(8, 2))
            ttk.Scale(ctrl, from_=lo, to=hi, variable=var, length=130,
                      command=lambda _v: (self._lab[key].set(fmt.format(var.get())), self._schedule_redraw())
                      ).grid(row=0, column=col * 3 + 1, sticky="ew")
            self._lab[key].set(fmt.format(var.get()))
            ttk.Label(ctrl, textvariable=self._lab[key], width=8).grid(row=0, column=col * 3 + 2, sticky="w")

        slider(0, "γ [°]", self.v_gamma, -6, 6, "g", "{:+.1f}°")
        slider(1, "κ (Fy/Mz)", self.v_kappa, -0.3, 0.3, "k", "{:+.3f}")
        slider(2, "α (Fx) [°]", self.v_alpha, -12, 12, "a", "{:+.1f}°")
        slider(3, "p [kPa]", self.v_p, 50, 130, "p", "{:.0f}")

        ttk.Label(ctrl, text="Fz [N]").grid(row=1, column=0, sticky="e", padx=(8, 2))
        e_fz = ttk.Entry(ctrl, textvariable=self.v_fz, width=22)
        e_fz.grid(row=1, column=1, columnspan=2, sticky="w")
        e_fz.bind("<Return>", lambda _e: self._schedule_redraw())
        e_fz.bind("<FocusOut>", lambda _e: self._schedule_redraw())
        e_fz.bind("<Key>", lambda _e: self.v_fz_auto.set(False))
        ttk.Button(ctrl, text="⟳ Auto Fz", width=10,
                   command=lambda: (self.v_fz_auto.set(True), self._sync_auto_fz(True), self._schedule_redraw())
                   ).grid(row=1, column=3, padx=(6, 2), sticky="w")
        ttk.Checkbutton(ctrl, text="Fz auto (TTC)", variable=self.v_fz_auto,
                        command=lambda: (self._sync_auto_fz(True), self._schedule_redraw())
                        ).grid(row=1, column=4, sticky="w")
        ttk.Checkbutton(ctrl, text="p nominal/neumático", variable=self.v_pauto,
                        command=self._schedule_redraw).grid(row=1, column=5, sticky="w", padx=(8, 0))
        ttk.Checkbutton(ctrl, text="Superponer datos TTC", variable=self.v_overlay,
                        command=self._schedule_redraw).grid(row=1, column=6, sticky="w", padx=(8, 0))

        self.nb = ttk.Notebook(right)
        self.nb.pack(fill="both", expand=True, padx=6, pady=4)
        for tab in TABS:
            fr = ttk.Frame(self.nb)
            self.nb.add(fr, text=tab)
            if tab == TABS[11]:
                self._build_compare_tab(fr, tab)
            else:
                self._figs[tab] = Figure(figsize=(8, 5), dpi=100, facecolor=_BG)
                self._attach_canvas(tab, fr)

        ttk.Label(right, textvariable=self.v_status, style="Status.TLabel").pack(
            fill="x", side="bottom", padx=6, pady=(0, 4))
        self.nb.bind("<<NotebookTabChanged>>", lambda _e: self._schedule_redraw())

    def _attach_canvas(self, tab: str, parent: tk.Widget) -> None:
        self._canv[tab] = FigureCanvasTkAgg(self._figs[tab], master=parent)
        nav = NavigationToolbar2Tk(self._canv[tab], parent)
        nav.config(background=_BG3)
        nav.update()
        w = self._canv[tab].get_tk_widget()
        w.configure(bg=_BG)
        w.pack(fill="both", expand=True)

    def _build_compare_tab(self, fr: ttk.Frame, tab: str) -> None:
        top = ttk.Frame(fr, padding=4)
        top.pack(fill="x")
        ttk.Label(top, text="Base:").pack(side="left")
        self.cb_ref = ttk.Combobox(top, textvariable=self.v_ref, values=list(self._labels), state="readonly", width=36)
        self.cb_ref.pack(side="left", padx=4)
        ttk.Label(top, text="Comparado:").pack(side="left")
        self.cb_cmp = ttk.Combobox(top, textvariable=self.v_cmp, values=list(self._labels), state="readonly", width=36)
        self.cb_cmp.pack(side="left", padx=4)
        self.cb_ref.bind("<<ComboboxSelected>>", lambda _e: self._schedule_redraw())
        self.cb_cmp.bind("<<ComboboxSelected>>", lambda _e: self._schedule_redraw())
        ttk.Button(top, text='Auto 7" vs 8"', command=self._pick_pair).pack(side="left", padx=4)
        ttk.Button(top, text="Exportar CSV", command=self._export_csv).pack(side="left")

        pw = ttk.PanedWindow(fr, orient="horizontal")
        pw.pack(fill="both", expand=True)
        tf = ttk.Frame(pw); pw.add(tf, weight=2)
        cols = ("fz", "metric", "ref", "cmp", "delta")
        self.tree = ttk.Treeview(tf, columns=cols, show="headings", height=22)
        for c, t_h, w in (("fz", "Fz [N]", 60), ("metric", "Métrica", 110), ("ref", "Base", 80),
                          ("cmp", "Comparado", 80), ("delta", "Δ%", 70)):
            self.tree.heading(c, text=t_h)
            self.tree.column(c, width=w, anchor="e" if c != "metric" else "w")
        self.tree.tag_configure("pos", foreground=_SUCCESS)
        self.tree.tag_configure("neg", foreground=_DANGER)
        self.tree.pack(fill="both", expand=True)
        gf = ttk.Frame(pw); pw.add(gf, weight=4)
        self._figs[tab] = Figure(figsize=(7, 5), dpi=100, facecolor=_BG)
        self._attach_canvas(tab, gf)

    # ---------------------------------------------------------------- list / selection
    def _bind_wheel(self, on: bool) -> None:
        def wheel(e: tk.Event) -> None:
            step = -1 if (getattr(e, "delta", 0) > 0 or e.num == 4) else 1
            self._canvas_l.yview_scroll(step, "units")

        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            if on:
                self._canvas_l.bind_all(seq, wheel)
            else:
                self._canvas_l.unbind_all(seq)

    def _pick_color(self, slug: str, swatch_widget: tk.Label) -> None:
        current_col = self.store.color.get(slug, "#58a6ff")
        _, hex_col = colorchooser.askcolor(color=current_col, title=f"Color para {slug}", parent=self)
        if hex_col:
            self.store.color[slug] = hex_col
            swatch_widget.configure(bg=hex_col)
            self._schedule_redraw()

    def _rebuild_list(self) -> None:
        for w in self._inner.winfo_children():
            w.destroy()
        sel = self.cb_rim.get()
        for e in self.store.entries:
            if sel != ALL_RIMS and not (np.isfinite(e.rim) and f'{e.rim:g}"' == sel):
                continue
            row = tk.Frame(self._inner, bg=_BG)
            row.pack(anchor="w", fill="x", pady=2)

            # Cuadro de color interactivo con cursor tipo mano
            swatch = tk.Label(row, bg=self.store.color[e.slug], width=2, height=1, relief="flat", cursor="hand2")
            swatch.pack(side="left", padx=(2, 6))
            swatch.bind("<Button-1>", lambda _evt, s=e.slug, sw=swatch: self._pick_color(s, sw))

            cb = tk.Checkbutton(
                row,
                text=f"{e.label}",
                variable=self._checks[e.slug],
                command=self._schedule_redraw,
                bg=_BG,
                fg=_FG,
                selectcolor="#1f6feb",
                activebackground=_BG3,
                activeforeground="#ffffff",
                font=("Segoe UI", 9),
                cursor="hand2",
                highlightthickness=0,
                bd=0,
            )
            cb.pack(side="left")

    def _set_all(self, val: bool) -> None:
        for v in self._checks.values():
            v.set(val)
        self._schedule_redraw()

    def _pick_pair(self) -> None:
        a, b = self.store.pair_7_8()
        if not a or not b:
            messagebox.showinfo('Par 7"/8"', 'No hay un mismo Tire_Name con 7" y 8".', parent=self)
            return
        self._set_all(False)
        self._checks[a].set(True); self._checks[b].set(True)
        self.v_ref.set(self.store.by_slug[a].label); self.v_cmp.set(self.store.by_slug[b].label)
        self._schedule_redraw()

    def _selected(self) -> list[str]:
        return [s for s, v in self._checks.items() if v.get()]

    def _sync_auto_fz(self, force: bool = False) -> None:
        if not self.v_fz_auto.get() and not force:
            return
        tab = self.nb.tab(self.nb.select(), "text") if self.nb.tabs() else ""
        slugs = ([self._labels.get(self.v_ref.get()), self._labels.get(self.v_cmp.get())]
                 if tab == TABS[11] else self._selected())
        raw: list[float] = []
        for s in filter(None, slugs):
            e = self.store.by_slug.get(s)
            if e:
                raw.extend(e.fz_levels())
        merged = merge_close_levels(raw)
        if merged:
            txt = ", ".join(f"{int(round(x))}" for x in merged)
            if self.v_fz.get() != txt:
                self.v_fz.set(txt)

    def _controls(self) -> Controls:
        return Controls(
            fz=parse_fz(self.v_fz.get(), [600.0, 1100.0, 1600.0]),
            gamma_deg=round(self.v_gamma.get(), 1),
            p_kpa=None if self.v_pauto.get() else float(round(self.v_p.get())),
            kappa=round(self.v_kappa.get(), 3),
            alpha_deg=round(self.v_alpha.get(), 1),
            overlay=self.v_overlay.get(),
            ref=self._labels.get(self.v_ref.get()),
            cmp=self._labels.get(self.v_cmp.get()),
        )

    # ---------------------------------------------------------------- drawing
    def _schedule_redraw(self) -> None:
        if not self._mounted:
            return
        if self._after_id is not None:
            self._cancel_after(self._after_id)
        self._after_id = self._schedule(40, self._redraw)

    def _publish_selection(self, slugs: list[str]) -> None:
        key = tuple(slugs)
        if key == self._last_pub or not slugs:
            return
        self._last_pub = key
        front = self.store.by_slug[slugs[0]]
        rear = self.store.by_slug[slugs[1]] if len(slugs) > 1 else front
        self._app_state.set("front_tyre_params", front.params)
        self._app_state.set("rear_tyre_params", rear.params)
        self._app_state.set("active_tyre_slugs", list(slugs))

    def _status(self, text: str) -> None:
        self.v_status.set(text)
        self._post_status(text)

    def _redraw(self) -> None:
        if self._after_id in self._pending_afters:
            self._pending_afters.remove(self._after_id)
        self._after_id = None
        tab = "?"
        try:
            self._sync_auto_fz()
            tab = self.nb.tab(self.nb.select(), "text")
            ctl = self._controls()
            slugs = self._selected()
            t0 = time.perf_counter()
            self._status("Calculando…")
            self.update_idletasks()
            self.engine.draw(self._figs[tab], tab, slugs, ctl)
            if tab == TABS[11]:
                res = self.engine.compare(ctl)
                self._last_cmp = res
                self.tree.delete(*self.tree.get_children())
                for r in (res or {}).get("rows", []):
                    self.tree.insert("", "end",
                                     values=(f"{r['fz']:.0f}", r["label"], f"{r['ref']:.4g}", f"{r['cmp']:.4g}",
                                             f"{r['delta_pct']:+.2f}"),
                                     tags=("pos" if r["delta_pct"] >= 0 else "neg",))
            self._canv[tab].draw_idle()
            self._publish_selection(slugs)
            ms = 1e3 * (time.perf_counter() - t0)
            self._app_state.set("last_draw_ms", ms)
            self._status(f"{tab} · {len(slugs)} neumático(s) · {len(ctl.fz)} carga(s) — {ms:.0f} ms")
        except Exception as exc:  # noqa: BLE001  # a failed redraw must never kill the Tk loop
            LOG.exception("Fallo al dibujar %s", tab)
            self._status(f"Error en {tab}: {exc}")

    # ---------------------------------------------------------------- file actions
    def _export_csv(self) -> None:
        res = self._last_cmp
        if not res:
            messagebox.showinfo("Exportar", "No hay comparativa activa.", parent=self)
            return
        p = filedialog.asksaveasfilename(parent=self, defaultextension=".csv", initialfile="comparativa_llantas.csv")
        if p:
            try:
                rows_to_csv(Path(p), res, self.store.by_slug[self._labels[self.v_ref.get()]],
                            self.store.by_slug[self._labels[self.v_cmp.get()]])
                self._status(f"Exportado {p}")
            except OSError as exc:
                self._status(f"Error exportando CSV: {exc}")

    def _save_png(self) -> None:
        tab = self.nb.tab(self.nb.select(), "text")
        p = filedialog.asksaveasfilename(parent=self, defaultextension=".png", initialfile=f"{_safe(tab)}.png")
        if p:
            try:
                self._figs[tab].savefig(p, dpi=200)
                self._status(f"Figura guardada en {p}")
            except OSError as exc:
                self._status(f"Error guardando figura: {exc}")

    def _open_index(self) -> None:
        p = filedialog.askopenfilename(parent=self, filetypes=[("index.json", "*.json")])
        if not p:
            return
        try:
            store = Store.from_index(Path(p))
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            self._status(f"Index inválido: {exc}")
            messagebox.showerror("index.json", str(exc), parent=self)
            return
        self._set_store(store, Path(p))
        self._status(f"Cargado {p}: {len(store.entries)} neumáticos")


# =====================================================================================
# Headless self-test + standalone entrypoint
# =====================================================================================
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
    if res and ref and cmp_:
        rows_to_csv(out / "comparativa.csv", res, store.by_slug[ref], store.by_slug[cmp_])
        print({k: round(v, 2) for k, v in res["mean_delta_pct"].items()})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="TeR-Twin · Tyre view (standalone)")
    ap.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    ap.add_argument("--demo", action="store_true", help="neumáticos sintéticos (sin datos TTC)")
    ap.add_argument("--selftest", type=Path, metavar="DIR", help="renderiza tabs a PNG y sale")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    store, path = resolve_store(a.index, a.demo)
    if a.selftest:
        selftest(store, a.selftest)
        return 0
    root = tk.Tk()
    root.title("TeR-Twin · Terminal de neumáticos MF6.1")
    root.geometry("1600x950")
    apply_global_theme(root)
    state = AppState.instance()
    state.set("index_path", path)
    view = TiresView(root, state, store=store, index_path=path)
    view.pack(fill="both", expand=True)
    view._do_mount()
    view.on_activate()
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())