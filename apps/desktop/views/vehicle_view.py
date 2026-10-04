#!/usr/bin/env python3
"""apps/desktop/views/vehicle_view.py
Dinámica vehicular: trim cuasi-estático, reparto de cargas, balance/subviraje, autoridad de TV y carpet MMM.

Usa el modelo dual-track de ``ter_twin.models.vehicle`` (JAX float64). Todo el cálculo corre en un hilo; el
primer cálculo compila con jit y puede tardar decenas de segundos. Los neumáticos salen de AppState
(front_tyre_params / rear_tyre_params, publicados por la vista de neumáticos) o del MF61Params nominal.
"""
from __future__ import annotations

import logging
import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
for _p in (_ROOT / "src", _ROOT):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import matplotlib  # noqa: E402

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from apps.desktop import theme as T  # noqa: E402
from apps.desktop.state import AppState  # noqa: E402
from apps.desktop.theme import (  # noqa: E402
    ACCENT_AMBER, ACCENT_BLUE, ACCENT_RED, BG_CARD, BG_DARK, BG_HOVER, BORDER, TEXT_MUTED, TEXT_PRIMARY,
    TYRE_PALETTE,
)
from apps.desktop.views.base_view import BaseView  # noqa: E402

__all__ = ["VehicleView", "build_vehicle"]
LOG = logging.getLogger("vehicle_view")
WHEELS = ("FL", "FR", "RL", "RR")


def build_vehicle(app_state: AppState, root: Path = _ROOT) -> Any:
    """VehicleParams desde config/vehicles/<ter26|ter27>/vehicle.yaml (si existe) + neumáticos de AppState."""
    from ter_twin.models import vehicle as VM

    name = str(app_state.get("active_vehicle") or "TeR27-4WD")
    yml = root / "config" / "vehicles" / ("ter26" if "26" in name else "ter27") / "vehicle.yaml"
    try:
        from ter_twin.models.vehicle.io import load_vehicle

        vp = load_vehicle(yml) if yml.exists() else VM.default_ter27()
    except Exception:  # noqa: BLE001
        LOG.exception("vehicle.yaml no cargable; usando default_ter27")
        vp = VM.default_ter27()
    front, rear = app_state.get("front_tyre_params"), app_state.get("rear_tyre_params")
    if front is not None:
        vp = VM.replace_in(vp, "tire", front=front, rear=rear if rear is not None else front)
    return vp


class VehicleView(BaseView):
    def __init__(self, parent: tk.Widget, app_state: AppState, **kwargs: Any) -> None:
        super().__init__(parent, app_state, **kwargs)
        self._q: queue.SimpleQueue[tuple[str, Any]] = queue.SimpleQueue()
        self._poll_id: str | None = None
        self._active = False
        self._busy = False

    # ------------------------------------------------------------------ lifecycle
    def on_mount(self) -> None:
        self.v_speed = tk.DoubleVar(value=60.0)
        self.v_ay = tk.DoubleVar(value=6.0)
        self.v_ax = tk.DoubleVar(value=0.0)
        self.v_mmm = tk.BooleanVar(value=True)
        self._lab: dict[str, tk.StringVar] = {k: tk.StringVar() for k in ("v", "ay", "ax")}

        pane = tk.PanedWindow(self, orient=tk.HORIZONTAL, bg=BORDER, sashwidth=3, bd=0)
        pane.pack(fill="both", expand=True)
        left = ttk.Frame(pane, padding=10)
        right = ttk.Frame(pane)
        pane.add(left, minsize=330, width=370)
        pane.add(right, minsize=700)

        ttk.Label(left, text="PUNTO DE OPERACIÓN", style="Section.TLabel").pack(anchor="w", pady=(0, 6))
        self._slider(left, "Velocidad [km/h]", self.v_speed, 20, 120, "v", "{:.0f}")
        self._slider(left, "Aceleración lateral ay [m/s²]", self.v_ay, 0, 16, "ay", "{:.1f}")
        self._slider(left, "Aceleración longitudinal ax [m/s²]", self.v_ax, -12, 8, "ax", "{:+.1f}")
        ttk.Checkbutton(left, text="Calcular carpet MMM (lento)", variable=self.v_mmm).pack(anchor="w", pady=6)
        self.btn = ttk.Button(left, text="RESOLVER TRIM / BALANCE", command=self._run)
        self.btn.pack(fill="x", pady=(2, 8))
        ttk.Label(left, text="INFORME", style="Section.TLabel").pack(anchor="w", pady=(4, 4))
        self.txt = tk.Text(left, height=26, bg=BG_CARD, fg=TEXT_PRIMARY, relief="flat", font=T.mono(8),
                           highlightthickness=1, highlightbackground=BORDER, wrap="none", insertbackground=ACCENT_BLUE)
        self.txt.pack(fill="both", expand=True)
        self._set_text("Pulsa «Resolver» para calcular el punto.\n"
                       "Vehículo: AppState.active_vehicle (vehicle.yaml o default_ter27).")

        self.fig = Figure(figsize=(8, 6), dpi=100, facecolor=BG_DARK)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        nav = NavigationToolbar2Tk(self.canvas, right, pack_toolbar=False)
        nav.config(background=BG_HOVER)
        nav.update()
        nav.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().configure(bg=BG_DARK, highlightthickness=0)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self._empty("Sin resultados")
        self.bind("<Destroy>", self._on_destroy, add="+")

    def on_activate(self) -> None:
        self._do_mount()
        self._active = True
        if self._poll_id is None:
            self._poll_id = self.after(100, self._poll)

    def on_deactivate(self) -> None:
        self._active = False
        if self._poll_id is not None:
            try:
                self.after_cancel(self._poll_id)
            except tk.TclError:
                pass
            self._poll_id = None
        super().on_deactivate()

    def _on_destroy(self, event: tk.Event) -> None:
        if event.widget is self:
            self._active = False

    # ------------------------------------------------------------------ UI helpers
    def _slider(self, parent: tk.Widget, text: str, var: tk.DoubleVar, lo: float, hi: float, key: str,
                fmt: str) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=3)
        ttk.Label(row, text=text, style="Muted.TLabel").pack(anchor="w")
        line = ttk.Frame(row)
        line.pack(fill="x")
        self._lab[key].set(fmt.format(var.get()))
        ttk.Scale(line, from_=lo, to=hi, variable=var, length=220,
                  command=lambda _v, k=key, f=fmt, v=var: self._lab[k].set(f.format(v.get()))
                  ).pack(side="left", fill="x", expand=True)
        ttk.Label(line, textvariable=self._lab[key], style="Mono.TLabel", width=7, anchor="e").pack(side="right")

    def _set_text(self, s: str) -> None:
        self.txt.configure(state="normal")
        self.txt.delete("1.0", "end")
        self.txt.insert("1.0", s)
        self.txt.configure(state="disabled")

    def _empty(self, msg: str) -> None:
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, msg, ha="center", va="center", color=TEXT_MUTED, fontsize=11)
        self.canvas.draw_idle()

    # ------------------------------------------------------------------ cómputo
    def _run(self) -> None:
        if self._busy:
            return
        self._busy = True
        self.btn.state(["disabled"])
        self._post_status("Resolviendo trim (la primera ejecución compila JAX)…")
        args = (self.v_speed.get() / 3.6, float(self.v_ay.get()), float(self.v_ax.get()), self.v_mmm.get())
        threading.Thread(target=self._worker, args=args, daemon=True).start()

    def _worker(self, v: float, ay: float, ax: float, do_mmm: bool) -> None:
        try:
            from ter_twin.models.vehicle.load_transfer import load_transfer_summary
            from ter_twin.models.vehicle.solvers import (balance_metrics, mmm_grid, mmm_metrics, solve_trim,
                                                         tv_authority, understeer_gradient)

            vp = build_vehicle(self._app_state)
            tr = solve_trim(vp, v, ay, ax)
            bal = balance_metrics(vp, tr)
            tv = tv_authority(vp, tr, v)
            lt = load_transfer_summary(vp, v, ax, ay)
            rep = understeer_gradient(vp, v)
            res, mm = None, None
            if do_mmm:
                res = mmm_grid(vp, v)
                mm = mmm_metrics(res)
            fz = np.asarray(tr.fz)
            lines = [
                f"VEHÍCULO      {vp.name}",
                f"PUNTO         V={v * 3.6:.0f} km/h  ay={ay:.2f}  ax={ax:+.2f} m/s²",
                f"CONVERGIDO    {bool(tr.converged)}  (|R|={float(tr.residual):.3g})",
                "",
                f"β             {np.degrees(float(tr.beta)):+.3f} °",
                f"δ medio       {np.degrees(float(tr.delta)):+.3f} °",
                f"roll φ        {np.degrees(float(tr.phi)):+.3f} °   pitch θ {np.degrees(float(tr.theta)):+.3f} °",
                f"ride height   F {float(tr.rh_f) * 1e3:.1f} mm   R {float(tr.rh_r) * 1e3:.1f} mm",
                "",
                "RUEDA     Fz[N]    α[°]     κ[%]     Fx[N]    Fy[N]   μx   μy",
            ]
            a, k = np.degrees(np.asarray(tr.alpha)), 100 * np.asarray(tr.kappa)
            fx, fy, mx, my = (np.asarray(x) for x in (tr.fx_w, tr.fy_w, tr.mux, tr.muy))
            for i, w in enumerate(WHEELS):
                lines.append(f"{w:<6} {fz[i]:8.0f} {a[i]:+7.2f} {k[i]:+8.2f} {fx[i]:+9.0f} {fy[i]:+8.0f} "
                             f"{mx[i]:4.2f} {my[i]:4.2f}")
            lines += [
                "",
                f"UTIL front/rear   {bal['util_front']:.2f} / {bal['util_rear']:.2f}   "
                f"índice {bal['balance_index']:+.3f} (>0 subviraje)",
                f"K_us lineal       {rep.k_us_linear_deg_g:+.2f} °/g",
                f"K_us trim         {rep.k_us_trim_deg_g:+.2f} °/g",
                f"C_α F/R           {rep.c_alpha_front:.0f} / {rep.c_alpha_rear:.0f} N/rad",
                f"margen estático   {rep.static_margin:+.3f}   v_car {rep.characteristic_speed:.1f} m/s",
                f"TV Mz máx         CCW {tv['mz_max_ccw']:+.0f}  CW {tv['mz_max_cw']:+.0f} N·m",
                f"Transf. long.     {lt['dFz_long']:+.0f} N   lat F/R {lt['dFz_lat_front']:+.0f}/{lt['dFz_lat_rear']:+.0f} N",
                f"Rigidez roll F    {100 * lt['roll_stiffness_front_frac']:.1f} %",
            ]
            if mm:
                lines += ["", f"MMM ay_max        {mm['ay_max_g']:.2f} g",
                          f"dN/dβ             {mm['dN_dbeta_Nm_per_deg']:+.1f} N·m/°",
                          f"dN/dδ             {mm['dN_ddelta_Nm_per_deg']:+.1f} N·m/°",
                          f"d ay/dδ           {mm['d_ay_ddelta_g_per_deg']:+.3f} g/°"]
            self._q.put(("ok", {"text": "\n".join(lines), "fz": fz, "util": np.asarray(bal["util_wheels"]),
                                "mmm": res, "v": v}))
        except Exception as exc:  # noqa: BLE001
            LOG.exception("vehicle worker failed")
            self._q.put(("err", f"{type(exc).__name__}: {exc}"))

    def _poll(self) -> None:
        self._poll_id = None
        if not self._active:
            return
        try:
            while True:
                tag, payload = self._q.get_nowait()
                self._busy = False
                self.btn.state(["!disabled"])
                if tag == "ok":
                    self._set_text(payload["text"])
                    self._draw(payload)
                    self._post_status("Trim resuelto")
                else:
                    self._set_text(f"ERROR\n{payload}")
                    self._post_status(f"Error: {payload}")
        except queue.Empty:
            pass
        except Exception:  # noqa: BLE001
            LOG.exception("vehicle poll failed")
        if self._active:
            self._poll_id = self.after(100, self._poll)

    def _draw(self, p: dict[str, Any]) -> None:
        fig = self.fig
        fig.clear()
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.5], hspace=0.45, wspace=0.25, left=0.08, right=0.98,
                              top=0.93, bottom=0.08)
        a1, a2, a3 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])
        cols = TYRE_PALETTE[:4]
        a1.bar(WHEELS, p["fz"], color=cols, width=0.65)
        a1.set_title("Carga vertical Fz [N]")
        a2.bar(WHEELS, p["util"], color=cols, width=0.65)
        a2.axhline(1.0, color=ACCENT_RED, lw=0.9, ls="--")
        a2.set_title("Utilización lateral |Fy|/(μy·Fz)")
        for a in (a1, a2):
            a.grid(True, axis="y")
        if p["mmm"] is not None:
            from ter_twin.models.vehicle.solvers import plot_mmm

            plot_mmm(p["mmm"], a3)
        else:
            a3.axis("off")
            a3.text(0.5, 0.5, "MMM desactivado", ha="center", va="center", color=TEXT_MUTED)
        self.canvas.draw_idle()