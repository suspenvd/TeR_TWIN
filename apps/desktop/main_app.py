#!/usr/bin/env python3
"""apps/desktop/main_app.py
TeR-Twin Studio: shell de cabina (MoTeC i2 / Bloomberg) con cuatro zonas.

    A  NavDock       dock lateral fijo 240 px (identidad, navegacion, estado del motor)
    B  TopStrip      barra superior 36 px (breadcrumb, selector de vehiculo, ticker del enlace CAN)
    C  viewport      contenedor central donde se montan las vistas (pack fill/expand)
    D  StatusStrip   barra inferior 24 px (operativo | buffers/memoria | sincronizacion/solver)

Ejecutar desde la raiz del repo:
    python3 -m apps.desktop.main_app [--view telemetry|tires|correlation|vehicle|suspension|aero|lts]
                                     [--index data/processed/mf61_fits/index.json]
                                     [--vehicle TeR27-4WD] [--selftest]

Contrato con las vistas (sin cambios respecto a la shell anterior)
------------------------------------------------------------------
* Vistas importadas y montadas en la primera navegacion: un fallo de importacion (jax, cantools...) muestra
  una tarjeta de error con "Reintentar" en vez de abortar el arranque.
* Ciclo de vida: _do_mount() una vez, on_activate() al entrar, on_deactivate() al salir. Todas las vistas
  montadas reciben on_state_change (visibles o no).
* AppState.set puede llamarse desde hilos de trabajo: los callbacks solo encolan; un pump Tk de 30 ms
  despacha y fusiona al ultimo valor por clave.
* ``self.app_state`` no se llama ``state`` porque tk.Tk.state() es el metodo de estado de ventana.
* El ticker CAN y los indicadores de buffer leen atributos de TelemetryView (_ingest, _buffer, _t, _log_name)
  con getattr defensivo: si la vista cambia, el shell degrada a "IDLE" sin lanzar excepciones.
"""
from __future__ import annotations

import argparse
import importlib
import logging
import os
import queue
import sys
import threading
import time
import tkinter as tk
import traceback
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk
from types import TracebackType
from typing import Any, Callable

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT / "src", _ROOT):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from apps.desktop import theme as T  # noqa: E402
from apps.desktop.state import AppState  # noqa: E402
from apps.desktop.theme import (  # noqa: E402
    ACCENT_AMBER, ACCENT_BLUE, ACCENT_GREEN, ACCENT_RED, BG_ACTIVE, BG_CARD, BG_DARK, BG_HOVER, BG_ROOT,
    BORDER, TEXT_BRIGHT, TEXT_MUTED, TEXT_PRIMARY, apply_global_theme,
)
from apps.desktop.views.base_view import BaseView  # noqa: E402

LOG = logging.getLogger("ter_twin_studio")

SIDEBAR_WIDTH = 240
HEADER_H = 36
STATUS_H = 24
PUMP_MS = 30
SLOW_MS = 500
DBC_PATH = _ROOT / "config" / "can" / "dbc" / "TER.dbc"
VEHICLES: tuple[str, ...] = ("TeR27-4WD", "TeR27-2WD", "TeR26")

# Claves repartidas a las vistas (status_text / last_draw_ms son telemetria de UI solo para la shell)
VIEW_STATE_KEYS: tuple[str, ...] = (
    "active_vehicle", "front_tyre_params", "rear_tyre_params", "active_tyre_slugs",
    "nominal_fz", "aero_balance", "roll_stiffness_dist", "index_path",
)
SHELL_KEYS: tuple[str, ...] = ("status_text", "last_draw_ms", "active_vehicle")


@dataclass(frozen=True, slots=True)
class ViewSpec:
    key: str
    title: str      # titulo para status/errores
    label: str      # texto de navegacion
    icon: str
    group: str
    module: str
    cls: str


VIEWS: tuple[ViewSpec, ...] = (
    ViewSpec("telemetry", "Telemetría en Vivo", "Telemetría en Vivo", "◉", "ANÁLISIS",
             "apps.desktop.views.telemetry_view", "TelemetryView"),
    ViewSpec("tires", "Neumáticos MF6.1", "Neumáticos MF6.1", "◎", "ANÁLISIS",
             "apps.desktop.views.tires_view", "TiresView"),
    ViewSpec("correlation", "Correlación Modelo ↔ Pista", "Correlación Modelo/Pista", "≋", "ANÁLISIS",
             "apps.desktop.views.correlation_view", "CorrelationView"),
    ViewSpec("vehicle", "Dinámica Vehicular", "Dinámica Vehicular", "▣", "MODELO",
             "apps.desktop.views.vehicle_view", "VehicleView"),
    ViewSpec("suspension", "Cinemática Suspensión", "Cinemática Suspensión", "⟂", "MODELO",
             "apps.desktop.views.suspension_view", "SuspensionView"),
    ViewSpec("aero", "Mapas Aerodinámicos", "Mapas Aerodinámicos", "≈", "MODELO",
             "apps.desktop.views.aerodynamics_view", "AeroView"),
    ViewSpec("lts", "Simulador Lap Time", "Simulador Lap Time", "◷", "MODELO",
             "apps.desktop.views.lts_view", "LTSView"),
)


# ======================================================================================
# Utilidades
# ======================================================================================
def _enable_hidpi() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        pass


def _rss_mb() -> float | None:
    """Memoria residente del proceso en MB (psutil -> /proc -> None)."""
    try:
        import psutil  # type: ignore[import-not-found]

        return float(psutil.Process().memory_info().rss) / 1048576.0
    except Exception:  # noqa: BLE001
        pass
    try:
        with open("/proc/self/statm", encoding="ascii") as fh:
            pages = int(fh.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / 1048576.0
    except (OSError, ValueError, AttributeError):
        return None


def _hairline(parent: tk.Misc, horizontal: bool = True) -> tk.Frame:
    return tk.Frame(parent, bg=BORDER, height=1 if horizontal else 0, width=0 if horizontal else 1)


# ======================================================================================
# ZONA A: dock lateral
# ======================================================================================
class NavDock(tk.Frame):
    """Filas planas con barra indicadora de 3 px (cyan) en el modulo activo."""

    def __init__(self, parent: tk.Misc, specs: tuple[ViewSpec, ...], on_select: Callable[[str], None]) -> None:
        super().__init__(parent, bg=BG_CARD, width=SIDEBAR_WIDTH)
        self.pack_propagate(False)
        self._on_select = on_select
        self._rows: dict[str, dict[str, tk.Widget]] = {}
        self._active: str | None = None
        self._hover: str | None = None

        # ---- encabezado institucional
        head = tk.Frame(self, bg=BG_CARD)
        head.pack(fill="x", padx=14, pady=(12, 8))
        tk.Label(head, text="TeR-TWIN", bg=BG_CARD, fg=TEXT_PRIMARY, font=T.ui(12, True), anchor="w").pack(anchor="w")
        tk.Label(head, text="TECNUN eRACING", bg=BG_CARD, fg=TEXT_MUTED, font=T.ui(8), anchor="w").pack(anchor="w")
        self._badge = tk.Label(head, text="TeR27 · 4WD", bg=BG_HOVER, fg=ACCENT_BLUE, font=T.mono(8, True),
                               padx=8, pady=2, highlightthickness=1, highlightbackground=BORDER,
                               highlightcolor=BORDER)
        self._badge.pack(anchor="w", pady=(8, 0))
        _hairline(self).pack(fill="x")

        # ---- navegacion
        nav = tk.Frame(self, bg=BG_CARD)
        nav.pack(fill="x", pady=(6, 0))
        group = None
        for i, spec in enumerate(specs, 1):
            if spec.group != group:
                group = spec.group
                tk.Label(nav, text=group, bg=BG_CARD, fg=TEXT_MUTED, font=T.mono(7, True), anchor="w").pack(
                    fill="x", padx=14, pady=(10, 3))
            self._build_row(nav, spec, i)

        # ---- pie: entorno y motor
        foot = tk.Frame(self, bg=BG_CARD)
        foot.pack(side="bottom", fill="x")
        _hairline(foot).pack(fill="x")
        box = tk.Frame(foot, bg=BG_CARD)
        box.pack(fill="x", padx=14, pady=8)
        venv = os.environ.get("VIRTUAL_ENV")
        env_txt = f"ENV  {Path(venv).name if venv else 'system'} · py{sys.version_info.major}.{sys.version_info.minor}"
        tk.Label(box, text=env_txt, bg=BG_CARD, fg=TEXT_MUTED, font=T.mono(8), anchor="w").pack(fill="x")
        eng = tk.Frame(box, bg=BG_CARD)
        eng.pack(fill="x", pady=(3, 0))
        self._eng_dot = tk.Label(eng, text="●", bg=BG_CARD, fg=ACCENT_AMBER, font=T.mono(8))
        self._eng_dot.pack(side="left")
        self._eng_txt = tk.Label(eng, text=" JAX · 200 Hz Engine", bg=BG_CARD, fg=TEXT_PRIMARY, font=T.mono(8))
        self._eng_txt.pack(side="left")

        # hairline derecha
        tk.Frame(self, bg=BORDER, width=1).place(relx=1.0, x=-1, rely=0.0, relheight=1.0)

    # ---------------------------------------------------------------- filas
    def _build_row(self, parent: tk.Misc, spec: ViewSpec, idx: int) -> None:
        row = tk.Frame(parent, bg=BG_CARD, height=30, cursor="hand2")
        row.pack(fill="x")
        row.pack_propagate(False)
        ind = tk.Frame(row, bg=BG_CARD, width=3)
        ind.pack(side="left", fill="y")
        icon = tk.Label(row, text=spec.icon, bg=BG_CARD, fg=TEXT_MUTED, font=T.mono(10), width=3)
        icon.pack(side="left")
        txt = tk.Label(row, text=spec.label, bg=BG_CARD, fg=TEXT_MUTED, font=T.ui(9), anchor="w")
        txt.pack(side="left", fill="x", expand=True)
        hint = tk.Label(row, text=f"^{idx}", bg=BG_CARD, fg=TEXT_MUTED, font=T.mono(7))
        hint.pack(side="right", padx=8)
        self._rows[spec.key] = {"row": row, "ind": ind, "icon": icon, "txt": txt, "hint": hint}
        for w in (row, icon, txt, hint):
            w.bind("<Enter>", lambda _e, k=spec.key: self._set_hover(k))
            w.bind("<Leave>", lambda _e, k=spec.key: self._set_hover(None, k))
            w.bind("<Button-1>", lambda _e, k=spec.key: self._on_select(k))

    def _set_hover(self, key: str | None, leaving: str | None = None) -> None:
        if key is None and self._hover != leaving:
            return
        self._hover = key
        self._paint()

    def _paint(self) -> None:
        for key, w in self._rows.items():
            active = key == self._active
            hover = key == self._hover
            bg = BG_ACTIVE if active else BG_HOVER if hover else BG_CARD
            fg = TEXT_BRIGHT if active else TEXT_PRIMARY if hover else TEXT_MUTED
            for name in ("row", "icon", "txt", "hint"):
                w[name].configure(bg=bg)
            w["ind"].configure(bg=ACCENT_BLUE if active else bg)
            w["txt"].configure(fg=fg, font=T.ui(9, active))
            w["icon"].configure(fg=ACCENT_BLUE if active else fg)

    # ---------------------------------------------------------------- API
    def set_active(self, key: str) -> None:
        self._active = key
        self._paint()

    def set_vehicle(self, name: str) -> None:
        self._badge.configure(text=str(name).replace("-", " · ", 1))

    def set_engine(self, text: str, color: str) -> None:
        self._eng_txt.configure(text=f" {text}")
        self._eng_dot.configure(fg=color)


# ======================================================================================
# ZONA B: barra superior
# ======================================================================================
class TopStrip(tk.Frame):
    def __init__(self, parent: tk.Misc, vehicles: tuple[str, ...], on_vehicle: Callable[[str], None]) -> None:
        super().__init__(parent, bg=BG_DARK, height=HEADER_H)
        self.pack_propagate(False)
        self._crumb_tail = tk.StringVar(value="")
        tk.Label(self, text="DASHBOARD // ", bg=BG_DARK, fg=TEXT_MUTED, font=T.mono(9, True)).pack(
            side="left", padx=(14, 0))
        tk.Label(self, textvariable=self._crumb_tail, bg=BG_DARK, fg=TEXT_PRIMARY, font=T.mono(9, True)).pack(
            side="left")

        # derecha: vehiculo | ticker
        self.v_vehicle = tk.StringVar(value=vehicles[0])
        cb = ttk.Combobox(self, textvariable=self.v_vehicle, values=list(vehicles), state="readonly", width=11)
        cb.pack(side="right", padx=(8, 12), pady=5)
        cb.bind("<<ComboboxSelected>>", lambda _e: on_vehicle(self.v_vehicle.get()))
        tk.Label(self, text="VEH", bg=BG_DARK, fg=TEXT_MUTED, font=T.mono(7, True)).pack(side="right")
        tk.Frame(self, bg=BORDER, width=1).pack(side="right", fill="y", pady=8, padx=10)
        self._link_txt = tk.Label(self, text="", bg=BG_DARK, fg=TEXT_MUTED, font=T.mono(8))
        self._link_txt.pack(side="right")
        self._link_dot = tk.Label(self, text="●", bg=BG_DARK, fg=TEXT_MUTED, font=T.mono(10))
        self._link_dot.pack(side="right", padx=(0, 4))
        tk.Frame(self, bg=BORDER, height=1).place(relx=0.0, rely=1.0, y=-1, relwidth=1.0)

    def set_crumb(self, text: str) -> None:
        self._crumb_tail.set(text.upper())

    def set_link(self, color: str, text: str) -> None:
        self._link_dot.configure(fg=color)
        self._link_txt.configure(text=text, fg=TEXT_PRIMARY if color != TEXT_MUTED else TEXT_MUTED)

    def set_vehicle(self, name: str) -> None:
        if self.v_vehicle.get() != name:
            self.v_vehicle.set(name)


# ======================================================================================
# ZONA D: barra de estado
# ======================================================================================
class StatusStrip(tk.Frame):
    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent, bg=BG_DARK, height=STATUS_H)
        self.pack_propagate(False)
        tk.Frame(self, bg=BORDER, height=1).pack(side="top", fill="x")
        inner = tk.Frame(self, bg=BG_DARK)
        inner.pack(fill="both", expand=True)
        for c, w in enumerate((5, 4, 5)):
            inner.columnconfigure(c, weight=w, uniform="st")
        inner.rowconfigure(0, weight=1)
        mk = lambda anchor, col: tk.Label(inner, text="", bg=BG_DARK, fg=TEXT_PRIMARY, font=T.mono(8),  # noqa: E731
                                          anchor=anchor)
        self._l, self._c, self._r = mk("w", 0), mk("center", 1), mk("e", 2)
        self._l.grid(row=0, column=0, sticky="ew", padx=(12, 6))
        self._c.grid(row=0, column=1, sticky="ew")
        self._r.grid(row=0, column=2, sticky="ew", padx=(6, 12))

    def set_left(self, text: str) -> None:
        self._l.configure(text=text)

    def set_center(self, text: str, color: str = TEXT_MUTED) -> None:
        self._c.configure(text=text, fg=color)

    def set_right(self, text: str, color: str = TEXT_MUTED) -> None:
        self._r.configure(text=text, fg=color)


# ======================================================================================
# Aplicacion
# ======================================================================================
class TeRTwinApp(tk.Tk):
    def __init__(self, initial: str = "telemetry") -> None:
        super().__init__()
        self._specs: dict[str, ViewSpec] = {v.key: v for v in VIEWS}
        if initial not in self._specs:
            raise KeyError(f"vista desconocida: {initial!r}")
        self.title("TeR-Twin Studio · Vehicle Dynamics & Telemetry Workstation")
        self.geometry("1680x980")
        self.minsize(1280, 720)

        self.app_state = AppState.instance()
        apply_global_theme(self)

        self._widgets: dict[str, tk.Widget] = {}
        self._active_key: str | None = None
        self._alive = True
        self._pump_id: str | None = None
        self._slow_id: str | None = None
        self._events: queue.SimpleQueue[tuple[str, Any]] = queue.SimpleQueue()
        self._solver_txt: str | None = None

        self.report_callback_exception = self._on_callback_exception  # type: ignore[assignment]
        self._build()
        self._on_vehicle_state(self.app_state.get("active_vehicle") or VEHICLES[0])

        for key in dict.fromkeys(VIEW_STATE_KEYS + SHELL_KEYS):
            self.app_state.subscribe(key, self._enqueue_state)
        self.protocol("WM_DELETE_WINDOW", self.close)
        for i, v in enumerate(VIEWS, 1):
            self.bind(f"<Control-Key-{i}>", lambda _e, k=v.key: self.navigate(k))
        self.bind("<Control-q>", lambda _e: self.close())
        self.bind("<F11>", lambda _e: self.attributes("-fullscreen", not bool(self.attributes("-fullscreen"))))
        self._pump_id = self.after(PUMP_MS, self._pump)
        self._slow_id = self.after(SLOW_MS, self._slow_tick)
        self.navigate(initial)
        self._refresh_chrome()

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        self.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=1)
        self.dock = NavDock(self, VIEWS, self._show)
        self.dock.grid(row=0, column=0, rowspan=2, sticky="ns")
        self.strip = TopStrip(self, VEHICLES, self._on_vehicle_pick)
        self.strip.grid(row=0, column=1, sticky="ew")
        self.content = tk.Frame(self, bg=BG_ROOT)
        self.content.grid(row=1, column=1, sticky="nsew")
        self.status = StatusStrip(self)
        self.status.grid(row=2, column=0, columnspan=2, sticky="ew")

    # ------------------------------------------------------------------ routing
    def navigate(self, key: str) -> None:
        if key not in self._specs:
            raise KeyError(f"vista desconocida: {key!r}")
        self._show(key)

    def _show(self, key: str) -> None:
        if key == self._active_key:
            return
        spec = self._specs[key]
        widget = self._widgets.get(key)
        if widget is None:
            widget = self._instantiate(spec)
            self._widgets[key] = widget

        prev = self._widgets.get(self._active_key) if self._active_key else None
        if prev is not None:
            if isinstance(prev, BaseView):
                self._guard(prev.on_deactivate, f"{self._active_key}.on_deactivate")
            prev.pack_forget()

        widget.pack(fill="both", expand=True)
        self._active_key = key
        self.dock.set_active(key)
        self.strip.set_crumb(spec.title)
        self.app_state.update_status(spec.title)
        if isinstance(widget, BaseView):
            self._guard(widget.on_activate, f"{key}.on_activate")
        self._refresh_chrome()

    def _instantiate(self, spec: ViewSpec) -> tk.Widget:
        self.app_state.update_status(f"Cargando {spec.title}…")
        self._refresh_chrome()
        self.update_idletasks()
        t0 = time.perf_counter()
        view: BaseView | None = None
        try:
            cls = getattr(importlib.import_module(spec.module), spec.cls)
            view = cls(self.content, self.app_state)
            view._do_mount()
        except Exception as exc:  # noqa: BLE001  # aisla un modulo roto del shell
            LOG.exception("No se pudo cargar la vista %s", spec.key)
            if view is not None:
                view.destroy()
            self.app_state.update_status(f"Error cargando {spec.title}: {exc}")
            return self._error_card(spec, exc)
        LOG.info("Vista %s montada en %.0f ms", spec.key, 1e3 * (time.perf_counter() - t0))
        return view

    def _error_card(self, spec: ViewSpec, exc: Exception) -> ttk.Frame:
        frame = ttk.Frame(self.content, style="Card.TFrame", padding=24)
        ttk.Label(frame, text=f"No se pudo cargar «{spec.title}»", style="CardH.TLabel",
                  foreground=ACCENT_RED).pack(anchor="w")
        ttk.Label(frame, text=f"{type(exc).__name__}: {exc}", style="Card.TLabel", wraplength=1000,
                  justify="left").pack(anchor="w", pady=(6, 12))
        txt = tk.Text(frame, height=18, bg=BG_ROOT, fg=TEXT_MUTED, relief="flat", font=T.mono(9),
                      wrap="none", highlightthickness=1, highlightbackground=BORDER)
        txt.insert("1.0", "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        txt.configure(state="disabled")
        txt.pack(fill="both", expand=True)

        def retry() -> None:
            frame.destroy()
            self._widgets.pop(spec.key, None)
            self._active_key = None
            self._show(spec.key)

        ttk.Button(frame, text="Reintentar", command=retry).pack(anchor="e", pady=(12, 0))
        return frame

    def _guard(self, fn: Callable[[], Any], what: str) -> bool:
        try:
            fn()
            return True
        except Exception as exc:  # noqa: BLE001  # los hooks no deben matar el event loop
            LOG.exception("Fallo en %s", what)
            self.app_state.update_status(f"Error en {what}: {exc}")
            return False

    # ------------------------------------------------------------------ estado
    def _enqueue_state(self, key: str, value: Any) -> None:
        self._events.put((key, value))  # thread-safe; nunca toca Tk

    def _pump(self) -> None:
        if not self._alive:
            return
        latest: dict[str, Any] = {}
        try:
            while True:
                k, v = self._events.get_nowait()
                latest[k] = v
        except queue.Empty:
            pass
        for k, v in latest.items():
            if k == "active_vehicle":
                self._on_vehicle_state(v)
            if k in VIEW_STATE_KEYS:
                for name, w in self._widgets.items():
                    if isinstance(w, BaseView) and w._mounted:
                        self._guard(lambda w=w, k=k, v=v: w.on_state_change(k, v), f"{name}.on_state_change")
        if "status_text" in latest:
            self._refresh_left()
        self._pump_id = self.after(PUMP_MS, self._pump)

    def _on_vehicle_state(self, name: Any) -> None:
        name = str(name)
        self.dock.set_vehicle(name)
        self.strip.set_vehicle(name)

    def _on_vehicle_pick(self, name: str) -> None:
        self.app_state.set("active_vehicle", name)

    # ------------------------------------------------------------------ chrome (ticker, status)
    def _tw(self) -> Any:
        return self._widgets.get("telemetry")

    def _link_state(self) -> tuple[str, str]:
        tw = self._tw()
        dbc = f" | DBC: {DBC_PATH.name}" if DBC_PATH.exists() else " | DBC: —"
        ing = getattr(tw, "_ingest", None)
        if ing is not None:
            try:
                s = ing.stats()
            except Exception:  # noqa: BLE001
                return ACCENT_AMBER, "CAN: STATS N/A" + dbc
            if s.get("error"):
                return ACCENT_RED, "CAN: ERROR" + dbc
            if s.get("connected"):
                return ACCENT_GREEN, f"CAN: ONLINE ({float(s.get('hz', 0.0)):.1f} Hz)" + dbc
            return ACCENT_AMBER, "CAN: DOWN" + dbc
        n = getattr(getattr(tw, "_t", None), "size", 0)
        if tw is not None and n and getattr(tw, "_mode", "offline") == "offline":
            return ACCENT_BLUE, "LOG: OFFLINE ANALYSIS" + dbc
        return TEXT_MUTED, "CAN: IDLE" + dbc

    def _solver_label(self) -> str:
        if self._solver_txt is not None:
            return self._solver_txt
        jax = sys.modules.get("jax")
        if jax is None:
            return "SOLVER: JAX n/c"
        try:
            backend = str(jax.default_backend()).upper()
        except Exception:  # noqa: BLE001
            backend = "?"
        try:
            x64 = bool(getattr(jax.config, "jax_enable_x64", False))
        except Exception:  # noqa: BLE001
            x64 = False
        self._solver_txt = f"SOLVER: JAX {backend} {'x64' if x64 else 'x32'}"
        self.dock.set_engine(f"JAX · {backend} · {'f64' if x64 else 'f32'} · 200 Hz Engine", ACCENT_GREEN)
        return self._solver_txt

    def _refresh_left(self) -> None:
        txt = str(self.app_state.get("status_text", "Listo"))
        tw = self._tw()
        if self._active_key == "telemetry" and tw is not None:
            n = int(getattr(getattr(tw, "_t", None), "size", 0) or 0)
            if n:
                txt += f"  ·  Log activo: {getattr(tw, '_log_name', '—')} | {n:,} muestras"
        self.status.set_left(txt)

    def _refresh_chrome(self) -> None:
        color, text = self._link_state()
        self.strip.set_link(color, text)
        self._refresh_left()
        buf = "BUFFER —"
        b = getattr(self._tw(), "_buffer", None)
        if b is not None:
            try:
                buf = f"BUFFER {100.0 * b.fill_fraction:4.1f}% ({b.count:,}/{b.capacity:,})"
            except Exception:  # noqa: BLE001
                pass
        mem = _rss_mb()
        mem_s = f"{mem:.0f} MB" if mem is not None else "n/a"
        self.status.set_center(f"{buf}  |  MEM {mem_s}  |  THR {threading.active_count()}",
                               ACCENT_AMBER if (mem or 0.0) > 3000 else TEXT_MUTED)
        try:
            draw = float(self.app_state.get("last_draw_ms", 0.0) or 0.0)
        except (TypeError, ValueError):
            draw = 0.0
        self.status.set_right(
            f"SYNC ● {time.strftime('%H:%M:%SZ', time.gmtime())}  |  {self._solver_label()}  |  DRAW {draw:.0f} ms",
            TEXT_PRIMARY)

    def _slow_tick(self) -> None:
        if not self._alive:
            return
        try:
            self._refresh_chrome()
        except Exception:  # noqa: BLE001  # el chrome nunca debe tumbar la app
            LOG.exception("refresh chrome failed")
        self._slow_id = self.after(SLOW_MS, self._slow_tick)

    # ------------------------------------------------------------------ errores / cierre
    def _on_callback_exception(self, exc: type[BaseException], val: BaseException,
                               tb: TracebackType | None) -> None:
        LOG.error("Excepción en callback de Tk", exc_info=(exc, val, tb))
        self.app_state.update_status(f"Error: {exc.__name__}: {val}")

    def close(self) -> None:
        if not self._alive:
            return
        self._alive = False
        for aid in (self._pump_id, self._slow_id):
            if aid is not None:
                try:
                    self.after_cancel(aid)
                except tk.TclError:
                    pass
        active = self._widgets.get(self._active_key) if self._active_key else None
        if isinstance(active, BaseView):
            self._guard(active.on_deactivate, "on_deactivate")
        for key in dict.fromkeys(VIEW_STATE_KEYS + SHELL_KEYS):
            self.app_state.unsubscribe(key, self._enqueue_state)
        self.destroy()


# ======================================================================================
# Selftest y entrada
# ======================================================================================
def selftest() -> int:
    """Comprobación sin ventana: contrato del tema + contrato de cada vista (importa, hereda BaseView, hooks)."""
    ok = True
    try:
        assert isinstance(T.TYRE_PALETTE, list) and len(T.TYRE_PALETTE) == 14
        assert all(isinstance(c, str) and c.startswith("#") for c in T.TYRE_PALETTE)
        for name in ("_ACCENT", "_SUCCESS", "_WARN", "_DANGER", "_PURPLE", "_BG", "_BG2", "_BG3", "_BORDER",
                     "_FG", "_FG2", "BG_DARK", "BG_CARD", "BG_HOVER", "BORDER", "TEXT_PRIMARY", "TEXT_MUTED",
                     "TEXT_BRIGHT", "ACCENT_BLUE", "ACCENT_GREEN", "ACCENT_AMBER", "ACCENT_RED", "ACCENT_PURPLE"):
            assert isinstance(getattr(T, name), str), name
        print("OK   theme")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print("FAIL theme", f"{type(exc).__name__}: {exc}")
    for spec in VIEWS:
        try:
            cls = getattr(importlib.import_module(spec.module), spec.cls)
            if not issubclass(cls, BaseView):
                raise TypeError("no hereda de BaseView")
            for hook in ("on_mount", "on_activate", "on_deactivate", "on_state_change"):
                if not callable(getattr(cls, hook, None)):
                    raise TypeError(f"falta {hook}")
            keys = getattr(getattr(cls, "SPEC", None), "state_keys", ())
            unknown = [k for k in keys if k not in VIEW_STATE_KEYS]
            if unknown:
                raise ValueError(f"state_keys no distribuidas por la shell: {unknown}")
            print("OK  ", spec.key)
        except Exception as exc:  # noqa: BLE001
            ok = False
            print("FAIL", spec.key, f"{type(exc).__name__}: {exc}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="TeR-Twin Studio")
    ap.add_argument("--view", choices=[v.key for v in VIEWS], default="telemetry")
    ap.add_argument("--index", type=Path, help="index.json de ajustes MF6.1 para neumáticos/correlación")
    ap.add_argument("--vehicle", help="perfil de vehículo activo (p. ej. TeR27-4WD)")
    ap.add_argument("--selftest", action="store_true", help="comprueba el contrato de tema y vistas sin ventana")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if a.selftest:
        return selftest()

    state = AppState.instance()
    if a.vehicle:
        state.set("active_vehicle", a.vehicle)
    if a.index:
        state.set("index_path", str(a.index))
    _enable_hidpi()
    TeRTwinApp(initial=a.view).mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())