#!/usr/bin/env python3
"""apps/desktop/main_app.py
TeR-Twin Studio shell: sidebar navigation, lazy view router, status bar, thread-safe state fan-out.

Run from the repository root:
    python -m apps.desktop.main_app [--view tires|suspension|aero|lts|telemetry]
                                    [--index data/processed/mf61_fits/index.json]
                                    [--vehicle TeR27-4WD] [--selftest]

Assumptions
-----------
* Views are imported and mounted lazily on first navigation, so a missing optional dependency
  (e.g. jax for the tyre view) shows an error card with a retry button instead of aborting startup.
* The active view gets on_activate / on_deactivate; every mounted view (visible or not) receives
  on_state_change so background modules stay current.
* AppState.set may be called from worker threads. Callbacks only enqueue; a 30 ms Tk-side pump
  dispatches, coalescing to the latest value per key (intermediate values are dropped).
* ``self.app_state`` is deliberately not called ``state``: tk.Tk.state() is the window-state method.
"""
from __future__ import annotations

import argparse
import importlib
import logging
import queue
import sys
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

from apps.desktop.components.sidebar import Sidebar  # noqa: E402
from apps.desktop.components.status_bar import StatusBar  # noqa: E402
from apps.desktop.state import AppState  # noqa: E402
from apps.desktop.theme import ACCENT_RED, BG_DARK, TEXT_MUTED, apply_global_theme  # noqa: E402
from apps.desktop.views.base_view import BaseView  # noqa: E402

LOG = logging.getLogger("ter_twin_studio")

SIDEBAR_WIDTH = 232
PUMP_MS = 30

# State keys fanned out to views. status_text / last_draw_ms are UI telemetry for the status bar only.
VIEW_STATE_KEYS: tuple[str, ...] = (
    "active_vehicle", "front_tyre_params", "rear_tyre_params", "active_tyre_slugs",
    "nominal_fz", "aero_balance", "roll_stiffness_dist", "index_path",
)


@dataclass(frozen=True, slots=True)
class ViewSpec:
    key: str
    title: str
    label: str
    module: str
    cls: str


VIEWS: tuple[ViewSpec, ...] = (
    ViewSpec("tires", "Tyre Model MF6.1", "🏎️  Tyre Model MF6.1", "apps.desktop.views.tires_view", "TiresView"),
    ViewSpec("suspension", "Suspension Kinematics", "📐  Suspension Kinematics",
             "apps.desktop.views.suspension_view", "SuspensionView"),
    ViewSpec("aero", "Aerodynamic Maps", "🪽  Aerodynamic Maps", "apps.desktop.views.aerodynamics_view", "AeroView"),
    ViewSpec("lts", "Lap Time Simulator & MMM", "⏱️  Lap Time Simulator & MMM",
             "apps.desktop.views.lts_view", "LTSView"),
    ViewSpec("telemetry", "Realtime Telemetry", "📡  Realtime Telemetry",
             "apps.desktop.views.telemetry_view", "TelemetryView"),
)


def _enable_hidpi() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        pass


class TeRTwinApp(tk.Tk):
    def __init__(self, initial: str = "tires") -> None:
        super().__init__()
        self._specs: dict[str, ViewSpec] = {v.key: v for v in VIEWS}
        if initial not in self._specs:
            raise KeyError(f"vista desconocida: {initial!r}")
        self.title("TeR-Twin · Vehicle Dynamics & Simulation Suite")
        self.geometry("1600x960")
        self.minsize(1280, 720)
        self.configure(bg=BG_DARK)

        self.app_state = AppState.instance()
        apply_global_theme(self)

        self._widgets: dict[str, tk.Widget] = {}
        self._active_key: str | None = None
        self._alive = True
        self._pump_id: str | None = None
        self._events: queue.SimpleQueue[tuple[str, Any]] = queue.SimpleQueue()

        self.report_callback_exception = self._on_callback_exception  # type: ignore[assignment]
        self._build()
        for key in VIEW_STATE_KEYS:
            self.app_state.subscribe(key, self._enqueue_state)
        self.protocol("WM_DELETE_WINDOW", self.close)
        for i, v in enumerate(VIEWS, 1):
            self.bind(f"<Control-Key-{i}>", lambda _e, k=v.key: self.navigate(k))
        self._pump_id = self.after(PUMP_MS, self._pump)
        self.navigate(initial)

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        self.sidebar = Sidebar(self, self._show, [(v.key, v.label) for v in VIEWS])
        self.sidebar.configure(width=SIDEBAR_WIDTH)
        self.sidebar.grid_propagate(False)
        self.sidebar.grid(row=0, column=0, sticky="ns")

        self.content = ttk.Frame(self)
        self.content.grid(row=0, column=1, sticky="nsew")

        self.status_bar = StatusBar(self, self.app_state)
        self.status_bar.grid(row=1, column=0, columnspan=2, sticky="ew")

    # ------------------------------------------------------------------ routing
    def navigate(self, key: str) -> None:
        if key not in self._specs:
            raise KeyError(f"vista desconocida: {key!r}")
        self.sidebar.set_active(key)
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
        self.sidebar.set_active(key)
        self.app_state.update_status(spec.title)
        if isinstance(widget, BaseView):
            self._guard(widget.on_activate, f"{key}.on_activate")

    def _instantiate(self, spec: ViewSpec) -> tk.Widget:
        self.app_state.update_status(f"Cargando {spec.title}…")
        self.update_idletasks()
        t0 = time.perf_counter()
        view: BaseView | None = None
        try:
            cls = getattr(importlib.import_module(spec.module), spec.cls)
            view = cls(self.content, self.app_state)
            view._do_mount()
        except Exception as exc:  # noqa: BLE001  # isolate a broken module from the shell
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
        txt = tk.Text(frame, height=18, bg=BG_DARK, fg=TEXT_MUTED, relief="flat", font=("Consolas", 9),
                      wrap="none", highlightthickness=0)
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
        except Exception as exc:  # noqa: BLE001  # lifecycle hooks must not kill the event loop
            LOG.exception("Fallo en %s", what)
            self.app_state.update_status(f"Error en {what}: {exc}")
            return False

    # ------------------------------------------------------------------ state fan-out
    def _enqueue_state(self, key: str, value: Any) -> None:
        self._events.put((key, value))  # thread-safe; never touches Tk

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
            for name, w in self._widgets.items():
                if isinstance(w, BaseView) and w._mounted:
                    self._guard(lambda w=w, k=k, v=v: w.on_state_change(k, v), f"{name}.on_state_change")
        self._pump_id = self.after(PUMP_MS, self._pump)

    # ------------------------------------------------------------------ errors / shutdown
    def _on_callback_exception(self, exc: type[BaseException], val: BaseException,
                               tb: TracebackType | None) -> None:
        LOG.error("Excepción en callback de Tk", exc_info=(exc, val, tb))
        self.app_state.update_status(f"Error: {exc.__name__}: {val}")

    def close(self) -> None:
        if not self._alive:
            return
        self._alive = False
        if self._pump_id is not None:
            try:
                self.after_cancel(self._pump_id)
            except tk.TclError:
                pass
        active = self._widgets.get(self._active_key) if self._active_key else None
        if isinstance(active, BaseView):
            self._guard(active.on_deactivate, "on_deactivate")
        for key in VIEW_STATE_KEYS:
            self.app_state.unsubscribe(key, self._enqueue_state)
        self.destroy()


def selftest() -> int:
    """Display-free contract check: every view imports, subclasses BaseView and declares valid state keys."""
    ok = True
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
    ap.add_argument("--view", choices=[v.key for v in VIEWS], default="tires")
    ap.add_argument("--index", type=Path, help="index.json de ajustes MF6.1 para la vista de neumáticos")
    ap.add_argument("--vehicle", help="perfil de vehículo activo (p. ej. TeR27-4WD)")
    ap.add_argument("--selftest", action="store_true", help="comprueba el contrato de las vistas sin abrir ventana")
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