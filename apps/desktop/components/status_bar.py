"""apps/desktop/components/status_bar.py
Shared bottom system bar for TeR-Twin Studio.

Displays: active vehicle profile, dataset (index.json) path, background compute latency, status text.

Fixes vs. the previous revision
-------------------------------
* ``_refresh_index`` computed ``str(p.parent.name / p.name)`` (str / str -> TypeError). That crashed
  startup whenever ``index_path`` was already set (``--index``) and was silently swallowed by
  AppState.set afterwards, leaving the label stuck on "demo mode".
* AppState callbacks may fire on worker threads, but Tk widgets are main-thread only. Updates from
  the Tk thread apply immediately (so "Calculando…" shows before a blocking redraw); updates from
  other threads are queued and applied by a 100 ms poll, coalesced to the latest value per key.
"""
from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Any, Callable

from apps.desktop.state import AppState
from apps.desktop.theme import (
    ACCENT_BLUE, ACCENT_GREEN, BG_HOVER, BORDER, TEXT_MUTED, TEXT_PRIMARY,
)

__all__ = ["StatusBar"]


class StatusBar(ttk.Frame):
    """Horizontal status strip pinned to the bottom of the main window."""

    POLL_MS = 100
    KEYS: tuple[str, ...] = ("active_vehicle", "index_path", "status_text", "last_draw_ms")

    def __init__(self, parent: tk.Widget, app_state: AppState, **kwargs: Any) -> None:
        super().__init__(parent, style="TFrame", **kwargs)
        self._state = app_state
        self._main_ident = threading.get_ident()
        self._lock = threading.Lock()
        self._pending: dict[str, Any] = {}
        self._poll_id: str | None = None
        self._handlers: dict[str, Callable[[Any], None]] = {}
        self._build()
        self._handlers = {
            "active_vehicle": self._refresh_vehicle,
            "index_path": self._refresh_index,
            "status_text": self._refresh_status,
            "last_draw_ms": self._refresh_latency,
        }
        for key in self.KEYS:
            self._handlers[key](self._state.get(key, self._default(key)))
            self._state.subscribe(key, self._on_state)
        self._poll_id = self.after(self.POLL_MS, self._poll)
        self.bind("<Destroy>", self._on_destroy, add="+")

    @staticmethod
    def _default(key: str) -> Any:
        return {"active_vehicle": "", "status_text": "Listo", "last_draw_ms": 0.0}.get(key)

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        tk.Frame(self, bg=BORDER, height=1).pack(side="top", fill="x")
        bar = tk.Frame(self, bg=BG_HOVER, height=26)
        bar.pack(fill="both", expand=True)
        bar.pack_propagate(False)

        self._lbl_vehicle = tk.Label(bar, text="", bg=BG_HOVER, fg=ACCENT_BLUE,
                                     font=("Segoe UI", 8, "bold"), anchor="w")
        self._lbl_vehicle.pack(side="left", padx=(10, 0), pady=3)
        tk.Label(bar, text=" │ ", bg=BG_HOVER, fg=BORDER).pack(side="left")
        self._lbl_dataset = tk.Label(bar, text="", bg=BG_HOVER, fg=TEXT_MUTED, font=("Segoe UI", 8), anchor="w")
        self._lbl_dataset.pack(side="left")

        self._lbl_latency = tk.Label(bar, text="", bg=BG_HOVER, fg=ACCENT_GREEN, font=("Segoe UI", 8), anchor="e")
        self._lbl_latency.pack(side="right", padx=(0, 10))
        tk.Label(bar, text=" │ ", bg=BG_HOVER, fg=BORDER).pack(side="right")

        self._lbl_status = tk.Label(bar, text="Listo", bg=BG_HOVER, fg=TEXT_PRIMARY,
                                    font=("Segoe UI", 8), anchor="w")
        self._lbl_status.pack(side="left", padx=(4, 0))

    # ------------------------------------------------------------------ thread-safe plumbing
    def _on_state(self, key: str, value: Any) -> None:
        if threading.get_ident() == self._main_ident:
            self._apply(key, value)
            return
        with self._lock:
            self._pending[key] = value

    def _apply(self, key: str, value: Any) -> None:
        handler = self._handlers.get(key)
        if handler is None:
            return
        try:
            handler(value)
        except tk.TclError:  # widget already destroyed
            pass

    def _poll(self) -> None:
        with self._lock:
            batch, self._pending = self._pending, {}
        for key, value in batch.items():
            self._apply(key, value)
        self._poll_id = self.after(self.POLL_MS, self._poll)

    def _on_destroy(self, event: tk.Event) -> None:
        if event.widget is not self:
            return
        if self._poll_id is not None:
            try:
                self.after_cancel(self._poll_id)
            except tk.TclError:
                pass
            self._poll_id = None
        for key in self.KEYS:
            self._state.unsubscribe(key, self._on_state)

    # ------------------------------------------------------------------ refreshers
    def _refresh_vehicle(self, value: Any) -> None:
        self._lbl_vehicle.configure(text=f"🏎  {value}")

    def _refresh_index(self, value: Any) -> None:
        if value is None:
            self._lbl_dataset.configure(text="demo mode")
            return
        p = Path(str(value))
        self._lbl_dataset.configure(text=f"{p.parent.name}/{p.name}" if p.parent.name else p.name)

    def _refresh_status(self, value: Any) -> None:
        self._lbl_status.configure(text=str(value))

    def _refresh_latency(self, value: Any) -> None:
        try:
            ms = float(value)
        except (TypeError, ValueError):
            ms = 0.0
        self._lbl_latency.configure(text=f"⚡ {ms:.0f} ms" if ms > 0 else "")