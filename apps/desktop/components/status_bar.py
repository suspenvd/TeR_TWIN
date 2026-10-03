"""apps/desktop/components/status_bar.py
Shared bottom system bar for TeR-Twin Studio.

Displays:
  • Active vehicle profile
  • Dataset (index.json) path
  • Background compute latency
  • Live status / error messages
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable

from apps.desktop.state import AppState
from apps.desktop.theme import (
    ACCENT_BLUE, ACCENT_GREEN, BG_HOVER, BORDER, TEXT_MUTED, TEXT_PRIMARY,
)


class StatusBar(ttk.Frame):
    """Horizontal status strip pinned to the bottom of the main window.

    The bar subscribes to ``AppState`` keys and auto-updates without polling.
    """

    def __init__(
        self,
        parent: tk.Widget,
        app_state: AppState,
        **kwargs: tk.Any,
    ) -> None:
        super().__init__(parent, style="TFrame", **kwargs)
        self._state = app_state
        self._build()
        self._subscribe()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        self.configure(style="TFrame")

        # Thin top separator
        sep = tk.Frame(self, bg=BORDER, height=1)
        sep.pack(side="top", fill="x")

        bar = tk.Frame(self, bg=BG_HOVER, height=26)
        bar.pack(fill="both", expand=True)
        bar.pack_propagate(False)

        # Left: vehicle + dataset
        self._lbl_vehicle = tk.Label(
            bar,
            text="",
            bg=BG_HOVER, fg=ACCENT_BLUE,
            font=("Segoe UI", 8, "bold"),
            anchor="w",
        )
        self._lbl_vehicle.pack(side="left", padx=(10, 0), pady=3)

        self._lbl_sep1 = tk.Label(bar, text=" │ ", bg=BG_HOVER, fg=BORDER)
        self._lbl_sep1.pack(side="left")

        self._lbl_dataset = tk.Label(
            bar,
            text="",
            bg=BG_HOVER, fg=TEXT_MUTED,
            font=("Segoe UI", 8),
            anchor="w",
        )
        self._lbl_dataset.pack(side="left")

        # Right: compute latency
        self._lbl_latency = tk.Label(
            bar,
            text="",
            bg=BG_HOVER, fg=ACCENT_GREEN,
            font=("Segoe UI", 8),
            anchor="e",
        )
        self._lbl_latency.pack(side="right", padx=(0, 10))

        self._lbl_sep2 = tk.Label(bar, text=" │ ", bg=BG_HOVER, fg=BORDER)
        self._lbl_sep2.pack(side="right")

        # Center: status message
        self._lbl_status = tk.Label(
            bar,
            text="Listo",
            bg=BG_HOVER, fg=TEXT_PRIMARY,
            font=("Segoe UI", 8),
            anchor="w",
        )
        self._lbl_status.pack(side="left", padx=(4, 0))

        # Initial populate
        self._refresh_vehicle(None, self._state.get("active_vehicle", ""))
        self._refresh_index(None, self._state.get("index_path"))
        self._refresh_status(None, self._state.get("status_text", "Listo"))
        self._refresh_latency(None, self._state.get("last_draw_ms", 0.0))

    # ------------------------------------------------------------------
    def _subscribe(self) -> None:
        self._state.subscribe("active_vehicle", self._refresh_vehicle)
        self._state.subscribe("index_path",     self._refresh_index)
        self._state.subscribe("status_text",    self._refresh_status)
        self._state.subscribe("last_draw_ms",   self._refresh_latency)

    # ------------------------------------------------------------------
    def _refresh_vehicle(self, _key: str | None, value: str) -> None:
        self._lbl_vehicle.configure(text=f"🏎  {value}")

    def _refresh_index(self, _key: str | None, value: object) -> None:
        if value is None:
            self._lbl_dataset.configure(text="demo mode")
        else:
            # Show only the last two path components to save space
            from pathlib import Path
            p = Path(str(value))
            self._lbl_dataset.configure(text=str(p.parent.name / p.name))

    def _refresh_status(self, _key: str | None, value: str) -> None:
        self._lbl_status.configure(text=str(value))

    def _refresh_latency(self, _key: str | None, value: float) -> None:
        if value and float(value) > 0:
            self._lbl_latency.configure(text=f"⚡ {value:.0f} ms")
        else:
            self._lbl_latency.configure(text="")
