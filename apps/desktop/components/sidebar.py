"""apps/desktop/components/sidebar.py
Collapsible navigation drawer for TeR-Twin Studio.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable

from apps.desktop.theme import (
    ACCENT_BLUE, BG_CARD, BG_DARK, BG_HOVER,
    BORDER, TEXT_MUTED, TEXT_PRIMARY,
)


class Sidebar(ttk.Frame):
    """Left navigation panel with per-module buttons and active-state highlight.

    Parameters
    ----------
    parent:
        Parent container widget.
    on_select:
        Callback invoked with the module key (str) when user clicks a nav item.
    nav_items:
        Ordered sequence of ``(key, label)`` tuples defining the module list.
    """

    def __init__(
        self,
        parent: tk.Widget,
        on_select: Callable[[str], None],
        nav_items: list[tuple[str, str]],
        **kwargs: tk.Any,
    ) -> None:
        super().__init__(parent, style="TFrame", **kwargs)
        self._on_select = on_select
        self._nav_items = nav_items
        self._buttons: dict[str, ttk.Button] = {}
        self._active_key: str | None = None
        self._build()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        self.configure(style="TFrame")
        self.columnconfigure(0, weight=1)

        # ── Logo / header ──────────────────────────────────────────────
        header = tk.Frame(self, bg=BG_DARK, height=64)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_propagate(False)

        logo_line1 = tk.Label(
            header,
            text="TeR-TWIN",
            bg=BG_DARK,
            fg=ACCENT_BLUE,
            font=("Segoe UI", 14, "bold"),
            anchor="w",
        )
        logo_line1.pack(anchor="w", padx=14, pady=(14, 0))

        logo_line2 = tk.Label(
            header,
            text="STUDIO",
            bg=BG_DARK,
            fg=TEXT_MUTED,
            font=("Segoe UI", 9),
            anchor="w",
        )
        logo_line2.pack(anchor="w", padx=16, pady=(0, 2))

        # ── Thin accent ruler ──────────────────────────────────────────
        sep = tk.Frame(self, bg=BORDER, height=1)
        sep.grid(row=1, column=0, sticky="ew")

        # ── Nav buttons ────────────────────────────────────────────────
        nav_container = ttk.Frame(self)
        nav_container.grid(row=2, column=0, sticky="nsew", pady=(8, 0))
        nav_container.columnconfigure(0, weight=1)

        for row_idx, (key, label) in enumerate(self._nav_items):
            btn = ttk.Button(
                nav_container,
                text=label,
                style="Nav.TButton",
                command=lambda k=key: self._handle_click(k),
                cursor="hand2",
            )
            btn.grid(row=row_idx, column=0, sticky="ew", padx=0, pady=1)
            self._buttons[key] = btn

        # ── Bottom version tag ─────────────────────────────────────────
        self.rowconfigure(3, weight=1)
        bottom = tk.Frame(self, bg=BG_DARK)
        bottom.grid(row=3, column=0, sticky="sew")
        tk.Label(
            bottom,
            text="TeR-Twin Studio  v2.0",
            bg=BG_DARK,
            fg=TEXT_MUTED,
            font=("Segoe UI", 7),
        ).pack(side="bottom", pady=6)

    # ------------------------------------------------------------------
    def _handle_click(self, key: str) -> None:
        self.set_active(key)
        self._on_select(key)

    def set_active(self, key: str) -> None:
        """Visually highlight *key* as the active nav item."""
        if self._active_key == key:
            return
        # De-highlight previous
        if self._active_key and self._active_key in self._buttons:
            self._buttons[self._active_key].configure(style="Nav.TButton")
        # Highlight new
        if key in self._buttons:
            self._buttons[key].configure(style="NavActive.TButton")
        self._active_key = key
