"""apps/desktop/views/stub_base.py
Shared dark-card scaffold for the not-yet-implemented engineering modules.

A concrete stub only declares a ``StubSpec``; layout, scrolling, text wrapping and the live
AppState panel live here so the four stubs stay visually and behaviourally identical.

Assumptions
-----------
* ``StubSpec.state_keys`` must be a subset of the keys the shell fans out
  (``main_app.VIEW_STATE_KEYS``); ``main_app --selftest`` enforces it.
* Live values are display-only: the stub never writes to AppState.
* Mouse-wheel binding is global while the pointer is over the canvas and is released on
  ``on_deactivate`` so a hidden view can never hold the wheel.
"""
from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk
from typing import Any, ClassVar

from apps.desktop.state import AppState
from apps.desktop.theme import (
    ACCENT_AMBER, ACCENT_BLUE, BG_CARD, BG_DARK, TEXT_MUTED,
)
from apps.desktop.views.base_view import BaseView

__all__ = ["SpecRow", "StubSpec", "StubView", "format_state_value"]

_MONO = ("Consolas", 9)
_UI_BOLD = ("Segoe UI", 9, "bold")


@dataclass(frozen=True, slots=True)
class SpecRow:
    name: str
    detail: str


@dataclass(frozen=True, slots=True)
class StubSpec:
    title: str
    subtitle: str
    summary: str
    inputs: tuple[SpecRow, ...]
    outputs: tuple[SpecRow, ...]
    stack: tuple[SpecRow, ...]
    roadmap: tuple[str, ...]
    repo_paths: tuple[str, ...] = ()
    state_keys: tuple[str, ...] = ()
    status: str = "STUB · NOT WIRED"
    status_color: str = ACCENT_AMBER


def format_state_value(value: Any) -> str:
    """Compact, side-effect-free rendering of AppState values (duck-types MF61Params)."""
    if value is None:
        return "— (not set yet)"
    if hasattr(value, "FNOMIN") and hasattr(value, "R0") and hasattr(value, "NOMPRES"):
        try:
            return (f"MF6.1 · FNOMIN={float(value.FNOMIN):.0f} N · R0={float(value.R0) * 1e3:.1f} mm"
                    f" · NOMPRES={float(value.NOMPRES):.0f} kPa")
        except (TypeError, ValueError):
            return "MF6.1 params (batched / unreadable)"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(format_state_value(v) for v in value) + "]" if value else "[]"
    return str(value)


class StubView(BaseView):
    """BaseView that renders ``SPEC`` as a scrollable grid of dark cards."""

    SPEC: ClassVar[StubSpec]

    def __init__(self, parent: tk.Widget, app_state: AppState, **kwargs: Any) -> None:
        super().__init__(parent, app_state, **kwargs)
        self._live: dict[str, tk.StringVar] = {}
        self._canvas: tk.Canvas | None = None

    # ------------------------------------------------------------------ lifecycle
    def on_mount(self) -> None:
        canvas = tk.Canvas(self, bg=BG_DARK, highlightthickness=0, bd=0)
        sb = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        self._canvas = canvas

        body = ttk.Frame(canvas)
        win = canvas.create_window((0, 0), window=body, anchor="nw")
        body.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(win, width=e.width))
        canvas.bind("<Enter>", lambda _e: self._bind_wheel(True))
        canvas.bind("<Leave>", lambda _e: self._bind_wheel(False))
        body.columnconfigure((0, 1), weight=1, uniform="col")

        s = self.SPEC
        self._build_header(body)
        self._rows_card(body, 1, 0, "Inputs from the vehicle model", s.inputs)
        self._rows_card(body, 1, 1, "Outputs / consumers", s.outputs)
        self._rows_card(body, 2, 0, "Technical specification", s.stack)
        self._roadmap_card(body, 2, 1, s.roadmap)
        self._live_card(body, 3, 0, s.state_keys)
        self._repo_card(body, 3, 1, s.repo_paths)
        self._refresh_live()

    def on_activate(self) -> None:
        self._refresh_live()

    def on_deactivate(self) -> None:
        self._bind_wheel(False)
        super().on_deactivate()

    def on_state_change(self, key: str, value: Any) -> None:
        var = self._live.get(key)
        if var is not None:
            var.set(format_state_value(value))

    # ------------------------------------------------------------------ building blocks
    @staticmethod
    def _autowrap(card: ttk.Frame, labels: list[ttk.Label]) -> None:
        def _apply(e: tk.Event) -> None:
            wrap = max(140, e.width - 44)
            for lbl in labels:
                lbl.configure(wraplength=wrap)

        card.bind("<Configure>", _apply)

    @staticmethod
    def _card(parent: tk.Widget, row: int, col: int, title: str, colspan: int = 1) -> ttk.Frame:
        card = ttk.Frame(parent, style="Card.TFrame", padding=14)
        card.grid(row=row, column=col, columnspan=colspan, sticky="nsew", padx=8, pady=8)
        ttk.Label(card, text=title, style="CardH.TLabel").pack(anchor="w", pady=(0, 6))
        return card

    def _build_header(self, body: ttk.Frame) -> None:
        s = self.SPEC
        card = ttk.Frame(body, style="Card.TFrame", padding=18)
        card.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=8, pady=(12, 8))
        top = ttk.Frame(card, style="Card.TFrame")
        top.pack(fill="x")
        ttk.Label(top, text=s.title, style="CardH.TLabel", font=("Segoe UI", 18, "bold")).pack(side="left")
        tk.Label(top, text=f" {s.status} ", bg=s.status_color, fg=BG_DARK, font=("Segoe UI", 8, "bold"),
                 padx=6, pady=2).pack(side="left", padx=12)
        ttk.Label(card, text=s.subtitle, style="CardS.TLabel").pack(anchor="w", pady=(2, 10))
        summary = ttk.Label(card, text=s.summary, style="Card.TLabel", justify="left")
        summary.pack(anchor="w", fill="x")
        self._autowrap(card, [summary])

    def _rows_card(self, body: ttk.Frame, row: int, col: int, title: str, rows: tuple[SpecRow, ...]) -> None:
        card = self._card(body, row, col, title)
        wraps: list[ttk.Label] = []
        for r in rows:
            ttk.Label(card, text=r.name, style="Card.TLabel", font=_UI_BOLD).pack(anchor="w", pady=(6, 0))
            d = ttk.Label(card, text=r.detail, style="CardS.TLabel", justify="left")
            d.pack(anchor="w", fill="x")
            wraps.append(d)
        self._autowrap(card, wraps)

    def _roadmap_card(self, body: ttk.Frame, row: int, col: int, items: tuple[str, ...]) -> None:
        card = self._card(body, row, col, "Implementation roadmap")
        wraps: list[ttk.Label] = []
        for i, item in enumerate(items, 1):
            lbl = ttk.Label(card, text=f"{i}.  {item}", style="Card.TLabel", justify="left")
            lbl.pack(anchor="w", fill="x", pady=(4, 0))
            wraps.append(lbl)
        self._autowrap(card, wraps)

    def _live_card(self, body: ttk.Frame, row: int, col: int, keys: tuple[str, ...]) -> None:
        card = self._card(body, row, col, "Live shared state (AppState)")
        if not keys:
            ttk.Label(card, text="This module consumes no shared state keys.", style="CardS.TLabel").pack(anchor="w")
            return
        wraps: list[ttk.Label] = []
        for key in keys:
            tk.Label(card, text=key, bg=BG_CARD, fg=ACCENT_BLUE, font=_MONO, anchor="w").pack(
                anchor="w", pady=(6, 0))
            var = tk.StringVar(value="—")
            lbl = ttk.Label(card, textvariable=var, style="Card.TLabel", justify="left")
            lbl.pack(anchor="w", fill="x")
            wraps.append(lbl)
            self._live[key] = var
        self._autowrap(card, wraps)

    def _repo_card(self, body: ttk.Frame, row: int, col: int, paths: tuple[str, ...]) -> None:
        card = self._card(body, row, col, "Backing repository modules")
        if not paths:
            ttk.Label(card, text="—", style="CardS.TLabel").pack(anchor="w")
            return
        for p in paths:
            tk.Label(card, text=p, bg=BG_CARD, fg=TEXT_MUTED, font=_MONO, anchor="w").pack(anchor="w", pady=1)

    # ------------------------------------------------------------------ state / scrolling
    def _refresh_live(self) -> None:
        for key, var in self._live.items():
            var.set(format_state_value(self._app_state.get(key)))

    def _bind_wheel(self, on: bool) -> None:
        cv = self._canvas
        if cv is None:
            return
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            if on:
                cv.bind_all(seq, self._on_wheel)
            else:
                cv.unbind_all(seq)

    def _on_wheel(self, e: tk.Event) -> None:
        if self._canvas is not None:
            up = getattr(e, "delta", 0) > 0 or getattr(e, "num", 0) == 4
            self._canvas.yview_scroll(-1 if up else 1, "units")