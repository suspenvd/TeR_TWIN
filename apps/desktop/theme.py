"""apps/desktop/theme.py
Centralized GitHub-Dark design system for TeR-Twin Studio.
Provides color constants, ttk style configuration, and Matplotlib rcParams.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import matplotlib

# ---------------------------------------------------------------------------
# Color tokens — GitHub Dark palette
# ---------------------------------------------------------------------------
BG_DARK   = "#0d1117"
BG_CARD   = "#161b22"
BG_HOVER  = "#21262d"
BORDER    = "#30363d"

TEXT_PRIMARY = "#c9d1d9"
TEXT_MUTED   = "#8b949e"
TEXT_BRIGHT  = "#ffffff"

ACCENT_BLUE   = "#58a6ff"
ACCENT_GREEN  = "#3fb950"
ACCENT_AMBER  = "#d29922"
ACCENT_RED    = "#f85149"
ACCENT_PURPLE = "#bc8cff"

# Short aliases used throughout the original tire_terminal
_ACCENT  = ACCENT_BLUE
_SUCCESS = ACCENT_GREEN
_WARN    = ACCENT_AMBER
_DANGER  = ACCENT_RED
_PURPLE  = ACCENT_PURPLE
_BG      = BG_DARK
_BG2     = BG_CARD
_BG3     = BG_HOVER
_BORDER  = BORDER
_FG      = TEXT_PRIMARY
_FG2     = TEXT_MUTED

# ---------------------------------------------------------------------------
# Tyre-list palette (14 high-contrast slugs)
# ---------------------------------------------------------------------------
TYRE_PALETTE = [
    "#58a6ff",  # electric blue
    "#f78166",  # coral
    "#3fb950",  # lime green
    "#bc8cff",  # violet
    "#d29922",  # amber
    "#79c0ff",  # sky blue
    "#ff7b72",  # salmon red
    "#39d353",  # mint green
    "#f0883e",  # deep orange
    "#d2a8ff",  # lavender
    "#56d364",  # emerald
    "#db61a2",  # magenta
    "#e3b341",  # warm yellow
    "#a5d6ff",  # ice blue
]


# ---------------------------------------------------------------------------
# Matplotlib integration
# ---------------------------------------------------------------------------
def apply_matplotlib_theme() -> None:
    """Patch global Matplotlib rcParams so embedded plots match the dark theme."""
    matplotlib.rcParams.update({
        "figure.facecolor":  BG_DARK,
        "axes.facecolor":    BG_CARD,
        "axes.edgecolor":    BORDER,
        "axes.labelcolor":   TEXT_PRIMARY,
        "axes.titlecolor":   TEXT_BRIGHT,
        "xtick.color":       TEXT_MUTED,
        "ytick.color":       TEXT_MUTED,
        "text.color":        TEXT_PRIMARY,
        "grid.color":        BG_HOVER,
        "grid.linewidth":    0.6,
        "legend.facecolor":  BG_CARD,
        "legend.edgecolor":  BORDER,
        "legend.fontsize":   7,
        "lines.linewidth":   1.6,
        "font.family":       "sans-serif",
        "font.size":         9,
    })


# ---------------------------------------------------------------------------
# ttk styling
# ---------------------------------------------------------------------------
def apply_global_theme(root: tk.Tk) -> ttk.Style:
    """Configure ttk.Style with the GitHub-Dark design system.

    Returns the configured Style object so callers can extend it.
    """
    style = ttk.Style(root)
    style.theme_use("clam")

    style.configure(
        ".",
        background=BG_DARK,
        foreground=TEXT_PRIMARY,
        fieldbackground=BG_CARD,
        bordercolor=BORDER,
        troughcolor=BG_HOVER,
        selectbackground=ACCENT_BLUE,
        selectforeground=BG_DARK,
        font=("Segoe UI", 9),
    )

    style.configure("TFrame",  background=BG_DARK)
    style.configure("TLabel",  background=BG_DARK, foreground=TEXT_PRIMARY)

    style.configure(
        "TButton",
        background=BG_HOVER,
        foreground=TEXT_PRIMARY,
        bordercolor=BORDER,
        focuscolor=ACCENT_BLUE,
        padding=(8, 4),
    )
    style.map(
        "TButton",
        background=[("active", ACCENT_BLUE), ("pressed", "#1f6feb")],
        foreground=[("active", BG_DARK)],
    )

    style.configure(
        "TCheckbutton",
        background=BG_DARK,
        foreground=TEXT_PRIMARY,
        indicatorbackground=BG_CARD,
        indicatorcolor=ACCENT_BLUE,
    )
    style.map("TCheckbutton", background=[("active", BG_HOVER)])

    style.configure(
        "TCombobox",
        fieldbackground=BG_CARD,
        background=BG_CARD,
        foreground=TEXT_PRIMARY,
        arrowcolor=TEXT_MUTED,
        selectbackground=ACCENT_BLUE,
        selectforeground=BG_DARK,
    )
    style.map("TCombobox", fieldbackground=[("readonly", BG_CARD)])

    style.configure(
        "TEntry",
        fieldbackground=BG_CARD,
        foreground=TEXT_PRIMARY,
        insertcolor=TEXT_PRIMARY,
        bordercolor=BORDER,
    )

    style.configure(
        "TLabelframe",
        background=BG_DARK,
        foreground=TEXT_MUTED,
        bordercolor=BORDER,
    )
    style.configure(
        "TLabelframe.Label",
        background=BG_DARK,
        foreground=ACCENT_BLUE,
        font=("Segoe UI", 9, "bold"),
    )

    style.configure("TNotebook", background=BG_DARK, tabmargins=0)
    style.configure(
        "TNotebook.Tab",
        background=BG_HOVER,
        foreground=TEXT_MUTED,
        padding=(8, 4),
        bordercolor=BORDER,
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", BG_CARD)],
        foreground=[("selected", ACCENT_BLUE)],
    )

    style.configure(
        "TScale",
        background=BG_DARK,
        troughcolor=BG_HOVER,
        sliderthickness=14,
        slidercolor=ACCENT_BLUE,
    )

    style.configure(
        "Treeview",
        background=BG_CARD,
        foreground=TEXT_PRIMARY,
        fieldbackground=BG_CARD,
        rowheight=22,
        bordercolor=BORDER,
    )
    style.configure(
        "Treeview.Heading",
        background=BG_HOVER,
        foreground=TEXT_MUTED,
        bordercolor=BORDER,
    )
    style.map(
        "Treeview",
        background=[("selected", ACCENT_BLUE)],
        foreground=[("selected", BG_DARK)],
    )

    style.configure(
        "TScrollbar",
        background=BG_HOVER,
        troughcolor=BG_DARK,
        arrowcolor=TEXT_MUTED,
        bordercolor=BORDER,
    )

    # --- Sidebar nav button variants ----------------------------------------
    style.configure(
        "Nav.TButton",
        background=BG_DARK,
        foreground=TEXT_MUTED,
        bordercolor=BG_DARK,
        focuscolor=BG_DARK,
        anchor="w",
        padding=(12, 8),
        font=("Segoe UI", 10),
    )
    style.map(
        "Nav.TButton",
        background=[("active", BG_HOVER), ("pressed", BG_CARD)],
        foreground=[("active", TEXT_PRIMARY)],
    )
    style.configure(
        "NavActive.TButton",
        background=BG_CARD,
        foreground=ACCENT_BLUE,
        bordercolor=ACCENT_BLUE,
        focuscolor=BG_CARD,
        anchor="w",
        padding=(12, 8),
        font=("Segoe UI", 10, "bold"),
    )
    style.map(
        "NavActive.TButton",
        background=[("active", BG_HOVER)],
        foreground=[("active", ACCENT_BLUE)],
    )

    # --- Status bar ---------------------------------------------------------
    style.configure(
        "Status.TLabel",
        background=BG_HOVER,
        foreground=TEXT_MUTED,
        relief="flat",
        padding=(6, 2),
    )

    # --- Stub card ----------------------------------------------------------
    style.configure("Card.TFrame",  background=BG_CARD,  relief="flat")
    style.configure("Card.TLabel",  background=BG_CARD,  foreground=TEXT_PRIMARY)
    style.configure("CardH.TLabel", background=BG_CARD,  foreground=ACCENT_BLUE,
                    font=("Segoe UI", 13, "bold"))
    style.configure("CardS.TLabel", background=BG_CARD,  foreground=TEXT_MUTED,
                    font=("Segoe UI", 9))
    style.configure("Tag.TLabel",   background=BG_HOVER, foreground=TEXT_MUTED,
                    padding=(4, 2), relief="flat",
                    font=("Segoe UI", 8))

    apply_matplotlib_theme()
    return style
