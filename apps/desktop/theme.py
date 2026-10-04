"""apps/desktop/theme.py
Sistema de diseño centralizado de TeR-Twin Studio (estética MoTeC i2 Pro / Bloomberg Terminal).

Compatibilidad hacia atrás (contrato con tires_view, telemetry_view, tire_terminal, sidebar, status_bar):
* Se conservan TODOS los tokens (BG_DARK, BG_CARD, BG_HOVER, BORDER, TEXT_*, ACCENT_*) y los alias cortos
  (_ACCENT, _SUCCESS, _WARN, _DANGER, _PURPLE, _BG, _BG2, _BG3, _BORDER, _FG, _FG2).
* TYRE_PALETTE sigue siendo list[str] (14 hex).
* apply_global_theme(root) devuelve el ttk.Style configurado; apply_matplotlib_theme() sigue existiendo.
* Todos los estilos TTK previos siguen definidos (Nav.TButton, NavActive.TButton, Card.*, Tag.TLabel,
  Status.TLabel, TLabelframe(.Label), TNotebook(.Tab), Treeview(.Heading), TScrollbar, Vertical.TScrollbar...).

Notas
-----
* Tk solo admite tamaños de fuente enteros en puntos: 8.5 pt -> 9, 7.5 pt -> 8, 10-11 pt -> 11 (titulos).
* La fuente real se resuelve en apply_global_theme (JetBrains Mono > Consolas > DejaVu Sans Mono > Courier New).
  Usa mono()/ui() en vistas nuevas para obtener tuplas de fuente ya resueltas.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk
from typing import Any

import matplotlib
from cycler import cycler

# ---------------------------------------------------------------------------
# Tokens de color
# ---------------------------------------------------------------------------
BG_ROOT = "#06080c"      # ventana raiz / viewport
BG_DARK = "#080b10"      # fondo de vistas
BG_CARD = "#0f141d"      # paneles y tarjetas
BG_HOVER = "#18202c"     # hover
BG_ACTIVE = "#1c2636"    # activo / pulsado
BORDER = "#1f2838"       # hairline estructural
GRID = "#151a24"         # rejilla de graficas

TEXT_PRIMARY = "#e1e7f0"
TEXT_MUTED = "#5d6b82"
TEXT_BRIGHT = "#ffffff"

ACCENT_BLUE = "#00e5ff"    # cyan hielo: seleccion, velocidad, telemetria primaria
ACCENT_GREEN = "#00e676"   # OK, acelerador, delta-t < 0
ACCENT_AMBER = "#ffab00"   # potencia, avisos
ACCENT_RED = "#ff1744"     # frenada, alarmas, limite 80 kW
ACCENT_PURPLE = "#b388ff"

# Alias cortos (NO eliminar: los importan terminales y vistas heredadas)
_ACCENT = ACCENT_BLUE
_SUCCESS = ACCENT_GREEN
_WARN = ACCENT_AMBER
_DANGER = ACCENT_RED
_PURPLE = ACCENT_PURPLE
_BG = BG_DARK
_BG2 = BG_CARD
_BG3 = BG_HOVER
_BORDER = BORDER
_FG = TEXT_PRIMARY
_FG2 = TEXT_MUTED

# 14 colores de alta visibilidad sobre fondo oscuro (list[str], nunca dict)
TYRE_PALETTE: list[str] = [
    "#00e5ff", "#ff6e40", "#00e676", "#b388ff", "#ffab00", "#40c4ff", "#ff5252",
    "#69f0ae", "#ff9100", "#ea80fc", "#c6ff00", "#ff4081", "#ffd740", "#84ffff",
]

# ---------------------------------------------------------------------------
# Tipografia
# ---------------------------------------------------------------------------
FONT_UI = "Segoe UI"
FONT_MONO = "Consolas"
_UI_CANDIDATES = ("Segoe UI", "Inter", "DejaVu Sans", "Helvetica", "Arial")
_MONO_CANDIDATES = ("JetBrains Mono", "Consolas", "DejaVu Sans Mono", "Menlo", "Courier New")


def resolve_fonts(root: tk.Misc | None = None) -> tuple[str, str]:
    """Elige la primera fuente disponible de cada familia y actualiza FONT_UI / FONT_MONO."""
    global FONT_UI, FONT_MONO
    try:
        fams = {f.lower(): f for f in tkfont.families(root)}
    except tk.TclError:
        return FONT_UI, FONT_MONO
    for cand in _UI_CANDIDATES:
        if cand.lower() in fams:
            FONT_UI = fams[cand.lower()]
            break
    for cand in _MONO_CANDIDATES:
        if cand.lower() in fams:
            FONT_MONO = fams[cand.lower()]
            break
    return FONT_UI, FONT_MONO


def mono(size: int = 9, bold: bool = False) -> tuple[Any, ...]:
    """Fuente monoespacio tabular (valores numericos, marcas de tiempo)."""
    return (FONT_MONO, size, "bold") if bold else (FONT_MONO, size)


def ui(size: int = 9, bold: bool = False) -> tuple[Any, ...]:
    """Fuente de interfaz (etiquetas, titulos)."""
    return (FONT_UI, size, "bold") if bold else (FONT_UI, size)


# ---------------------------------------------------------------------------
# Matplotlib
# ---------------------------------------------------------------------------
def apply_matplotlib_theme() -> None:
    """Parchea rcParams globales para que todas las figuras adopten el estilo de cabina."""
    matplotlib.rcParams.update({
        "figure.facecolor": BG_DARK,
        "figure.edgecolor": BG_DARK,
        "savefig.facecolor": BG_DARK,
        "savefig.edgecolor": BG_DARK,
        "axes.facecolor": BG_CARD,
        "axes.edgecolor": BORDER,
        "axes.linewidth": 0.8,
        "axes.labelcolor": TEXT_PRIMARY,
        "axes.titlecolor": TEXT_BRIGHT,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "axes.grid": False,
        "axes.prop_cycle": cycler(color=TYRE_PALETTE),
        "xtick.color": TEXT_MUTED,
        "ytick.color": TEXT_MUTED,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "text.color": TEXT_PRIMARY,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": ":",
        "legend.facecolor": BG_CARD,
        "legend.edgecolor": BORDER,
        "legend.framealpha": 0.9,
        "legend.fontsize": 7,
        "lines.linewidth": 1.4,
        "lines.solid_capstyle": "butt",
        "lines.dash_capstyle": "butt",
        "patch.edgecolor": BORDER,
        "font.family": "sans-serif",
        "font.size": 8,
    })


# ---------------------------------------------------------------------------
# TTK
# ---------------------------------------------------------------------------
def apply_global_theme(root: tk.Tk) -> ttk.Style:
    """Configura 'clam' sin biseles nativos. Devuelve el ttk.Style para que el llamador lo extienda."""
    resolve_fonts(root)
    root.configure(background=BG_ROOT)
    style = ttk.Style(root)
    style.theme_use("clam")

    # Widgets tk clasicos y popdown de Combobox
    root.option_add("*Font", ui(9))
    root.option_add("*TCombobox*Listbox.background", BG_CARD)
    root.option_add("*TCombobox*Listbox.foreground", TEXT_PRIMARY)
    root.option_add("*TCombobox*Listbox.selectBackground", BG_ACTIVE)
    root.option_add("*TCombobox*Listbox.selectForeground", ACCENT_BLUE)
    root.option_add("*TCombobox*Listbox.font", mono(9))
    root.option_add("*TCombobox*Listbox.borderWidth", 0)
    root.option_add("*Menu.background", BG_CARD)
    root.option_add("*Menu.foreground", TEXT_PRIMARY)
    root.option_add("*Menu.activeBackground", BG_ACTIVE)
    root.option_add("*Menu.activeForeground", ACCENT_BLUE)

    style.configure(
        ".",
        background=BG_DARK, foreground=TEXT_PRIMARY, fieldbackground=BG_CARD,
        bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER, troughcolor=BG_DARK,
        selectbackground=ACCENT_BLUE, selectforeground=BG_ROOT, focuscolor=BG_DARK,
        insertcolor=TEXT_PRIMARY, font=ui(9), borderwidth=0, relief="flat",
    )

    style.configure("TFrame", background=BG_DARK)
    style.configure("TLabel", background=BG_DARK, foreground=TEXT_PRIMARY)
    style.configure("TSeparator", background=BORDER)

    # --- Botones ---------------------------------------------------------
    style.configure(
        "TButton", background=BG_CARD, foreground=TEXT_PRIMARY, bordercolor=BORDER,
        lightcolor=BG_CARD, darkcolor=BG_CARD, focuscolor=BG_CARD, focusthickness=0,
        relief="flat", borderwidth=1, padding=(10, 3), font=ui(9),
    )
    style.map(
        "TButton",
        background=[("disabled", BG_DARK), ("pressed", BG_ACTIVE), ("active", BG_HOVER)],
        lightcolor=[("pressed", BG_ACTIVE), ("active", BG_HOVER)],
        darkcolor=[("pressed", BG_ACTIVE), ("active", BG_HOVER)],
        bordercolor=[("disabled", BORDER), ("pressed", ACCENT_BLUE), ("active", ACCENT_BLUE)],
        foreground=[("disabled", TEXT_MUTED), ("active", TEXT_BRIGHT)],
    )

    # --- Checkbutton / Radiobutton ---------------------------------------
    for name in ("TCheckbutton", "TRadiobutton"):
        style.configure(
            name, background=BG_DARK, foreground=TEXT_PRIMARY, indicatorbackground=BG_CARD,
            indicatorforeground=BG_ROOT, indicatorcolor=BG_CARD, upperbordercolor=BORDER,
            lowerbordercolor=BORDER, bordercolor=BORDER, lightcolor=BG_CARD, darkcolor=BG_CARD,
            focuscolor=BG_DARK, padding=2,
        )
        style.map(
            name,
            background=[("active", BG_HOVER)],
            indicatorcolor=[("selected", ACCENT_BLUE), ("!selected", BG_CARD)],
            foreground=[("disabled", TEXT_MUTED)],
        )

    # --- Combobox / Entry / Spinbox --------------------------------------
    style.configure(
        "TCombobox", fieldbackground=BG_CARD, background=BG_HOVER, foreground=TEXT_PRIMARY,
        arrowcolor=TEXT_MUTED, bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER,
        selectbackground=BG_CARD, selectforeground=TEXT_PRIMARY, arrowsize=12, padding=(4, 2),
        font=mono(9),
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", BG_CARD), ("disabled", BG_DARK)],
        selectbackground=[("readonly", BG_CARD)],
        selectforeground=[("readonly", TEXT_PRIMARY)],
        background=[("active", BG_ACTIVE), ("readonly", BG_HOVER)],
        arrowcolor=[("active", ACCENT_BLUE)],
        bordercolor=[("focus", ACCENT_BLUE), ("active", ACCENT_BLUE)],
        foreground=[("disabled", TEXT_MUTED)],
    )
    for name in ("TEntry", "TSpinbox"):
        style.configure(
            name, fieldbackground=BG_CARD, foreground=TEXT_PRIMARY, insertcolor=ACCENT_BLUE,
            bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER, padding=(4, 2), font=mono(9),
            arrowcolor=TEXT_MUTED, background=BG_HOVER,
        )
        style.map(
            name,
            bordercolor=[("focus", ACCENT_BLUE)],
            lightcolor=[("focus", ACCENT_BLUE)],
            darkcolor=[("focus", ACCENT_BLUE)],
            fieldbackground=[("disabled", BG_DARK)],
        )

    # --- LabelFrame -------------------------------------------------------
    style.configure(
        "TLabelframe", background=BG_DARK, foreground=TEXT_MUTED, bordercolor=BORDER,
        lightcolor=BORDER, darkcolor=BORDER, relief="solid", borderwidth=1,
    )
    style.configure(
        "TLabelframe.Label", background=BG_DARK, foreground=ACCENT_BLUE, font=ui(8, True),
    )

    # --- Notebook ---------------------------------------------------------
    style.configure(
        "TNotebook", background=BG_DARK, bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER,
        tabmargins=(0, 0, 0, 0), borderwidth=0,
    )
    style.configure(
        "TNotebook.Tab", background=BG_CARD, foreground=TEXT_MUTED, padding=(14, 5),
        bordercolor=BORDER, lightcolor=BG_CARD, darkcolor=BG_CARD, focuscolor=BG_CARD,
        borderwidth=1, font=ui(8, True),
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", BG_DARK), ("active", BG_HOVER)],
        foreground=[("selected", ACCENT_BLUE), ("active", TEXT_PRIMARY)],
        lightcolor=[("selected", BG_DARK), ("active", BG_HOVER)],
        darkcolor=[("selected", BG_DARK), ("active", BG_HOVER)],
        bordercolor=[("selected", ACCENT_BLUE)],
    )

    # --- Scale / Progressbar ---------------------------------------------
    style.configure(
        "TScale", background=ACCENT_BLUE, troughcolor=BG_HOVER, bordercolor=BORDER,
        lightcolor=ACCENT_BLUE, darkcolor=ACCENT_BLUE, sliderthickness=12, borderwidth=0,
    )
    style.map("TScale", background=[("active", TEXT_BRIGHT)],
              lightcolor=[("active", TEXT_BRIGHT)], darkcolor=[("active", TEXT_BRIGHT)])
    for name in ("TProgressbar", "Horizontal.TProgressbar"):
        style.configure(
            name, troughcolor=BG_HOVER, background=ACCENT_GREEN, bordercolor=BORDER,
            lightcolor=ACCENT_GREEN, darkcolor=ACCENT_GREEN, thickness=6, borderwidth=0,
        )

    # --- Treeview ---------------------------------------------------------
    style.layout("Treeview", [("Treeview.treearea", {"sticky": "nswe"})])
    style.configure(
        "Treeview", background=BG_CARD, foreground=TEXT_PRIMARY, fieldbackground=BG_CARD,
        rowheight=20, bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER, borderwidth=0,
        font=mono(9),
    )
    style.configure(
        "Treeview.Heading", background=BG_HOVER, foreground=TEXT_MUTED, bordercolor=BORDER,
        lightcolor=BG_HOVER, darkcolor=BG_HOVER, relief="flat", font=ui(8, True), padding=(4, 3),
    )
    style.map(
        "Treeview",
        background=[("selected", BG_ACTIVE)],
        foreground=[("selected", ACCENT_BLUE)],
    )
    style.map("Treeview.Heading", background=[("active", BG_ACTIVE)], foreground=[("active", TEXT_PRIMARY)])

    # --- Scrollbars (sin flechas, hairline) -------------------------------
    for orient, nm in (("Vertical", "Vertical.TScrollbar"), ("Horizontal", "Horizontal.TScrollbar")):
        style.layout(nm, [(f"{orient}.Scrollbar.trough", {
            "sticky": "nswe" if orient == "Vertical" else "nswe",
            "children": [(f"{orient}.Scrollbar.thumb", {"expand": "1", "sticky": "nswe"})],
        })])
        style.configure(
            nm, background=BG_HOVER, troughcolor=BG_DARK, bordercolor=BG_DARK,
            lightcolor=BG_HOVER, darkcolor=BG_HOVER, arrowcolor=TEXT_MUTED, gripcount=0,
            width=9, relief="flat", borderwidth=0,
        )
        style.map(
            nm,
            background=[("pressed", ACCENT_BLUE), ("active", BG_ACTIVE)],
            lightcolor=[("pressed", ACCENT_BLUE), ("active", BG_ACTIVE)],
            darkcolor=[("pressed", ACCENT_BLUE), ("active", BG_ACTIVE)],
        )
    style.configure(
        "TScrollbar", background=BG_HOVER, troughcolor=BG_DARK, bordercolor=BG_DARK,
        lightcolor=BG_HOVER, darkcolor=BG_HOVER, arrowcolor=TEXT_MUTED, gripcount=0, width=9,
    )

    # --- Paned windows ----------------------------------------------------
    style.configure("TPanedwindow", background=BORDER)
    style.configure("Sash", sashthickness=4, gripcount=0, background=BORDER, bordercolor=BORDER,
                    lightcolor=BORDER, darkcolor=BORDER)

    # --- Navegacion heredada (Sidebar antiguo) ----------------------------
    style.configure(
        "Nav.TButton", background=BG_CARD, foreground=TEXT_MUTED, bordercolor=BG_CARD,
        lightcolor=BG_CARD, darkcolor=BG_CARD, focuscolor=BG_CARD, anchor="w", padding=(12, 8),
        font=ui(10), borderwidth=0,
    )
    style.map(
        "Nav.TButton",
        background=[("active", BG_HOVER), ("pressed", BG_ACTIVE)],
        lightcolor=[("active", BG_HOVER)], darkcolor=[("active", BG_HOVER)],
        foreground=[("active", TEXT_PRIMARY)],
    )
    style.configure(
        "NavActive.TButton", background=BG_ACTIVE, foreground=ACCENT_BLUE, bordercolor=ACCENT_BLUE,
        lightcolor=BG_ACTIVE, darkcolor=BG_ACTIVE, focuscolor=BG_ACTIVE, anchor="w", padding=(12, 8),
        font=ui(10, True), borderwidth=1,
    )
    style.map("NavActive.TButton", background=[("active", BG_HOVER)], foreground=[("active", ACCENT_BLUE)])

    # --- Barra de estado --------------------------------------------------
    style.configure("Status.TLabel", background=BG_DARK, foreground=TEXT_MUTED, relief="flat",
                    padding=(6, 2), font=mono(8))

    # --- Tarjetas ---------------------------------------------------------
    style.configure("Card.TFrame", background=BG_CARD, relief="flat", bordercolor=BORDER)
    style.configure("Card.TLabel", background=BG_CARD, foreground=TEXT_PRIMARY)
    style.configure("CardH.TLabel", background=BG_CARD, foreground=ACCENT_BLUE, font=ui(11, True))
    style.configure("CardS.TLabel", background=BG_CARD, foreground=TEXT_MUTED, font=ui(8))
    style.configure("Tag.TLabel", background=BG_HOVER, foreground=TEXT_MUTED, padding=(4, 2),
                    relief="flat", font=ui(8))

    # --- Extras para vistas nuevas ----------------------------------------
    style.configure("Mono.TLabel", background=BG_DARK, foreground=TEXT_PRIMARY, font=mono(9))
    style.configure("Muted.TLabel", background=BG_DARK, foreground=TEXT_MUTED, font=ui(8))
    style.configure("Section.TLabel", background=BG_DARK, foreground=ACCENT_BLUE, font=ui(8, True))
    style.configure("Ok.TLabel", background=BG_DARK, foreground=ACCENT_GREEN, font=mono(9, True))
    style.configure("Warn.TLabel", background=BG_DARK, foreground=ACCENT_AMBER, font=mono(9, True))
    style.configure("Crit.TLabel", background=BG_DARK, foreground=ACCENT_RED, font=mono(9, True))

    apply_matplotlib_theme()
    return style


__all__ = [
    "BG_ROOT", "BG_DARK", "BG_CARD", "BG_HOVER", "BG_ACTIVE", "BORDER", "GRID",
    "TEXT_PRIMARY", "TEXT_MUTED", "TEXT_BRIGHT",
    "ACCENT_BLUE", "ACCENT_GREEN", "ACCENT_AMBER", "ACCENT_RED", "ACCENT_PURPLE",
    "_ACCENT", "_SUCCESS", "_WARN", "_DANGER", "_PURPLE", "_BG", "_BG2", "_BG3", "_BORDER", "_FG", "_FG2",
    "TYRE_PALETTE", "FONT_UI", "FONT_MONO", "resolve_fonts", "mono", "ui",
    "apply_matplotlib_theme", "apply_global_theme",
]