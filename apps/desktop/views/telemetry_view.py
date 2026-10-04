#!/usr/bin/env python3
"""apps/desktop/views/telemetry_view.py
TeR-Twin Studio · Telemetry workstation (MoTeC i2 Pro / Bloomberg-terminal aesthetic).

Layout: [track map | delta-T | sector matrix]  [4 linked strips w/ 20 px header banners + scrub bar]
        [G-G + MF6.1 boundary | damper-velocity histograms | diagnostics watchdog grid]

Assumptions (explicit)
----------------------
* No matplotlib navigation toolbar. Zoom/pan are implemented on the canvases: wheel = zoom at cursor,
  left click/drag = scrub, right drag = box-zoom, middle drag = pan, double click = reset.
* Each strip is its own Figure/canvas (so a Tk header bar can sit directly above it). X-limits are linked via
  ``xlim_changed`` with a re-entrancy guard instead of ``sharex`` (shared axes across figures would force redraws
  of every canvas on each limit change).
* Tyre-temperature harness swap (logged RL=physical FR, logged FR=physical RL) is applied ONCE inside
  ``resolve_channel_name`` by every loader/decoder. This view consumes canonical ``tire_temp_{fl,fr,rl,rr}`` and
  never swaps again (re-swapping would undo the fix).
* All IO, math-channel evaluation and JAX friction-ellipse evaluation run in daemon threads; results return via a
  ``queue.SimpleQueue`` drained by the 30 ms Tk tick. The Tk thread never touches sockets/files.
* Decimation: vectorised min-max (peak preserving) for strips, LTTB for delta-T / ghost (single line, finite).
* MF6.1 boundary: ``friction_ellipses_g`` evaluated at the cursor speed quantised to 5 km/h, cached; the ellipse is
  the single-load-sensitive-mu reference (no load transfer / combined slip). Braking semi-axis mirrored.
* Watchdog thresholds (amber/red) are engineering assumptions: min cell 3.20/3.00 V, SoC 20/10 %, inverter 85/100 C,
  motor 110/125 C, accumulator > 80 kW flashes red. Damper bins (mm/s): LSC [0,50], HSC >50, LSR [-50,0), HSR <-50.
* Track reconstruction: GPS when spread > 5 m, otherwise yaw-bias-corrected, loop-closed dead reckoning per lap.
"""
from __future__ import annotations

import logging
import math
import queue
import sys
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, NamedTuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
for _p in (_ROOT / "src", _ROOT):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import matplotlib  # noqa: E402

matplotlib.use("TkAgg")
from matplotlib.artist import Artist  # noqa: E402
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Circle, Ellipse, Rectangle  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

from apps.desktop.state import AppState  # noqa: E402
from apps.desktop.views.base_view import BaseView  # noqa: E402
from ter_twin.telemetry import (  # noqa: E402
    CORNERS, POWER_LIMIT_KW, RAW_CHANNELS, CanIngest, Lap, LapAnalyzer, LogData, ReplayIngest, RingBuffer,
    compute_math_channels, decimate_indices, format_lap_time, friction_ellipses_g, get_spec, load_log,
    make_demo_log, minmax_indices,
)
from ter_twin.telemetry.math_channels import cumtrapz, damper_velocity, fill_nan, gps_to_local  # noqa: E402

__all__ = ["TelemetryView"]
LOG = logging.getLogger("telemetry_view")

# ---------------------------------------------------------------------------------------------- palette
K_BLACK, K_PLOT = "#030507", "#06080c"
K_CARD, K_CARD2 = "#0d1117", "#11141c"
K_LINE, K_LINE2, K_GRID = "#1e2430", "#1a1f2b", "#161b24"
K_TXT, K_DIM, K_BRIGHT, K_ACCENT, K_SEL = "#c9d1d9", "#6e7681", "#e6edf3", "#58a6ff", "#1f3a5f"
GREEN, RED, AMBER, CYAN = "#00e676", "#ff1744", "#ffab00", "#00e5ff"
COBALT, ORANGE, MAGENTA, WHITE = "#2979ff", "#ff6d00", "#e040fb", "#ffffff"
WHEEL_COL = {"fl": "#00b0ff", "fr": "#ff5252", "rl": "#69f0ae", "rr": "#e040fb"}
MONO_CANDIDATES = ("JetBrains Mono", "Roboto Mono", "Consolas", "Menlo", "DejaVu Sans Mono", "Courier New")

POLL_MS = 30
FS = 200.0
MAX_PTS = 2000
LIVE_CAPACITY = 120_000
DBC_PATH = _ROOT / "config" / "can" / "dbc" / "TER.dbc"
HEAT_CHANNELS = ("vx", "ay", "brake_press_front", "ax", "throttle_pct", "battery_power_kw", "yaw_rate")
CMAPS: dict[str, Any] = {
    "slate-cyan": LinearSegmentedColormap.from_list("slate_cyan", ["#1c2733", "#2b6a8a", CYAN]),
    "viridis": "viridis", "plasma": "plasma",
}
HIST_LABELS = ("LSC", "HSC", "LSR", "HSR")
HIST_COLORS = ("#1f5fbf", COBALT, "#a36a00", AMBER)
_OFF_KEYS = ("_t", "_ch", "_dist", "_an", "_laps", "_best", "_lap_sel", "_ref", "_seg", "_xs", "_ci",
             "_log_name", "_fs", "_gps")


class Ch(NamedTuple):
    key: str
    short: str
    color: str
    ls: str = "-"
    lw: float = 0.9
    fill: bool = False


class Strip(NamedTuple):
    title: str
    left: tuple[Ch, ...]
    right: tuple[Ch, ...]
    yl: str
    yr: str
    units: bool
    ghost: str | None


def _wheel(fmt: str, short: str, ls: str = "-", lw: float = 0.85) -> tuple[Ch, ...]:
    return tuple(Ch(fmt.format(c), f"{short}{c.upper()}", WHEEL_COL[c], ls, lw) for c in CORNERS)


STRIPS: tuple[Strip, ...] = (
    Strip("SPEED·ENERGY",
          (Ch("vx", "vx", CYAN, "-", 1.0), Ch("throttle_pct", "Throt", GREEN, "-", 0.85),
           Ch("brake_press_front", "BrkF", RED, "-", 0.9, True), Ch("brake_press_rear", "BrkR", "#a3283f", "--", 0.7)),
          (Ch("battery_power_kw", "P_batt", AMBER, "-", 1.0),), "km/h · % · bar", "kW", True, "vx"),
    Strip("CHASSIS",
          (Ch("ay", "ay", COBALT, "-", 0.9), Ch("ax", "ax", WHITE, "-", 0.8)),
          (Ch("steer_angle", "Steer", ORANGE, "-", 0.9), Ch("yaw_rate", "Yaw", MAGENTA, "-", 0.85)),
          "g", "deg · deg/s", True, "ay"),
    Strip("WHEELS km/h│κ %", _wheel("wheel_speed_{}", "WS"), _wheel("slip_ratio_{}", "κ", "--", 0.75),
          "km/h", "slip %", False, None),
    Strip("SUSP mm│TIRE °C", _wheel("damper_travel_{}", "DT"), _wheel("tire_temp_{}", "TT", "-.", 0.75),
          "mm", "°C", False, None),
)


def _pick_mono(root: tk.Misc) -> str:
    fams = set(tkfont.families(root))
    return next((f for f in MONO_CANDIDATES if f in fams), "TkFixedFont")


def _nanfn(fn: Any, arr: Any) -> float:
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    return float(fn(a)) if a.size else float("nan")


class _Blit:
    """Animated artists are repainted over a cached clean background (no full redraw per cursor move)."""

    def __init__(self, canvas: FigureCanvasTkAgg) -> None:
        self.canvas, self.fig = canvas, canvas.figure
        self.bg: Any = None
        self.artists: list[Artist] = []
        canvas.mpl_connect("draw_event", self._on_draw)

    def add(self, *arts: Artist) -> None:
        for a in arts:
            a.set_animated(True)
            self.artists.append(a)

    def _on_draw(self, _e: Any) -> None:
        self.bg = self.canvas.copy_from_bbox(self.fig.bbox)
        self._paint()

    def _paint(self) -> None:
        for a in self.artists:
            if a.get_visible():
                self.fig.draw_artist(a)

    def update(self) -> None:
        if self.bg is None:
            self.canvas.draw_idle()
            return
        self.canvas.restore_region(self.bg)
        self._paint()
        self.canvas.blit(self.fig.bbox)


class TelemetryView(BaseView):
    """MoTeC-style telemetry workstation (offline + live)."""

    def __init__(self, parent: tk.Widget, app_state: AppState, **kwargs: Any) -> None:
        super().__init__(parent, app_state, **kwargs)
        self._q: queue.SimpleQueue[tuple[str, Any]] = queue.SimpleQueue()
        self._poll_id: str | None = None
        self._deb_id: str | None = None
        self._active = False
        self._mode = "offline"
        self._busy = False
        self._tick_n = 0
        self._t = np.zeros(0)
        self._ch: dict[str, np.ndarray] = {}
        self._dist = np.zeros(0)
        self._an: LapAnalyzer | None = None
        self._laps: list[Lap] = []
        self._best: Lap | None = None
        self._lap_sel: Lap | None = None
        self._ref: Lap | None = None
        self._seg = (0, 0)
        self._xs = np.zeros(0)
        self._ci = 0
        self._log_name = "—"
        self._fs = FS
        self._gps: tuple[np.ndarray, np.ndarray] | None = None
        self._stash: dict[str, Any] | None = None
        self._map_x = np.zeros(0)
        self._map_y = np.zeros(0)
        self._map_decor: list[Artist] = []
        self._loading = False
        self._playing = False
        self._play_wall = 0.0
        self._play_t = 0.0
        self._buffer: RingBuffer | None = None
        self._ingest: Any = None
        self._math_cache: tuple[np.ndarray, dict[str, np.ndarray]] | None = None
        self._math_busy = False
        self._last_math = 0.0
        self._live_frozen = False
        self._live_n = 0
        self._last_ms = 0.0
        self._ell_cache: dict[int, tuple[float, float]] = {}
        self._ell_pending: set[int] = set()
        self._ghost_cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
        self._key_ids: list[tuple[str, str]] = []
        self._drag: dict[str, Any] | None = None
        self._flash = False

    # ------------------------------------------------------------------ lifecycle
    def on_mount(self) -> None:
        mono = _pick_mono(self)
        self._f8 = (mono, 8)
        self._f8b = (mono, 8, "bold")
        self._f12 = (mono, 12, "bold")
        self.v_mode = tk.StringVar(value="offline")
        self.v_xmode = tk.StringVar(value="time")
        self.v_lap = tk.StringVar()
        self.v_isref = tk.BooleanVar(value=False)
        self.v_speed = tk.StringVar(value="1.0x")
        self.v_play = tk.StringVar(value="PLAY")
        self.v_heat = tk.StringVar(value="vx")
        self.v_cmap = tk.StringVar(value="slate-cyan")
        self.v_src = tk.StringVar(value="socketcan")
        self.v_chan = tk.StringVar(value="vcan0")
        self.v_win = tk.StringVar(value="30 s")
        self.v_badge = tk.StringVar(value="NO LOG")
        self.v_best = tk.StringVar(value="")
        self.v_ref_lbl = tk.StringVar(value="REF —")
        self.v_conn = tk.StringVar(value="CONNECT")
        self.v_rate = tk.StringVar(value="OFFLINE")
        self.v_frames = tk.StringVar(value="FRM 0")
        self.v_drop = tk.StringVar(value="DROP 0")
        self.v_scrub = tk.StringVar(value="")
        self.v_s_buf = tk.StringVar(value="BUF —")
        self.v_s_hz = tk.StringVar(value="FS — Hz")
        self.v_s_cur = tk.StringVar(value="CUR —")
        self.v_s_loss = tk.StringVar(value="LOSS 0")
        self.v_s_file = tk.StringVar(value="FILE —")
        self.v_map_hdr = tk.StringVar(value="")
        self.v_dt_hdr = tk.StringVar(value="")
        self.v_gg_hdr = tk.StringVar(value="")
        self._build_toolbar()
        self._build_statusbar()
        self._build_body()
        self.bind("<Destroy>", self._on_destroy, add="+")

    def on_activate(self) -> None:
        self._do_mount()
        self._active = True
        self._bind_keys()
        self._draw_all()
        if self._poll_id is None:
            self._poll_id = self.after(POLL_MS, self._render_tick)

    def on_deactivate(self) -> None:
        self._active = False
        self._pause()
        self._unbind_keys()
        for attr in ("_poll_id", "_deb_id"):
            aid = getattr(self, attr)
            if aid is not None:
                try:
                    self.after_cancel(aid)
                except tk.TclError:
                    pass
                setattr(self, attr, None)
        super().on_deactivate()

    def on_state_change(self, key: str, value: Any) -> None:
        if self._mounted and key in ("front_tyre_params", "rear_tyre_params"):
            self._ell_cache.clear()
            self._ell_pending.clear()

    def _on_destroy(self, event: tk.Event) -> None:
        if event.widget is not self:
            return
        self._active = False
        self._disconnect_live()
        self._unbind_keys()
        for attr in ("_poll_id", "_deb_id"):
            aid = getattr(self, attr)
            if aid is not None:
                try:
                    self.after_cancel(aid)
                except tk.TclError:
                    pass
                setattr(self, attr, None)

    # ------------------------------------------------------------------ widget helpers
    def _btn(self, parent: tk.Widget, text: str | None, cmd: Any, textvariable: tk.StringVar | None = None,
             width: int | None = None) -> tk.Button:
        return tk.Button(parent, text=text, textvariable=textvariable, command=cmd, bg=K_CARD2, fg=K_TXT,
                         activebackground=K_LINE, activeforeground=K_BRIGHT, relief="flat", bd=0,
                         highlightthickness=1, highlightbackground=K_LINE, highlightcolor=K_LINE, padx=8, pady=2,
                         font=self._f8b, cursor="hand2", takefocus=0, width=width)

    def _seg_btn(self, parent: tk.Widget, text: str, value: str, var: tk.StringVar, cmd: Any) -> tk.Radiobutton:
        return tk.Radiobutton(parent, text=text, value=value, variable=var, command=cmd, indicatoron=False,
                              bg=K_CARD2, fg=K_TXT, selectcolor=K_SEL, activebackground=K_LINE,
                              activeforeground=K_BRIGHT, relief="flat", bd=0, highlightthickness=1,
                              highlightbackground=K_LINE, padx=10, pady=2, font=self._f8b, cursor="hand2",
                              takefocus=0)

    def _label(self, parent: tk.Widget, var: tk.StringVar | None = None, text: str = "", fg: str = K_TXT,
               font: Any = None, **kw: Any) -> tk.Label:
        return tk.Label(parent, textvariable=var, text=text, bg=kw.pop("bg", K_CARD), fg=fg,
                        font=font or self._f8, **kw)

    def _panel_header(self, parent: tk.Widget, title: str, var: tk.StringVar | None = None) -> tk.Frame:
        f = tk.Frame(parent, bg=K_CARD, height=20, highlightbackground=K_LINE, highlightthickness=1)
        f.pack_propagate(False)
        self._label(f, text=title, fg=K_ACCENT, font=self._f8b).pack(side="left", padx=(6, 8))
        if var is not None:
            self._label(f, var, fg=K_DIM).pack(side="left")
        return f

    @staticmethod
    def _canvas(parent: tk.Widget, fig: Figure) -> tuple[FigureCanvasTkAgg, _Blit]:
        cv = FigureCanvasTkAgg(fig, master=parent)
        w = cv.get_tk_widget()
        w.configure(bg=K_BLACK, highlightthickness=0, bd=0)
        w.pack(fill="both", expand=True)
        return cv, _Blit(cv)

    def _style_ax(self, ax: Any, grid: bool = True) -> None:
        ax.set_facecolor(K_PLOT)
        for s in ax.spines.values():
            s.set_color(K_LINE)
            s.set_linewidth(0.8)
        ax.tick_params(colors=K_DIM, labelsize=6.5, length=2, width=0.6)
        try:
            ax.tick_params(labelfontfamily=list(MONO_CANDIDATES))
        except (AttributeError, TypeError, ValueError):
            pass
        if grid:
            ax.grid(True, color=K_GRID, lw=0.5, ls=":", alpha=0.6)

    # ------------------------------------------------------------------ top command & HUD strip
    def _build_toolbar(self) -> None:
        bar = tk.Frame(self, bg=K_CARD, highlightbackground=K_LINE, highlightthickness=1)
        bar.pack(side="top", fill="x")
        r1 = tk.Frame(bar, bg=K_CARD)
        r1.pack(fill="x", padx=6, pady=(4, 2))
        r2 = tk.Frame(bar, bg=K_CARD)
        r2.pack(fill="x", padx=6, pady=(2, 4))

        seg = tk.Frame(r1, bg=K_LINE)
        seg.pack(side="left", padx=(0, 10))
        self._seg_btn(seg, "OFFLINE ANALYSIS", "offline", self.v_mode, self._on_mode).pack(side="left", padx=(0, 1))
        self._seg_btn(seg, "LIVE CAN STREAM", "live", self.v_mode, self._on_mode).pack(side="left")

        self._slot = tk.Frame(r1, bg=K_CARD)
        self._slot.pack(side="left")
        self._grp_off = tk.Frame(self._slot, bg=K_CARD)
        self._btn(self._grp_off, "OPEN LOG…", self._open_log).pack(side="left")
        self._btn(self._grp_off, "LOAD DEMO", lambda: self._load_async(None)).pack(side="left", padx=2)
        self._btn(self._grp_off, "DEMO REPLAY", self._replay_demo).pack(side="left")
        self._label(self._grp_off, self.v_badge, fg=K_ACCENT, padx=8).pack(side="left")
        self._grp_off.pack(side="left")
        self._grp_live = tk.Frame(self._slot, bg=K_CARD)
        cb = ttk.Combobox(self._grp_live, textvariable=self.v_src, state="readonly", width=10, font=self._f8,
                          values=("socketcan", "serial", "udp"))
        cb.pack(side="left")
        cb.bind("<<ComboboxSelected>>", self._on_src)
        ttk.Entry(self._grp_live, textvariable=self.v_chan, width=16, font=self._f8).pack(side="left", padx=3)
        self._btn(self._grp_live, None, self._toggle_connect, textvariable=self.v_conn, width=11).pack(side="left")
        self._btn(self._grp_live, "DEMO REPLAY", self._replay_demo).pack(side="left", padx=3)
        self._label(self._grp_live, text="WINDOW", fg=K_DIM).pack(side="left", padx=(8, 2))
        ttk.Combobox(self._grp_live, textvariable=self.v_win, state="readonly", width=6, font=self._f8,
                     values=("10 s", "30 s", "60 s", "120 s")).pack(side="left")

        self._label(r1, text="LAP", fg=K_DIM).pack(side="left", padx=(14, 2))
        self._cb_lap = ttk.Combobox(r1, textvariable=self.v_lap, state="readonly", width=46, font=self._f8)
        self._cb_lap.pack(side="left")
        self._cb_lap.bind("<<ComboboxSelected>>", self._on_lap_select)
        self._label(r1, self.v_best, fg=GREEN, font=self._f8b, padx=6).pack(side="left")
        tk.Checkbutton(r1, text="SET REFERENCE LAP", variable=self.v_isref, indicatoron=False, command=self._toggle_ref,
                       bg=K_CARD2, fg=K_TXT, selectcolor=K_SEL, activebackground=K_LINE, relief="flat", bd=0,
                       highlightthickness=1, highlightbackground=K_LINE, padx=8, pady=2, font=self._f8b,
                       cursor="hand2", takefocus=0).pack(side="left", padx=4)
        self._label(r1, self.v_ref_lbl, fg=AMBER).pack(side="left")
        xs = tk.Frame(r1, bg=K_LINE)
        xs.pack(side="right")
        self._seg_btn(xs, "LAP DISTANCE [m]", "dist", self.v_xmode, self._on_xmode).pack(side="right", padx=(1, 0))
        self._seg_btn(xs, "TIME [s]", "time", self.v_xmode, self._on_xmode).pack(side="right")

        for txt, fn in (("|<", lambda: self._jump(False)), ("<", lambda: self._step(-0.05))):
            self._btn(r2, txt, fn, width=3).pack(side="left", padx=(0, 1))
        self._btn(r2, None, self._toggle_play, textvariable=self.v_play, width=6).pack(side="left", padx=1)
        for txt, fn in ((">", lambda: self._step(0.05)), (">|", lambda: self._jump(True))):
            self._btn(r2, txt, fn, width=3).pack(side="left", padx=1)
        ttk.Combobox(r2, textvariable=self.v_speed, state="readonly", width=6, font=self._f8,
                     values=("0.25x", "0.5x", "1.0x", "2.0x", "5.0x", "10.0x")).pack(side="left", padx=6)
        self._label(r2, text="TRACK COLOUR", fg=K_DIM).pack(side="left", padx=(14, 2))
        c1 = ttk.Combobox(r2, textvariable=self.v_heat, state="readonly", width=16, font=self._f8,
                          values=HEAT_CHANNELS)
        c1.pack(side="left")
        c1.bind("<<ComboboxSelected>>", lambda _e: self._on_heat())
        c2 = ttk.Combobox(r2, textvariable=self.v_cmap, state="readonly", width=10, font=self._f8,
                          values=tuple(CMAPS))
        c2.pack(side="left", padx=3)
        c2.bind("<<ComboboxSelected>>", lambda _e: self._on_heat())

        hud = tk.Frame(r2, bg=K_CARD2, highlightbackground=K_LINE, highlightthickness=1)
        hud.pack(side="right")
        self._lbl_rate = self._label(hud, self.v_rate, fg=K_DIM, font=self._f8b, bg=K_CARD2, width=10, padx=6)
        self._lbl_rate.pack(side="left")
        self._label(hud, self.v_frames, fg=K_TXT, bg=K_CARD2, width=14).pack(side="left")
        self._label(hud, self.v_drop, fg=K_TXT, bg=K_CARD2, width=10).pack(side="left")
        self._fill_cv = tk.Canvas(hud, width=80, height=8, bg=K_BLACK, highlightthickness=1,
                                  highlightbackground=K_LINE)
        self._fill_cv.pack(side="left", padx=8, pady=3)

    def _build_statusbar(self) -> None:
        bar = tk.Frame(self, bg=K_CARD, height=20, highlightbackground=K_LINE, highlightthickness=1)
        bar.pack(side="bottom", fill="x")
        for var, col in ((self.v_s_buf, K_TXT), (self.v_s_hz, GREEN), (self.v_s_cur, K_BRIGHT),
                         (self.v_s_loss, K_TXT), (self.v_s_file, K_ACCENT)):
            self._label(bar, var, fg=col, padx=8).pack(side="left")
            tk.Frame(bar, bg=K_LINE, width=1).pack(side="left", fill="y", pady=3)

    # ------------------------------------------------------------------ body
    def _build_body(self) -> None:
        self._pane = tk.PanedWindow(self, orient=tk.HORIZONTAL, bg=K_LINE, sashwidth=3, bd=0, sashrelief="flat",
                                    opaqueresize=True)
        self._pane.pack(fill="both", expand=True)
        left = tk.PanedWindow(self._pane, orient=tk.VERTICAL, bg=K_LINE, sashwidth=3, bd=0)
        center = tk.Frame(self._pane, bg=K_BLACK)
        right = tk.PanedWindow(self._pane, orient=tk.VERTICAL, bg=K_LINE, sashwidth=3, bd=0)
        self._pane.add(left, minsize=240, stretch="always")
        self._pane.add(center, minsize=520, stretch="always")
        self._pane.add(right, minsize=260, stretch="always")
        self._sash_done = False
        self._pane.bind("<Configure>", self._init_sashes)

        f_map, f_dt, f_sec = (tk.Frame(left, bg=K_BLACK) for _ in range(3))
        left.add(f_map, minsize=200, stretch="always")
        left.add(f_dt, minsize=120, stretch="always")
        left.add(f_sec, minsize=110, stretch="never")
        self._build_map(f_map)
        self._build_delta(f_dt)
        self._build_sectors(f_sec)
        self._build_strips(center)
        f_gg, f_h, f_d = (tk.Frame(right, bg=K_BLACK) for _ in range(3))
        right.add(f_gg, minsize=220, stretch="always")
        right.add(f_h, minsize=200, stretch="always")
        right.add(f_d, minsize=150, stretch="never")
        self._build_gg(f_gg)
        self._build_hist(f_h)
        self._build_diag(f_d)

    def _init_sashes(self, ev: tk.Event) -> None:
        if self._sash_done or ev.width < 400:
            return
        self._sash_done = True
        try:
            self._pane.sash_place(0, int(ev.width * 0.22), 0)
            self._pane.sash_place(1, int(ev.width * 0.76), 0)
        except tk.TclError:
            pass

    # ---- map
    def _build_map(self, parent: tk.Widget) -> None:
        self._panel_header(parent, "TRACK MAP", self.v_map_hdr).pack(fill="x")
        fig = Figure(figsize=(3, 3), facecolor=K_BLACK)
        ax = fig.add_axes([0.01, 0.01, 0.98, 0.98])
        ax.set_facecolor(K_BLACK)
        ax.set_axis_off()
        ax.set_aspect("equal", adjustable="datalim")
        self._ax_map = ax
        self._map_lc = LineCollection([], linewidths=1.2, capstyle="butt", zorder=2)
        ax.add_collection(self._map_lc)
        self._blip_ring, = ax.plot([], [], "o", ms=4.5, mfc="none", mec=WHITE, mew=0.9, zorder=10)
        self._blip_dot, = ax.plot([], [], "o", ms=1.6, color=RED, zorder=11)
        self._blip_vec, = ax.plot([], [], color=WHITE, lw=0.9, solid_capstyle="butt", zorder=10)
        self._cv_map, self._bl_map = self._canvas(parent, fig)
        self._bl_map.add(self._blip_ring, self._blip_dot, self._blip_vec)

    # ---- delta
    def _build_delta(self, parent: tk.Widget) -> None:
        self._panel_header(parent, "Δt vs REF", self.v_dt_hdr).pack(fill="x")
        fig = Figure(figsize=(3, 2), facecolor=K_BLACK)
        ax = fig.add_axes([0.17, 0.2, 0.79, 0.74])
        self._style_ax(ax)
        ax.axhline(0, color=K_LINE, lw=0.8)
        ax.set_xlabel("Lap distance [m]", color=K_DIM, fontsize=6.5)
        ax.set_ylabel("Δt [s]", color=K_DIM, fontsize=6.5)
        self._ax_dt = ax
        self._dt_line, = ax.plot([], [], color=K_BRIGHT, lw=0.8, solid_capstyle="butt")
        self._dt_cur = ax.axvline(0, color=WHITE, lw=0.7, alpha=0.85)
        self._cv_dt, self._bl_dt = self._canvas(parent, fig)
        self._bl_dt.add(self._dt_cur)

    # ---- sector matrix
    def _build_sectors(self, parent: tk.Widget) -> None:
        self._panel_header(parent, "SECTOR DELTA MATRIX").pack(fill="x")
        grid = tk.Frame(parent, bg=K_CARD, highlightbackground=K_LINE2, highlightthickness=1)
        grid.pack(fill="both", expand=True)
        for c, h in enumerate(("", "CUR", "REF", "Δ ms")):
            self._label(grid, text=h, fg=K_DIM, font=self._f8b, width=9 if c else 5, anchor="e").grid(
                row=0, column=c, padx=2, sticky="e")
        self._sec_cells: list[tuple[tk.Label, tk.Label, tk.Label]] = []
        for r, name in enumerate(("S1", "S2", "S3", "LAP"), 1):
            self._label(grid, text=name, fg=K_ACCENT, font=self._f8b, width=5, anchor="w").grid(
                row=r, column=0, padx=2, sticky="w")
            cells = tuple(self._label(grid, text="--", fg=K_TXT, width=9, anchor="e") for _ in range(3))
            for c, w in enumerate(cells, 1):
                w.grid(row=r, column=c, padx=2, sticky="e")
            self._sec_cells.append(cells)  # type: ignore[arg-type]

    # ---- strips
    def _build_strips(self, parent: tk.Widget) -> None:
        parent.columnconfigure(0, weight=1)
        self._figs: list[Figure] = []
        self._cvs: list[FigureCanvasTkAgg] = []
        self._bls: list[_Blit] = []
        self._axl: list[Any] = []
        self._axr: list[Any] = []
        self._lines: list[list[tuple[Ch, Any]]] = []
        self._ghosts: list[Any] = []
        self._cursors: list[Any] = []
        self._zrects: list[Rectangle] = []
        self._hdr: list[list[list[Any]]] = []
        self._fills: dict[tuple[int, str], Any] = {}
        self._cv_idx: dict[Any, int] = {}
        n = len(STRIPS)
        for si, strip in enumerate(STRIPS):
            last = si == n - 1
            self._build_strip_header(parent, 2 * si, strip)
            holder = tk.Frame(parent, bg=K_BLACK)
            holder.grid(row=2 * si + 1, column=0, sticky="nsew")
            parent.rowconfigure(2 * si + 1, weight=118 if last else 100)
            fig = Figure(figsize=(8, 2), facecolor=K_BLACK)
            b = 0.22 if last else 0.03
            ax = fig.add_axes([0.07, b, 0.86, 0.96 - b])
            axr = ax.twinx()
            self._style_ax(ax)
            self._style_ax(axr, grid=False)
            ax.margins(y=0.06)
            axr.margins(y=0.06)
            if not last:
                ax.tick_params(labelbottom=False)
            ax.set_ylabel(strip.yl, color=K_DIM, fontsize=6.5)
            axr.set_ylabel(strip.yr, color=K_DIM, fontsize=6.5)
            lines: list[tuple[Ch, Any]] = []
            for chs, tgt in ((strip.left, ax), (strip.right, axr)):
                for ch in chs:
                    ln, = tgt.plot([], [], color=ch.color, lw=ch.lw, ls=ch.ls, solid_capstyle="butt",
                                   dash_capstyle="butt", solid_joinstyle="miter", zorder=3)
                    lines.append((ch, ln))
            if strip.ghost:
                gh, = ax.plot([], [], color=K_DIM, lw=0.7, ls="--", dash_capstyle="butt", zorder=2)
                self._ghosts.append(gh)
            else:
                self._ghosts.append(None)
            cur = ax.axvline(0, color=WHITE, lw=0.7, alpha=0.85, zorder=6)
            rect = Rectangle((0, 0), 0, 1, transform=blended_transform_factory(ax.transData, ax.transAxes),
                             fc=COBALT, alpha=0.18, ec=COBALT, lw=0.7, visible=False, zorder=7)
            ax.add_artist(rect)
            cv, bl = self._canvas(holder, fig)
            bl.add(cur, rect)
            self._cv_idx[cv] = si
            for ev, fn in (("motion_notify_event", self._on_motion), ("button_press_event", self._on_press),
                           ("button_release_event", self._on_release), ("scroll_event", self._on_scroll)):
                cv.mpl_connect(ev, fn)
            ax.callbacks.connect("xlim_changed", self._on_xlim)
            self._figs.append(fig)
            self._cvs.append(cv)
            self._bls.append(bl)
            self._axl.append(ax)
            self._axr.append(axr)
            self._lines.append(lines)
            self._cursors.append(cur)
            self._zrects.append(rect)
        self._axl[-1].set_xlabel("Time [s]", color=K_DIM, fontsize=6.5)
        axr0 = self._axr[0]
        self._lim_line = axr0.axhline(POWER_LIMIT_KW, color="#8a2a38", ls="--", lw=0.9, dash_capstyle="butt", zorder=4)
        self._ln_over, = axr0.plot([], [], color=RED, lw=1.4, solid_capstyle="butt", zorder=5)
        self._bls[0].add(self._lim_line)
        self._build_scrub(parent, 2 * n)

    def _build_strip_header(self, parent: tk.Widget, row: int, strip: Strip) -> None:
        hdr = tk.Frame(parent, bg=K_CARD, height=20, highlightbackground=K_LINE, highlightthickness=1)
        hdr.grid(row=row, column=0, sticky="ew")
        hdr.pack_propagate(False)
        self._label(hdr, text=strip.title, fg=K_ACCENT, font=self._f8b).pack(side="left", padx=(6, 10))
        cells: list[list[Any]] = []
        for ch in strip.left + strip.right:
            tk.Label(hdr, text="■", bg=K_CARD, fg=ch.color, font=self._f8).pack(side="left", padx=(0, 1))
            tmpl = self._hdr_text(ch, strip, float("nan"))
            lb = self._label(hdr, text=tmpl, fg=K_TXT, width=len(tmpl), anchor="w")
            lb.pack(side="left", padx=(0, 8))
            cells.append([ch, lb, tmpl, strip])
        self._hdr.append(cells)

    @staticmethod
    def _hdr_text(ch: Ch, strip: Strip, v: float) -> str:
        sp = get_spec(ch.key)
        wv = 7 if strip.units else 6
        val = "--" if v is None or not np.isfinite(v) else f"{v:.{sp.precision}f}"
        return f"{ch.short} {val:>{wv}}" + (f" {sp.units}" if strip.units else "")

    def _build_scrub(self, parent: tk.Widget, row: int) -> None:
        f = tk.Frame(parent, bg=K_CARD, height=22, highlightbackground=K_LINE, highlightthickness=1)
        f.grid(row=row, column=0, sticky="ew")
        f.pack_propagate(False)
        self._label(f, self.v_scrub, fg=K_BRIGHT, width=48, anchor="e").pack(side="right", padx=6)
        cv = tk.Canvas(f, bg=K_CARD, height=18, highlightthickness=0, bd=0, cursor="sb_h_double_arrow")
        cv.pack(side="left", fill="x", expand=True, padx=6)
        cv.bind("<Configure>", lambda _e: self._draw_scrub())
        cv.bind("<Button-1>", self._scrub_event)
        cv.bind("<B1-Motion>", self._scrub_event)
        self._scrub_cv = cv

    # ---- G-G
    def _build_gg(self, parent: tk.Widget) -> None:
        self._panel_header(parent, "G-G TRACTION CIRCLE", self.v_gg_hdr).pack(fill="x")
        fig = Figure(figsize=(3, 3), facecolor=K_BLACK)
        ax = fig.add_axes([0.13, 0.11, 0.83, 0.86])
        self._style_ax(ax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-2.4, 2.4)
        ax.set_ylim(-2.4, 2.4)
        ax.axhline(0, color=K_LINE, lw=0.6)
        ax.axvline(0, color=K_LINE, lw=0.6)
        for r in (1.0, 1.5, 2.0):
            ax.add_patch(Circle((0, 0), r, fill=False, ec=K_LINE, lw=0.6, zorder=1))
        ax.set_xlabel("ay [g] (+ left)", color=K_DIM, fontsize=6.5)
        ax.set_ylabel("ax [g]", color=K_DIM, fontsize=6.5)
        self._ax_gg = ax
        self._gg_sc = ax.scatter([], [], s=1.2, c="#8b97a8", alpha=0.3, linewidths=0, rasterized=True, zorder=2)
        self._gg_ell = Ellipse((0, 0), 0.0, 0.0, fill=False, ec=AMBER, ls="--", lw=0.8, zorder=3, visible=False)
        ax.add_patch(self._gg_ell)
        self._gg_trail, = ax.plot([], [], color=K_TXT, lw=0.6, alpha=0.5, solid_capstyle="butt", zorder=4)
        self._gg_vec, = ax.plot([], [], color=RED, lw=0.9, solid_capstyle="butt", zorder=5)
        self._gg_pt, = ax.plot([], [], "o", ms=3.5, color=RED, zorder=6)
        self._cv_gg, self._bl_gg = self._canvas(parent, fig)
        self._bl_gg.add(self._gg_ell, self._gg_trail, self._gg_vec, self._gg_pt)

    # ---- histograms
    def _build_hist(self, parent: tk.Widget) -> None:
        self._panel_header(parent, "DAMPER VELOCITY [%]").pack(fill="x")
        rows = tk.Frame(parent, bg=K_CARD, highlightbackground=K_LINE2, highlightthickness=1)
        rows.pack(side="bottom", fill="x")
        self._hist_lbls: dict[str, tk.Label] = {}
        for c in CORNERS:
            lb = self._label(rows, text="", fg=WHEEL_COL[c], anchor="w")
            lb.pack(fill="x", padx=6)
            self._hist_lbls[c] = lb
        fig = Figure(figsize=(3, 3), facecolor=K_BLACK)
        axs = fig.subplots(2, 2).ravel()
        fig.subplots_adjust(left=0.1, right=0.97, top=0.92, bottom=0.08, hspace=0.5, wspace=0.28)
        self._hist_bars: dict[str, Any] = {}
        self._hist_ax: dict[str, Any] = {}
        for ax, c in zip(axs, CORNERS):
            self._style_ax(ax)
            ax.bar(range(4), [0, 0, 0, 0], color=HIST_COLORS, width=0.7, ec=K_LINE, lw=0.6)
            self._hist_bars[c] = ax.containers[0]
            ax.set_xticks(range(4))
            ax.set_xticklabels(HIST_LABELS, fontsize=6)
            ax.set_ylim(0, 100)
            ax.set_title(c.upper(), fontsize=7, color=WHEEL_COL[c], pad=2)
            self._hist_ax[c] = ax
        self._cv_h, _ = self._canvas(parent, fig)

    # ---- watchdog
    def _build_diag(self, parent: tk.Widget) -> None:
        self._panel_header(parent, "DIAGNOSTICS WATCHDOG").pack(fill="x")
        grid = tk.Frame(parent, bg=K_BLACK)
        grid.pack(fill="both", expand=True)

        def lo_hi(v: float, amber: float, red: float, low: bool) -> str:
            bad = (lambda x, t: x < t) if low else (lambda x, t: x > t)
            return RED if bad(v, red) else AMBER if bad(v, amber) else GREEN

        self._diag: list[dict[str, Any]] = [
            dict(name="MIN CELL V", unit="V", fmt="{:.3f}", key="min_cell_voltage",
                 col=lambda v: GREEN if v > 3.2 else RED if v <= 3.0 else AMBER),
            dict(name="ACCU POWER", unit="kW", fmt="{:.1f}", key="battery_power_kw", power=True,
                 col=lambda v: AMBER),
            dict(name="MAX INV T", unit="°C", fmt="{:.1f}", pre="inverter_temp_",
                 col=lambda v: lo_hi(v, 85.0, 100.0, False)),
            dict(name="MAX MOT T", unit="°C", fmt="{:.1f}", pre="motor_temp_",
                 col=lambda v: lo_hi(v, 110.0, 125.0, False)),
            dict(name="STATE OF CHG", unit="%", fmt="{:.1f}", key="soc", col=lambda v: lo_hi(v, 20.0, 10.0, True)),
            dict(name="TV MOMENT", unit="N·m", fmt="{:+.0f}", key="tv_yaw_moment", col=lambda v: K_BRIGHT),
        ]
        for k, d in enumerate(self._diag):
            cell = tk.Frame(grid, bg=K_CARD, highlightbackground=K_LINE2, highlightthickness=1)
            cell.grid(row=k // 2, column=k % 2, sticky="nsew")
            grid.columnconfigure(k % 2, weight=1)
            grid.rowconfigure(k // 2, weight=1)
            self._label(cell, text=d["name"], fg=K_DIM, font=self._f8b).pack(anchor="w", padx=6, pady=(3, 0))
            d["val"] = self._label(cell, text="--", fg=K_BRIGHT, font=self._f12)
            d["val"].pack(anchor="w", padx=6)
            d["sub"] = self._label(cell, text="", fg=K_DIM)
            d["sub"].pack(anchor="w", padx=6, pady=(0, 3))

    # ================================================================== data helpers
    def _val(self, key: str, i: int) -> float:
        y = self._ch.get(key)
        if y is None or i < 0 or i >= y.size:
            return float("nan")
        return float(y[i])

    def _win_s(self) -> float:
        try:
            return float(self.v_win.get().split()[0])
        except (ValueError, IndexError):
            return 30.0

    def _speed(self) -> float:
        try:
            return float(self.v_speed.get().rstrip("x"))
        except ValueError:
            return 1.0

    def _x_for(self, lap_i0: int, lap_i1: int) -> np.ndarray:
        if self.v_xmode.get() == "dist" and self._dist.size == self._t.size:
            return self._dist[lap_i0:lap_i1] - self._dist[lap_i0]
        return self._t[lap_i0:lap_i1] - self._t[lap_i0]

    def _rebuild_x(self) -> None:
        n = self._t.size
        i0, i1 = self._seg
        if n == 0 or i1 - i0 < 2:
            self._xs = np.zeros(0)
            return
        self._xs = np.asarray(self._t[i0:i1] if self._mode == "live" else self._x_for(i0, i1), dtype=float)
        lbl = "Time [s]" if (self._mode == "live" or self.v_xmode.get() == "time") else "Lap distance [m]"
        self._axl[-1].set_xlabel(lbl, color=K_DIM, fontsize=6.5)
        self._ghost_cache.clear()

    def _gps_xy(self, ch: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray] | None:
        if "gps_lat" not in ch or "gps_lon" not in ch:
            return None
        g = gps_to_local(ch["gps_lat"], ch["gps_lon"])
        if g is None or (np.ptp(g[0]) + np.ptp(g[1])) < 5.0:
            return None
        return g

    def _track_for(self, i0: int, i1: int) -> tuple[np.ndarray, np.ndarray, str]:
        ch = self._ch
        empty = np.zeros(0)
        if i1 - i0 < 3:
            return empty, empty, "-"
        if self._mode == "live":
            tx, ty = ch.get("track_x"), ch.get("track_y")
            if tx is None or ty is None:
                return empty, empty, "-"
            return np.asarray(tx[i0:i1]), np.asarray(ty[i0:i1]), "live"
        if self._gps is not None:
            return self._gps[0][i0:i1], self._gps[1][i0:i1], "GPS"
        if all(k in ch for k in ("vx", "yaw_rate", "vy_est")):
            t = self._t[i0:i1]
            vx = np.maximum(fill_nan(ch["vx"][i0:i1]) / 3.6, 0.0)
            vy = fill_nan(ch["vy_est"][i0:i1])
            r = np.radians(fill_nan(ch["yaw_rate"][i0:i1]))
            closed = self._lap_sel is not None
            psi = cumtrapz(r, t)
            if closed:  # remove yaw-rate bias so the heading integral closes to n*2pi
                total = float(psi[-1])
                target = 2.0 * math.pi * (round(total / (2.0 * math.pi)) or (1 if total >= 0 else -1))
                r = r - (total - target) / max(float(t[-1] - t[0]), 1e-6)
                psi = cumtrapz(r, t)
            x = cumtrapz(vx * np.cos(psi) - vy * np.sin(psi), t)
            y = cumtrapz(vx * np.sin(psi) + vy * np.cos(psi), t)
            if closed:
                s = np.linspace(0.0, 1.0, x.size)
                x = x - (x[-1] - x[0]) * s
                y = y - (y[-1] - y[0]) * s
            return x, y, "odometry (loop-closed)" if closed else "odometry (raw)"
        tx, ty = ch.get("track_x"), ch.get("track_y")
        if tx is None or ty is None:
            return empty, empty, "-"
        return np.asarray(tx[i0:i1]), np.asarray(ty[i0:i1]), "math"

    @staticmethod
    def _dec(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if x.size <= MAX_PTS:
            return x, y
        sel = minmax_indices(y, MAX_PTS)
        return x[sel], y[sel]

    # ================================================================== background workers
    def _run_bg(self, fn: Any, *args: Any) -> None:
        threading.Thread(target=fn, args=args, daemon=True).start()

    def _open_log(self) -> None:
        p = filedialog.askopenfilename(parent=self, title="Open telemetry log", filetypes=[
            ("Telemetry logs", "*.log *.asc *.mf4 *.mdf *.csv *.npz *.mat"), ("All files", "*.*")])
        if p:
            self._load_async(Path(p))

    def _load_async(self, path: Path | None) -> None:
        if self._loading:
            return
        if self._mode != "offline":
            self.v_mode.set("offline")
            self._on_mode()
        self._disconnect_live()
        self._loading = True
        self.v_badge.set("LOADING…")
        self._run_bg(self._load_worker, path)

    def _load_worker(self, path: Path | None) -> None:
        try:
            log: LogData = make_demo_log(n_laps=6) if path is None else load_log(path)
            front = self._app_state.get("front_tyre_params")
            rear = self._app_state.get("rear_tyre_params")
            ch = {**log.channels, **compute_math_channels(log.t, log.channels, None, front, rear)}
            an: LapAnalyzer | None = None
            laps: list[Lap] = []
            if "vx" in ch:
                an = LapAnalyzer(log.t, ch["vx"])
                if "lap_beacon" in ch:
                    laps = an.detect_from_beacon(ch["lap_beacon"])
                if not laps and "track_x" in ch and "track_y" in ch:
                    laps = an.detect_from_gate(ch["track_x"], ch["track_y"])
            self._q.put(("loaded", dict(t=log.t, ch=ch, an=an, laps=laps, fs=log.fs,
                                        dist=an.dist if an is not None else np.zeros(log.t.size),
                                        name="DEMO (synthetic)" if path is None else path.name,
                                        gps=self._gps_xy(ch))))
        except Exception as exc:  # noqa: BLE001
            LOG.exception("load failed")
            self._q.put(("load_error", f"{type(exc).__name__}: {exc}"))

    def _ellipse_worker(self, bucket: int, front: Any, rear: Any) -> None:
        try:
            res = friction_ellipses_g(front, rear, None, (bucket / 3.6,))
        except Exception:  # noqa: BLE001
            LOG.exception("friction ellipse failed")
            res = []
        self._q.put(("ellipse", (bucket, res)))

    def _request_ellipse(self, bucket: int) -> None:
        if bucket in self._ell_cache or bucket in self._ell_pending:
            return
        self._ell_pending.add(bucket)
        self._run_bg(self._ellipse_worker, bucket, self._app_state.get("front_tyre_params"),
                     self._app_state.get("rear_tyre_params"))

    def _math_worker(self, t: np.ndarray, ch: dict[str, np.ndarray], front: Any, rear: Any) -> None:
        try:
            self._q.put(("math", (t, compute_math_channels(t, ch, None, front, rear))))
        except Exception:  # noqa: BLE001
            LOG.exception("live math channels failed")
            self._q.put(("math", None))

    def _demo_worker(self) -> None:
        try:
            self._q.put(("replay_log", make_demo_log(n_laps=6)))
        except Exception as exc:  # noqa: BLE001
            self._q.put(("load_error", f"{type(exc).__name__}: {exc}"))

    def _drain_queue(self) -> None:
        for _ in range(64):
            try:
                tag, payload = self._q.get_nowait()
            except queue.Empty:
                return
            if tag == "loaded":
                self._loading = False
                if self._mode == "offline":
                    self._apply_loaded(payload)
            elif tag == "load_error":
                self._loading = False
                self.v_badge.set("LOAD FAILED")
                messagebox.showerror("Telemetry", str(payload), parent=self)
            elif tag == "ellipse":
                bucket, res = payload
                self._ell_pending.discard(bucket)
                if res:
                    _, ay_g, ax_g = res[0]
                    if np.isfinite(ay_g) and np.isfinite(ax_g):
                        self._ell_cache[bucket] = (float(ay_g), float(ax_g))
            elif tag == "math":
                self._math_busy = False
                if payload is not None:
                    self._math_cache = payload
            elif tag == "replay_log":
                self._start_replay(payload)

    # ================================================================== apply data
    def _apply_loaded(self, p: dict[str, Any]) -> None:
        self._pause()
        self._t, self._ch, self._an, self._laps = p["t"], p["ch"], p["an"], p["laps"]
        self._dist, self._fs, self._log_name, self._gps = p["dist"], p["fs"], p["name"], p["gps"]
        valid = [la for la in self._laps if la.valid]
        self._best = min(valid, key=lambda la: la.lap_time) if valid else None
        self._ref = self._best
        self.v_isref.set(self._ref is not None)
        self._populate_laps()
        start = self._best
        self.v_lap.set(self._cb_lap["values"][(self._laps.index(start) + 1) if start else 0])
        dur = float(self._t[-1] - self._t[0]) if self._t.size > 1 else 0.0
        self.v_badge.set(f"{self._log_name} · {dur:.1f} s · {self._t.size} smp · {self._fs:.1f} Hz")
        self.v_s_file.set(f"FILE {self._log_name} · {self._t.size} smp")
        self._set_segment(start)

    def _populate_laps(self) -> None:
        vals = ["Session (all)"]
        for la in self._laps:
            tag = ("  [BEST]" if la is self._best else "") + ("" if la.valid else "  (invalid)")
            vals.append(f"Lap {la.index}: {format_lap_time(la.lap_time)} (Vmax: {la.v_max_kmh:.0f} km/h){tag}")
        self._cb_lap["values"] = vals
        self.v_lap.set(vals[0])
        self.v_ref_lbl.set(f"REF L{self._ref.index}" if self._ref else "REF —")
        self.v_best.set(f"■ BEST L{self._best.index} {format_lap_time(self._best.lap_time)}" if self._best else "")

    # ================================================================== segment / panels
    def _set_segment(self, lap: Lap | None, reset_view: bool = True) -> None:
        n = self._t.size
        self._lap_sel = lap
        self._seg = (lap.i0, lap.i1) if lap is not None else (0, n)
        self._rebuild_x()
        self._ci = self._seg[0]
        self._play_t = float(self._t[self._ci]) if n else 0.0
        if reset_view:
            self._reset_view()
        self._refresh_strips()
        self._update_map()
        self._update_delta()
        self._update_sectors()
        self._update_gg()
        self._update_hist()
        self._update_diag_extremes()
        self._update_cursor(blit=False)
        self._draw_all()

    def _reset_view(self) -> None:
        if self._xs.size < 2:
            return
        self._busy = True
        try:
            self._axl[0].set_xlim(float(self._xs[0]), float(self._xs[-1]))
            for a in self._axl + self._axr:
                a.set_autoscaley_on(True)
        finally:
            self._busy = False
        for a in self._axl[1:]:
            self._busy = True
            try:
                a.set_xlim(float(self._xs[0]), float(self._xs[-1]))
            finally:
                self._busy = False

    def _draw_all(self) -> None:
        for cv in (*self._cvs, self._cv_map, self._cv_dt, self._cv_gg, self._cv_h):
            cv.draw_idle()
        self._draw_scrub()

    def _blit_all(self) -> None:
        for bl in (*self._bls, self._bl_map, self._bl_dt, self._bl_gg):
            bl.update()

    def _set_fill(self, si: int, ch: Ch, x: np.ndarray, y: np.ndarray) -> None:
        old = self._fills.pop((si, ch.key), None)
        if old is not None:
            try:
                old.remove()
            except (ValueError, NotImplementedError):
                pass
        if x.size > 1:
            self._fills[(si, ch.key)] = self._axl[si].fill_between(x, 0.0, y, color=ch.color, alpha=0.15, lw=0,
                                                                   zorder=1)

    def _ghost_xy(self, key: str) -> tuple[np.ndarray, np.ndarray] | None:
        ref, lap = self._ref, self._lap_sel
        y = self._ch.get(key)
        if ref is None or lap is None or ref is lap or y is None or y.size != self._t.size or self._mode != "offline":
            return None
        ck = (key, ref.index, self.v_xmode.get())
        if ck not in self._ghost_cache:
            gx = self._x_for(ref.i0, ref.i1)
            gy = y[ref.i0:ref.i1]
            sel = decimate_indices(gx, gy, MAX_PTS, "lttb")
            self._ghost_cache[ck] = (gx[sel], gy[sel])
        return self._ghost_cache[ck]

    def _refresh_strips(self) -> None:
        xs = self._xs
        i0, i1 = self._seg
        nt = self._t.size
        if xs.size < 2:
            for lines in self._lines:
                for _, ln in lines:
                    ln.set_data([], [])
            self._ln_over.set_data([], [])
            return
        self._busy = True
        try:
            lo, hi = self._axl[0].get_xlim()
            a = max(0, int(np.searchsorted(xs, lo)) - 1)
            b = min(xs.size, int(np.searchsorted(xs, hi, side="right")) + 1)
            if b - a < 2:
                a, b = 0, xs.size
            xv = xs[a:b]
            for si, strip in enumerate(STRIPS):
                for ch, ln in self._lines[si]:
                    y = self._ch.get(ch.key)
                    if y is None or y.size != nt:
                        ln.set_data([], [])
                        if ch.fill:
                            self._set_fill(si, ch, np.zeros(0), np.zeros(0))
                        continue
                    x2, y2 = self._dec(xv, y[i0 + a:i0 + b])
                    ln.set_data(x2, y2)
                    if ch.fill:
                        self._set_fill(si, ch, x2, y2)
                    if ch.key == "battery_power_kw":
                        self._ln_over.set_data(x2, np.where(y2 > POWER_LIMIT_KW, y2, np.nan))
                gh = self._ghosts[si]
                if gh is not None and strip.ghost:
                    g = self._ghost_xy(strip.ghost)
                    if g is None:
                        gh.set_data([], [])
                    else:
                        m = (g[0] >= lo) & (g[0] <= hi)
                        gh.set_data(g[0][m], g[1][m])
                for ax in (self._axl[si], self._axr[si]):
                    ax.relim()
                    ax.autoscale_view(scalex=False, scaley=True)
            axr0 = self._axr[0]
            y0, y1 = axr0.get_ylim()
            if y1 < POWER_LIMIT_KW * 1.1:
                axr0.set_ylim(y0, POWER_LIMIT_KW * 1.1)
        finally:
            self._busy = False

    def _update_map(self) -> None:
        for art in self._map_decor:
            try:
                art.remove()
            except (ValueError, NotImplementedError):
                pass
        self._map_decor.clear()
        i0, i1 = self._seg
        x, y, src = self._track_for(i0, i1)
        self._map_x, self._map_y = x, y
        if not (x.size >= 3 and np.isfinite(x).any() and np.isfinite(y).any()):
            self._map_lc.set_segments([])
            self.v_map_hdr.set("NO POSITION DATA (GPS / vx+yaw_rate)")
            for ln in (self._blip_ring, self._blip_dot, self._blip_vec):
                ln.set_data([], [])
            return
        n = x.size
        idx = np.linspace(0, n - 1, min(n, 4000)).astype(int)
        px, py = x[idx], y[idx]
        pts = np.column_stack([px, py]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        key = self.v_heat.get()
        v = self._ch.get(key)
        lo, hi = 0.0, 1.0
        vals = np.zeros(len(idx) - 1)
        if v is not None and v.size == self._t.size:
            vv = v[i0:i1][idx]
            vals = 0.5 * (vv[:-1] + vv[1:])
            fin = vals[np.isfinite(vals)]
            if fin.size:
                lo, hi = (float(q) for q in np.percentile(fin, [2, 98]))
                if hi - lo < 1e-9:
                    hi = lo + 1.0
            vals = np.nan_to_num(vals, nan=lo)
        self._map_lc.set_segments(segs)
        self._map_lc.set_array(vals)
        self._map_lc.set_cmap(CMAPS[self.v_cmap.get()])
        self._map_lc.set_clim(lo, hi)
        sp = get_spec(key)
        self.v_map_hdr.set(f"{sp.display_name} {lo:.1f}→{hi:.1f} {sp.units} · {src}")
        xmn, xmx, ymn, ymx = float(np.nanmin(px)), float(np.nanmax(px)), float(np.nanmin(py)), float(np.nanmax(py))
        span = max(xmx - xmn, ymx - ymn, 1.0)
        pad = 0.08 * span
        self._ax_map.set_xlim(xmn - pad, xmx + pad)
        self._ax_map.set_ylim(ymn - pad, ymx + pad)
        self._span_map = span

        def gate(k: int, label: str, color: str) -> None:
            j = min(max(k, 1), n - 2)
            dx, dy = x[j + 1] - x[j - 1], y[j + 1] - y[j - 1]
            nrm = math.hypot(dx, dy) or 1.0
            qx, qy = -dy / nrm, dx / nrm
            ln_ = 0.03 * span
            art, = self._ax_map.plot([x[k] - qx * ln_, x[k] + qx * ln_], [y[k] - qy * ln_, y[k] + qy * ln_],
                                     color=color, lw=1.2, solid_capstyle="butt", zorder=5)
            tx = self._ax_map.text(x[k] + qx * ln_ * 2.2, y[k] + qy * ln_ * 2.2, label, fontsize=6, color=color,
                                   ha="center", va="center", zorder=6)
            self._map_decor.extend([art, tx])

        gate(0, "S/F", GREEN)
        lap = self._lap_sel
        if lap is not None and self._mode == "offline":
            ts = lap.t_start + np.cumsum(lap.sectors)[:2]
            ks = np.clip(np.searchsorted(self._t[i0:i1], ts), 1, n - 2)
            for q, k in enumerate(ks):
                gate(int(k), f"S{q + 1}", AMBER)

    def _update_delta(self) -> None:
        ax = self._ax_dt
        for c in list(ax.collections):
            c.remove()
        lap, ref = self._lap_sel, self._ref
        if lap is None or ref is None or self._an is None or self._mode != "offline":
            self._dt_line.set_data([], [])
            self.v_dt_hdr.set("select lap + reference")
            return
        d, dt = self._an.delta_time(lap, ref)
        sel = decimate_indices(d, dt, MAX_PTS, "lttb")
        d, dt = d[sel], dt[sel]
        self._dt_line.set_data(d, dt)
        ax.fill_between(d, 0, dt, where=dt <= 0, color=GREEN, alpha=0.5, interpolate=True, lw=0)
        ax.fill_between(d, 0, dt, where=dt > 0, color=RED, alpha=0.5, interpolate=True, lw=0)
        ax.set_xlim(float(d[0]), float(d[-1]) if d[-1] > d[0] else float(d[0]) + 1.0)
        m = max(float(np.nanmax(np.abs(dt))), 0.05) * 1.15
        ax.set_ylim(-m, m)
        self.v_dt_hdr.set(f"L{lap.index} vs L{ref.index} · end {dt[-1]:+.3f} s")

    def _update_sectors(self) -> None:
        lap, ref = self._lap_sel, self._ref
        for r, cells in enumerate(self._sec_cells):
            vals: list[tuple[float | None, str]] = []
            for la in (lap, ref):
                if la is None or self._mode != "offline":
                    vals.append((None, "--"))
                elif r < 3:
                    vals.append((la.sectors[r], f"{la.sectors[r]:.3f}"))
                else:
                    vals.append((la.lap_time, format_lap_time(la.lap_time)))
            cells[0].configure(text=vals[0][1])
            cells[1].configure(text=vals[1][1])
            if vals[0][0] is None or vals[1][0] is None:
                cells[2].configure(text="--", fg=K_TXT)
            else:
                dms = (vals[0][0] - vals[1][0]) * 1000.0
                cells[2].configure(text=f"{dms:+.0f} ms", fg=GREEN if dms < 0 else RED if dms > 0 else K_TXT)

    def _update_gg(self) -> None:
        i0, i1 = self._seg
        ay, ax_ = self._ch.get("ay"), self._ch.get("ax")
        if ay is None or ax_ is None or ay.size != self._t.size or i1 - i0 < 2:
            self._gg_sc.set_offsets(np.zeros((0, 2)))
            return
        s = max(1, (i1 - i0) // 5000)
        a, b = ay[i0:i1:s], ax_[i0:i1:s]
        m = np.isfinite(a) & np.isfinite(b)
        self._gg_sc.set_offsets(np.column_stack([a[m], b[m]]))

    def _update_hist(self) -> None:
        i0, i1 = self._seg
        cnt: dict[str, np.ndarray] = {}
        for c in CORNERS:
            v = self._ch.get(f"damper_velocity_{c}")
            z = self._ch.get(f"damper_travel_{c}")
            if v is None and z is not None and i1 - i0 > 8:
                v = np.concatenate([np.zeros(i0), damper_velocity(self._t[i0:i1], z[i0:i1]),
                                    np.zeros(max(self._t.size - i1, 0))])
            if v is None or v.size != self._t.size:
                cnt[c] = np.zeros(4)
                continue
            vv = v[i0:i1]
            vv = vv[np.isfinite(vv)]
            cnt[c] = np.array([((vv >= 0) & (vv <= 50)).sum(), (vv > 50).sum(),
                               ((vv < 0) & (vv >= -50)).sum(), (vv < -50).sum()], dtype=float)
        for c in CORNERS:
            pct = 100.0 * cnt[c] / max(cnt[c].sum(), 1.0)
            self._hist_ax[c].set_ylim(0, max(float(pct.max()) * 1.2, 10.0))
            for bar, h in zip(self._hist_bars[c], pct):
                bar.set_height(float(h))
            self._hist_lbls[c].configure(
                text=f"{c.upper()}  " + "  ".join(f"{lab} {pct[k]:4.1f}" for k, lab in enumerate(HIST_LABELS)))
        self._cv_h.draw_idle()

    def _update_diag_extremes(self) -> None:
        i0, i1 = self._seg
        for d in self._diag:
            if "key" in d:
                y = self._ch.get(d["key"])
                seg = y[i0:i1] if y is not None and y.size == self._t.size else np.zeros(0)
                if d["key"] in ("min_cell_voltage", "soc"):
                    d["ext"], d["ext_lbl"] = (_nanfn(np.min, seg) if seg.size else float("nan")), "min"
                else:
                    d["ext"] = _nanfn(lambda a: a[np.argmax(np.abs(a))], seg) if seg.size else float("nan")
                    d["ext_lbl"] = "pk"
            else:
                vals = [_nanfn(np.max, self._ch[f"{d['pre']}{c}"][i0:i1]) for c in CORNERS
                        if f"{d['pre']}{c}" in self._ch and self._ch[f"{d['pre']}{c}"].size == self._t.size]
                vals = [v for v in vals if np.isfinite(v)]
                d["ext"], d["ext_lbl"] = (max(vals) if vals else float("nan")), "max"

    def _update_diag(self, i: int) -> None:
        for d in self._diag:
            if "key" in d:
                v = self._val(d["key"], i)
            else:
                vs = [x for x in (self._val(f"{d['pre']}{c}", i) for c in CORNERS) if np.isfinite(x)]
                v = max(vs) if vs else float("nan")
            fin = np.isfinite(v)
            col = d["col"](v) if fin else K_DIM
            if d.get("power") and fin and v > POWER_LIMIT_KW:
                col = RED if self._flash else "#661020"
            d["val"].configure(text=("--" if not fin else d["fmt"].format(v)) + f" {d['unit']}", fg=col)
            e = d.get("ext", float("nan"))
            d["sub"].configure(text=f"{d.get('ext_lbl', '')} " + ("--" if not np.isfinite(e) else d["fmt"].format(e)))

    # ================================================================== cursor
    def _update_headers(self, i: int) -> None:
        for cells in self._hdr:
            for cell in cells:
                ch, lb, last, strip = cell
                txt = self._hdr_text(ch, strip, self._val(ch.key, i))
                if txt != last:
                    lb.configure(text=txt)
                    cell[2] = txt

    def _update_cursor(self, blit: bool = True) -> None:
        n = self._t.size
        i0, i1 = self._seg
        if n == 0 or self._xs.size == 0:
            for ln in self._cursors:
                ln.set_visible(False)
            if blit:
                self._blit_all()
            return
        i = int(np.clip(self._ci, i0, i1 - 1))
        self._ci = i
        x = float(self._xs[i - i0])
        for ln in self._cursors:
            ln.set_visible(True)
            ln.set_xdata([x, x])
        self._update_headers(i)
        t_rel = float(self._t[i] - self._t[i0]) if self._mode == "offline" else float(self._t[i])
        d_rel = float(self._dist[i] - self._dist[i0]) if self._dist.size == n else float("nan")
        lbl = f"t={t_rel:9.3f} s" + (f"  d={d_rel:7.1f} m" if np.isfinite(d_rel) else "")
        if self._lap_sel is not None:
            lbl += f"  L{self._lap_sel.index}"
        self.v_scrub.set(lbl)
        self.v_s_cur.set(f"CUR {lbl.strip()}")
        self._draw_scrub()
        k = i - i0
        if self._map_x.size > k and np.isfinite(self._map_x[k]) and np.isfinite(self._map_y[k]):
            px, py = float(self._map_x[k]), float(self._map_y[k])
            j = min(k + 8, self._map_x.size - 1)
            j0 = j if j != k else max(k - 8, 0)
            hx, hy = float(self._map_x[j] - self._map_x[j0]), float(self._map_y[j] - self._map_y[j0])
            nrm = math.hypot(hx, hy) or 1.0
            ln_ = 0.05 * getattr(self, "_span_map", 1.0)
            sgn = 1.0 if j != k else -1.0
            self._blip_ring.set_data([px], [py])
            self._blip_dot.set_data([px], [py])
            self._blip_vec.set_data([px, px + sgn * hx / nrm * ln_], [py, py + sgn * hy / nrm * ln_])
        else:
            for ln in (self._blip_ring, self._blip_dot, self._blip_vec):
                ln.set_data([], [])
        if self._lap_sel is not None and self._dist.size == n and self._mode == "offline":
            self._dt_cur.set_visible(True)
            dx = float(self._dist[i] - self._dist[i0])
            self._dt_cur.set_xdata([dx, dx])
        else:
            self._dt_cur.set_visible(False)
        ay, ax_ = self._val("ay", i), self._val("ax", i)
        if np.isfinite(ay) and np.isfinite(ax_):
            self._gg_pt.set_data([ay], [ax_])
            self._gg_vec.set_data([0.0, ay], [0.0, ax_])
            a = max(i0, i - 60)
            self._gg_trail.set_data(self._ch["ay"][a:i + 1], self._ch["ax"][a:i + 1])
        else:
            for ln in (self._gg_pt, self._gg_vec, self._gg_trail):
                ln.set_data([], [])
        self._update_ellipse(self._val("vx", i))
        self._update_diag(i)
        pw = self._val("battery_power_kw", i)
        over = np.isfinite(pw) and pw > POWER_LIMIT_KW
        self._lim_line.set_color((RED if self._flash else "#661020") if over else "#8a2a38")
        if blit:
            self._blit_all()

    def _update_ellipse(self, v_kmh: float) -> None:
        if not np.isfinite(v_kmh):
            self._gg_ell.set_visible(False)
            return
        bucket = max(5, int(round(v_kmh / 5.0)) * 5)
        self._request_ellipse(bucket)
        hit = self._ell_cache.get(bucket)
        if hit is None and self._ell_cache:
            hit = self._ell_cache[min(self._ell_cache, key=lambda b: abs(b - bucket))]
        if hit is None:
            self._gg_ell.set_visible(False)
            return
        self._gg_ell.set_width(2 * hit[0])
        self._gg_ell.set_height(2 * hit[1])
        self._gg_ell.set_visible(True)
        self.v_gg_hdr.set(f"MF6.1 @ {bucket} km/h · ay {hit[0]:.2f} g · ax {hit[1]:.2f} g")

    def _draw_scrub(self) -> None:
        cv = getattr(self, "_scrub_cv", None)
        if cv is None:
            return
        cv.delete("all")
        w, h = max(cv.winfo_width(), 2), max(cv.winfo_height(), 2)
        mid = h // 2
        cv.create_line(0, mid, w, mid, fill=K_LINE)
        i0, i1 = self._seg
        if self._t.size == 0 or i1 - i0 < 2:
            return
        span = max(i1 - 1 - i0, 1)
        frac = float(np.clip((self._ci - i0) / span, 0.0, 1.0))
        lap = self._lap_sel
        if lap is not None and self._mode == "offline" and lap.lap_time > 0:
            for cs in np.cumsum(lap.sectors)[:2]:
                xx = int(cs / lap.lap_time * w)
                cv.create_line(xx, 2, xx, h - 2, fill=AMBER)
        xp = int(frac * (w - 1))
        cv.create_line(0, mid, xp, mid, fill=COBALT, width=2)
        cv.create_rectangle(xp - 1, 1, xp + 1, h - 1, fill=WHITE, outline="")

    def _scrub_event(self, e: tk.Event) -> None:
        if self._xs.size < 2:
            return
        self._pause()
        w = max(self._scrub_cv.winfo_width(), 1)
        i0, i1 = self._seg
        self._ci = i0 + int(round(float(np.clip(e.x / w, 0.0, 1.0)) * (i1 - 1 - i0)))
        self._update_cursor()

    # ================================================================== mouse (strip canvases)
    def _x_from(self, ev: Any, si: int) -> float:
        return float(self._axl[si].transData.inverted().transform((ev.x, ev.y))[0])

    def _set_cursor_from_x(self, xdata: float) -> None:
        xs = self._xs
        if xs.size < 2:
            return
        k = int(np.clip(np.searchsorted(xs, xdata), 0, xs.size - 1))
        if k > 0 and abs(xs[k - 1] - xdata) < abs(xs[k] - xdata):
            k -= 1
        self._ci = self._seg[0] + k
        self._update_cursor()

    def _clamp_view(self, lo: float, hi: float) -> tuple[float, float]:
        a, b = float(self._xs[0]), float(self._xs[-1])
        w = min(hi - lo, b - a)
        lo = min(max(lo, a), b - w)
        return lo, lo + w

    def _on_motion(self, ev: Any) -> None:
        si = self._cv_idx.get(ev.canvas)
        if si is None or self._xs.size < 2:
            return
        d = self._drag
        if d is not None:
            if d["kind"] == "box":
                x0, x1 = sorted((d["x0"], self._x_from(ev, si)))
                for r in self._zrects:
                    r.set_x(x0)
                    r.set_width(x1 - x0)
                    r.set_visible(True)
                for bl in self._bls:
                    bl.update()
            elif d["kind"] == "pan":
                span = d["hi"] - d["lo"]
                shift = (ev.x - d["px"]) * span / max(self._axl[si].bbox.width, 1.0)
                self._axl[0].set_xlim(*self._clamp_view(d["lo"] - shift, d["hi"] - shift))
            elif d["kind"] == "scrub":
                self._set_cursor_from_x(self._x_from(ev, si))
            return
        if self._playing or (self._mode == "live" and not self._live_frozen):
            return
        if ev.inaxes in (self._axl[si], self._axr[si]) and ev.xdata is not None:
            self._set_cursor_from_x(self._x_from(ev, si))

    def _on_press(self, ev: Any) -> None:
        si = self._cv_idx.get(ev.canvas)
        if si is None or self._xs.size < 2 or ev.inaxes not in (self._axl[si], self._axr[si]):
            return
        if ev.dblclick:
            self._reset_view()
            self._refresh_strips()
            self._draw_all()
            return
        lo, hi = self._axl[0].get_xlim()
        if ev.button == 1:
            self._pause()
            self._drag = {"kind": "scrub"}
            self._set_cursor_from_x(self._x_from(ev, si))
        elif ev.button == 3:
            self._drag = {"kind": "box", "x0": self._x_from(ev, si), "px": ev.x}
        elif ev.button == 2:
            self._drag = {"kind": "pan", "px": ev.x, "lo": lo, "hi": hi}

    def _on_release(self, ev: Any) -> None:
        d, self._drag = self._drag, None
        if d is None:
            return
        if d["kind"] == "box":
            si = self._cv_idx.get(ev.canvas, 0)
            for r in self._zrects:
                r.set_visible(False)
            if abs(ev.x - d["px"]) > 4:
                x0, x1 = sorted((d["x0"], self._x_from(ev, si)))
                if x1 - x0 > 1e-9:
                    self._axl[0].set_xlim(*self._clamp_view(x0, x1))
                    return
            for bl in self._bls:
                bl.update()

    def _on_scroll(self, ev: Any) -> None:
        si = self._cv_idx.get(ev.canvas)
        if si is None or self._xs.size < 2 or ev.inaxes not in (self._axl[si], self._axr[si]):
            return
        if self._mode == "live" and not self._live_frozen:
            return
        lo, hi = self._axl[0].get_xlim()
        f = 0.8 if ev.button == "up" else 1.25
        x = self._x_from(ev, si)
        nlo, nhi = x - (x - lo) * f, x + (hi - x) * f
        if nhi - nlo < 1e-4 * max(float(self._xs[-1] - self._xs[0]), 1e-9):
            return
        self._axl[0].set_xlim(*self._clamp_view(nlo, nhi))

    def _on_xlim(self, ax: Any) -> None:
        if self._busy:
            return
        lo, hi = ax.get_xlim()
        self._busy = True
        try:
            for a in self._axl:
                if a is not ax:
                    a.set_xlim(lo, hi)
        finally:
            self._busy = False
        if self._deb_id is not None:
            try:
                self.after_cancel(self._deb_id)
            except tk.TclError:
                pass
        self._deb_id = self.after(25, self._xlim_refresh)

    def _xlim_refresh(self) -> None:
        self._deb_id = None
        self._refresh_strips()
        for cv in self._cvs:
            cv.draw_idle()

    # ================================================================== keyboard / transport
    def _bind_keys(self) -> None:
        if self._key_ids:
            return
        top = self.winfo_toplevel()
        for seq, fn in (("<space>", lambda e: self._key(e, self._toggle_play)),
                        ("<Left>", lambda e: self._key(e, lambda: self._step(-0.05))),
                        ("<Right>", lambda e: self._key(e, lambda: self._step(0.05))),
                        ("<Home>", lambda e: self._key(e, lambda: self._jump(False))),
                        ("<End>", lambda e: self._key(e, lambda: self._jump(True)))):
            self._key_ids.append((seq, top.bind(seq, fn, add="+")))

    def _unbind_keys(self) -> None:
        try:
            top = self.winfo_toplevel()
            for seq, fid in self._key_ids:
                top.unbind(seq, fid)
        except tk.TclError:
            pass
        self._key_ids.clear()

    @staticmethod
    def _key(e: tk.Event, fn: Any) -> None:
        if isinstance(e.widget, (tk.Entry, ttk.Entry, ttk.Combobox)):
            return
        fn()

    def _pause(self) -> None:
        if self._playing:
            self._playing = False
            self.v_play.set("PLAY")

    def _toggle_play(self) -> None:
        if self._mode == "live":
            self._live_frozen = not self._live_frozen
            self.v_play.set("RESUME" if self._live_frozen else "FREEZE")
            return
        if self._t.size == 0:
            return
        if self._playing:
            self._pause()
            return
        i0, i1 = self._seg
        if self._ci >= i1 - 1:
            self._ci = i0
        self._playing = True
        self._play_wall = time.perf_counter()
        self._play_t = float(self._t[self._ci])
        self.v_play.set("PAUSE")

    def _jump(self, end: bool) -> None:
        if self._t.size == 0:
            return
        self._pause()
        i0, i1 = self._seg
        self._ci = (i1 - 1) if end else i0
        self._update_cursor()

    def _step(self, dt: float) -> None:
        if self._t.size == 0 or (self._mode == "live" and not self._live_frozen):
            return
        self._pause()
        i0, i1 = self._seg
        self._ci = int(np.clip(np.searchsorted(self._t, float(self._t[self._ci]) + dt), i0, i1 - 1))
        self._update_cursor()

    # ================================================================== toolbar handlers
    def _on_lap_select(self, _e: Any = None) -> None:
        k = self._cb_lap.current()
        self._pause()
        self._set_segment(None if k <= 0 else self._laps[k - 1])
        self.focus_set()

    def _toggle_ref(self) -> None:
        if self.v_isref.get():
            self._ref = self._lap_sel or self._best
            if self._ref is None:
                self.v_isref.set(False)
        else:
            self._ref = None
        self.v_ref_lbl.set(f"REF L{self._ref.index}" if self._ref else "REF —")
        self._ghost_cache.clear()
        self._refresh_strips()
        self._update_delta()
        self._update_sectors()
        self._draw_all()

    def _on_xmode(self) -> None:
        if self._mode == "live":
            self.v_xmode.set("time")
            return
        self._rebuild_x()
        self._reset_view()
        self._refresh_strips()
        self._update_cursor(blit=False)
        self._draw_all()

    def _on_heat(self) -> None:
        self._update_map()
        self._update_cursor(blit=False)
        self._cv_map.draw_idle()
        self.focus_set()

    def _on_src(self, _e: Any = None) -> None:
        self.v_chan.set({"socketcan": "vcan0", "serial": "/dev/ttyUSB0", "udp": "0.0.0.0:5005"}[self.v_src.get()])

    def _on_mode(self) -> None:
        m = self.v_mode.get()
        if m == self._mode:
            return
        self._pause()
        if m == "live":
            self._stash = {k: getattr(self, k) for k in _OFF_KEYS}
            self._grp_off.pack_forget()
            self._grp_live.pack(side="left")
            self._mode = "live"
            self.v_xmode.set("time")
            self._live_frozen = False
            self.v_play.set("FREEZE")
            self._cb_lap.configure(state="disabled")
            self._clear_data()
        else:
            self._disconnect_live()
            self._grp_live.pack_forget()
            self._grp_off.pack(side="left")
            self._mode = "offline"
            self._cb_lap.configure(state="readonly")
            self.v_play.set("PLAY")
            if self._stash is not None:
                for k, v in self._stash.items():
                    setattr(self, k, v)
                self._stash = None
                self._populate_laps()
                if self._lap_sel in self._laps:
                    self.v_lap.set(self._cb_lap["values"][self._laps.index(self._lap_sel) + 1])
                self._set_segment(self._lap_sel)
            else:
                self._clear_data()

    def _clear_data(self) -> None:
        self._t, self._ch, self._dist = np.zeros(0), {}, np.zeros(0)
        self._an, self._laps, self._best, self._lap_sel, self._ref = None, [], None, None, None
        self._seg, self._xs, self._ci, self._gps = (0, 0), np.zeros(0), 0, None
        self._math_cache = None
        self._populate_laps()
        self._refresh_strips()
        self._update_map()
        self._update_delta()
        self._update_sectors()
        self._update_gg()
        self._update_hist()
        self._update_cursor(blit=False)
        self._draw_all()

    # ================================================================== live ingest
    def _toggle_connect(self) -> None:
        if self._ingest is not None:
            self._disconnect_live()
            return
        if not DBC_PATH.exists():
            messagebox.showerror("Live CAN", f"DBC not found:\n{DBC_PATH}", parent=self)
            return
        self._buffer = RingBuffer(RAW_CHANNELS, capacity=LIVE_CAPACITY)
        ing = CanIngest(self._buffer, self.v_src.get(), self.v_chan.get().strip(), DBC_PATH)
        ing.start()
        self._ingest = ing
        self._math_cache = None
        self.v_conn.set("DISCONNECT")

    def _replay_demo(self) -> None:
        if self._mode != "live":
            self.v_mode.set("live")
            self._on_mode()
        self._disconnect_live()
        self.v_badge.set("BUILDING DEMO…")
        self._run_bg(self._demo_worker)

    def _start_replay(self, log: LogData) -> None:
        self._buffer = RingBuffer(RAW_CHANNELS, capacity=LIVE_CAPACITY)
        ing = ReplayIngest(self._buffer, log.t, log.channels)
        ing.start()
        self._ingest = ing
        self._math_cache = None
        self.v_conn.set("DISCONNECT")
        self.v_badge.set("DEMO REPLAY (loop)")

    def _disconnect_live(self) -> None:
        ing, self._ingest = self._ingest, None
        if ing is not None:
            ing.stop()
            ing.join(timeout=1.0)
        try:
            self.v_conn.set("CONNECT")
        except tk.TclError:
            pass

    def _update_hud(self) -> None:
        ing = self._ingest
        cv = self._fill_cv
        cv.delete("all")
        if ing is None:
            self.v_rate.set("OFFLINE" if self._mode == "offline" else "NO LINK")
            self._lbl_rate.configure(fg=K_DIM if self._mode == "offline" else RED)
            self.v_frames.set("FRM 0")
            self.v_drop.set("DROP 0")
            self.v_s_buf.set("BUF —")
            self.v_s_hz.set(f"FS {self._fs:.0f} Hz")
            self.v_s_loss.set("LOSS 0")
            return
        s = ing.stats()
        hz = float(s["hz"])
        self._lbl_rate.configure(fg=RED if not s["connected"] else GREEN if hz >= 180.0 else AMBER)
        self.v_rate.set(f"{hz:5.0f} Hz")
        self.v_frames.set(f"FRM {s['frames']}")
        self.v_drop.set(f"DROP {s['dropped']}")
        fill = float(s["fill"])
        cv.create_rectangle(0, 0, int(80 * fill), 8, fill=GREEN if fill < 0.9 else AMBER, outline="")
        self.v_s_buf.set(f"BUF {100.0 * fill:5.1f} %")
        self.v_s_hz.set(f"FS {hz:.0f} Hz" + ("" if s["connected"] else " (down)"))
        self.v_s_loss.set(f"LOSS {s['dropped']}")
        self.v_s_file.set(f"SRC {self.v_src.get()}:{self.v_chan.get()}" if self._buffer is not None else "FILE —")
        if s.get("error"):
            self._post_status(f"Live ingest error: {s['error']}")

    # ================================================================== render tick
    def _render_tick(self) -> None:
        self._poll_id = None
        if not self._active:
            return
        t0 = time.perf_counter()
        try:
            self._drain_queue()
            if self._mode == "offline":
                self._tick_playback()
            else:
                self._tick_live()
            self._tick_n += 1
            if self._tick_n % 8 == 0:
                self._flash = not self._flash
                if not self._playing and self._mode == "offline":
                    pw = self._val("battery_power_kw", self._ci)
                    if np.isfinite(pw) and pw > POWER_LIMIT_KW:
                        self._update_cursor()
            if self._tick_n % 15 == 0:
                self._update_hud()
                self._app_state.set("last_draw_ms", self._last_ms)
        except Exception:  # noqa: BLE001  # the loop must survive any rendering error
            LOG.exception("telemetry render tick failed")
        self._last_ms = 1e3 * (time.perf_counter() - t0)
        if self._active and self._poll_id is None:
            self._poll_id = self.after(POLL_MS, self._render_tick)

    def _tick_playback(self) -> None:
        if not self._playing or self._t.size == 0:
            return
        now = time.perf_counter()
        self._play_t += (now - self._play_wall) * self._speed()
        self._play_wall = now
        i0, i1 = self._seg
        i = int(np.searchsorted(self._t, self._play_t))
        if i >= i1 - 1:
            i = i1 - 1
            self._pause()
        self._ci = max(i, i0)
        self._update_cursor()

    def _tick_live(self) -> None:
        buf = self._buffer
        if buf is None or self._live_frozen:
            return
        win = self._win_s()
        t, blk = buf.get_latest(int(win * FS))
        if t.size < 16:
            return
        ch = dict(blk)
        cache = self._math_cache
        if cache is not None:
            tm, mc = cache
            for k, v in mc.items():
                if k not in ch and tm.size > 1 and v.size == tm.size:
                    ch[k] = np.interp(t, tm, v)
        self._t, self._ch = t, ch
        self._dist = np.zeros(0)
        self._seg = (0, t.size)
        self._xs = t
        self._ci = t.size - 1
        self._live_n += 1
        self._log_name = "LIVE"
        now = time.perf_counter()
        if not self._math_busy and now - self._last_math > 1.5 and "vx" in ch:
            self._math_busy = True
            self._last_math = now
            n = min(t.size, int(60 * FS))
            self._run_bg(self._math_worker, t[-n:].copy(), {k: v[-n:].copy() for k, v in blk.items()},
                         self._app_state.get("front_tyre_params"), self._app_state.get("rear_tyre_params"))
        if self._live_n % 3 == 0:
            self._busy = True
            try:
                for a in self._axl:
                    a.set_xlim(float(t[-1]) - win, float(t[-1]))
            finally:
                self._busy = False
            self._refresh_strips()
            for cv in self._cvs:
                cv.draw_idle()
        if self._live_n % 15 == 0:
            self._update_map()
            self._update_gg()
            self._update_hist()
            self._update_diag_extremes()
            self._cv_map.draw_idle()
            self._cv_gg.draw_idle()
        self._update_cursor()


# ====================================================================== standalone entrypoint
def main() -> int:
    from apps.desktop.theme import apply_global_theme

    root = tk.Tk()
    root.title("TeR-Twin · Telemetry")
    root.geometry("1900x1000")
    apply_global_theme(root)
    state = AppState.instance()
    view = TelemetryView(root, state)
    view.pack(fill="both", expand=True)
    view._do_mount()
    view.on_activate()
    root.after(300, lambda: view._load_async(None))
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())