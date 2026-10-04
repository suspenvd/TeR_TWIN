#!/usr/bin/env python3
"""apps/desktop/views/telemetry_view.py
TeR-Twin Studio · Telemetry Analysis workstation (MoTeC i2 Pro style).

Offline analysis (.mf4 .mdf .csv .npz .mat) and live 200 Hz CAN streaming, 3 workspaces:
  left   : reconstructed 2D track map (speed/ay/brake heat ribbon, sector gates, animated car) + delta-t strip
  center : 4 stacked, x-linked strip charts + master scrubbar, crosshair with live numeric readouts
  right  : G-G diagram (+ friction ellipses of the active tyres), damper-velocity histograms, diagnostics

Assumptions (explicit)
----------------------
* Backend contract = ``ter_twin.telemetry``. ``CORNER_COLORS`` and the helpers ``gps_to_local``, ``cumtrapz``,
  ``damper_velocity``, ``fill_nan`` are NOT re-exported by ``ter_twin.telemetry.__init__``; they are imported from
  ``channel_definitions`` / ``math_channels`` directly.
* All file IO, math-channel computation, lap detection and friction-ellipse (JAX) evaluation run in daemon
  threads; results return through a ``queue.SimpleQueue`` drained by the 33 ms Tk poll. The Tk thread never
  touches sockets or files.
* Cursor/scrub/playback are sample-index based on the 200 Hz grid. X axis = time since segment start (offline)
  or absolute log time (live); lap-distance mode uses the trapezoid distance of ``LapAnalyzer`` (offline only).
* Track reconstruction: GPS (``gps_to_local``) when its spread is > 5 m; otherwise per-lap dead reckoning
  with (1) yaw-rate bias removal so the heading integral closes to n*2*pi and (2) a linear residual closure.
  "Session" view without GPS is raw (drifting) odometry. Live view uses the math-channel ``track_x/track_y``.
* Delta-t uses ``LapAnalyzer.delta_time``. The best valid lap is auto-selected as view lap and reference.
* Damper bins (mm/s): LSC [0,50], HSC >50, LSR [-50,0), HSR <-50; zero velocity counts as LSC.
* Watchdog thresholds are engineering assumptions (amber/red): min cell < 3.30/3.00 V, SoC < 20/10 %,
  inverter > 85/100 degC, motor > 110/125 degC, accumulator power > 72/80 kW.
* G-G ellipses come from ``friction_ellipses_g`` (40 and 90 km/h) with the tyres published by the Tyre view
  (``front_tyre_params`` / ``rear_tyre_params``); the braking-limit semi-axis is mirrored (accel is
  powertrain-limited, not shown). Without published tyres the nominal ``MF61Params()`` is used.
* Live DBC: ``config/can/dbc/TER.dbc``. ``Replay demo`` feeds the ring buffer from ``make_demo_log`` (no hardware).
* Blitting: cursors, readouts, blips and the G-G trail are animated artists; data lines redraw only on
  data / zoom / pan changes. Lines are updated with ``set_data`` (no ``clf``), min-max decimated to <= 2000 pts.
"""
from __future__ import annotations

import logging
import math
import queue
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Optional

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
for _p in (_ROOT / "src", _ROOT):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import matplotlib  # noqa: E402

matplotlib.use("TkAgg")
from matplotlib.artist import Artist  # noqa: E402
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

from apps.desktop.state import AppState  # noqa: E402
from apps.desktop.theme import (  # noqa: E402
    ACCENT_AMBER, ACCENT_BLUE, ACCENT_GREEN, ACCENT_RED, BG_CARD, BG_DARK, BG_HOVER, BORDER,
    TEXT_BRIGHT, TEXT_MUTED, TEXT_PRIMARY,
)
from apps.desktop.views.base_view import BaseView  # noqa: E402
from ter_twin.telemetry import (  # noqa: E402
    CORNERS, POWER_LIMIT_KW, RAW_CHANNELS, CanIngest, Lap, LapAnalyzer, LogData, ReplayIngest, RingBuffer,
    compute_math_channels, format_lap_time, friction_ellipses_g, get_spec, load_log, make_demo_log,
    minmax_indices,
)
from ter_twin.telemetry.channel_definitions import CORNER_COLORS  # noqa: E402
from ter_twin.telemetry.math_channels import cumtrapz, damper_velocity, fill_nan, gps_to_local  # noqa: E402

__all__ = ["TelemetryView"]
LOG = logging.getLogger("telemetry_view")

TXT = "#e6edf3"
POLL_MS = 33
FS = 200.0
MAX_PTS = 2000
LIVE_CAPACITY = 120_000
DBC_PATH = _ROOT / "config" / "can" / "dbc" / "TER.dbc"
HEAT_CHANNELS = ("vx", "ay", "brake_press_front", "ax", "throttle_pct", "battery_power_kw", "yaw_rate")
HIST_LABELS = ("HSR", "LSR", "LSC", "HSC")
HIST_COLORS = ("#d29922", "#f0883e", "#58a6ff", "#1f6feb")
_OFF_KEYS = ("_t", "_ch", "_dist", "_an", "_laps", "_best", "_lap_sel", "_ref", "_seg", "_xs", "_ci",
             "_log_name", "_fs", "_gps")


def _corner(fmt: str, short: str) -> tuple[tuple[str, str], ...]:
    return tuple((fmt.format(c), f"{short} {c.upper()}") for c in CORNERS)


STRIPS: tuple[dict[str, Any], ...] = (
    dict(L=(("vx", "Speed"), ("brake_press_front", "BrkF"), ("throttle_pct", "Thr")),
         R=(("battery_power_kw", "Pwr"),), yl="km/h · bar · %", yr="kW"),
    dict(L=(("ay", "Ay"), ("ax", "Ax")), R=(("steer_angle", "Steer"), ("yaw_rate", "Yaw")),
         yl="g", yr="deg · deg/s"),
    dict(L=_corner("wheel_speed_{}", "WS"), R=_corner("slip_ratio_{}", "SR"), yl="km/h", yr="slip %"),
    dict(L=_corner("damper_travel_{}", "DT"), R=_corner("tire_temp_{}", "TT"), yl="mm", yr="°C"),
)


def _fmt(key: str, v: float) -> str:
    return "--" if v is None or not np.isfinite(v) else f"{v:.{get_spec(key).precision}f}"


def _style(ax: Any, grid: bool = True) -> None:
    ax.set_facecolor(BG_CARD)
    for s in ax.spines.values():
        s.set_color(BORDER)
    ax.tick_params(colors=TEXT_MUTED, labelsize=7, length=2)
    if grid:
        ax.grid(True, color=BG_HOVER, lw=0.6, alpha=0.9)


def _nanfn(fn: Any, arr: Any) -> float:
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    return float(fn(a)) if a.size else float("nan")


class _Blit:
    """Blitting helper: animated artists are repainted over a cached clean background."""

    def __init__(self, canvas: FigureCanvasTkAgg) -> None:
        self.canvas = canvas
        self.fig = canvas.figure
        self.bg: Any = None
        self.artists: list[Artist] = []
        canvas.mpl_connect("draw_event", self._on_draw)

    def add(self, *arts: Artist) -> None:
        for a in arts:
            a.set_animated(True)
            self.artists.append(a)

    def _on_draw(self, _event: Any) -> None:
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
        self._poll_id: Optional[str] = None
        self._deb_id: Optional[str] = None
        self._active = False
        self._mode = "offline"
        self._busy = False                      # guards programmatic xlim changes
        self._tick_n = 0
        # data
        self._t = np.zeros(0)
        self._ch: dict[str, np.ndarray] = {}
        self._dist = np.zeros(0)
        self._an: Optional[LapAnalyzer] = None
        self._laps: list[Lap] = []
        self._best: Optional[Lap] = None
        self._lap_sel: Optional[Lap] = None
        self._ref: Optional[Lap] = None
        self._seg = (0, 0)
        self._xs = np.zeros(0)
        self._ci = 0
        self._log_name = "—"
        self._fs = FS
        self._gps: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._stash: Optional[dict[str, Any]] = None
        self._map_x = np.zeros(0)
        self._map_y = np.zeros(0)
        self._map_decor: list[Artist] = []
        self._ell_patches: list[Artist] = []
        self._loading = False
        # playback
        self._playing = False
        self._play_wall = 0.0
        self._play_t = 0.0
        self._scrub_lock = False
        self._blink = 0
        # live
        self._buffer: Optional[RingBuffer] = None
        self._ingest: Any = None
        self._math_cache: Optional[tuple[np.ndarray, dict[str, np.ndarray]]] = None
        self._math_busy = False
        self._last_math = 0.0
        self._live_frozen = False
        self._live_n = 0
        self._ell_busy = False
        self._ell_dirty = False
        self._last_ms = 0.0

    # ------------------------------------------------------------------ lifecycle
    def on_mount(self) -> None:
        self.v_mode = tk.StringVar(value="offline")
        self.v_xmode = tk.StringVar(value="time")
        self.v_lap = tk.StringVar()
        self.v_isref = tk.BooleanVar(value=False)
        self.v_speed = tk.StringVar(value="1.0x")
        self.v_play = tk.StringVar(value="▶")
        self.v_heat = tk.StringVar(value="vx")
        self.v_cmap = tk.StringVar(value="turbo")
        self.v_src = tk.StringVar(value="socketcan")
        self.v_chan = tk.StringVar(value="vcan0")
        self.v_win = tk.StringVar(value="30 s")
        self.v_badge = tk.StringVar(value="no log loaded")
        self.v_hud = tk.StringVar(value="— Hz · 0 frames · 0 dropped")
        self.v_fill = tk.DoubleVar(value=0.0)
        self.v_conn = tk.StringVar(value="Connect")
        self.v_ref_lbl = tk.StringVar(value="Ref: —")
        self.v_s_log = tk.StringVar(value="Log: —")
        self.v_s_fs = tk.StringVar(value="fs: — Hz")
        self.v_s_cur = tk.StringVar(value="cursor: —")
        self.v_s_rate = tk.StringVar(value="packets: — Hz")
        self.v_s_fill = tk.StringVar(value="buffer: —")
        self.v_scrub = tk.StringVar(value="")
        self._build_toolbar()
        self._build_statusbar()
        self._build_body()
        self.bind("<Destroy>", self._on_destroy, add="+")
        self._start_ellipse_worker()

    def on_activate(self) -> None:
        self._do_mount()
        self._active = True
        self._draw_all()
        if self._poll_id is None:
            self._poll_id = self.after(POLL_MS, self._poll_tick)

    def on_deactivate(self) -> None:
        self._active = False
        self._pause()
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
            self._start_ellipse_worker()

    def _on_destroy(self, event: tk.Event) -> None:
        if event.widget is not self:
            return
        self._active = False
        self._disconnect_live()
        for attr in ("_poll_id", "_deb_id"):
            aid = getattr(self, attr)
            if aid is not None:
                try:
                    self.after_cancel(aid)
                except tk.TclError:
                    pass
                setattr(self, attr, None)

    # ------------------------------------------------------------------ toolbar
    def _build_toolbar(self) -> None:
        bar = tk.Frame(self, bg=BG_CARD)
        bar.pack(side="top", fill="x")
        r1 = tk.Frame(bar, bg=BG_CARD)
        r1.pack(fill="x", padx=6, pady=(5, 2))
        r2 = tk.Frame(bar, bg=BG_CARD)
        r2.pack(fill="x", padx=6, pady=(2, 5))
        tk.Frame(bar, bg=BORDER, height=1).pack(fill="x")

        seg = tk.Frame(r1, bg=BORDER)
        seg.pack(side="left", padx=(0, 10))
        for txt, val in (("OFFLINE LOG ANALYSIS", "offline"), ("LIVE CAN STREAM", "live")):
            tk.Radiobutton(seg, text=txt, value=val, variable=self.v_mode, indicatoron=False, command=self._on_mode,
                           bg=BG_HOVER, fg=TEXT_PRIMARY, selectcolor="#1f6feb", activebackground=BG_HOVER,
                           activeforeground=TEXT_BRIGHT, relief="flat", bd=0, padx=12, pady=4, highlightthickness=0,
                           font=("Segoe UI", 8, "bold"), cursor="hand2").pack(side="left", padx=(0, 1), pady=1)

        self._slot = tk.Frame(r1, bg=BG_CARD)
        self._slot.pack(side="left")
        self._grp_off = tk.Frame(self._slot, bg=BG_CARD)
        ttk.Button(self._grp_off, text="Open Log…", command=self._open_log).pack(side="left")
        ttk.Button(self._grp_off, text="Load Demo", command=lambda: self._load_async(None)).pack(side="left", padx=3)
        tk.Label(self._grp_off, textvariable=self.v_badge, bg=BG_HOVER, fg=ACCENT_BLUE, padx=8, pady=3,
                 font=("Consolas", 8)).pack(side="left", padx=4)
        self._grp_off.pack(side="left")
        self._grp_live = tk.Frame(self._slot, bg=BG_CARD)
        self._cb_src = ttk.Combobox(self._grp_live, textvariable=self.v_src, state="readonly", width=10,
                                    values=("socketcan", "serial", "udp"))
        self._cb_src.pack(side="left")
        self._cb_src.bind("<<ComboboxSelected>>", self._on_src)
        ttk.Entry(self._grp_live, textvariable=self.v_chan, width=16).pack(side="left", padx=3)
        ttk.Button(self._grp_live, textvariable=self.v_conn, command=self._toggle_connect, width=10).pack(side="left")
        ttk.Button(self._grp_live, text="Replay demo", command=self._replay_demo).pack(side="left", padx=3)
        ttk.Label(self._grp_live, text="Window").pack(side="left", padx=(8, 2))
        ttk.Combobox(self._grp_live, textvariable=self.v_win, state="readonly", width=6,
                     values=("10 s", "30 s", "60 s", "120 s")).pack(side="left")

        ttk.Label(r1, text="  Lap").pack(side="left", padx=(12, 2))
        self._cb_lap = ttk.Combobox(r1, textvariable=self.v_lap, state="readonly", width=40)
        self._cb_lap.pack(side="left")
        self._cb_lap.bind("<<ComboboxSelected>>", self._on_lap_select)
        tk.Checkbutton(r1, text="Set Reference Lap", variable=self.v_isref, indicatoron=False, command=self._toggle_ref,
                       bg=BG_HOVER, fg=TEXT_PRIMARY, selectcolor="#1f6feb", activebackground=BG_HOVER,
                       relief="flat", padx=8, pady=3, highlightthickness=0, font=("Segoe UI", 8),
                       cursor="hand2").pack(side="left", padx=6)
        tk.Label(r1, textvariable=self.v_ref_lbl, bg=BG_CARD, fg=ACCENT_AMBER, font=("Consolas", 8)).pack(side="left")
        self._rb_x: list[tk.Radiobutton] = []
        for txt, val in (("Time [s]", "time"), ("Lap Distance [m]", "dist")):
            rb = tk.Radiobutton(r1, text=txt, value=val, variable=self.v_xmode, indicatoron=False,
                                command=self._on_xmode, bg=BG_HOVER, fg=TEXT_PRIMARY, selectcolor="#1f6feb",
                                activebackground=BG_HOVER, relief="flat", padx=8, pady=3, highlightthickness=0,
                                font=("Segoe UI", 8), cursor="hand2")
            rb.pack(side="right", padx=1)
            self._rb_x.append(rb)

        for txt, fn in (("|<", lambda: self._jump(False)), ("<", lambda: self._step(-0.05))):
            ttk.Button(r2, text=txt, width=3, command=fn).pack(side="left")
        ttk.Button(r2, textvariable=self.v_play, width=4, command=self._toggle_play).pack(side="left", padx=2)
        for txt, fn in ((">", lambda: self._step(0.05)), (">|", lambda: self._jump(True))):
            ttk.Button(r2, text=txt, width=3, command=fn).pack(side="left")
        ttk.Combobox(r2, textvariable=self.v_speed, state="readonly", width=6,
                     values=("0.25x", "0.5x", "1.0x", "2.0x", "5.0x", "10.0x")).pack(side="left", padx=6)
        ttk.Label(r2, text="Track colour").pack(side="left", padx=(14, 2))
        cb = ttk.Combobox(r2, textvariable=self.v_heat, state="readonly", width=16, values=HEAT_CHANNELS)
        cb.pack(side="left")
        cb.bind("<<ComboboxSelected>>", lambda _e: self._on_heat())
        cb2 = ttk.Combobox(r2, textvariable=self.v_cmap, state="readonly", width=9,
                           values=("turbo", "coolwarm", "viridis"))
        cb2.pack(side="left", padx=3)
        cb2.bind("<<ComboboxSelected>>", lambda _e: self._on_heat())

        hud = tk.Frame(r2, bg=BG_HOVER)
        hud.pack(side="right")
        tk.Label(hud, text="INGEST", bg=BG_HOVER, fg=TEXT_MUTED, font=("Segoe UI", 7, "bold")).pack(side="left", padx=(8, 4))
        tk.Label(hud, textvariable=self.v_hud, bg=BG_HOVER, fg=ACCENT_GREEN, font=("Consolas", 8)).pack(side="left")
        ttk.Progressbar(hud, variable=self.v_fill, maximum=100.0, length=90).pack(side="left", padx=8, pady=3)

    def _build_statusbar(self) -> None:
        bar = tk.Frame(self, bg=BG_HOVER)
        bar.pack(side="bottom", fill="x")
        for var, col in ((self.v_s_log, ACCENT_BLUE), (self.v_s_fs, TEXT_MUTED), (self.v_s_cur, TXT),
                         (self.v_s_rate, ACCENT_GREEN), (self.v_s_fill, TEXT_MUTED)):
            tk.Label(bar, textvariable=var, bg=BG_HOVER, fg=col, font=("Consolas", 8), padx=10).pack(side="left")

    # ------------------------------------------------------------------ body
    def _build_body(self) -> None:
        self._pane = tk.PanedWindow(self, orient=tk.HORIZONTAL, bg=BG_DARK, sashwidth=5, sashrelief="flat",
                                    bd=0, opaqueresize=True)
        self._pane.pack(fill="both", expand=True)
        left = tk.PanedWindow(self._pane, orient=tk.VERTICAL, bg=BG_DARK, sashwidth=5, bd=0)
        center = tk.Frame(self._pane, bg=BG_DARK)
        right = tk.PanedWindow(self._pane, orient=tk.VERTICAL, bg=BG_DARK, sashwidth=5, bd=0)
        self._pane.add(left, minsize=240, stretch="always")
        self._pane.add(center, minsize=500, stretch="always")
        self._pane.add(right, minsize=260, stretch="always")
        self._sash_done = False
        self._pane.bind("<Configure>", self._init_sashes)

        f_map = tk.Frame(left, bg=BG_DARK)
        f_dt = tk.Frame(left, bg=BG_DARK)
        left.add(f_map, minsize=200, stretch="always")
        left.add(f_dt, minsize=120, stretch="always")
        self._build_map(f_map)
        self._build_delta(f_dt)
        self._build_strips(center)
        f_gg = tk.Frame(right, bg=BG_DARK)
        f_h = tk.Frame(right, bg=BG_DARK)
        f_d = tk.Frame(right, bg=BG_CARD)
        right.add(f_gg, minsize=200, stretch="always")
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
            self._pane.sash_place(0, int(ev.width * 0.20), 0)
            self._pane.sash_place(1, int(ev.width * 0.75), 0)
        except tk.TclError:
            pass

    @staticmethod
    def _canvas(parent: tk.Widget, fig: Figure, toolbar: bool = False) -> tuple[FigureCanvasTkAgg, _Blit, Any]:
        canvas = FigureCanvasTkAgg(fig, master=parent)
        nav = None
        if toolbar:
            nav = NavigationToolbar2Tk(canvas, parent, pack_toolbar=False)
            nav.config(background=BG_HOVER)
            for w in nav.winfo_children():
                try:
                    w.configure(background=BG_HOVER)
                except tk.TclError:
                    pass
            nav.update()
            nav.pack(side="bottom", fill="x")
        w = canvas.get_tk_widget()
        w.configure(bg=BG_DARK, highlightthickness=0)
        w.pack(fill="both", expand=True)
        return canvas, _Blit(canvas), nav

    # ---- map
    def _build_map(self, parent: tk.Widget) -> None:
        fig = Figure(figsize=(3, 3), facecolor=BG_DARK)
        ax = fig.add_axes([0.01, 0.01, 0.98, 0.98])
        ax.set_facecolor(BG_DARK)
        ax.set_axis_off()
        ax.set_aspect("equal", adjustable="datalim")
        self._ax_map = ax
        self._map_under, = ax.plot([], [], color="#010409", lw=8.5, solid_capstyle="round", zorder=1)
        self._map_lc = LineCollection([], linewidths=4.0, capstyle="round", zorder=2)
        self._map_lc.set_cmap("turbo")
        ax.add_collection(self._map_lc)
        self._map_txt = ax.text(0.015, 0.012, "", transform=ax.transAxes, fontsize=7, color=TEXT_MUTED, va="bottom")
        ax.text(0.015, 0.985, "TRACK MAP", transform=ax.transAxes, fontsize=7, color=ACCENT_BLUE, va="top",
                fontweight="bold")
        self._blip_glow, = ax.plot([], [], "o", ms=16, color=ACCENT_RED, alpha=0.35, zorder=9)
        self._blip, = ax.plot([], [], "o", ms=7, color="#ffffff", mec=ACCENT_RED, mew=1.8, zorder=10)
        self._cv_map, self._bl_map, _ = self._canvas(parent, fig)
        self._bl_map.add(self._blip_glow, self._blip)

    # ---- delta
    def _build_delta(self, parent: tk.Widget) -> None:
        fig = Figure(figsize=(3, 2), facecolor=BG_DARK)
        ax = fig.add_axes([0.16, 0.2, 0.80, 0.68])
        _style(ax)
        ax.axhline(0, color=BORDER, lw=0.9)
        ax.set_xlabel("Lap distance [m]", color=TEXT_MUTED, fontsize=7)
        ax.set_ylabel("Δt vs ref [s]", color=TEXT_MUTED, fontsize=7)
        self._ax_dt = ax
        self._dt_line, = ax.plot([], [], color=TXT, lw=1.0)
        self._dt_msg = ax.text(0.5, 0.5, "Select a lap and a reference lap", transform=ax.transAxes, ha="center",
                               va="center", fontsize=8, color=TEXT_MUTED)
        self._dt_title = ax.set_title("", fontsize=8, color=TXT, pad=3)
        self._dt_cur = ax.axvline(0, color=TXT, lw=0.9)
        self._cv_dt, self._bl_dt, _ = self._canvas(parent, fig)
        self._bl_dt.add(self._dt_cur)

    # ---- strips
    def _build_strips(self, parent: tk.Widget) -> None:
        scrub_row = tk.Frame(parent, bg=BG_CARD)
        scrub_row.pack(side="bottom", fill="x")
        self._scrub = ttk.Scale(scrub_row, from_=0, to=1000, orient="horizontal", command=self._on_scrub)
        self._scrub.pack(side="left", fill="x", expand=True, padx=(8, 8), pady=4)
        tk.Label(scrub_row, textvariable=self.v_scrub, bg=BG_CARD, fg=TXT, font=("Consolas", 8), width=44,
                 anchor="e").pack(side="right", padx=8)

        fig = Figure(figsize=(8, 8), facecolor=BG_DARK)
        axs = list(fig.subplots(4, 1, sharex=True))
        fig.subplots_adjust(left=0.065, right=0.93, top=0.992, bottom=0.06, hspace=0.05)
        self._axl = axs
        self._axr = [a.twinx() for a in axs]
        self._all_axes = set(self._axl) | set(self._axr)
        self._lines: list[tuple[str, Any, Any]] = []
        self._readouts: list[tuple[str, str, Any]] = []
        self._cursors: list[Any] = []
        for si, spec in enumerate(STRIPS):
            ax, axr = self._axl[si], self._axr[si]
            _style(ax)
            _style(axr, grid=False)
            ax.margins(y=0.08)
            axr.margins(y=0.08)
            ax.set_ylabel(spec["yl"], color=TEXT_MUTED, fontsize=7)
            axr.set_ylabel(spec["yr"], color=TEXT_MUTED, fontsize=7)
            j = 0
            for side, tgt in (("L", ax), ("R", axr)):
                for key, short in spec[side]:
                    sp = get_spec(key)
                    ln, = tgt.plot([], [], color=sp.color_hex, lw=1.0, ls=sp.linestyle)
                    self._lines.append((key, ln, tgt))
                    txt = ax.text(0.008 + (j % 4) * 0.245, 0.965 - (j // 4) * 0.135, "", transform=ax.transAxes,
                                  fontsize=6.5, color=sp.color_hex, va="top", family="monospace",
                                  bbox=dict(facecolor=BG_DARK, alpha=0.6, pad=1.2, edgecolor="none"))
                    self._readouts.append((key, short, txt))
                    j += 1
            cur = ax.axvline(0, color=TXT, lw=0.9, alpha=0.9)
            self._cursors.append(cur)
        self._axl[3].set_xlabel("Time [s]", color=TEXT_MUTED, fontsize=7)
        axr0 = self._axr[0]
        axr0.axhline(POWER_LIMIT_KW, color=ACCENT_RED, ls="--", lw=1.1, alpha=0.9)
        axr0.text(0.995, POWER_LIMIT_KW, f"{POWER_LIMIT_KW:.0f} kW LIMIT", fontsize=6.5, color=ACCENT_RED, ha="right",
                  va="bottom", transform=blended_transform_factory(axr0.transAxes, axr0.transData))
        self._ln_over, = axr0.plot([], [], color=ACCENT_RED, lw=2.4, alpha=0.95)
        self._time_txt = self._axl[3].text(0, 0.02, "", transform=blended_transform_factory(
            self._axl[3].transData, self._axl[3].transAxes), fontsize=7, color=TXT, va="bottom", family="monospace",
            bbox=dict(facecolor="#1f6feb", alpha=0.85, pad=1.5, edgecolor="none"))

        self._cv_st, self._bl_st, self._nav = self._canvas(parent, fig, toolbar=True)
        self._bl_st.add(*self._cursors, *[t for _, _, t in self._readouts], self._time_txt)
        for a in self._all_axes:
            a.callbacks.connect("xlim_changed", self._on_xlim)
        self._cv_st.mpl_connect("motion_notify_event", self._on_motion)
        self._cv_st.mpl_connect("button_press_event", self._on_press)
        self._cv_st.mpl_connect("button_release_event", self._on_release)
        self._cv_st.mpl_connect("scroll_event", self._on_scroll)
        self._dragging = False

    # ---- G-G
    def _build_gg(self, parent: tk.Widget) -> None:
        fig = Figure(figsize=(3, 3), facecolor=BG_DARK)
        ax = fig.add_axes([0.14, 0.12, 0.82, 0.80])
        _style(ax)
        ax.axhline(0, color=BORDER, lw=0.9)
        ax.axvline(0, color=BORDER, lw=0.9)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-2.4, 2.4)
        ax.set_ylim(-2.4, 2.4)
        ax.set_xlabel("ay [g]  (+ = left)", color=TEXT_MUTED, fontsize=7)
        ax.set_ylabel("ax [g]", color=TEXT_MUTED, fontsize=7)
        ax.set_title("G-G DIAGRAM", fontsize=8, color=ACCENT_BLUE, pad=3, fontweight="bold")
        self._ax_gg = ax
        self._gg_norm = Normalize(0, 100)
        self._gg_sc = ax.scatter([], [], s=3, c=[], cmap="turbo", norm=self._gg_norm, alpha=0.8, linewidths=0,
                                 rasterized=True)
        self._gg_trail, = ax.plot([], [], color="#ffffff", lw=1.2, alpha=0.8)
        self._gg_pt, = ax.plot([], [], "o", ms=8, color="#ffffff", mec=ACCENT_RED, mew=2.0)
        self._cv_gg, self._bl_gg, _ = self._canvas(parent, fig)
        self._bl_gg.add(self._gg_trail, self._gg_pt)

    # ---- histograms
    def _build_hist(self, parent: tk.Widget) -> None:
        self._lbl_fr = tk.Label(parent, text="", bg=BG_DARK, fg=TXT, font=("Consolas", 8), anchor="w")
        self._lbl_rr = tk.Label(parent, text="", bg=BG_DARK, fg=TXT, font=("Consolas", 8), anchor="w")
        self._lbl_rr.pack(side="bottom", fill="x", padx=6)
        self._lbl_fr.pack(side="bottom", fill="x", padx=6)
        fig = Figure(figsize=(3, 3), facecolor=BG_DARK)
        axs = fig.subplots(2, 2).ravel()
        fig.subplots_adjust(left=0.1, right=0.97, top=0.9, bottom=0.08, hspace=0.45, wspace=0.28)
        fig.suptitle("DAMPER VELOCITY DISTRIBUTION [%]", fontsize=8, color=ACCENT_BLUE, fontweight="bold", y=0.985)
        self._hist_bars: dict[str, Any] = {}
        self._hist_txt: dict[str, list[Any]] = {}
        self._hist_ax: dict[str, Any] = {}
        for ax, c in zip(axs, CORNERS):
            _style(ax)
            bars = ax.bar(range(4), [0, 0, 0, 0], color=HIST_COLORS, width=0.72)
            ax.set_xticks(range(4))
            ax.set_xticklabels(HIST_LABELS, fontsize=6.5)
            ax.set_ylim(0, 100)
            ax.set_title(c.upper(), fontsize=8, color=CORNER_COLORS[c], pad=2, fontweight="bold")
            self._hist_bars[c] = bars
            self._hist_ax[c] = ax
            self._hist_txt[c] = [ax.text(i, 0, "", ha="center", va="bottom", fontsize=6.5, color=TXT)
                                 for i in range(4)]
        self._cv_h, self._bl_h, _ = self._canvas(parent, fig)

    # ---- diagnostics
    def _build_diag(self, parent: tk.Widget) -> None:
        tk.Label(parent, text="DIAGNOSTICS WATCHDOG", bg=BG_CARD, fg=ACCENT_BLUE,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=8, pady=(6, 2))
        grid = tk.Frame(parent, bg=BG_CARD)
        grid.pack(fill="x", padx=6, pady=(0, 6))
        self._diag: list[dict[str, Any]] = [
            dict(name="MIN CELL V", unit="V", fmt="{:.3f}", warn=(3.30, 3.00, "lo"), key="min_cell_voltage"),
            dict(name="SoC", unit="%", fmt="{:.1f}", warn=(20.0, 10.0, "lo"), key="soc"),
            dict(name="MAX INVERTER", unit="°C", fmt="{:.1f}", warn=(85.0, 100.0, "hi"), pre="inverter_temp_"),
            dict(name="MAX MOTOR", unit="°C", fmt="{:.1f}", warn=(110.0, 125.0, "hi"), pre="motor_temp_"),
            dict(name="TV YAW MOMENT", unit="N·m", fmt="{:+.0f}", warn=None, key="tv_yaw_moment"),
            dict(name="ACCU POWER", unit="kW", fmt="{:.1f}", warn=(72.0, 80.0, "hi"), key="battery_power_kw"),
        ]
        for k, d in enumerate(self._diag):
            cell = tk.Frame(grid, bg=BG_HOVER)
            cell.grid(row=k // 2, column=k % 2, sticky="nsew", padx=2, pady=2)
            grid.columnconfigure(k % 2, weight=1)
            tk.Label(cell, text=d["name"], bg=BG_HOVER, fg=TEXT_MUTED, font=("Segoe UI", 7, "bold")).pack(anchor="w", padx=6)
            d["val"] = tk.Label(cell, text="--", bg=BG_HOVER, fg=TEXT_BRIGHT, font=("Consolas", 15, "bold"))
            d["val"].pack(anchor="w", padx=6)
            d["sub"] = tk.Label(cell, text="", bg=BG_HOVER, fg=TEXT_MUTED, font=("Consolas", 7))
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

    def _rebuild_x(self) -> None:
        n = self._t.size
        i0, i1 = self._seg
        if n == 0 or i1 - i0 < 2:
            self._xs = np.zeros(0)
            return
        if self._mode == "live":
            full = self._t
        elif self.v_xmode.get() == "dist" and self._dist.size == n:
            full = self._dist - self._dist[i0]
        else:
            full = self._t - self._t[i0]
        self._xs = np.asarray(full[i0:i1], dtype=float)
        lbl = "Time [s]" if (self._mode == "live" or self.v_xmode.get() == "time") else "Lap distance [m]"
        self._axl[3].set_xlabel(lbl, color=TEXT_MUTED, fontsize=7)

    def _gps_xy(self, ch: dict[str, np.ndarray]) -> Optional[tuple[np.ndarray, np.ndarray]]:
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
            if closed:
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

    # ================================================================== background workers
    def _run_bg(self, fn: Any, *args: Any) -> None:
        threading.Thread(target=fn, args=args, daemon=True).start()

    def _open_log(self) -> None:
        p = filedialog.askopenfilename(parent=self, title="Open telemetry log", filetypes=[
            ("Telemetry logs", "*.mf4 *.mdf *.csv *.npz *.mat"), ("All files", "*.*")])
        if p:
            self._load_async(Path(p))

    def _load_async(self, path: Optional[Path]) -> None:
        if self._loading:
            return
        if self._mode != "offline":
            self.v_mode.set("offline")
            self._on_mode()
        self._loading = True
        self.v_badge.set("loading…")
        self._run_bg(self._load_worker, path)

    def _load_worker(self, path: Optional[Path]) -> None:
        try:
            log: LogData = make_demo_log(n_laps=6) if path is None else load_log(path)
            front = self._app_state.get("front_tyre_params")
            rear = self._app_state.get("rear_tyre_params")
            math_ch = compute_math_channels(log.t, log.channels, None, front, rear)
            ch = {**log.channels, **math_ch}
            an: Optional[LapAnalyzer] = None
            laps: list[Lap] = []
            if "vx" in ch:
                an = LapAnalyzer(log.t, ch["vx"])
                if "lap_beacon" in ch:
                    laps = an.detect_from_beacon(ch["lap_beacon"])
                if not laps and "track_x" in ch and "track_y" in ch:
                    laps = an.detect_from_gate(ch["track_x"], ch["track_y"])
            dist = an.dist if an is not None else np.zeros(log.t.size)
            self._q.put(("loaded", dict(t=log.t, ch=ch, an=an, laps=laps, dist=dist, fs=log.fs,
                                        name="DEMO (synthetic)" if path is None else path.name,
                                        gps=self._gps_xy(ch))))
        except Exception as exc:  # noqa: BLE001
            LOG.exception("load failed")
            self._q.put(("load_error", f"{type(exc).__name__}: {exc}"))

    def _ellipse_worker(self, front: Any, rear: Any) -> None:
        try:
            res = friction_ellipses_g(front, rear, None, (40.0 / 3.6, 90.0 / 3.6))
        except Exception:  # noqa: BLE001
            LOG.exception("friction ellipses failed")
            res = []
        self._q.put(("ellipses", res))

    def _start_ellipse_worker(self) -> None:
        if self._ell_busy:
            self._ell_dirty = True
            return
        self._ell_busy = True
        self._ell_dirty = False
        self._run_bg(self._ellipse_worker, self._app_state.get("front_tyre_params"),
                     self._app_state.get("rear_tyre_params"))

    def _math_worker(self, t: np.ndarray, ch: dict[str, np.ndarray], front: Any, rear: Any) -> None:
        try:
            out = compute_math_channels(t, ch, None, front, rear)
            self._q.put(("math", (t, out)))
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
                self.v_badge.set("load failed")
                messagebox.showerror("Telemetry", str(payload), parent=self)
            elif tag == "ellipses":
                self._ell_busy = False
                self._apply_ellipses(payload)
                if self._ell_dirty:
                    self._start_ellipse_worker()
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
        self.v_badge.set(f"{self._log_name} · {dur:.1f} s · {self._fs:.0f} Hz")
        self._set_segment(start)

    def _populate_laps(self) -> None:
        vals = ["Session (all)"]
        for la in self._laps:
            tag = ("  [BEST]" if la is self._best else "") + ("" if la.valid else "  (invalid)")
            vals.append(f"Lap {la.index}: {format_lap_time(la.lap_time)} (Vmax: {la.v_max_kmh:.0f} km/h){tag}")
        self._cb_lap["values"] = vals
        self.v_lap.set(vals[0])
        self.v_ref_lbl.set(f"Ref: Lap {self._ref.index}" if self._ref else "Ref: —")

    def _apply_ellipses(self, res: list[tuple[float, float, float]]) -> None:
        for a in self._ell_patches:
            try:
                a.remove()
            except (ValueError, NotImplementedError):
                pass
        self._ell_patches.clear()
        ax = self._ax_gg
        cols = ("#d29922", "#f85149", "#bc8cff")
        lim = 2.4
        for k, (v, ay_g, ax_g) in enumerate(res):
            if not (np.isfinite(ay_g) and np.isfinite(ax_g)):
                continue
            e = Ellipse((0, 0), 2 * ay_g, 2 * ax_g, fill=False, ec=cols[k % 3], ls="--", lw=1.2, zorder=3)
            ax.add_patch(e)
            tx = ax.text(0, ax_g, f" {v * 3.6:.0f} km/h", fontsize=6.5, color=cols[k % 3], va="bottom", ha="center")
            self._ell_patches += [e, tx]
            lim = max(lim, 1.15 * max(ay_g, ax_g))
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        self._cv_gg.draw_idle()

    # ================================================================== segment / panels
    def _set_segment(self, lap: Optional[Lap], reset_view: bool = True) -> None:
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
            for a in self._all_axes:
                a.set_autoscaley_on(True)
        finally:
            self._busy = False
        try:
            self._nav.update()
        except Exception:  # noqa: BLE001
            pass

    def _draw_all(self) -> None:
        for cv in (self._cv_st, self._cv_map, self._cv_dt, self._cv_gg, self._cv_h):
            cv.draw_idle()

    def _decimate(self, y: np.ndarray) -> np.ndarray:
        return minmax_indices(y, MAX_PTS)

    def _refresh_strips(self) -> None:
        xs = self._xs
        i0, i1 = self._seg
        if xs.size < 2:
            for _, ln, _ in self._lines:
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
            nt = self._t.size
            for key, ln, _ in self._lines:
                y = self._ch.get(key)
                if y is None or y.size != nt:
                    ln.set_data([], [])
                    continue
                yv = y[i0 + a:i0 + b]
                if xv.size > MAX_PTS:
                    sel = self._decimate(yv)
                    ln.set_data(xv[sel], yv[sel])
                else:
                    ln.set_data(xv, yv)
            p = self._ch.get("battery_power_kw")
            if p is not None and p.size == nt:
                pv = p[i0 + a:i0 + b]
                sel = self._decimate(pv) if xv.size > MAX_PTS else np.arange(xv.size)
                ps = pv[sel]
                self._ln_over.set_data(xv[sel], np.where(ps > POWER_LIMIT_KW, ps, np.nan))
            else:
                self._ln_over.set_data([], [])
            for ax in self._all_axes:
                ax.relim()
                ax.autoscale_view(scalex=False, scaley=True)
            axr0 = self._axr[0]
            if axr0.get_autoscaley_on():
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
        ok = x.size >= 3 and np.isfinite(x).any() and np.isfinite(y).any()
        if not ok:
            self._map_under.set_data([], [])
            self._map_lc.set_segments([])
            self._map_txt.set_text("no position data (GPS / vx+yaw_rate required)")
            self._blip.set_data([], [])
            self._blip_glow.set_data([], [])
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
        self._map_under.set_data(px, py)
        self._map_lc.set_segments(segs)
        self._map_lc.set_array(vals)
        self._map_lc.set_cmap(self.v_cmap.get())
        self._map_lc.set_clim(lo, hi)
        sp = get_spec(key)
        self._map_txt.set_text(f"{sp.display_name}: {lo:.1f} → {hi:.1f} {sp.units}   ·   {src}")
        xmn, xmx, ymn, ymx = float(np.nanmin(px)), float(np.nanmax(px)), float(np.nanmin(py)), float(np.nanmax(py))
        span = max(xmx - xmn, ymx - ymn, 1.0)
        pad = 0.08 * span
        self._ax_map.set_xlim(xmn - pad, xmx + pad)
        self._ax_map.set_ylim(ymn - pad, ymx + pad)
        self._ax_map.set_aspect("equal", adjustable="datalim")

        def gate(k: int, label: str, color: str) -> None:
            j = min(max(k, 1), n - 2)
            dx, dy = x[j + 1] - x[j - 1], y[j + 1] - y[j - 1]
            nrm = math.hypot(dx, dy) or 1.0
            qx, qy = -dy / nrm, dx / nrm
            ln_ = 0.035 * span
            art, = self._ax_map.plot([x[k] - qx * ln_, x[k] + qx * ln_], [y[k] - qy * ln_, y[k] + qy * ln_],
                                     color=color, lw=2.4, zorder=5)
            tx = self._ax_map.text(x[k] + qx * ln_ * 2.0, y[k] + qy * ln_ * 2.0, label, fontsize=7, color=color,
                                   ha="center", va="center", fontweight="bold", zorder=6)
            self._map_decor += [art, tx]

        gate(0, "S/F", ACCENT_GREEN)
        lap = self._lap_sel
        if lap is not None:
            ts = lap.t_start + np.cumsum(lap.sectors)[:2]
            ks = np.clip(np.searchsorted(self._t[i0:i1], ts), 1, n - 2)
            for q, k in enumerate(ks):
                gate(int(k), f"S{q + 1}|S{q + 2}", ACCENT_AMBER)

    def _update_delta(self) -> None:
        ax = self._ax_dt
        for c in list(ax.collections):
            c.remove()
        lap, ref = self._lap_sel, self._ref
        if lap is None or ref is None or self._an is None:
            self._dt_line.set_data([], [])
            self._dt_msg.set_visible(True)
            self._dt_title.set_text("")
            return
        d, dt = self._an.delta_time(lap, ref)
        self._dt_msg.set_visible(False)
        sel = self._decimate(dt) if d.size > MAX_PTS else np.arange(d.size)
        d, dt = d[sel], dt[sel]
        self._dt_line.set_data(d, dt)
        ax.fill_between(d, 0, dt, where=dt <= 0, color=ACCENT_GREEN, alpha=0.55, interpolate=True, lw=0)
        ax.fill_between(d, 0, dt, where=dt > 0, color=ACCENT_RED, alpha=0.55, interpolate=True, lw=0)
        ax.set_xlim(float(d[0]), float(d[-1]) if d[-1] > d[0] else float(d[0]) + 1.0)
        m = max(float(np.nanmax(np.abs(dt))), 0.05) * 1.15
        ax.set_ylim(-m, m)
        self._dt_title.set_text(f"Lap {lap.index} vs Lap {ref.index}   Δ end = {dt[-1]:+.3f} s")

    def _update_gg(self) -> None:
        i0, i1 = self._seg
        ay, ax_, v = self._ch.get("ay"), self._ch.get("ax"), self._ch.get("vx")
        if ay is None or ax_ is None or ay.size != self._t.size or i1 - i0 < 2:
            self._gg_sc.set_offsets(np.zeros((0, 2)))
            return
        s = max(1, (i1 - i0) // 4000)
        a, b = ay[i0:i1:s], ax_[i0:i1:s]
        c = v[i0:i1:s] if v is not None and v.size == self._t.size else np.zeros_like(a)
        m = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
        self._gg_sc.set_offsets(np.column_stack([a[m], b[m]]))
        self._gg_sc.set_array(c[m])
        vmax = _nanfn(np.max, c[m]) if m.any() else 100.0
        self._gg_norm.vmin, self._gg_norm.vmax = 0.0, max(vmax, 1.0)

    def _update_hist(self) -> None:
        i0, i1 = self._seg
        cnt: dict[str, np.ndarray] = {}
        for c in CORNERS:
            v = self._ch.get(f"damper_velocity_{c}")
            z = self._ch.get(f"damper_travel_{c}")
            if v is None and z is not None and i1 - i0 > 8:
                v = damper_velocity(self._t[i0:i1], z[i0:i1])
                v = np.concatenate([np.zeros(i0), v, np.zeros(max(self._t.size - i1, 0))])
            if v is None or v.size != self._t.size:
                cnt[c] = np.zeros(4)
                continue
            vv = v[i0:i1]
            vv = vv[np.isfinite(vv)]
            cnt[c] = np.array([(vv < -50).sum(), ((vv < 0) & (vv >= -50)).sum(),
                               ((vv >= 0) & (vv <= 50)).sum(), (vv > 50).sum()], dtype=float)
        for c in CORNERS:
            tot = max(cnt[c].sum(), 1.0)
            pct = 100.0 * cnt[c] / tot
            top = max(float(pct.max()) * 1.3, 10.0)
            self._hist_ax[c].set_ylim(0, top)
            for k, (bar, tx) in enumerate(zip(self._hist_bars[c], self._hist_txt[c])):
                bar.set_height(float(pct[k]))
                tx.set_position((k, float(pct[k])))
                tx.set_text(f"{pct[k]:.0f}")

        def row(name: str, a: str, b: str) -> str:
            tot = cnt[a] + cnt[b]
            p = 100.0 * tot / max(tot.sum(), 1.0)
            return f"{name}  " + " | ".join(f"{lab} {p[k]:4.1f}" for k, lab in enumerate(HIST_LABELS))

        self._lbl_fr.configure(text=row("FRONT", "fl", "fr"))
        self._lbl_rr.configure(text=row("REAR ", "rl", "rr"))

    def _update_diag_extremes(self) -> None:
        i0, i1 = self._seg
        for d in self._diag:
            if "key" in d:
                y = self._ch.get(d["key"])
                seg = y[i0:i1] if y is not None and y.size == self._t.size else np.zeros(0)
                if d["key"] in ("min_cell_voltage", "soc"):
                    d["ext"] = _nanfn(np.min, seg) if seg.size else float("nan")
                    d["ext_lbl"] = "seg min"
                else:
                    d["ext"] = _nanfn(lambda a: a[np.argmax(np.abs(a))], seg) if seg.size and np.isfinite(seg).any() \
                        else float("nan")
                    d["ext_lbl"] = "seg peak"
            else:
                vals = [_nanfn(np.max, self._ch[f"{d['pre']}{c}"][i0:i1]) for c in CORNERS
                        if f"{d['pre']}{c}" in self._ch and self._ch[f"{d['pre']}{c}"].size == self._t.size]
                vals = [v for v in vals if np.isfinite(v)]
                d["ext"] = max(vals) if vals else float("nan")
                d["ext_lbl"] = "seg max"

    def _update_diag(self, i: int) -> None:
        for d in self._diag:
            if "key" in d:
                v = self._val(d["key"], i)
            else:
                vs = [self._val(f"{d['pre']}{c}", i) for c in CORNERS]
                vs = [x for x in vs if np.isfinite(x)]
                v = max(vs) if vs else float("nan")
            col = TEXT_BRIGHT
            if np.isfinite(v) and d["warn"] is not None:
                amber, red, mode = d["warn"]
                bad = (lambda x, th: x < th) if mode == "lo" else (lambda x, th: x > th)
                col = ACCENT_RED if bad(v, red) else ACCENT_AMBER if bad(v, amber) else ACCENT_GREEN
            d["val"].configure(text=("--" if not np.isfinite(v) else d["fmt"].format(v)) + f" {d['unit']}", fg=col)
            e = d.get("ext", float("nan"))
            d["sub"].configure(text=f"{d.get('ext_lbl', '')} " + ("--" if not np.isfinite(e) else d["fmt"].format(e)))

    # ================================================================== cursor
    def _update_cursor(self, blit: bool = True) -> None:
        n = self._t.size
        i0, i1 = self._seg
        if n == 0 or self._xs.size == 0:
            for ln in self._cursors:
                ln.set_visible(False)
            if blit:
                self._bl_st.update()
            return
        i = int(np.clip(self._ci, i0, i1 - 1))
        self._ci = i
        x = float(self._xs[i - i0])
        for ln in self._cursors:
            ln.set_visible(True)
            ln.set_xdata([x, x])
        for key, short, txt in self._readouts:
            txt.set_text(f"{short} {_fmt(key, self._val(key, i))}")
        lo, hi = self._axl[0].get_xlim()
        frac = (x - lo) / (hi - lo) if hi > lo else 0.0
        self._time_txt.set_x(x)
        self._time_txt.set_ha("right" if frac > 0.8 else "left")
        t_rel = float(self._t[i] - self._t[i0]) if self._mode == "offline" else float(self._t[i])
        d_rel = float(self._dist[i] - self._dist[i0]) if self._dist.size == n else float("nan")
        lbl = f"t={t_rel:8.3f} s" + (f"  d={d_rel:7.1f} m" if np.isfinite(d_rel) else "")
        self._time_txt.set_text(lbl)
        self.v_scrub.set(lbl + (f"   lap {self._lap_sel.index}" if self._lap_sel else ""))
        self.v_s_cur.set(f"cursor: {lbl.strip()}")
        span = max(i1 - 1 - i0, 1)
        self._scrub_lock = True
        self._scrub.set(1000.0 * (i - i0) / span)
        self._scrub_lock = False
        # map blip
        k = i - i0
        if self._map_x.size > k and np.isfinite(self._map_x[k]):
            self._blip.set_data([self._map_x[k]], [self._map_y[k]])
            self._blip_glow.set_data([self._map_x[k]], [self._map_y[k]])
        else:
            self._blip.set_data([], [])
            self._blip_glow.set_data([], [])
        # delta cursor
        if self._lap_sel is not None and self._dist.size == n:
            self._dt_cur.set_visible(True)
            dx = float(self._dist[i] - self._dist[i0])
            self._dt_cur.set_xdata([dx, dx])
        else:
            self._dt_cur.set_visible(False)
        # G-G
        ay, ax_ = self._val("ay", i), self._val("ax", i)
        self._blink += 1
        if np.isfinite(ay) and np.isfinite(ax_):
            self._gg_pt.set_data([ay], [ax_])
            self._gg_pt.set_markersize(10 if (self._blink // 6) % 2 else 6)
            a = max(i0, i - 60)
            self._gg_trail.set_data(self._ch["ay"][a:i + 1], self._ch["ax"][a:i + 1])
        else:
            self._gg_pt.set_data([], [])
            self._gg_trail.set_data([], [])
        self._update_diag(i)
        if blit:
            for bl in (self._bl_st, self._bl_map, self._bl_dt, self._bl_gg):
                bl.update()

    def _set_cursor_from_x(self, xdata: float) -> None:
        xs = self._xs
        if xs.size < 2:
            return
        k = int(np.clip(np.searchsorted(xs, xdata), 0, xs.size - 1))
        if k > 0 and abs(xs[k - 1] - xdata) < abs(xs[k] - xdata):
            k -= 1
        self._ci = self._seg[0] + k
        self._update_cursor()

    # ================================================================== mouse / scrub
    def _tool_active(self) -> bool:
        m = getattr(self._nav, "mode", "")
        return bool(getattr(m, "value", m))

    def _on_motion(self, ev: Any) -> None:
        if ev.inaxes not in self._all_axes or ev.xdata is None or self._tool_active():
            return
        if self._playing or (self._mode == "live" and not self._live_frozen):
            return
        self._set_cursor_from_x(float(ev.xdata))

    def _on_press(self, ev: Any) -> None:
        if ev.inaxes not in self._all_axes or ev.xdata is None or self._tool_active():
            return
        if ev.dblclick:
            self._reset_view()
            self._refresh_strips()
            self._cv_st.draw_idle()
            return
        if ev.button == 1:
            self._pause()
            self._dragging = True
            self._set_cursor_from_x(float(ev.xdata))

    def _on_release(self, _ev: Any) -> None:
        self._dragging = False

    def _on_scroll(self, ev: Any) -> None:
        if ev.inaxes not in self._all_axes or ev.xdata is None or self._xs.size < 2:
            return
        if self._mode == "live" and not self._live_frozen:
            return
        lo, hi = self._axl[0].get_xlim()
        f = 0.8 if ev.button == "up" else 1.25
        x = float(ev.xdata)
        nlo = max(x - (x - lo) * f, float(self._xs[0]))
        nhi = min(x + (hi - x) * f, float(self._xs[-1]))
        if nhi - nlo < 0.05 * max(float(self._xs[-1] - self._xs[0]), 1e-9) * 0.1:
            return
        self._axl[0].set_xlim(nlo, nhi)

    def _on_scrub(self, val: str) -> None:
        if self._scrub_lock or self._xs.size < 2:
            return
        self._pause()
        i0, i1 = self._seg
        self._ci = i0 + int(round(float(val) / 1000.0 * (i1 - 1 - i0)))
        self._update_cursor()

    def _on_xlim(self, _ax: Any) -> None:
        if self._busy:
            return
        if self._deb_id is not None:
            try:
                self.after_cancel(self._deb_id)
            except tk.TclError:
                pass
        self._deb_id = self.after(30, self._xlim_refresh)

    def _xlim_refresh(self) -> None:
        self._deb_id = None
        self._refresh_strips()
        self._cv_st.draw_idle()

    # ================================================================== transport
    def _pause(self) -> None:
        if self._playing:
            self._playing = False
            self.v_play.set("▶")

    def _toggle_play(self) -> None:
        if self._mode == "live":
            self._live_frozen = not self._live_frozen
            self.v_play.set("▶" if self._live_frozen else "⏸")
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
        self.v_play.set("⏸")

    def _jump(self, end: bool) -> None:
        if self._t.size == 0:
            return
        self._pause()
        i0, i1 = self._seg
        self._ci = (i1 - 1) if end else i0
        self._update_cursor()

    def _step(self, dt: float) -> None:
        if self._t.size == 0 or self._mode == "live" and not self._live_frozen:
            return
        self._pause()
        i0, i1 = self._seg
        tgt = float(self._t[self._ci]) + dt
        self._ci = int(np.clip(np.searchsorted(self._t, tgt), i0, i1 - 1))
        self._update_cursor()

    # ================================================================== toolbar handlers
    def _on_lap_select(self, _e: Any = None) -> None:
        k = self._cb_lap.current()
        lap = None if k <= 0 else self._laps[k - 1]
        self._pause()
        self._set_segment(lap)

    def _toggle_ref(self) -> None:
        if self.v_isref.get():
            self._ref = self._lap_sel or self._best
            if self._ref is None:
                self.v_isref.set(False)
        else:
            self._ref = None
        self.v_ref_lbl.set(f"Ref: Lap {self._ref.index}" if self._ref else "Ref: —")
        self._update_delta()
        self._cv_dt.draw_idle()

    def _on_xmode(self) -> None:
        if self._mode == "live":
            self.v_xmode.set("time")
            return
        self._rebuild_x()
        self._reset_view()
        self._refresh_strips()
        self._update_cursor(blit=False)
        self._cv_st.draw_idle()

    def _on_heat(self) -> None:
        self._update_map()
        self._update_cursor(blit=False)
        self._cv_map.draw_idle()

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
            self.v_play.set("⏸")
            self._cb_lap.configure(state="disabled")
            self._clear_data()
        else:
            self._grp_live.pack_forget()
            self._grp_off.pack(side="left")
            self._mode = "offline"
            self._cb_lap.configure(state="readonly")
            self.v_play.set("▶")
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
        self.v_conn.set("Disconnect")

    def _replay_demo(self) -> None:
        if self._ingest is not None:
            self._disconnect_live()
        self.v_hud.set("building demo…")
        self._run_bg(self._demo_worker)

    def _start_replay(self, log: LogData) -> None:
        self._buffer = RingBuffer(RAW_CHANNELS, capacity=LIVE_CAPACITY)
        ing = ReplayIngest(self._buffer, log.t, log.channels)
        ing.start()
        self._ingest = ing
        self._math_cache = None
        self.v_conn.set("Disconnect")

    def _disconnect_live(self) -> None:
        ing, self._ingest = self._ingest, None
        if ing is not None:
            ing.stop()
            ing.join(timeout=1.0)
        try:
            self.v_conn.set("Connect")
        except tk.TclError:
            pass

    def _update_hud(self) -> None:
        ing = self._ingest
        if ing is None:
            self.v_hud.set("— Hz · 0 frames · 0 dropped")
            self.v_fill.set(0.0)
            self.v_s_rate.set("packets: — Hz")
            self.v_s_fill.set("buffer: —")
        else:
            s = ing.stats()
            self.v_hud.set(f"{s['hz']:5.0f} Hz · {s['frames']} frames · {s['dropped']} dropped"
                           + (" · ERR" if s.get("error") else ""))
            self.v_fill.set(100.0 * s["fill"])
            self.v_s_rate.set(f"packets: {s['hz']:.0f} Hz" + ("" if s["connected"] else " (down)"))
            self.v_s_fill.set(f"buffer: {100.0 * s['fill']:.1f} %")
            if s.get("error"):
                self._post_status(f"Live ingest error: {s['error']}")
        self.v_s_log.set(f"Log: {self._log_name if self._mode == 'offline' else 'LIVE'}")
        self.v_s_fs.set(f"fs: {self._fs:.0f} Hz")

    # ================================================================== poll loop
    def _poll_tick(self) -> None:
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
            if self._tick_n % 15 == 0:
                self._update_hud()
                self._app_state.set("last_draw_ms", self._last_ms)
        except Exception:  # noqa: BLE001  # the loop must survive any rendering error
            LOG.exception("telemetry poll tick failed")
        self._last_ms = 1e3 * (time.perf_counter() - t0)
        if self._active and self._poll_id is None:
            self._poll_id = self.after(POLL_MS, self._poll_tick)

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
                self._axl[0].set_xlim(float(t[-1]) - win, float(t[-1]))
            finally:
                self._busy = False
            self._refresh_strips()
            self._cv_st.draw_idle()
        if self._live_n % 15 == 0:
            self._update_map()
            self._update_gg()
            self._update_hist()
            self._update_diag_extremes()
            self._cv_map.draw_idle()
            self._cv_gg.draw_idle()
            self._cv_h.draw_idle()
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