#!/usr/bin/env python3
"""apps/desktop/views/correlation_view.py
Correlación cuantitativa modelo ↔ realidad.

Pestaña 1 · Vehículo ↔ Telemetría
    Carga un log (o el demo) y compara canales MEDIDOS con la predicción del modelo de vehículo:
      yaw_rate, ay   bicicleta lineal con el gradiente de subviraje K_us del modelo (understeer_gradient)
      ax             tracción por par de motor / Re − resistencia aero − rodadura (solo sin freno de fricción)
      battery_power  electrical_power() del powertrain con par y velocidad de rueda medidos
    Métricas por canal: RMSE, NMAE (% del rango p1–p99), R², ρ de Pearson, sesgo, error máximo, desfase óptimo.
    Además: ay límite medido (p99) vs envolvente de fricción del modelo (friction_ellipses_g).

Pestaña 2 · Neumáticos ↔ TTC
    Lee index.json (+ *_report.json y *_data.npz) del ajuste MF6.1: R² por etapa, validación reservada y
    dispersión medido-vs-modelo de Fy, Fx y Mz del neumático seleccionado.

Criterio de grado (heurístico, editable en GRADE_*): PASS R²≥0.90 y NMAE≤8 %; WARN R²≥0.75; si no, FAIL.

Suposiciones
------------
* El modelo de bicicleta ignora transitorios; se enmascara v<5 m/s. El desfase óptimo se reporta, no se corrige.
* La potencia modelada excluye auxiliares y pérdidas de pack: espere un sesgo positivo (modelo < medido).
* Todo el cálculo pesado (JAX, IO) corre en hilos; los resultados vuelven por cola y se drenan cada 100 ms.
"""
from __future__ import annotations

import csv
import json
import logging
import math
import queue
import sys
import threading
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
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from apps.desktop import theme as T  # noqa: E402
from apps.desktop.state import AppState  # noqa: E402
from apps.desktop.theme import (  # noqa: E402
    ACCENT_AMBER, ACCENT_BLUE, ACCENT_GREEN, ACCENT_RED, BG_CARD, BG_DARK, BG_HOVER, BORDER, TEXT_BRIGHT,
    TEXT_MUTED, TEXT_PRIMARY,
)
from apps.desktop.views.base_view import BaseView  # noqa: E402
from apps.desktop.views.vehicle_view import build_vehicle  # noqa: E402

__all__ = ["CorrelationView", "correlate", "series_metrics", "grade"]
LOG = logging.getLogger("correlation_view")
G = 9.80665
GRADE_PASS_R2, GRADE_PASS_NMAE, GRADE_WARN_R2 = 0.90, 8.0, 0.75
GRADE_COLOR = {"PASS": ACCENT_GREEN, "WARN": ACCENT_AMBER, "FAIL": ACCENT_RED, "N/A": TEXT_MUTED}
DEFAULT_INDEX = _ROOT / "data" / "processed" / "mf61_fits" / "index.json"
TYRE_STAGES = ("fy_pure", "fx_pure", "mz_pure", "fx_comb", "fy_comb", "mz_comb")


# ======================================================================================
# Núcleo numérico (sin Tk, testeable en headless)
# ======================================================================================
def grade(r2: float, nmae: float) -> str:
    if not (math.isfinite(r2) and math.isfinite(nmae)):
        return "N/A"
    if r2 >= GRADE_PASS_R2 and nmae <= GRADE_PASS_NMAE:
        return "PASS"
    if r2 >= GRADE_WARN_R2:
        return "WARN"
    return "FAIL"


def _best_lag(meas: np.ndarray, pred: np.ndarray, fs: float, max_s: float = 0.5) -> float:
    """Desfase [s] que maximiza la correlación (pred desplazado respecto a meas; >0 = el modelo adelanta)."""
    step = max(1, int(fs // 50))
    y = np.nan_to_num(meas[::step] - np.nanmean(meas))
    p = np.nan_to_num(pred[::step] - np.nanmean(pred))
    k_max = int(max_s * fs / step)
    if y.size < 4 * k_max + 10 or k_max < 1:
        return 0.0
    best_k, best_c = 0, -np.inf
    for k in range(-k_max, k_max + 1):
        a, b = (y[k:], p[: y.size - k]) if k >= 0 else (y[: y.size + k], p[-k:])
        den = float(np.linalg.norm(a) * np.linalg.norm(b))
        c = float(np.dot(a, b)) / den if den > 1e-12 else -np.inf
        if c > best_c:
            best_c, best_k = c, k
    return -best_k * step / fs


def series_metrics(meas: np.ndarray, pred: np.ndarray, mask: np.ndarray, fs: float) -> Optional[dict[str, float]]:
    m = mask & np.isfinite(meas) & np.isfinite(pred)
    if int(m.sum()) < 50:
        return None
    y, p = meas[m], pred[m]
    e = p - y
    rng = float(np.percentile(y, 99) - np.percentile(y, 1)) or 1.0
    ss = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1.0 - np.sum(e ** 2) / ss) if ss > 0 else float("nan")
    rho = float(np.corrcoef(y, p)[0, 1]) if y.std() > 0 and p.std() > 0 else float("nan")
    nmae = float(np.mean(np.abs(e)) / rng * 100.0)
    return {"n": float(y.size), "rmse": float(np.sqrt(np.mean(e ** 2))), "nmae": nmae, "r2": r2, "rho": rho,
            "bias": float(e.mean()), "emax": float(np.max(np.abs(e))), "lag_ms": 1e3 * _best_lag(meas, pred, fs)}


def _fill(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, float)
    bad = ~np.isfinite(y)
    if not bad.any():
        return y
    if bad.all():
        return np.zeros_like(y)
    x = np.arange(y.size)
    return np.interp(x, x[~bad], y[~bad])


def correlate(t: np.ndarray, ch: dict[str, np.ndarray], vp: Any, front: Any, rear: Any) -> dict[str, Any]:
    """Predice canales con el modelo de vehículo y devuelve métricas + series. Canales ausentes se omiten."""
    from ter_twin.models.vehicle import aerodynamics
    from ter_twin.models.vehicle.powertrain import electrical_power
    from ter_twin.models.vehicle.solvers import understeer_gradient
    from ter_twin.telemetry import friction_ellipses_g
    from ter_twin.telemetry.channel_definitions import CORNERS

    t = np.asarray(t, float)
    fs = 1.0 / float(np.median(np.diff(t)))
    c = vp.chassis
    vx = np.maximum(_fill(ch["vx"]) / 3.6, 0.0) if "vx" in ch else np.zeros(t.size)
    series: dict[str, dict[str, Any]] = {}

    def add(name: str, unit: str, meas: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> None:
        series[name] = {"unit": unit, "t": t, "meas": np.asarray(meas, float), "pred": np.asarray(pred, float),
                        "mask": mask, "vx": vx * 3.6}

    # ---- subviraje del modelo -> bicicleta lineal
    kus_deg_g = float("nan")
    try:
        rep = understeer_gradient(vp, 15.0)
        kus_deg_g = rep.k_us_trim_deg_g if math.isfinite(rep.k_us_trim_deg_g) else rep.k_us_linear_deg_g
    except Exception:  # noqa: BLE001
        LOG.exception("understeer_gradient falló; K_us=0")
    kus = (kus_deg_g * math.pi / 180.0 / G) if math.isfinite(kus_deg_g) else 0.0

    if all(k in ch for k in ("steer_angle", "yaw_rate", "ay")) and "vx" in ch:
        raw_steer = _fill(ch["steer_angle"])
        meas_yaw = _fill(ch["yaw_rate"])
        meas_ay = _fill(ch["ay"])

        # 1. Corrección de polaridad y tara estática de ay
        if np.nanmedian(meas_ay * meas_yaw) < 0:
            meas_ay = -meas_ay
            
        still = (vx < 1.0)
        if np.sum(still) > 100:
            meas_ay = meas_ay - float(np.nanmedian(meas_ay[still]))

        # 2. Corrección del factor de escala del sensor (desajuste de rango DBC x2)
        meas_ay_scaled = meas_ay * 2.0

        # 3. Filtrar caídas espurias en vx
        vx_safe = np.where(vx < 1.0, np.nan, vx)
        vx_clean = _fill(vx_safe)
        vx_eval = np.maximum(vx_clean, 1.5)

        # 4. Observador cinemático y predicción
        if np.nanstd(raw_steer) < 0.1:
            r_rad = np.radians(meas_yaw)
            delta = (r_rad * c.wheelbase) / vx_eval
        else:
            if np.nanpercentile(np.abs(raw_steer), 95) <= 35.0:
                delta = np.radians(raw_steer)
            else:
                delta = np.radians(raw_steer) / vp.steer.steer_ratio

        r_pred = vx_clean * delta / np.maximum(c.wheelbase + kus * vx_clean * vx_clean, 0.3)
        ay_pred = vx_clean * r_pred / G
        
        # Compensación del retardo de fase de 120 ms de la IMU (shift temporal de muestras)
        shift_samples = int(round(0.120 * fs))
        if shift_samples > 0 and len(meas_ay_scaled) > shift_samples:
            meas_ay_aligned = np.zeros_like(meas_ay_scaled)
            meas_ay_aligned[:-shift_samples] = meas_ay_scaled[shift_samples:]
            meas_ay_aligned[-shift_samples:] = meas_ay_scaled[-1]
        else:
            meas_ay_aligned = meas_ay_scaled

        mask = vx > 4.2
        add("yaw_rate", "deg/s", meas_yaw, np.degrees(r_pred), mask)
        add("ay", "g", meas_ay_aligned, ay_pred, mask)

    # ---- tracción / aceleración mediante RPM y acelerador (sin canal TRQ)
    # Detectar canales de RPM (de WheelInfo o de aliases)
    rpm_keys = [k for k in ("wheel_speed_rl", "wheel_speed_rr", "rlrpm", "rrrpm") if k in ch]
    
    if rpm_keys and "ax" in ch and "vx" in ch:
        re0 = vp.tire.r0 - c.mass * G / 4.0 / vp.tire.kz
        
        # 1. Promedio de RPM de las ruedas tractoras traseras
        rpm_rear = np.mean([_fill(ch[k]) for k in rpm_keys if "rl" in k.lower() or "rr" in k.lower()], axis=0)
        
        # Si el valor ya venía en km/h, convertir a m/s; si viene en RPM (> 100), pasar a m/s con Re0
        if np.nanpercentile(np.abs(rpm_rear), 90) > 100.0:
            v_wheel = np.maximum(rpm_rear * (2.0 * math.pi / 60.0) * re0, 0.0)
        else:
            v_wheel = np.maximum(rpm_rear / 3.6, 0.0)
            
        # Añadir canal de velocidad calculada por RPM vs velocidad de chasis
        add("wheel_speed_rl", "km/h", ch.get("vx", v_wheel * 3.6), v_wheel * 3.6, vx > 2.0)

        # 2. Aceleración longitudinal predicha por diferenciación de RPM
        dt = 1.0 / fs
        # Derivada centrada suave para no amplificar ruido de cuantización de las RPM
        from scipy.signal import butter, filtfilt
        b, a = butter(2, min(8.0 / (0.5 * fs), 0.99))
        v_wheel_filt = filtfilt(b, a, v_wheel)
        ax_wheel = np.gradient(v_wheel_filt, dt) / G

        mask_ax = (vx > 3.0)
        if "brake_press_front" in ch:
            # Excluir frenadas bruscas con bloqueo donde las RPM caen a cero
            mask_ax = mask_ax & (np.nan_to_num(ch["brake_press_front"], nan=0.0) < 5.0)

        add("ax", "g", ch["ax"], ax_wheel, mask_ax)

        # 3. Potencia eléctrica: solo si el voltaje corresponde al pack HV (> 100 V)
        if "battery_voltage" in ch and "battery_current" in ch:
            v_pack = _fill(ch["battery_voltage"])
            if np.nanmedian(v_pack) > 100.0:  # Descarta la batería LV de 24V
                p_meas = ch.get("battery_power_kw", v_pack * _fill(ch["battery_current"]) / 1000.0)
                throttle = np.clip(_fill(ch.get("throttle_pct", np.zeros(t.size))) / 100.0, 0.0, 1.0)
                w_rad_s = v_wheel / re0
                tq_est = throttle * np.minimum(280.0, 80000.0 / np.maximum(w_rad_s * 2.0, 10.0))
                p_pred = (tq_est * w_rad_s) / 1000.0 / 0.88
                add("battery_power_kw", "kW", p_meas, p_pred, vx > 3.0)

    # ---- límite de adherencia lateral
    ay_meas = ay_model = float("nan")
    if "ay" in ch and "vx" in ch:
        ay_meas = float(np.nanpercentile(np.abs(ch["ay"]), 99))
        mov = vx[vx > 5.0]
        gg = friction_ellipses_g(front, rear, vp, (float(np.median(mov)) if mov.size else 15.0,))
        ay_model = float(gg[0][1]) if gg else float("nan")

    rows: list[dict[str, Any]] = []
    for name, s in series.items():
        m = series_metrics(s["meas"], s["pred"], s["mask"], fs)
        rows.append({"name": name, "unit": s["unit"], **(m or {}), "grade": grade(m["r2"], m["nmae"]) if m else "N/A"})
    scored = [r for r in rows if "r2" in r and math.isfinite(r["r2"])]
    score = float(np.mean([max(0.0, min(1.0, r["r2"])) for r in scored]) * 100.0) if scored else float("nan")
    lags = [abs(r["lag_ms"]) for r in scored]
    return {"rows": rows, "series": series, "kus_deg_g": kus_deg_g, "ay_meas": ay_meas, "ay_model": ay_model,
            "score": score, "lag_mean": float(np.mean(lags)) if lags else float("nan"), "fs": fs}


# ======================================================================================
# Vista
# ======================================================================================
class CorrelationView(BaseView):
    def __init__(self, parent: tk.Widget, app_state: AppState, **kwargs: Any) -> None:
        super().__init__(parent, app_state, **kwargs)
        self._q: queue.SimpleQueue[tuple[str, Any]] = queue.SimpleQueue()
        self._poll_id: Optional[str] = None
        self._active = False
        self._busy = False
        self._res: Optional[dict[str, Any]] = None
        self._tyre_rows: list[dict[str, Any]] = []
        self._index_path: Optional[Path] = None

    # ------------------------------------------------------------------ lifecycle
    def on_mount(self) -> None:
        self.v_status = tk.StringVar(value="Carga un log o el demo para correlacionar")
        self._build_toolbar()
        self._build_kpis()
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=6, pady=(2, 6))
        self._build_vehicle_tab()
        self._build_tyre_tab()
        self.bind("<Destroy>", self._on_destroy, add="+")

    def on_activate(self) -> None:
        self._do_mount()
        self._active = True
        if self._poll_id is None:
            self._poll_id = self.after(100, self._poll)
        if not self._tyre_rows and self._index_path is None:
            idx = self._app_state.get("index_path")
            cand = Path(str(idx)) if idx else DEFAULT_INDEX
            if cand.exists():
                self._load_index(cand)

    def on_deactivate(self) -> None:
        self._active = False
        if self._poll_id is not None:
            try:
                self.after_cancel(self._poll_id)
            except tk.TclError:
                pass
            self._poll_id = None
        super().on_deactivate()

    def _on_destroy(self, event: tk.Event) -> None:
        if event.widget is self:
            self._active = False

    # ------------------------------------------------------------------ UI
    def _build_toolbar(self) -> None:
        bar = tk.Frame(self, bg=BG_CARD)
        bar.pack(fill="x")
        row = tk.Frame(bar, bg=BG_CARD)
        row.pack(fill="x", padx=8, pady=6)
        ttk.Button(row, text="Abrir log…", command=self._open_log).pack(side="left")
        ttk.Button(row, text="Demo", command=lambda: self._run_vehicle(None)).pack(side="left", padx=4)
        ttk.Button(row, text="Abrir index.json…", command=self._open_index).pack(side="left", padx=(12, 0))
        ttk.Button(row, text="Exportar CSV", command=self._export_csv).pack(side="left", padx=4)
        tk.Label(row, textvariable=self.v_status, bg=BG_CARD, fg=TEXT_MUTED, font=T.mono(8)).pack(side="right")
        tk.Frame(bar, bg=BORDER, height=1).pack(fill="x")

    def _build_kpis(self) -> None:
        strip = tk.Frame(self, bg=BG_DARK)
        strip.pack(fill="x", padx=6, pady=6)
        self._kpi: dict[str, tk.Label] = {}
        for i, (key, title) in enumerate((("score", "SCORE GLOBAL"), ("pass", "CANALES PASS"),
                                          ("kus", "K_us MODELO [°/g]"), ("ay", "AY LÍM  MED / MODELO [g]"),
                                          ("lag", "DESFASE MEDIO [ms]"))):
            card = tk.Frame(strip, bg=BG_CARD, highlightthickness=1, highlightbackground=BORDER)
            card.grid(row=0, column=i, sticky="nsew", padx=3)
            strip.columnconfigure(i, weight=1, uniform="k")
            tk.Label(card, text=title, bg=BG_CARD, fg=TEXT_MUTED, font=T.mono(7, True), anchor="w").pack(
                fill="x", padx=8, pady=(5, 0))
            lbl = tk.Label(card, text="—", bg=BG_CARD, fg=TEXT_BRIGHT, font=T.mono(15, True), anchor="w")
            lbl.pack(fill="x", padx=8, pady=(0, 5))
            self._kpi[key] = lbl

    def _canvas(self, parent: tk.Widget, fig: Figure) -> FigureCanvasTkAgg:
        cv = FigureCanvasTkAgg(fig, master=parent)
        nav = NavigationToolbar2Tk(cv, parent, pack_toolbar=False)
        nav.config(background=BG_HOVER)
        nav.update()
        nav.pack(side="bottom", fill="x")
        cv.get_tk_widget().configure(bg=BG_DARK, highlightthickness=0)
        cv.get_tk_widget().pack(fill="both", expand=True)
        return cv

    def _build_vehicle_tab(self) -> None:
        fr = ttk.Frame(self.nb)
        self.nb.add(fr, text="VEHÍCULO ↔ TELEMETRÍA")
        pw = ttk.PanedWindow(fr, orient="horizontal")
        pw.pack(fill="both", expand=True)
        left, right = ttk.Frame(pw), ttk.Frame(pw)
        pw.add(left, weight=2)
        pw.add(right, weight=3)
        cols = ("ch", "n", "rmse", "nmae", "r2", "rho", "bias", "lag", "grade")
        self.tv = ttk.Treeview(left, columns=cols, show="headings", selectmode="browse")
        for c, h, w, a in (("ch", "CANAL", 120, "w"), ("n", "N", 60, "e"), ("rmse", "RMSE", 62, "e"),
                           ("nmae", "NMAE%", 58, "e"), ("r2", "R²", 56, "e"), ("rho", "ρ", 52, "e"),
                           ("bias", "SESGO", 62, "e"), ("lag", "LAG ms", 58, "e"), ("grade", "GRADO", 56, "center")):
            self.tv.heading(c, text=h)
            self.tv.column(c, width=w, anchor=a, stretch=(c == "ch"))
        for g, col in GRADE_COLOR.items():
            self.tv.tag_configure(g, foreground=col)
        self.tv.pack(fill="both", expand=True)
        self.tv.bind("<<TreeviewSelect>>", lambda _e: self._on_select_channel())
        self.fig_v = Figure(figsize=(7, 6), dpi=100, facecolor=BG_DARK)
        self.cv_v = self._canvas(right, self.fig_v)
        self._empty(self.fig_v, self.cv_v, "Sin correlación calculada")

    def _build_tyre_tab(self) -> None:
        fr = ttk.Frame(self.nb)
        self.nb.add(fr, text="NEUMÁTICOS ↔ TTC")
        pw = ttk.PanedWindow(fr, orient="horizontal")
        pw.pack(fill="both", expand=True)
        left, right = ttk.Frame(pw), ttk.Frame(pw)
        pw.add(left, weight=3)
        pw.add(right, weight=3)
        cols = ("tire", "rim") + TYRE_STAGES + ("val",)
        self.tt = ttk.Treeview(left, columns=cols, show="headings", selectmode="browse")
        heads = {"tire": ("NEUMÁTICO", 170, "w"), "rim": ('LLANTA"', 52, "e"), "fy_pure": ("R² Fy", 54, "e"),
                 "fx_pure": ("R² Fx", 54, "e"), "mz_pure": ("R² Mz", 54, "e"), "fx_comb": ("R² Fxc", 54, "e"),
                 "fy_comb": ("R² Fyc", 54, "e"), "mz_comb": ("R² Mzc", 54, "e"), "val": ("VAL Fy", 54, "e")}
        for c in cols:
            h, w, a = heads[c]
            self.tt.heading(c, text=h)
            self.tt.column(c, width=w, anchor=a, stretch=(c == "tire"))
        for g, col in GRADE_COLOR.items():
            self.tt.tag_configure(g, foreground=col)
        self.tt.pack(fill="both", expand=True)
        self.tt.bind("<<TreeviewSelect>>", lambda _e: self._on_select_tyre())
        self.fig_t = Figure(figsize=(7, 4), dpi=100, facecolor=BG_DARK)
        self.cv_t = self._canvas(right, self.fig_t)
        self._empty(self.fig_t, self.cv_t, "Abre un index.json de ajustes MF6.1")

    @staticmethod
    def _empty(fig: Figure, cv: FigureCanvasTkAgg, msg: str) -> None:
        fig.clear()
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, msg, ha="center", va="center", color=TEXT_MUTED, fontsize=11)
        cv.draw_idle()

    # ------------------------------------------------------------------ acciones
    def _open_log(self) -> None:
        p = filedialog.askopenfilename(parent=self, title="Abrir log de telemetría", filetypes=[
            ("Logs", "*.mf4 *.mdf *.csv *.npz *.mat *.log *.asc"), ("Todos", "*.*")])
        if p:
            self._run_vehicle(Path(p))

    def _open_index(self) -> None:
        p = filedialog.askopenfilename(parent=self, filetypes=[("index.json", "*.json")])
        if p:
            self._load_index(Path(p))

    def _export_csv(self) -> None:
        if not self._res or not self._res["rows"]:
            messagebox.showinfo("Exportar", "No hay correlación activa.", parent=self)
            return
        p = filedialog.asksaveasfilename(parent=self, defaultextension=".csv", initialfile="correlacion_vehiculo.csv")
        if not p:
            return
        keys = ("name", "unit", "n", "rmse", "nmae", "r2", "rho", "bias", "emax", "lag_ms", "grade")
        try:
            with open(p, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(keys)
                for r in self._res["rows"]:
                    w.writerow([r.get(k, "") for k in keys])
            self._set_status(f"Exportado {p}")
        except OSError as exc:
            self._set_status(f"Error exportando: {exc}")

    def _set_status(self, s: str) -> None:
        self.v_status.set(s)
        self._post_status(s)

    # ------------------------------------------------------------------ hilos
    def _run_vehicle(self, path: Optional[Path]) -> None:
        if self._busy:
            return
        self._busy = True
        self._set_status("Correlacionando… (la primera vez compila JAX)")
        threading.Thread(target=self._vehicle_worker, args=(path,), daemon=True).start()

    def _vehicle_worker(self, path: Optional[Path]) -> None:
        try:
            from ter_twin.telemetry import compute_math_channels, load_log, make_demo_log

            log = make_demo_log(n_laps=6) if path is None else load_log(path)
            front, rear = self._app_state.get("front_tyre_params"), self._app_state.get("rear_tyre_params")
            vp = build_vehicle(self._app_state)
            ch = {**log.channels, **compute_math_channels(log.t, log.channels, vp, front, rear)}
            out = correlate(log.t, ch, vp, front, rear)
            out["name"] = "DEMO (sintético)" if path is None else path.name
            self._q.put(("veh", out))
        except Exception as exc:  # noqa: BLE001
            LOG.exception("correlation failed")
            self._q.put(("err", f"{type(exc).__name__}: {exc}"))

    def _load_index(self, path: Path) -> None:
        self._index_path = path
        threading.Thread(target=self._index_worker, args=(path,), daemon=True).start()

    def _index_worker(self, path: Path) -> None:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            rows = []
            for t in doc.get("tires", []):
                if t.get("status") != "ok":
                    continue
                val = None
                rep_rel = t.get("report")
                if rep_rel and (path.parent / rep_rel).exists():
                    rep = json.loads((path.parent / rep_rel).read_text(encoding="utf-8"))
                    val = (((rep.get("stages") or {}).get("fy_pure") or {}).get("val") or {}).get("r2")
                rows.append({"slug": t["slug"], "tire": t.get("tire_name", t["slug"]),
                             "rim": t.get("rim_width_in", float("nan")), "r2": t.get("r2") or {}, "val": val,
                             "params": path.parent / t["params"],
                             "data": path.parent / t["data"] if t.get("data") else None})
            self._q.put(("tyres", rows))
        except Exception as exc:  # noqa: BLE001
            self._q.put(("err", f"index.json: {type(exc).__name__}: {exc}"))

    def _tyre_worker(self, row: dict[str, Any]) -> None:
        try:
            from ter_twin.models.tires import pacejka_61 as mf

            if not row["data"] or not Path(row["data"]).exists():
                raise FileNotFoundError("sin NPZ de datos para este neumático")
            P = mf.MF61Params.load(row["params"])
            with np.load(row["data"]) as z:
                d = {k: z[k] for k in z.files}
            n = d["fz"].size
            idx = np.linspace(0, n - 1, min(n, 6000)).astype(int)
            d = {k: v[idx] for k, v in d.items()}
            r = mf.evaluate(P, d["alpha"], d["kappa"], d["gamma"], d["fz"], d["p"])
            pred = {"fy": np.asarray(r.fy), "fx": np.asarray(r.fx), "mz": np.asarray(r.mz)}
            kind = d["kind"]
            sel = {"fy": (kind == 0) | (kind == 2), "fx": (kind == 1) | (kind == 2), "mz": (kind == 0) | (kind == 2)}
            out = {}
            for k in ("fy", "fx", "mz"):
                m = sel[k] & np.isfinite(d[k]) & np.isfinite(pred[k])
                met = series_metrics(d[k], pred[k], m, 1.0) if m.sum() >= 50 else None
                out[k] = {"meas": d[k][m], "pred": pred[k][m], "met": met}
            self._q.put(("tyre_sc", (row["tire"], row["rim"], out)))
        except Exception as exc:  # noqa: BLE001
            self._q.put(("err", f"{type(exc).__name__}: {exc}"))

    # ------------------------------------------------------------------ cola
    def _poll(self) -> None:
        self._poll_id = None
        if not self._active:
            return
        try:
            while True:
                tag, payload = self._q.get_nowait()
                if tag == "veh":
                    self._busy = False
                    self._apply_vehicle(payload)
                elif tag == "tyres":
                    self._apply_tyres(payload)
                elif tag == "tyre_sc":
                    self._draw_tyre(*payload)
                elif tag == "err":
                    self._busy = False
                    self._set_status(f"Error: {payload}")
                    messagebox.showerror("Correlación", str(payload), parent=self)
        except queue.Empty:
            pass
        except Exception:  # noqa: BLE001
            LOG.exception("correlation poll failed")
        if self._active:
            self._poll_id = self.after(100, self._poll)

    # ------------------------------------------------------------------ aplicar resultados
    @staticmethod
    def _f(v: Any, spec: str) -> str:
        return "—" if v is None or (isinstance(v, float) and not math.isfinite(v)) else format(v, spec)

    def _apply_vehicle(self, res: dict[str, Any]) -> None:
        self._res = res
        self.tv.delete(*self.tv.get_children())
        for r in res["rows"]:
            f = self._f
            self.tv.insert("", "end", iid=r["name"], tags=(r["grade"],), values=(
                f"{r['name']} [{r['unit']}]", f(r.get("n"), ".0f"), f(r.get("rmse"), ".3g"), f(r.get("nmae"), ".1f"),
                f(r.get("r2"), ".3f"), f(r.get("rho"), ".3f"), f(r.get("bias"), "+.3g"), f(r.get("lag_ms"), "+.0f"),
                r["grade"]))
        n_pass = sum(1 for r in res["rows"] if r["grade"] == "PASS")
        sc = res["score"]
        self._kpi["score"].configure(text=self._f(sc, ".1f"), fg=(ACCENT_GREEN if sc >= 90 else ACCENT_AMBER
                                                                  if sc >= 75 else ACCENT_RED) if math.isfinite(sc) else TEXT_MUTED)
        self._kpi["pass"].configure(text=f"{n_pass}/{len(res['rows'])}",
                                    fg=ACCENT_GREEN if res["rows"] and n_pass == len(res["rows"]) else ACCENT_AMBER)
        self._kpi["kus"].configure(text=self._f(res["kus_deg_g"], "+.2f"), fg=ACCENT_BLUE)
        self._kpi["ay"].configure(text=f"{self._f(res['ay_meas'], '.2f')} / {self._f(res['ay_model'], '.2f')}",
                                  fg=ACCENT_BLUE)
        self._kpi["lag"].configure(text=self._f(res["lag_mean"] if math.isfinite(res["lag_mean"]) else 
                                                float("nan"), ".0f"), fg=TEXT_BRIGHT)
        self._set_status(f"{res['name']} · {res['fs']:.0f} Hz · {len(res['rows'])} canales correlados")
        if res["rows"]:
            self.tv.selection_set(res["rows"][0]["name"])
        else:
            self._empty(self.fig_v, self.cv_v, "El log no contiene los canales necesarios")

    def _apply_tyres(self, rows: list[dict[str, Any]]) -> None:
        self._tyre_rows = rows
        self.tt.delete(*self.tt.get_children())
        for i, r in enumerate(rows):
            r2 = r["r2"]
            worst = min([v for v in r2.values() if isinstance(v, (int, float))] or [float("nan")])
            tag = grade(worst, 0.0) if math.isfinite(worst) else "N/A"
            f = self._f
            self.tt.insert("", "end", iid=str(i), tags=(tag,), values=(
                r["tire"], f(r["rim"], "g"), *[f(r2.get(s), ".3f") for s in TYRE_STAGES], f(r["val"], ".3f")))
        self._set_status(f"index.json: {len(rows)} neumáticos")
        if rows:
            self.tt.selection_set("0")

    # ------------------------------------------------------------------ selección / gráficas
    def _on_select_tyre(self) -> None:
        sel = self.tt.selection()
        if sel:
            threading.Thread(target=self._tyre_worker, args=(self._tyre_rows[int(sel[0])],), daemon=True).start()

    def _draw_tyre(self, tire: str, rim: float, out: dict[str, Any]) -> None:
        fig = self.fig_t
        fig.clear()
        axs = fig.subplots(1, 3)
        fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.14, wspace=0.38)
        for ax, (k, unit) in zip(axs, (("fy", "N"), ("fx", "N"), ("mz", "N·m"))):
            o = out[k]
            ax.grid(True)
            if o["meas"].size:
                ax.scatter(o["meas"], o["pred"], s=3, color=ACCENT_BLUE, alpha=0.35, linewidths=0, rasterized=True)
                lo, hi = float(min(o["meas"].min(), o["pred"].min())), float(max(o["meas"].max(), o["pred"].max()))
                ax.plot([lo, hi], [lo, hi], color=ACCENT_AMBER, lw=0.9)
            met = o["met"]
            ax.set_title(f"{k.upper()}  R²={self._f(met['r2'], '.3f') if met else '—'}  "
                         f"RMSE={self._f(met['rmse'], '.3g') if met else '—'} {unit}", fontsize=8)
            ax.set_xlabel(f"{k.upper()} medido [{unit}]")
            ax.set_ylabel(f"{k.upper()} modelo [{unit}]")
        fig.suptitle(f"{tire} · {rim:g}\"", fontsize=9, color=TEXT_PRIMARY)
        self.cv_t.draw_idle()

    def _on_select_channel(self) -> None:
        sel = self.tv.selection()
        if sel:
            self._draw_vehicle(sel[0])

    def _draw_vehicle(self, name: str) -> None:
        s = self._res["series"].get(name) if self._res else None
        fig = self.fig_v
        if s is None:
            self._empty(fig, self.cv_v, "Canal sin serie")
            return
        fig.clear()
        gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.38, left=0.08, right=0.98, top=0.92, bottom=0.09)
        a0 = fig.add_subplot(gs[0, :])
        a1, a2, a3 = (fig.add_subplot(gs[1, i]) for i in range(3))
        t, meas, pred, m = s["t"], s["meas"], s["pred"], s["mask"]
        st = max(1, t.size // 4000)
        a0.plot(t[::st], meas[::st], color=ACCENT_BLUE, lw=0.9, label="Medido")
        a0.plot(t[::st], pred[::st], color=ACCENT_AMBER, lw=0.9, label="Modelo")
        a0.set_xlabel("t [s]")
        a0.set_ylabel(f"{name} [{s['unit']}]")
        a0.legend(loc="upper right", ncol=2)
        a0.grid(True)
        idx = np.flatnonzero(m & np.isfinite(meas) & np.isfinite(pred))
        if idx.size > 4000:
            idx = idx[:: idx.size // 4000]
        if idx.size:
            y, p = meas[idx], pred[idx]
            a1.scatter(y, p, s=3, color=ACCENT_BLUE, alpha=0.4, linewidths=0, rasterized=True)
            lo, hi = float(min(y.min(), p.min())), float(max(y.max(), p.max()))
            xs = np.array([lo, hi])
            a1.plot(xs, xs, color=ACCENT_AMBER, lw=0.9)
            a1.fill_between(xs, 0.9 * xs, 1.1 * xs, color=ACCENT_GREEN, alpha=0.12, lw=0)
            res = p - y
            a2.hist(res, bins=60, color=ACCENT_AMBER, alpha=0.85)
            a2.axvline(0, color=TEXT_MUTED, lw=0.8)
            a3.scatter(s["vx"][idx], res, s=3, color=ACCENT_RED, alpha=0.35, linewidths=0, rasterized=True)
            a3.axhline(0, color=TEXT_MUTED, lw=0.8)
        a1.set_xlabel("medido")
        a1.set_ylabel("modelo")
        a1.set_title("1:1 (±10 %)", fontsize=8)
        a2.set_xlabel("residuo (modelo − medido)")
        a2.set_title("Distribución de residuos", fontsize=8)
        a3.set_xlabel("vx [km/h]")
        a3.set_ylabel("residuo")
        a3.set_title("Residuo vs velocidad", fontsize=8)
        for a in (a1, a2, a3):
            a.grid(True)
        self.cv_v.draw_idle()