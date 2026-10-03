#!/usr/bin/env python3
"""Genera .dat sinteticos con formato TTC (SI, FZ<0, signos SAE) a partir de MF6.1 con ruido.

Sirve para validar parser, clasificador, ajuste y terminal sin datos reales de Calspan.
Verdad-terreno: 7" y 8" difieren en pKy1 (+8%), pDy1 (+2%) y qDz1 (-6%).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
from ter_twin.models.tires import pacejka_61 as mf  # noqa: E402

TIRE = "Hoosier 16.0x7.5-10 R20"
COLS = "ET V SA IA SL RL RE P FX FY FZ MX MZ".split()
UNITS = "s km/h deg deg - cm cm kPa N N N N·m N·m".split()


def truth(rim: float) -> mf.MF61Params:
    k = 1.0 if rim < 7.5 else 1.0
    up = rim >= 7.5
    return mf.MF61Params(
        FNOMIN=1100.0, R0=0.2032, NOMPRES=83.0,
        pKy1=45.0 * (1.08 if up else 1.0), pDy1=2.4 * (1.02 if up else 1.0),
        qDz1=0.15 * (0.94 if up else 1.0), pDy3=5.0, pKy6=3.0, ppy3=0.3, ppy1=0.2,
    ).replace(pDx1=2.2 * k)


def write(path: Path, rim: float, t, v, sa, ia, kappa, p, fz, fx, fy, mz, rng, noise):
    n = len(t)
    fy_, mz_, fx_ = -fy, -mz, fx  # SAE TTC: Fy y Mz con signo opuesto a la convencion del modelo
    fy_ = fy_ + rng.normal(0, noise * 2400, n)
    fx_ = fx_ + rng.normal(0, noise * 2400, n)
    mz_ = mz_ + rng.normal(0, noise * 60, n)
    arr = np.column_stack([t, np.full(n, v), np.degrees(sa), np.degrees(ia), kappa, np.full(n, 19.5), np.full(n, 20.0),
                           p, fx_, fy_, -fz, np.zeros(n), mz_])
    hdr = (f"Tire_Name\t{TIRE}\nRim_Width\t{rim:g} in\nTest\tSynthetic\n"
           + "\t".join(COLS) + "\n" + "\t".join(UNITS) + "\n")
    path.write_text(hdr + "\n".join("\t".join(f"{x:.6g}" for x in row) for row in arr) + "\n", encoding="latin-1")


def generate(out: Path, seed: int = 0, noise: float = 0.004) -> None:
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    dt = 0.05
    for rim in (7.0, 8.0):
        P = truth(rim)
        tag, run = f"{int(rim)}in", 0

        def emit(label, t, sa, kappa, ia, p, fz):
            nonlocal run
            run += 1
            r = mf.mf61(P, sa, kappa, ia, fz, p)
            write(out / f"synth_{tag}_run{run:02d}_{label}.dat", rim, t, 40.0, sa, ia, kappa, p, fz,
                  np.asarray(r.fx), np.asarray(r.fy), np.asarray(r.mz), rng, noise)

        t = np.arange(0, 48, dt)
        sa_sw = np.radians(12) * np.sin(2 * np.pi * t / 48 * 2)  # ~1 °/s
        emit("warmup", t[:400], np.radians(1.0) * np.sin(t[:400]), np.zeros(400), np.zeros(400),
             np.full(400, 83.0), np.full(400, 1100.0))
        n = len(t)
        for fz in (250.0, 650.0, 1100.0, 1550.0):
            for ia in (0.0, 2.0, 4.0):
                emit("cornering", t, sa_sw, np.zeros(n), np.full(n, np.radians(ia)), np.full(n, 83.0), np.full(n, fz))
        tk = np.arange(0, 24, dt)
        kap = 0.25 * np.sin(2 * np.pi * tk / 24 * 2)
        nk = len(tk)
        for fz in (250.0, 650.0, 1100.0, 1550.0):
            emit("drivebrake", tk, np.zeros(nk), kap, np.zeros(nk), np.full(nk, 83.0), np.full(nk, fz))
        for a_deg in (3.0, 6.0):
            for fz in (650.0, 1100.0):
                emit("combined", tk, np.full(nk, np.radians(a_deg)), kap, np.zeros(nk), np.full(nk, 83.0), np.full(nk, fz))
        for pr in (70.0, 83.0, 97.0):
            emit("pressure", t, sa_sw, np.zeros(n), np.zeros(n), np.full(n, pr), np.full(n, 1100.0))
        tt = np.arange(0, 20, 0.01)
        emit("transient", tt, np.radians(4) * np.sin(2 * np.pi * 2 * tt), np.zeros(len(tt)), np.zeros(len(tt)),
             np.full(len(tt), 83.0), np.full(len(tt), 1100.0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/raw/ttc_synth"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", type=float, default=0.004)
    a = ap.parse_args()
    generate(a.out, a.seed, a.noise)
    print(f"Escrito en {a.out}")