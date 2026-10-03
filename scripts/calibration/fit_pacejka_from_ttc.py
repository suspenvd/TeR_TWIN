#!/usr/bin/env python3
# scripts/calibration/fit_pacejka_from_ttc.py
# Lector universal de datos TTC y ajuste de coeficientes Pacejka 6.1

import argparse
import numpy as np
import scipy.io as sio
from scipy.optimize import least_squares
from pathlib import Path

def load_ttc_file(file_path: Path):
    """
    Lee archivos de ensayo Flat-Trac (.mat) o exportaciones en CSV.
    """
    if file_path.suffix == ".mat":
        mat = sio.loadmat(file_path, squeeze_me=True)
        data = mat.get("run", mat)
        sa = data["SA"]        # Slip angle [deg]
        ia = data["IA"]        # Camber angle [deg]
        fz = np.abs(data["FZ"])# Carga vertical [N]
        fy = data["FY"]        # Fuerza lateral [N]
    else:
        raw = np.genfromtxt(file_path, delimiter=",", names=True)
        sa, ia, fz, fy = raw["SA"], raw["IA"], np.abs(raw["FZ"]), raw["FY"]
    return sa, ia, fz, fy

def mf61_residual(params, sa_deg, ia_deg, fz_n, fy_meas):
    pcy1, pdy1, pdy2, pky1, pky2, pey1, pey2 = params
    sa_rad = np.radians(sa_deg)
    ia_rad = np.radians(ia_deg)
    
    fz0 = 654.0
    dfz = (fz_n - fz0) / fz0
    
    mu = pdy1 + pdy2 * dfz
    D = mu * fz_n
    C = pcy1
    Kya = pky1 * fz0 * np.sin(2.0 * np.arctan(fz_n / (pky2 * fz0)))
    B = Kya / (C * D + 1e-9)
    E = np.clip(pey1 + pey2 * dfz, -10.0, 1.0)
    
    Bx = B * sa_rad
    fy_pred = D * np.sin(C * np.arctan(Bx - E * (Bx - np.arctan(Bx))))
    
    # Ponderación por carga normal para evitar sesgo en cargas bajas
    return (fy_meas - fy_pred) / np.sqrt(fz_n)

def calibrate_rim(file_path: Path):
    print(f"[*] Leyendo datos de ensayo desde: {file_path}")
    sa, ia, fz, fy = load_ttc_file(file_path)
    
    # Valores iniciales lógicos para PCY1, PDY1, PDY2, PKY1, PKY2, PEY1, PEY2
    p0 = [1.5, 1.6, -0.15, 50.0, 2.3, -0.5, -0.1]
    bounds = (
        [1.0, 0.8, -0.8, 10.0, 0.5, -5.0, -1.0],
        [2.2, 2.6,  0.2, 90.0, 5.0,  0.5,  1.0]
    )
    
    res = least_squares(mf61_residual, p0, args=(sa, ia, fz, fy), bounds=bounds, loss="huber")
    print(f"[+] Ajuste completado con éxito. Coste residual: {res.cost:.4f}")
    return res.x

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrador Pacejka 6.1 desde TTC")
    parser.add_argument("--rim7", type=Path, required=True, help="Fichero TTC para llanta 7 pulgadas")
    parser.add_argument("--rim8", type=Path, required=True, help="Fichero TTC para llanta 8 pulgadas")
    args = parser.parse_args()

    print("--- CALIBRACIÓN LLANTA 7 PULGADAS ---")
    coeffs_7 = calibrate_rim(args.rim7)
    
    print("\n--- CALIBRACIÓN LLANTA 8 PULGADAS ---")
    coeffs_8 = calibrate_rim(args.rim8)
    
    print("\n==================================================")
    print(" RESULTADOS FINALES DE COEFICIENTES MF6.1")
    print("==================================================")
    labels = ["PCY1", "PDY1", "PDY2", "PKY1", "PKY2", "PEY1", "PEY2"]
    for lbl, v7, v8 in zip(labels, coeffs_7, coeffs_8):
        diff = (v8 / v7 - 1.0) * 100.0
        print(f"{lbl:<10} | 7\": {v7:>8.4f} | 8\": {v8:>8.4f} | Δ: {diff:>+6.2f}%")
    print("==================================================")