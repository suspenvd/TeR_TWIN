#!/usr/bin/env python3
"""Ajuste en lote + comparativa cruzada de llantas (por defecto 7" vs 8") para el TeR27.

1. Ejecuta fit_all_ttc_dat.run_batch (o reutiliza index.json con --skip-fit).
2. Por cada Tire_Name con >=2 anchos de llanta, empareja base (--ref-rim) vs comparada (--cmp-rim)
   y calcula Δ% de Cα, μy/μx pico, Fy pico, Mz pico, traza t0 y Cκ en una rejilla de cargas
   (y, opcionalmente, caidas) con mf.compare_params.
3. Escribe comparison_<ref>v<cmp>.{json,csv,md} junto al index.json y muestra la tabla.

Suposiciones: Δ% = 100 (cmp - ref)/ref; cada neumatico se evalua a su propia presion nominal salvo
--pressure; las cargas por defecto cubren el rango de Formula Student (400-1800 N).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
import fit_all_ttc_dat as fit  # noqa: E402
from ter_twin.models.tires import pacejka_61 as mf  # noqa: E402

LOG = logging.getLogger("batch_compare")


def load_index(index_path: Path) -> list[dict]:
    doc = json.loads(index_path.read_text(encoding="utf-8"))
    return [t for t in doc["tires"] if t.get("status") == "ok"]


def pair_entries(entries: list[dict], ref_rim: float | None, cmp_rim: float | None) -> list[tuple[dict, dict]]:
    by_tire: dict[str, list[dict]] = {}
    for e in entries:
        by_tire.setdefault(e["tire_name"], []).append(e)
    pairs = []
    for tire, lst in by_tire.items():
        lst = sorted((e for e in lst if np.isfinite(e["rim_width_in"])), key=lambda e: e["rim_width_in"])
        if ref_rim is not None and cmp_rim is not None:
            a = next((e for e in lst if abs(e["rim_width_in"] - ref_rim) < 0.05), None)
            b = next((e for e in lst if abs(e["rim_width_in"] - cmp_rim) < 0.05), None)
            if a and b:
                pairs.append((a, b))
        else:
            pairs.extend(zip(lst[:-1], lst[1:]))
    return pairs


def compare_pair(a: dict, b: dict, index_dir: Path, fz: list[float], gammas_deg: list[float],
                 pressure: float | None) -> dict:
    Pa = mf.MF61Params.load(index_dir / a["params"])
    Pb = mf.MF61Params.load(index_dir / b["params"])
    out = {"tire_name": a["tire_name"], "ref": a["slug"], "cmp": b["slug"],
           "ref_rim_in": a["rim_width_in"], "cmp_rim_in": b["rim_width_in"], "cases": []}
    for g in gammas_deg:
        res = mf.compare_params(Pa, Pb, fz, np.radians(g), pressure or Pa.NOMPRES, pressure or Pb.NOMPRES)
        out["cases"].append(res)
    return out


def to_markdown(cmp_: dict) -> str:
    lines = [f"## {cmp_['tire_name']}: {cmp_['ref_rim_in']:g}\" (base) vs {cmp_['cmp_rim_in']:g}\" (comparada)", ""]
    for case in cmp_["cases"]:
        lines += [f"**Caída γ = {case['gamma_deg']:.1f}°** — Δ% medio sobre el rango de carga:", "",
                  "| Métrica | Δ% medio |", "|---|---:|"]
        lines += [f"| {mf.METRIC_LABELS[k]} | {v:+.2f} |" for k, v in case["mean_delta_pct"].items()]
        lines += ["", "| Fz [N] | Métrica | Base | Comparada | Δ% |", "|---:|---|---:|---:|---:|"]
        lines += [f"| {r['fz']:.0f} | {r['label']} | {r['ref']:.4g} | {r['cmp']:.4g} | {r['delta_pct']:+.2f} |"
                  for r in case["rows"]]
        lines.append("")
    return "\n".join(lines)


def write_outputs(cmps: list[dict], out_dir: Path, tag: str) -> None:
    (out_dir / f"comparison_{tag}.json").write_text(json.dumps(cmps, indent=2, default=float), encoding="utf-8")
    with open(out_dir / f"comparison_{tag}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["tire_name", "ref", "cmp", "gamma_deg", "fz_N", "metric", "ref_value", "cmp_value", "delta_pct"])
        for c in cmps:
            for case in c["cases"]:
                for r in case["rows"]:
                    w.writerow([c["tire_name"], c["ref"], c["cmp"], case["gamma_deg"], r["fz"], r["metric"],
                                r["ref"], r["cmp"], r["delta_pct"]])
    (out_dir / f"comparison_{tag}.md").write_text("\n".join(to_markdown(c) for c in cmps), encoding="utf-8")


def main(argv=None) -> int:
    ap = fit.build_parser()
    ap.description = "Ajuste en lote MF6.1 + comparativa de llantas"
    ap.add_argument("--skip-fit", action="store_true", help="reutiliza <output>/index.json")
    ap.add_argument("--ref-rim", type=float, default=7.0)
    ap.add_argument("--cmp-rim", type=float, default=8.0)
    ap.add_argument("--any-pairs", action="store_true", help="comparar anchos adyacentes sin fijar 7/8")
    ap.add_argument("--fz", type=float, nargs="+", default=[400.0, 800.0, 1200.0, 1600.0])
    ap.add_argument("--gamma-deg", type=float, nargs="+", default=[0.0, 2.0])
    ap.add_argument("--pressure", type=float, help="kPa comunes; por defecto NOMPRES de cada ajuste")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = fit.config_from_args(a)
    index_path = cfg.output / "index.json"
    if a.dry_run:
        fit.dry_run(cfg)
        return 0
    entries = load_index(index_path) if a.skip_fit else [e for e in fit.run_batch(cfg) if e.get("status") == "ok"]
    pairs = pair_entries(entries, None if a.any_pairs else a.ref_rim, None if a.any_pairs else a.cmp_rim)
    if not pairs:
        LOG.error("Sin pares %s\" vs %s\" con ajuste valido", a.ref_rim, a.cmp_rim)
        return 2
    cmps = [compare_pair(x, y, cfg.output, a.fz, a.gamma_deg, a.pressure) for x, y in pairs]
    tag = "any" if a.any_pairs else f"{a.ref_rim:g}v{a.cmp_rim:g}"
    write_outputs(cmps, cfg.output, tag)
    for c in cmps:
        print(f"\n{c['tire_name']}: {c['ref_rim_in']:g}\" → {c['cmp_rim_in']:g}\"")
        for case in c["cases"]:
            print(f"  γ={case['gamma_deg']:.1f}°  " + "  ".join(
                f"{mf.METRIC_LABELS[k]}: {v:+.1f}%" for k, v in case["mean_delta_pct"].items()
                if k in ("c_alpha_N_deg", "mu_y_peak", "mu_x_peak", "mz_peak_Nm", "trail0_mm")))
    print(f"\nInformes en {cfg.output}/comparison_{tag}.(json|csv|md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())