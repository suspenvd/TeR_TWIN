"""YAML/JSON parser and serializer for vehicle setup profiles.

Schema (``config/vehicles/<veh>/vehicle.yaml``) — every key optional, defaults come from ``default_ter27``::

    name: TeR27-4WD
    chassis:    {mass: 290.0, wheelbase: 1.55, ...}      # ChassisParams fields
    susp:       {k_wheel_f: 24000.0, camber_static_f_deg: -2.2, ...}
    steer:      {ackermann: 0.6, ...}
    aero:       {map_file: aero_map.yaml, h_aero: 0.4}   # grids inline or in map_file (relative to this file)
    pt:         {p_acc_max: 80000.0, ...}
    tire:       {r0: 0.2032, kz: 90000.0, inertia_w: 0.32, crr: 0.015}
    tires:      {index: ../../../data/processed/mf61_fits/index.json, front: <slug>, rear: <slug>}
                # or front_yaml / rear_yaml: path to a MF61Params YAML

Angles may be given as ``<field>_deg`` for any field holding radians. Unknown keys raise ``KeyError``.
aero_map.yaml: ``rh_f_grid: [..m..]``, ``rh_r_grid: [..]``, ``cla_f/cla_r/cda: [[..]]`` (rows = rh_f).
"""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import yaml

from ter_twin.models.tires import pacejka_61 as mf

from ..parameters import (AeroParams, ChassisParams, PowertrainParams, SteeringParams, SuspensionParams,
                          TireParams, VehicleParams, default_ter27)

_SECTIONS = {"chassis": ChassisParams, "susp": SuspensionParams, "steer": SteeringParams,
             "aero": AeroParams, "pt": PowertrainParams, "tire": TireParams}
_ARRAY_FIELDS = {"rh_f_grid", "rh_r_grid", "cla_f", "cla_r", "cda"}
_ANGLE_FIELDS = {"camber_static_f", "camber_static_r", "toe_static_f", "toe_static_r", "delta_max"}


def _read(path: Path) -> dict:
    txt = path.read_text(encoding="utf-8")
    return json.loads(txt) if path.suffix.lower() == ".json" else (yaml.safe_load(txt) or {})


def _section(cls, base, data: dict) -> Any:
    names = {f.name for f in dataclasses.fields(cls)}
    kw = {}
    for k, v in (data or {}).items():
        if k in ("map_file", "tires"):
            continue
        name = k[:-4] if k.endswith("_deg") and k[:-4] in _ANGLE_FIELDS else k
        if name not in names:
            raise KeyError(f"{cls.__name__}: unknown field {k!r}")
        if name in ("front", "rear"):
            continue
        if k.endswith("_deg") and name in _ANGLE_FIELDS:
            v = math.radians(float(v))
        kw[name] = jnp.asarray(v, dtype=float) if name in _ARRAY_FIELDS else (
            int(v) if name == "elasto_iter" else float(v))
    return dataclasses.replace(base, **kw)


def _load_tire(spec: dict, base: Path, which: str):
    if spec.get(f"{which}_yaml"):
        return mf.MF61Params.load(base / spec[f"{which}_yaml"])
    slug = spec.get(which)
    if not slug:
        return None
    idx_path = (base / spec["index"]) if spec.get("index") else None
    if idx_path is None:
        raise ValueError(f"tires.{which}={slug!r} needs tires.index")
    doc = json.loads(idx_path.read_text(encoding="utf-8"))
    for t in doc.get("tires", []):
        if t.get("slug") == slug and t.get("status") == "ok":
            return mf.MF61Params.load(idx_path.parent / t["params"])
    raise KeyError(f"slug {slug!r} not found (status ok) in {idx_path}")


def vehicle_from_dict(d: dict, base_dir: str | Path = ".", tire_index: str | Path | None = None) -> VehicleParams:
    base_dir = Path(base_dir)
    vp = default_ter27()
    secs = {}
    for key, cls in _SECTIONS.items():
        data = dict(d.get(key) or {})
        if key == "aero" and data.get("map_file"):
            grid = _read(base_dir / data["map_file"])
            data = {**grid, **{k: v for k, v in data.items() if k != "map_file"}}
        secs[key] = _section(cls, getattr(vp, key), data)
    tsp = dict(d.get("tires") or {})
    if tire_index is not None:
        tsp["index"] = str(Path(tire_index).resolve())
        base_for_tires = Path("/")
    else:
        base_for_tires = base_dir
    front = _load_tire(tsp, base_for_tires, "front")
    rear = _load_tire(tsp, base_for_tires, "rear") or front
    if front is not None:
        secs["tire"] = dataclasses.replace(secs["tire"], front=front, rear=rear)
    return VehicleParams(name=str(d.get("name", vp.name)), **secs)


def load_vehicle(path: str | Path, tire_index: str | Path | None = None) -> VehicleParams:
    path = Path(path)
    return vehicle_from_dict(_read(path), path.parent, tire_index)


def _plain(v: Any) -> Any:
    if isinstance(v, (jnp.ndarray, np.ndarray)):
        return np.asarray(v).tolist()
    if isinstance(v, (float, int, np.floating, np.integer)):
        return float(v) if not isinstance(v, (int, np.integer)) else int(v)
    return v


def vehicle_to_dict(vp: VehicleParams, tire_refs: dict | None = None) -> dict:
    """Plain-dict form. Tyre coefficients are NOT embedded: pass ``tire_refs`` (index/front/rear slugs or *_yaml)."""
    out: dict[str, Any] = {"name": vp.name}
    for key in _SECTIONS:
        sec = getattr(vp, key)
        out[key] = {f.name: _plain(getattr(sec, f.name)) for f in dataclasses.fields(sec)
                    if f.name not in ("front", "rear")}
    if tire_refs:
        out["tires"] = dict(tire_refs)
    return out


def save_vehicle(vp: VehicleParams, path: str | Path, tire_refs: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = vehicle_to_dict(vp, tire_refs)
    path.write_text(json.dumps(doc, indent=2) if path.suffix.lower() == ".json"
                    else yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")