"""apps/desktop/views/aerodynamics_view.py
Aerodynamic maps module (stub): downforce, drag and CoP vs ride heights, attitude and DRS.
"""
from __future__ import annotations

from typing import ClassVar

from apps.desktop.views.stub_base import SpecRow, StubSpec, StubView

__all__ = ["AeroView"]


class AeroView(StubView):
    SPEC: ClassVar[StubSpec] = StubSpec(
        title="Aerodynamic Maps",
        subtitle="Downforce · drag · centre of pressure vs ride height, attitude and DRS",
        summary=(
            "Quasi-static aero polars for the TeR27 package. Ride heights, attitude and DRS state map to "
            "axle loads, drag and moments through fitted, smooth response surfaces. Not implemented yet: "
            "the modules under src/ter_twin/models/aerodynamics/ were empty skeletons when this view was written."
        ),
        inputs=(
            SpecRow("Aero map", "config/vehicles/<veh>/aero_map.yaml — CL·A, CD·A and CoP surfaces."),
            SpecRow("Vehicle attitude",
                    "Front/rear ride height, pitch, roll and yaw/sideslip β (heights from the suspension kinematics)."),
            SpecRow("Air state", "Air density ρ and relative airspeed (headwind / crosswind)."),
            SpecRow("DRS command", "Logic-gate output (control/drs/logic_gate.py) and actuator state."),
            SpecRow("CFD / track data", "Calibration points consumed by scripts/calibration/fit_aero_surfaces.py."),
        ),
        outputs=(
            SpecRow("Axle loads", "Aerodynamic ΔFz_front / ΔFz_rear [N] for the load-transfer model."),
            SpecRow("Drag force", "Longitudinal aero force Fx_aero [N] for the powertrain / QSS solver."),
            SpecRow("Aero moments", "Pitch, roll and yaw moments (aero_moments.py)."),
            SpecRow("Aero balance", "Front-fraction balance; shown live below via AppState.aero_balance."),
            SpecRow("DRS delta", "ΔCL·A and ΔCD·A between DRS open/closed and lap-time sensitivity."),
        ),
        stack=(
            SpecRow("Surfaces",
                    "Response surfaces in (h_f, h_r, pitch, yaw, δ_DRS) fitted with scipy least_squares; smooth "
                    "basis so jax.grad is finite across the whole envelope."),
            SpecRow("Quasi-static assumption",
                    "No wake or flow-lag dynamics; valid while vehicle motion is slow versus flow time scales."),
            SpecRow("DRS actuator", "First-order lag with angle and rate saturation (drs_actuator.py)."),
            SpecRow("Validation",
                    "validation/benchmarks_subsystems/evaluate_aero.py; scripts/studies/drs_laptime_sensitivity.py."),
        ),
        roadmap=(
            "aero_map.yaml schema and loader (ter26 / ter27).",
            "Surface fitting pipeline with held-out validation.",
            "quasi_static_map.py evaluation, jit / grad tests.",
            "DRS actuator model and logic-gate integration.",
            "Polars, CoP-migration and DRS lap-time sensitivity plots.",
        ),
        repo_paths=(
            "src/ter_twin/models/aerodynamics/quasi_static_map.py",
            "src/ter_twin/models/aerodynamics/aero_moments.py",
            "src/ter_twin/models/aerodynamics/drs_actuator.py",
            "src/ter_twin/control/drs/logic_gate.py",
            "config/vehicles/ter27/aero_map.yaml",
            "scripts/calibration/fit_aero_surfaces.py",
        ),
        state_keys=("active_vehicle", "aero_balance"),
    )