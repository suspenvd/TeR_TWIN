"""apps/desktop/views/suspension_view.py
Suspension kinematics module (stub): roll centres, camber gain, bump steer.
"""
from __future__ import annotations

from typing import ClassVar

from apps.desktop.views.stub_base import SpecRow, StubSpec, StubView

__all__ = ["SuspensionView"]


class SuspensionView(StubView):
    SPEC: ClassVar[StubSpec] = StubSpec(
        title="Suspension Kinematics",
        subtitle="Double-wishbone kinematics · roll centres · camber gain · bump steer",
        summary=(
            "Spatial loop-closure solver for the TeR26 / TeR27 corners. Hardpoints go in; smooth, "
            "differentiable kinematic lookups come out and feed the planar 15-DOF model and the lap-time "
            "simulator. Not implemented yet: the backing modules under src/ter_twin/models/suspension/ "
            "were empty skeletons when this view was written."
        ),
        inputs=(
            SpecRow("Hardpoints",
                    "config/vehicles/<ter26|ter27>/hardpoints.yaml — wishbone, tie-rod, pushrod, rocker and ARB "
                    "pickup points in the chassis frame [m]."),
            SpecRow("Tyre geometry",
                    "R0 and loaded radius from the selected MF6.1 tyre (front_tyre_params / rear_tyre_params)."),
            SpecRow("Spring, damper, ARB rates",
                    "Parameter sets for springs_dampers.py and arb.py, referenced to motion ratios."),
            SpecRow("Excitation grid", "Heave z, roll φ, pitch θ and rack displacement x_r sweeps."),
            SpecRow("Setup", "nominal_fz and roll_stiffness_dist from the shared AppState."),
        ),
        outputs=(
            SpecRow("Camber / toe curves", "γ(z, φ) and toe(z, φ): camber gain and bump steer [°/m]."),
            SpecRow("Roll centres & anti-geometry",
                    "Front/rear RC height, anti-dive / anti-squat [%], track change, scrub radius."),
            SpecRow("Motion ratios", "MR_spring(z), MR_damper(z), MR_ARB(z) for wheel rate and roll stiffness."),
            SpecRow("Roll stiffness split",
                    "Writes roll_stiffness_dist [-, front fraction] back to AppState for the LTS."),
            SpecRow("Differentiable lookups",
                    "jnp interpolants (core/math/interpolators.py) for in-loop use by planar_15dof."),
        ),
        stack=(
            SpecRow("Solver",
                    "Newton–Raphson on the 6-DOF upright pose: 5 link-length constraints (4 wishbone links + "
                    "tie rod) plus one prescribed vertical travel. Jacobian via jax.jacfwd, float64."),
            SpecRow("Parametrisation",
                    "Pre-tabulated (z, φ, x_r) grids with tensor-product interpolation; branch-free so "
                    "jax.grad / jax.jit stay valid."),
            SpecRow("Instant centres", "Velocity-pole / screw-axis method evaluated on the solved poses."),
            SpecRow("Validation",
                    "validation/benchmarks_subsystems/evaluate_kinematics.py against the CAD export; design "
                    "notes in docs/engineering_design/suspension_kinematics.md."),
        ),
        roadmap=(
            "Hardpoint loader with schema validation (ter26 / ter27).",
            "Loop-closure solver with closure-residual unit tests.",
            "Camber / toe / RC sweeps and lookup export.",
            "ARB + spring/damper wheel-rate model.",
            "Camber-gain, bump-steer and RC-migration plots; CAD correlation report.",
        ),
        repo_paths=(
            "src/ter_twin/models/suspension/roll_centers.py",
            "src/ter_twin/models/suspension/lookups.py",
            "src/ter_twin/models/suspension/springs_dampers.py",
            "src/ter_twin/models/suspension/arb.py",
            "config/vehicles/ter27/hardpoints.yaml",
            "validation/benchmarks_subsystems/evaluate_kinematics.py",
        ),
        state_keys=("active_vehicle", "nominal_fz", "roll_stiffness_dist", "front_tyre_params"),
    )