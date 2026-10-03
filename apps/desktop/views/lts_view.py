"""apps/desktop/views/lts_view.py
Quasi-steady-state lap-time simulator and Milliken Moment Method (stub).

The live panel already shows the tyres published by TiresView, so the tyre -> LTS coupling can be
verified before the solver exists.
"""
from __future__ import annotations

from typing import ClassVar

from apps.desktop.views.stub_base import SpecRow, StubSpec, StubView

__all__ = ["LTSView"]


class LTSView(StubView):
    SPEC: ClassVar[StubSpec] = StubSpec(
        title="Lap Time Simulator & MMM",
        subtitle="QSS forward–backward lap time · Milliken Moment Method · minimum-time OCP",
        summary=(
            "Consumes the tyre, aero and suspension models to produce speed profiles, GG(V) envelopes and "
            "lap times for the FSG events. Not implemented yet: the solvers will live under "
            "src/ter_twin/models/full_vehicle/ and src/ter_twin/control/driver/."
        ),
        inputs=(
            SpecRow("Tyres", "front_tyre_params / rear_tyre_params published by the Tyre Model view (MF6.1)."),
            SpecRow("Aero", "Axle loads, drag and balance from the Aerodynamic Maps module."),
            SpecRow("Suspension", "Roll stiffness distribution, roll-centre heights, motion ratios."),
            SpecRow("Powertrain", "Motor / inverter torque-speed limits, accumulator power, thermal derating."),
            SpecRow("Event & track", "config/events/{skidpad_fsg, acceleration_75m, autocross_default, "
                                     "endurance_stint}.yaml and the track curvature κ(s)."),
        ),
        outputs=(
            SpecRow("Lap result", "Lap time, v(s), a_x(s), a_y(s) and energy per lap for endurance."),
            SpecRow("MMM diagram", "AY–N map over (β, δ) at fixed speed: limit balance, stability and control."),
            SpecRow("GGV envelope", "Per-speed GG boundary that the QSS solver and torque vectoring reuse."),
            SpecRow("Sensitivities", "d(t_lap)/d(setup) through jax.grad — e.g. tyre, aero balance, roll split."),
            SpecRow("Validation", "GG-diagram match against telemetry and the baseline lap-time regression."),
        ),
        stack=(
            SpecRow("QSS solver",
                    "Forward–backward speed-profile passes on the point-mass / single-track model: the lowest of "
                    "the cornering limit, the traction-limited forward pass and the braking-limited backward pass."),
            SpecRow("Milliken Moment Method",
                    "Trim solve over a (β, δ) grid at each speed with MF6.1 corners, load transfer from the "
                    "roll-stiffness split, and aero loads; yields AY–N and the maximum lateral acceleration."),
            SpecRow("Optimal control",
                    "CasADi minimum-time OCP (direct collocation, IPOPT) as the reference against the QSS result; "
                    "JAX path kept for differentiable setup sweeps."),
            SpecRow("Integrators", "Heun RK2 and GLRK-4 from core/integrators for transient replays (JAX, float64)."),
        ),
        roadmap=(
            "Point-mass QSS with an MF6.1-derived friction envelope.",
            "Single-track GGV with aero and powertrain limits.",
            "Milliken Moment Method solver and AY–N plots.",
            "Event runners for the four FSG disciplines.",
            "CasADi OCP reference and comparison report.",
            "Regression baseline: tests/regression/test_baseline_laptime.py.",
        ),
        repo_paths=(
            "src/ter_twin/models/full_vehicle/point_mass.py",
            "src/ter_twin/models/full_vehicle/single_track.py",
            "src/ter_twin/models/full_vehicle/planar_15dof.py",
            "src/ter_twin/control/driver/optimal_preview.py",
            "src/ter_twin/validation/benchmarks_vehicle/gg_diagram_match.py",
            "config/events/*.yaml",
        ),
        state_keys=("front_tyre_params", "rear_tyre_params", "active_tyre_slugs", "nominal_fz",
                    "aero_balance", "roll_stiffness_dist"),
    )