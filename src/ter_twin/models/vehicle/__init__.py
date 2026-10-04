"""TeR-Twin: Ensamblajes de chasis completo y dinámica vehicular."""
from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)

from .parameters import (
    ChassisParams,
    PowertrainParams,
    SteeringParams,
    SuspensionParams,
    TireParams,
    VehicleParams,
    default_ter27,
    replace_in,
)
from .dual_track_11dof import (
    Controls,
    N_STATES,
    compute_forces,
    derivatives,
    initial_state,
    rk4_step,
    simulate,
    zero_controls,
)

__all__ = [
    "VehicleParams",
    "ChassisParams",
    "SuspensionParams",
    "SteeringParams",
    "PowertrainParams",
    "TireParams",
    "default_ter27",
    "replace_in",
    "Controls",
    "N_STATES",
    "compute_forces",
    "derivatives",
    "initial_state",
    "rk4_step",
    "simulate",
    "zero_controls",
]