"""TeR-Twin full-vehicle dynamics (dual-track, 4WD in-hub, MF6.1, ride-height aero).

JAX float64 is required for the solvers; enabled on import.
"""
import jax

jax.config.update("jax_enable_x64", True)

from . import aerodynamics, kinematics, load_transfer, powertrain, tire_interface  # noqa: E402
from .dual_track_model import (Controls, N_STATES, compute_forces, derivatives, initial_state, rk4_step,  # noqa: E402
                               simulate, zero_controls)
from .parameters import (AeroParams, ChassisParams, PowertrainParams, SteeringParams, SuspensionParams,  # noqa: E402
                         TireParams, VehicleParams, default_ter27, replace_in)

__all__ = [
    "aerodynamics", "kinematics", "load_transfer", "powertrain", "tire_interface",
    "Controls", "N_STATES", "compute_forces", "derivatives", "initial_state", "rk4_step", "simulate",
    "zero_controls", "AeroParams", "ChassisParams", "PowertrainParams", "SteeringParams",
    "SuspensionParams", "TireParams", "VehicleParams", "default_ter27", "replace_in",
]