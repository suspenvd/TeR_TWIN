"""Steering kinematics (rack, Ackermann) and camber/toe elasto-kinematics.

All functions accept ``xp`` (jnp | numpy). Vectors are ordered FL, FR, RL, RR.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ter_twin.models.vehicle.parameters import SIDE, VehicleParams

def rack_to_delta(vp: VehicleParams, x_rack):
    """Rack travel [m] -> mean road-wheel steer angle [rad] (cubic nonlinearity optional)."""
    s = vp.steer
    return s.rack_ratio * x_rack + s.rack_cubic * x_rack ** 3


def delta_to_rack(vp: VehicleParams, delta, n_iter: int = 6):
    """Inverse of :func:`rack_to_delta` (Newton, exact for the linear case)."""
    s = vp.steer
    x = delta / s.rack_ratio
    for _ in range(n_iter):
        x = x - (s.rack_ratio * x + s.rack_cubic * x ** 3 - delta) / (s.rack_ratio + 3 * s.rack_cubic * x * x)
    return x


def steer_angles(vp: VehicleParams, delta_c, xp=jnp):
    """Road-wheel angles [FL, FR, RL, RR] for a mean steer ``delta_c``.

    Small-angle Ackermann: delta_inner/outer = delta_c +/- k*delta_c*|delta_c|, k = ack*t_f/(2L).
    Smooth |.| keeps the derivative finite at zero steer.
    """
    ch = vp.chassis
    k = vp.steer.ackermann * ch.track_f / (2.0 * ch.wheelbase)
    dd = k * delta_c * xp.sqrt(delta_c * delta_c + 1e-8)
    z = xp.zeros_like(delta_c + 0.0)
    return xp.stack([delta_c + dd, delta_c - dd, z, z], axis=-1)


def ackermann_error(vp: VehicleParams, delta_c):
    """Deviation from true Ackermann: cot(d_out) - cot(d_in) - t/L  [-]  (0 = perfect)."""
    d = steer_angles(vp, delta_c, np)
    ch = vp.chassis
    inner, outer = d[..., 0], d[..., 1]
    return 1.0 / np.tan(outer) - 1.0 / np.tan(inner) + ch.track_f / ch.wheelbase


def camber_toe(vp: VehicleParams, phi, dz, fy=None, xp=jnp):
    """Wheel inclination (tyre-model sign) and steer-equivalent toe angle per wheel.

    Parameters
    ----------
    phi : body roll [rad] (>0 = roll to the right).
    dz  : (4,) bump travel per wheel [m] (>0 = compression), roll contribution included.
    fy  : (4,) tyre lateral forces [N] for compliance (optional).

    Returns
    -------
    gamma : (4,) tilt of the wheel top towards +y [rad]  (= SIDE * SAE camber - rigid body roll).
    steer_toe : (4,) additive steer angle from toe, bump steer and compliance [rad].

    The roll camber gain follows from camber_bump*dz: g = camber_bump * track/2 (typ. 0.6-0.8).
    """
    su = vp.susp
    side = xp.asarray(SIDE)
    one = xp.ones(2)
    cam0 = xp.concatenate([su.camber_static_f * one, su.camber_static_r * one])
    kb = xp.concatenate([su.camber_bump_f * one, su.camber_bump_r * one])
    toe0 = xp.concatenate([su.toe_static_f * one, su.toe_static_r * one])
    bs = xp.concatenate([su.bump_steer_f * one, su.bump_steer_r * one])
    cam_sae = cam0 - kb * dz
    gamma = side * cam_sae - phi
    toe = toe0 + bs * dz
    steer_toe = -side * toe
    if fy is not None:
        ct = xp.concatenate([su.toe_compliance_f * one, su.toe_compliance_r * one])
        cc = xp.concatenate([su.camber_compliance_f * one, su.camber_compliance_r * one])
        steer_toe = steer_toe - ct * fy
        gamma = gamma - cc * fy
    return gamma, steer_toe