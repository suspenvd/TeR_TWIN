"""Wheel-load solver: static + aero + longitudinal + lateral (geometric vs elastic) transfer.

Derivation (pitch plane, a = CG->front axle, b = CG->rear axle, L = a + b)
    N_f = m g b/L + DF_f - dN ,   N_r = m g a/L + DF_r + dN
    dN  = (m ax h_cg + D h_aero) / L        (ax = sum(F_x)/m of the CG, D = aero drag)
Lateral, per axle (ay = sum(F_y)/m):
    dFz_axle = [ (ms_axle h_rc + mu_axle h_u) ay + K_axle phi + C_axle p ] / track
    geometric (links)                          elastic (springs+ARB+dampers)
Roll angle phi is either a dynamic state or the steady value :func:`steady_roll`.
Ride heights are found with ``n_aero`` fixed-point passes (aero load <-> ride height).
Anti-dive/squat only removes the linked share of dN from the spring deflection (ride height).
"""
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from .aerodynamics import AeroOut, aero_forces
from .parameters import G, VehicleParams


class LoadState(NamedTuple):
    fz: object        # (4,) wheel loads [N]
    dz: object        # (4,) bump travel [m]
    rh_f: object
    rh_r: object
    theta: object     # pitch, nose-down positive [rad]
    aero: AeroOut
    d_long: object    # longitudinal transfer dN (rear +) [N]
    d_lat_f: object   # front axle lateral transfer (right +) [N]
    d_lat_r: object


def roll_stiffness(vp: VehicleParams):
    """(K_f, K_r) [N m/rad] = springs (0.5 k t^2) + ARB."""
    ch, su = vp.chassis, vp.susp
    return (0.5 * su.k_wheel_f * ch.track_f ** 2 + su.k_arb_f,
            0.5 * su.k_wheel_r * ch.track_r ** 2 + su.k_arb_r)


def roll_damping(vp: VehicleParams):
    ch, su = vp.chassis, vp.susp
    return 0.5 * su.c_wheel_f * ch.track_f ** 2, 0.5 * su.c_wheel_r * ch.track_r ** 2


def roll_arm(vp: VehicleParams):
    """Distance from the roll axis (at the CG station) to the sprung CG [m]."""
    ch = vp.chassis
    h_ra = ch.h_rc_f + (ch.h_rc_r - ch.h_rc_f) * ch.a / ch.wheelbase
    return ch.h_sprung - h_ra


def steady_roll(vp: VehicleParams, ay):
    """Steady roll angle [rad] for lateral acceleration ``ay``."""
    kf, kr = roll_stiffness(vp)
    ms, d = vp.chassis.ms, roll_arm(vp)
    return ms * ay * d / (kf + kr - ms * G * d)


def _sfloor(x, eps=2.0, xp=jnp):
    """Smooth max(x, 0) (keeps wheel loads positive and differentiable)."""
    return 0.5 * (x + xp.sqrt(x * x + eps * eps))


def wheel_loads(vp: VehicleParams, v, ax, ay, phi=0.0, p=0.0, drs=0.0, n_aero: int = 3, xp=jnp) -> LoadState:
    ch, su, ae = vp.chassis, vp.susp, vp.aero
    L, a, b, m = ch.wheelbase, ch.a, ch.b, ch.mass
    rh_f, rh_r = ch.rh0_f, ch.rh0_r
    for i in range(n_aero + 1):
        A = aero_forces(ae, v, rh_f, rh_r, phi, drs, L, xp)
        dN = (m * ax * ch.h_cg + A.drag * ae.h_aero) / L
        d_f = (A.df_f - (1.0 - su.anti_f) * dN) / (2.0 * su.k_wheel_f)   # axle compression [m]
        d_r = (A.df_r + (1.0 - su.anti_r) * dN) / (2.0 * su.k_wheel_r)
        if i < n_aero:
            rh_f, rh_r = ch.rh0_f - d_f, ch.rh0_r - d_r

    n_f = m * G * b / L + A.df_f - dN
    n_r = m * G * a / L + A.df_r + dN

    ms = ch.ms
    kf, kr = roll_stiffness(vp)
    cf, cr = roll_damping(vp)
    geo_f = (ms * b / L * ch.h_rc_f + 2.0 * ch.mu_f * ch.h_unsprung) * ay / ch.track_f
    geo_r = (ms * a / L * ch.h_rc_r + 2.0 * ch.mu_r * ch.h_unsprung) * ay / ch.track_r
    dl_f = geo_f + (kf * phi + cf * p) / ch.track_f
    dl_r = geo_r + (kr * phi + cr * p) / ch.track_r

    fz = xp.stack([0.5 * n_f - dl_f, 0.5 * n_f + dl_f, 0.5 * n_r - dl_r, 0.5 * n_r + dl_r])
    fz = _sfloor(fz, 2.0, xp)
    dz = xp.stack([d_f - 0.5 * ch.track_f * phi, d_f + 0.5 * ch.track_f * phi,
                   d_r - 0.5 * ch.track_r * phi, d_r + 0.5 * ch.track_r * phi])
    theta = (d_f - d_r) / L
    return LoadState(fz, dz, rh_f, rh_r, theta, A, dN, dl_f, dl_r)


def load_transfer_summary(vp: VehicleParams, v, ax, ay, drs=0.0) -> dict:
    """Convenience (NumPy-friendly) report at steady roll: geometric vs elastic split."""
    import numpy as np

    phi = steady_roll(vp, ay)
    ls = wheel_loads(vp, v, ax, ay, phi, 0.0, drs, xp=np)
    ch = vp.chassis
    kf, kr = roll_stiffness(vp)
    return {
        "fz": np.asarray(ls.fz), "phi_deg": float(np.degrees(phi)), "pitch_deg": float(np.degrees(ls.theta)),
        "rh_f_mm": float(ls.rh_f * 1e3), "rh_r_mm": float(ls.rh_r * 1e3),
        "dFz_long": float(ls.d_long), "dFz_lat_front": float(ls.d_lat_f), "dFz_lat_rear": float(ls.d_lat_r),
        "roll_stiffness_front_frac": float(kf / (kf + kr)),
        "dFz_elastic_front": float(kf * phi / ch.track_f), "dFz_elastic_rear": float(kr * phi / ch.track_r),
    }