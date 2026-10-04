"""Non-linear dual-track model: longitudinal, lateral, yaw, roll + 4 wheel spins.

State  x = [vx, vy, r, phi, p, w_FL, w_FR, w_RL, w_RR, ax_f, ay_f]   (11)
    vx, vy [m/s] body-frame CG velocity; r yaw rate; phi/p roll angle/rate (phi>0 = roll right);
    w_i wheel spin [rad/s]; ax_f, ay_f = first-order-filtered force accelerations (sum F / m) that
    drive the load transfer (tau = susp.tau_load). The lag removes the algebraic loop
    "tyre force -> load transfer -> tyre force" and emulates suspension response.
Heave/pitch are quasi-static (ride heights from axle loads); roll is dynamic.
Input  u = Controls(delta, t_drive[4], t_brake[4], drs)

    m (vx' - r vy) = sum Fx_b - D vx/V          m (vy' + r vx) = sum Fy_b - D vy/V
    Izz r' = sum(x_i Fy_b,i - y_i Fx_b,i + Mz_i)
    (Ixx + ms d^2) phi'' = ms ay d + ms g d phi - K phi - C p
    Iw w_i' = T_drive,i - T_brake,i tanh(w/e) - Fx_w,i Re_i - Crr Fz_i Re_i tanh(w/e)
"""
from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from ter_twin.models.suspension.kinematics import camber_toe, steer_angles
from ter_twin.models.suspension.load_transfer import LoadState, roll_arm, roll_damping, roll_stiffness, wheel_loads
from ter_twin.models.tires.interface import (
    TireOut, body_loads, kappa_from_wheel_speed, tire_forces, wheel_slips
)
from .parameters import G, VehicleParams

N_STATES = 11
IX = dict(vx=0, vy=1, r=2, phi=3, p=4, w=slice(5, 9), ax=9, ay=10)


class Controls(NamedTuple):
    delta: object      # mean road-wheel steer [rad]
    t_drive: object    # (4,) signed motor wheel torque [N m]
    t_brake: object    # (4,) friction brake torque >= 0 [N m]
    drs: object = 0.0


class Forces(NamedTuple):
    loads: LoadState
    tires: TireOut
    fx: object
    fy: object
    mz: object
    ax_c: object       # sum(Fx)/m incl. aero
    ay_c: object


def initial_state(vp: VehicleParams, vx0: float) -> jnp.ndarray:
    re0 = vp.tire.r0 - vp.chassis.mass * G / 4.0 / vp.tire.kz
    x = jnp.zeros(N_STATES)
    return x.at[0].set(vx0).at[5:9].set(vx0 / re0)


def zero_controls(delta=0.0) -> Controls:
    return Controls(jnp.asarray(delta), jnp.zeros(4), jnp.zeros(4), jnp.asarray(0.0))


def compute_forces(vp: VehicleParams, x, u: Controls) -> Forces:
    ch, su = vp.chassis, vp.susp
    vx, vy, r, phi, p = x[0], x[1], x[2], x[3], x[4]
    w = x[5:9]
    ax_f, ay_f = x[9], x[10]
    V = jnp.sqrt(vx * vx + vy * vy + 1e-3)
    ld = wheel_loads(vp, V, ax_f, ay_f, phi, p, u.drs)
    steer0 = steer_angles(vp, u.delta)
    fy_prev = None
    for _ in range(int(su.elasto_iter) + 1):          # fixed-point on compliance steer/camber
        gamma, st_toe = camber_toe(vp, phi, ld.dz, fy_prev)
        steer = steer0 + st_toe
        alpha, vxw, re = wheel_slips(vp, vx, vy, r, steer, ld.fz)
        kappa = kappa_from_wheel_speed(w, re, vxw)
        t = tire_forces(vp, alpha, kappa, gamma, ld.fz, steer, re)
        fy_prev = t.fy_w
    fxb, fyb, mzb = body_loads(vp, t)
    D = ld.aero.drag
    fx = fxb - D * vx / V
    fy = fyb - D * vy / V
    return Forces(ld, t, fx, fy, mzb, fx / ch.mass, fy / ch.mass)


def derivatives(vp: VehicleParams, x, u: Controls, t=0.0):
    ch, su, tp = vp.chassis, vp.susp, vp.tire
    f = compute_forces(vp, x, u)
    vx, vy, r, phi, p = x[0], x[1], x[2], x[3], x[4]
    w = x[5:9]
    kf, kr = roll_stiffness(vp)
    cf, cr = roll_damping(vp)
    d = roll_arm(vp)
    ms = ch.ms
    ixx_axis = ch.ixx + ms * d * d
    p_dot = (ms * f.ay_c * d + ms * G * d * phi - (kf + kr) * phi - (cf + cr) * p) / ixx_axis
    s = jnp.tanh(w / 2.0)
    t_rr = tp.crr * f.tires.fz * f.tires.re * s
    w_dot = (u.t_drive - u.t_brake * s - f.tires.fx_w * f.tires.re - t_rr) / tp.inertia_w
    tau = su.tau_load
    return jnp.stack([
        f.ax_c + r * vy, f.ay_c - r * vx, f.mz / ch.izz,
        p, p_dot, w_dot[0], w_dot[1], w_dot[2], w_dot[3],
        (f.ax_c - x[9]) / tau, (f.ay_c - x[10]) / tau,
    ])


def pose_rates(x, psi):
    """Inertial-frame kinematics: (X', Y', psi')."""
    c, s = jnp.cos(psi), jnp.sin(psi)
    return x[0] * c - x[1] * s, x[0] * s + x[1] * c, x[2]


def rk4_step(vp: VehicleParams, x, u: Controls, dt):
    k1 = derivatives(vp, x, u)
    k2 = derivatives(vp, x + 0.5 * dt * k1, u)
    k3 = derivatives(vp, x + 0.5 * dt * k2, u)
    k4 = derivatives(vp, x + dt * k3, u)
    return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate(vp: VehicleParams, x0, control_fn: Callable[[float, jnp.ndarray], Controls], dt: float, n_steps: int):
    """Fixed-step RK4 via ``lax.scan``. ``control_fn(t, x) -> Controls`` (pure JAX). Returns (t, X[n+1, 11])."""

    def step(x, k):
        xn = rk4_step(vp, x, control_fn(k * dt, x), dt)
        return xn, xn

    _, xs = jax.lax.scan(step, x0, jnp.arange(n_steps))
    X = jnp.concatenate([x0[None], xs])
    return jnp.arange(n_steps + 1) * dt, X