"""Layer 2 - transient dual-track (11 DOF: vx, vy, r, roll, 4 wheel spins, lagged ax/ay) + EKF, JAX.

Why an observer: wheel spins and the longitudinal/lateral velocity states are not measured directly and an
open-loop integration of a stiff tyre/wheel system drifts. The EKF anchors them to the measured channels.

Anti-leakage rule (important for honest R2):
* channels listed in ``assimilate`` are scored on the ONE-STEP-AHEAD PRIOR  y_{k|k-1}  (pure innovation),
* every other channel is scored on the posterior state, which never saw that channel (held-out).
By default vx and wheel speeds are assimilated; yaw rate and ay are held out.

Implementation: the full predict/update recursion is one ``jax.lax.scan`` (jit once per (vehicle, dt) pair).
Components with NaN/dropout are masked; innovations beyond ``gate_sigma`` inflate R (robust to CAN glitches);
below ``v_min`` the state is re-initialised from measurements (wheel slip is singular at standstill); any
non-finite state triggers a reset that is COUNTED (``n_resets`` is the divergence metric).
To plug the 108-DOF model, replace ``f`` and ``C`` in ``_build`` (its repo imports must be fixed first).

Runtime: the filter runs at ``target_hz`` (default 100 Hz, 8 RK4 sub-steps/step to keep the wheel dynamics
stable at 1.25 ms) and is interpolated back to the log grid.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .dsp import fill_nan
from .types import SeriesResult

G = 9.80665
MEAS = ("vx", "yaw_rate", "ay", "ws_fl", "ws_fr", "ws_rl", "ws_rr")
# Friction brake model (same constants as full_car_108dof): T = 2*mu_pad*(P*A)*r_eff per corner, A = 2*pi*d^2/4.
_BRAKE_K = 2.0 * 0.40 * (2.0 * np.pi * 0.025 ** 2 / 4.0) * 0.080 * 1e5     # N m per bar


@dataclass(frozen=True)
class ObserverConfig:
    assimilate: tuple[str, ...] = ("vx", "ws_fl", "ws_fr", "ws_rl", "ws_rr")
    target_hz: float = 100.0
    max_substep: float = 1.25e-3
    v_min: float = 3.0
    gate_sigma: float = 5.0
    brake_bias_front: float = 0.6
    # per-step process std: vx, vy, r, phi, p, w x4, ax_f, ay_f
    q_std: tuple[float, ...] = (0.02, 0.01, 0.003, 1e-4, 1e-3, 0.2, 0.2, 0.2, 0.2, 0.05, 0.05)
    # measurement std: vx [m/s], r [rad/s], ay [m/s2], wheel rolling speeds [m/s]
    r_std: tuple[float, ...] = (0.2, 0.02, 0.3, 0.3, 0.3, 0.3, 0.3)
    p0_std: tuple[float, ...] = (1.0, 0.5, 0.1, 0.02, 0.1, 5.0, 5.0, 5.0, 5.0, 1.0, 1.0)


@dataclass
class ObserverOutput:
    x_hat: np.ndarray        # (N, 11) posterior on the log grid (NaN before first moving sample)
    y_prior: np.ndarray      # (N, 7) one-step-ahead predicted measurements (SI)
    n_resets: int
    dt_filter: float
    k0: int


_CACHE: dict = {}


def _build(vp, dt: float, cfg: ObserverConfig):
    import jax
    import jax.numpy as jnp

    from ter_twin.models.vehicle import rk4_step

    key = (id(vp), round(dt, 9), cfg)
    if key in _CACHE:
        return _CACHE[key][1]
    re0 = float(vp.tire.r0 - vp.chassis.mass * G / 4.0 / vp.tire.kz)
    nsub = max(1, int(np.ceil(dt / cfg.max_substep)))
    h = dt / nsub

    def f(x, u):
        for _ in range(nsub):
            x = rk4_step(vp, x, u, h)
        return x

    C = np.zeros((7, 11))
    C[0, 0], C[1, 2], C[2, 10] = 1.0, 1.0, 1.0
    for i in range(4):
        C[3 + i, 5 + i] = re0
    Cm, Qm = jnp.asarray(C), jnp.diag(jnp.asarray(cfg.q_std) ** 2)
    Rd, P0 = jnp.asarray(cfg.r_std) ** 2, jnp.diag(jnp.asarray(cfg.p0_std) ** 2)
    eye, g2, vmin = jnp.eye(11), cfg.gate_sigma ** 2, cfg.v_min

    def x_reset(z):
        return jnp.zeros(11).at[0].set(z[0]).at[5:9].set(z[0] / re0)   # zero-slip rolling start

    @jax.jit
    def run(x0, U, Z, V, SLOW):
        def step(carry, inp):
            x, P = carry
            u, z, v, slow = inp
            xr = x_reset(z)
            x, P = jnp.where(slow, xr, x), jnp.where(slow, P0, P)
            F = jax.jacfwd(f)(x, u)
            xp = f(x, u)
            Pp = F @ P @ F.T + Qm
            yp = Cm @ xp
            innov = jnp.where(v, z - yp, 0.0)
            S0 = Cm @ Pp @ Cm.T
            nis = innov ** 2 / (jnp.diag(S0) + Rd)
            Re = jnp.where(v, Rd * jnp.maximum(1.0, nis / g2), 1e12)      # robust gating / dropout masking
            K = jnp.linalg.solve(S0 + jnp.diag(Re), Cm @ Pp).T
            xu = xp + K @ innov
            IKC = eye - K @ Cm
            Pu = IKC @ Pp @ IKC.T + K @ jnp.diag(Re) @ K.T                 # Joseph form
            Pu = 0.5 * (Pu + Pu.T)
            bad = ~(jnp.all(jnp.isfinite(xu)) & jnp.all(jnp.isfinite(Pu)))
            reset = bad | slow
            xo, Po = jnp.where(reset, xr, xu), jnp.where(reset, P0, Pu)
            return (xo, Po), (xo, jnp.nan_to_num(yp), bad)

        _, ys = jax.lax.scan(step, (x0, P0), (U, Z, V, SLOW))
        return ys

    _CACHE[key] = (vp, (run, re0))
    return run, re0


def _inputs(ch, vp, cfg, idx):
    import jax.numpy as jnp

    from ter_twin.models.vehicle import Controls

    n = idx.size
    delta = np.radians(fill_nan(ch["steer_angle"]))[idx] / vp.steer.steer_ratio
    corners = ("fl", "fr", "rl", "rr")
    td = np.stack([fill_nan(ch[f"motor_torque_{c}"])[idx] if f"motor_torque_{c}" in ch else np.zeros(n)
                   for c in corners], axis=1)
    pf = fill_nan(ch["brake_press_front"])[idx] if "brake_press_front" in ch else np.zeros(n)
    b = cfg.brake_bias_front
    pr = fill_nan(ch["brake_press_rear"])[idx] if "brake_press_rear" in ch else pf * (1.0 - b) / b
    tb = _BRAKE_K * np.stack([pf, pf, pr, pr], axis=1)
    return Controls(jnp.asarray(delta), jnp.asarray(td), jnp.asarray(tb), jnp.zeros(n))


def _measurements(ch, idx):
    n = idx.size
    z = np.full((n, 7), np.nan)
    z[:, 0] = fill_nan(ch["vx"])[idx] / 3.6
    if "yaw_rate" in ch:
        z[:, 1] = np.radians(ch["yaw_rate"][idx])
    if "ay" in ch:
        z[:, 2] = ch["ay"][idx] * G
    for i, c in enumerate(("fl", "fr", "rl", "rr")):
        if f"wheel_speed_{c}" in ch:
            z[:, 3 + i] = ch[f"wheel_speed_{c}"][idx] / 3.6
    return z


def run_observer(t: np.ndarray, ch: dict[str, np.ndarray], vp, cfg: ObserverConfig = ObserverConfig(),
                 open_loop: bool = False) -> ObserverOutput:
    import jax.numpy as jnp

    t = np.asarray(t, float)
    n, fs = t.size, 1.0 / float(np.median(np.diff(t)))
    stride = max(1, int(round(fs / cfg.target_hz)))
    dt = stride / fs
    run, re0 = _build(vp, dt, cfg)
    vx = fill_nan(ch["vx"]) / 3.6
    k0 = int(np.argmax(vx >= cfg.v_min)) if (vx >= cfg.v_min).any() else 0
    idx = np.arange(k0, n, stride)
    z = _measurements(ch, idx)
    use = np.array([m in cfg.assimilate for m in MEAS])
    valid = np.isfinite(z) & use[None, :]
    if open_loop:
        valid[:] = False
    slow = (np.nan_to_num(z[:, 0], nan=0.0) < cfg.v_min) & (not open_loop)
    x0 = np.zeros(11)
    x0[0], x0[5:9] = z[0, 0], z[0, 0] / re0
    xs, yp, bad = run(jnp.asarray(x0), _inputs(ch, vp, cfg, idx), jnp.asarray(np.nan_to_num(z)),
                      jnp.asarray(valid), jnp.asarray(slow))
    xs, yp, bad = np.asarray(xs), np.asarray(yp), np.asarray(bad)
    grid = np.arange(n)
    x_full = np.column_stack([np.interp(grid, idx, xs[:, j]) for j in range(11)])
    y_full = np.column_stack([np.interp(grid, idx, yp[:, j]) for j in range(7)])
    x_full[:k0] = np.nan
    y_full[:k0] = np.nan
    return ObserverOutput(x_full, y_full, int(bad.sum()), dt, k0)


def observer_series(t: np.ndarray, ch: dict[str, np.ndarray], vp, cfg: ObserverConfig = ObserverConfig(),
                    with_open_loop: bool = True) -> tuple[list[SeriesResult], dict[str, float]]:
    t = np.asarray(t, float)
    vxk = np.nan_to_num(ch["vx"])
    mask = vxk / 3.6 > 4.2
    runs = {"L2": run_observer(t, ch, vp, cfg, False)}
    if with_open_loop:
        runs["L2-OL"] = run_observer(t, ch, vp, cfg, True)
    spec = (("yaw_rate", "deg/s", 2, 1, np.degrees(1.0)), ("ay", "g", 10, 2, 1.0 / G), ("vx", "km/h", 0, 0, 3.6))
    out: list[SeriesResult] = []
    for layer, o in runs.items():
        for name, unit, ix, iy, sc in spec:
            if name not in ch:
                continue
            held_out = (name not in cfg.assimilate) if name != "yaw_rate" else ("yaw_rate" not in cfg.assimilate)
            if layer == "L2" and not held_out:
                pred, mode = o.y_prior[:, iy] * sc, "one-step-ahead prior (assimilated channel)"
            else:
                pred = o.x_hat[:, ix] * sc
                mode = "open-loop (no assimilation)" if layer == "L2-OL" else "held-out posterior"
            s = SeriesResult(f"{name}@{layer}", unit, layer, mode, t, np.asarray(ch[name], float), pred, mask, vxk)
            s.compute()
            out.append(s)
    info = {k: float(v.n_resets) for k, v in runs.items()}
    return out, info