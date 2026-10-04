"""
models/differentiable_ekf.py

Differentiable EKF for real-time parameter estimation from standard MoTeC sensors.
State: θ = [λ_μ_f, λ_μ_r, T_opt, h_cg, α_peak] — 5 scalars.
Obs:   y = [ay_measured, wz_measured] — 2 channels.

FIX (this revision): _simulate_with_params previously called simulate_step()
with lambda_mu_f/lambda_mu_r/T_opt_override/alpha_scale kwargs that do not
exist anywhere in DifferentiableMultiBodyVehicle.simulate_step — the EKF has
never actually executed. Now routes through the real `tire_cal` (6,) 
[mu_f, mu_r, T_opt_override, alpha_scale, rby1_scale, rby2_scale] argument
added to simulate_step, and expands the 2-channel [steer, force] control into
the 6-channel [δ, T_fl..T_rr, F_brake] vector the 108-DOF physics expects
(equal 4-way torque split, matching step_with_params' convention).
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
from functools import partial
from models.vehicle_dynamics import DifferentiableMultiBodyVehicle

IDX_LAMBDA_MU_F = 0
IDX_LAMBDA_MU_R = 1
IDX_T_OPT       = 2
IDX_H_CG        = 3
IDX_ALPHA_PEAK  = 4
PARAM_DIM = 5


class DifferentiableEKF:
    def __init__(self, vehicle: DifferentiableMultiBodyVehicle):
        self.vehicle = vehicle
        self.theta_hat = jnp.array([1.0, 1.0, 90.0, 0.285, 0.13], dtype=jnp.float32)
        self.P = jnp.diag(jnp.array([0.04, 0.04, 100.0, 0.0004, 0.0016], dtype=jnp.float32))
        self.Q = jnp.diag(jnp.array([1e-5, 1e-5, 0.01, 1e-7, 1e-4], dtype=jnp.float32))
        self.R_meas = jnp.diag(jnp.array([
            (0.02 * 9.81) ** 2,
            (0.5 * jnp.pi / 180) ** 2,
        ], dtype=jnp.float32))

    @partial(jax.jit, static_argnums=(0,))
    def _simulate_with_params(
        self,
        x_state: jax.Array,   # (108,) full physics state
        u:       jax.Array,   # (2,) [steer, net_force] — MoTeC-derived
        setup:   jax.Array,   # (28,)
        theta:   jax.Array,   # [λ_μ_f, λ_μ_r, T_opt, h_cg, α_peak]
        dt:      float,
    ) -> jax.Array:
        lambda_mu_f = theta[IDX_LAMBDA_MU_F]
        lambda_mu_r = theta[IDX_LAMBDA_MU_R]
        t_opt       = theta[IDX_T_OPT]
        h_cg_theta  = theta[IDX_H_CG]
        alpha_peak  = theta[IDX_ALPHA_PEAK]

        # Peak-slip-angle shift → cornering-stiffness scale (matches the
        # alpha_scale hook added to compute_force_and_sigma: larger alpha_peak
        # ⇒ softer/"lazier" tire ⇒ smaller stiffness scale).
        alpha_peak_nominal = 0.13
        b_scale = alpha_peak_nominal / jnp.clip(alpha_peak, 0.05, 0.30)

        tire_cal = jnp.array(
            [lambda_mu_f, lambda_mu_r, t_opt, b_scale, 1.0, 1.0],
            dtype=jnp.float32,
        )
        setup_patched = setup.at[25].set(h_cg_theta)

        # Expand [steer, force] → 6-channel [δ, T_fl,T_fr,T_rl,T_rr,F_brake].
        # Equal 4-way split: positive force → drive torque, negative → hydraulic brake.
        steer, force = u[0], u[1]
        R_wheel = self.vehicle.R_wheel
        T_each  = jax.nn.relu(force) * R_wheel / 4.0
        F_brake = jax.nn.relu(-force)
        u_full  = jnp.array([steer, T_each, T_each, T_each, T_each, F_brake])

        x_next = self.vehicle.simulate_step(
            x_state, u_full, setup_patched, dt=dt, n_substeps=1, tire_cal=tire_cal,
        )

        vx_next = x_next[14]
        wz_next = x_next[19]
        dvy_dt  = (x_next[15] - x_state[15]) / dt
        ay_pred = vx_next * wz_next + dvy_dt
        return jnp.array([ay_pred, wz_next])

    def update(self, x_state, u, setup, ay_measured, wz_measured, dt=0.005):
        theta_pred = self.theta_hat
        P_pred     = self.P + self.Q

        H = jax.jacobian(
            lambda th: self._simulate_with_params(x_state, u, setup, th, dt)
        )(theta_pred)

        y_pred = self._simulate_with_params(x_state, u, setup, theta_pred, dt)
        y_meas = jnp.array([ay_measured, wz_measured])
        innov  = y_meas - y_pred

        S = H @ P_pred @ H.T + self.R_meas
        K = P_pred @ H.T @ jnp.linalg.inv(S)

        theta_new = theta_pred + K @ innov
        P_new     = (jnp.eye(PARAM_DIM) - K @ H) @ P_pred

        theta_new = jnp.clip(
            theta_new,
            jnp.array([0.5, 0.5, 60.0, 0.22, 0.07]),
            jnp.array([1.5, 1.5, 120.0, 0.40, 0.25]),
        )

        self.theta_hat, self.P = theta_new, P_new
        return theta_new, P_new

    def get_calibrated_params(self) -> dict:
        return {
            'lambda_mu_front': float(self.theta_hat[IDX_LAMBDA_MU_F]),
            'lambda_mu_rear':  float(self.theta_hat[IDX_LAMBDA_MU_R]),
            'T_opt_estimated': float(self.theta_hat[IDX_T_OPT]),
            'h_cg_estimated':  float(self.theta_hat[IDX_H_CG]),
            'alpha_peak_estimated_deg': float(jnp.degrees(self.theta_hat[IDX_ALPHA_PEAK])),
            'uncertainty_lambda': float(jnp.sqrt(self.P[IDX_LAMBDA_MU_F, IDX_LAMBDA_MU_F])),
        }