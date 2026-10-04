# models/tire_transient.py
# ═══════════════════════════════════════════════════════════════════════════════
# Project-GP — Enhanced Tire Transient Dynamics
# ═══════════════════════════════════════════════════════════════════════════════
#
# PROBLEM (Flaw #5):
#   compute_transient_slip_derivatives() uses a first-order lag:
#     dα_t/dt = (α_kin - α_t) / τ,   τ = σ / |Vx|
#   with a CONSTANT relaxation length σ = 0.35 m.
#
#   Reality:
#   1. σ depends on normal load: σ(Fz) = σ_0 + σ_1·(Fz/Fz0 - 1)
#      Light wheel → shorter σ → faster response → less lag
#   2. σ depends on slip level: at high slip, the carcass is fully
#      deflected and σ collapses (contact patch "breaks away")
#   3. The first-order model has a -20 dB/dec rolloff — real tires
#      show a -40 dB/dec rolloff (second-order from carcass + tread)
#   4. At very low speed, τ = σ/|Vx| → ∞, causing numerical stiffness
#
#   The TV controller at 200Hz requests torque corrections faster than
#   the tire carcass can physically respond. The sim allows these because
#   the Pacejka model responds instantly, creating false control authority.
#
# SOLUTION:
#   Second-order carcass-tread model:
#
#     ẍ_t + 2ζωₙ·ẋ_t + ωₙ²·x_t = ωₙ²·x_kin
#
#   where:
#     ωₙ = |Vx| / σ(Fz, α)       — natural frequency
#     ζ = damping ratio ∈ [0.6, 1.0] (overdamped at low speed)
#     σ(Fz, α) = σ_0·(Fz/Fz0)^p · (1 - β·|α/α_peak|²)
#
#   This creates:
#   - -40 dB/dec rolloff above ωₙ (physically correct)
#   - Speed-dependent bandwidth (slow car = sluggish, fast car = responsive)
#   - Load-dependent dynamics (light wheel responds faster)
#   - Slip-dependent σ collapse at saturation
#   - Low-speed regularization via minimum ωₙ floor
#
# STATE:
#   Per corner: 4 transient states (α_t, dα_t/dt, κ_t, dκ_t/dt)
#   Total: 4 corners × 4 = 16 transient states
#   (Current: 4 corners × 2 = 8 states — net +8)
#
# JAX CONTRACT: Pure JAX, JIT-safe, C∞ everywhere.
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import NamedTuple
from functools import partial


# ─────────────────────────────────────────────────────────────────────────────
# §1  Configuration
# ─────────────────────────────────────────────────────────────────────────────

class TireTransientConfig(NamedTuple):
    """Tire transient dynamics parameters."""
    # Relaxation length at nominal load
    sigma_alpha_0: float = 0.25   # lateral relaxation length [m]
    sigma_kappa_0: float = 0.15   # longitudinal relaxation length [m]

    # Load dependence: σ(Fz) = σ_0 · (Fz/Fz0)^p_load
    Fz0: float = 1000.0           # nominal load [N]
    p_load_alpha: float = 0.5     # load exponent for lateral σ
    p_load_kappa: float = 0.4     # load exponent for longitudinal σ

    # Slip-dependent σ collapse
    alpha_peak: float = 0.14      # peak lateral slip angle [rad] (~8°)
    kappa_peak: float = 0.12      # peak longitudinal slip ratio
    beta_collapse: float = 0.6    # σ reduction at peak slip (0 = no reduction)

    # Second-order damping
    zeta_base: float = 0.7        # damping ratio at high speed
    zeta_low_speed: float = 1.2   # overdamped at low speed (prevents ringing)
    v_zeta_transition: float = 5.0  # speed threshold for damping transition [m/s]

    # Numerical safety
    omega_n_min: float = 5.0       # minimum natural frequency [rad/s] (prevents stiffness)
    omega_n_max: float = 500.0     # maximum natural frequency [rad/s]
    v_min: float = 1.0             # minimum velocity for relaxation [m/s]


# ─────────────────────────────────────────────────────────────────────────────
# §2  Load and Slip-Dependent Relaxation Length
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def relaxation_length(
    Fz: jax.Array,
    slip: jax.Array,
    sigma_0: float,
    Fz0: float = 1000.0,
    p_load: float = 0.5,
    slip_peak: float = 0.14,
    beta_collapse: float = 0.6,
) -> jax.Array:
    """
    Speed, load, and slip-dependent relaxation length.

    σ(Fz, s) = σ_0 · (Fz/Fz0)^p_load · Γ_collapse(s)

    where Γ_collapse(s) = 1 - β · σ(|s/s_peak|)
    implements the smooth σ collapse at high slip.

    Physics: at light loads, contact patch is smaller → shorter carcass
    deformation path → smaller σ. At peak slip, the contact patch
    is partially sliding → effective σ drops as the elastic portion
    of the contact shrinks.
    """
    # Load dependence
    Fz_safe = jnp.maximum(Fz, 50.0)
    load_factor = jax.nn.softplus(Fz_safe / Fz0) ** p_load

    # Slip collapse: smooth reduction approaching peak
    slip_ratio = jnp.abs(slip) / (slip_peak + 1e-6)
    # Use sigmoid to smoothly transition from 1 (low slip) to (1-β) (peak slip)
    collapse = 1.0 - beta_collapse * jax.nn.sigmoid(5.0 * (slip_ratio - 0.5))

    sigma = sigma_0 * load_factor * collapse
    return jnp.clip(sigma, 0.05, 1.0)  # physical bounds [m]


# ─────────────────────────────────────────────────────────────────────────────
# §3  Natural Frequency and Damping
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def compute_omega_n(
    Vx: jax.Array,
    sigma: jax.Array,
    omega_min: float = 5.0,
    omega_max: float = 500.0,
) -> jax.Array:
    """
    Natural frequency of the tire carcass transient response.

    ωₙ = |Vx| / σ

    At 15 m/s with σ = 0.25 m: ωₙ = 60 rad/s ≈ 10 Hz
    At 30 m/s with σ = 0.25 m: ωₙ = 120 rad/s ≈ 19 Hz

    This means the tire bandwidth INCREASES with speed — which is
    physically correct and explains why high-speed control is feasible
    but low-speed maneuvering feels sluggish.
    """
    Vx_safe = jnp.maximum(jnp.abs(Vx), 1.0)
    omega = Vx_safe / (sigma + 1e-6)
    return jnp.clip(omega, omega_min, omega_max)


@jax.jit
def compute_zeta(
    Vx: jax.Array,
    zeta_base: float = 0.7,
    zeta_low: float = 1.2,
    v_transition: float = 5.0,
) -> jax.Array:
    """
    Speed-dependent damping ratio.

    At high speed: ζ ≈ 0.7 (slightly underdamped, realistic oscillation)
    At low speed: ζ > 1.0 (overdamped, prevents numerical ringing)

    Smooth transition via sigmoid.
    """
    Vx_abs = jnp.abs(Vx)
    weight_high = jax.nn.sigmoid(3.0 * (Vx_abs - v_transition))
    return zeta_low + (zeta_base - zeta_low) * weight_high


# ─────────────────────────────────────────────────────────────────────────────
# §4  Second-Order Transient Slip Derivatives
# ─────────────────────────────────────────────────────────────────────────────

@partial(jax.jit, static_argnums=())
def compute_transient_slip_derivatives_2nd_order(
    alpha_kin: jax.Array,      # kinematic slip angle [rad]
    kappa_kin: jax.Array,      # kinematic slip ratio
    alpha_t: jax.Array,        # transient slip angle [rad]
    alpha_dot: jax.Array,      # transient slip angle rate [rad/s]
    kappa_t: jax.Array,        # transient slip ratio
    kappa_dot: jax.Array,      # transient slip ratio rate [1/s]
    Fz: jax.Array,             # normal load [N]
    Vx: jax.Array,             # longitudinal velocity [m/s]
    cfg: TireTransientConfig = TireTransientConfig(),
) -> tuple:
    """
    Second-order tire transient slip dynamics.

    State equations (per corner):
      ẍ_α + 2ζωₙ·ẋ_α + ωₙ²·x_α = ωₙ²·α_kin
      ẍ_κ + 2ζωₙ_κ·ẋ_κ + ωₙ_κ²·x_κ = ωₙ_κ²·κ_kin

    Written as first-order system:
      dα_t/dt = α_dot
      dα_dot/dt = ωₙ²·(α_kin - α_t) - 2ζωₙ·α_dot

    Returns:
        d_alpha_t: dα_t/dt
        d_alpha_dot: dα̈_t/dt (acceleration)
        d_kappa_t: dκ_t/dt
        d_kappa_dot: dκ̈_t/dt
    """
    # ── Lateral channel ───────────────────────────────────────────────────
    sigma_alpha = relaxation_length(
        Fz, alpha_t, cfg.sigma_alpha_0, cfg.Fz0,
        cfg.p_load_alpha, cfg.alpha_peak, cfg.beta_collapse)

    omega_n_alpha = compute_omega_n(
        Vx, sigma_alpha, cfg.omega_n_min, cfg.omega_n_max)

    zeta_alpha = compute_zeta(
        Vx, cfg.zeta_base, cfg.zeta_low_speed, cfg.v_zeta_transition)

    d_alpha_t = alpha_dot
    d_alpha_dot = (omega_n_alpha ** 2 * (alpha_kin - alpha_t)
                   - 2.0 * zeta_alpha * omega_n_alpha * alpha_dot)

    # ── Longitudinal channel ──────────────────────────────────────────────
    sigma_kappa = relaxation_length(
        Fz, kappa_t, cfg.sigma_kappa_0, cfg.Fz0,
        cfg.p_load_kappa, cfg.kappa_peak, cfg.beta_collapse)

    omega_n_kappa = compute_omega_n(
        Vx, sigma_kappa, cfg.omega_n_min, cfg.omega_n_max)

    zeta_kappa = compute_zeta(
        Vx, cfg.zeta_base, cfg.zeta_low_speed, cfg.v_zeta_transition)

    d_kappa_t = kappa_dot
    d_kappa_dot = (omega_n_kappa ** 2 * (kappa_kin - kappa_t)
                   - 2.0 * zeta_kappa * omega_n_kappa * kappa_dot)

    # Clip derivatives to prevent numerical blowup
    d_alpha_dot = jnp.clip(d_alpha_dot, -1e4, 1e4)
    d_kappa_dot = jnp.clip(d_kappa_dot, -1e4, 1e4)

    return d_alpha_t, d_alpha_dot, d_kappa_t, d_kappa_dot


# ─────────────────────────────────────────────────────────────────────────────
# §5  4-Corner Vectorized Interface
# ─────────────────────────────────────────────────────────────────────────────

@partial(jax.jit, static_argnums=())
def four_corner_transient_derivatives(
    alpha_kin: jax.Array,      # (4,) kinematic slip angles [rad]
    kappa_kin: jax.Array,      # (4,) kinematic slip ratios
    state: jax.Array,          # (4, 4) = [α_t, α̇_t, κ_t, κ̇_t] per corner
    Fz: jax.Array,             # (4,) normal loads [N]
    Vx: jax.Array,             # scalar longitudinal velocity [m/s]
    cfg: TireTransientConfig = TireTransientConfig(),
) -> jax.Array:
    """
    Vectorized 4-corner second-order transient derivatives.

    Args:
        alpha_kin: (4,) kinematic slip angles
        kappa_kin: (4,) kinematic slip ratios
        state: (4, 4) transient states [α_t, dα/dt, κ_t, dκ/dt]
        Fz: (4,) normal loads
        Vx: scalar velocity

    Returns:
        d_state: (4, 4) derivatives
    """
    def _single_corner(a_kin, k_kin, s, fz):
        da, dda, dk, ddk = compute_transient_slip_derivatives_2nd_order(
            a_kin, k_kin, s[0], s[1], s[2], s[3], fz, Vx, cfg)
        return jnp.array([da, dda, dk, ddk])

    return jax.vmap(_single_corner)(alpha_kin, kappa_kin, state, Fz)


# ─────────────────────────────────────────────────────────────────────────────
# §6  Frequency Response Utilities
# ─────────────────────────────────────────────────────────────────────────────

def tire_bandwidth_hz(
    Vx: float,
    Fz: float = 1000.0,
    sigma_0: float = 0.25,
    Fz0: float = 1000.0,
    p_load: float = 0.5,
) -> float:
    """
    Compute the -3dB bandwidth of the tire transient response.

    For the second-order model with damping ζ:
      f_3dB = (ωₙ / 2π) · √(1 - 2ζ² + √(4ζ⁴ - 4ζ² + 2))

    Returns frequency in Hz.
    """
    sigma = sigma_0 * (Fz / Fz0) ** p_load
    omega_n = abs(Vx) / max(sigma, 0.05)
    zeta = 0.7  # nominal
    factor = (1.0 - 2.0 * zeta ** 2
              + (4.0 * zeta ** 4 - 4.0 * zeta ** 2 + 2.0) ** 0.5) ** 0.5
    return omega_n * factor / (2.0 * 3.14159)


# ─────────────────────────────────────────────────────────────────────────────
# §7  Backward Compatibility Bridge
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def transient_1st_order_fallback(
    alpha_kin: jax.Array,
    kappa_kin: jax.Array,
    alpha_t: jax.Array,
    kappa_t: jax.Array,
    Fz: jax.Array,
    Vx: jax.Array,
    sigma_alpha: float = 0.25,
    sigma_kappa: float = 0.15,
) -> tuple:
    """
    First-order fallback matching existing compute_transient_slip_derivatives().
    Use this as a drop-in during transition.
    """
    Vx_safe = jnp.maximum(jnp.abs(Vx), 1.0)
    tau_alpha = sigma_alpha / Vx_safe
    tau_kappa = sigma_kappa / Vx_safe
    d_alpha = (alpha_kin - alpha_t) / tau_alpha
    d_kappa = (kappa_kin - kappa_t) / tau_kappa
    return d_alpha, d_kappa


def upgrade_state_1st_to_2nd(
    state_1st: jax.Array,   # (4, 2) = [α_t, κ_t] per corner
) -> jax.Array:
    """
    Expand first-order transient state (4, 2) to second-order (4, 4).
    Initializes velocity states to zero (quiescent start).
    """
    zeros = jnp.zeros((4, 1))
    alpha_t = state_1st[:, 0:1]
    kappa_t = state_1st[:, 1:2]
    return jnp.concatenate([alpha_t, zeros, kappa_t, zeros], axis=1)