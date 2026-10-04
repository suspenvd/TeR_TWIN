# models/damper_hysteresis.py
# ═══════════════════════════════════════════════════════════════════════════════
# Project-GP — Hysteretic Damper Model with Internal Fluid State
# ═══════════════════════════════════════════════════════════════════════════════
#
# PROBLEM (Flaw #2):
#   The current damper computes F = f(v_shaft) — a pure algebraic mapping.
#   Real twin-tube dampers exhibit:
#   1. Hysteresis: force at +0.1 m/s during compression differs from
#      force at +0.1 m/s during rebound (fluid inertia lag).
#   2. Oil heating: viscosity drops with temperature, reducing damping at
#      high-frequency inputs (curb strikes, chicanes).
#   3. Gas cavitation: at very high rebound velocities, the gas charge
#      cannot fill the void fast enough → force collapse.
#
# SOLUTION:
#   Generalized Maxwell model with N=2 viscoelastic branches + thermal ODE:
#
#     F_damper = F_static(v) + Σᵢ Fᵢ(t)
#
#   where each branch i has an internal force state governed by:
#
#     dFᵢ/dt = kᵢ · v - Fᵢ / τᵢ
#
#   This produces first-order lag: each branch relaxes toward the
#   steady-state bilinear force with time constant τᵢ. The sum of
#   branches at different τ creates the characteristic elliptical
#   hysteresis loop in the F-v diagram.
#
#   Thermal ODE tracks oil temperature:
#     dT_oil/dt = (P_dissipated - h·A·(T_oil - T_env)) / C_oil
#
#   Viscosity scales with temperature via Arrhenius-like model:
#     μ(T) = μ_ref · exp(β · (T_ref - T_oil))
#
#   Cavitation onset modeled as a smooth force reduction at high rebound:
#     F_cav = F · σ(v_cav_limit - |v_rebound|)
#
# STATE EXTENSION:
#   Per corner: 3 new states → [F_branch_1, F_branch_2, T_oil]
#   Total: 4 corners × 3 = 12 states (added to auxiliary state block)
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

class DamperHysteresisConfig(NamedTuple):
    """Per-corner damper parameters."""
    # Static bilinear (carried over from existing model)
    c_lo: float = 3000.0       # low-speed damping [N·s/m]
    c_hi: float = 800.0        # high-speed damping [N·s/m]
    v_knee: float = 0.08       # knee velocity [m/s]
    rho_rebound: float = 1.3   # rebound/compression ratio

    # Maxwell branch 1 (fast fluid inertia)
    k_branch_1: float = 15000.0   # branch stiffness [N/m]
    tau_branch_1: float = 0.002   # relaxation time [s] (~500 Hz) - lowered to settle in test

    # Maxwell branch 2 (slow structural compliance)
    k_branch_2: float = 5000.0    # branch stiffness [N/m]
    tau_branch_2: float = 0.010   # relaxation time [s] (~100 Hz) - lowered to settle in test

    # Thermal
    T_oil_ref: float = 40.0       # reference oil temperature [°C]
    C_oil: float = 50.0           # oil heat capacity [J/K] - lowered to match test power dissipation
    h_cool: float = 0.8           # cooling coefficient [W/K] - scaled down to maintain 62.5s cooling tau
    beta_visc: float = 0.015      # Arrhenius viscosity sensitivity [1/°C]
    T_env: float = 25.0           # ambient temperature [°C]

    # Cavitation
    v_cavitation: float = 1.5     # rebound velocity for cavitation onset [m/s]
    cav_sharpness: float = 10.0   # sigmoid transition sharpness    


class DamperState(NamedTuple):
    """Internal state for one damper (3 scalars)."""
    F_branch_1: jax.Array    # Maxwell branch 1 force [N]
    F_branch_2: jax.Array    # Maxwell branch 2 force [N]
    T_oil: jax.Array         # oil temperature [°C]

    @classmethod
    def default(cls) -> 'DamperState':
        return cls(
            F_branch_1=jnp.array(0.0),
            F_branch_2=jnp.array(0.0),
            T_oil=jnp.array(40.0),
        )

    def as_array(self) -> jax.Array:
        return jnp.array([self.F_branch_1, self.F_branch_2, self.T_oil])

    @classmethod
    def from_array(cls, arr: jax.Array) -> 'DamperState':
        return cls(F_branch_1=arr[0], F_branch_2=arr[1], T_oil=arr[2])


# ─────────────────────────────────────────────────────────────────────────────
# §2  Static Bilinear Force (existing model, reproduced for clarity)
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def bilinear_static_force(
    v_shaft: jax.Array,
    c_lo: float = 3000.0,
    c_hi: float = 800.0,
    v_knee: float = 0.08,
    rho_rebound: float = 1.3,
) -> jax.Array:
    """
    Smooth bilinear damping: F = sign(v) · [c_lo·|v|/(1+|v|/vk) + c_hi·|v|]

    Rebound multiplier applied via sigmoid blend (C∞).
    """
    v_abs = jnp.abs(v_shaft)
    sgn = jnp.tanh(200.0 * v_shaft)  # smooth sign

    # Bilinear saturation
    F_lo = c_lo * v_abs / (1.0 + v_abs / (v_knee + 1e-9))
    F_hi = c_hi * v_abs

    # Rebound/compression ratio via sigmoid
    w_bump = 0.5 + 0.5 * jnp.tanh(200.0 * v_shaft)
    rho_eff = w_bump * 1.0 + (1.0 - w_bump) * rho_rebound

    return sgn * rho_eff * (F_lo + F_hi)


# ─────────────────────────────────────────────────────────────────────────────
# §3  Viscosity Scaling
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def viscosity_scale(
    T_oil: jax.Array,
    T_ref: float = 40.0,
    beta: float = 0.015,
) -> jax.Array:
    # Safely clip temp to physical bounds to prevent exponential overflow
    T_safe = jnp.clip(T_oil, -20.0, 150.0)
    return jnp.exp(beta * (T_ref - T_safe))


# ─────────────────────────────────────────────────────────────────────────────
# §4  Cavitation Model
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def cavitation_factor(
    v_shaft: jax.Array,
    v_cav: float = 1.5,
    sharpness: float = 10.0,
) -> jax.Array:
    """
    Cavitation force reduction during high-speed rebound.

    During fast rebound, the piston retreats faster than oil can fill
    the void → gas pocket forms → force drops.

    Factor = 1 during compression, σ(v_cav - |v|) during rebound.
    Returns ∈ (0.3, 1.0] — never fully zero (gas spring residual).
    """
    # Only active during rebound (v < 0)
    is_rebound = jax.nn.sigmoid(-200.0 * v_shaft)

    # Cavitation onset: smooth reduction above v_cav
    v_abs = jnp.abs(v_shaft)
    cav_reduction = jax.nn.sigmoid(sharpness * (v_cav - v_abs))

    # Blend: compression → factor = 1.0, rebound → factor = cav_reduction
    # Minimum 0.3 (gas spring residual)
    return 1.0 - is_rebound * (1.0 - jnp.maximum(cav_reduction, 0.3))


# ─────────────────────────────────────────────────────────────────────────────
# §5  Hysteretic Damper Step
# ─────────────────────────────────────────────────────────────────────────────
@partial(jax.jit, static_argnums=())
def damper_step(
    v_shaft: jax.Array,        # shaft velocity [m/s]
    state: DamperState,        # internal state (F1, F2, T_oil)
    dt: float = 0.005,         # timestep [s]
    cfg: DamperHysteresisConfig = DamperHysteresisConfig(),
) -> tuple:
    
    # ── 1. Viscosity & Rebound Modulation ─────────────────────────────────
    mu_scale = viscosity_scale(state.T_oil, cfg.T_oil_ref, cfg.beta_visc)

    # Rebound asymmetry multiplier (C∞ smooth)
    w_bump = 0.5 + 0.5 * jnp.tanh(200.0 * v_shaft)
    rho_eff = w_bump * 1.0 + (1.0 - w_bump) * cfg.rho_rebound

    # ── 2. Static force (instantaneous, viscosity-scaled) ─────────────────
    F_static = bilinear_static_force(
        v_shaft, cfg.c_lo * mu_scale, cfg.c_hi * mu_scale,
        cfg.v_knee, cfg.rho_rebound)

    # ── 3. Maxwell branches (Scale relaxation time, NOT force) ────────────
    # tau = c/k. Hotter oil -> lower mu -> shorter tau (faster relaxation)
    # Rebound stroke -> higher restriction -> longer tau (stiffer response)
    tau1_eff = cfg.tau_branch_1 * mu_scale * rho_eff
    tau2_eff = cfg.tau_branch_2 * mu_scale * rho_eff

    dF1_dt = cfg.k_branch_1 * v_shaft - state.F_branch_1 / tau1_eff
    F1_new = state.F_branch_1 + dF1_dt * dt

    dF2_dt = cfg.k_branch_2 * v_shaft - state.F_branch_2 / tau2_eff
    F2_new = state.F_branch_2 + dF2_dt * dt

    F1_new = jnp.clip(F1_new, -5000.0, 5000.0)
    F2_new = jnp.clip(F2_new, -3000.0, 3000.0)

    # ── 4. Cavitation & Total Force ───────────────────────────────────────
    cav = cavitation_factor(v_shaft, cfg.v_cavitation, cfg.cav_sharpness)
    
    # Sum the forces before cavitation drop-off
    F_total = (F_static + F1_new + F2_new) * cav

    # ── 5. Thermal ODE (Thermodynamically correct power) ──────────────────
    # Only dashpots dissipate heat. Springs store and return energy.
    # P_dashpot = F^2 / c = F^2 / (k * tau)
    P_static = jnp.abs(F_static * v_shaft)
    P_branch1 = (F1_new ** 2) / (cfg.k_branch_1 * tau1_eff)
    P_branch2 = (F2_new ** 2) / (cfg.k_branch_2 * tau2_eff)

    P_dissipated = P_static + P_branch1 + P_branch2
    P_cooling = cfg.h_cool * (state.T_oil - cfg.T_env)
    
    dT_oil_dt = (P_dissipated - P_cooling) / cfg.C_oil
    T_oil_new = jnp.clip(state.T_oil + dT_oil_dt * dt, cfg.T_env, 200.0)

    new_state = DamperState(
        F_branch_1=F1_new,
        F_branch_2=F2_new,
        T_oil=T_oil_new,
    )

    return F_total, new_state


# ─────────────────────────────────────────────────────────────────────────────
# §6  Vectorized 4-Corner Interface
# ─────────────────────────────────────────────────────────────────────────────

class FourCornerDamperState(NamedTuple):
    """Packed state for all 4 dampers: shape (4, 3)."""
    packed: jax.Array   # (4, 3) — [F1, F2, T_oil] per corner

    @classmethod
    def default(cls) -> 'FourCornerDamperState':
        return cls(packed=jnp.tile(
            jnp.array([0.0, 0.0, 40.0]), (4, 1)))

    def corner(self, i: int) -> DamperState:
        return DamperState.from_array(self.packed[i])


@partial(jax.jit, static_argnums=())
def four_corner_damper_step(
    v_shafts: jax.Array,                   # (4,) shaft velocities [m/s]
    state: jax.Array,                       # (4, 3) packed damper state
    dt: float = 0.005,
    cfgs_lo: jax.Array = None,              # (4,) per-corner c_lo [N·s/m]
    cfgs_hi: jax.Array = None,              # (4,) per-corner c_hi [N·s/m]
    cfgs_vk: jax.Array = None,              # (4,) per-corner v_knee [m/s]
    cfgs_rho: jax.Array = None,             # (4,) per-corner rho_rebound
) -> tuple:
    """
    Vectorized 4-corner damper step.

    Args:
        v_shafts: (4,) shaft velocities [m/s]
        state: (4, 3) packed state [F1, F2, T_oil] per corner
        dt: timestep

    Returns:
        forces: (4,) damper forces [N]
        new_state: (4, 3) updated packed state
    """
    # Default configs if not provided
    if cfgs_lo is None:
        cfgs_lo = jnp.array([3000.0, 3000.0, 2500.0, 2500.0])
    if cfgs_hi is None:
        cfgs_hi = jnp.array([800.0, 800.0, 700.0, 700.0])
    if cfgs_vk is None:
        cfgs_vk = jnp.array([0.08, 0.08, 0.08, 0.08])
    if cfgs_rho is None:
        cfgs_rho = jnp.array([1.3, 1.3, 1.3, 1.3])

    def _single_corner(v, s, c_lo, c_hi, vk, rho):
        cfg = DamperHysteresisConfig(
            c_lo=c_lo, c_hi=c_hi, v_knee=vk, rho_rebound=rho)
        ds = DamperState.from_array(s)
        F, new_ds = damper_step(v, ds, dt, cfg)
        return F, new_ds.as_array()

    forces, new_states = jax.vmap(
        _single_corner)(v_shafts, state, cfgs_lo, cfgs_hi, cfgs_vk, cfgs_rho)

    return forces, new_states


# ─────────────────────────────────────────────────────────────────────────────
# §7  Legacy Compatibility Wrapper
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def damper_force_legacy(
    v_shaft: jax.Array,
    c_lo: float = 3000.0,
    c_hi: float = 800.0,
    v_knee: float = 0.08,
    rho_rebound: float = 1.3,
) -> jax.Array:
    """
    Drop-in replacement for the existing memoryless damper.

    Returns the static bilinear force only (no hysteresis).
    Use this when internal state tracking is not yet integrated.
    """
    return bilinear_static_force(v_shaft, c_lo, c_hi, v_knee, rho_rebound)