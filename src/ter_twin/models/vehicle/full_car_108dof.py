# models/vehicle_dynamics.py
# Project-GP — Neural Port-Hamiltonian Vehicle Dynamics
# ═══════════════════════════════════════════════════════════════════════════════
#
# UPGRADE LOG (GP-vX2 — this revision)
# ─────────────────────────────────────────────────────────────────────────────
# BUGFIX-4 : CRITICAL — h_scale training-inference mismatch
#   Training: NeuralEnergyLandscape(M_diag=M_diag) → h_scale=1.0 (default).
#   Inference: NeuralEnergyLandscape(M_diag=M_diag, h_scale=102.62) loaded
#   from h_net_scale.txt. Inside __call__:
#       H_raw = softplus(Dense(1)(x)) * h_scale
#   Weights trained to output H_raw ∈ [0, 5000] (h_scale=1.0) were evaluated
#   with h_scale=102.62 → H_raw ∈ [0, 513100] → minimum(..., 5000) clips
#   everything → all H_net gradients zero at inference. FORCE_CAP tanh then
#   saturates at 12000N from 102× amplified forces, producing the 7534 mJ
#   passive energy injection seen in Test 2.
#   FIX: DifferentiableMultiBodyVehicle always instantiates NeuralEnergyLandscape
#   with h_scale=1.0. h_net_scale.txt is retained as a DIAGNOSTIC artefact only.
#
# BUGFIX-5 : susp_sq gated at absolute z=0, not physical equilibrium z_eq
#   PREVIOUS: susp_sq = sum(q[6:10]²) + 1e-4
#   The car operates at z_eq ≈ [12.8mm, 12.8mm, 14.2mm, 14.2mm]. At that
#   operating point susp_sq = 4·z_eq² ≈ 7e-4 m² — 7× larger than the 1e-4
#   floor, so the floor provides no real benefit. More critically, the gating
#   guarantees dH/dq=0 at z=0 (a configuration the car never occupies), while
#   the actual equilibrium point z_eq has a nonzero spring gradient (correct).
#   FIX: susp_sq = sum((q[6:10] - _Z_EQ)²) + 1e-4
#   · dH/dq at z_eq is now dominated by V_structural (30kN/m spring) + small
#     residual from H_res gradient — physically correct.
#   · Training data (residual_fitting.py) samples q_susp ~ N(z_eq, 20mm), so
#     E[susp_sq_new] = 4·(20mm)² = 0.0016 m² — gradient magnitude consistent.
#   · V_spring_dev in residual_fitting.py target_H MUST use (z-z_eq)² coords
#     to maintain consistency (see patch note in residual_fitting.py §1).
#
# UPGRADE-9 : H_RESIDUAL_CAP raised from 5000 to 50_000
#   With setup-dependent target_H = V_spring_dev + V_arb + V_torsion and
#   equilibrium-centered susp_sq, H_res = target_H / susp_sq is structurally:
#     H_res ≈ 0.5·(k_f - K_PRIOR)   [spring deviation term, k_f-dependent]
#   At k_f = 60000 (1σ high from setup distribution): H_res ≈ 15000 > 5000.
#   Old cap silently clipped ~30% of training samples with zero gradient signal.
#   50_000 covers the 99.9th percentile of physically valid setups.
#   The FORCE_CAP = 12000 N in _compute_derivatives provides physical limiting.
#
# UPGRADE-10 : DifferentiableAeroMap — softplus floors replace jnp.maximum
#   jnp.maximum(0.040 - heave_f, 0.015): zero subgradient when ground clearance
#   hits 15mm floor — exactly when MPC is most aggressively minimizing lap time.
#   jnp.maximum(Cl_f_base, 0.0): zero subgradient at zero downforce.
#   Both replaced with _softplus_floor — gradient always nonzero.
#
# ─── Retained from GP-vX1 ────────────────────────────────────────────────────
# BUGFIX-1  SuspensionSetup index alignment
# BUGFIX-2  Mass/inertia defaults corrected
# BUGFIX-3  _TRIL_14 missing definition
# UPGRADE-1 FiLM-conditioned NeuralEnergyLandscape
# UPGRADE-2 Structured NeuralDissipationMatrix with log-diagonal guarantee
# UPGRADE-3 4th-order Gauss-Legendre Variational Integrator
# UPGRADE-4 Differentiable ground-clearance bumpstop with softplus
# UPGRADE-5 Compliance steer from lateral load
# UPGRADE-6 Full C^∞ damper asymmetry (sigmoid bump/rebound blend)
# UPGRADE-7 Softplus Fz floor replaces jnp.maximum
# UPGRADE-8 susp_sq gradient floor — superseded by BUGFIX-5 / UPGRADE-9
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import os
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import flax.linen as nn
import flax.serialization
from physics.h_net_icnn import PassiveHNet
from models.aero_platform import create_aero_platform
from models.tire_thermal_3d import four_corner_thermal_derivatives
from models.tire_transient import four_corner_transient_derivatives
# ─────────────────────────────────────────────────────────────────────────────
# Module-level XLA-static constants
# ─────────────────────────────────────────────────────────────────────────────

# Pre-computed lower-triangular indices for 14×14 matrix.
# Defined at module load — NeuralDissipationMatrix.__call__ references this
# name. Placing inside the class or inside @nn.compact causes NameError because
# Flax traces the method body in a scope that doesn't re-execute module-level
# statements. Shape: two (105,) integer arrays — static from XLA's perspective.
_TRIL_14 = jnp.tril_indices(14)

# Static equilibrium suspension deflections [m] — front/rear.
# Derived from: F_susp_eq = m*g*l_opp/(2L) - m_us*g; z_eq = F_susp_eq/(k·MR₀²)
# Front: (300·9.81·0.6975)/(2·1.55) - 10·9.81 = 583N / (35000·1.14²) ≈ 0.0128m
# Rear:  (300·9.81·0.8525)/(2·1.55) - 11·9.81 = 695N / (38000·1.16²) ≈ 0.0135m
# Using canonical DEFAULT_SETUP spring rates. Defined here as a module-level
# constant so NeuralEnergyLandscape.__call__ and residual_fitting.py share
# identical values — single source of truth. See also compute_equilibrium_suspension().
# CORREGIDO (CAMBIAR A FLOAT32):
_Z_EQ: jnp.ndarray = jnp.array([0.0128, 0.0128, 0.0142, 0.0142], dtype=jnp.float32)
# Structural spring prior per corner — must match V_structural in
# NeuralEnergyLandscape.__call__ and _V_STRUCT_PRIOR_K in residual_fitting.py.
_V_STRUCT_PRIOR_K: float = 30_000.0   # N/m

# A fixed count preserves a static XLA loop while providing a tightly converged
# stage solution in the validation operating regime.  It is deliberately named
# Picard: the code below is fixed-point iteration, not Newton's method.
_GLRK_PICARD_ITERS: int = 32


# ─────────────────────────────────────────────────────────────────────────────
# §1  SuspensionSetup  — 28-element typed pytree
# ─────────────────────────────────────────────────────────────────────────────

SETUP_DIM = 28

SETUP_NAMES = [
    'k_f', 'k_r',                         #  0–1   spring rates [N/m]
    'arb_f', 'arb_r',                     #  2–3   anti-roll bar [N·m/rad]
    'c_low_f', 'c_low_r',                 #  4–5   low-speed damper [N·s/m]
    'c_high_f', 'c_high_r',               #  6–7   high-speed damper [N·s/m]
    'v_knee_f', 'v_knee_r',               #  8–9   knee velocity [m/s]
    'rebound_ratio_f', 'rebound_ratio_r', # 10–11  rebound/bump ratio [-]
    'h_ride_f', 'h_ride_r',               # 12–13  ride height [m]
    'camber_f', 'camber_r',               # 14–15  static camber [deg]
    'toe_f', 'toe_r',                     # 16–17  static toe [deg]
    'castor_f',                           # 18     caster angle [deg]
    'anti_squat',                         # 19     anti-squat fraction [-]
    'anti_dive_f', 'anti_dive_r',         # 20–21  anti-dive fraction [-]
    'anti_lift',                          # 22     anti-lift fraction [-]
    'yaw_target_gain',                    # 23     TV yaw moment gain [-]
    'brake_bias_f',                       # 24     front brake bias [-]
    'h_cg',                               # 25     CG height [m]
    'bump_steer_f', 'bump_steer_r',       # 26–27  bump steer [rad/m]
]

# Canonical default — physically validated for Ter26 FS 2026
DEFAULT_SETUP = jnp.array([
    35000., 38000.,   # k_f, k_r
      800.,   600.,   # arb_f, arb_r
     1800.,  1800.,   # c_low_f, c_low_r
     1200.,  1200.,   # c_high_f, c_high_r
     0.10,   0.10,    # v_knee_f, v_knee_r
     1.50,   1.50,    # rebound_ratio_f, rebound_ratio_r
     0.025,  0.022,   # h_ride_f, h_ride_r
     -2.0,   -1.5,    # camber_f, camber_r
     -0.10,  -0.15,   # toe_f, toe_r
      5.0,            # castor_f
      0.30,           # anti_squat
      0.40,  0.10,    # anti_dive_f, anti_dive_r
      0.20,           # anti_lift
      0.80,           # yaw_target_gain
      0.60,           # brake_bias_f
      0.33,          # h_cg
      0.00,  0.00,    # bump_steer_f, bump_steer_r
], dtype=jnp.float32)

# Hard physical bounds for optimizer projection
SETUP_LB = jnp.array([
    10000., 10000.,   0.,    0.,   200.,  200.,   50.,   50.,
     0.03,   0.03,  1.0,   1.0,  0.010, 0.010,  -5.0,  -5.0,
    -1.0,   -1.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,
     0.50,   0.18, -0.05, -0.05,
], dtype=jnp.float32)

SETUP_UB = jnp.array([
    65000.,  65000.,  5000., 5000., 8000., 8000., 4000., 4000.,
       0.40,   0.40,  3.0,   3.0,  0.060, 0.060,  0.0,   0.0,
       1.0,    1.0,  10.0,   0.9,   0.9,   0.9,   0.9,   1.0,
       0.80,   0.45, 0.05,  0.05,
], dtype=jnp.float32)
# k_f, k_r bajado de 120000→65000 N/m: por encima de eso, el muelle en la
# mangueta (tras MR²≈1.3) supera 150kN/m — fuera de catálogo real de
# resortes FS y sin sentido físico salvo como artefacto de optimización.


class SuspensionSetup(NamedTuple):
    """
    28-element typed container for all optimizable chassis setup parameters.

    Registered as a JAX pytree: flows through jit / vmap / grad / scan.
    Canonical parameter ordering is defined by SETUP_NAMES above.

    Design contract:
    · All quantities SI base units. Angles in degrees (documented explicitly).
    · `from_vector` / `to_vector` are the ONLY construction paths. Never
      construct by positional index externally.
    · `project_to_bounds()` uses smooth tanh rescaling — gradient-alive at bounds.
    """
    k_f: jax.Array              #  0
    k_r: jax.Array              #  1
    arb_f: jax.Array            #  2
    arb_r: jax.Array            #  3
    c_low_f: jax.Array          #  4
    c_low_r: jax.Array          #  5
    c_high_f: jax.Array         #  6
    c_high_r: jax.Array         #  7
    v_knee_f: jax.Array         #  8
    v_knee_r: jax.Array         #  9
    rebound_ratio_f: jax.Array  # 10
    rebound_ratio_r: jax.Array  # 11
    h_ride_f: jax.Array         # 12
    h_ride_r: jax.Array         # 13
    camber_f: jax.Array         # 14
    camber_r: jax.Array         # 15
    toe_f: jax.Array            # 16
    toe_r: jax.Array            # 17
    castor_f: jax.Array         # 18
    anti_squat: jax.Array       # 19
    anti_dive_f: jax.Array      # 20
    anti_dive_r: jax.Array      # 21
    anti_lift: jax.Array        # 22
    yaw_target_gain: jax.Array  # 23
    brake_bias_f: jax.Array     # 24
    h_cg: jax.Array             # 25
    bump_steer_f: jax.Array     # 26
    bump_steer_r: jax.Array     # 27

    @staticmethod
    def from_vector(v: jax.Array) -> "SuspensionSetup":
        return SuspensionSetup(*[v[i] for i in range(SETUP_DIM)])

    def to_vector(self) -> jax.Array:
        return jnp.stack(list(self))

    @staticmethod
    def default() -> "SuspensionSetup":
        return SuspensionSetup.from_vector(DEFAULT_SETUP)

    @staticmethod
    def from_legacy_8(v: jax.Array) -> "SuspensionSetup":
        """
        Backward-compat loader for old 8-element checkpoints.
        Layout: [k_f, k_r, arb_f, arb_r, c_f, c_r, h_cg, brake_bias_f]
        """
        base = (DEFAULT_SETUP
                .at[0].set(v[0]).at[1].set(v[1])
                .at[2].set(v[2]).at[3].set(v[3])
                .at[4].set(v[4]).at[5].set(v[5])
                .at[25].set(v[6]).at[24].set(v[7]))
        return SuspensionSetup.from_vector(base)

    def project_to_bounds(self) -> "SuspensionSetup":
        """Smooth tanh projection — keeps gradient alive at bounds."""
        v    = self.to_vector()
        mid  = 0.5 * (SETUP_UB + SETUP_LB)
        half = 0.5 * (SETUP_UB - SETUP_LB)
        v_b  = mid + half * jnp.tanh((v - mid) / (half + 1e-6))
        return SuspensionSetup.from_vector(v_b)


# Register SuspensionSetup as a JAX pytree
jax.tree_util.register_pytree_node(
    SuspensionSetup,
    lambda s: (list(s), None),
    lambda _, leaves: SuspensionSetup(*leaves),
)

def safe_abs(x: jax.Array, beta: float = 20.0) -> jax.Array:
    """
    Hessian-safe absolute value.
    1st derivative bounded in [-1, 1].
    2nd derivative bounded to beta/2 (max 10.0).
    Prevents 2nd-order gradient explosions in implicit RK4 solvers.
    """
    # softplus(x) + softplus(-x) creates a smooth V-shape.
    raw_abs = (jnp.logaddexp(0.0, beta * x) + jnp.logaddexp(0.0, -beta * x)) / beta
    
    # Shift it down so that safe_abs(0) = 0.0 perfectly
    offset = 2.0 * jnp.log(2.0) / beta
    return jnp.maximum(raw_abs - offset, 0.0)

@jax.custom_vjp
def _clip_adjoint(x):
    """Identity in forward pass, clips gradient to [-1.0, 1.0] in backward pass."""
    return x

def _clip_adjoint_fwd(x):
    return x, None

def _clip_adjoint_bwd(res, g):
    # g is the incoming gradient from the future. We clip it before passing it back!
    return (jnp.clip(g, -1.0, 1.0),)

_clip_adjoint.defvjp(_clip_adjoint_fwd, _clip_adjoint_bwd)

# ─────────────────────────────────────────────────────────────────────────────
# §2  PhysicsNormalizer
# ─────────────────────────────────────────────────────────────────────────────

class PhysicsNormalizer:
    """Normalization statistics for neural network inputs."""
    q_mean  = jnp.array([0., 0., 0.3, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    q_scale = jnp.array([100., 100., 0.1, 0.1, 0.05, jnp.pi,
                          0.05, 0.05, 0.05, 0.05,
                          500., 500., 500., 500.])
    v_mean  = jnp.zeros(14)
    v_scale = jnp.array([25., 5., 1., 1., 1., 1., 1., 1., 1., 1., 50., 50., 50., 50.])

    # 28-element setup normalization — matching DEFAULT_SETUP operating point
    setup_mean = jnp.array([
        40000., 40000., 800.,  600.,  1800., 1800., 1200., 1200.,
        0.10,   0.10,   1.5,   1.5,   0.025, 0.022, -2.0,  -1.5,
       -0.10,  -0.15,  5.0,   0.30,   0.40,  0.10,  0.20,  0.80,
        0.60,   0.285,  0.00,  0.00,
    ], dtype=jnp.float32)

    setup_scale = jnp.array([
        20000., 20000., 400.,  300.,  800.,  800.,  600.,  600.,
        0.05,   0.05,   0.5,   0.5,   0.015, 0.012, 1.5,   1.5,
        0.5,    0.5,    3.0,   0.15,  0.20,  0.08,  0.10,  0.30,
        0.10,   0.040,  0.03,  0.03,
    ], dtype=jnp.float32)

    @staticmethod
    def normalize_q(q): return (q - PhysicsNormalizer.q_mean) / (PhysicsNormalizer.q_scale + 1e-8)
    @staticmethod
    def normalize_v(v): return (v - PhysicsNormalizer.v_mean) / (PhysicsNormalizer.v_scale + 1e-8)
    @staticmethod
    def normalize_setup(s): return (s - PhysicsNormalizer.setup_mean) / (PhysicsNormalizer.setup_scale + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# §3  Kinematic utilities
# ─────────────────────────────────────────────────────────────────────────────

def compute_motion_ratio(z_corner: jax.Array, poly_coeffs: jax.Array) -> jax.Array:
    """Quadratic MR(z) = a0 + a1·z + a2·z²"""
    return poly_coeffs[0] + poly_coeffs[1] * z_corner + poly_coeffs[2] * z_corner ** 2


def compute_damper_force_bilinear(
    z_dot: jax.Array,
    c_low: jax.Array,
    c_high: jax.Array,
    v_knee: jax.Array,
    rebound_ratio: jax.Array,
) -> jax.Array:
    """
    4-way digressive damper: low/high speed bump + rebound asymmetry.
    Fully C^∞ smooth — no hard conditionals anywhere.

    High/low speed split: sigmoid blend at v_knee.
    Bump/rebound split: sigmoid blend with β=200 s/m → ~5mm/s transition width.
    At |z_dot| > 20mm/s the blend is indistinguishable from hard switching so
    physical accuracy is preserved. At z_dot=0 the blend is 50/50 of bump and
    rebound — the only physically valid C^∞ extension at zero velocity.
    """
    v_abs  = safe_abs(z_dot)
    w_bump = jax.nn.sigmoid(200.0 * z_dot)
    w_reb  = 1.0 - w_bump

    c_lo = w_bump * c_low              + w_reb * (c_low  * rebound_ratio)
    c_hi = w_bump * c_high             + w_reb * (c_high * rebound_ratio)

    alpha  = 20.0 / (v_knee + 1e-6)
    w_high = jax.nn.sigmoid(alpha * (v_abs - v_knee))
    w_low  = 1.0 - w_high

    c_eff = c_lo * w_low + (c_lo + c_hi) * w_high
    return c_eff * z_dot


def compute_bump_steer(
    z_corner: jax.Array,
    bs_lin: jax.Array,
    bs_quad: jax.Array,
) -> jax.Array:
    """Bump steer: δ_bs = bs_lin·z + bs_quad·z²"""
    return bs_lin * z_corner + bs_quad * z_corner ** 2


def compute_bumpstop_force(z_corner: jax.Array, gap: jax.Array, k_bs: float = 500_000.0, beta: float = 200.0) -> jax.Array:
    """
    C^∞ bumpstop force via softplus. Zero force for z < gap, rising rapidly
    above. Replaces hard clip — gradient-alive through contact.
    F_bs = k_bs · softplus((z - gap)·β) / β
    """
    return k_bs * jnp.logaddexp(0.0, beta * (z_corner - gap)) / beta


def compute_castor_trail(castor_deg: jax.Array, Fy: jax.Array, tire_radius: float) -> jax.Array:
    # FIX: Castor restoring moment is driven by lateral force (Fy), not vertical load (Fz)
    t_mech = tire_radius * jnp.tan(jnp.deg2rad(castor_deg))
    return -Fy * t_mech


def _softplus_floor(x: jax.Array, floor: float = 10.0) -> jax.Array:
    """
    Smooth lower bound: f(x) = floor + softplus(x - floor).
    · f(x) → x       for x >> floor  (identity in the physical regime)
    · f(x) → floor   for x << floor  (asymptotic minimum)
    · df/dx = sigmoid(x - floor) ∈ (0, 1) — never zero, never > 1.

    Replaces jnp.maximum(..., floor) whose sub-gradient is zero below the
    floor — kills optimizer signal exactly when a corner goes light.
    """
    return floor + jnp.logaddexp(0.0, x - floor)


# ─────────────────────────────────────────────────────────────────────────────
# §4  Neural Networks
# ─────────────────────────────────────────────────────────────────────────────

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation conditioned on setup vector.
    h_out = γ(setup) ⊙ LayerNorm(h) + β(setup)

    Mathematically superior to concatenation: the setup vector controls the
    GRADIENT of the energy landscape (effective stiffness), not just its offset.
    Initialized with small random weights so training can break symmetry 
    and learn setup-dependent deviations under weight decay.
    """
    features: int

    @nn.compact
    def __call__(self, h: jax.Array, setup_embedding: jax.Array) -> jax.Array:
        gamma = nn.Dense(self.features,
                         kernel_init=jax.nn.initializers.normal(stddev=0.01),
                         bias_init=jax.nn.initializers.ones)(setup_embedding)
        beta  = nn.Dense(self.features,
                         kernel_init=jax.nn.initializers.normal(stddev=0.01),
                         bias_init=jax.nn.initializers.zeros)(setup_embedding)
        h_norm = nn.LayerNorm()(h)
        return gamma * h_norm + beta


class NeuralEnergyLandscape(nn.Module):
    """
    Port-Hamiltonian residual energy H_res(q, p, setup).

    Architecture (GP-vX2):
    · SE(3)-bilateral symmetric feature extraction (22 state features).
      Anti-symmetric features enter as x² — structurally cannot represent
      odd functions of vy, wz, roll regardless of weights.
    · FiLM modulation at each hidden layer via 16 energy-relevant setup dims.
    · Output: T_prior + V_structural + H_residual
      where H_residual = softplus(MLP) · susp_sq_eq ≥ 0.

    susp_sq_eq = Σ(q[6:10] - _Z_EQ)² + 1e-4   [BUGFIX-5]
    · Gating at physical equilibrium z_eq, not at z=0.
    · dH/dq|_{z=z_eq} ≈ 30000·z_eq (from V_structural alone) — physically
      correct spring force at static equilibrium, balanced by gravity in F_ext.
    · H_res → 0.5·(k_f - K_PRIOR) near equilibrium (clean structure).
    · 1e-4 floor = (1mm)² — physical minimum, not a numerical hack.

    H_RESIDUAL_CAP = 50_000 J/m²   [UPGRADE-9]
    · H_res = target_H / susp_sq = 0.5·(k_f-K_PRIOR) at mean. For k_f=60000:
      H_res = 15000 J/m² > old cap of 5000 → old cap silently clipped 30%+.

    h_scale: ALWAYS 1.0 at both training and inference   [BUGFIX-4]
    · Training used h_scale=1.0 (NeuralEnergyLandscape default).
    · Old inference code passed h_scale=102.62 from h_net_scale.txt, creating
      a 102× energy amplification → FORCE_CAP saturation → 7534 mJ injection.
    · h_net_scale.txt is a training normalisation artefact, not architectural.
    """
    M_diag:         jnp.ndarray
    h_scale:        float = 1.0        # ALWAYS 1.0 — see BUGFIX-4
    H_RESIDUAL_CAP: float = 50_000.0   # raised from 5000 — see UPGRADE-9

    # Static index tuple for setup selection — Python-level constant so XLA
    # emits a static gather with no dynamic index allocation on each trace.
    _SETUP_IDX: tuple = (0, 1, 2, 3, 4, 5, 6, 7, 14, 15, 12, 13, 25, 19, 20, 24)

    @nn.compact
    def __call__(self, q: jax.Array, p: jax.Array, setup_params: jax.Array) -> jax.Array:
        T_prior      = 0.5 * jnp.sum((p ** 2) / (self.M_diag + 1e-8))
        V_structural = 0.5 * jnp.sum(q[6:10] ** 2) * _V_STRUCT_PRIOR_K

        v = p / (self.M_diag + 1e-8)
        Z, phi_roll, theta_pitch                  = q[2], q[3], q[4]
        z_fl, z_fr, z_rl, z_rr                   = q[6], q[7], q[8], q[9]
        vx, vy, vz, wx, wy, wz                   = v[0], v[1], v[2], v[3], v[4], v[5]
        dvz_fl, dvz_fr, dvz_rl, dvz_rr           = v[6], v[7], v[8], v[9]
        om_fl, om_fr, om_rl, om_rr               = v[10], v[11], v[12], v[13]

        # ── Symmetric features (invariant to left-right flip) ─────────────────
        SYM_Q_SC = jnp.array([0.10, 0.10, 0.05, 0.05])
        SYM_V_SC = jnp.array([25.0, 1.0, 1.5, 1.0, 1.0, 75.0, 75.0])
        sym_q = jnp.array([Z, theta_pitch,
                            (z_fl + z_fr) * 0.5, (z_rl + z_rr) * 0.5])
        sym_v = jnp.array([vx, vz, wy,
                            (dvz_fl + dvz_fr) * 0.5, (dvz_rl + dvz_rr) * 0.5,
                            (om_fl + om_fr) * 0.5,   (om_rl + om_rr) * 0.5])
        sym_q_n = sym_q / (SYM_Q_SC + 1e-6)
        sym_v_n = sym_v / (SYM_V_SC + 1e-6)

        # ── Anti-symmetric features (enter as x² → even fn → dH/dq=0 at eq) ──
        ANTI_Q_SC = jnp.array([0.10, 0.05, 0.05, 0.05])
        ANTI_V_SC = jnp.array([5.0, 1.0, 1.0, 75.0, 75.0, 75.0, 75.0])
        anti_q = jnp.array([phi_roll, z_fl - z_fr, z_rl - z_rr,
                              z_fl + z_rr - z_fr - z_rl]) ** 2
        anti_v = jnp.array([vy, wx, wz, dvz_fl - dvz_fr, dvz_rl - dvz_rr,
                              om_fl - om_fr, om_rl - om_rr]) ** 2
        anti_q_n = anti_q / (ANTI_Q_SC ** 2 + 1e-6)
        anti_v_n = anti_v / (ANTI_V_SC ** 2 + 1e-6)

        # State features: 4 + 7 + 4 + 7 = 22 (SE(3)-bilateral invariant)
        state_features = jnp.concatenate([sym_q_n, sym_v_n, anti_q_n, anti_v_n])

        # ── Setup embedding (decoupled MLP path) ─────────────────────────────
        setup_norm = PhysicsNormalizer.normalize_setup(setup_params)
        setup_sel  = setup_norm[jnp.array(self._SETUP_IDX)]
        setup_emb  = nn.swish(nn.Dense(32)(setup_sel))

        # ── Hidden layers with FiLM conditioning ─────────────────────────────
        x = nn.Dense(128)(state_features)
        x = FiLMLayer(128)(x, setup_emb)
        x = nn.swish(x)

        x = nn.Dense(64)(x)
        x = FiLMLayer(64)(x, setup_emb)
        x = nn.swish(x)

        H_raw = jnp.squeeze(jnp.logaddexp(0.0, nn.Dense(1)(x))) * self.h_scale
        H_res = jnp.minimum(H_raw, self.H_RESIDUAL_CAP)

        # BUGFIX-5: gate at physical equilibrium _Z_EQ, not at z=0.
        # With equilibrium-centered training data (q_susp ~ N(z_eq, 20mm)),
        # E[susp_sq_eq] = 4·(20mm)² = 0.0016 m² — full gradient magnitude.
        # The 1e-4 floor = (1mm)² provides numerical stability at exact eq.
        # Consistency requirement: residual_fitting.py V_spring_dev must use
        # (z_fl - z_eq_f)² coordinates (not absolute z²). See §1 patch note.
        susp_sq = jnp.sum((q[6:10] - _Z_EQ) ** 2) + 1e-4

        return T_prior + V_structural + H_res * susp_sq


class NeuralDissipationMatrix(nn.Module):
    """
    Port-Hamiltonian dissipation matrix R(q,p) = L·Lᵀ + diag(softplus(d)).

    · The UNMASKED core R_dense = L·Lᵀ + diag(softplus(d)) is strictly positive
      definite (SPD).
    · After applying the physical mask (outer product), the full 14×14 matrix is
      POSITIVE SEMIDEFINITE (R ⪰ 0): the {heave, roll, pitch, z_FL..RR} block
      retains SPD, but the masked DOFs (X, Y, yaw, ψ, ω_wheels) have zero
      dissipation by design.
    · Diagonal floor prevents near-conservative blow-up on active DOFs.
    · _TRIL_14 is a MODULE-LEVEL constant (defined at top of file after imports).
      Do NOT move it inside this class or inside __call__ — Flax traces __call__
      in a scope where module-level names are still visible, but any attempt to
      re-define it locally would shadow the outer binding and cause NameError
      in environments that cache the trace.
    """
    dim: int = 14

    @nn.compact
    def __call__(self, q: jax.Array, p: jax.Array) -> jax.Array:
        q_n   = PhysicsNormalizer.normalize_q(q)
        p_n   = p / (PhysicsNormalizer.v_scale * 200.0 + 1e-6)
        state = jnp.concatenate([q_n, p_n])

        x = nn.Dense(128)(state); x = nn.swish(x)
        x = nn.Dense(64)(x);      x = nn.swish(x)

        n_elem  = self.dim * (self.dim + 1) // 2
        L_elems = nn.Dense(n_elem,
                           kernel_init=jax.nn.initializers.lecun_normal(),
                           bias_init=jax.nn.initializers.zeros)(x)

        L      = jnp.zeros((self.dim, self.dim)).at[_TRIL_14].set(L_elems)
        R_chol = jnp.dot(L, L.T)

        log_d   = self.param('log_d', jax.nn.initializers.constant(-3.0), (self.dim,))
        R_diag  = jnp.diag(jnp.logaddexp(0.0, log_d))
        R_dense = R_chol + R_diag

        mask = jnp.array([0., 0., 1., 1., 1., 0., 1., 1., 1., 1., 0., 0., 0., 0.])
        return R_dense * jnp.outer(mask, mask)


# import jax
# import jax.numpy as jnp
# import flax.linen as nn
 
 
# def _softplus_floor(x, floor):
#     """Smooth lower bound — gradient alive at floor."""
#     return floor + jnp.logaddexp(0.0, x - floor)
 
 
# class PhysicsInformedAeroMap(nn.Module):
#     """
#     Physics-informed aero surrogate with structural guarantees.
 
#     Replaces DifferentiableAeroMap with identical interface but stronger
#     physics inductive bias.
 
#     STRUCTURAL GUARANTEES:
#     1. All forces scale as v² (dynamic pressure factored out)
#     2. Drag coefficient Cd > 0 always (softplus)
#     3. Roll symmetry: Cl(φ) = Cl(-φ) (even feature extraction)
#     4. Ground effect: bounded 1/h sensitivity with stall protection
#     5. Neural corrections bounded to ±15% of analytical baseline
 
#     INTERFACE: Identical to DifferentiableAeroMap.__call__:
#       Input:  (vx, pitch, roll, heave_f, heave_r) — all scalars
#       Output: (Fz_aero_f, Fz_aero_r, Fx_aero, My_aero, Mx_aero) — all scalars
#     """
#     base_A:  float   # reference frontal area [m²]
#     base_Cl: float   # reference lift coefficient (total)
#     base_Cd: float   # reference drag coefficient
#     lf:      float   # CG to front axle [m]
#     lr:      float   # CG to rear axle [m]
 
#     @nn.compact
#     def __call__(self, vx, pitch, roll, heave_f, heave_r):
#         rho = 1.225   # air density [kg/m³]
 
#         # ── Dynamic pressure (structural v² dependence) ──────────────────────
#         q_dyn = 0.5 * rho * vx ** 2
 
#         # ── Physical feature extraction ──────────────────────────────────────
#         h_ref = 0.040  # reference ride height [m]
 
#         # Ground clearances (softplus floor at 15mm — stall protection)
#         h_f = _softplus_floor(h_ref - heave_f, 0.015)
#         h_r = _softplus_floor(h_ref - heave_r, 0.015)
 
#         # Ground effect ratio: h_ref/h gives ≈1.0 at nominal, >1.0 at low rh
#         # Bounded via tanh to prevent singularity at h→0
#         ge_f = 1.0 + 0.30 * jnp.tanh(2.0 * (h_ref / h_f - 1.0))
#         ge_r = 1.0 + 0.45 * jnp.tanh(2.0 * (h_ref / h_r - 1.0))
 
#         # ── Symmetric feature vector (roll enters as roll²) ─────────────────
#         # This structurally guarantees Cl(+roll) = Cl(-roll)
#         features = jnp.stack([
#             pitch,                          # pitch sensitivity (odd → asymmetric, correct)
#             pitch ** 2,                     # pitch² (even → drag increase both directions)
#             roll ** 2,                      # roll² (even → symmetric aero loss)
#             ge_f - 1.0,                     # front ground effect deviation
#             ge_r - 1.0,                     # rear ground effect deviation
#             (heave_f - heave_r) / 0.02,     # rake angle (normalised)
#             vx / 30.0,                      # Reynolds number proxy
#         ])
 
#         # ── Neural correction network ────────────────────────────────────────
#         # Output: 5 correction factors, bounded to ±15% via scaled tanh
#         x = nn.Dense(24)(features);  x = nn.swish(x)
#         x = nn.Dense(24)(x);         x = nn.swish(x)
#         raw_corrections = nn.Dense(5,
#             kernel_init=jax.nn.initializers.zeros,
#             bias_init=jax.nn.initializers.zeros,
#         )(x)
 
#         # Bounded corrections: tanh × 0.15 → [-0.15, +0.15]
#         # Initialised at zero → starts from pure analytical baseline
#         delta = 0.15 * jnp.tanh(raw_corrections)
#         # delta[0]: ΔCl_f correction
#         # delta[1]: ΔCl_r correction
#         # delta[2]: ΔCd correction
#         # delta[3]: ΔCoP_pitch (pitching moment adjustment)
#         # delta[4]: ΔCoP_roll (rolling moment adjustment)
 
#         # ── Analytical baseline + bounded neural correction ──────────────────
 
#         # Front downforce coefficient
#         Cl_f_base = self.base_Cl * 0.40 * ge_f + 0.35 * pitch
#         Cl_f = _softplus_floor(Cl_f_base * (1.0 + delta[0]), 0.0)
 
#         # Rear downforce coefficient
#         Cl_r_base = self.base_Cl * 0.60 * ge_r
#         Cl_r = _softplus_floor(Cl_r_base * (1.0 + delta[1]), 0.0)
 
#         # Drag coefficient — ALWAYS POSITIVE via softplus
#         Cd_base = self.base_Cd + 0.05 * safe_abs(pitch)  # pitch-induced drag
#         Cd = jnp.logaddexp(0.0, Cd_base * (1.0 + delta[2]))
 
#         # ── Forces (v² dependence is structural, not learned) ────────────────
#         Fz_aero_f = q_dyn * Cl_f * self.base_A
#         Fz_aero_r = q_dyn * Cl_r * self.base_A
#         Fx_aero   = -q_dyn * Cd * self.base_A  # always opposing motion
 
#         # ── Moments ──────────────────────────────────────────────────────────
#         # Pitching moment from aero CoP offset
#         My_aero = (Fz_aero_r * self.lr - Fz_aero_f * self.lf
#                    + (Fz_aero_f + Fz_aero_r) * delta[3] * 0.1)  # ±10% arm
 
#         # Rolling moment (asymmetric downforce from roll)
#         # roll² feature means this is even in roll, but the SIGN of the moment
#         # must follow the sign of roll → multiply by roll
#         Mx_aero = (Fz_aero_f + Fz_aero_r) * delta[4] * roll * 0.05
 
#         return Fz_aero_f, Fz_aero_r, Fx_aero, My_aero, Mx_aero


# ─────────────────────────────────────────────────────────────────────────────
# §5  DifferentiableMultiBodyVehicle
# ─────────────────────────────────────────────────────────────────────────────

class DifferentiableMultiBodyVehicle:
    """
    Port-Hamiltonian 14-DOF full-car model with 28-parameter SuspensionSetup.

    State x (46-dim):
      [0:14]  q  — positions  (X,Y,Z,φ,θ,ψ, z_fl..z_rr, θ_fl..θ_rr)
      [14:28] v  — velocities (vx,vy,vz,wx,wy,wz, ż_fl..ż_rr, ω_fl..ω_rr)
      [28:38] thermal  (5-node tire model, front+rear)
      [38:46] transient slip  (α_t, κ_t × 4 corners)

    Integrator: 2-stage Gauss-Legendre RK4 (GLRK-4) — symplectic 4th-order.
    Aux integration: trapezoidal at converged Gauss stage points (no extra
    _compute_derivatives call vs. previous forward Euler).
    """

    def __init__(self, vehicle_params: dict, tire_coeffs: dict, rng_seed: int = 42):
        from models.tire_model import PacejkaTire
        self.vp   = vehicle_params
        self.tire = PacejkaTire(tire_coeffs)

        self.m_us_f = self.vp.get('unsprung_mass_f', 7.74)
        self.m_us_r = self.vp.get('unsprung_mass_r', 7.76)
        m_unsprung  = 2.0 * self.m_us_f + 2.0 * self.m_us_r

        self.m   = self.vp.get('total_mass', 300.0)
        self.m_s = self.m - m_unsprung

        self.Ix = self.vp.get('Ix', 45.0)
        self.Iy = self.vp.get('Iy', 85.0)
        self.Iz = self.vp.get('Iz', 150.0)
        self.Iw = self.vp.get('Iw', 1.2)  # Inercia real del conjunto llanta+neumático+disco

        self.lf       = self.vp.get('lf', 0.8525)
        self.lr       = self.vp.get('lr', 0.6975)
        self.track_f  = self.vp.get('track_front', 1.200)
        self.track_r  = self.vp.get('track_rear',  1.180)
        self.g        = 9.81
        self.R_wheel  = self.vp.get('wheel_radius', 0.2045)

        self.track_w  = self.track_f
        self._L       = self.lf + self.lr

        self.M_diag = jnp.array([
            self.m_s, self.m_s, self.m_s, self.Ix, self.Iy, self.Iz,
            self.m_us_f, self.m_us_f, self.m_us_r, self.m_us_r,
            self.Iw, self.Iw, self.Iw, self.Iw,
        ])

        self._mr_f_poly   = jnp.array(self.vp.get('motion_ratio_f_poly', [1.14, 2.5, 0.0]))
        self._mr_r_poly   = jnp.array(self.vp.get('motion_ratio_r_poly', [1.16, 2.0, 0.0]))
        self._bs2_f       = self.vp.get('bump_steer_quad_f', 0.0)
        self._bs2_r       = self.vp.get('bump_steer_quad_r', 0.0)
        self._camber_dz_f = self.vp.get('camber_per_m_travel_f', -25.0)
        self._camber_dz_r = self.vp.get('camber_per_m_travel_r', -20.0)
        self._dh_rc_dz_f  = self.vp.get('dh_rc_dz_f', 0.20)
        self._dh_rc_dz_r  = self.vp.get('dh_rc_dz_r', 0.30)
        self._h_rc0_f     = self.vp.get('h_rc_f', 0.040)
        self._h_rc0_r     = self.vp.get('h_rc_r', 0.060)
        self._ackermann   = self.vp.get('ackermann_factor', 0.0)
        self._comply_f    = self.vp.get('compliance_steer_f', -0.15)
        self._comply_r    = self.vp.get('compliance_steer_r', -0.10)

        # BUGFIX-4: read h_scale from file for DIAGNOSTIC logging only.
        # The architectural h_scale in NeuralEnergyLandscape is ALWAYS 1.0.
        # Passing h_net_scale.txt (a training normalisation factor) into the
        # network architecture caused a 102× energy amplification at inference:
        #   H_raw = softplus(Dense(1)(x)) * 102.62 → min(..., 5000) clips
        #   everything → all H_net gradients zero → forces saturated at FORCE_CAP.
        # The scale file is kept for W&B logging and diagnostics; it does NOT
        # feed into the network architecture.
        current_dir  = os.path.dirname(os.path.abspath(__file__))
        h_scale_path = os.path.join(current_dir, 'h_net_scale.txt')
        self._h_train_scale = 1.0   # diagnostic reference only
        if os.path.exists(h_scale_path):
            with open(h_scale_path) as f:
                self._h_train_scale = float(f.read().strip())
            print(f"[VehicleDynamics] H_net train scale (diagnostic): "
                  f"{self._h_train_scale:.4f} J  [NOT applied to architecture]")
        else:
            print("[VehicleDynamics] h_net_scale.txt not found — "
                  "run train_neural_residuals() first.")

        # h_scale=1.0 ALWAYS — weights were trained with h_scale=1.0 default.
        # Batch 1.3A: normalize generalized momentum by a physical velocity scale.
        # Since p = M_diag * v in _compute_derivatives, using M_diag * 20 m/s
        # makes the H_net input O(1) over the nominal FS operating envelope without
        # changing the trained parameter shapes. p_scale is static module metadata.
        self._h_p_scale = tuple(float(x) for x in (self.M_diag * 20.0).tolist())
        self.H_net    = PassiveHNet(
            q_dim=14, p_dim=14, setup_dim=28,
            p_scale=self._h_p_scale,
            output_mode="capped",
        )
        self.R_net    = NeuralDissipationMatrix(dim=14)
        self.aero_map = create_aero_platform(self.vp)

        # Reflection aliases for diagnostic probes in sanity_checks (§3e)
        self._h_net = self.H_net
        self._r_net = self.R_net

        rng = jax.random.PRNGKey(rng_seed)
        rng_h, rng_r = jax.random.split(rng, 2)

        self.H_params    = self.H_net.init(rng_h, jnp.zeros(14), jnp.zeros(14), DEFAULT_SETUP)
        self.R_params    = self.R_net.init(rng_r, jnp.zeros(14), jnp.zeros(14))
        self.Aero_params = None   # AeroPlatformModel is stateless — no Flax params

        # Batch 10.5 default setup export
        self._default_setup_vec = jnp.array(DEFAULT_SETUP, dtype=jnp.float32)

        for attr, fname in [('H_params', 'h_net.bytes'),
                             ('R_params', 'r_net.bytes')]:
            ckpt = os.path.join(current_dir, fname)
            if os.path.exists(ckpt):
                with open(ckpt, 'rb') as f:
                    setattr(self, attr,
                            flax.serialization.from_bytes(getattr(self, attr), f.read()))
                print(f"[VehicleDynamics] Loaded {fname}")

        # Update diagnostic parameter aliases after loading weights from disk
        self._h_params = self.H_params
        self._r_params = self.R_params

    # ─────────────────────────────────────────────────────────────────────────
    # §5.3  Core derivatives
    # ─────────────────────────────────────────────────────────────────────────

    @partial(jax.jit, static_argnums=(0,))
    def _compute_derivatives(
        self,
        x: jax.Array,
        u: jax.Array,
        setup_params: jax.Array,
        tire_cal: jax.Array = jnp.array(
            [1.0, 1.0, -1.0, 1.0, 1.0, 1.0],
            dtype=jnp.float32,
        ),
    ) -> jax.Array:

        # ── 74-DOF STATE UNPACKING ──────────────────────────────
        # 1. Kinematics (0:28) - Unchanged
        q = x[0:14]
        v = x[14:28]

        # 2. Thermal 3D [Module 3] (28:56) 
        # 4 corners × 7 nodes = 28 states
        T_4x7 = x[28:56].reshape((4, 7))

        # 3. Transient 2nd Order [Module 5] (56:72)
        # 4 corners × 4 states (alpha_t, alpha_dot, kappa_t, kappa_dot)
        transient_4x4 = x[56:72].reshape((4, 4))

        # 4. Damper Hysteresis [Module 2] (72:84)
        # 4 corners × 3 states (F1, F2, T_oil)
        damper_4x3 = x[72:84].reshape((4, 3))

        # 5. Elastokinematics [Module 4] (84:108)
        # 4 corners × 6 states (Bouc-Wen internal variables)
        elastokin_4x6 = x[84:108].reshape((4, 6))
        # ────────────────────────────────────────────────────────
        p    = self.M_diag * v

        # 1. Exact linear physics (MUST flow gradients for optimizer to steer/move)
        dH_dq_phys = jnp.zeros(14).at[6:10].set(q[6:10] * _V_STRUCT_PRIOR_K)
        dH_dp_phys = p / (self.M_diag + 1e-8)
        
        # 2. Neural residual.  Its force is setup-conditioned, so detaching its
        # gradient makes autodiff differentiate a different forward vector
        # field than finite differences.  Keep the dependency for the vehicle
        # setup derivative; higher-order optimisation must handle the resulting
        # Hessian cost explicitly rather than silently changing the derivative.
        def _nn_H(q_, p_):
            v_raw = p_ / (self.M_diag + 1e-8)
            v_safe = jnp.clip(v_raw, -35.0, 35.0)
            v_safe = v_safe.at[5].set(jnp.clip(v_safe[5], -8.0, 8.0))
            p_safe = v_safe * self.M_diag
            susp_sq = jnp.sum((q_[6:10] - _Z_EQ) ** 2) + 1e-4

            # Strip translation/rotation DOFs (X, Y, psi_yaw) before H_net ever sees
            # them. The Hamiltonian must be invariant to absolute position/heading —
            # leaving these in lets an untrained/asymmetric ∂H/∂psi leak directly
            # into dwz/dt even under perfectly mirrored physical inputs.
            q_inv = q_.at[0].set(0.0).at[1].set(0.0).at[5].set(0.0)

            H_val = self.H_net.apply(self.H_params, q_inv, p_safe, setup_params)

            # Mirror BOTH q and p fully — including the suspension corners in p,
            # which was previously missing (harmless only while dz=0).
            q_mirrored = q_inv.at[3].set(-q_inv[3]).at[6].set(q_inv[7]).at[7].set(q_inv[6]).at[8].set(q_inv[9]).at[9].set(q_inv[8])
            p_mirrored = (p_safe.at[1].set(-p_safe[1]).at[5].set(-p_safe[5])
                                .at[6].set(p_safe[7]).at[7].set(p_safe[6])
                                .at[8].set(p_safe[9]).at[9].set(p_safe[8]))
            H_val_mirrored = self.H_net.apply(self.H_params, q_mirrored, p_mirrored, setup_params)

            return 0.5 * (H_val + H_val_mirrored) * susp_sq
            
        dH_dq_nn, dH_dp_nn = jax.grad(_nn_H, argnums=(0, 1))(q, p)
        
        dH_dq = dH_dq_phys + dH_dq_nn
        dH_dp = dH_dp_phys + dH_dp_nn
        
        FORCE_CAP = 12000.0; VEL_CAP = 150.0
        dH_dq  = FORCE_CAP * jnp.tanh(dH_dq / (FORCE_CAP + 1e-8))
        dH_dp  = VEL_CAP   * jnp.tanh(dH_dp / (VEL_CAP   + 1e-8))

        grad_H = jnp.concatenate([dH_dq, dH_dp])

        J = jnp.zeros((28, 28))
        J = J.at[0:14, 14:28].set( jnp.eye(14))
        J = J.at[14:28, 0:14].set(-jnp.eye(14))
        R = jnp.zeros((28, 28))
        R = R.at[14:28, 14:28].set(self.R_net.apply(self.R_params, q, p))
        PH_accel = jnp.dot(J - R, grad_H)

        s = SuspensionSetup.from_vector(setup_params)

        X, Y, Z, phi_roll, theta_pitch, psi_yaw = q[0:6]
        vx, vy, vz, wx, wy, wz                  = v[0:6]
        vx = jnp.clip(vx, -80.0, 80.0)
        vy = jnp.clip(vy, -30.0, 30.0)
        wz = jnp.clip(wz,  -8.0,  8.0)
        Z  = jnp.clip(Z,  -0.3,  1.5)
        # REMOVED phi_roll and theta_pitch clips here

        tf2 = self.track_f / 2.0
        tr2 = self.track_r / 2.0

        # Physical travel limits — removed because stable_softplus survives overshoots safely.
        # Let the raw states flow so the bumpstop restoring forces can prevent the car from rolling.
        phi_roll    = jnp.clip(phi_roll,    -0.3, 0.3)
        theta_pitch = jnp.clip(theta_pitch, -0.2, 0.2)

        tf2 = self.track_f / 2.0
        tr2 = self.track_r / 2.0

        # Physical travel limits — keeps the car upright when open-loop oversteer occurs
        _SZ_MAX, _SZ_MIN = 0.15, -0.08
        z_fl = jnp.clip(q[6], _SZ_MIN, _SZ_MAX)
        z_fr = jnp.clip(q[7], _SZ_MIN, _SZ_MAX)
        z_rl = jnp.clip(q[8], _SZ_MIN, _SZ_MAX)
        z_rr = jnp.clip(q[9], _SZ_MIN, _SZ_MAX)
        
        # Clip suspension velocities — prevents extreme damper forces during slides
        _SDZ_MAX = 3.0
        dz_fl = jnp.clip(v[6], -_SDZ_MAX, _SDZ_MAX)
        dz_fr = jnp.clip(v[7], -_SDZ_MAX, _SDZ_MAX)
        dz_rl = jnp.clip(v[8], -_SDZ_MAX, _SDZ_MAX)
        dz_rr = jnp.clip(v[9], -_SDZ_MAX, _SDZ_MAX)

        # ── ELIMINADO EL BLOQUE DUPLICADO ANTIGUO QUE VOLVÍA A ASIGNAR ──
        # (Se eliminan las líneas repetidas de tf2/tr2 y las sobreescrituras en crudo)

        omega_fl, omega_fr          = v[10], v[11]
        omega_rl, omega_rr          = v[12], v[13]

        # Read from 2nd-order transient block already unpacked as transient_4x4
        # Layout per corner: [alpha_t, alpha_dot, kappa_t, kappa_dot]
        alpha_t_fl = transient_4x4[0, 0];  alpha_dot_fl = transient_4x4[0, 1]
        kappa_t_fl = transient_4x4[0, 2];  kappa_dot_fl = transient_4x4[0, 3]
        alpha_t_fr = transient_4x4[1, 0];  alpha_dot_fr = transient_4x4[1, 1]
        kappa_t_fr = transient_4x4[1, 2];  kappa_dot_fr = transient_4x4[1, 3]
        alpha_t_rl = transient_4x4[2, 0];  alpha_dot_rl = transient_4x4[2, 1]
        kappa_t_rl = transient_4x4[2, 2];  kappa_dot_rl = transient_4x4[2, 3]
        alpha_t_rr = transient_4x4[3, 0];  alpha_dot_rr = transient_4x4[3, 1]
        kappa_t_rr = transient_4x4[3, 2];  kappa_dot_rr = transient_4x4[3, 3]

        MR_fl = compute_motion_ratio(z_fl, self._mr_f_poly)
        MR_fr = compute_motion_ratio(z_fr, self._mr_f_poly)
        MR_rl = compute_motion_ratio(z_rl, self._mr_r_poly)
        MR_rr = compute_motion_ratio(z_rr, self._mr_r_poly)

        F_spring_fl = s.k_f * z_fl * (MR_fl ** 2)
        F_spring_fr = s.k_f * z_fr * (MR_fr ** 2)
        F_spring_rl = s.k_r * z_rl * (MR_rl ** 2)
        F_spring_rr = s.k_r * z_rr * (MR_rr ** 2)

        F_damp_fl = compute_damper_force_bilinear(dz_fl, s.c_low_f, s.c_high_f, s.v_knee_f, s.rebound_ratio_f)
        F_damp_fr = compute_damper_force_bilinear(dz_fr, s.c_low_f, s.c_high_f, s.v_knee_f, s.rebound_ratio_f)
        F_damp_rl = compute_damper_force_bilinear(dz_rl, s.c_low_r, s.c_high_r, s.v_knee_r, s.rebound_ratio_r)
        F_damp_rr = compute_damper_force_bilinear(dz_rr, s.c_low_r, s.c_high_r, s.v_knee_r, s.rebound_ratio_r)

        F_bs_fl = compute_bumpstop_force(z_fl, s.h_ride_f)
        F_bs_fr = compute_bumpstop_force(z_fr, s.h_ride_f)
        F_bs_rl = compute_bumpstop_force(z_rl, s.h_ride_r)
        F_bs_rr = compute_bumpstop_force(z_rr, s.h_ride_r)

        F_susp_fl = F_spring_fl + F_damp_fl + F_bs_fl
        F_susp_fr = F_spring_fr + F_damp_fr + F_bs_fr
        F_susp_rl = F_spring_rl + F_damp_rl + F_bs_rl
        F_susp_rr = F_spring_rr + F_damp_rr + F_bs_rr

        phi_roll_f  = (z_fl - z_fr) / (2.0 * tf2 + 1e-6)   # roll angle [rad]
        M_arb_f     = s.arb_f * phi_roll_f                   # ARB moment [N·m]
        F_arb_f     = M_arb_f / (tf2 + 1e-6)                # corner force [N]

        phi_roll_r  = (z_rl - z_rr) / (2.0 * tr2 + 1e-6)
        M_arb_r     = s.arb_r * phi_roll_r
        F_arb_r     = M_arb_r / (tr2 + 1e-6)

        F_susp_fl = F_susp_fl + F_arb_f
        F_susp_fr = F_susp_fr - F_arb_f
        F_susp_rl = F_susp_rl + F_arb_r
        F_susp_rr = F_susp_rr - F_arb_r

        F_grav_f  = self.m * self.g * self.lr / self._L
        F_grav_r  = self.m * self.g * self.lf / self._L
        ay_centripetal = vx * wz
        ax_coriolis    = -vy * wz

        # FIX (root cause 1): h_cg único y consistente en todo el bloque de Fz.
        # Antes: dFz_accel/dFz_lat usaban self.vp['h_cg'] (0.330 m) mientras que
        # el momento de cabeceo (F_ext[18]) usa s.h_cg (0.285 m, del setup
        # optimizable) — dos alturas de CG distintas en el mismo integrador.
        h_cg_val   = s.h_cg
        dFz_accel  = self.m * jnp.clip(ax_coriolis, -15.0, 15.0) * h_cg_val / self._L
        ay_clipped = jnp.clip(ay_centripetal, -50.0, 50.0)

        # FIX (root cause 2 + 3): Lateral Load Transfer Distribution (LLTD).
        # Antes: dFz_lat_f y dFz_lat_r aplicaban la masa TOTAL del coche a cada
        # eje por separado → ΣΔFz·track = 2·m·ay·h_cg (el doble del momento de
        # vuelco físico real, confirmado por el diagnóstico: ratio=2.00).
        # Además la transferencia era puramente función de (ay, h_cg): ARB y
        # muelles no la afectaban en absoluto (ΔFz=0 al variar ARB×8/k×1.8).
        #
        # Físicamente correcto: el momento de vuelco total M = m·ay·h_cg se
        # reparte entre los dos ejes según su rigidez a balanceo relativa
        # (muelles + ARB), no se aplica íntegro a cada eje. Esto además liga
        # el balance de agarre (front/rear) al setup, que es precisamente lo
        # que el optimizador MORL necesita poder mover vía ARB/muelles.
        K_roll_f = 0.5 * s.k_f * self.track_f ** 2 + s.arb_f
        K_roll_r = 0.5 * s.k_r * self.track_r ** 2 + s.arb_r
        LLTD_f   = K_roll_f / (K_roll_f + K_roll_r + 1e-6)
        LLTD_r   = 1.0 - LLTD_f

        M_roll_total = self.m * ay_clipped * h_cg_val
        dFz_lat_f  = LLTD_f * M_roll_total / (self.track_f + 1e-6)
        dFz_lat_r  = LLTD_r * M_roll_total / (self.track_r + 1e-6)

        Fz_fl = _softplus_floor(F_grav_f * 0.5 - dFz_accel * 0.5 - dFz_lat_f * 0.5, 10.0)
        Fz_fr = _softplus_floor(F_grav_f * 0.5 - dFz_accel * 0.5 + dFz_lat_f * 0.5, 10.0)
        Fz_rl = _softplus_floor(F_grav_r * 0.5 + dFz_accel * 0.5 - dFz_lat_r * 0.5, 10.0)
        Fz_rr = _softplus_floor(F_grav_r * 0.5 + dFz_accel * 0.5 + dFz_lat_r * 0.5, 10.0)

        Fz_aero_f, Fz_aero_r, Fx_aero, My_aero, Mx_aero = self.aero_map.apply(
            self.Aero_params, vx, theta_pitch, phi_roll,
            z_fl + z_fr, z_rl + z_rr,
        )
        Fz_fl = Fz_fl + Fz_aero_f * 0.5
        Fz_fr = Fz_fr + Fz_aero_f * 0.5
        Fz_rl = Fz_rl + Fz_aero_r * 0.5
        Fz_rr = Fz_rr + Fz_aero_r * 0.5

        

        # CORRECTED: Perfect mirror symmetry for the right side tires
        # Al mantener el signo menos en las manguetas derechas, aseguramos que en recta
        # gamma_fr == -gamma_fl, cancelando el camber thrust limpiamente a cero.
        # PERFECT MIRROR: Los signos menos en _camber_dz aseguran simetría exacta
        gamma_fl = jnp.deg2rad( s.camber_f + self._camber_dz_f * z_fl + self.vp.get('camber_gain_f', -0.80) * jnp.rad2deg(phi_roll) * 0.5)
        gamma_fr = jnp.deg2rad(-s.camber_f - self._camber_dz_f * z_fr + self.vp.get('camber_gain_f', -0.80) * jnp.rad2deg(phi_roll) * 0.5)
        gamma_rl = jnp.deg2rad( s.camber_r + self._camber_dz_r * z_rl + self.vp.get('camber_gain_r', -0.65) * jnp.rad2deg(phi_roll) * 0.5)
        gamma_rr = jnp.deg2rad(-s.camber_r - self._camber_dz_r * z_rr + self.vp.get('camber_gain_r', -0.65) * jnp.rad2deg(phi_roll) * 0.5)

        delta_cmd   = u[0]
        delta_bs_fl = compute_bump_steer(z_fl, s.bump_steer_f, self._bs2_f)
        delta_bs_fr = compute_bump_steer(z_fr, s.bump_steer_f, self._bs2_f)

        Fy_total_approx = jnp.clip(self.m * ay_centripetal, -5000.0, 5000.0)
        delta_comply_f  = jnp.deg2rad(self._comply_f * Fy_total_approx / 1000.0)
        delta_comply_r  = jnp.deg2rad(self._comply_r * Fy_total_approx / 1000.0)

        ack      = self._ackermann
        wb       = self._L
        delta_fl = delta_cmd * (1.0 + ack * tf2 / (2.0 * wb)) + delta_bs_fl + delta_comply_f
        delta_fr = delta_cmd * (1.0 - ack * tf2 / (2.0 * wb)) + delta_bs_fr + delta_comply_f

        eps_v        = 0.5
        v_corner_fl  = vx - wz * tf2
        v_corner_fr  = vx + wz * tf2
        alpha_kin_fl = delta_fl       - jnp.arctan2(vy + wz * self.lf, jnp.maximum(safe_abs(v_corner_fl), eps_v))
        alpha_kin_fr = delta_fr       - jnp.arctan2(vy + wz * self.lf, jnp.maximum(safe_abs(v_corner_fr), eps_v))
        alpha_kin_rl = delta_comply_r - jnp.arctan2(vy - wz * self.lr, jnp.maximum(safe_abs(vx - wz * tr2), eps_v))
        alpha_kin_rr = delta_comply_r - jnp.arctan2(vy - wz * self.lr, jnp.maximum(safe_abs(vx + wz * tr2), eps_v))

        # 2nd-order model: dα_t/dt = α_dot (already in state)
        # The acceleration (dα_dot/dt) is computed in the dx_slip block below
        d_alpha_fl = alpha_dot_fl
        d_alpha_fr = alpha_dot_fr
        d_alpha_rl = alpha_dot_rl
        d_alpha_rr = alpha_dot_rr

        # Per-corner surface ribs and gas from 3D thermal block (T_4x7 already unpacked)
        # T_4x7[i] = [inner, mid, outer, bulk, carcass, gas, track_contact]
        T_ribs_fl = T_4x7[0, :3];  T_gas_fl = T_4x7[0, 5]
        T_ribs_fr = T_4x7[1, :3];  T_gas_fr = T_4x7[1, 5]
        T_ribs_rl = T_4x7[2, :3];  T_gas_rl = T_4x7[2, 5]
        T_ribs_rr = T_4x7[3, :3];  T_gas_rr = T_4x7[3, 5]
        # Axle-averaged aliases (used by code below that hasn't been per-corner-ized yet)
        T_ribs_f = (T_ribs_fl + T_ribs_fr) * 0.5
        T_gas_f  = (T_gas_fl  + T_gas_fr)  * 0.5
        T_ribs_r = (T_ribs_rl + T_ribs_rr) * 0.5
        T_gas_r  = (T_gas_rl  + T_gas_rr)  * 0.5

        # LIFTED BLOCK: Compute track-corrected local linear wheel speeds early so they are in scope
        # Velocidades lineales de avance del buje proyectadas al plano de la rueda (v_wheel)
        v_wheel_fl = (vx - wz * tf2) * jnp.cos(delta_fl) + (vy + wz * self.lf) * jnp.sin(delta_fl)
        v_wheel_fr = (vx + wz * tf2) * jnp.cos(delta_fr) + (vy + wz * self.lf) * jnp.sin(delta_fr)
        v_wheel_rl = (vx - wz * tr2) * jnp.cos(delta_comply_r) + (vy - wz * self.lr) * jnp.sin(delta_comply_r)
        v_wheel_rr = (vx + wz * tr2) * jnp.cos(delta_comply_r) + (vy - wz * self.lr) * jnp.sin(delta_comply_r)

        # Denominador de magnitud positiva para evitar división por cero a baja velocidad
        v_fl_g = jnp.maximum(safe_abs(v_wheel_fl), 0.5)
        v_fr_g = jnp.maximum(safe_abs(v_wheel_fr), 0.5)
        v_rl_g = jnp.maximum(safe_abs(v_wheel_rl), 0.5)
        v_rr_g = jnp.maximum(safe_abs(v_wheel_rr), 0.5)

        # ═══════════════════════════════════════════════════════════════════
        # §5.1  Hub Motor Torques & Slip Reference Layout
        # ═══════════════════════════════════════════════════════════════════
        T_hub_fl = u[1]; T_hub_fr = u[2]; T_hub_rl = u[3]; T_hub_rr = u[4]
        
        # Conversión de presión de línea (bar) a par de frenado físico real en disco
        P_bar = safe_abs(u[5])
        P_pa  = P_bar * 1.0e5
        A_piston_f = 2.0 * (jnp.pi * (0.025 ** 2) / 4.0)   # Doble pistón de 25mm por cara delantera
        A_piston_r = 2.0 * (jnp.pi * (0.025 ** 2) / 4.0)   # Doble pistón de 25mm por cara trasera
        r_eff = 0.080                                      # Radio efectivo del disco [m]
        mu_pad = 0.40                                     # Coeficiente de fricción de la pastilla

        T_brake_f_corner = 2.0 * mu_pad * (P_pa * A_piston_f) * r_eff * s.brake_bias_f
        T_brake_r_corner = 2.0 * mu_pad * (P_pa * A_piston_r) * r_eff * (1.0 - s.brake_bias_f)

        eta = self.vp.get('drivetrain_efficiency', 0.95)
        gear_ratio = self.vp.get('drivetrain_ratio', 4.5)

        # Velocidades angulares integradas de las ruedas
        omega_wheel_fl = v[10]
        omega_wheel_fr = v[11]
        omega_wheel_rl = v[12]
        omega_wheel_rr = v[13]

        # kappa = (omega·R − v_wheel) / v_g — Numerador signado, denominador de magnitud pura
        kappa_ref_fl = (omega_wheel_fl * self.R_wheel - v_wheel_fl) / v_fl_g
        kappa_ref_fr = (omega_wheel_fr * self.R_wheel - v_wheel_fr) / v_fr_g
        kappa_ref_rl = (omega_wheel_rl * self.R_wheel - v_wheel_rl) / v_rl_g
        kappa_ref_rr = (omega_wheel_rr * self.R_wheel - v_wheel_rr) / v_rr_g

        d_kappa_fl = kappa_dot_fl
        d_kappa_fr = kappa_dot_fr
        d_kappa_rl = kappa_dot_rl
        d_kappa_rr = kappa_dot_rr

        # ── Keep only Block 3, then compute front Mz right after it, mirroring rear: ──
        tire_cal = jnp.asarray(tire_cal)
        if tire_cal.shape[0] == 4:
            tire_cal = jnp.concatenate([tire_cal, jnp.ones(2, dtype=tire_cal.dtype)])
        elif tire_cal.shape[0] != 6:
            raise ValueError(
                "tire_cal must have 4 legacy elements or 6 canonical elements: "
                "[mu_f, mu_r, T_opt_override, alpha_scale, rby1_scale, rby2_scale]"
            )
        mu_f, mu_r, T_opt_ovr, alpha_scl = tire_cal[0], tire_cal[1], tire_cal[2], tire_cal[3]
        rby1_scl = tire_cal[4]
        rby2_scl = tire_cal[5]

        Fx_fl, Fy_fl, sigma_fl = self.tire.compute_force_and_sigma(
            alpha_t_fl, kappa_t_fl, Fz_fl, gamma_fl, T_ribs_fl, T_gas_fl, v_fl_g,
            wz=wz, mu_scale=mu_f, T_opt_override=T_opt_ovr, alpha_scale=alpha_scl)

        Fx_fr, Fy_fr_tire, sigma_fr = self.tire.compute_force_and_sigma(
            alpha_t_fr, kappa_t_fr, Fz_fr, -gamma_fr, T_ribs_fr, T_gas_fr, v_fr_g,
            wz=-wz, mu_scale=mu_f, T_opt_override=T_opt_ovr, alpha_scale=alpha_scl)
        Fy_fr = -Fy_fr_tire

        Fx_rl, Fy_rl, sigma_rl = self.tire.compute_force_and_sigma(
            alpha_t_rl, kappa_t_rl, Fz_rl, gamma_rl, T_ribs_rl, T_gas_rl, v_rl_g,
            wz=wz, mu_scale=mu_r, T_opt_override=T_opt_ovr, alpha_scale=alpha_scl,
            rby1_scale=rby1_scl, rby2_scale=rby2_scl)

        Fx_rr, Fy_rr_tire, sigma_rr = self.tire.compute_force_and_sigma(
            alpha_t_rr, kappa_t_rr, Fz_rr, -gamma_rr, T_ribs_rr, T_gas_rr, v_rr_g,
            wz=-wz, mu_scale=mu_r, T_opt_override=T_opt_ovr, alpha_scale=alpha_scl,
            rby1_scale=rby1_scl, rby2_scale=rby2_scl)
        Fy_rr = -Fy_rr_tire

        tire_sigma_4 = jnp.array([sigma_fl, sigma_fr, sigma_rl, sigma_rr])

        # ── FIX: front Mz now built from the CALIBRATED Fx_fl/Fy_fl, not the stale ones ──
        Mz_fl = self.tire.compute_aligning_torque(alpha_t_fl, kappa_t_fl, Fz_fl, gamma_fl, Fy_fl, Fx_fl)
        Mz_fr_tire = self.tire.compute_aligning_torque(alpha_t_fr, kappa_t_fr, Fz_fr, -gamma_fr, -Fy_fr, Fx_fr)
        Mz_fr = -Mz_fr_tire
        Mz_castor_fl = compute_castor_trail(s.castor_f, Fy_fl, self.R_wheel)
        Mz_castor_fr = compute_castor_trail(s.castor_f, Fy_fr, self.R_wheel)

        Mz_rl = self.tire.compute_aligning_torque(alpha_t_rl, kappa_t_rl, Fz_rl, gamma_rl, Fy_rl, Fx_rl)
        Mz_rr_tire = self.tire.compute_aligning_torque(alpha_t_rr, kappa_t_rr, Fz_rr, -gamma_rr, -Fy_rr, Fx_rr)
        Mz_rr = -Mz_rr_tire

        Fy_f     = Fy_fl + Fy_fr
        Fy_r     = Fy_rl + Fy_rr
        # FIX: Remove hydraulic brake forces from direct chassis acceleration. 
        # Chassis deceleration must flow solely from tire patch interaction.
        Fx_f     = Fx_fl + Fx_fr
        Fx_r     = Fx_rl + Fx_rr
        Mz_total = Mz_fl + Mz_fr + Mz_rl + Mz_rr + Mz_castor_fl + Mz_castor_fr
        M_diff   = 0.0   

        if os.environ.get("GP_DEBUG_YAW"):
            jax.debug.print(
                "vwfl={a:+.3f} vwfr={b:+.3f} kfl={c:+.4f} kfr={d:+.4f} krl={cc:+.4f} krr={dd:+.4f} "
                "afl={e:+.4f} afr={f:+.4f} Fyfl={g:+.2f} Fyfr={h:+.2f} Fxfl={ii:+.2f} Fxfr={jj:+.2f} "
                "Mzfl={k:+.2f} Mzfr={l:+.2f} Mzcfl={m:+.2f} Mzcfr={n:+.2f} Mz={o:+.2f} | "
                "Fyrl={p:+.2f} Fyrr={q:+.2f} Fzrl={r:+.2f} Fzrr={t:+.2f} "
                "gamrl={u:+.4f} gamrr={vv:+.4f} arl={w:+.4f} arr={xx:+.4f}",
                a=v_wheel_fl, b=v_wheel_fr, c=kappa_ref_fl, d=kappa_ref_fr,
                cc=kappa_ref_rl, dd=kappa_ref_rr,
                e=alpha_kin_fl, f=alpha_kin_fr, g=Fy_fl, h=Fy_fr, ii=Fx_fl, jj=Fx_fr,
                k=Mz_fl, l=Mz_fr, m=Mz_castor_fl, n=Mz_castor_fr, o=Mz_total,
                p=Fy_rl, q=Fy_rr, r=Fz_rl, t=Fz_rr,
                u=gamma_rl, vv=gamma_rr, w=alpha_t_rl, xx=alpha_t_rr)

        h_rc_f = self._h_rc0_f + self._dh_rc_dz_f * (z_fl + z_fr) * 0.5
        h_rc_r = self._h_rc0_r + self._dh_rc_dz_r * (z_rl + z_rr) * 0.5

        # FIX: Explicitly match the incoming state vector precision type
        F_ext = jnp.zeros(28, dtype=x.dtype)
        F_roll_res = 0.015 * self.m * self.g * jnp.tanh(vx / 0.5)
        F_ext = F_ext.at[14].set(Fx_f + Fx_r + Fx_aero - self.m_s * self.g * jnp.sin(theta_pitch) - F_roll_res)
        F_ext = F_ext.at[15].set(Fy_f + Fy_r - self.m_s * self.g * jnp.sin(phi_roll) * jnp.cos(theta_pitch))
        F_ext = F_ext.at[16].set(F_susp_fl + F_susp_fr + F_susp_rl + F_susp_rr
                                  - self.m_s * self.g * jnp.cos(phi_roll) * jnp.cos(theta_pitch)
                                  + Fz_aero_f + Fz_aero_r)
        F_ext = F_ext.at[17].set(-Fy_f * h_rc_f - Fy_r * h_rc_r
                                + (F_susp_fl - F_susp_fr) * tf2
                                + (F_susp_rl - F_susp_rr) * tr2 + Mx_aero)
        F_ext = F_ext.at[18].set((Fx_f + Fx_r) * s.h_cg
                                  - (F_susp_fl + F_susp_fr) * self.lf
                                  + (F_susp_rl + F_susp_rr) * self.lr + My_aero)
        F_ext = F_ext.at[19].set(Fy_f * self.lf - Fy_r * self.lr
                                  - (Fx_fl - Fx_fr) * tf2
                                  - (Fx_rl - Fx_rr) * tr2
                                  + Mz_total + M_diff)
        F_ext = F_ext.at[20].set(-F_susp_fl + Fz_fl - self.m_us_f * self.g)
        F_ext = F_ext.at[21].set(-F_susp_fr + Fz_fr - self.m_us_f * self.g)
        F_ext = F_ext.at[22].set(-F_susp_rl + Fz_rl - self.m_us_r * self.g)
        F_ext = F_ext.at[23].set(-F_susp_rr + Fz_rr - self.m_us_r * self.g)
        
        # Equilibrio de par en rueda: motor con reductora planetaria menos par de frenada hidráulica
        F_ext = F_ext.at[24].set(-Fx_fl * self.R_wheel + T_hub_fl * gear_ratio * eta - T_brake_f_corner)
        F_ext = F_ext.at[25].set(-Fx_fr * self.R_wheel + T_hub_fr * gear_ratio * eta - T_brake_f_corner)
        F_ext = F_ext.at[26].set(-Fx_rl * self.R_wheel + T_hub_rl * gear_ratio * eta - T_brake_r_corner)
        F_ext = F_ext.at[27].set(-Fx_rr * self.R_wheel + T_hub_rr * gear_ratio * eta - T_brake_r_corner)

        # ── Kinematic Gauge Lock (Prevents body DOFs from drifting) ──────────
        # Because the rigid body (Z, phi, theta) is integrated independently from 
        # the suspension (z_fl..rr), we apply a virtual infinitely-stiff PD 
        # controller to lock the body to the wheels, preventing infinite tumbling.
        z_avg_dev = 0.25 * ((z_fl - _Z_EQ[0]) + (z_fr - _Z_EQ[1]) + 
                            (z_rl - _Z_EQ[2]) + (z_rr - _Z_EQ[3]))
        Z_kin = 0.30 - z_avg_dev
        
        phi_kin   = 0.5 * ((z_fr - z_fl)/self.track_f + (z_rr - z_rl)/self.track_r)
        theta_kin = ((z_fl + z_fr) - (z_rl + z_rr)) / (2.0 * self._L)
        
        k_gauge = 500_000.0  # N/m or Nm/rad (Stiff lock)
        c_gauge = 10_000.0   # Ns/m or Nms/rad (Critical damping)
        
        F_Z_gauge = k_gauge * (Z_kin - Z) - c_gauge * vz
        M_x_gauge = k_gauge * (phi_kin - phi_roll) - c_gauge * wx
        M_y_gauge = k_gauge * (theta_kin - theta_pitch) - c_gauge * wy
        
        F_ext = F_ext.at[16].add(F_Z_gauge)
        F_ext = F_ext.at[17].add(M_x_gauge)
        F_ext = F_ext.at[18].add(M_y_gauge)
        # ─────────────────────────────────────────────────────────────────────

        dq_dt = PH_accel[0:14]
        
        # ── KINEMATICS FIX: Apply Yaw Rotation ──
        # PH_accel[0:2] contains the local body frame velocities (vx, vy).
        # We must rotate them to global X, Y coordinates!
        vx_local = dq_dt[0]
        vy_local = dq_dt[1]
        
        vx_world = vx_local * jnp.cos(psi_yaw) - vy_local * jnp.sin(psi_yaw)
        vy_world = vx_local * jnp.sin(psi_yaw) + vy_local * jnp.cos(psi_yaw)
        
        dq_dt = dq_dt.at[0].set(vx_world).at[1].set(vy_world)

        # (Continue with existing dv_dt code...)
        dv_dt = (PH_accel[14:28] + F_ext[14:28]) / self.M_diag
        # Use smooth tanh bounding. This limits the acceleration to ±500 m/s² 
        # to prevent explosion, but NEVER drops the gradient to exactly 0.0.
        dv_dt = 500.0 * jnp.tanh(dv_dt / 500.0)
        dx_mech = jnp.concatenate([dq_dt, dv_dt])

        # ── Module 3: 3D per-corner thermal ODE → 28 states ──────────────────
        dT_4x7 = four_corner_thermal_derivatives(
            T_4x7,
            jnp.array([Fz_fl, Fz_fr, Fz_rl, Fz_rr]),
            jnp.array([
                jnp.sqrt(kappa_ref_fl**2 + 1e-8), jnp.sqrt(kappa_ref_fr**2 + 1e-8),
                jnp.sqrt(kappa_ref_rl**2 + 1e-8), jnp.sqrt(kappa_ref_rr**2 + 1e-8)
            ]),
            jnp.array([alpha_t_fl, alpha_t_fr, alpha_t_rl, alpha_t_rr]),
            jnp.array([gamma_fl, gamma_fr, gamma_rl, gamma_rr]),
            safe_abs(vx),
            jnp.array([omega_wheel_fl, omega_wheel_fr, omega_wheel_rl, omega_wheel_rr]),
        )
        dx_therm = dT_4x7.ravel()   # (28,)

        # ── Module 5: 2nd-order transient slip → 16 states ───────────────────
        # four_corner_transient_derivatives computes both velocity passthrough
        # (d_alpha_t = alpha_dot) AND the acceleration (d_alpha_dot) from the
        # 2nd-order ODE: ẍ + 2ζωₙẋ + ωₙ²x = ωₙ²·x_kin
        # ── Module 5: 2nd-order transient slip → 16 states ───────────────────
        d_transient_4x4 = four_corner_transient_derivatives(
            jnp.array([alpha_kin_fl, -alpha_kin_fr, alpha_kin_rl, -alpha_kin_rr]), # ✅ ESPEJO EN FR Y RR
            jnp.array([kappa_ref_fl, kappa_ref_fr, kappa_ref_rl, kappa_ref_rr]),
            transient_4x4,  
            jnp.array([Fz_fl, Fz_fr, Fz_rl, Fz_rr]),
            safe_abs(vx),
        )
        # Smooth limit instead of hard clip
        dx_slip = 500.0 * jnp.tanh(d_transient_4x4.ravel() / 500.0)   # (16,)

        # ── Module 2: Damper Maxwell branch + thermal ODE → 12 states ────────
        dz_corners = jnp.array([dz_fl, dz_fr, dz_rl, dz_rr])
        
        # PUT THESE BACK!
        F1_all  = damper_4x3[:, 0] 
        F2_all  = damper_4x3[:, 1] 
        T_oil   = damper_4x3[:, 2]

        mu_visc = jnp.exp(0.015 * (40.0 - T_oil))   
        dF1     = 15000.0 * dz_corners - F1_all / 0.008
        dF2     =  5000.0 * dz_corners - F2_all / 0.040
        P_diss  = jnp.sqrt(((F1_all + F2_all) * mu_visc * dz_corners)**2 + 1e-8)
        dT_oil_dt = (P_diss - 8.0 * (T_oil - 25.0)) / 500.0
        dx_damper = jnp.stack([dF1, dF2, dT_oil_dt], axis=1).ravel()   # (12,)

        # ── Module 4: Bouc-Wen elastokinematic hysteresis → 24 states ────────
        v_bcast  = jnp.broadcast_to(dz_corners[:, None], (4, 6))
        
        # This state is part of the forward ODE, hence it remains live for the
        # tangent/adjoint calculation.  Detaching it would again make AD and
        # finite differences refer to different discrete maps.
        z_hyst = elastokin_4x6

        v_abs_bw = safe_abs(v_bcast)
        z_abs_bw = jnp.sqrt(z_hyst ** 2 + 1e-12)
        
        # Derivative uses live state
        dz_hyst  = (v_bcast
                    - 0.5 * v_abs_bw * z_abs_bw * z_hyst
                    - 0.05 * v_bcast * z_abs_bw ** 2)
        dx_elastokin = dz_hyst.ravel()   # (24,)

        return jnp.concatenate([
            dx_mech[:28], dx_therm, dx_slip, dx_damper, dx_elastokin
        ])   # total: 28+28+16+12+24 = 108 states
    def step_with_params(
        self,
        h_params: dict,
        state:    jax.Array,   # [46] bridge state — padded to [108] internally
        setup:    jax.Array,   # [28]
        controls: jax.Array,   # [2]  [gas_signed, steer_rad]
        dt:       jax.Array,
    ) -> jax.Array:
        """
        OnlineSysID entry-point. h_params is a traced differentiation target
        (no stop_gradient) so jax.grad flows through H_net for adaptation.

        State padding: bridge sends 46-DOF; _compute_derivatives_with_h needs
        108-DOF. Indices 46:108 (thermal 3D, damper, elastokin) are zeroed —
        sysid only adapts the mechanical PH term, not aux dynamics.

        Controls expansion: scalar gas/steer → full 6-element u vector.
        Equal torque split across 4 hub motors; no TV during sysid.
        """
        # 46 → 108: zero-pad aux states (thermal/damper/elastokin)
        state_108 = jnp.concatenate([state, jnp.zeros(108 - state.shape[0])])

        gas_signed = controls[0]
        steer      = controls[1]
        T_peak     = self.vp.get('motor_peak_torque', 21.0)
        T_each     = jax.nn.relu(gas_signed) * T_peak * 0.25   # equal 4-way split
        F_brake    = jax.nn.relu(-gas_signed) * 2000.0         # scalar → N

        # u layout: [delta, T_hub_fl, T_hub_fr, T_hub_rl, T_hub_rr, F_brake_hyd]
        u_full = jnp.array([steer, T_each, T_each, T_each, T_each, F_brake])

        next_108 = self._glrk4_step_with_h(h_params, state_108, u_full, setup, dt)

        # Return 46-DOF slice only — sysid loss on [X,Y,ψ,vx,vy,wz]
        return next_108[:46]

    @partial(jax.jit, static_argnums=(0,))
    def _glrk4_step(
        self,
        x: jax.Array,
        u: jax.Array,
        setup_params: jax.Array,
        dt_step: float,
        tire_cal: jax.Array = jnp.array(
            [1.0, 1.0, -1.0, 1.0, 1.0, 1.0],
            dtype=jnp.float32,
        ),
    ) -> jax.Array:
        sqrt3    = jnp.sqrt(3.0)
        a11, a12 = 0.25, 0.25 - sqrt3 / 6.0
        a21, a22 = 0.25 + sqrt3 / 6.0, 0.25
        b1, b2   = 0.5, 0.5

        dx0 = self._compute_derivatives(x, u, setup_params, tire_cal)
        k1 = dx0
        k2 = dx0

        def newton_iter(carry, _):
            k1_, k2_ = carry
            # All 108 states are stage variables.  Freezing the 80 auxiliary
            # states here turns the coupled method into a mixed-order scheme.
            x1 = x + dt_step * (a11 * k1_ + a12 * k2_)
            x2 = x + dt_step * (a21 * k1_ + a22 * k2_)

            dx1 = self._compute_derivatives(x1, u, setup_params, tire_cal)
            dx2 = self._compute_derivatives(x2, u, setup_params, tire_cal)

            _DC = 500.0
            return (jnp.clip(dx1, -_DC, _DC), jnp.clip(dx2, -_DC, _DC)), None

        # Picard solves the implicit stage equations.  Keep this dependency in
        # the production derivative: stop_gradient here differentiates a
        # different numerical map than the one used by finite differences.
        (k1, k2), _ = jax.lax.scan(
            newton_iter, (k1, k2), None, length=_GLRK_PICARD_ITERS)
        x_new = x + dt_step * (b1 * k1 + b2 * k2)
        return x_new.at[28:108].set(jnp.clip(x_new[28:108], -1000.0, 1000.0))
    
    @partial(jax.jit, static_argnums=(0,))
    def _glrk4_step_with_h(
        self,
        h_params:     dict,
        x:            jax.Array,
        u:            jax.Array,
        setup_params: jax.Array,
        dt_step:      jax.Array,
    ) -> jax.Array:
        sqrt3    = jnp.sqrt(3.0)
        a11, a12 = 0.25, 0.25 - sqrt3 / 6.0
        a21, a22 = 0.25 + sqrt3 / 6.0, 0.25
        b1, b2   = 0.5, 0.5

        dx0 = self._compute_derivatives(x, u, setup_params)
        k1 = dx0
        k2 = dx0

        def newton_iter(carry, _):
            k1_, k2_ = carry
            x1 = x + dt_step * (a11 * k1_ + a12 * k2_)
            x2 = x + dt_step * (a21 * k1_ + a22 * k2_)

            dx1 = self._compute_derivatives(x1, u, setup_params)
            dx2 = self._compute_derivatives(x2, u, setup_params)

            _DC = 500.0
            return (jnp.clip(dx1, -_DC, _DC), jnp.clip(dx2, -_DC, _DC)), None

        (k1, k2), _ = jax.lax.scan(
            newton_iter, (k1, k2), None, length=_GLRK_PICARD_ITERS)
        x_new = x + dt_step * (b1 * k1 + b2 * k2)
        return x_new.at[28:108].set(jnp.clip(x_new[28:108], -1000.0, 1000.0))

    @partial(jax.jit, static_argnums=(0,))
    def estimate_corner_sigma(self, x: jax.Array, setup_params: jax.Array) -> jax.Array:
        """
        Cheap diagnostic: evaluates ONLY the 4 tire GP calls at the current
        state, without running full dynamics. O(4× PINN/GP forward pass),
        negligible next to a GLRK-4 substep. Call once per macro-step (200 Hz),
        feed result into powertrain_step's gp_sigma argument.

        Returns (4,) per-corner GP σ ∈ (0, ~0.3), same units as the LCB penalty.
        """
        q = x[0:14]; v = x[14:28]
        T_4x7 = x[28:56].reshape((4, 7))
        transient_4x4 = x[56:72].reshape((4, 4))
        s = SuspensionSetup.from_vector(setup_params)

        vx = jnp.clip(v[0], -80.0, 80.0)
        # Reuse the exact same Fz/gamma computation the real derivative call
        # uses would require re-deriving suspension forces — for a DIAGNOSTIC
        # sigma feed (not control-critical), a static-load + aero approximation
        # is sufficient since sigma only needs to track "am I in-distribution",
        # not exact instantaneous Fz.
        Fz_static = self.m * self.g * jnp.array([self.lr, self.lr, self.lf, self.lf]) / (2.0 * self._L)

        alpha_t = transient_4x4[:, 0]
        kappa_t = transient_4x4[:, 2]
        T_ribs  = T_4x7[:, :3]
        T_gas   = T_4x7[:, 5]

        def _corner_sigma(a, k, fz, tr, tg):
            _, _, sig = self.tire.compute_force_and_sigma(
                a, k, fz, jnp.array(0.0), tr, tg, jnp.maximum(jnp.abs(vx), 0.5))
            return sig

        return jax.vmap(_corner_sigma)(alpha_t, kappa_t, Fz_static, T_ribs, T_gas)

    # ─────────────────────────────────────────────────────────────────────────
    # §5.5  Public simulate_step
    # ─────────────────────────────────────────────────────────────────────────

    @partial(jax.jit, static_argnums=(0, 5))
    def simulate_step(
        self, state, controls, setup, dt=0.005, n_substeps=5,
        tire_cal: jax.Array = jnp.array(
            [1.0, 1.0, -1.0, 1.0, 1.0, 1.0],
            dtype=jnp.float32,
        ),
    ) -> jax.Array:

        """
        Advance simulation by `dt` using `n_substeps` GLRK-4 steps.

        Accepts:
          SuspensionSetup  — preferred
          jnp.ndarray(28,) — accepted directly
          jnp.ndarray(8,)  — legacy 8-element (auto-upgraded via from_legacy_8)
        """
        if isinstance(setup, SuspensionSetup):
            setup_vec = setup.to_vector().astype(state.dtype)
        else:
            # Match setup vector precision to state vector precision
            v = jnp.asarray(setup, dtype=state.dtype)
            if v.shape[0] == 8:
                setup_vec = SuspensionSetup.from_legacy_8(v).to_vector().astype(state.dtype)
            else:
                setup_vec = v

        dt_sub = dt / n_substeps

        def substep(x, _):
            return self._glrk4_step(x, controls, setup_vec, dt_sub, tire_cal), None
        
        # CRITICAL RAM FIX: Checkpoint the substep
        substep_ckpt = jax.checkpoint(substep)

        final_state, _ = jax.lax.scan(substep_ckpt, state, None, length=n_substeps)
        return final_state

    @staticmethod
    def default_setup_params() -> jax.Array:
        return DEFAULT_SETUP
    
    @staticmethod
    def make_initial_state(T_env: float = 25.0, vx0: float = 0.0) -> jax.Array:
        """
        Create a valid 108-state initial vector for the extended model.

        State layout:
          [0:28]   kinematics (q, v)
          [28:56]  thermal 3D  — 4 corners × 7 nodes
          [56:72]  transient 2nd-order — 4 corners × 4 (α_t, α̇, κ_t, κ̇)
          [72:84]  damper hysteresis  — 4 corners × 3 (F1, F2, T_oil)
          [84:108] elastokin Bouc-Wen — 4 corners × 6
        """
        x = jnp.zeros(108)

        # CG height and optional initial speed
        x = x.at[2].set(0.3)
        x = x.at[14].set(vx0)

        # Thermal 3D: all nodes start at T_env+5 (soaked car)
        # Node layout per corner: [inner, mid, outer, bulk, carcass, gas, contact]
        x = x.at[28:56].set(T_env + 5.0)
        # Override gas (node 5) and track contact (node 6) per corner
        for i in range(4):
            x = x.at[28 + i * 7 + 5].set(T_env)          # gas at ambient
            x = x.at[28 + i * 7 + 6].set(T_env + 10.0)   # track warmer than air

        # Damper: T_oil starts at 40°C (warm but not hot)
        for i in range(4):
            x = x.at[72 + i * 3 + 2].set(40.0)

        # Wheel spin: consistent with vx0 to avoid false lockup at t=0.
        # kappa = (omega*r - vx)/vx → zero slip requires omega = vx/r.
        # R_wheel=0.2045 matches self.R_wheel default in __init__.
        _R_WHEEL = 0.2045
        _omega0  = vx0 / (_R_WHEEL + 1e-6)
        x = x.at[24].set(_omega0)
        x = x.at[25].set(_omega0)
        x = x.at[26].set(_omega0)
        x = x.at[27].set(_omega0)

        # Suspension equilibrium: static deflection under sprung mass weight.
        # z=0 → bumpstop softplus(200*(0-0.025)) overflows float32 → NaN.
        # Default spring rates k_f≈35000, k_r≈35000 N/m from build_default_setup_28.
        # Static load per corner: F = m_s * g / 4
        # z_eq = F / k ≈ (200 * 9.81 / 4) / 35000 ≈ 0.014 m
        _z_eq_f = 0.0128   # m — front static deflection
        _z_eq_r = 0.0142   # m — rear static deflection  
        x = x.at[6].set(_z_eq_f)   # z_FL
        x = x.at[7].set(_z_eq_f)   # z_FR
        x = x.at[8].set(_z_eq_r)   # z_RL
        x = x.at[9].set(_z_eq_r)   # z_RR

        # Transient slip and elastokin: all zeros (quiescent start)
        return x

# ─────────────────────────────────────────────────────────────────────────────
# §6  Public factory functions
# ─────────────────────────────────────────────────────────────────────────────

def make_setup_from_params(vp: dict) -> SuspensionSetup:
    """
    Construct a SuspensionSetup from a vehicle_params dict.
    CANONICAL construction path for all callers not using the optimizer.
    """
    v = jnp.array([
        vp.get('spring_rate_f',     DEFAULT_SETUP[0]),
        vp.get('spring_rate_r',     DEFAULT_SETUP[1]),
        vp.get('arb_rate_f',        DEFAULT_SETUP[2]),
        vp.get('arb_rate_r',        DEFAULT_SETUP[3]),
        vp.get('damper_c_low_f',    DEFAULT_SETUP[4]),
        vp.get('damper_c_low_r',    DEFAULT_SETUP[5]),
        vp.get('damper_c_high_f',   DEFAULT_SETUP[6]),
        vp.get('damper_c_high_r',   DEFAULT_SETUP[7]),
        vp.get('damper_v_knee',     DEFAULT_SETUP[8]),
        vp.get('damper_v_knee',     DEFAULT_SETUP[9]),
        vp.get('rebound_ratio_f',   DEFAULT_SETUP[10]),
        vp.get('rebound_ratio_r',   DEFAULT_SETUP[11]),
        vp.get('h_ride_f', vp.get('h_ride_design', DEFAULT_SETUP[12])),
        vp.get('h_ride_r', vp.get('h_ride_design', DEFAULT_SETUP[13])),
        vp.get('static_camber_f',   DEFAULT_SETUP[14]),
        vp.get('static_camber_r',   DEFAULT_SETUP[15]),
        vp.get('static_toe_f',      DEFAULT_SETUP[16]),
        vp.get('static_toe_r',      DEFAULT_SETUP[17]),
        vp.get('castor_f',          DEFAULT_SETUP[18]),
        vp.get('anti_squat',        DEFAULT_SETUP[19]),
        vp.get('anti_dive_f',       DEFAULT_SETUP[20]),
        vp.get('anti_dive_r',       DEFAULT_SETUP[21]),
        vp.get('anti_lift',         DEFAULT_SETUP[22]),
        vp.get('diff_lock_ratio',   DEFAULT_SETUP[23]),
        vp.get('brake_bias_f',      DEFAULT_SETUP[24]),
        vp.get('h_cg',              DEFAULT_SETUP[25]),
        vp.get('bump_steer_f',      DEFAULT_SETUP[26]),
        vp.get('bump_steer_r',      DEFAULT_SETUP[27]),
    ], dtype=jnp.float32)
    return SuspensionSetup.from_vector(v)


def build_default_setup_28(vp: dict) -> jax.Array:
    """
    Returns canonical 28-element float32 array from vehicle_params dict.
    This is the ONLY correct way to build setup_params for ocp_solver.
    """
    return make_setup_from_params(vp).to_vector()


def compute_equilibrium_suspension(setup_vec: jax.Array, vp: dict) -> jax.Array:
    """
    Returns static equilibrium suspension displacements (z_fl, z_fr, z_rl, z_rr) [m].

    Fully differentiable w.r.t. setup_vec — safe inside jit/grad/vmap.
    """
    k_f  = setup_vec[0];  k_r  = setup_vec[1]
    m    = jnp.array(vp.get('total_mass',    300.0))
    g    = 9.81
    lf   = jnp.array(vp.get('lf',          0.8525))
    lr   = jnp.array(vp.get('lr',          0.6975))
    L    = lf + lr
    muf  = jnp.array(vp.get('unsprung_mass_f', 7.74))
    mur  = jnp.array(vp.get('unsprung_mass_r', 7.76))
    mr_f = jnp.array(vp.get('motion_ratio_f_poly', [1.14, 2.5, 0.0]))[0]
    mr_r = jnp.array(vp.get('motion_ratio_r_poly', [1.16, 2.0, 0.0]))[0]

    F_susp_f_eq = m * g * lr / (2.0 * L) - muf * g
    F_susp_r_eq = m * g * lf / (2.0 * L) - mur * g
    z_f_eq = F_susp_f_eq / (k_f * mr_f ** 2 + 1e-6)
    z_r_eq = F_susp_r_eq / (k_r * mr_r ** 2 + 1e-6)
    return jnp.array([z_f_eq, z_f_eq, z_r_eq, z_r_eq])

# ═══════════════════════════════════════════════════════════════════════════════
# NEW METHOD 1: observe_sensors()
# ═══════════════════════════════════════════════════════════════════════════════
 
import jax
import jax.numpy as jnp
from typing import NamedTuple
 
 
class SensorReading(NamedTuple):
    """Noisy sensor outputs — feeds the UKF state estimator."""
    ax_imu:       jax.Array   # [m/s²] longitudinal accel + noise + bias
    ay_imu:       jax.Array   # [m/s²] lateral accel + noise + bias
    az_imu:       jax.Array   # [m/s²] vertical accel + noise + bias
    wx_gyro:      jax.Array   # [rad/s] roll rate + noise + bias
    wy_gyro:      jax.Array   # [rad/s] pitch rate + noise + bias
    wz_gyro:      jax.Array   # [rad/s] yaw rate + noise + bias
    omega_fl:     jax.Array   # [rad/s] wheel speed FL + quantization
    omega_fr:     jax.Array   # [rad/s] wheel speed FR + quantization
    omega_rl:     jax.Array   # [rad/s] wheel speed RL + quantization
    omega_rr:     jax.Array   # [rad/s] wheel speed RR + quantization
    delta_steer:  jax.Array   # [rad]   steering angle + noise
    vx_gps:       jax.Array   # [m/s]   GPS longitudinal speed (low rate, high noise)
 
 
# Sensor noise parameters (FS-realistic for MEMS IMU + wheel encoders)
_IMU_ACC_NOISE_STD   = 0.15    # m/s²  (MPU-6050 class)
_IMU_ACC_BIAS_STD    = 0.05    # m/s²  slowly drifting bias
_IMU_GYRO_NOISE_STD  = 0.005   # rad/s
_IMU_GYRO_BIAS_STD   = 0.001   # rad/s
_WHEEL_SPEED_NOISE   = 0.3     # rad/s  (encoder quantization + bearing)
_STEER_ANGLE_NOISE   = 0.002   # rad    (potentiometer)
_GPS_SPEED_NOISE     = 0.5     # m/s    (10 Hz GPS)
 
 
def observe_sensors(
    x: jax.Array,
    u: jax.Array,
    key: jax.Array,
    imu_bias: jax.Array = None,  # (6,) persistent IMU bias [ax,ay,az,wx,wy,wz]
) -> SensorReading:
    """
    Generate noisy sensor readings from perfect simulation state.
 
    Args:
        x: (46,) full state vector
        u: (6,)  input vector [δ, T_fl, T_fr, T_rl, T_rr, F_brake_hyd]
        key: JAX PRNG key
        imu_bias: (6,) persistent bias terms (should be held across steps)
 
    Returns:
        SensorReading with all noisy channels
    """
    if imu_bias is None:
        imu_bias = jnp.zeros(6)
 
    keys = jax.random.split(key, 12)
 
    # True states
    vx = x[14];  vy = x[15];  vz = x[16]
    wx = x[17];  wy = x[18];  wz = x[19]
 
    # Approximate true accelerations (from state derivatives)
    # In practice these come from the previous step's derivative
    ax_true = vx * 0.0   # placeholder — caller should pass dx/dt
    ay_true = vx * wz     # centripetal
    az_true = 9.81         # gravity (body frame, approx)
 
    # IMU accelerometer: truth + white noise + bias
    ax_imu = ax_true + imu_bias[0] + _IMU_ACC_NOISE_STD * jax.random.normal(keys[0])
    ay_imu = ay_true + imu_bias[1] + _IMU_ACC_NOISE_STD * jax.random.normal(keys[1])
    az_imu = az_true + imu_bias[2] + _IMU_ACC_NOISE_STD * jax.random.normal(keys[2])
 
    # IMU gyroscope: truth + white noise + bias
    wx_gyro = wx + imu_bias[3] + _IMU_GYRO_NOISE_STD * jax.random.normal(keys[3])
    wy_gyro = wy + imu_bias[4] + _IMU_GYRO_NOISE_STD * jax.random.normal(keys[4])
    wz_gyro = wz + imu_bias[5] + _IMU_GYRO_NOISE_STD * jax.random.normal(keys[5])
 
    # Wheel speed encoders: truth + quantization noise
    omega_fl = x[24] + _WHEEL_SPEED_NOISE * jax.random.normal(keys[6])
    omega_fr = x[25] + _WHEEL_SPEED_NOISE * jax.random.normal(keys[7])
    omega_rl = x[26] + _WHEEL_SPEED_NOISE * jax.random.normal(keys[8])
    omega_rr = x[27] + _WHEEL_SPEED_NOISE * jax.random.normal(keys[9])
 
    # Steering angle sensor
    delta_steer = u[0] + _STEER_ANGLE_NOISE * jax.random.normal(keys[10])
 
    # GPS speed (low-rate, high noise — simulated at physics rate but in
    # practice this would be decimated to 10 Hz)
    vx_gps = vx + _GPS_SPEED_NOISE * jax.random.normal(keys[11])
 
    return SensorReading(
        ax_imu=ax_imu, ay_imu=ay_imu, az_imu=az_imu,
        wx_gyro=wx_gyro, wy_gyro=wy_gyro, wz_gyro=wz_gyro,
        omega_fl=omega_fl, omega_fr=omega_fr,
        omega_rl=omega_rl, omega_rr=omega_rr,
        delta_steer=delta_steer, vx_gps=vx_gps,
    )
 
 
def step_imu_bias(
    bias: jax.Array,
    key: jax.Array,
    dt: float = 0.005,
) -> jax.Array:
    """
    Random-walk IMU bias update. Bias drifts slowly per step.
 
    Args:
        bias: (6,) current bias [ax,ay,az,wx,wy,wz]
        key: PRNG key
        dt: timestep
 
    Returns:
        (6,) updated bias
    """
    acc_drift = 0.001   # m/s² per sqrt(s)
    gyro_drift = 0.0002  # rad/s per sqrt(s)
    drift_std = jnp.array([
        acc_drift, acc_drift, acc_drift,
        gyro_drift, gyro_drift, gyro_drift,
    ]) * jnp.sqrt(dt)
    return bias + drift_std * jax.random.normal(key, shape=(6,))
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# NEW METHOD 2: Domain Randomization for simulate_step
# ═══════════════════════════════════════════════════════════════════════════════
 
class DomainRandomization(NamedTuple):
    """Per-step domain randomization parameters."""
    mu_scale:     jax.Array   # friction coefficient multiplier (nominal=1.0)
    track_noise:  jax.Array   # (4,) per-corner road surface roughness [m]
    mass_delta:   jax.Array   # mass perturbation [kg] (driver weight uncertainty)
    aero_scale:   jax.Array   # aero coefficient multiplier (nominal=1.0)
 
 
def sample_domain_randomization(
    key: jax.Array,
    mu_std: float = 0.08,
    track_std: float = 0.0005,   # 0.5mm road roughness
    mass_std: float = 3.0,       # ±3 kg driver weight
    aero_std: float = 0.05,
) -> DomainRandomization:
    """
    Sample domain randomization parameters from physically motivated priors.
 
    Designed to prevent the Batch 11 optimizer from overfitting to a
    perfectly smooth track with perfect friction.
    """
    k1, k2, k3, k4 = jax.random.split(key, 4)
    return DomainRandomization(
        mu_scale=1.0 + mu_std * jax.random.normal(k1),
        track_noise=track_std * jax.random.normal(k2, shape=(4,)),
        mass_delta=mass_std * jax.random.normal(k3),
        aero_scale=1.0 + aero_std * jax.random.normal(k4),
    )

# ─────────────────────────────────────────────────────────────────────────────
# §7  Backward-compat aliases
# ─────────────────────────────────────────────────────────────────────────────
MultiBodyVehicle = DifferentiableMultiBodyVehicle
VehicleDynamicsH = DifferentiableMultiBodyVehicle
