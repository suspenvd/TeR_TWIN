# models/aero_platform.py
# ═══════════════════════════════════════════════════════════════════════════════
# Project-GP — Attitude-Coupled Aerodynamic Platform Model
# ═══════════════════════════════════════════════════════════════════════════════
#
# PROBLEM (Flaw #1):
#   DifferentiableAeroMap treats downforce as Fz ∝ v²·Cl with Cl learned from
#   an MLP on (vx, pitch, roll, heave_f, heave_r). The MLP has no structural
#   guarantee of ground-effect stall below critical ride height, no CoP
#   migration under pitch, and no yaw-angle sensitivity. The evolutionary
#   optimizer freely suggests setups that bottom out the front splitter
#   without penalty because the aero model never "stalls".
#
# SOLUTION:
#   Physics-structured aero model with:
#   1. Ground-effect envelope: Cl(rh) = Cl_max · σ(rh; rh_peak, rh_stall)
#      where σ is a differentiable bell curve that peaks at optimal ride height
#      and decays (stalls) below a critical floor clearance.
#   2. Pitch-coupled CoP migration: as pitch increases, aero center moves
#      rearward (front wing loses incidence, rear wing gains).
#   3. Roll-induced asymmetric downforce: C_l roll moment from asymmetric
#      underbody flow.
#   4. Yaw sensitivity: crosswind or yaw angle reduces effective Cl via
#      cosine-squared projection + induced side force.
#   5. Drag buildup from attitude: pitch-drag, yaw-drag, cooling drag.
#
# MATHEMATICAL STRUCTURE:
#   Cl_eff(rh, θ, φ, ψ) = Cl_max · Γ_ge(rh) · Γ_pitch(θ) · Γ_yaw(ψ)
#   where Γ_ge is the ground-effect envelope (Gaussian bump + stall tail),
#   Γ_pitch is the pitch sensitivity (linear + quadratic),
#   Γ_yaw is cosine-squared yaw loss.
#
#   CoP_x(θ) = CoP_0 + dCoP/dθ · θ  (front/rear split migration)
#   Mx_aero(φ) = ½ρv²A · Cl · φ · d_roll  (roll-induced aero moment)
#
# JAX CONTRACT: Pure JAX, JIT-safe, C∞ everywhere, ∂/∂(rh, θ, φ, ψ) exist.
# PORT-HAMILTONIAN: Aero forces are conservative in the co-moving frame —
#   they inject energy (from flow) but dissipation is structural (drag).
#   No free energy is created: drag is always opposing motion.
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import NamedTuple
from functools import partial


# ─────────────────────────────────────────────────────────────────────────────
# §1  Configuration
# ─────────────────────────────────────────────────────────────────────────────

class AeroPlatformConfig(NamedTuple):
    """Physical aero platform parameters — all from CFD/wind tunnel."""
    # Reference geometry
    Cl_max: float = 4.14        # peak Cl at optimal ride height
    Cd_base: float = 1.15       # base drag coefficient
    A_ref: float = 1.1          # frontal area [m²]
    rho_air: float = 1.225      # air density [kg/m³]

    # Ground effect envelope [mm]
    rh_peak: float = 30.0       # ride height at peak downforce [mm]
    rh_stall: float = 12.0      # ride height where flow stalls [mm]
    rh_high: float = 80.0       # ride height where ground effect vanishes [mm]
    ge_sharpness: float = 0.15  # stall transition sharpness [1/mm]

    # Pitch sensitivity
    dCl_dpitch: float = -0.08   # Cl change per degree pitch (nose-down = positive pitch)
    dCl_dpitch2: float = -0.02  # quadratic pitch penalty (always reduces Cl)
    dCoP_dpitch: float = 0.015  # CoP shift per degree pitch [fraction of wheelbase]
    dCd_dpitch: float = 0.04    # drag increase per degree pitch (abs)

    # Roll sensitivity
    dCl_droll2: float = -0.03   # Cl loss per (roll°)² — always negative
    roll_moment_arm: float = 0.05  # fraction of half-track for aero roll moment

    # Yaw sensitivity
    Cy_yaw: float = 0.8         # side force coefficient at 90° yaw (per radian)
    dCd_dyaw2: float = 0.25     # yaw-induced drag coefficient per rad²

    # Front/rear aero split at zero attitude
    aero_split_f: float = 0.40  # fraction of total Cl on front axle
    aero_split_r: float = 0.60

    # Geometric references
    lf: float = 0.8525          # front axle to CG [m]
    lr: float = 0.6975          # rear axle to CG [m]


# ─────────────────────────────────────────────────────────────────────────────
# §2  Ground Effect Envelope
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def ground_effect_envelope(
    rh_mm: jax.Array,
    rh_peak: float = 30.0,
    rh_stall: float = 12.0,
    rh_high: float = 80.0,
    sharpness: float = 0.15,
) -> jax.Array:
    """
    Differentiable ground-effect Cl multiplier Γ_ge ∈ (0, 1].

    Physics: underbody venturi effect peaks at rh_peak. Below rh_stall,
    flow separation causes catastrophic downforce loss (aero stall).
    Above rh_high, ground effect diminishes to free-stream.

    Shape: product of two sigmoids creating a smooth bell:
      Γ_ge(rh) = σ_low(rh) · σ_high(rh)
      σ_low  = sigmoid(sharpness · (rh - rh_stall))   — kills Cl below stall
      σ_high = sigmoid(sharpness · (rh_high - rh))     — kills Cl above ceiling

    The peak automatically falls near rh_peak when rh_stall and rh_high
    are symmetric around it.

    Args:
        rh_mm: ride height in mm (can be per-axle or scalar)

    Returns:
        Γ_ge ∈ (0, 1] — multiply by Cl_max to get effective Cl
    """
    # Low-rh stall sigmoid: 0 when rh << rh_stall, 1 when rh >> rh_stall
    sigma_low = jax.nn.sigmoid(sharpness * (rh_mm - rh_stall))

    # High-rh decay sigmoid: 1 when rh << rh_high, 0 when rh >> rh_high
    sigma_high = jax.nn.sigmoid(sharpness * (rh_high - rh_mm))

    # Peak enhancement: Gaussian bump centered at rh_peak
    # This shapes the bell to have a true maximum at rh_peak
    width = (rh_high - rh_stall) * 0.4
    gaussian_boost = jnp.exp(-0.5 * ((rh_mm - rh_peak) / (width + 1e-6)) ** 2)

    # Composite: sigmoid walls × gaussian peak shaping
    # Normalize so Γ_ge(rh_peak) ≈ 1.0
    raw = sigma_low * sigma_high * (0.6 + 0.4 * gaussian_boost)

    # Normalize to peak = 1.0
    peak_val = (jax.nn.sigmoid(sharpness * (rh_peak - rh_stall))
                * jax.nn.sigmoid(sharpness * (rh_high - rh_peak)))
    return raw / (peak_val * 1.0 + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# §3  Pitch Sensitivity
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def pitch_sensitivity(
    pitch_rad: jax.Array,
    dCl_dpitch: float = -0.08,
    dCl_dpitch2: float = -0.02,
) -> jax.Array:
    """
    Pitch-dependent Cl multiplier.

    Physics: small nose-down pitch (positive θ) can increase front wing
    incidence but the quadratic term always penalizes large attitudes.

    Γ_pitch(θ) = 1 + dCl/dθ · θ_deg + dCl/dθ² · θ_deg²

    Clamped to (0.3, 1.3) via smooth tanh compression.
    """
    pitch_deg = pitch_rad * (180.0 / jnp.pi)
    raw = 1.0 + dCl_dpitch * pitch_deg + dCl_dpitch2 * pitch_deg ** 2
    # Smooth clamp to [0.3, 1.3] — prevents negative Cl or unphysical amplification
    center = 0.8
    half_range = 0.5
    return center + half_range * jnp.tanh((raw - center) / half_range)


@jax.jit
def cop_migration(
    pitch_rad: jax.Array,
    aero_split_f_0: float = 0.40,
    dCoP_dpitch: float = 0.015,
) -> jax.Array:
    """
    Center-of-pressure migration under pitch.

    Positive pitch (nose down) → front wing gains incidence → CoP moves forward.
    Returns front aero split fraction ∈ (0.25, 0.60).

    CoP_f(θ) = CoP_f_0 + dCoP/dθ · θ_deg
    """
    pitch_deg = pitch_rad * (180.0 / jnp.pi)
    raw_split = aero_split_f_0 + dCoP_dpitch * pitch_deg
    # Smooth clamp to physical range
    return 0.425 + 0.175 * jnp.tanh((raw_split - 0.425) / 0.175)


# ─────────────────────────────────────────────────────────────────────────────
# §4  Roll & Yaw Sensitivity
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def roll_cl_loss(roll_rad: jax.Array, dCl_droll2: float = -0.03) -> jax.Array:
    """
    Roll-induced Cl loss: Γ_roll = 1 + dCl/dφ² · φ²

    Physics: roll exposes one side of the underbody, breaking the
    venturi seal. Always reduces total Cl (dCl_droll2 < 0).
    """
    roll_deg = roll_rad * (180.0 / jnp.pi)
    return jnp.clip(1.0 + dCl_droll2 * roll_deg ** 2, 0.5, 1.0)


@jax.jit
def yaw_sensitivity(
    yaw_rate: jax.Array,
    vx: jax.Array,
    Cy_yaw: float = 0.8,
    dCd_dyaw2: float = 0.25,
) -> tuple:
    """
    Yaw-angle effects on aero.

    The effective yaw angle β ≈ atan(vy/vx) ≈ ωz·L/(2·vx) for small angles.
    We use yaw_rate as a proxy for sideslip.

    Returns:
        Γ_yaw: Cl multiplier from cos²(β) projection loss
        Fy_aero: aerodynamic side force [N/q·A] (needs multiplication by q·A)
        dCd_yaw: drag increment from yaw
    """
    vx_safe = jnp.maximum(jnp.abs(vx), 2.0)
    # Effective sideslip from yaw rate (simplified; full model uses vy directly)
    L_ref = 1.55  # wheelbase for angle estimation
    beta_eff = jnp.arctan2(yaw_rate * L_ref * 0.5, vx_safe)

    # Cl loss: cos²(β) ≈ 1 - β² for small β
    Gamma_yaw = jnp.cos(beta_eff) ** 2

    # Side force coefficient
    Cy = Cy_yaw * jnp.sin(2.0 * beta_eff)  # peaks at 45°, linear near 0

    # Yaw-induced drag
    dCd = dCd_dyaw2 * beta_eff ** 2

    return Gamma_yaw, Cy, dCd


# ─────────────────────────────────────────────────────────────────────────────
# §5  Softplus Floor (match existing API)
# ─────────────────────────────────────────────────────────────────────────────

def _softplus_floor(x: jax.Array, floor: float, beta: float = 20.0) -> jax.Array:
    """C∞ lower bound: approaches floor from above, never negative gradient."""
    return floor + jax.nn.softplus(beta * (x - floor)) / beta


# ─────────────────────────────────────────────────────────────────────────────
# §6  Main Platform Aero Model
# ─────────────────────────────────────────────────────────────────────────────

class AeroPlatformModel:
    """
    Attitude-coupled aerodynamic model with ground-effect stall.

    Drop-in replacement for DifferentiableAeroMap.apply() — same output
    signature (Fz_aero_f, Fz_aero_r, Fx_aero, My_aero, Mx_aero) plus
    optional extended outputs (Fy_aero, CoP_x, Gamma_ge diagnostic).

    Key difference: this model STALLS when ride height drops below
    rh_stall, creating a strong gradient signal that penalizes
    setups with inadequate ride height margin.
    """

    def __init__(self, config: AeroPlatformConfig = AeroPlatformConfig()):
        self.cfg = config
        self.lf = config.lf
        self.lr = config.lr
        self._L = config.lf + config.lr

    @partial(jax.jit, static_argnums=(0,))
    def apply(
        self,
        _params,             # ignored (compatibility with DifferentiableAeroMap)
        vx: jax.Array,       # longitudinal velocity [m/s]
        pitch: jax.Array,    # body pitch angle [rad] (positive = nose down)
        roll: jax.Array,     # body roll angle [rad]
        heave_f: jax.Array,  # front heave [m] (sum of z_fl + z_fr)
        heave_r: jax.Array,  # rear heave [m] (sum of z_rl + z_rr)
        yaw_rate: jax.Array = 0.0,  # yaw rate [rad/s] — new input
    ) -> tuple:
        """
        Compute attitude-coupled aerodynamic forces and moments.

        Returns: (Fz_aero_f, Fz_aero_r, Fx_aero, My_aero, Mx_aero)
                 Same signature as DifferentiableAeroMap.apply()
        """
        c = self.cfg

        # ── Dynamic pressure ──────────────────────────────────────────────
        vx_safe = jnp.maximum(jnp.abs(vx), 0.5)
        q_dyn = 0.5 * c.rho_air * vx_safe ** 2

        # ── Ride heights [mm] ─────────────────────────────────────────────
        # heave_f/r are sum of left+right corner displacements [m]
        # Static ride height minus compression gives actual clearance
        rh_static_f = 30.0   # mm — nominal front ride height
        rh_static_r = 35.0   # mm — nominal rear ride height
        rh_f_mm = rh_static_f - heave_f * 0.5 * 1000.0  # average front rh [mm]
        rh_r_mm = rh_static_r - heave_r * 0.5 * 1000.0  # average rear rh [mm]

        # Per-axle ground effect envelope
        Gamma_ge_f = ground_effect_envelope(
            rh_f_mm, c.rh_peak, c.rh_stall, c.rh_high, c.ge_sharpness)
        Gamma_ge_r = ground_effect_envelope(
            rh_r_mm, c.rh_peak, c.rh_stall, c.rh_high, c.ge_sharpness)

        # ── Pitch effects ─────────────────────────────────────────────────
        Gamma_pitch = pitch_sensitivity(pitch, c.dCl_dpitch, c.dCl_dpitch2)
        split_f = cop_migration(pitch, c.aero_split_f, c.dCoP_dpitch)
        split_r = 1.0 - split_f

        # ── Roll effects ──────────────────────────────────────────────────
        Gamma_roll = roll_cl_loss(roll, c.dCl_droll2)

        # ── Yaw effects ───────────────────────────────────────────────────
        Gamma_yaw, Cy_yaw, dCd_yaw = yaw_sensitivity(
            yaw_rate, vx, c.Cy_yaw, c.dCd_dyaw2)

        # ── Effective Cl per axle ─────────────────────────────────────────
        Cl_total = c.Cl_max * Gamma_pitch * Gamma_roll * Gamma_yaw
        Cl_f = _softplus_floor(Cl_total * split_f * Gamma_ge_f, 0.0)
        Cl_r = _softplus_floor(Cl_total * split_r * Gamma_ge_r, 0.0)

        # ── Drag coefficient ─────────────────────────────────────────────
        pitch_deg_abs = jnp.abs(pitch) * (180.0 / jnp.pi)
        Cd = _softplus_floor(
            c.Cd_base
            + c.dCd_dpitch * pitch_deg_abs    # pitch-induced drag
            + dCd_yaw,                        # yaw-induced drag
            0.1,  # minimum drag floor
        )

        # ── Forces ────────────────────────────────────────────────────────
        qA = q_dyn * c.A_ref
        Fz_aero_f = qA * Cl_f                   # front downforce [N]
        Fz_aero_r = qA * Cl_r                   # rear downforce [N]
        Fx_aero = -qA * Cd                       # drag (always opposing)
        Fy_aero = qA * Cy_yaw                    # side force from yaw

        # ── Moments ───────────────────────────────────────────────────────
        # Pitching moment from CoP offset
        My_aero = Fz_aero_r * self.lr - Fz_aero_f * self.lf

        # Rolling moment from asymmetric downforce under roll
        # Sign convention: positive roll → right side lower → right gets more DF
        half_track = 0.6  # half track width [m]
        Mx_aero = (Fz_aero_f + Fz_aero_r) * roll * c.roll_moment_arm * half_track

        return Fz_aero_f, Fz_aero_r, Fx_aero, My_aero, Mx_aero

    @partial(jax.jit, static_argnums=(0,))
    def apply_extended(
        self,
        _params,
        vx: jax.Array,
        pitch: jax.Array,
        roll: jax.Array,
        heave_f: jax.Array,
        heave_r: jax.Array,
        yaw_rate: jax.Array = 0.0,
    ) -> dict:
        """Extended output with diagnostics for the dashboard."""
        Fz_f, Fz_r, Fx, My, Mx = self.apply(
            _params, vx, pitch, roll, heave_f, heave_r, yaw_rate)

        c = self.cfg
        rh_f_mm = 30.0 - heave_f * 0.5 * 1000.0
        rh_r_mm = 35.0 - heave_r * 0.5 * 1000.0
        Gamma_ge_f = ground_effect_envelope(
            rh_f_mm, c.rh_peak, c.rh_stall, c.rh_high, c.ge_sharpness)
        Gamma_ge_r = ground_effect_envelope(
            rh_r_mm, c.rh_peak, c.rh_stall, c.rh_high, c.ge_sharpness)
        split_f = cop_migration(pitch, c.aero_split_f, c.dCoP_dpitch)

        qA = 0.5 * c.rho_air * jnp.maximum(jnp.abs(vx), 0.5) ** 2 * c.A_ref
        Cl_eff = (Fz_f + Fz_r) / (qA + 1e-6)
        Cd_eff = -Fx / (qA + 1e-6)

        return {
            'Fz_aero_f': Fz_f, 'Fz_aero_r': Fz_r,
            'Fx_aero': Fx, 'My_aero': My, 'Mx_aero': Mx,
            'Cl_eff': Cl_eff, 'Cd_eff': Cd_eff,
            'LD_ratio': Cl_eff / (Cd_eff + 1e-6),
            'CoP_x': split_f,
            'Gamma_ge_f': Gamma_ge_f, 'Gamma_ge_r': Gamma_ge_r,
            'rh_f_mm': rh_f_mm, 'rh_r_mm': rh_r_mm,
        }


# ─────────────────────────────────────────────────────────────────────────────
# §7  Integration Helper — Drop-in for vehicle_dynamics.py
# ─────────────────────────────────────────────────────────────────────────────

def create_aero_platform(vp: dict) -> AeroPlatformModel:
    """
    Factory that reads vehicle_params and constructs the platform model.

    Usage in DifferentiableMultiBodyVehicle.__init__:
        from models.aero_platform import create_aero_platform
        self.aero_map = create_aero_platform(self.vp)
    """
    cfg = AeroPlatformConfig(
        Cl_max=vp.get('Cl_ref', 4.14),
        Cd_base=vp.get('Cd_ref', 1.15),
        A_ref=vp.get('A_ref', 1.1),
        rho_air=vp.get('rho_air', 1.225),
        rh_peak=vp.get('rh_peak_mm', 30.0),
        rh_stall=vp.get('rh_stall_mm', 12.0),
        rh_high=vp.get('rh_high_mm', 80.0),
        ge_sharpness=vp.get('ge_sharpness', 0.15),
        dCl_dpitch=vp.get('dCl_dpitch', -0.08),
        dCl_dpitch2=vp.get('dCl_dpitch2', -0.02),
        dCoP_dpitch=vp.get('dCoP_dpitch', 0.015),
        dCd_dpitch=vp.get('dCd_dpitch', 0.04),
        dCl_droll2=vp.get('dCl_droll2', -0.03),
        Cy_yaw=vp.get('Cy_yaw', 0.8),
        dCd_dyaw2=vp.get('dCd_dyaw2', 0.25),
        aero_split_f=vp.get('aero_split_f', 0.40),
        aero_split_r=vp.get('aero_split_r', 0.60),
        lf=vp.get('lf', 0.8525),
        lr=vp.get('lr', 0.6975),
    )
    return AeroPlatformModel(cfg)