# models/tire_thermal_3d.py
# ═══════════════════════════════════════════════════════════════════════════════
# Project-GP — 3D Tire Thermal Model with Lateral Distribution
# ═══════════════════════════════════════════════════════════════════════════════
#
# PROBLEM (Flaw #3):
#   The current 5-node thermal model (tire_model.py §4.2) averages the front
#   and rear axles, producing 10 states: 3 surface ribs + gas + core per axle.
#   It cannot differentiate between a boiling outer shoulder and a cold inner
#   edge on the same wheel. The TV and setup optimizer are blind to camber-
#   induced thermal asymmetry — the dominant failure mode in endurance.
#
# SOLUTION:
#   Per-corner 7-node thermal model with full lateral distribution:
#
#   Layer 0 (surface):  T_inner, T_mid, T_outer  — 3 rib surface temps
#   Layer 1 (bulk):     T_bulk                    — tread rubber bulk
#   Layer 2 (carcass):  T_carcass                 — structural carcass
#   Layer 3 (gas):      T_gas                     — inflation gas
#   Layer 4 (contact):  T_track_contact           — track surface under patch
#
#   Total: 4 corners × 7 nodes = 28 thermal states
#
#   Heat generation is CAMBER-WEIGHTED: under negative camber, the outer
#   rib carries more normal load → more slip energy → hotter.
#
#   Through-thickness conduction: surface ↔ bulk ↔ carcass ↔ gas
#   Lateral conduction: inner ↔ mid ↔ outer (within surface layer)
#   Track contact: surface ↔ track (conduction through contact patch)
#
# STATE EXTENSION:
#   Current: x[28:38] = 10 thermal states (axle-averaged)
#   New:     x[28:56] = 28 thermal states (per-corner, 7 each)
#   This extends the 46-DOF state to 74-DOF (net +18 states).
#
# JAX CONTRACT: Pure JAX, JIT-safe, C∞, vmapped over corners.
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import NamedTuple
from functools import partial


# ─────────────────────────────────────────────────────────────────────────────
# §1  Thermal Material Properties
# ─────────────────────────────────────────────────────────────────────────────

class TireThermalProps(NamedTuple):
    """Hoosier R25B thermal material properties."""
    # Conductivity
    k_rubber: float = 0.25          # W/m/K — rubber thermal conductivity
    k_carcass: float = 0.35         # W/m/K — nylon/steel cord carcass
    k_lateral: float = 0.15         # W/m/K — lateral (within tread) conduction

    # Volumetric heat capacity
    rho_c_rubber: float = 2.0e6    # J/m³/K — tread compound
    rho_c_carcass: float = 1.5e6   # J/m³/K — carcass (lighter)

    # Geometry
    tread_thickness: float = 0.006  # m — tread depth
    carcass_thickness: float = 0.003  # m — structural carcass
    rib_width: float = 0.055        # m — width of each rib (inner/mid/outer)
    contact_length: float = 0.15    # m — contact patch length
    contact_width: float = 0.165    # m — total contact patch width
    tire_radius: float = 0.2045     # m

    # Convection
    h_surface_air: float = 80.0     # W/m²/K — tread-to-air convection
    h_track_contact: float = 500.0  # W/m²/K — tread-to-track conduction
    h_gas_internal: float = 12.0    # W/m²/K — gas-to-carcass internal

    # Thermal masses
    V_rib: float = 0.0004           # m³ — volume per rib (~55mm × 150mm × 6mm × π ratio)
    V_bulk: float = 0.0008          # m³ — bulk tread volume
    V_carcass: float = 0.0012       # m³ — carcass volume
    V_gas: float = 0.008            # m³ — gas volume

    # Environment
    T_env: float = 25.0             # °C — ambient air temperature
    T_track: float = 35.0           # °C — track surface temperature

    # Friction
    mu_nominal: float = 1.5         # nominal friction coefficient


class CornerThermalState(NamedTuple):
    """7-node thermal state for one corner."""
    T_inner: jax.Array    # surface inner rib [°C]
    T_mid: jax.Array      # surface middle rib [°C]
    T_outer: jax.Array    # surface outer rib [°C]
    T_bulk: jax.Array     # tread bulk [°C]
    T_carcass: jax.Array  # structural carcass [°C]
    T_gas: jax.Array      # inflation gas [°C]
    T_contact: jax.Array  # track contact temperature [°C]

    @classmethod
    def default(cls, T_env: float = 25.0) -> 'CornerThermalState':
        return cls(
            T_inner=jnp.array(T_env + 5.0),
            T_mid=jnp.array(T_env + 5.0),
            T_outer=jnp.array(T_env + 5.0),
            T_bulk=jnp.array(T_env + 3.0),
            T_carcass=jnp.array(T_env + 2.0),
            T_gas=jnp.array(T_env + 1.0),
            T_contact=jnp.array(T_env + 10.0),
        )


# ─────────────────────────────────────────────────────────────────────────────
# §2  Camber-Weighted Load Distribution
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def camber_load_distribution(
    gamma_rad: jax.Array,
    Fz_total: jax.Array,
) -> jax.Array:
    """
    Compute per-rib normal load based on camber angle.

    Under negative camber (typical FS: -1.5° to -3.5°), the outer rib
    carries more load due to contact patch migration. This drives
    asymmetric heat generation.

    Returns: (3,) = [Fz_inner, Fz_mid, Fz_outer]

    Model: quadratic camber pressure distribution:
      p(y) = p_mean · (1 + α · y/w)   where y is lateral position
      α = f(γ), positive γ loads outer edge

    For negative camber (γ < 0): outer rib is loaded more.
    """
    gamma_deg = gamma_rad * (180.0 / jnp.pi)

    # Camber shift: positive = load shifts to outer rib
    # Typical FS: γ = -2° → shift_factor ≈ +0.15 (outer gets 15% more)
    shift_factor = -0.08 * gamma_deg  # negative gamma → positive shift

    # Smooth clamping
    shift = jnp.tanh(shift_factor)  # ∈ (-1, 1)

    # Distribution: [inner, mid, outer] sums to 1.0
    w_inner = (1.0 - shift) / 3.0
    w_mid = 1.0 / 3.0
    w_outer = (1.0 + shift) / 3.0

    # Normalize to sum exactly to 1.0
    total = w_inner + w_mid + w_outer
    weights = jnp.array([w_inner, w_mid, w_outer]) / total

    return weights * Fz_total


# ─────────────────────────────────────────────────────────────────────────────
# §3  Flash Temperature (per-rib)
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def compute_rib_flash_temperature(
    mu: jax.Array,
    Fz_rib: jax.Array,
    V_slide: jax.Array,
    k_rubber: float = 0.25,
    rho_c: float = 2.0e6,
    contact_half_length: float = 0.075,
    rib_width: float = 0.055,
) -> jax.Array:
    """
    Jaeger flash temperature per rib.

    T_flash = (q · a) / (k · √(π · V · a / α))

    where q = heat flux = μ·Fz·V / (2·a·w) over the rib contact area.
    """
    thermal_diff = k_rubber / rho_c
    V_safe = jnp.maximum(jnp.abs(V_slide), 1e-3)

    # Heat flux over rib area
    q_flux = (mu * jnp.abs(Fz_rib) * V_safe) / (
        2.0 * contact_half_length * rib_width + 1e-9)

    T_flash = (q_flux * contact_half_length) / (
        k_rubber * jnp.sqrt(
            jnp.pi * (V_safe * contact_half_length) / (thermal_diff + 1e-9)
        ) + 1e-9
    )
    return jnp.clip(T_flash, 0.0, 400.0)


# ─────────────────────────────────────────────────────────────────────────────
# §4  Per-Corner 7-Node Thermal ODE
# ─────────────────────────────────────────────────────────────────────────────

@partial(jax.jit, static_argnums=())
def corner_thermal_derivatives(
    T_state: jax.Array,      # (7,) = [T_inner, T_mid, T_outer, T_bulk, T_carcass, T_gas, T_contact]
    Fz: jax.Array,            # normal load [N]
    kappa: jax.Array,          # longitudinal slip ratio
    alpha: jax.Array,          # slip angle [rad]
    gamma: jax.Array,          # camber angle [rad]
    Vx: jax.Array,             # longitudinal velocity [m/s]
    omega: jax.Array,          # wheel angular velocity [rad/s]
    props: TireThermalProps = TireThermalProps(),
) -> jax.Array:
    """
    7-node thermal ODE for a single corner.

    Heat flow paths:
      1. Friction → surface ribs (camber-weighted)
      2. Flash spike → surface (Jaeger transient conduction)
      3. Surface ↔ bulk (through-thickness conduction)
      4. Bulk ↔ carcass (conduction)
      5. Carcass ↔ gas (internal convection, Gay-Lussac)
      6. Surface ↔ air (external convection, rotational velocity dependent)
      7. Surface ↔ track (contact conduction, velocity dependent)
      8. Inner ↔ mid ↔ outer (lateral conduction within tread)
      9. Hysteresis → carcass (internal heat generation from cyclic deformation)

    Returns: dT/dt (7,) in °C/s
    """
    T_inner, T_mid, T_outer, T_bulk, T_carcass, T_gas, T_contact = (
        T_state[0], T_state[1], T_state[2], T_state[3],
        T_state[4], T_state[5], T_state[6])

    # ── Thermal capacitances ──────────────────────────────────────────────
    C_rib = props.rho_c_rubber * props.V_rib          # J/K per rib
    C_bulk = props.rho_c_rubber * props.V_bulk
    C_carcass = props.rho_c_carcass * props.V_carcass
    C_gas = 1004.0 * 0.010  # J/K (air at ~10g in tire)

    # ── Contact patch area per rib ────────────────────────────────────────
    A_rib = props.contact_length * props.rib_width
    A_total = props.contact_length * props.contact_width

    # ── Sliding velocity ──────────────────────────────────────────────────
    Vx_safe = jnp.maximum(jnp.abs(Vx), 1.0)
    V_slide_lon = Vx_safe * jnp.abs(kappa)
    V_slide_lat = Vx_safe * jnp.abs(jnp.tan(alpha))
    V_slide = jnp.sqrt(V_slide_lon ** 2 + V_slide_lat ** 2 + 1e-6)

    # ── Camber-weighted load distribution ─────────────────────────────────
    Fz_ribs = camber_load_distribution(gamma, Fz)  # (3,)

    # ── Flash temperatures per rib ────────────────────────────────────────
    T_flash = jax.vmap(
        lambda fz: compute_rib_flash_temperature(
            props.mu_nominal, fz, V_slide, props.k_rubber,
            props.rho_c_rubber, props.contact_length * 0.5, props.rib_width)
    )(Fz_ribs)

    # ── Frictional heat generation per rib [W] ────────────────────────────
    Q_fric_ribs = props.mu_nominal * Fz_ribs * V_slide  # (3,) [W]

    # ── Rotational convection enhancement ─────────────────────────────────
    # h_conv increases with wheel speed (forced convection)
    V_periph = jnp.abs(omega) * props.tire_radius
    h_conv_eff = props.h_surface_air * (1.0 + 0.5 * V_periph / (20.0 + 1e-6))

    # ── Contact time fraction ─────────────────────────────────────────────
    # Each rib is in contact for fraction of the rotation
    f_contact = props.contact_length / (2.0 * jnp.pi * props.tire_radius + 1e-6)
    f_free = 1.0 - f_contact

    # ── Conduction resistances ────────────────────────────────────────────
    R_surf_bulk = props.tread_thickness / (props.k_rubber * A_total + 1e-9)
    R_bulk_carc = props.carcass_thickness / (props.k_carcass * A_total + 1e-9)
    R_lateral = props.rib_width / (props.k_lateral * props.contact_length *
                                    props.tread_thickness + 1e-9)

    T_ribs = jnp.array([T_inner, T_mid, T_outer])

    # ── Surface rib ODEs ──────────────────────────────────────────────────
    # dT/dt = (Q_fric + Q_flash + Q_track - Q_conv - Q_cond_to_bulk ± Q_lateral) / C_rib

    # Friction heat input
    dT_fric = Q_fric_ribs / (C_rib + 1e-6)

    # Flash temperature coupling (Jaeger transient)
    dT_flash = h_conv_eff * A_rib * f_contact * (T_flash - T_ribs) / (C_rib + 1e-6)

    # Track contact conduction (during contact phase)
    dT_track = props.h_track_contact * A_rib * f_contact * (T_contact - T_ribs) / (C_rib + 1e-6)

    # Air convection (during free phase)
    dT_conv = -h_conv_eff * A_rib * f_free * (T_ribs - props.T_env) / (C_rib + 1e-6)

    # Through-thickness conduction to bulk
    dT_to_bulk = -(T_ribs - T_bulk) / (R_surf_bulk * C_rib + 1e-6)

    # Lateral conduction: inner↔mid, mid↔outer
    dT_lat_inner = (T_mid - T_inner) / (R_lateral * C_rib + 1e-6)
    dT_lat_mid = ((T_inner - T_mid) + (T_outer - T_mid)) / (R_lateral * C_rib + 1e-6)
    dT_lat_outer = (T_mid - T_outer) / (R_lateral * C_rib + 1e-6)

    dT_surface = dT_fric + dT_flash + dT_track + dT_conv + dT_to_bulk
    dT_inner_total = dT_surface[0] + dT_lat_inner
    dT_mid_total = dT_surface[1] + dT_lat_mid
    dT_outer_total = dT_surface[2] + dT_lat_outer

    # ── Bulk ODE ──────────────────────────────────────────────────────────
    T_surf_avg = (T_inner + T_mid + T_outer) / 3.0
    dT_bulk = ((T_surf_avg - T_bulk) / (R_surf_bulk * C_bulk + 1e-6)
               - (T_bulk - T_carcass) / (R_bulk_carc * C_bulk + 1e-6))

    # ── Carcass ODE ───────────────────────────────────────────────────────
    # Hysteresis heat generation: proportional to deformation rate × load
    Q_hysteresis = 0.05 * Fz * jnp.abs(omega) * props.tire_radius * 0.01  # [W]
    dT_carcass = ((T_bulk - T_carcass) / (R_bulk_carc * C_carcass + 1e-6)
                  + Q_hysteresis / (C_carcass + 1e-6)
                  - props.h_gas_internal * A_total * (T_carcass - T_gas) / (C_carcass + 1e-6))

    # ── Gas ODE (Gay-Lussac coupling) ─────────────────────────────────────
    dT_gas = props.h_gas_internal * A_total * (T_carcass - T_gas) / (C_gas + 1e-6)

    # ── Track contact ODE ─────────────────────────────────────────────────
    # Track surface heats up under tire, cools by conduction into asphalt bulk
    T_surf_in_contact = (T_inner + T_mid + T_outer) / 3.0
    dT_contact = (
        props.h_track_contact * A_total * f_contact * (T_surf_in_contact - T_contact)
        - 200.0 * (T_contact - props.T_track)  # conduction into asphalt bulk
    ) / (3000.0 + 1e-6)  # track thermal mass [J/K] (asphalt cap)

    dT = jnp.array([
        dT_inner_total, dT_mid_total, dT_outer_total,
        dT_bulk, dT_carcass, dT_gas, dT_contact,
    ])
    return jnp.clip(dT, -500.0, 500.0)


# ─────────────────────────────────────────────────────────────────────────────
# §5  4-Corner Vectorized Interface
# ─────────────────────────────────────────────────────────────────────────────

@partial(jax.jit, static_argnums=())
def four_corner_thermal_derivatives(
    T_all: jax.Array,          # (4, 7) per-corner thermal states
    Fz_corners: jax.Array,     # (4,) normal loads [N]
    kappa: jax.Array,           # (4,) longitudinal slip
    alpha: jax.Array,           # (4,) slip angles [rad]
    gamma: jax.Array,           # (4,) camber angles [rad]
    Vx: jax.Array,              # scalar longitudinal velocity [m/s]
    omega: jax.Array,           # (4,) wheel speeds [rad/s]
    props: TireThermalProps = TireThermalProps(),
) -> jax.Array:
    """
    Vectorized 4-corner thermal ODE.

    Args:
        T_all: (4, 7) thermal states per corner
        Fz_corners: (4,) vertical loads
        kappa: (4,) longitudinal slip ratios
        alpha: (4,) slip angles [rad]
        gamma: (4,) camber angles [rad]
        Vx: longitudinal velocity [m/s]
        omega: (4,) wheel angular velocities [rad/s]

    Returns:
        dT_all: (4, 7) thermal derivatives per corner
    """
    return jax.vmap(
        lambda t, fz, k, a, g, w: corner_thermal_derivatives(
            t, fz, k, a, g, Vx, w, props)
    )(T_all, Fz_corners, kappa, alpha, gamma, omega)


# ─────────────────────────────────────────────────────────────────────────────
# §6  Thermal Grip Factor (per-corner, rib-resolved)
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def thermal_grip_factor_3d(
    T_ribs: jax.Array,    # (3,) inner/mid/outer surface temps [°C]
    gamma: jax.Array,      # camber angle [rad]
    T_opt: float = 85.0,
    beta_T: float = 5e-4,
) -> jax.Array:
    """
    Per-corner thermal grip factor using camber-weighted rib temperatures.

    The effective temperature is a camber-weighted average of the rib temps,
    reflecting that the loaded rib dominates grip.

    μ_T = exp(-β_T · (T_eff - T_opt)²)
    """
    # Camber-weighted average (same logic as load distribution)
    gamma_deg = gamma * (180.0 / jnp.pi)
    shift = jnp.tanh(-0.08 * gamma_deg)
    weights = jnp.array([(1.0 - shift) / 3.0, 1.0 / 3.0, (1.0 + shift) / 3.0])
    weights = weights / (jnp.sum(weights) + 1e-8)

    T_eff = jnp.sum(weights * T_ribs)
    return jnp.exp(-beta_T * (T_eff - T_opt) ** 2)


@jax.jit
def four_corner_thermal_grip(
    T_all: jax.Array,       # (4, 7) per-corner thermal states
    gamma: jax.Array,        # (4,) camber angles [rad]
    T_opt: float = 85.0,
    beta_T: float = 5e-4,
) -> jax.Array:
    """
    Returns (4,) thermal grip factors, one per corner.
    """
    T_ribs = T_all[:, :3]   # (4, 3) surface inner/mid/outer
    return jax.vmap(
        lambda t, g: thermal_grip_factor_3d(t, g, T_opt, beta_T)
    )(T_ribs, gamma)


# ─────────────────────────────────────────────────────────────────────────────
# §7  Backward-Compatibility Bridge
# ─────────────────────────────────────────────────────────────────────────────

def pack_to_legacy_10(T_all_4x7: jax.Array) -> jax.Array:
    """
    Convert (4, 7) per-corner thermal state to legacy (10,) axle-averaged
    layout for compatibility with existing vehicle_dynamics.py state vector.

    Legacy layout:
      [0:3] = T_surf_inner/mid/outer_f (average of FL and FR)
      [3]   = T_gas_f (average)
      [4:7] = T_surf_inner/mid/outer_r (average of RL and RR)
      [7]   = T_gas_r (average)
      [8]   = T_core_f (average of bulk+carcass front)
      [9]   = T_core_r (average of bulk+carcass rear)
    """
    T_f = (T_all_4x7[0] + T_all_4x7[1]) * 0.5  # average FL, FR
    T_r = (T_all_4x7[2] + T_all_4x7[3]) * 0.5  # average RL, RR

    return jnp.array([
        T_f[0], T_f[1], T_f[2],   # front surface ribs
        T_f[5],                     # front gas
        T_r[0], T_r[1], T_r[2],   # rear surface ribs
        T_r[5],                     # rear gas
        (T_f[3] + T_f[4]) * 0.5,  # front "core" = avg(bulk, carcass)
        (T_r[3] + T_r[4]) * 0.5,  # rear "core"
    ])


def unpack_from_legacy_10(T_legacy_10: jax.Array) -> jax.Array:
    """
    Expand legacy (10,) to (4, 7) per-corner with uniform left-right split.
    Used for initialization from existing state vectors.
    """
    T_sf = T_legacy_10[0:3]     # front surface ribs
    T_gf = T_legacy_10[3]       # front gas
    T_sr = T_legacy_10[4:7]     # rear surface ribs
    T_gr = T_legacy_10[7]       # rear gas
    T_cf = T_legacy_10[8]       # front core
    T_cr = T_legacy_10[9]       # rear core

    # Expand to per-corner (7 nodes each)
    corner_f = jnp.array([T_sf[0], T_sf[1], T_sf[2], T_cf, T_cf, T_gf, 35.0])
    corner_r = jnp.array([T_sr[0], T_sr[1], T_sr[2], T_cr, T_cr, T_gr, 35.0])

    return jnp.array([corner_f, corner_f, corner_r, corner_r])