"""Strongly-typed parameter containers for the TeR-Twin full-vehicle model.

Conventions (used by EVERY module in this package)
--------------------------------------------------
* Body axes: x forward, y LEFT, z up (ISO 8855). SI units, angles in rad.
* Wheel index order everywhere: 0=FL, 1=FR, 2=RL, 3=RR  (``SIDE = [+1, -1, +1, -1]``).
* ay > 0 is a LEFT turn. Roll angle ``phi`` > 0 = body rolls to the RIGHT (outward in a left turn),
  i.e. the right wheels are in bump. Pitch ``theta`` > 0 = nose down.
* Tyre model convention is the one of ``pacejka_61`` (positive slope): Fy>0 for alpha>0,
  Fx>0 for kappa>0, Mz<0 for alpha>0. alpha = -atan(vy_w / vx_w).
* Wheel inclination handed to the tyre model (``gamma``) is the tilt of the wheel top towards +y.
  Setup sheets use SAE camber (negative = top inboard); ``gamma = SIDE * camber_sae``.
* Toe: positive = toe-in. Steer angle of wheel i from static toe = -SIDE_i * toe_i.

Every dataclass is registered as a JAX pytree (all fields are leaves except ``name``), so whole
parameter sets can be passed through ``jit``/``vmap``/``grad`` and swept with ``dataclasses.replace``.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ter_twin.models.tires import pacejka_61 as mf

G = 9.80665
RHO_AIR = 1.18
SIDE = np.array([1.0, -1.0, 1.0, -1.0])
WHEEL_NAMES = ("FL", "FR", "RL", "RR")


def pytree_dataclass(cls=None, *, static: tuple[str, ...] = ()):
    """Frozen dataclass registered as a pytree. Fields in ``static`` become aux data."""

    def wrap(c):
        c = dataclasses.dataclass(frozen=True)(c)
        names = [f.name for f in dataclasses.fields(c)]
        dyn = [n for n in names if n not in static]
        sta = [n for n in names if n in static]

        def flat(o):
            return tuple(getattr(o, n) for n in dyn), tuple(getattr(o, n) for n in sta)

        def unflat(aux, children):
            obj = object.__new__(c)
            for n, v in zip(dyn, children):
                object.__setattr__(obj, n, v)
            for n, v in zip(sta, aux):
                object.__setattr__(obj, n, v)
            return obj

        jax.tree_util.register_pytree_node(c, flat, unflat)
        return c

    return wrap(cls) if cls is not None else wrap


# --------------------------------------------------------------------------------------
@pytree_dataclass
class ChassisParams:
    mass: float = 290.0            # total mass incl. driver [kg]
    mu_f: float = 11.0             # unsprung mass per FRONT corner (wheel+upright+hub motor) [kg]
    mu_r: float = 11.0             # unsprung mass per REAR corner [kg]
    wheelbase: float = 1.55        # [m]
    cg_to_front: float = 0.8525    # a: CG -> front axle [m]  (front weight fraction = b/L = 0.45)
    h_cg: float = 0.285            # total CG height [m]
    h_unsprung: float = 0.2032     # unsprung CG height ~ wheel radius [m]
    track_f: float = 1.20
    track_r: float = 1.16
    h_rc_f: float = 0.035          # geometric roll-centre heights [m]
    h_rc_r: float = 0.060
    rh0_f: float = 0.035           # static ride height at the aero reference point [m]
    rh0_r: float = 0.050
    ixx: float = 38.0              # sprung roll inertia about CG axis [kg m^2]
    iyy: float = 105.0
    izz: float = 112.0             # total yaw inertia [kg m^2]

    @property
    def b(self) -> float:
        return self.wheelbase - self.cg_to_front

    @property
    def a(self) -> float:
        return self.cg_to_front

    @property
    def ms(self):
        return self.mass - 2.0 * (self.mu_f + self.mu_r)

    @property
    def h_sprung(self):
        return (self.mass * self.h_cg - 2.0 * (self.mu_f + self.mu_r) * self.h_unsprung) / self.ms

    def x_wheels(self):
        return jnp.array([self.a, self.a, -self.b, -self.b])

    def y_wheels(self):
        return jnp.array([0.5 * self.track_f, -0.5 * self.track_f, 0.5 * self.track_r, -0.5 * self.track_r])


@pytree_dataclass
class SuspensionParams:
    k_wheel_f: float = 24000.0     # ride rate at the wheel (spring*MR^2) [N/m]
    k_wheel_r: float = 26000.0
    c_wheel_f: float = 1400.0      # damper rate at the wheel [N s/m]
    c_wheel_r: float = 1500.0
    k_arb_f: float = 3500.0        # ARB roll stiffness contribution [N m/rad]
    k_arb_r: float = 2500.0
    camber_static_f: float = math.radians(-2.2)   # SAE camber [rad]
    camber_static_r: float = math.radians(-1.6)
    toe_static_f: float = math.radians(-0.15)     # toe-in positive (slight toe-out)
    toe_static_r: float = math.radians(0.20)
    camber_bump_f: float = 1.20    # d(camber_sae)/d(-bump) [rad/m] -> roll camber gain = value*track/2
    camber_bump_r: float = 1.00
    bump_steer_f: float = 0.0      # d(toe-in)/d(bump) [rad/m]
    bump_steer_r: float = 0.0
    toe_compliance_f: float = 2.0e-6   # steer per unit tyre Fy [rad/N] (>0: understeer)
    toe_compliance_r: float = 1.0e-6
    camber_compliance_f: float = 1.0e-6  # tilt per unit tyre Fy [rad/N]
    camber_compliance_r: float = 1.0e-6
    anti_f: float = 0.15           # fraction of longitudinal transfer through links (anti-dive)
    anti_r: float = 0.20           # anti-squat
    tau_load: float = 0.025        # lag of the load-transfer accelerations [s]
    elasto_iter: int = 1           # fixed-point passes for compliance steer/camber (static)


@pytree_dataclass
class SteeringParams:
    rack_ratio: float = 0.55       # road-wheel angle per metre of rack travel [rad/m]
    rack_cubic: float = 0.0        # cubic nonlinearity of rack->road-wheel [rad/m^3]
    steer_ratio: float = 4.5       # steering-wheel angle / road-wheel angle [-] (reporting only)
    ackermann: float = 0.6         # 0 = parallel, 1 = 100 % Ackermann
    delta_max: float = math.radians(28.0)


@pytree_dataclass
class AeroParams:
    rh_f_grid: Any = None          # [m], shape (n,)
    rh_r_grid: Any = None          # [m], shape (m,)
    cla_f: Any = None              # front downforce area  ClA_f(rh_f, rh_r) [m^2], shape (n, m)
    cla_r: Any = None              # rear
    cda: Any = None                # drag area
    h_aero: float = 0.40           # height of the drag resultant [m]
    roll_sens: float = 1.5         # ClA multiplier (1 - roll_sens*phi^2)
    drs_dcla_f: float = 0.0        # additive change of ClA when DRS fully open
    drs_dcla_r: float = -0.55
    drs_dcda: float = -0.20
    rho: float = RHO_AIR


@pytree_dataclass
class PowertrainParams:
    gear_ratio: float = 14.0       # motor -> wheel reduction [-]
    eta_drive: float = 0.92        # motor+inverter+gear, battery -> wheel
    eta_regen: float = 0.88
    motor_tq_peak: float = 21.0    # motor shaft torque [N m]
    motor_p_peak: float = 35.0e3   # per-motor mechanical power [W]
    motor_w_max: float = 2094.0    # motor speed limit [rad/s] (20 000 rpm)
    regen_tq_peak: float = 14.0    # [N m] motor side
    regen_p_peak: float = 20.0e3
    regen_w_min: float = 120.0     # no regen below this motor speed [rad/s]
    p_acc_max: float = 80.0e3      # FS rules: accumulator power limit [W]
    p_regen_max: float = 40.0e3    # recuperation limit [W]


@pytree_dataclass
class TireParams:
    r0: float = 0.2032             # unloaded radius [m]
    kz: float = 90.0e3             # vertical stiffness [N/m]
    inertia_w: float = 0.32        # wheel+rotor spin inertia [kg m^2]
    crr: float = 0.015             # rolling-resistance coefficient [-]
    front: Any = None              # MF61Params (scalar)
    rear: Any = None


@pytree_dataclass(static=("name",))
class VehicleParams:
    chassis: ChassisParams
    susp: SuspensionParams
    steer: SteeringParams
    aero: AeroParams
    pt: PowertrainParams
    tire: TireParams
    name: str = "TeR27-4WD"


# --------------------------------------------------------------------------------------
def default_aero_grid(cla_f0: float = 1.45, cla_r0: float = 1.85, cda0: float = 1.10) -> AeroParams:
    """Analytic placeholder map (replace with ``aero_map.yaml`` / CFD fit).

    Lower front ride height -> more front downforce; higher rake (rear up) -> more front/less rear.
    """
    rf = np.linspace(0.015, 0.060, 7)
    rr = np.linspace(0.025, 0.090, 8)
    RF, RR = np.meshgrid(rf, rr, indexing="ij")
    f = 1.0 + 6.0 * (0.035 - RF) + 2.0 * (RR - RF - 0.015)
    r = 1.0 + 3.0 * (0.050 - RR) - 1.5 * (RR - RF - 0.015)
    d = 1.0 + 1.0 * (0.035 - RF) + 0.5 * (RR - 0.05)
    return AeroParams(
        rh_f_grid=jnp.asarray(rf), rh_r_grid=jnp.asarray(rr),
        cla_f=jnp.asarray(cla_f0 * f), cla_r=jnp.asarray(cla_r0 * r), cda=jnp.asarray(cda0 * d),
    )


def default_ter27(front: Any = None, rear: Any = None) -> VehicleParams:
    """Plausible TeR27-4WD parameter set. Tyres default to nominal ``MF61Params()`` if not given."""
    front = front if front is not None else mf.MF61Params()
    rear = rear if rear is not None else front
    return VehicleParams(
        chassis=ChassisParams(), susp=SuspensionParams(), steer=SteeringParams(),
        aero=default_aero_grid(), pt=PowertrainParams(),
        tire=TireParams(front=front, rear=rear),
    )


def replace_in(vp: VehicleParams, section: str, **kw: Any) -> VehicleParams:
    """Immutable nested update: ``replace_in(vp, 'chassis', mass=300.0)``."""
    return dataclasses.replace(vp, **{section: dataclasses.replace(getattr(vp, section), **kw)})