# physics/h_net_icnn.py
# Project-GP — Structurally Passive Neural Hamiltonian Residual
# ═══════════════════════════════════════════════════════════════════════════════
#
#     H_net(q, p, setup) = K(p) · ψ(q, setup) + V(q, setup)
#
# Structural properties guaranteed for ANY weight values:
#   P1  H_net ≥ 0
#   P2  H_net(q_eq, 0, setup) = 0
#   P3  ∇_p H_net(q, 0, setup) = 0
#   P4  pᵀ ∇_p H_net ≥ 0
#
# KEY DESIGN DECISION — why a submodule, not a nested function:
#   K(p) = ICNN(p²) - ICNN(0)  requires calling the same ICNN twice with
#   identical weights. Flax's nn.compact stores params by string name — calling
#   a nested function twice inside one __call__ tries to register e.g.
#   "k_W0_raw" twice → NameInUseError.
#
#   Fix: _KineticICNN is a standalone nn.Module. KineticNet instantiates it
#   once with name="core". Flax binds params on the first call `core(p_sq)`
#   and REUSES the same param dict on the second call `core(zeros)` because
#   the module name is the same. The subtraction K(p²) - K(0) is then
#   computed with provably identical weights.
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.linen as nn


# -----------------------------------------------------------------------------
# Numerically calibrated positive-weight initializer
# -----------------------------------------------------------------------------
# ``normal(0.01)`` followed by ``softplus`` does NOT produce ~0.01 positive
# weights: it produces weights near softplus(0) ~= 0.693.  In a multilayer ICNN
# this compounds across layers and can inflate H_raw by orders of magnitude
# before training.  We initialize the unconstrained raw parameter so that its
# post-softplus value is a deliberately small positive number.
_POSITIVE_W_INIT = 0.10
_POSITIVE_W_RAW0 = float(jnp.log(jnp.expm1(_POSITIVE_W_INIT)))

def _positive_small_init(key, shape, dtype=jnp.float32):
    del key
    return jnp.full(shape, _POSITIVE_W_RAW0, dtype=dtype)



# ─────────────────────────────────────────────────────────────────────────────
# Equilibrium — matches residual_fitting.py _Z_EQ
# ─────────────────────────────────────────────────────────────────────────────

_Z_EQ_DEFAULT: jax.Array = jnp.array(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     0.0128, 0.0128, 0.0142, 0.0142,
     0.0, 0.0, 0.0, 0.0]
)


# ─────────────────────────────────────────────────────────────────────────────
# §1  _KineticICNN — raw all-positive-weight ICNN on x = p²
# ─────────────────────────────────────────────────────────────────────────────

class _KineticICNN(nn.Module):
    """
    All-non-negative-weight ICNN. Monotone non-decreasing in each input
    component when inputs are ≥ 0 (which p² always is).

    Extracted as a standalone module so KineticNet can call it twice
    (at p² and at 0) with provably identical parameters — Flax reuses
    the param dict for any submodule invoked under the same name.
    """
    hidden: tuple[int, ...]

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        # Input layer — all weights non-negative via softplus
        W0_raw = self.param("W0_raw", _positive_small_init,
                            (x.shape[-1], self.hidden[0]))
        b0 = self.param("b0", nn.initializers.zeros, (self.hidden[0],))
        z = nn.softplus(x @ nn.softplus(W0_raw) + b0)

        for i, h in enumerate(self.hidden[1:], start=1):
            Wz_raw = self.param(f"Wz{i}_raw", _positive_small_init,
                                (z.shape[-1], h))
            Wx_raw = self.param(f"Wx{i}_raw", _positive_small_init,
                                (x.shape[-1], h))
            b = self.param(f"b{i}", nn.initializers.zeros, (h,))
            z = nn.softplus(z @ nn.softplus(Wz_raw) + x @ nn.softplus(Wx_raw) + b)

        w_raw = self.param("w_out_raw", _positive_small_init, (z.shape[-1],))
        return jnp.sum(z * nn.softplus(w_raw))


# ─────────────────────────────────────────────────────────────────────────────
# §2  KineticNet — K(p) with P1–P4 via squared input + submodule reuse
# ─────────────────────────────────────────────────────────────────────────────

class KineticNet(nn.Module):
    """
    Structurally passive kinetic residual with dimensionless momentum input.

    The ICNN is evaluated on

        p_tilde = p / p_scale

    before forming ``p_tilde**2``.  This is only a numerical/unit normalization:
    it does not alter the positivity proof because ``p_scale`` is strictly positive.

    ``p_scale`` is stored as static module metadata (not a trainable parameter).
    For the vehicle model it should be chosen consistently with the 14 generalized
    masses/inertias, e.g. ``p_scale = M_diag * v_scale``.

    P1–P4 remain algebraic for every positive ``p_scale``.
    """
    hidden: tuple[int, ...]
    p_scale: float | tuple[float, ...] = 5000.0

    @nn.compact
    def __call__(self, p: jax.Array) -> jax.Array:
        scale = jnp.asarray(self.p_scale, dtype=p.dtype)
        scale = jnp.maximum(scale, jnp.asarray(1e-12, dtype=p.dtype))
        p_tilde = p / scale
        p_sq = p_tilde * p_tilde
        core = _KineticICNN(self.hidden, name="core")
        # Shared ICNN parameters: exact subtraction of the same core at zero.
        return core(p_sq) - core(jnp.zeros_like(p_sq))


# ─────────────────────────────────────────────────────────────────────────────
# §3  PsiGate — ψ(q, setup) ≥ 0
# ─────────────────────────────────────────────────────────────────────────────

class PsiGate(nn.Module):
    hidden: tuple[int, ...]

    @nn.compact
    def __call__(self, q: jax.Array, setup: jax.Array) -> jax.Array:
        h = jnp.concatenate([q, setup])
        for i, w in enumerate(self.hidden):
            h = nn.swish(nn.Dense(w, name=f"dense_{i}")(h))
        return nn.softplus(nn.Dense(1, name="out")(h)[0])


# ─────────────────────────────────────────────────────────────────────────────
# §4  _PotentialICNN — ICNN core for V(q, setup)
# ─────────────────────────────────────────────────────────────────────────────

class _PotentialICNN(nn.Module):
    """
    Standard ICNN convex in q (hidden-to-hidden weights non-negative).
    Input weights unconstrained because q can be negative.
    """
    hidden: tuple[int, ...]

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        z = nn.softplus(nn.Dense(self.hidden[0], name="b0_x")(x))

        for i, h in enumerate(self.hidden[1:], start=1):
            Wz_raw = self.param(f"Wz{i}_raw", _positive_small_init,
                                (z.shape[-1], h))
            z = nn.softplus(
                z @ nn.softplus(Wz_raw) + nn.Dense(h, name=f"b{i}_x")(x)
            )

        w_raw = self.param("w_out_raw", nn.initializers.normal(0.01), (z.shape[-1],))
        return jnp.sum(z * nn.softplus(w_raw))


# ─────────────────────────────────────────────────────────────────────────────
# §5  PotentialNet — V(q, setup), grounded at q_eq
# ─────────────────────────────────────────────────────────────────────────────

class PotentialNet(nn.Module):
    """
    Convex, equilibrium-grounded Bregman potential.

    Let g(z, setup) be convex in z and let z_eq = g-input evaluated at
    q = q_eq. The potential is the Bregman divergence

        V(q, setup)
          = g(z) - g(z_eq) - ∇_z g(z_eq)^T (z - z_eq)

    with
        z = gamma(setup) * (q - q_eq) + beta(setup)
        z_eq = beta(setup)

    Since g is convex:

        V(q, setup) >= 0

    for all q, while at equilibrium:

        V(q_eq, setup) = 0
        ∇_q V(q_eq, setup) = 0

    The FiLM map is affine in q, so convexity in q is preserved.
    """

    hidden: tuple[int, ...]
    q_dim: int
    setup_dim: int

    @nn.compact
    def __call__(self, q: jax.Array, setup: jax.Array) -> jax.Array:
        q_centered = q - _Z_EQ_DEFAULT

        # ---------------------------------------------------------------
        # 1. Setup-conditioned affine FiLM transformation
        # ---------------------------------------------------------------
        setup_duplicated = jnp.concatenate([setup, -setup], axis=-1)

        film = nn.Dense(
            2 * self.q_dim,
            name="film",
            kernel_init=jax.nn.initializers.glorot_normal(),
            bias_init=jax.nn.initializers.zeros,
        )(setup_duplicated)

        gamma = 1.0 + 0.5 * jnp.tanh(film[: self.q_dim])
        beta = 0.30 * jnp.tanh(film[self.q_dim:])

        # z(q,s)
        z = gamma * q_centered + beta

        # z(q_eq,s)
        z_eq = beta

        # ---------------------------------------------------------------
        # 2. Shared convex ICNN
        # ---------------------------------------------------------------
        core = _PotentialICNN(self.hidden, name="core")

        g_z = core(z)
        g_eq = core(z_eq)

        # ---------------------------------------------------------------
        # 3. Bregman tangent-plane subtraction
        #
        # D_g(z,z_eq) = g(z) - g(z_eq)
        #               - grad_g(z_eq)^T (z-z_eq)
        #
        # stop_gradient is intentionally NOT used here:
        # the tangent gradient is part of the exact mathematical
        # definition of the Bregman divergence.
        # ---------------------------------------------------------------
        grad_g_eq = jax.grad(
            lambda z_: core(z_)
        )(z_eq)

        dz = z - z_eq

        return g_z - g_eq - jnp.dot(grad_g_eq, dz)

# ─────────────────────────────────────────────────────────────────────────────
# §6  PassiveHNet — full residual H_net(q, p, setup)
# ─────────────────────────────────────────────────────────────────────────────

class PassiveHNet(nn.Module):
    """
    H_net(q, p, setup) = K(p) · ψ(q, setup) + V(q, setup)

    Structural properties of H_raw (guaranteed for ANY weight values):
      P1  H_raw ≥ 0                   K≥0 (KineticNet), ψ≥0 (PsiGate), V≥0 (Bregman)
      P2  H_raw(q_eq, 0, s) = 0       K(0)=0, V(q_eq,s)=0 (Bregman)
      P3  ∇_p H_raw(q, 0, s) = 0      ∂K/∂p|₀=0 (KineticNet)
      P4  pᵀ ∇_p H_raw ≥ 0           pᵀ ∂K/∂p ≥ 0 (KineticNet)
      P5  ∇_q V(q_eq, s) = 0          Bregman divergence property

    NOTE ON SATURATION CAP:
      The tanh saturation H = h_cap · tanh(H_raw / h_cap) is a NUMERICAL
      SAFETY MEASURE to prevent float32 overflow in extreme maneuvers.
      It is NOT a structural passivity guarantee. All passivity properties
      (P1–P5) hold for H_raw. The cap preserves P2, P3, P5 exactly and
      preserves P1, P4 approximately when H_raw ≪ h_cap (typical regime).

    Drop-in for NeuralEnergyLandscape:
        model.apply({"params": params}, q, p, setup)  →  scalar [J]

    Returns H_neural only. Callers that need H_total must add T_prior + V_struct.
    See vehicle_dynamics.py _compute_derivatives for the correct wrapper.
    """
    q_dim:      int          = 14
    p_dim:      int          = 14
    setup_dim:  int          = 28
    k_hidden:   tuple[int, ...] = (64, 64, 32)
    v_hidden:   tuple[int, ...] = (64, 64, 32)
    psi_hidden: tuple[int, ...] = (64, 32)
    h_cap:      float        = 50_000.0
    # Generalized-momentum normalization. 5000 is a safe standalone default;
    # vehicle_dynamics.py overrides it with M_diag * 20 m/s.
    p_scale:    float | tuple[float, ...] = 5000.0
    # Explicit validation/production view of the same network. This is static
    # configuration, not a trainable parameter.
    output_mode: str = "capped"

    @nn.compact
    def __call__(self, q: jax.Array, p: jax.Array, setup: jax.Array) -> jax.Array:
        K   = KineticNet(self.k_hidden, self.p_scale,                  name="K_net")(p)
        psi = PsiGate(self.psi_hidden,                               name="psi_gate")(q, setup)
        V   = PotentialNet(self.v_hidden, self.q_dim, self.setup_dim, name="V_net")(q, setup)

        H_raw = K * psi + V

        if self.output_mode == "raw":
            # Structural passivity verification must use this branch.
            return H_raw
        if self.output_mode == "potential":
            # Direct access for P5/P6 without relying on Linen bound-submodule names.
            return V
        if self.output_mode != "capped":
            raise ValueError(
                f"Unknown output_mode={self.output_mode!r}; "
                "expected 'capped', 'raw', or 'potential'."
            )

        # NUMERICAL SATURATION ONLY — not a structural guarantee.
        return self.h_cap * jnp.tanh(H_raw / self.h_cap)


# ─────────────────────────────────────────────────────────────────────────────
# §7  Init helper
# ─────────────────────────────────────────────────────────────────────────────

def init_passive_hnet(
    rng: jax.Array,
    q_dim: int = 14, p_dim: int = 14, setup_dim: int = 28,
    **kwargs,
) -> tuple[PassiveHNet, dict]:
    model  = PassiveHNet(q_dim=q_dim, p_dim=p_dim, setup_dim=setup_dim, **kwargs)
    params = model.init(
        rng,
        jnp.zeros(q_dim),
        jnp.zeros(p_dim),
        jnp.zeros(setup_dim),
    )["params"]
    return model, params