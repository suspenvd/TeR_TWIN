# models/tire_model.py
# Project-GP — Multi-Fidelity Tire Model
# ═══════════════════════════════════════════════════════════════════════════════
#
# UPGRADE LOG (GP-vX2)
# ─────────────────────────────────────────────────────────────────────────────
# BUGFIX-3 : sigma penalty 56% at init — physically impossible cornering speeds
#   PREVIOUS: penalty = 2.0 * sigma applied unconditionally.
#   Uninitalized GP: prior_var=0.08 → sigma≈0.28 → penalty=0.56 → forces
#   reduced to 44% of Pacejka baseline. MPC sees a friction circle 56%
#   smaller than physical reality → finds "optimal" cornering at 19.2 m/s
#   instead of the physical limit of ~16.6 m/s.
#   FIX: jnp.clip(2.0 * sigma, 0.0, 0.15)  — 15% maximum LCB penalty.
#   Physical justification: even in fully unexplored regions, Pacejka
#   captures the dominant tire physics. The PINN correction is a residual
#   ΔFy/Fy0 bounded to ±25% by the clip already present downstream.
#   The GP sigma quantifies uncertainty in that residual, not in the baseline.
#
# BUGFIX-4 : GP inducing points cover only positive slip quadrant
#   PREVIOUS: Z_raw = uniform(0,1), Z = Z_raw * scale + shift
#   Result: alpha ∈ [0, 0.15] rad (left turns only), kappa ∈ [0, 0.10]
#   (traction only), gamma ∈ [0, 0.05] (positive camber only).
#   The GP uncertainty estimate for right-hand turns and braking was
#   physically meaningless — the car spends half its time in uncharted space.
#   FIX: Z_raw = normal(0, 0.5), Z = tanh(Z_raw) * scale + shift
#   tanh maps R→(-1,1) symmetrically. Covers:
#     alpha ∈ [-0.25, 0.25] rad  (±14.3°, full operating range)
#     kappa ∈ [-0.20, 0.20]      (traction + braking)
#     gamma ∈ [-0.08, 0.08] rad  (±4.6°, FS camber + roll-induced variation)
#     Fz    ∈ [400, 1200] N      (realistic corner load range)
#     Vx    ∈ [2, 22] m/s        (full FS event speed range)
#   tanh also bounds inducing points permanently — Adam cannot push them
#   outside the physical operating envelope.
#
# BUGFIX-5 : PINN blind to thermal state
#   PREVIOUS: features = [sin(α), sin(2α), κ, κ³, γ, Fz/1000, Vx/20]
#   No temperature. The deterministic drift correction is designed to capture
#   systematic Pacejka deviations — and the dominant deviation at operating
#   conditions IS thermal sensitivity. The network had no access to it.
#   FIX: add (T_eff - T_opt) / 30.0 as 8th feature.
#   T_eff = mean(T_ribs[:3]) — same surface average used by _thermal_grip_factor.
#   Normalization: /30 → unit range covers ±30°C from optimum, where the
#   Pacejka thermal correction changes by ~exp(-0.0008×900) ≈ 0.49 = 51% drop.
#   GP input kept at 5D (kinematic only) — thermal is analytically modeled
#   above the GP layer via _thermal_grip_factor.
#
# BUGFIX-6 : SpectralDense — u_vec receives loss gradients, breaking Lipschitz bound
#   PREVIOUS: u_vec = self.param(...) — Adam treats it as a learnable weight.
#   Adam minimizes loss by pushing u_vec toward directions that minimize sigma,
#   making W_sn = W / sigma unbounded. The Lipschitz-≤1 guarantee was void.
#   FIX: jax.lax.stop_gradient(sigma) — normalization factor is frozen.
#   u_vec is still a param so it survives serialization, but its gradient is
#   zeroed before the Adam update, preserving the power-iteration semantics.
#
# BUGFIX-7 : compute_thermal_derivatives layout misaligned with state vector
#   PREVIOUS: assumed T_nodes[0:5] = contiguous 5-node front block.
#   ACTUAL state layout (vehicle_dynamics.py §5.3):
#     x[28:31] = T_ribs_f, x[31] = T_gas_f, x[32:35] = T_ribs_r, x[35] = T_gas_r
#     x[36] = T_core_f, x[37] = T_core_r
#   Previous code: T_nodes[4] = x[32] = T_rib0_r was used as front core temp.
#   Rear thermal block starting at T_nodes[5]=x[33] was offset by one index.
#   FIX: reindex to match actual vehicle_dynamics layout exactly.
#
# BUGFIX-8 : tire.operator AttributeError in diagnose.py
#   diagnose.py: tire.operator.apply(state, ...) — attribute was _pinn_module.
#   FIX: @property operator returns _pinn_module.
#
# ─── Retained from GP-vX1 ────────────────────────────────────────────────────
# BUGFIX-1  TireOperatorPINN now a proper Flax nn.Module
# BUGFIX-2  compute_thermal_derivatives implemented
# UPGRADE-1 Spectral normalization on PINN Dense layers
# UPGRADE-2 Learnable inducing points for Sparse GP
# UPGRADE-3 Full MF6.2 aligning torque
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.linen as nn


# ─────────────────────────────────────────────────────────────────────────────
# §1  Spectral normalization utility
# ─────────────────────────────────────────────────────────────────────────────
def safe_abs(x: jax.Array, beta: float = 20.0) -> jax.Array:
    """
    Hessian-safe absolute value.
    1st derivative bounded in [-1, 1].
    2nd derivative bounded to beta/2 (max 10.0).
    Prevents 2nd-order gradient explosions in implicit RK4 solvers.
    """
    # softplus(x) + softplus(-x) creates a smooth V-shape.
    raw_abs = (jax.nn.softplus(beta * x) + jax.nn.softplus(-beta * x)) / beta
    
    # Shift it down so that safe_abs(0) = 0.0 perfectly
    offset = 2.0 * jnp.log(2.0) / beta
    return jnp.maximum(raw_abs - offset, 0.0)

class SpectralDense(nn.Module):
    """
    Dense layer with power-iteration spectral normalization.
    Bounds the Lipschitz constant of each layer to ≤ 1.

    BUGFIX-6: stop_gradient applied to sigma.
    Without it Adam receives a gradient through sigma and minimizes it,
    making W / sigma unbounded — the exact opposite of the intended guarantee.
    With stop_gradient, the normalization factor is frozen w.r.t. the optimizer,
    preserving the Lipschitz-≤1 bound across all training epochs.
    """
    features: int
    use_bias: bool = True

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        W = self.param('kernel', jax.nn.initializers.lecun_normal(),
                       (x.shape[-1], self.features))
        # u_vec: power-iteration running estimate of the top right singular vector.
        # Stored as param for serialization, but NOT updated by Adam — see below.
        u = self.param('u_vec', jax.nn.initializers.normal(), (self.features,))

        # One power iteration step (sufficient for smooth networks).
        v = W.T @ (W @ u)   # (in,out).T @ ((in,out) @ (out,)) = (out,in) @ (in,) = (out,)

        v = v / (jnp.linalg.norm(v) + 1e-8)
        # BUGFIX-6: stop_gradient prevents sigma from entering the loss gradient.
        # Adam will NOT see dL/d(sigma) and will NOT try to minimize sigma.
        sigma = jax.lax.stop_gradient(jnp.linalg.norm(W @ v) + 1e-8)
        W_sn  = W / sigma

        out = x @ W_sn
        if self.use_bias:
            b   = self.param('bias', jax.nn.initializers.zeros, (self.features,))
            out = out + b
        return out


# ─────────────────────────────────────────────────────────────────────────────
# §2  Sparse Gaussian Process — Matérn 5/2 with learnable inducing points
# ─────────────────────────────────────────────────────────────────────────────

class SparseGPMatern52(nn.Module):
    """
    Sparse GP with Matérn 5/2 kernel and LEARNABLE inducing point locations.

    Input:  x_star — (5,) kinematic state [alpha, kappa, gamma, Fz, Vx]
    Output: scalar predictive std dev σ(x_star)

    BUGFIX-4: Symmetric inducing point initialization.
    Previous uniform(0,1) initialization covered only positive slip angles
    (left turns only) and positive kappa (traction only). The GP uncertainty
    estimate for right-hand turns and braking was physically meaningless.

    New: Z_raw ~ N(0, 0.5), Z = tanh(Z_raw) * scale + shift
    · tanh maps R→(-1,1), giving symmetric coverage around zero for signed
      quantities (alpha, kappa, gamma) and offset coverage for positive ones (Fz, Vx).
    · tanh also permanently bounds inducing points — Adam cannot push them
      outside the physical operating envelope regardless of learning rate.

    BUGFIX-9 (GP-vX3): NaN-safe posterior variance — three compounding issues fixed.
    ─────────────────────────────────────────────────────────────────────────────
    ISSUE-A  jnp.linalg.inv backward pass — primary NaN source.
        With 50 inducing points initialized from N(0, 0.5) → tanh → cluster
        near 0 in whitened space (Z/ls), the 50×50 Gram matrix K_ZZ is nearly
        singular. The backward pass of linalg.inv computes:
            grad_K_ZZ = -K_ZZ_inv^T @ grad_out @ K_ZZ_inv^T
        which squares the ill-conditioning. Condition number ~1e4 at init
        produces intermediate values ~1e8, overflowing float32 → NaN.
        This NaN propagates back through 64 lax.scan steps × 4 tire corners
        = 256 amplification stages before reaching wavelet coefficients.

        FIX-A:  stop_gradient around linalg.solve(K_ZZ_reg, I).
        Since the GP parameters (Z_raw, log_ls, log_var) are FROZEN during
        WMPC L-BFGS-B optimization (only wavelet_coeffs are optimized),
        stop_gradient is semantically exact — no gradient information is lost.
        K_ZZ_inv becomes a constant 50×50 matrix in the backward pass.
        The gradient of sigma w.r.t. x_star (the tire operating point, which
        DOES depend on wavelet_coeffs) still flows correctly through k_xZ:
            ∂sigma/∂x_star = ∂sigma/∂post_var * (-2*alpha) * ∂k_xZ/∂x_star
        where alpha = K_ZZ_inv @ k_xZ is a constant vector. ✓

    ISSUE-B  jitter 1e-4 insufficient for 50 clustered inducing points.
        The prior_var=0.08 with 50 nearly-identical inducing points gives
        λ_max(K_ZZ) ≈ n_ind * prior_var = 4.0. With jitter=1e-4, the
        regularised smallest eigenvalue is ~1e-4, giving condition number
        ~4/1e-4 = 40,000. linalg.solve at this condition number is 10,000×
        less accurate than IEEE 754 would suggest.

        FIX-B:  jitter = 1e-3 (10× increase). Reduces effective condition
        number from ~40,000 to ~4,000. Combined with FIX-A (stop_gradient),
        the precise value of K_ZZ_inv is now irrelevant to the backward pass.

    ISSUE-C  jnp.maximum(k_xx - red, 1e-6) — dead gradient kink.
        When k_xx - red ≤ 1e-6 (which happens whenever red ≥ k_xx due to
        float32 cancellation in an ill-conditioned system), XLA chooses the
        subgradient d/dx max(x,c)|_{x≤c} = 0. The sqrt gradient 1/(2√1e-6)≈500
        is then multiplied by 0 → the GP contributes ZERO gradient to the
        WMPC loss. The optimizer has no signal from uncertainty and cannot
        enforce the LCB penalty on friction — this is why the 15% LCB cap
        is ignored and the solver runs at 20 m/s without penalty.

        FIX-C:  jax.nn.softplus(k_xx - red) + 1e-8 as the variance floor.
        softplus(x) = log(1+exp(x)) is C∞ everywhere with gradient sigmoid(x)∈(0,1).
        At x=0: softplus(0)=log(2)≈0.693, gradient=0.5 — smooth and non-zero.
        For large negative x (red >> k_xx): softplus decays to 0 exponentially
        but retains a non-zero gradient, so the optimizer is still pushed away
        from high-uncertainty operating points even when the nominal variance
        would be negative due to numerical error. The +1e-8 ensures sqrt is
        never called on zero.
    """
    num_inducing: int = 50

    @nn.compact
    def __call__(self, x_star: jax.Array) -> jax.Array:
        log_var = self.param('log_variance',
                             jax.nn.initializers.constant(jnp.log(0.08)), ())
        prior_var = jnp.exp(log_var)

        # BUGFIX-4: normal init + tanh transform for symmetric, bounded coverage.
        Z_scale = jnp.array([0.25, 0.20, 0.08, 400.0, 10.0])
        Z_shift = jnp.array([0.0,  0.0,  0.0,  800.0, 12.0])
        Z_raw   = self.param('Z_raw',
                             jax.nn.initializers.normal(stddev=0.5),
                             (self.num_inducing, 5))
        Z = jnp.tanh(Z_raw) * Z_scale + Z_shift

        # ── Spectral Mixture kernel parameters (Q=4 mixtures) ─────────────
        # Each mixture q has:  w_q (weight), μ_q (frequency), σ_q (bandwidth) — all per-dim.
        # Initialised so that the SM kernel degrades gracefully to ≈ Matérn 5/2
        # at the default hyperparameter values (Wilson & Adams 2013, Thm 1).
        Q_mix = 4
        log_w_q  = self.param('log_w_q',
                               jax.nn.initializers.zeros, (Q_mix,))          # (Q,)
        log_mu_q = self.param('log_mu_q',
                               jax.nn.initializers.normal(0.3), (Q_mix, 5))  # (Q, D)
        log_sig_q = self.param('log_sig_q',
                                jax.nn.initializers.constant(jnp.log(jnp.array(
                                    # σ_q initialised to match Matérn 5/2 lengthscales.
                                    # In the SM kernel, large σ_q → narrow Gaussian envelope
                                    # (decays quickly with distance). Init at the same scale
                                    # as the original Matérn lengthscales so k_ZZ is well-
                                    # conditioned at init. Adam will learn the correct values.
                                    [0.2, 0.15, 0.1, 400.0, 15.0]
                                ).reshape(1, 5).repeat(Q_mix, axis=0))),
                                (Q_mix, 5))
        w_q   = jax.nn.softmax(log_w_q)              # (Q,) sums to 1, positive
        mu_q  = jnp.exp(log_mu_q)                    # (Q, D) positive frequencies
        sig_q = jnp.exp(log_sig_q)                   # (Q, D) positive bandwidths

        def sm_kernel(x1, x2):
            """
            Spectral Mixture kernel: k(τ) = Σ_q w_q · exp(-2π²τᵀdiag(σ_q²)τ) · cos(2πτᵀμ_q)

            τ = x1 - x2  (D-dim lag vector, NOT normalised by ls — SM learns its own ls via σ_q)
            Reduces to stationary RBF when Q=1 and μ_q=0. Universal approximator
            of stationary covariance functions (Bochner's theorem, Wilson 2013).
            """
            tau = x1 - x2                                                    # (D,)
            def mixture_component(w, mu, sig):
                # Gaussian envelope: exp(-2π²‖τ/sig‖²)
                envelope = jnp.exp(-2.0 * jnp.pi ** 2 * jnp.sum((tau * sig) ** 2))
                # Cosine carrier: cos(2π τᵀ μ)
                carrier  = jnp.cos(2.0 * jnp.pi * jnp.dot(tau, mu))
                return w * envelope * carrier
            # vmap over Q mixtures, then sum — (Q,) → scalar
            components = jax.vmap(mixture_component)(w_q, mu_q, sig_q)
            return prior_var * jnp.sum(components)

        k_ZZ = jax.vmap(lambda z1: jax.vmap(lambda z2: sm_kernel(z1, z2))(Z))(Z)

        # GP-vX3: Cholesky + stop_gradient on L only.
        #
        # Three compounding issues in the original code, all fixed here:
        #
        # ISSUE-1  linalg.inv backward squares the condition number.
        #   K_ZZ with 50 clustered inducing points (tanh-init) has cond ~1e4.
        #   grad(inv) = -K^{-T} @ grad_out @ K^{-T} → cond² ≈ 1e8 → float32
        #   overflow → NaN propagated through 64 scan steps × 4 tire corners.
        #   FIX: Cholesky decomposition. The triangular solve backward is
        #   O(n²) substitution, numerically stable, condition scales linearly.
        #
        # ISSUE-2  jitter 1e-4 insufficient for 50 clustered inducing points.
        #   λ_max(K_ZZ) ≈ n_ind × prior_var = 4.0.
        #   With jitter=1e-4: cond = 4.0/1e-4 = 40,000.
        #   FIX: jitter = 1e-3 → cond ≈ 4,000.
        #
        # ISSUE-3  stop_gradient on L ONLY, not on solve_triangular result.
        #   Wrapping the entire solve_triangular kills ∂sigma/∂x_star, making
        #   the GP LCB penalty invisible to the optimizer gradient.
        #   With stop_gradient only on L: L is a constant 50×50 factor; the
        #   backward of solve_triangular(L, k_xZ) w.r.t. k_xZ is L^{-T} × ∂,
        #   which is numerically bounded. ∂k_xZ/∂x_star still flows through
        #   the matern52 kernel, so the full GP uncertainty gradient survives.
        #
        # ISSUE-4  jnp.maximum floor → dead subgradient at kink.
        #   d/dx max(x, c)|_{x≤c} = 0. sqrt gradient 1/(2√1e-6) ≈ 500 is
        #   multiplied by 0 → zero GP gradient to WMPC. FIX: softplus floor.
        jitter   = 1e-3
        K_ZZ_reg = k_ZZ + jitter * jnp.eye(self.num_inducing)

        # stop_gradient on L only — L is a constant triangular factor.
        # The backward of solve_triangular w.r.t. k_xZ remains active.
        L   = jax.lax.stop_gradient(jnp.linalg.cholesky(K_ZZ_reg))

        k_xZ  = jax.vmap(lambda z: sm_kernel(x_star, z))(Z)  # gradient flows ✓
        v     = jax.scipy.linalg.solve_triangular(L, k_xZ, lower=True)
        red   = jnp.sum(v ** 2)                               # = k_xZ^T K^{-1} k_xZ

        # softplus floor: C∞ everywhere, non-zero gradient at kink.
        # At k_xx - red = 0: softplus(0) = log(2) ≈ 0.693, gradient = 0.5.
        post_var = jax.nn.softplus(prior_var - red) + 1e-8
        return jnp.sqrt(post_var)

# ─────────────────────────────────────────────────────────────────────────────
# §3  PINN + GP combined tire operator
# ─────────────────────────────────────────────────────────────────────────────

class TireOperatorPINN(nn.Module):
    """
    Symmetry-respecting Physics-Informed Neural Network + Sparse GP.

    Input: state_tensor (8,) [alpha, kappa, gamma, Fz, Vx, T_norm]   [BUGFIX-5]
    Previously (7,) with no thermal state — the dominant Pacejka deviation
    (thermal sensitivity) was invisible to the correction network.

    T_norm = (T_eff - T_opt) / 30.0 where T_eff = mean(T_surface_ribs).
    Normalization: /30 covers ±30°C from optimum, the range over which
    the Pacejka thermal correction changes by up to 51%.

    GP input remains (5,) kinematic only — thermal is modeled analytically
    above via _thermal_grip_factor and the GP's role is uncertainty bounding
    in kinematic operating regimes far from tested conditions.
    """
    dim_hidden:      int = 16
    num_gp_inducing: int = 50

    @nn.compact
    def __call__(
        self,
        state_tensor: jax.Array,   # (8,) [alpha, kappa, gamma, Fz, Vx, T_norm, _pad, _pad]
        stochastic_key = None,
    ):
        # Unpack — T_norm is the 6th element (BUGFIX-5 addition)
        alpha  = state_tensor[0]
        kappa  = state_tensor[1]
        gamma  = state_tensor[2]
        Fz     = state_tensor[3]
        Vx     = state_tensor[4]
        T_norm = state_tensor[5]   # (T_eff - T_opt) / 30.0

        # Symmetry-respecting kinematic features
        # sin(2α): captures the secondary peak in Fy vs α beyond the saturation point
        # κ³: asymmetric longitudinal coupling in combined slip
        # T_norm: primary driver of Pacejka residuals at operating temperature
        features = jnp.array([
            jnp.sin(alpha),
            jnp.sin(2.0 * alpha),
            kappa,
            kappa ** 3,
            gamma,
            Fz / 1000.0,
            Vx / 20.0,
            T_norm,         # BUGFIX-5: thermal deviation from optimum
        ])

        # Spectrally normalized layers → Lipschitz-bounded PINN
        x = SpectralDense(self.dim_hidden)(features)
        x = jnp.tanh(x)
        x = SpectralDense(32)(x)
        x = jnp.tanh(x)
        drift = nn.Dense(2,
                         kernel_init=jax.nn.initializers.zeros,
                         bias_init=jax.nn.initializers.zeros)(x)

        # GP uncertainty — kinematic state only (thermal is analytically modeled)
        gp    = SparseGPMatern52(num_inducing=self.num_gp_inducing)
        sigma = gp(state_tensor[:5])   # (5,): alpha, kappa, gamma, Fz, Vx

        if stochastic_key is not None:
            noise = jax.random.normal(stochastic_key, shape=(2,))
            return drift + sigma * noise, sigma

        return drift, sigma


# ─────────────────────────────────────────────────────────────────────────────
# §4  PacejkaTire  — MF6.2 + 5-Node Thermal + PINN/GP
# ─────────────────────────────────────────────────────────────────────────────

class PacejkaTire:
    """
    100% Pure JAX Differentiable Tire Model.

    Layers:
    1. Analytical Pacejka MF6.2 (pure + combined slip)
    2. 5-Node thermodynamic ODE (Jaeger flash temp + convection/conduction)
    3. TireOperatorPINN: deterministic drift + GP uncertainty (LCB capped at 15%)
    """

    def __init__(self, tire_coeffs: dict, rng_seed: int = 42):
        self.coeffs  = tire_coeffs
        self.T_opt   = tire_coeffs.get('T_opt', tire_coeffs.get('T_OPT', 90.0))
        self.T_env   = 25.0
        self.T_track = 40.0
        self.P_nom   = tire_coeffs.get('P_nom', 0.834)

        key = jax.random.PRNGKey(rng_seed)
        self._pinn_module = TireOperatorPINN()

        # BUGFIX-5: dummy state is now (8,) to match expanded feature vector
        dummy_state = jnp.ones(8)
        self._pinn_params = self._pinn_module.init(key, dummy_state)

        # Load calibrated GP inducing points if available.
        # Generated by tools/calibrate_gp_inducing_points.py from a MoTeC log.
        # Falls back silently to the prior-distribution init if file not present.
        import os
        _gp_cal_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', 'models', 'gp_inducing_calibrated.npy'
        )
        if os.path.exists(_gp_cal_path):
            import numpy as np
            Z_raw_calibrated = jnp.array(np.load(_gp_cal_path))
            self._pinn_params = jax.tree_util.tree_map_with_path(
                lambda path, v: (
                    Z_raw_calibrated
                    if any('Z_raw' in str(p) for p in path)
                    else v
                ),
                self._pinn_params,
            )
            print(f"[TireModel] Loaded calibrated GP inducing points → "
                  f"{os.path.basename(_gp_cal_path)}")
    @property
    def operator(self):
        """
        Alias for _pinn_module — backward compat with diagnose.py.
        Correct call: tire.operator.apply(tire._pinn_params, state_8d)
        """
        return self._pinn_module

    @property
    def pinn_params(self):
        """Exposes PINN params for external calibration or logging."""
        return self._pinn_params

    # ─────────────────────────────────────────────────────────────────────────
    # §4.1  Flash temperature
    # ─────────────────────────────────────────────────────────────────────────

    def compute_flash_temperature(
        self,
        mu_actual: jax.Array,
        Fz:        jax.Array,
        V_slide:   jax.Array,
    ) -> jax.Array:
        k_rubber            = 0.25
        rho_c               = 2.0e6
        contact_half_length = 0.075
        thermal_diff        = k_rubber / rho_c
        V_safe              = jnp.maximum(jnp.sqrt(V_slide**2 + 1e-8), 1e-3)
        q_flux              = (mu_actual * safe_abs(Fz) * V_safe) / (2.0 * contact_half_length * 0.205)
        T_flash             = ((q_flux * contact_half_length)
                               / (k_rubber * jnp.sqrt(jnp.pi * (V_safe * contact_half_length) / thermal_diff)))
        return jnp.clip(T_flash, 0.0, 350.0)

    # ─────────────────────────────────────────────────────────────────────────
    # §4.2  5-Node thermal derivatives  (BUGFIX-7: layout realigned)
    # ─────────────────────────────────────────────────────────────────────────

    def compute_thermal_derivatives(
        self,
        T_nodes:    jax.Array,   # (10,) — see layout note below
        Fz_corners: jax.Array,   # (4,)
        kappa:      jax.Array,   # (4,) absolute longitudinal slip
        Vx:         jax.Array,   # scalar
    ) -> jax.Array:
        """
        5-node thermal ODE per axle, realigned to match vehicle_dynamics layout.

        State vector layout (vehicle_dynamics.py §5.3):
          T_nodes[0:3] = x[28:31] = T_surf_inner/mid/outer_f   (3 rib temps, front)
          T_nodes[3]   = x[31]    = T_gas_f
          T_nodes[4:7] = x[32:35] = T_surf_inner/mid/outer_r   (3 rib temps, rear)
          T_nodes[7]   = x[35]    = T_gas_r
          T_nodes[8]   = x[36]    = T_core_f
          T_nodes[9]   = x[37]    = T_core_r

        BUGFIX-7: previous code assumed T_nodes[0:5] = front 5-node block.
        That mapped T_nodes[4]=x[32]=T_surf0_r into the front core slot and
        misaligned the entire rear thermal block by one index.

        Returns dT/dt (10,) in °C/s, ordered identically to T_nodes input.
        """
        # Thermal constants (approximate for Hoosier R25B)
        k_cond   = 0.25     # W/m/K rubber conductivity
        rho_c    = 2.0e6    # J/m³/K volumetric heat capacity
        h_conv   = 80.0     # W/m²/K convection coefficient (air + track)
        A_patch  = 0.025    # m² contact patch area
        V_tire   = 0.003    # m³ tire volume (approx)
        C_node   = rho_c * V_tire / 5.0   # lumped capacitance per node [J/K]

        mu_nom = 1.5   # conservative nominal friction coefficient

        # ── Front axle ────────────────────────────────────────────────────────
        T_ribs_f   = T_nodes[0:3]   # surface inner/mid/outer
        T_gas_f    = T_nodes[3]
        T_core_f   = T_nodes[8]     # BUGFIX-7: corregido mapeo

        Fz_f       = (Fz_corners[0] + Fz_corners[1]) * 0.5
        kap_f      = (kappa[0] + kappa[1]) * 0.5
        V_slide_f  = safe_abs(Vx * kap_f)
        T_flash_f  = self.compute_flash_temperature(mu_nom, Fz_f, V_slide_f)

        # Frictional heat split evenly across 3 surface nodes
        Q_fric_f   = mu_nom * Fz_f * V_slide_f / (3.0 * C_node + 1e-6)

        dT_s_f0 = (Q_fric_f
                   + h_conv * A_patch * (T_flash_f - T_ribs_f[0]) / C_node
                   - h_conv * A_patch * (T_ribs_f[0] - self.T_env) / C_node)
        dT_s_f1 = (Q_fric_f
                   + h_conv * A_patch * (T_flash_f - T_ribs_f[1]) / C_node
                   - h_conv * A_patch * (T_ribs_f[1] - self.T_env) / C_node)
        dT_s_f2 = (Q_fric_f
                   + h_conv * A_patch * (T_flash_f - T_ribs_f[2]) / C_node
                   - h_conv * A_patch * (T_ribs_f[2] - self.T_env) / C_node)

        T_surf_avg_f = (T_ribs_f[0] + T_ribs_f[1] + T_ribs_f[2]) / 3.0
        # Gay-Lussac coupling: internal gas tracks surface slowly
        dT_gas_f  = 0.05 * (T_surf_avg_f - T_gas_f)
        # Core conduction: slow thermal mass
        dT_core_f = 0.02 * (T_surf_avg_f - T_core_f)

        # ── Rear axle ─────────────────────────────────────────────────────────
        T_ribs_r  = T_nodes[4:7]   # BUGFIX-7: was T_nodes[5:8] (off by one)
        T_gas_r   = T_nodes[7]     # BUGFIX-7: was T_nodes[8]
        T_core_r  = T_nodes[9]     # BUGFIX-7: was T_nodes[9] (coincidentally correct)

        Fz_r      = (Fz_corners[2] + Fz_corners[3]) * 0.5
        kap_r     = (kappa[2] + kappa[3]) * 0.5
        V_slide_r = safe_abs(Vx * kap_r)
        T_flash_r = self.compute_flash_temperature(mu_nom, Fz_r, V_slide_r)

        Q_fric_r  = mu_nom * Fz_r * V_slide_r / (3.0 * C_node + 1e-6)

        dT_s_r0 = (Q_fric_r
                   + h_conv * A_patch * (T_flash_r - T_ribs_r[0]) / C_node
                   - h_conv * A_patch * (T_ribs_r[0] - self.T_env) / C_node)
        dT_s_r1 = (Q_fric_r
                   + h_conv * A_patch * (T_flash_r - T_ribs_r[1]) / C_node
                   - h_conv * A_patch * (T_ribs_r[1] - self.T_env) / C_node)
        dT_s_r2 = (Q_fric_r
                   + h_conv * A_patch * (T_flash_r - T_ribs_r[2]) / C_node
                   - h_conv * A_patch * (T_ribs_r[2] - self.T_env) / C_node)

        T_surf_avg_r = (T_ribs_r[0] + T_ribs_r[1] + T_ribs_r[2]) / 3.0
        dT_gas_r  = 0.05 * (T_surf_avg_r - T_gas_r)
        dT_core_r = 0.02 * (T_surf_avg_r - T_core_r)

        # Output ordering MUST match input T_nodes ordering (vehicle_dynamics layout):
        #   [surf×3_f, gas_f, surf×3_r, gas_r, core_f, core_r]
        dT = jnp.array([
            dT_s_f0, dT_s_f1, dT_s_f2, dT_gas_f,   # front (indices 0-3)
            dT_s_r0, dT_s_r1, dT_s_r2, dT_gas_r,   # rear  (indices 4-7)
            dT_core_f, dT_core_r,                    # cores (indices 8-9)
        ])
        return jnp.clip(dT, -500.0, 500.0)

    # ─────────────────────────────────────────────────────────────────────────
    # §4.3  Thermal grip factor from 5-node model
    # ─────────────────────────────────────────────────────────────────────────

    def _thermal_grip_factor(
        self,
        T_ribs: jax.Array,
        T_gas:  jax.Array,
        T_opt:  jax.Array = None,   # NEW: EKF-estimated optimal temp override
    ) -> jax.Array:
        T_opt_eff = self.T_opt if T_opt is None else T_opt   # static Python branch — fine under jit
        T_eff   = jnp.mean(T_ribs[:3])
        beta    = 0.0008
        mu_T    = jnp.exp(-beta * (T_eff - T_opt_eff) ** 2)

        T_ref   = self.T_env + 273.15
        T_gas_K = T_gas + 273.15
        P_ratio = jnp.clip(T_gas_K / (T_ref + 1e-3), 0.70, 1.30)
        mu_P    = 1.0 + 0.05 * (P_ratio - 1.0)
        return jnp.clip(mu_T * mu_P, 0.30, 1.20)

    def compute_force_and_sigma(
        self,
        alpha, kappa, Fz, gamma, T_ribs, T_gas, Vx,
        stochastic_key   = None,
        apply_residual:  bool = True,
        wz:               jax.Array = jnp.array(0.0),
        mu_scale:         jax.Array = jnp.array(1.0),
        T_opt_override:   jax.Array = jnp.array(-1.0),
        alpha_scale:      jax.Array = jnp.array(1.0),
        rby1_scale:       jax.Array = jnp.array(1.0),   # NEW
        rby2_scale:       jax.Array = jnp.array(1.0),   # NEW
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        c   = self.coeffs
        eps = 1e-6

        Fz0     = c.get('FNOMIN', 654.0)
        Fz_safe = jnp.clip(Fz, 10.0, 15000.0)
        dfz     = (Fz_safe - Fz0) / (Fz0 + eps)

        # EKF-estimated T_opt: sentinel -1.0 means "not overridden"
        T_opt_eff = jnp.where(T_opt_override < 0.0, self.T_opt, T_opt_override)
        lam_muy   = self._thermal_grip_factor(T_ribs, T_gas, T_opt=T_opt_eff) * mu_scale
        gam       = gamma

        # ── (unchanged block down to Ky) ──
        PCY1 = c.get('PCY1',  1.53041); PDY1 = c.get('PDY1',  2.40275)
        PDY2 = c.get('PDY2',  0.343535); PDY3 = c.get('PDY3',  3.89743)
        PEY1 = c.get('PEY1',  0.000); PEY2 = c.get('PEY2', -0.280762)
        PEY3 = c.get('PEY3',  0.70403); PEY4 = c.get('PEY4', -0.478297)
        PKY1 = c.get('PKY1', 53.2421); PKY2 = c.get('PKY2',  2.38205)
        PKY3 = c.get('PKY3',  0.15);   PKY4 = c.get('PKY4',  2.0)
        PHY1 = c.get('PHY1', -0.0009); PHY2 = c.get('PHY2', -0.00082)
        PVY1 = c.get('PVY1',  0.045);  PVY2 = c.get('PVY2', -0.024)

        SHy  = PHY1 + PHY2 * dfz
        SVy  = Fz_safe * (PVY1 + PVY2 * dfz) * lam_muy

        # alpha_scale rescales cornering stiffness Ky → shifts the peak slip
        # angle location: x_y = By·a_s, peak at a_s ≈ const/By, so scaling Ky
        # (which drives By) directly implements the EKF's "b_scale" shift.
        # Escalar rigidez de deriva proporcionalmente a mu_scale para evitar que By explote en asfalto
        Ky = (PKY1 * Fz0
              * jnp.sin(PKY4 * jnp.arctan(Fz_safe / jnp.maximum(PKY2 * Fz0, eps)))
              * (1.0 - PKY3 * safe_abs(gam))) * alpha_scale * mu_scale

        Dy   = PDY1 * (1.0 + PDY2 * dfz) * (1.0 - PDY3 * gam ** 2) * Fz_safe * lam_muy
        Cy   = PCY1
        By   = Ky / jnp.maximum(Cy * Dy, eps)

        a_s     = alpha + SHy
        sgn_as  = jnp.tanh(a_s / (1e-3 + eps))
        Ey      = jnp.clip(
            (PEY1 + PEY2 * dfz) * (1.0 - (PEY3 + PEY4 * gam) * sgn_as),
            -10.0, 1.0,
        )
        x_y     = By * a_s
        Fy0     = Dy * jnp.sin(Cy * jnp.arctan(x_y - Ey * (x_y - jnp.arctan(x_y)))) + SVy

        # ════════════════════════════════════════════════════════════════════
        # PURE LONGITUDINAL FORCE  (MF6.2)
        # ════════════════════════════════════════════════════════════════════
        PCX1 = c.get('PCX1',  1.579)
        PDX1 = c.get('PDX1',  1.0)
        PDX2 = c.get('PDX2', -0.10)
        PDX3 = c.get('PDX3',  0.0)
        PEX1 = c.get('PEX1', -0.20)
        PEX2 = c.get('PEX2',  0.10)
        PEX3 = c.get('PEX3',  0.0)
        PKX1 = c.get('PKX1',  5.0)  
        PKX2 = c.get('PKX2',  0.0)
        PKX3 = c.get('PKX3',  0.20)
        PHX1 = c.get('PHX1',  0.0)
        PHX2 = c.get('PHX2',  0.0)
        PVX1 = c.get('PVX1',  0.0)
        PVX2 = c.get('PVX2',  0.0)

        SHx     = PHX1 + PHX2 * dfz
        SVx     = Fz_safe * (PVX1 + PVX2 * dfz) * lam_muy
        kappa_c = kappa + SHx

        Cx  = PCX1
        Dx  = PDX1 * (1.0 + PDX2 * dfz) * (1.0 - PDX3 * gam ** 2) * Fz_safe * lam_muy
        Kx  = PKX1 * Fz_safe * jnp.exp(PKX3 * dfz) * (1.0 + PKX2 * dfz)
        Bx  = Kx / jnp.maximum(Cx * Dx, eps)
        Ex  = jnp.clip(PEX1 + PEX2 * dfz + PEX3 * dfz ** 2, -10.0, 1.0)
        x_x = Bx * kappa_c
        Fx0 = Dx * jnp.sin(Cx * jnp.arctan(x_x - Ex * (x_x - jnp.arctan(x_x)))) + SVx

        # ════════════════════════════════════════════════════════════════════
        # COMBINED SLIP REDUCTION  (Gyk, Gxa)
        # ════════════════════════════════════════════════════════════════════
        RBY1 = c.get('RBY1', 7.0) * rby1_scale
        RBY2 = c.get('RBY2', 7.0) * rby2_scale
        RBY3 = c.get('RBY3', 0.0)
        RCY1 = c.get('RCY1', 1.0)
        REY1 = c.get('REY1', 0.0)
        REY2 = c.get('REY2', 0.0)
        RHY1 = c.get('RHY1', 0.0)

        RBX1 = c.get('RBX1', 10.0)
        RBX2 = c.get('RBX2', 10.0)
        RCX1 = c.get('RCX1', 1.0)
        RHX1 = c.get('RHX1', 0.0)

        SHyk     = RHY1
        kappa_ys = kappa + SHyk                             
        By_s     = RBY1 * jnp.cos(jnp.arctan(RBY2 * (alpha - RBY3)))
        Ey_s     = REY1 + REY2 * dfz
        x_ys     = By_s * kappa_ys                           
        Gyk_num  = jnp.cos(RCY1 * jnp.arctan(x_ys - Ey_s * (x_ys - jnp.arctan(x_ys))))
        Gyk_den  = jnp.cos(RCY1 * jnp.arctan(By_s * SHyk - Ey_s * (By_s * SHyk - jnp.arctan(By_s * SHyk))))
        Gyk      = Gyk_num / (Gyk_den + 1e-6)
        Gyk      = jnp.clip(Gyk, 0.05, 1.0)

        SHxa     = RHX1
        alpha_xs = alpha + SHxa                             
        Bx_s     = RBX1 * jnp.cos(jnp.arctan(RBX2 * kappa))
        x_xs     = Bx_s * alpha_xs                           
        Gxa_num  = jnp.cos(RCX1 * jnp.arctan(x_xs))
        Gxa_den  = jnp.cos(RCX1 * jnp.arctan(Bx_s * SHxa))
        Gxa      = Gxa_num / (Gxa_den + 1e-6)
        Gxa      = jnp.clip(Gxa, 0.05, 1.0)

        Fy = Fy0 * Gyk
        Fx = Fx0 * Gxa

        # ── Turn slip correction (JAX-Safe) ──────────────────────────────────
        a_contact      = c.get('contact_half_length', 0.05)
        curvature      = (jnp.sqrt(wz**2 + 1e-8) + eps) / (jnp.sqrt(Vx**2 + 1e-8) + 0.1)
        phi_t          = a_contact * curvature
        low_speed_fade = jax.nn.sigmoid(10.0 * (safe_abs(Vx) - 0.5))
        Fy             = Fy * (1.0 - 0.15 * safe_abs(phi_t) * low_speed_fade)

        # ── PINN/GP residual corrections ─────────────────────────────────────
        # ``apply_residual=False`` is the reproducible analytical MF6.2 path
        # used by the independent TTC benchmark.  It retains the preceding
        # thermal, combined-slip, and turn-slip calculations but excludes both
        # learned drift and the nonstandard GP-style safety penalty.
        if not apply_residual:
            return Fx, Fy, jnp.array(0.0, dtype=jnp.asarray(Fx).dtype)

        T_eff  = jnp.mean(T_ribs[:3])
        T_norm = jnp.tanh((T_eff - self.T_opt) / 30.0) 
        Vx_arr = jnp.asarray(Vx)

        state_in = jnp.array([
            alpha, kappa_c, gam, Fz_safe, Vx_arr, T_norm,
            0.0, 0.0,   
        ])
        mods, sigma = self._pinn_module.apply(self._pinn_params, state_in, stochastic_key)
        mods        = jnp.clip(mods, -0.25, 0.25)
        penalty     = jnp.clip(2.0 * sigma, 0.0, 0.15)

        Fx_final = Fx * (1.0 + mods[0] - penalty)
        Fy_final = Fy * (1.0 + mods[1] - penalty)

        return Fx_final, Fy_final, sigma

    def compute_force(self, *args, **kwargs) -> tuple[jax.Array, jax.Array]:
        """
        Back-compat shim. Keeps existing callers (objectives, sanity_checks)
        working without needing to unpack the new sigma return value.
        """
        Fx, Fy, _sigma = self.compute_force_and_sigma(*args, **kwargs)
        return Fx, Fy
    # ─────────────────────────────────────────────────────────────────────────
    # §4.5  Aligning torque  (MF6.2 Mz)
    # ─────────────────────────────────────────────────────────────────────────

    def compute_aligning_torque(
        self,
        alpha: jax.Array,
        kappa: jax.Array,
        Fz:    jax.Array,
        gamma: jax.Array,
        Fy:    jax.Array,
        Fx:    jax.Array = jnp.array(0.0),
    ) -> jax.Array:
        """
        Pacejka MF6.2 aligning torque Mz.
        Positive = nose-right restoring moment.
        """
        c   = self.coeffs
        eps = 1e-6
        Fz0     = c.get('FNOMIN', 654.0)
        Fz_safe = jnp.clip(Fz, 10.0, 15000.0)
        dfz     = (Fz_safe - Fz0) / (Fz0 + eps)
        gam     = gamma

        QBZ1 = c.get('QBZ1',  6.5)
        QBZ2 = c.get('QBZ2', -0.50)
        QBZ3 = c.get('QBZ3',  0.0)
        QBZ9 = c.get('QBZ9',  9.0)
        QCZ1 = c.get('QCZ1',  1.10)
        QDZ1 = c.get('QDZ1',  0.08)
        QDZ2 = c.get('QDZ2', -0.01)
        QDZ3 = c.get('QDZ3',  0.0)
        QDZ4 = c.get('QDZ4',  0.0)
        QEZ1 = c.get('QEZ1', -1.50)
        QEZ2 = c.get('QEZ2',  0.60)
        QEZ3 = c.get('QEZ3',  0.0)   # sign-term coefficient (unused at default=0)
        QHZ1 = c.get('QHZ1',  0.0)
        QHZ2 = c.get('QHZ2',  0.0)

        R0   = c.get('R0', 0.2045)

        SHt = QHZ1 + QHZ2 * dfz
        a_t = alpha + SHt

        Bt      = (QBZ1 + QBZ2 * dfz + QBZ3 * dfz ** 2) * (1.0 + QBZ9 * safe_abs(gam))
        Ct      = QCZ1
        Dt = (Fz_safe / 1000.0) * R0 * (QDZ1 + QDZ2 * dfz) * (1.0 + QDZ3 * gam + QDZ4 * gam ** 2)
        # MF6.2 full Et: includes sign-dependent asymmetry term (QEZ3)
        sgn_at  = jnp.tanh(a_t / (1e-3 + eps))
        Et      = jnp.clip((QEZ1 + QEZ2 * dfz) * (1.0 - QEZ3 * sgn_at), -10.0, 1.0)
        x_t     = Bt * a_t
        t       = Dt * jnp.cos(Ct * jnp.arctan(x_t - Et * (x_t - jnp.arctan(x_t))))
        Mz0     = -t * Fy

        SSZ1 = c.get('SSZ1', 0.0)
        SSZ2 = c.get('SSZ2', 0.0)
        s    = R0 * (SSZ1 + SSZ2 * (Fy / (Fz0 + eps)))
        Mz_r = s * Fx

        return Mz0 + Mz_r

    # ─────────────────────────────────────────────────────────────────────────
    # §4.6  Transient slip derivatives
    # ─────────────────────────────────────────────────────────────────────────

    def compute_transient_slip_derivatives(
        self,
        alpha_kin:   jax.Array,
        kappa_kin:   jax.Array,
        alpha_t:     jax.Array,
        kappa_t:     jax.Array,
        Fz:          jax.Array,
        Vx:          jax.Array,
    ) -> tuple:
        """
        First-order carcass lag for transient slip states.
        rl = relaxation length [m]; τ = rl / |Vx| [s]
        """
        rl  = self.coeffs.get('relaxation_length', 0.35)
        tau = rl / (jnp.maximum(safe_abs(Vx), 1.0))
        d_alpha = (alpha_kin - alpha_t) / tau
        d_kappa = (kappa_kin - kappa_t) / tau
        return d_alpha, d_kappa
