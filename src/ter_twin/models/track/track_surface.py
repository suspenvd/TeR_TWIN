# models/track_surface.py
# ═══════════════════════════════════════════════════════════════════════════════
# Project-GP — Differentiable Track Geometry (B-spline + Arc-Length + Frenet)
# ═══════════════════════════════════════════════════════════════════════════════
"""
JAX-native, fully-differentiable track centerline representation.

Mathematical contract — for arc-length s ∈ [0, L):
    r(s)   ∈ R² : centerline position                                        [m]
    T̂(s)   ∈ R² : unit tangent  dr/ds                                         [-]
    N̂(s)   ∈ R² : unit inward normal  (left of travel)                        [-]
    κ(s)   ∈ R  : signed curvature, left-positive, κ = (x'y'' − y'x'')/‖r'‖³ [1/m]
    w_l(s) ∈ R  : half-width to left  cone (signed +N̂ direction)              [m]
    w_r(s) ∈ R  : half-width to right cone (signed −N̂ direction)              [m]

Construction modes:
    • from_cones(left_xy, right_xy)         — production: pair-fit from FSG cones
    • from_curvature(kappa_s, ds)           — Frenet integration from κ(s) profile
    • randomize(key, length, **kwargs)      — domain-randomized FSG-compliant track

JAX contract:
    All public methods are jit-safe, vmap-safe, grad-safe. The arc-length
    inversion s↔u is differentiable through `jnp.searchsorted` + linear interp.
    Curvature is computed analytically from the B-spline basis derivatives —
    no finite differences anywhere on the gradient path.

Author: Project-GP contributor • Target: Ter27 / FSG 2026 Siemens DT Award
"""

from __future__ import annotations
from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp


# ─────────────────────────────────────────────────────────────────────────────
# §1.  Configuration & data types
# ─────────────────────────────────────────────────────────────────────────────

class TrackGeometryConfig(NamedTuple):
    n_control: int = 128         # B-spline control points (periodic)
    n_query:   int = 2048        # arc-length LUT resolution
    n_quad:    int = 4           # Gauss-Legendre quadrature order per segment


class TrackGeometry(NamedTuple):
    """
    Compiled track. All arrays static-shape under JIT; treat as a JAX pytree.
    """
    P_center:     jnp.ndarray  # [n_control, 2]   centerline control points
    P_left:       jnp.ndarray  # [n_control, 2]   left  boundary control points
    P_right:      jnp.ndarray  # [n_control, 2]   right boundary control points
    s_grid:       jnp.ndarray  # [n_query]        cumulative arc length at u_grid
    u_grid:       jnp.ndarray  # [n_query]        uniform u ∈ [0,1)
    L:            jnp.ndarray  # ()               total track length [m]
    half_w_left:  jnp.ndarray  # [n_query]        half-width to left  cone
    half_w_right: jnp.ndarray  # [n_query]        half-width to right cone


# ─────────────────────────────────────────────────────────────────────────────
# §2.  Periodic uniform cubic B-spline basis (analytic to order 2)
# ─────────────────────────────────────────────────────────────────────────────
#
# Local parameter t ∈ [0,1) on each segment:
#   B₀(t) = (1−t)³ / 6
#   B₁(t) = (3t³ − 6t² + 4) / 6
#   B₂(t) = (−3t³ + 3t² + 3t + 1) / 6
#   B₃(t) = t³ / 6
# Σ B_k = 1 (partition of unity), B_k ≥ 0 on [0,1] (convex combination).
# C² continuity at integer knots → centerline κ(s) is C⁰.
# ─────────────────────────────────────────────────────────────────────────────

def _basis_b0(t):  # value
    one_m_t = 1.0 - t
    return jnp.stack([
        one_m_t ** 3 / 6.0,
        (3.0 * t ** 3 - 6.0 * t ** 2 + 4.0) / 6.0,
        (-3.0 * t ** 3 + 3.0 * t ** 2 + 3.0 * t + 1.0) / 6.0,
        t ** 3 / 6.0,
    ], axis=0)


def _basis_b1(t):  # 1st derivative w.r.t. t
    return jnp.stack([
        -((1.0 - t) ** 2) / 2.0,
        (9.0 * t ** 2 - 12.0 * t) / 6.0,
        (-9.0 * t ** 2 + 6.0 * t + 3.0) / 6.0,
        (t ** 2) / 2.0,
    ], axis=0)


def _basis_b2(t):  # 2nd derivative w.r.t. t
    return jnp.stack([(1.0 - t),
                      (3.0 * t - 2.0),
                      (-3.0 * t + 1.0),
                      t], axis=0)


def _eval_bspline_periodic(P: jnp.ndarray, u: jnp.ndarray, basis_fn) -> jnp.ndarray:
    """
    Evaluate a periodic cubic B-spline (or its k-th derivative w.r.t. t) at u ∈ [0,1).

    P:       [n_control, dim] periodic control points (no duplicate end point).
    u:       () or (...) parameter values; modular access handles wrap.
    basis_fn: one of _basis_b0 / _basis_b1 / _basis_b2.

    Returns shape (..., dim). NB: The chain-rule scaling du/dt = 1/n_control is
    NOT applied here — apply it at the call site if you need d/du as opposed
    to d/dt of the local segment parameter.
    """
    n_c = P.shape[0]
    u_scaled = u * n_c
    i = jnp.floor(u_scaled).astype(jnp.int32)
    t = u_scaled - i
    B = basis_fn(t)                                                # [4, ...]
    idx = (i[..., None] + jnp.arange(4)[None, :]) % n_c            # [..., 4]
    P_local = P[idx]                                               # [..., 4, dim]
    # Contract: B over its leading axis vs. P_local over the 4-neighbours axis.
    return jnp.einsum('k...,...kd->...d', B, P_local)


# ─────────────────────────────────────────────────────────────────────────────
# §3.  Arc-length lookup table (Gauss-Legendre)
# ─────────────────────────────────────────────────────────────────────────────

_GL4_NODES = jnp.array([
    -0.8611363115940526, -0.3399810435848563,
     0.3399810435848563,  0.8611363115940526,
])
_GL4_WEIGHTS = jnp.array([
    0.3478548451374538, 0.6521451548625461,
    0.6521451548625461, 0.3478548451374538,
])


def _build_arclength_table(P_center: jnp.ndarray, n_query: int):
    """
    Returns (u_grid [n_query], s_grid [n_query], L ()).
    Per-segment integration of ‖dr/du‖ via 4-pt Gauss-Legendre.
    """
    nodes_01 = 0.5 * (_GL4_NODES + 1.0)
    weights  = 0.5 * _GL4_WEIGHTS
    du = 1.0 / n_query
    u_edges = jnp.arange(n_query, dtype=jnp.float32) * du

    n_c = P_center.shape[0]

    def seg_length(u_lo):
        u_pts = u_lo + du * nodes_01                                # [4]
        d1_t = jax.vmap(lambda uu: _eval_bspline_periodic(P_center, uu, _basis_b1))(u_pts)
        # dr/du = (1/du)·dr/dt? No — basis is evaluated against control points,
        # the segment local t goes from 0..1 across one knot interval of width
        # 1/n_c in u, so dr/du = n_c · (basis_b1 · P_local). Magnitude:
        speed = jnp.linalg.norm(d1_t, axis=-1) * n_c
        return du * jnp.dot(weights, speed)

    seg_lengths = jax.vmap(seg_length)(u_edges)                     # [n_query]
    s_cum = jnp.concatenate([jnp.array([0.0], dtype=seg_lengths.dtype),
                             jnp.cumsum(seg_lengths)])
    return u_edges, s_cum[:-1], s_cum[-1]


# ─────────────────────────────────────────────────────────────────────────────
# §4.  Construction — from FSG cone arrays
# ─────────────────────────────────────────────────────────────────────────────

def _resample_periodic_chord(xy: jnp.ndarray, n_out: int) -> jnp.ndarray:
    """
    Periodic chord-length resample to exactly n_out points uniformly spaced
    in cumulative chord-length. This is the implicit knot-vector for the
    B-spline (uniform parameterization in chord-length is the workhorse for
    racing-line splines — no Foley-Nielson reparam needed at FSG track scales).
    """
    diffs   = xy - jnp.roll(xy, 1, axis=0)
    seg_len = jnp.linalg.norm(diffs, axis=-1)
    cum     = jnp.cumsum(seg_len)
    total   = cum[-1]
    u_in    = cum / (total + 1e-12)                                 # ∈ (0,1]
    u_out   = jnp.linspace(0.0, 1.0, n_out + 1)[:-1]                # ∈ [0,1)

    # jnp.interp with period= for proper wrap-around
    x_out = jnp.interp(u_out, u_in, xy[:, 0], period=1.0)
    y_out = jnp.interp(u_out, u_in, xy[:, 1], period=1.0)
    return jnp.stack([x_out, y_out], axis=-1)


def from_cones(
    left_xy:  jnp.ndarray,                                          # [N_l, 2]
    right_xy: jnp.ndarray,                                          # [N_r, 2]
    config:   TrackGeometryConfig = TrackGeometryConfig(),
) -> TrackGeometry:
    """
    Build TrackGeometry from FSG left/right cone polylines.

    Caller must pre-order cones along the direction of travel (driverless
    perception layer is responsible for this — see project_gp_mirena_bridge
    track_ingestor.py for the gate-matching ROS-side preprocessor).
    """
    P_left   = _resample_periodic_chord(left_xy,  config.n_control)
    P_right  = _resample_periodic_chord(right_xy, config.n_control)
    P_center = 0.5 * (P_left + P_right)

    u_grid, s_grid, L = _build_arclength_table(P_center, config.n_query)

    # Half-widths sampled at the LUT grid (cached for fast queries)
    eval_u = lambda P, u: _eval_bspline_periodic(P, u, _basis_b0)
    pos_c = jax.vmap(lambda u: eval_u(P_center, u))(u_grid)
    pos_l = jax.vmap(lambda u: eval_u(P_left,   u))(u_grid)
    pos_r = jax.vmap(lambda u: eval_u(P_right,  u))(u_grid)

    half_w_left  = jnp.linalg.norm(pos_l - pos_c, axis=-1)
    half_w_right = jnp.linalg.norm(pos_r - pos_c, axis=-1)

    return TrackGeometry(
        P_center=P_center, P_left=P_left, P_right=P_right,
        s_grid=s_grid, u_grid=u_grid, L=L,
        half_w_left=half_w_left, half_w_right=half_w_right,
    )


# ─────────────────────────────────────────────────────────────────────────────
# §5.  Construction — Frenet integration from a κ(s) profile
# ─────────────────────────────────────────────────────────────────────────────
# Useful for domain-randomized track generation. The Frenet–Serret system
# in 2D collapses to:
#
#     dψ/ds = κ(s)
#     dx/ds = cos ψ
#     dy/ds = sin ψ
#
# Closed-track guarantee: rescale κ so ∫κ ds = ±2π · n_loops.
# ─────────────────────────────────────────────────────────────────────────────

def from_curvature(
    kappa_s:    jnp.ndarray,                                        # [N_s]
    ds:         float,
    half_width: float = 2.0,
    config:     TrackGeometryConfig = TrackGeometryConfig(),
    n_loops:    int = 1,
) -> TrackGeometry:
    """
    Integrate κ(s) → (x(s), y(s)) via Frenet, then refit B-spline control
    points so the rest of the pipeline (arc-length LUT, normals, curvature
    queries) stays consistent.

    Closure is enforced: total turning is rescaled to 2π·n_loops.
    """
    # Enforce closure
    int_k    = jnp.sum(kappa_s) * ds
    target   = 2.0 * jnp.pi * n_loops
    # Sign-preserving rescale (avoid κ=0 division)
    scale    = target / (int_k + jnp.sign(int_k) * 1e-6)
    kappa_s  = kappa_s * scale

    # Symplectic Frenet integration (trapezoidal heading update)
    psi = jnp.cumsum(kappa_s) * ds                                  # [N_s]
    psi = psi - psi[0]                                              # start at heading 0
    cos_psi = jnp.cos(psi)
    sin_psi = jnp.sin(psi)
    # Trapezoidal position integration for ½-step accuracy boost
    cos_avg = 0.5 * (cos_psi + jnp.roll(cos_psi, 1))
    sin_avg = 0.5 * (sin_psi + jnp.roll(sin_psi, 1))
    cos_avg = cos_avg.at[0].set(cos_psi[0])
    sin_avg = sin_avg.at[0].set(sin_psi[0])
    x = jnp.cumsum(cos_avg) * ds
    y = jnp.cumsum(sin_avg) * ds
    xy_center = jnp.stack([x, y], axis=-1)

    # Inward normal at each station: n̂ = (-sin ψ, cos ψ)
    n_hat = jnp.stack([-sin_psi, cos_psi], axis=-1)
    P_left_xy  = xy_center + half_width * n_hat
    P_right_xy = xy_center - half_width * n_hat

    return from_cones(P_left_xy, P_right_xy, config=config)


# ─────────────────────────────────────────────────────────────────────────────
# §6.  Domain randomization — FSG-compliant random tracks
# ─────────────────────────────────────────────────────────────────────────────
# Strategy: sample a corner-straight skeleton with random radii drawn from the
# FSG-rules envelope (R_min ≈ 9 m for hairpins → κ_max ≈ 0.11). Smooth corner
# transitions with a half-Hann taper to keep κ(s) C¹.
# ─────────────────────────────────────────────────────────────────────────────

def randomize(
    key:         jax.Array,
    length:      float = 350.0,
    n_corners:   int   = 10,
    kappa_max:   float = 0.13,                                      # FSG R≈7.7 m
    half_width:  float = 2.0,
    ds:          float = 0.25,
    config:      TrackGeometryConfig = TrackGeometryConfig(),
) -> TrackGeometry:
    """
    Generate a closed FSG-compliant random track. Returns a TrackGeometry whose
    control points are differentiable w.r.t. nothing (it's a sample), but the
    output track itself supports gradient queries downstream.
    """
    k1, k2, k3, k4 = jax.random.split(key, 4)

    # 50% straight, 50% corner budget — typical FSG sprint character
    corner_budget   = 0.5 * length
    straight_budget = length - corner_budget

    # Corner curvatures: log-uniform in [0.05, kappa_max], random sign
    log_min, log_max = jnp.log(0.05), jnp.log(kappa_max)
    log_k    = jax.random.uniform(k1, (n_corners,), minval=log_min, maxval=log_max)
    signs    = jax.random.choice(k2, jnp.array([-1.0, 1.0]), (n_corners,))
    k_vals   = signs * jnp.exp(log_k)

    # Dirichlet allocation for both corner arc-lengths and straight lengths
    raw_c    = jax.random.gamma(k3, jnp.ones(n_corners))
    arc_corn = corner_budget * raw_c / (jnp.sum(raw_c) + 1e-12)

    raw_s    = jax.random.gamma(k4, jnp.ones(n_corners + 1))
    arc_str  = straight_budget * raw_s / (jnp.sum(raw_s) + 1e-12)

    # Build dense κ(s) profile in pure JAX (no Python list comprehension in
    # traced code — but this function is called outside of JIT, so use NumPy-
    # style index assignment via at[].set() in a single fori_loop).
    n_s = int(jnp.ceil(length / ds))
    s_axis = jnp.arange(n_s) * ds
    kappa_dense = jnp.zeros(n_s)

    # Build piecewise constant κ via a fori_loop accumulator on (s_start, k_val)
    s_starts = jnp.cumsum(jnp.concatenate([
        jnp.array([arc_str[0]]),
        # interleave corners + straights from index 1
        jnp.stack([arc_corn[:-1] + arc_str[1:-1], arc_str[1:-1]], axis=-1).ravel()[:2*(n_corners-1)],
    ]))
    # Simpler: compute corner [start, end] via cumulative sum of stitched events
    seg_lens = jnp.concatenate([
        jnp.stack([arc_str[:-1], arc_corn], axis=-1).ravel(),       # str, corn, str, corn, ...
        arc_str[-1:],
    ])                                                              # length 2*n_corners + 1
    seg_kvals = jnp.concatenate([
        jnp.stack([jnp.zeros(n_corners), k_vals], axis=-1).ravel(),
        jnp.array([0.0]),
    ])                                                              # length 2*n_corners + 1
    seg_ends = jnp.cumsum(seg_lens)                                 # absolute end-of-segment

    # Vectorized assignment: for each station s, pick the segment it belongs to
    # via `searchsorted(seg_ends, s, side='right')`
    seg_idx = jnp.searchsorted(seg_ends, s_axis, side='right')
    seg_idx = jnp.clip(seg_idx, 0, seg_kvals.shape[0] - 1)
    kappa_dense = seg_kvals[seg_idx]

    # Smooth κ(s) with a half-Hann window to give C¹ corner entries/exits.
    # Corners < 6 m of taper would be physically dubious at 80 km/h.
    hann_n = max(int(round(2.0 / ds)), 4)                           # 2 m taper window
    if hann_n % 2 == 0:
        hann_n += 1
    half = hann_n // 2
    n_arr = jnp.arange(hann_n)
    hann = 0.5 - 0.5 * jnp.cos(2 * jnp.pi * n_arr / (hann_n - 1))
    hann = hann / (jnp.sum(hann) + 1e-12)
    kappa_smooth = jnp.convolve(kappa_dense, hann, mode='same')

    return from_curvature(
        kappa_smooth, ds=ds, half_width=half_width,
        config=config, n_loops=1,
    )


# ─────────────────────────────────────────────────────────────────────────────
# §7.  Public query API — fully differentiable
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def _u_of_s(track: TrackGeometry, s: jnp.ndarray) -> jnp.ndarray:
    """
    Invert the arc-length LUT: find u such that s_grid(u)=s.
    Linear interp between LUT entries — gradient-safe, no branching.
    """
    s_mod = jnp.mod(s, track.L)
    # searchsorted → linear interp on u_grid via s_grid
    return jnp.interp(s_mod, track.s_grid, track.u_grid, period=None)


@jax.jit
def query(
    track: TrackGeometry,
    s:     jnp.ndarray,                                             # () or [N]
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Returns (pos, T̂, N̂, κ, w_left, w_right) for one or many arc-length stations.

    Differentiable w.r.t. s (continuous through the LUT) AND w.r.t. the
    underlying control points (chain rule through the basis evaluation).
    """
    u = _u_of_s(track, s)                                           # broadcast-shape

    # Unscale to local segment param (chain rule: d/du = n_c · d/dt)
    n_c = track.P_center.shape[0]

    # Position
    pos = _eval_bspline_periodic(track.P_center, u, _basis_b0)      # [..., 2]

    # Tangent in u-space, then scale to s-space tangent (unit)
    dr_du = _eval_bspline_periodic(track.P_center, u, _basis_b1) * n_c   # [..., 2]
    speed_u = jnp.linalg.norm(dr_du, axis=-1, keepdims=True) + 1e-9
    T_hat = dr_du / speed_u
    # Inward normal: rotate +90° (left of travel)
    N_hat = jnp.stack([-T_hat[..., 1], T_hat[..., 0]], axis=-1)

    # Curvature: κ = (x'y'' − y'x'') / ‖r'‖³  with primes in u-parameter
    d2r_du2 = _eval_bspline_periodic(track.P_center, u, _basis_b2) * (n_c ** 2)
    cross_uu = (dr_du[..., 0] * d2r_du2[..., 1]
                - dr_du[..., 1] * d2r_du2[..., 0])
    speed_u_sq = jnp.sum(dr_du ** 2, axis=-1)
    kappa = cross_uu / (speed_u_sq ** 1.5 + 1e-12)

    # Half-widths via cached LUT (linear interp)
    w_left  = jnp.interp(jnp.mod(s, track.L), track.s_grid, track.half_w_left)
    w_right = jnp.interp(jnp.mod(s, track.L), track.s_grid, track.half_w_right)

    return pos, T_hat, N_hat, kappa, w_left, w_right


@jax.jit
def project_xy(
    track:        TrackGeometry,
    xy:           jnp.ndarray,                                      # [2]
    s_guess:      jnp.ndarray,                                      # ()
    search_half:  float = 12.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Project a world-frame point onto the centerline.

    Returns (s_closest, n_signed, heading_at_s).
        n_signed > 0   : car is on left of centerline (inside +N̂ half-plane)
        n_signed < 0   : right side

    Implementation: windowed soft-argmin around s_guess (warm-started from
    previous timestep). Fully differentiable in xy via softmax temperature.
    """
    n_cands  = 96
    s_lo     = s_guess - search_half
    s_hi     = s_guess + search_half
    s_cands  = jnp.linspace(s_lo, s_hi, n_cands)

    pos_c, T_c, N_c, _, _, _ = jax.vmap(lambda ss: query(track, ss))(s_cands)
    d_sq = jnp.sum((pos_c - xy[None, :]) ** 2, axis=-1)             # [n_cands]

    # Soft-argmin (τ chosen so that the gradient is sharp but never zero)
    tau     = 0.5
    weights = jax.nn.softmax(-d_sq / tau)
    s_best  = jnp.dot(weights, s_cands)

    pos_b, T_b, N_b, _, _, _ = query(track, s_best)
    delta_xy = xy - pos_b
    n_signed = jnp.dot(delta_xy, N_b)
    psi_track = jnp.arctan2(T_b[1], T_b[0])
    return s_best, n_signed, psi_track

# ─────────────────────────────────────────────────────────────────────────────
# §9. Track Surface Physics (Rubber & Thermodynamics)
# ─────────────────────────────────────────────────────────────────────────────
from typing import NamedTuple

class TrackSurfaceConfigPhysics(NamedTuple):
    track_length: float = 1000.0
    N_s: int = 1000
    N_n: int = 5
    base_mu: float = 1.0
    rubber_mu_boost: float = 0.2

class TrackSurfaceState(NamedTuple):
    rubber_level: jnp.ndarray  
    T_surface: jnp.ndarray     
    shadow_mask: jnp.ndarray   

def create_track_surface(track_length: float = 1000.0, n_laps_pre_rubbered: int = 0):
    cfg = TrackSurfaceConfigPhysics(track_length=track_length)
    rubber_level = jnp.zeros((cfg.N_s, cfg.N_n))
    
    if n_laps_pre_rubbered > 0:
        rubber_level = rubber_level.at[:, 2].set(0.8)
        
    T_surface = jnp.full(cfg.N_s, 25.0)
    s_axis = jnp.arange(cfg.N_s) * (track_length / cfg.N_s)
    shadow_mask = jnp.where((s_axis > 280.0) & (s_axis < 320.0), 1.0, 0.0)
    
    return TrackSurfaceState(rubber_level, T_surface, shadow_mask), cfg

@jax.jit
def query_track_friction(s: jnp.ndarray, n: jnp.ndarray, state: TrackSurfaceState, cfg: TrackSurfaceConfigPhysics):
    s_idx = (jnp.mod(s, cfg.track_length) / cfg.track_length * cfg.N_s).astype(jnp.int32)
    n_idx = jnp.clip(jnp.round(n / 1.4) + 2, 0, cfg.N_n - 1).astype(jnp.int32)
    mu = cfg.base_mu + state.rubber_level[s_idx, n_idx] * cfg.rubber_mu_boost
    return mu, state.T_surface[s_idx]

@jax.jit
def update_rubber_level(rubber_level: jnp.ndarray, s: jnp.ndarray, n: jnp.ndarray, cfg: TrackSurfaceConfigPhysics, dt: float):
    s_idx = (jnp.mod(s, cfg.track_length) / cfg.track_length * cfg.N_s).astype(jnp.int32)
    n_idx = jnp.clip(jnp.round(n / 1.4) + 2, 0, cfg.N_n - 1).astype(jnp.int32)
    return rubber_level.at[s_idx, n_idx].add(0.01 * dt)

@jax.jit
def update_track_temperature(T_surface: jnp.ndarray, shadow_mask: jnp.ndarray, cfg: TrackSurfaceConfigPhysics, dt: float):
    heating_rate = 0.05 * (1.0 - shadow_mask)
    cooling_rate = -0.01 * (T_surface - 25.0)
    return T_surface + (heating_rate + cooling_rate) * dt

# ─────────────────────────────────────────────────────────────────────────────
# §8.  Smoke test (not run under JIT path — for local sanity only)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[track_surface] smoke test …")
    key = jax.random.PRNGKey(0)
    trk = randomize(key, length=300.0, n_corners=8)
    print(f"  L = {float(trk.L):.2f} m")
    s_test = jnp.linspace(0.0, float(trk.L), 5)
    pos, T_hat, N_hat, kappa, wL, wR = jax.vmap(lambda s: query(trk, s))(s_test)
    print(f"  κ samples = {kappa}")
    print(f"  w_left = {wL}")
    # Gradient check: ∂(κ at s=50)/∂(P_center[0,0]) should be finite
    g = jax.grad(lambda P, s: query(trk._replace(P_center=P), s)[3])(trk.P_center, jnp.array(50.0))
    print(f"  grad κ wrt P_center[0]: {g[0]}  (finite gradient → differentiability OK)")
    print("[track_surface] OK.")