"""Vectorized MF6.1 parameter PyTree I/O for multi-wheel batch evaluation.

API
---
load_json(path)                        -> list[MF61Params]
stack_params(params_list)              -> MF61Params  (fields are arrays of shape (N,))

The stacked MF61Params is a valid JAX pytree, so:

    batch_result = jax.vmap(mf.evaluate)(stacked, alpha4, kappa4, gamma4, fz4, p4)

evaluates all 4 corner forces in parallel with no Python loop overhead.

Sign convention / units
-----------------------
Same as pacejka_61.py (positive-slope): Fy>0 for alpha>0, Fx>0 for kappa>0, Mz<0 for alpha>0; SI.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ter_twin.models.tires import pacejka_61 as mf

__all__ = ["load_json", "stack_params", "unstack_params"]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_json(path: str | Path) -> list[mf.MF61Params]:
    """Load all successfully-fitted tires from an *index.json* produced by
    ``fit_all_ttc_dat.py`` and return a list of :class:`MF61Params`.

    Only entries with ``status == "ok"`` are included; entries with missing
    YAML files are silently skipped with a warning.
    """
    import logging
    log = logging.getLogger(__name__)

    path = Path(path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    base = path.parent
    out: list[mf.MF61Params] = []
    for entry in doc.get("tires", []):
        if entry.get("status") != "ok":
            continue
        yaml_rel = entry.get("params")
        if not yaml_rel:
            log.warning("load_json: entry %r has no params path, skipping", entry.get("slug"))
            continue
        yaml_path = base / yaml_rel
        if not yaml_path.exists():
            log.warning("load_json: %s not found, skipping", yaml_path)
            continue
        try:
            out.append(mf.MF61Params.load(yaml_path))
        except Exception as exc:  # noqa: BLE001
            log.warning("load_json: failed to load %s: %s", yaml_path, exc)
    return out


# ---------------------------------------------------------------------------
# Stacking / unstacking
# ---------------------------------------------------------------------------

def stack_params(params_list: list[mf.MF61Params]) -> mf.MF61Params:
    """Stack a list of scalar :class:`MF61Params` into a single batched
    instance whose fields are 1-D JAX arrays of shape ``(N,)``.

    The result is a valid JAX pytree and can be used directly with
    :func:`jax.vmap`::

        stacked = stack_params([P_fl, P_fr, P_rl, P_rr])
        # alpha, kappa, ... must be shape (N,) arrays
        result = jax.vmap(mf.evaluate)(stacked, alpha, kappa, gamma, fz)

    Parameters
    ----------
    params_list:
        Non-empty list of scalar :class:`MF61Params` (one per wheel corner).

    Returns
    -------
    MF61Params
        Instance where every coefficient field is a 1-D JAX array of shape
        ``(len(params_list),)``.
    """
    if not params_list:
        raise ValueError("stack_params: params_list must not be empty")

    # Collect the leaf values for every named parameter across all instances.
    # MF61Params.tree_flatten returns (leaves, None) where leaves are in NAMES order.
    leaves_per_param = list(zip(*(
        jax.tree_util.tree_leaves(p) for p in params_list
    )))  # list[tuple[float, ...]] of length len(NAMES)

    stacked_leaves = [jnp.asarray(col) for col in leaves_per_param]

    # Reconstruct via tree_unflatten so the pytree structure is preserved.
    _, treedef = jax.tree_util.tree_flatten(params_list[0])
    return jax.tree_util.tree_unflatten(treedef, stacked_leaves)


def unstack_params(stacked: mf.MF61Params) -> list[mf.MF61Params]:
    """Inverse of :func:`stack_params`.

    Splits a batched :class:`MF61Params` (fields of shape ``(N,)``) back into
    a list of *N* scalar :class:`MF61Params`.
    """
    leaves = jax.tree_util.tree_leaves(stacked)
    if not leaves:
        return []
    n = int(np.asarray(leaves[0]).shape[0]) if np.ndim(np.asarray(leaves[0])) >= 1 else 1
    _, treedef = jax.tree_util.tree_flatten(mf.MF61Params())
    result = []
    for i in range(n):
        single_leaves = [
            float(np.asarray(leaf)[i]) if np.ndim(np.asarray(leaf)) >= 1 else float(np.asarray(leaf))
            for leaf in leaves
        ]
        result.append(jax.tree_util.tree_unflatten(treedef, single_leaves))
    return result


# ---------------------------------------------------------------------------
# Convenience: batched evaluation helper
# ---------------------------------------------------------------------------

def batch_evaluate(
    stacked: mf.MF61Params,
    alpha: "np.ndarray | jnp.ndarray",
    kappa: "np.ndarray | jnp.ndarray",
    gamma: "np.ndarray | jnp.ndarray",
    fz: "np.ndarray | jnp.ndarray",
    p: "np.ndarray | jnp.ndarray | None" = None,
) -> mf.MF61Result:
    """Evaluate MF6.1 across all stacked wheel corners in a single :func:`jax.vmap` call.

    Parameters
    ----------
    stacked:
        Batched params produced by :func:`stack_params`, with field shapes ``(N,)``.
    alpha, kappa, gamma, fz:
        Per-corner inputs, each shape ``(N,)``.
    p:
        Inflation pressure [kPa], shape ``(N,)`` or ``None`` (uses each corner's NOMPRES).

    Returns
    -------
    MF61Result
        Named-tuple of arrays each with shape ``(N,)`` — one result per corner.
    """
    if p is None:
        # Use each corner's own nominal pressure scalar — vmap sees it as a leaf.
        _eval = jax.vmap(lambda P, a, k, g, f: mf.evaluate(P, a, k, g, f))
        return _eval(stacked, jnp.asarray(alpha), jnp.asarray(kappa),
                     jnp.asarray(gamma), jnp.asarray(fz))
    _eval_p = jax.vmap(lambda P, a, k, g, f, pp: mf.evaluate(P, a, k, g, f, pp))
    return _eval_p(stacked, jnp.asarray(alpha), jnp.asarray(kappa),
                   jnp.asarray(gamma), jnp.asarray(fz), jnp.asarray(p))
