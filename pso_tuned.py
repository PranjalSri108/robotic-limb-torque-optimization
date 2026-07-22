"""
pso_tuned.py
============

Tuned Particle Swarm Optimization variant (from scratch; no scipy / pyswarm).

This is a control experiment for the paper: it isolates whether vanilla PSO's
poor performance (pso.py) is a *tuning artifact*. It uses widely-recommended,
well-regarded PSO settings and is otherwise IDENTICAL to pso.py (same update
equations, same interface, same n_pop / n_iter), so any difference is due to
the hyperparameters alone.

Tuned settings (vs pso.py)
--------------------------
  * inertia weight w: linearly decreasing 0.9 -> 0.4        (same as pso.py)
  * c1 = c2 = 1.49445 : the constriction-equivalent acceleration coefficients
        (Clerc & Kennedy 2002; with w~0.729 these reproduce the constriction
        factor chi. Here paired with the standard decreasing-inertia schedule,
        the canonical "well-tuned" recipe.)   [pso.py uses c1 = c2 = 2.0]
  * velocity clamp |v_d| <= 0.2 * (upper_d - lower_d)       [pso.py uses 1.0 *]

Reference
---------
Kennedy & Eberhart (1995); Shi & Eberhart (1998, inertia weight);
Clerc & Kennedy (2002, constriction).

Core update equations (per particle i, dimension d, iteration t)
----------------------------------------------------------------
    v_{i,d} <- w * v_{i,d}
               + c1 * r1 * (pbest_{i,d} - x_{i,d})      # cognitive pull
               + c2 * r2 * (gbest_d     - x_{i,d})      # social pull
    x_{i,d} <- x_{i,d} + v_{i,d}
with r1, r2 ~ U(0,1) per dimension, positions clamped to the box bounds and
velocities clamped to +/- v_max = 0.2 (upper - lower).

Interface (identical across pso / pso_tuned / gwo / acor)
---------------------------------------------------------
    optimize(objective_fn, bounds, n_pop=30, n_iter=100, seed=None)
        -> (best_x, best_cost, history)
"""

from __future__ import annotations

import numpy as np

# --- tuned PSO hyperparameters -----------------------------------------------
W_MAX, W_MIN = 0.9, 0.4     # inertia weight, linearly annealed w_max -> w_min
C1 = 1.49445                # cognitive coefficient (constriction-equivalent)
C2 = 1.49445                # social coefficient (constriction-equivalent)
V_CLAMP_FRAC = 0.2          # velocity clamp as a fraction of the box width


def optimize(objective_fn, bounds, n_pop=30, n_iter=100, seed=None):
    """Minimize objective_fn over box `bounds` with tuned PSO.

    Parameters / Returns: identical contract to pso.optimize (shared interface).
    """
    rng = np.random.default_rng(seed)
    bounds = np.asarray(bounds, dtype=float)
    lo, hi = bounds[:, 0], bounds[:, 1]
    n_dim = bounds.shape[0]
    v_max = V_CLAMP_FRAC * (hi - lo)                  # tighter velocity clamp

    # --- initialize positions (uniform in box) and velocities ---------------
    X = lo + (hi - lo) * rng.random((n_pop, n_dim))
    V = -v_max + 2.0 * v_max * rng.random((n_pop, n_dim))

    # Batch evaluation (numerically identical to per-row; fast for the
    # vectorized forward-dynamics objective).
    cost = np.asarray(objective_fn(X), dtype=float).reshape(-1)

    pbest = X.copy()
    pbest_cost = cost.copy()

    g_idx = np.argmin(pbest_cost)
    gbest = pbest[g_idx].copy()
    gbest_cost = pbest_cost[g_idx]

    history = np.empty(n_iter)

    for t in range(n_iter):
        w = W_MAX - (W_MAX - W_MIN) * t / max(1, n_iter - 1)   # anneal inertia

        r1 = rng.random((n_pop, n_dim))
        r2 = rng.random((n_pop, n_dim))

        # velocity + position update (vectorized over the whole swarm)
        V = w * V + C1 * r1 * (pbest - X) + C2 * r2 * (gbest - X)
        V = np.clip(V, -v_max, v_max)
        X = X + V
        X = np.clip(X, lo, hi)                         # respect bounds

        cost = np.asarray(objective_fn(X), dtype=float).reshape(-1)

        improved = cost < pbest_cost
        pbest[improved] = X[improved]
        pbest_cost[improved] = cost[improved]

        g_idx = np.argmin(pbest_cost)
        if pbest_cost[g_idx] < gbest_cost:
            gbest_cost = pbest_cost[g_idx]
            gbest = pbest[g_idx].copy()

        history[t] = gbest_cost

    return gbest, float(gbest_cost), history


if __name__ == "__main__":
    from objective import PARAM_BOUNDS, objective

    best_x, best_cost, hist = optimize(objective, PARAM_BOUNDS, seed=0)
    print("=== tuned PSO smoke test on exoskeleton objective J ===")
    print(f"initial best cost : {hist[0]:.4f}")
    print(f"final best cost   : {best_cost:.4f}")
    print(f"improvement       : {hist[0] - best_cost:.4f}")
