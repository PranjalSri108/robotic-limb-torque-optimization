"""
gwo.py
======

Grey Wolf Optimizer (GWO) implemented from scratch (no scipy / black-box libs).

Reference
---------
Mirjalili, S., Mirjalili, S. M. & Lewis, A. (2014). "Grey Wolf Optimizer."
Advances in Engineering Software, 69, 46-61.

Core update equations
----------------------
The three best wolves are named alpha, beta, delta (the leaders). A coefficient
`a` decreases linearly from 2 to 0 across the run:

    a = 2 - 2 * t / (n_iter - 1)

For each leader L in {alpha, beta, delta} and each search agent X:

    A = 2 * a * r1 - a          # r1 ~ U(0,1) per dimension
    C = 2 * r2                  # r2 ~ U(0,1) per dimension
    D_L = | C * X_L - X |       # distance to leader L
    X_L' = X_L - A * D_L        # candidate position pulled toward leader L

The new position is the mean of the three leader-guided candidates:

    X <- (X_alpha' + X_beta' + X_delta') / 3

Positions are clamped to the box bounds.

Interface (identical across pso / gwo / acor)
---------------------------------------------
    optimize(objective_fn, bounds, n_pop=30, n_iter=100, seed=None)
        -> (best_x, best_cost, history)
"""

from __future__ import annotations

import numpy as np


def optimize(objective_fn, bounds, n_pop=30, n_iter=100, seed=None):
    """Minimize objective_fn over box `bounds` with the Grey Wolf Optimizer.

    Parameters / Returns: see module docstring (shared interface).
    """
    rng = np.random.default_rng(seed)
    bounds = np.asarray(bounds, dtype=float)
    lo, hi = bounds[:, 0], bounds[:, 1]
    n_dim = bounds.shape[0]

    # --- initialize the wolf pack uniformly in the box ----------------------
    X = lo + (hi - lo) * rng.random((n_pop, n_dim))
    # Batch evaluation: the whole (n_pop, n_dim) pack is scored in one call
    # (identical to per-row, but fast for a vectorized objective).
    cost = np.asarray(objective_fn(X), dtype=float).reshape(-1)

    # --- identify the three leaders (alpha < beta < delta by cost) ----------
    def _rank_leaders(X, cost):
        order = np.argsort(cost)
        a_i, b_i, d_i = order[0], order[1], order[2]
        return (
            X[a_i].copy(), cost[a_i],
            X[b_i].copy(), cost[b_i],
            X[d_i].copy(), cost[d_i],
        )

    Xa, Ca, Xb, Cb, Xd, Cd = _rank_leaders(X, cost)

    history = np.empty(n_iter)

    for t in range(n_iter):
        a = 2.0 - 2.0 * t / max(1, n_iter - 1)         # linearly 2 -> 0

        # encircle/hunt: pull each wolf toward alpha, beta, delta.
        # Fresh random A, C coefficients are drawn per leader (per GWO spec).
        r1 = rng.random((n_pop, n_dim)); r2 = rng.random((n_pop, n_dim))
        A1 = 2 * a * r1 - a; C1 = 2 * r2
        X1 = Xa - A1 * np.abs(C1 * Xa - X)

        r1 = rng.random((n_pop, n_dim)); r2 = rng.random((n_pop, n_dim))
        A2 = 2 * a * r1 - a; C2 = 2 * r2
        X2 = Xb - A2 * np.abs(C2 * Xb - X)

        r1 = rng.random((n_pop, n_dim)); r2 = rng.random((n_pop, n_dim))
        A3 = 2 * a * r1 - a; C3 = 2 * r2
        X3 = Xd - A3 * np.abs(C3 * Xd - X)

        X = (X1 + X2 + X3) / 3.0
        X = np.clip(X, lo, hi)                          # respect bounds

        cost = np.asarray(objective_fn(X), dtype=float).reshape(-1)   # batch eval

        # refresh leaders using both the new pack and the incumbent leaders
        # (so the best-so-far is never lost -- elitism on alpha/beta/delta)
        pool_X = np.vstack([X, Xa, Xb, Xd])
        pool_cost = np.concatenate([cost, [Ca, Cb, Cd]])
        Xa, Ca, Xb, Cb, Xd, Cd = _rank_leaders(pool_X, pool_cost)

        history[t] = Ca

    return Xa, float(Ca), history


if __name__ == "__main__":
    from objective import PARAM_BOUNDS, objective

    best_x, best_cost, hist = optimize(objective, PARAM_BOUNDS, seed=0)
    print("=== GWO smoke test on exoskeleton objective J ===")
    print(f"initial best cost : {hist[0]:.4f}")
    print(f"final best cost   : {best_cost:.4f}")
    print(f"improvement       : {hist[0] - best_cost:.4f}")
