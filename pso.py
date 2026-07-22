"""
pso.py
======

Particle Swarm Optimization (PSO) implemented from scratch (no scipy / pyswarm).

Reference
---------
Kennedy, J. & Eberhart, R. (1995). "Particle Swarm Optimization." Proc. IEEE
Int. Conf. Neural Networks. Inertia-weight variant: Shi & Eberhart (1998).

Core update equations (per particle i, dimension d, iteration t)
----------------------------------------------------------------
    v_{i,d} <- w * v_{i,d}
               + c1 * r1 * (pbest_{i,d} - x_{i,d})      # cognitive pull
               + c2 * r2 * (gbest_d     - x_{i,d})      # social pull
    x_{i,d} <- x_{i,d} + v_{i,d}

where r1, r2 ~ U(0,1) drawn independently per dimension, w is the inertia
weight (linearly decreased from w_max to w_min over the run), and c1, c2 are
the cognitive/social acceleration coefficients. Positions are clamped to the
box bounds; velocities are clamped to +/- v_max = (upper - lower).

Interface (identical across pso / gwo / acor)
---------------------------------------------
    optimize(objective_fn, bounds, n_pop=30, n_iter=100, seed=None)
        -> (best_x, best_cost, history)
"""

from __future__ import annotations

import numpy as np

# --- PSO hyperparameters -----------------------------------------------------
W_MAX, W_MIN = 0.9, 0.4     # inertia weight, linearly annealed w_max -> w_min
C1 = 2.0                    # cognitive coefficient (pull to personal best)
C2 = 2.0                    # social coefficient (pull to global best)


def optimize(objective_fn, bounds, n_pop=30, n_iter=100, seed=None):
    """Minimize objective_fn over box `bounds` with PSO.

    Parameters
    ----------
    objective_fn : callable, maps a (n_dim,) vector -> scalar cost.
    bounds       : (n_dim, 2) array-like of [lower, upper] per dimension.
    n_pop        : swarm size (number of particles).
    n_iter       : number of iterations.
    seed         : RNG seed for reproducibility.

    Returns
    -------
    best_x    : (n_dim,) best decision vector found.
    best_cost : float, its objective value.
    history   : (n_iter,) best cost per iteration (convergence curve).
    """
    rng = np.random.default_rng(seed)
    bounds = np.asarray(bounds, dtype=float)
    lo, hi = bounds[:, 0], bounds[:, 1]
    n_dim = bounds.shape[0]
    v_max = hi - lo                                   # velocity clamp magnitude

    # --- initialize positions (uniform in box) and velocities ---------------
    X = lo + (hi - lo) * rng.random((n_pop, n_dim))
    V = -v_max + 2.0 * v_max * rng.random((n_pop, n_dim))

    # Batch evaluation: objective_fn is called on the whole (n_pop, n_dim)
    # population at once and returns (n_pop,) costs. This is numerically
    # identical to evaluating row-by-row but far faster for a vectorized
    # objective (the forward-dynamics simulation batches over the population).
    cost = np.asarray(objective_fn(X), dtype=float).reshape(-1)   # current costs

    pbest = X.copy()                                  # personal-best positions
    pbest_cost = cost.copy()

    g_idx = np.argmin(pbest_cost)                     # global-best
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

        cost = np.asarray(objective_fn(X), dtype=float).reshape(-1)   # batch eval

        # update personal bests
        improved = cost < pbest_cost
        pbest[improved] = X[improved]
        pbest_cost[improved] = cost[improved]

        # update global best
        g_idx = np.argmin(pbest_cost)
        if pbest_cost[g_idx] < gbest_cost:
            gbest_cost = pbest_cost[g_idx]
            gbest = pbest[g_idx].copy()

        history[t] = gbest_cost

    return gbest, float(gbest_cost), history


if __name__ == "__main__":
    from objective import PARAM_BOUNDS, objective

    best_x, best_cost, hist = optimize(objective, PARAM_BOUNDS, seed=0)
    print("=== PSO smoke test on exoskeleton objective J ===")
    print(f"initial best cost : {hist[0]:.4f}")
    print(f"final best cost   : {best_cost:.4f}")
    print(f"improvement       : {hist[0] - best_cost:.4f}")
