"""
acor.py
=======

Ant Colony Optimization for Continuous Domains (ACO_R) from scratch.

Reference
---------
Socha, K. & Dorigo, M. (2008). "Ant colony optimization for continuous
domains." European Journal of Operational Research, 185(3), 1155-1173.

Idea
----
ACO_R replaces the discrete pheromone table of classic ACO with a *solution
archive* of the k best solutions found so far. Each new solution is sampled
from a Gaussian-kernel probability density built from that archive.

Core equations
--------------
Archive holds k solutions sorted best -> worst (rank l = 1..k). Each rank gets
a Gaussian weight (smaller q => stronger bias toward the best-ranked solution):

    w_l = 1 / (q * k * sqrt(2*pi)) * exp( -(l-1)^2 / (2 * q^2 * k^2) )

Selection probability of archive solution l:   p_l = w_l / sum_j w_j.

To build one new ant, for each dimension i:
    1. pick a guiding solution l with probability p_l (roulette),
    2. mean   mu   = archive[l, i],
    3. spread sigma = xi * sum_{e != l} |archive[e, i] - archive[l, i]| / (k - 1),
    4. sample  x_i ~ Normal(mu, sigma).

n_pop new ants are sampled per iteration, merged with the archive, and the best
k are kept (archive size = k). Samples are clamped to the box bounds.

Interface (identical across pso / gwo / acor)
---------------------------------------------
    optimize(objective_fn, bounds, n_pop=30, n_iter=100, seed=None)
        -> (best_x, best_cost, history)
"""

from __future__ import annotations

import numpy as np

# --- ACO_R hyperparameters ---------------------------------------------------
Q = 0.5     # intensification factor (smaller => greedier toward best archive member)
XI = 0.85   # pheromone evaporation / deviation-distance factor (like a learning rate)


def optimize(objective_fn, bounds, n_pop=30, n_iter=100, seed=None):
    """Minimize objective_fn over box `bounds` with ACO_R.

    The archive size k is set equal to n_pop for a fair comparison (same number
    of stored/managed candidate solutions as the other optimizers' population).

    Parameters / Returns: see module docstring (shared interface).
    """
    rng = np.random.default_rng(seed)
    bounds = np.asarray(bounds, dtype=float)
    lo, hi = bounds[:, 0], bounds[:, 1]
    n_dim = bounds.shape[0]
    k = n_pop                                          # archive size

    # --- rank-based Gaussian weights and selection probabilities ------------
    ranks = np.arange(1, k + 1)
    w = (1.0 / (Q * k * np.sqrt(2 * np.pi))) * np.exp(-((ranks - 1) ** 2) / (2 * Q**2 * k**2))
    p = w / w.sum()                                    # p_l, fixed across iterations

    # --- initialize the archive uniformly, sorted best -> worst -------------
    A = lo + (hi - lo) * rng.random((k, n_dim))
    # Batch evaluation: the whole archive is scored in one call (identical to
    # per-row, but fast for a vectorized objective).
    A_cost = np.asarray(objective_fn(A), dtype=float).reshape(-1)
    order = np.argsort(A_cost)
    A, A_cost = A[order], A_cost[order]

    history = np.empty(n_iter)

    for t in range(n_iter):
        # roulette-select a guiding archive solution per new ant
        guides = rng.choice(k, size=n_pop, p=p)        # (n_pop,)

        new_ants = np.empty((n_pop, n_dim))
        for a in range(n_pop):
            l = guides[a]
            for i in range(n_dim):
                mu = A[l, i]
                # sigma = xi * mean absolute distance from guide l to all others
                sigma = XI * np.sum(np.abs(A[:, i] - A[l, i])) / max(1, k - 1)
                new_ants[a, i] = rng.normal(mu, sigma)

        new_ants = np.clip(new_ants, lo, hi)           # respect bounds
        new_cost = np.asarray(objective_fn(new_ants), dtype=float).reshape(-1)  # batch

        # merge new ants with the archive, keep the best k (elitist)
        merged_X = np.vstack([A, new_ants])
        merged_cost = np.concatenate([A_cost, new_cost])
        order = np.argsort(merged_cost)[:k]
        A, A_cost = merged_X[order], merged_cost[order]

        history[t] = A_cost[0]                          # best archive member

    return A[0].copy(), float(A_cost[0]), history


if __name__ == "__main__":
    from objective import PARAM_BOUNDS, objective

    best_x, best_cost, hist = optimize(objective, PARAM_BOUNDS, seed=0)
    print("=== ACOR smoke test on exoskeleton objective J ===")
    print(f"initial best cost : {hist[0]:.4f}")
    print(f"final best cost   : {best_cost:.4f}")
    print(f"improvement       : {hist[0] - best_cost:.4f}")
