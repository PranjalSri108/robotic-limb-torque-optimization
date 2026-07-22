"""
objective.py
============

Forward-dynamics *tracking* objective for the 3-DOF sagittal-plane exoskeleton.

Problem (upgraded from inverse-dynamics effort minimization)
------------------------------------------------------------
A fixed reference gait q_ref(t) (see exo_model.REF_GAIT_PARAMS) is tracked by a
closed loop with a hand-tuned PD base controller plus an optimized feedforward
torque:

    tau(t) = tau_ff(t) + Kp (q_ref - q) + Kd (q_dot_ref - q_dot)

The closed loop is forward-simulated (fixed-step RK4) via
q_ddot = M^-1 (tau - C q_dot - G). The cost trades actuation effort against
tracking accuracy:

    J = integral_0^T ( sum_j tau_j^2 ) dt
      + w_track * integral_0^T ( sum_j (q_j - q_ref,j)^2 ) dt
      + hard penalty if RMS tracking error exceeds a threshold
      + large finite penalty if the forward simulation diverges.

J returns a single float, and (for a big speed-up) also accepts a 2-D batch of
decision vectors and returns one cost per row.

Decision variables (27 total)
-----------------------------
The feedforward torque tau_ff(t) per joint is a truncated Fourier series
(4 harmonics + offset = 9 coeffs/joint, 3 joints => 27). Flat layout is
joint-major, per joint:
    [ a0,  a1..a4 (cos coeffs),  b1..b4 (sin coeffs) ]

Bounds
------
Every coefficient is bounded to +/- TAU_MAX = 50 N m. This comfortably brackets
the optimum (a least-squares Fourier fit to the reference gait's inverse-dynamics
torque peaks at ~39 N m) while keeping the search box symmetric and interpretable
(each coefficient is a torque contribution) and the closed loop stably integrable.
"""

from __future__ import annotations

import numpy as np

from exo_model import (
    GAIT_PERIOD,
    N_DOF,
    N_FF_PARAMS,
    forward_simulate,
)

# =============================================================================
# COST WEIGHTS / PENALTY CONSTANTS  (documented)
# =============================================================================
# w_track balances tracking against effort. At the PD-only baseline the effort
# integral is ~4.6e3 and the tracking integral ~3.5e-2; W_TRACK = 5e4 makes the
# tracking term (~1.7e3 at baseline) the same order as effort, so the optimizer
# is meaningfully rewarded for reducing tracking error without ignoring torque.
# This term also grows steeply as tracking worsens (it is ~integral of error^2),
# giving a smooth descent gradient across the whole search box.
W_TRACK = 5.0e4

# Hard feasibility penalty on RMS tracking error (over time & joints). It is a
# STEEP but CONTINUOUS quadratic in the threshold-exceedance: zero below
# RMS_THRESH, growing as HARD_PENALTY * (rms - thresh)^2 above it. Continuity at
# the threshold (no flat cliff) keeps the landscape descendable for the
# population optimizers while still strongly rejecting poor-tracking solutions.
# PD-only tracks at ~0.10 rad, so 0.15 rad flags "worse than doing nothing".
RMS_THRESH = 0.15                 # [rad]
HARD_PENALTY = 1.0e5              # [cost / rad^2] steepness above the threshold

# Divergence (NaN / exploding state) -> large but finite cost so optimizers keep
# working with real numbers instead of crashing. (In practice RK4@300 steps does
# not diverge on this box, but the guard is kept for robustness.)
DIVERGENCE_PENALTY = 1.0e12

# --- decision-variable bounds: (27, 2) array of [lower, upper] ---------------
# +/-50 N m per coefficient comfortably contains the optimum (a least-squares
# Fourier fit to the reference gait's inverse-dynamics torque peaks at ~39 N m)
# while keeping the closed-loop dynamics gentle enough to integrate stably.
TAU_MAX = 50.0                    # [N m] per Fourier coefficient
N_PARAMS = N_FF_PARAMS            # 27
PARAM_BOUNDS = np.column_stack([
    np.full(N_PARAMS, -TAU_MAX),
    np.full(N_PARAMS, +TAU_MAX),
])


def random_valid_params(rng: np.random.Generator | None = None) -> np.ndarray:
    """Draw a random decision vector uniformly within PARAM_BOUNDS."""
    rng = np.random.default_rng() if rng is None else rng
    lo, hi = PARAM_BOUNDS[:, 0], PARAM_BOUNDS[:, 1]
    return lo + (hi - lo) * rng.random(N_PARAMS)


def _costs_from_sim(sim) -> np.ndarray:
    """Assemble per-member scalar costs from a forward_simulate() result dict."""
    tau2 = sim["integral_tau2"]
    err2 = sim["integral_err2"]
    rms = sim["rms_err"]
    diverged = sim["diverged"]

    cost = tau2 + W_TRACK * err2

    # steep-but-continuous quadratic hard penalty above the RMS threshold
    exceed = np.maximum(0.0, rms - RMS_THRESH)
    cost = cost + HARD_PENALTY * exceed**2

    # divergence dominates everything with a large finite value
    cost = np.where(diverged, DIVERGENCE_PENALTY, cost)
    return cost


def objective(params: np.ndarray):
    """Scalar tracking cost J(params) to be minimized.

    Accepts a single (27,) vector -> float, or a (n, 27) batch -> (n,) array.
    The batch path forward-simulates the whole population in one vectorized RK4
    sweep, which is a large speed-up for the 30-run experiment.
    """
    arr = np.asarray(params, dtype=float)
    single = arr.ndim == 1
    P = arr.reshape(1, -1) if single else arr
    if P.shape[1] != N_PARAMS:
        raise ValueError(f"expected {N_PARAMS} decision variables, got {P.shape[1]}")

    sim = forward_simulate(P)
    cost = _costs_from_sim(sim)
    return float(cost[0]) if single else cost


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    # --- SANITY CHECK: PD alone (tau_ff = 0) should roughly track, no blow-up.
    zero_ff = np.zeros(N_PARAMS)
    sim = forward_simulate(zero_ff)
    print("=== PD-only sanity check (feedforward tau_ff = 0) ===")
    print(f"  diverged        : {bool(sim['diverged'][0])}")
    print(f"  RMS tracking err : {sim['rms_err'][0]:.5f} rad "
          f"({np.rad2deg(sim['rms_err'][0]):.2f} deg)")
    print(f"  integral tau^2   : {sim['integral_tau2'][0]:.2f}")
    print(f"  J (PD only)      : {objective(zero_ff):.2f}")
    assert not sim["diverged"][0], "PD-only closed loop diverged -- retune gains!"

    # --- decision-variable summary ------------------------------------------
    print(f"\nDecision variables: {N_PARAMS} "
          f"(9 per joint x {N_DOF} joints); bounds +/-{TAU_MAX} N m each")

    # --- evaluate J on a random valid vector (proves it runs) ---------------
    rng = np.random.default_rng(0)
    x = random_valid_params(rng)
    print(f"\nJ(random valid params) = {objective(x):.4f}")

    # --- batch == per-row consistency (vectorization sanity) ----------------
    X = np.array([random_valid_params(np.random.default_rng(s)) for s in range(5)])
    batch = objective(X)
    perrow = np.array([objective(row) for row in X])
    print(f"batch vs per-row max diff: {np.max(np.abs(batch - perrow)):.2e}")
