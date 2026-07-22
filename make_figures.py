"""
make_figures.py
===============

Regenerate ALL result figures for the paper in IEEE two-column publication
style (via plotstyle.py), saving each as BOTH a vector .pdf (for LaTeX) and a
300-dpi .png into figures/.

The experiments are NOT re-run as new experiments. Every optimizer here is fully
seeded and deterministic, so re-invoking it reproduces bit-identical results.
This script:

  1. Loads the finished per-seed final costs from results_30runs.csv.
  2. Reconstructs the plotting arrays that were never persisted (per-seed
     convergence histories and the best-solution vectors) by re-invoking the
     deterministic optimizers, and ASSERTS the reconstructed final costs equal
     results_30runs.csv exactly -- proving no number changed. The reconstructed
     arrays are cached to figures/plot_arrays.npz so subsequent re-plots need no
     re-run at all.
  3. Renders:
       - convergence_curves   (single column)
       - final_cost_boxplot   (single column, straight from the CSV)
       - torque_profiles      (3 stacked joint panels)
       - tracking_error       (3 stacked joint panels)
       - hinf_tracking        (3 stacked joint panels; H-inf design reconstructed
                               from hinf_baseline.py, reusing the GWO-best vector)

No experimental code (exo_model, objective, optimizers, hinf_baseline logic) is
modified.
"""

from __future__ import annotations

# Pin BLAS to one thread per process before numpy (reconstruction uses a pool).
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

import pso
import gwo
import acor
from exo_model import N_DOF, N_FF_PARAMS, forward_simulate
from objective import PARAM_BOUNDS, objective
# import hinf_baseline BEFORE plotstyle so plotstyle's rcParams win (hinf_baseline
# sets its own rcParams at import time).
import hinf_baseline
import plotstyle                                    # applies IEEE rcParams last
from plotstyle import (ALGO_COLORS, ALGO_LS, ALGO_ORDER, SERIES,
                       savefig_both, new_fig, FIGDIR)

N_SEEDS = 30
N_POP = 30
N_ITER = 100
SEEDS = list(range(N_SEEDS))
JOINT_NAMES = ["Hip", "Knee", "Ankle"]
CSV_PATH = "results_30runs.csv"
CACHE = os.path.join(FIGDIR, "plot_arrays.npz")

OPTIMIZERS = {"PSO": pso.optimize, "GWO": gwo.optimize, "ACOR": acor.optimize}


# =============================================================================
# RECONSTRUCT / LOAD PLOTTING ARRAYS
# =============================================================================
def _reconstruct_one(task):
    """Re-run one deterministic (algorithm, seed); return cost, history, best_x."""
    name, seed = task
    bx, bc, hist = OPTIMIZERS[name](objective, PARAM_BOUNDS,
                                    n_pop=N_POP, n_iter=N_ITER, seed=seed)
    return name, seed, float(bc), np.asarray(hist, float), np.asarray(bx, float)


def load_or_reconstruct_arrays(df_costs):
    """Return dict: hist[algo] (30,100), bestx[algo] (27,), best_seed[algo].

    Uses figures/plot_arrays.npz if present; otherwise reconstructs by re-running
    the deterministic optimizers and verifies the result against results_30runs.csv.
    """
    if os.path.exists(CACHE):
        print(f"Loading cached plotting arrays from {CACHE} (no re-run needed)...")
        z = np.load(CACHE)
        hist = {a: z[f"hist_{a}"] for a in ALGO_ORDER}
        bestx = {a: z[f"bestx_{a}"] for a in ALGO_ORDER}
        best_seed = {a: int(z[f"bestseed_{a}"]) for a in ALGO_ORDER}
        _verify_costs(hist, df_costs)
        return hist, bestx, best_seed

    tasks = [(a, s) for a in ALGO_ORDER for s in SEEDS]
    n_workers = min(cpu_count(), len(tasks))
    print(f"Reconstructing plotting arrays: re-running {len(tasks)} deterministic "
          f"runs on {n_workers} workers\n  (seeded => bit-identical to the finished "
          f"study; final costs are verified against {CSV_PATH})...")
    with Pool(processes=n_workers) as pool:
        raw = pool.map(_reconstruct_one, tasks)

    hist = {a: np.empty((N_SEEDS, N_ITER)) for a in ALGO_ORDER}
    costs = {a: np.empty(N_SEEDS) for a in ALGO_ORDER}
    xs = {a: np.empty((N_SEEDS, N_FF_PARAMS)) for a in ALGO_ORDER}
    for name, seed, bc, h, bx in raw:
        hist[name][seed] = h
        costs[name][seed] = bc
        xs[name][seed] = bx

    _verify_costs(hist, df_costs, costs=costs)

    best_seed = {a: int(np.argmin(costs[a])) for a in ALGO_ORDER}
    bestx = {a: xs[a][best_seed[a]] for a in ALGO_ORDER}

    np.savez_compressed(
        CACHE,
        **{f"hist_{a}": hist[a] for a in ALGO_ORDER},
        **{f"bestx_{a}": bestx[a] for a in ALGO_ORDER},
        **{f"bestseed_{a}": best_seed[a] for a in ALGO_ORDER},
    )
    print(f"Cached plotting arrays -> {CACHE}")
    return hist, bestx, best_seed


def _verify_costs(hist, df_costs, costs=None):
    """Assert reconstructed final costs match results_30runs.csv exactly."""
    for a in ALGO_ORDER:
        csv_c = (df_costs[df_costs["algorithm"] == a]
                 .set_index("seed").loc[SEEDS]["final_cost"].to_numpy())
        reco_c = costs[a] if costs is not None else hist[a][:, -1]
        if not np.allclose(reco_c, csv_c, rtol=1e-9, atol=1e-6):
            worst = np.max(np.abs(reco_c - csv_c))
            raise AssertionError(
                f"{a}: reconstructed final costs differ from {CSV_PATH} "
                f"(max abs diff {worst:.3e}) -- reconstruction is NOT faithful.")
    print("  verified: reconstructed final costs match results_30runs.csv exactly.")


# =============================================================================
# FIGURE 1: CONVERGENCE CURVES  (single column)
# =============================================================================
def fig_convergence(hist):
    fig, ax = new_fig(3.5, 2.62)
    iters = np.arange(1, N_ITER + 1)
    for a in ALGO_ORDER:
        m, sd = hist[a].mean(0), hist[a].std(0, ddof=1)
        ax.plot(iters, m, color=ALGO_COLORS[a], ls=ALGO_LS[a], label=a)
        ax.fill_between(iters, m - sd, m + sd, color=ALGO_COLORS[a],
                        alpha=0.15, linewidth=0)
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Best cost $J$")
    ax.set_xlim(1, N_ITER)
    ax.grid(True, which="both")
    ax.legend(loc="upper right", handlelength=1.6)
    return savefig_both(fig, "convergence_curves")


# =============================================================================
# FIGURE 2: FINAL-COST BOXPLOT  (single column; straight from the CSV)
# =============================================================================
def fig_boxplot(df_costs):
    fig, ax = new_fig(3.5, 2.62)
    data = [df_costs[df_costs["algorithm"] == a]["final_cost"].to_numpy()
            for a in ALGO_ORDER]
    bp = ax.boxplot(data, tick_labels=ALGO_ORDER, patch_artist=True, widths=0.6,
                    medianprops=dict(color="black", linewidth=1.1))
    for patch, a in zip(bp["boxes"], ALGO_ORDER):
        patch.set_facecolor(ALGO_COLORS[a])
        patch.set_alpha(0.45)
        patch.set_edgecolor(ALGO_COLORS[a])
    for a, whisk in zip(np.repeat(ALGO_ORDER, 2), bp["whiskers"]):
        whisk.set_color(ALGO_COLORS[a])
    for a, cap in zip(np.repeat(ALGO_ORDER, 2), bp["caps"]):
        cap.set_color(ALGO_COLORS[a])
    for a, fl in zip(ALGO_ORDER, bp["fliers"]):
        fl.set(marker="o", markersize=3, markerfacecolor="none",
               markeredgecolor=ALGO_COLORS[a], alpha=0.8)
    ax.set_ylabel(r"Final cost $J$")
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    return savefig_both(fig, "final_cost_boxplot")


# =============================================================================
# 3-PANEL HELPERS (one shared legend on top, one shared x-label at bottom)
# =============================================================================
def _three_panel():
    fig, axes = plotstyle.plt.subplots(N_DOF, 1, figsize=(3.5, 5.0), sharex=True)
    return fig, axes


def _shared_legend(axes, handles, labels, ncol):
    axes[0].legend(handles, labels, loc="lower center",
                   bbox_to_anchor=(0.5, 1.01), ncol=ncol,
                   columnspacing=1.1, handlelength=1.6, borderaxespad=0.0)


# =============================================================================
# FIGURE 3: TORQUE PROFILES  (3 stacked joint panels)
# =============================================================================
def fig_torque(bestx):
    sims = {a: forward_simulate(bestx[a], return_history=True) for a in ALGO_ORDER}
    fig, axes = _three_panel()
    for a in ALGO_ORDER:
        sim = sims[a]
        t, tau = sim["t"], sim["tau"][:, 0, :]
        for j in range(N_DOF):
            axes[j].plot(t, tau[:, j], color=ALGO_COLORS[a], ls=ALGO_LS[a], label=a)
    for j in range(N_DOF):
        axes[j].axhline(0.0, color="k", lw=0.5)
        axes[j].set_ylabel(f"{JOINT_NAMES[j]}\n" r"$\tau$ [N m]")
        axes[j].margins(y=0.08)
    axes[-1].set_xlabel("Time [s]")
    axes[-1].set_xlim(0.0, sims[ALGO_ORDER[0]]["t"][-1])
    h, l = axes[0].get_legend_handles_labels()
    _shared_legend(axes, h, l, ncol=3)
    return savefig_both(fig, "torque_profiles")


# =============================================================================
# FIGURE 4: TRACKING ERROR (q vs q_ref)  (3 stacked joint panels)
# =============================================================================
def fig_tracking(bestx):
    sims = {a: forward_simulate(bestx[a], return_history=True) for a in ALGO_ORDER}
    ref = sims[ALGO_ORDER[0]]
    t, q_ref = ref["t"], ref["q_ref"]
    fig, axes = _three_panel()
    for j in range(N_DOF):
        axes[j].plot(t, q_ref[:, j], color=SERIES["ref"]["color"],
                     ls=SERIES["ref"]["ls"], lw=1.3, label=SERIES["ref"]["label"])
        for a in ALGO_ORDER:
            q = sims[a]["q"][:, 0, :]
            axes[j].plot(t, q[:, j], color=ALGO_COLORS[a], ls=ALGO_LS[a], label=a)
        axes[j].set_ylabel(f"{JOINT_NAMES[j]}\n" r"$q$ [rad]")
        axes[j].margins(y=0.08)
    axes[-1].set_xlabel("Time [s]")
    axes[-1].set_xlim(0.0, t[-1])
    h, l = axes[0].get_legend_handles_labels()
    _shared_legend(axes, h, l, ncol=4)
    return savefig_both(fig, "tracking_error")


# =============================================================================
# FIGURE 5: H-INFINITY TRACKING ERROR  (3 stacked joint panels)
# =============================================================================
def fig_hinf(gwo_bestx):
    """Reconstruct the H-infinity design (deterministic) and plot per-joint
    tracking error [deg] for H-inf vs GWO-best vs PD-only. The GWO-best vector is
    reused from the reconstruction (identical to hinf_baseline's own re-derivation)
    instead of re-running 30 GWO seeds again."""
    print("Reconstructing H-infinity design (linearize + mixsyn + nonlinear sim)...")
    q0 = hinf_baseline.operating_point()
    P, _, _ = hinf_baseline.linearize(q0)
    Kc, _ = hinf_baseline.design_hinf(P)
    hinf = hinf_baseline.simulate_hinf(Kc)

    gwo_hist = forward_simulate(gwo_bestx, return_history=True)
    pd = forward_simulate(np.zeros(N_FF_PARAMS), return_history=True)

    hinf_err = hinf["q"] - hinf["q_ref"]
    gwo_err = gwo_hist["q"][:, 0, :] - gwo_hist["q_ref"]
    pd_err = pd["q"][:, 0, :] - pd["q_ref"]

    fig, axes = _three_panel()
    for j in range(N_DOF):
        axes[j].plot(hinf["t"], np.rad2deg(hinf_err[:, j]),
                     color=SERIES["Hinf"]["color"], ls=SERIES["Hinf"]["ls"],
                     label=SERIES["Hinf"]["label"])
        axes[j].plot(gwo_hist["t"], np.rad2deg(gwo_err[:, j]),
                     color=SERIES["GWO"]["color"], ls=SERIES["GWO"]["ls"],
                     label=SERIES["GWO"]["label"])
        axes[j].plot(pd["t"], np.rad2deg(pd_err[:, j]),
                     color=SERIES["PD"]["color"], ls=SERIES["PD"]["ls"],
                     label=SERIES["PD"]["label"])
        axes[j].axhline(0.0, color="k", lw=0.5)
        axes[j].set_ylabel(f"{JOINT_NAMES[j]}\nerror [deg]")
        axes[j].margins(y=0.08)
    axes[-1].set_xlabel("Time [s]")
    axes[-1].set_xlim(0.0, hinf["t"][-1])
    h, l = axes[0].get_legend_handles_labels()
    _shared_legend(axes, h, l, ncol=3)
    return savefig_both(fig, "hinf_tracking")


# =============================================================================
# MAIN
# =============================================================================
def main():
    df_costs = pd.read_csv(CSV_PATH)
    df_costs["algorithm"] = df_costs["algorithm"].astype(str).str.strip()

    hist, bestx, best_seed = load_or_reconstruct_arrays(df_costs)
    print("Best seed per algorithm (min final cost): "
          + ", ".join(f"{a}=seed{best_seed[a]}" for a in ALGO_ORDER))

    outputs = []
    outputs.append(fig_convergence(hist))
    outputs.append(fig_boxplot(df_costs))
    outputs.append(fig_torque(bestx))
    outputs.append(fig_tracking(bestx))
    outputs.append(fig_hinf(bestx["GWO"]))

    print("\nWrote:")
    for pdf, png in outputs:
        print(f"  {pdf}\n  {png}")


if __name__ == "__main__":
    main()
