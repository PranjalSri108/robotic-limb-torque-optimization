"""
compare.py
==========

Publication-grade comparison of the three from-scratch optimizers
(PSO / GWO / ACOR) on the FORWARD-DYNAMICS TRACKING objective J (objective.py):
a fixed reference gait tracked by a PD base controller plus an optimized
feedforward torque tau_ff(t) (27 Fourier coefficients).

Pipeline (same 30-seed protocol and statistics as the inverse-dynamics study)
-----------------------------------------------------------------------------
1. PD-only sanity check (tau_ff = 0) printed first: confirm the closed loop
   tracks roughly and does not diverge.
2. Run every optimizer for N_SEEDS = 30 seeds (0..29), same n_pop / n_iter.
   Runs execute across a process pool; results are exactly reproducible per
   seed (each optimizer seeds its own RNG; parallelism changes only timing).
3. Metrics per run: final cost, iterations-to-within-5%-of-best-known, wall
   time, and whether the final solution is INFEASIBLE (RMS tracking error above
   threshold, or a diverged simulation).
4. Statistics (scipy.stats): Friedman + pairwise Wilcoxon signed-rank with
   Holm-Bonferroni correction and rank-biserial effect sizes.
5. Outputs:
     * results_30runs.csv   -- one row per (algorithm, seed).
     * stats_summary.txt     -- metric table + all tests + conclusion.
     * convergence_curves.png, final_cost_boxplot.png, torque_profiles.png,
       tracking_error.png   (300 dpi, IEEE two-column sizing).

Dependencies: numpy, scipy, matplotlib, pandas (+ stdlib multiprocessing).
The optimizers and the objective are NOT modified here.
"""

from __future__ import annotations

# IMPORTANT: pin BLAS/OpenMP to a single thread PER PROCESS *before* numpy is
# imported. The forward-dynamics objective is dominated by many tiny linear
# solves; with the process pool, letting each of the 16 workers also spawn
# multi-threaded BLAS oversubscribes the CPU and thrashes (a ~5 min job ballooned
# to ~28 min and made per-run wall-clock timing meaningless). One thread/process
# lets the 16 workers map cleanly onto the 16 cores.
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import time
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pso
import gwo
import acor
from exo_model import GAIT_PERIOD, N_DOF, N_FF_PARAMS, forward_simulate
from objective import PARAM_BOUNDS, RMS_THRESH, objective

# --- experiment configuration ------------------------------------------------
N_SEEDS = 30
N_POP = 30
N_ITER = 100
SEEDS = list(range(N_SEEDS))
CONV_TOL = 0.05                  # "convergence" = within 5% of best-known cost

ALGORITHMS = {"PSO": pso.optimize, "GWO": gwo.optimize, "ACOR": acor.optimize}
ALGO_ORDER = ["PSO", "GWO", "ACOR"]
COLORS = {"PSO": "tab:blue", "GWO": "tab:orange", "ACOR": "tab:green"}
JOINT_NAMES = ["Hip", "Knee", "Ankle"]

# --- IEEE two-column figure styling (single-column width ~3.5 in, 300 dpi) ---
COL_WIDTH = 3.5
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 300, "savefig.dpi": 300, "lines.linewidth": 1.2,
})


# =============================================================================
# WORKER (top-level so it is picklable by multiprocessing)
# =============================================================================
def _run_one(task):
    """Run one (algorithm, seed) optimization and time it."""
    name, seed = task
    solve = ALGORITHMS[name]
    t0 = time.perf_counter()
    best_x, best_cost, history = solve(objective, PARAM_BOUNDS,
                                       n_pop=N_POP, n_iter=N_ITER, seed=seed)
    wall = time.perf_counter() - t0
    return {
        "algorithm": name, "seed": seed,
        "best_x": np.asarray(best_x, dtype=float),
        "final_cost": float(best_cost),
        "history": np.asarray(history, dtype=float),
        "wall_time_s": wall,
    }


def run_experiments():
    tasks = [(name, seed) for name in ALGO_ORDER for seed in SEEDS]
    n_workers = min(cpu_count(), len(tasks))
    print(f"Running {len(ALGO_ORDER)} optimizers x {N_SEEDS} seeds = {len(tasks)} runs "
          f"on {n_workers} workers (n_pop={N_POP}, n_iter={N_ITER})...")
    with Pool(processes=n_workers) as pool:
        return pool.map(_run_one, tasks)


def measure_uncontended_timing():
    """Time one run per algorithm serially in-process (no pool contention).

    The per-seed wall times recorded during the parallel sweep are inflated by
    memory-bandwidth contention among the 16 workers, so they are NOT a fair
    per-run cost. This measures each optimizer once on an otherwise-idle CPU to
    report a representative uncontended wall-clock time per run. (Costs are
    deterministic in the seed, so this does not affect any result -- only timing.)
    """
    timing = {}
    for name in ALGO_ORDER:
        t0 = time.perf_counter()
        ALGORITHMS[name](objective, PARAM_BOUNDS, n_pop=N_POP, n_iter=N_ITER, seed=0)
        timing[name] = time.perf_counter() - t0
    return timing


# =============================================================================
# PER-RUN METRICS
# =============================================================================
def _infeasible(best_x) -> bool:
    """True if the optimized solution violates the tracking constraint.

    The feasibility constraint of the tracking problem is RMS tracking error
    <= RMS_THRESH (or the simulation diverging). Mirrors objective.py's hard
    penalty condition.
    """
    sim = forward_simulate(best_x)
    return bool(sim["diverged"][0] or sim["rms_err"][0] > RMS_THRESH)


def _iters_to_converge(history, threshold) -> int:
    reached = np.where(history <= threshold)[0]
    return int(reached[0] + 1) if reached.size else N_ITER


def build_dataframe(results):
    best_known = min(r["final_cost"] for r in results)
    threshold = best_known * (1.0 + CONV_TOL)
    rows = []
    for r in results:
        rows.append({
            "algorithm": r["algorithm"], "seed": r["seed"],
            "final_cost": r["final_cost"],
            "iters_to_5pct": _iters_to_converge(r["history"], threshold),
            "wall_time_s": r["wall_time_s"],
            "constraint_active": _infeasible(r["best_x"]),
        })
    df = pd.DataFrame(rows).sort_values(["algorithm", "seed"]).reset_index(drop=True)
    return df, best_known, threshold


# =============================================================================
# STATISTICS  (identical protocol to the inverse-dynamics study)
# =============================================================================
def _final_cost_matrix(df):
    seeds = sorted(df["seed"].unique())
    return {n: df[df["algorithm"] == n].set_index("seed").loc[seeds]["final_cost"].to_numpy()
            for n in ALGO_ORDER}


def rank_biserial(a, b):
    """Matched-pairs rank-biserial correlation (Kerby 2014). r>0 => a > b."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[d != 0.0]
    if d.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    w_plus, w_minus = ranks[d > 0].sum(), ranks[d < 0].sum()
    total = w_plus + w_minus
    return float((w_plus - w_minus) / total) if total > 0 else 0.0


def holm_bonferroni(pvals):
    p = np.asarray(pvals, float)
    m = p.size
    order = np.argsort(p)
    corrected = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        corrected[idx] = min(running, 1.0)
    return corrected


def run_statistics(df):
    cols = _final_cost_matrix(df)
    fried_chi2, fried_p = stats.friedmanchisquare(*[cols[n] for n in ALGO_ORDER])
    pairs = [("PSO", "GWO"), ("PSO", "ACOR"), ("GWO", "ACOR")]
    raw_p, wstats, effects = [], [], []
    for a, b in pairs:
        w, p = stats.wilcoxon(cols[a], cols[b])
        wstats.append(float(w)); raw_p.append(float(p))
        effects.append(rank_biserial(cols[a], cols[b]))
    return {
        "friedman": (float(fried_chi2), float(fried_p)),
        "pairs": pairs, "wstats": wstats, "raw_p": raw_p,
        "corr_p": holm_bonferroni(raw_p).tolist(), "effects": effects,
    }


# =============================================================================
# SUMMARY TABLE + TEXT REPORT
# =============================================================================
def summarize(df):
    g = df.groupby("algorithm")
    return pd.DataFrame({
        "mean": g["final_cost"].mean(),
        "std": g["final_cost"].std(ddof=1),
        "best": g["final_cost"].min(),
        "worst": g["final_cost"].max(),
        "median": g["final_cost"].median(),
        "conv_speed_iters": g["iters_to_5pct"].mean(),
        "constraint_runs": g["constraint_active"].sum().astype(int),
    }).loc[ALGO_ORDER]


def write_report(summary, stat, best_known, threshold, pd_rms, timing,
                 path="stats_summary.txt"):
    lines = []
    lines.append("=" * 72)
    lines.append("PUBLICATION-GRADE COMPARISON: PSO vs GWO vs ACOR")
    lines.append("Forward-dynamics gait TRACKING objective J  |  lower cost is better")
    lines.append("J = int(sum tau^2) dt + w_track int(sum (q-q_ref)^2) dt + hard-RMS penalty")
    lines.append(f"{N_SEEDS} seeds (0..{N_SEEDS-1}), n_pop={N_POP}, n_iter={N_ITER} "
                 f"(~{N_POP*(N_ITER+1)} evals/run)")
    lines.append(f"PD-only baseline (tau_ff=0): RMS tracking error = {pd_rms:.4f} rad")
    lines.append("=" * 72)

    lines.append("\n[1] PER-ALGORITHM METRICS (final cost unless noted)")
    lines.append(f"  best-known cost over all runs : {best_known:.4f}")
    lines.append(f"  5% convergence threshold      : {threshold:.4f}")
    lines.append("")
    header = (f"  {'algo':<5} {'mean':>9} {'std':>8} {'best':>9} {'worst':>9} "
              f"{'median':>9} {'conv(it)':>9} {'time(s)':>8} {'infeas':>7}")
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for name in ALGO_ORDER:
        r = summary.loc[name]
        lines.append(f"  {name:<5} {r['mean']:>9.3f} {r['std']:>8.3f} {r['best']:>9.3f} "
                     f"{r['worst']:>9.3f} {r['median']:>9.3f} {r['conv_speed_iters']:>9.2f} "
                     f"{timing[name]:>8.3f} {int(r['constraint_runs']):>7d}")
    lines.append("")
    lines.append("  conv(it) = mean iterations to reach within 5% of best-known "
                 "(censored at n_iter)")
    lines.append("  time(s)  = uncontended single-run wall clock (see note); the "
                 "per-seed\n             wall_time_s in results_30runs.csv is the "
                 "parallel-pool time,\n             which is inflated by memory-bandwidth "
                 "contention across workers.")
    lines.append(f"  infeas   = # of runs whose final solution violates the tracking "
                 f"constraint (RMS > {RMS_THRESH} rad or diverged)")

    chi2, fp = stat["friedman"]
    lines.append("\n[2] FRIEDMAN TEST (paired across the 3 algorithms, blocked by seed)")
    lines.append(f"  chi^2 = {chi2:.4f}   p = {fp:.3e}   "
                 f"({'significant' if fp < 0.05 else 'not significant'} at alpha=0.05)")

    lines.append("\n[3] PAIRWISE WILCOXON SIGNED-RANK (Holm-Bonferroni corrected)")
    lines.append(f"  {'pair':<12} {'W':>10} {'raw p':>12} {'corr p':>12} "
                 f"{'r (rank-bis.)':>14} {'sig':>5}")
    lines.append("  " + "-" * 68)
    for (a, b), w, rp, cp, eff in zip(stat["pairs"], stat["wstats"],
                                      stat["raw_p"], stat["corr_p"], stat["effects"]):
        sig = "yes" if cp < 0.05 else "no"
        lines.append(f"  {a+'-'+b:<12} {w:>10.2f} {rp:>12.3e} {cp:>12.3e} "
                     f"{eff:>14.3f} {sig:>5}")
    lines.append("\n  r = rank-biserial effect size; r>0 => first algorithm has HIGHER "
                 "(worse) cost.")
    lines.append("  |r|: ~0.1 small, ~0.3 medium, ~0.5 large (Cohen-style guide).")

    ranking = summary["mean"].sort_values().index.tolist()
    order_str = " < ".join(ranking)
    best_algo = ranking[0]
    fried_sig = fp < 0.05
    best_sig = []
    for (a, b), cp in zip(stat["pairs"], stat["corr_p"]):
        if best_algo in (a, b) and cp < 0.05:
            best_sig.append(b if a == best_algo else a)

    lines.append("\n[4] CONCLUSION")
    if fried_sig and best_sig:
        lines.append(f"  Ranking by mean final cost (best first): {order_str}. "
                     f"Friedman is significant (p={fp:.2e}); {best_algo} significantly "
                     f"outperforms {', '.join(best_sig)} after Holm correction.")
    elif fried_sig:
        lines.append(f"  Ranking by mean final cost (best first): {order_str}. "
                     f"Friedman is significant (p={fp:.2e}), but no pairwise difference "
                     f"involving {best_algo} survives Holm correction.")
    else:
        lines.append(f"  Ranking by mean final cost (best first): {order_str}. "
                     f"Friedman is not significant (p={fp:.2e}); differences may be "
                     f"due to chance.")

    text = "\n".join(lines) + "\n"
    with open(path, "w") as f:
        f.write(text)
    return text


# =============================================================================
# FIGURES
# =============================================================================
def _best_solution_per_algo(results):
    best = {}
    for name in ALGO_ORDER:
        runs = [r for r in results if r["algorithm"] == name]
        best[name] = min(runs, key=lambda r: r["final_cost"])["best_x"]
    return best


def plot_convergence(results):
    hist = {n: np.vstack([r["history"] for r in results if r["algorithm"] == n])
            for n in ALGO_ORDER}
    fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.6))
    iters = np.arange(1, N_ITER + 1)
    for name in ALGO_ORDER:
        h = hist[name]
        mean, std = h.mean(axis=0), h.std(axis=0, ddof=1)
        ax.plot(iters, mean, color=COLORS[name], label=name)
        ax.fill_between(iters, mean - std, mean + std, color=COLORS[name],
                        alpha=0.2, linewidth=0)
    ax.set_xlabel("Iteration"); ax.set_ylabel("Best cost $J$")
    ax.set_yscale("log")
    ax.set_title(f"Convergence (mean $\\pm$1 std, {N_SEEDS} seeds)")
    ax.legend(); ax.grid(True, which="both", ls=":", alpha=0.5)
    fig.tight_layout(); fig.savefig("convergence_curves.png"); plt.close(fig)


def plot_boxplot(df):
    fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.6))
    data = [df[df["algorithm"] == n]["final_cost"].to_numpy() for n in ALGO_ORDER]
    bp = ax.boxplot(data, tick_labels=ALGO_ORDER, patch_artist=True, widths=0.6)
    for patch, name in zip(bp["boxes"], ALGO_ORDER):
        patch.set_facecolor(COLORS[name]); patch.set_alpha(0.5)
    for med in bp["medians"]:
        med.set_color("black")
    ax.set_ylabel("Final cost $J$")
    ax.set_title(f"Final-cost distribution ({N_SEEDS} seeds)")
    ax.grid(True, axis="y", ls=":", alpha=0.5)
    fig.tight_layout(); fig.savefig("final_cost_boxplot.png"); plt.close(fig)


def plot_torque_profiles(results):
    """Realized closed-loop joint torque over the gait cycle (best run/algo)."""
    best_x = _best_solution_per_algo(results)
    fig, axes = plt.subplots(N_DOF, 1, figsize=(COL_WIDTH, 5.0), sharex=True)
    for name in ALGO_ORDER:
        sim = forward_simulate(best_x[name], return_history=True)
        t, tau = sim["t"], sim["tau"][:, 0, :]        # (S+1,), (S+1, 3)
        for j in range(N_DOF):
            axes[j].plot(t, tau[:, j], color=COLORS[name], label=name)
    for j in range(N_DOF):
        axes[j].set_ylabel(f"{JOINT_NAMES[j]}\n$\\tau$ [N m]")
        axes[j].grid(True, ls=":", alpha=0.5)
        axes[j].axhline(0.0, color="k", lw=0.5)
    axes[0].set_title("Closed-loop joint torques (best run per algorithm)")
    axes[0].legend(loc="upper right", ncol=3, columnspacing=1.0, handlelength=1.2)
    axes[-1].set_xlabel("Time [s]")
    fig.tight_layout(); fig.savefig("torque_profiles.png"); plt.close(fig)


def plot_tracking_error(results):
    """Tracking: q(t) vs q_ref(t) per joint for each algorithm's best solution."""
    best_x = _best_solution_per_algo(results)
    sims = {n: forward_simulate(best_x[n], return_history=True) for n in ALGO_ORDER}
    ref = sims[ALGO_ORDER[0]]                          # fixed reference (same for all)
    t, q_ref = ref["t"], ref["q_ref"]                  # (S+1,), (S+1, 3)

    fig, axes = plt.subplots(N_DOF, 1, figsize=(COL_WIDTH, 5.0), sharex=True)
    for j in range(N_DOF):
        axes[j].plot(t, q_ref[:, j], color="k", ls="--", lw=1.4, label="$q_{ref}$")
        for name in ALGO_ORDER:
            q = sims[name]["q"][:, 0, :]               # (S+1, 3)
            axes[j].plot(t, q[:, j], color=COLORS[name], label=name)
        axes[j].set_ylabel(f"{JOINT_NAMES[j]}\n$q$ [rad]")
        axes[j].grid(True, ls=":", alpha=0.5)
    axes[0].set_title("Tracking: $q$ vs $q_{ref}$ (best run per algorithm)")
    axes[0].legend(loc="upper right", ncol=4, columnspacing=0.9, handlelength=1.1)
    axes[-1].set_xlabel("Time [s]")
    fig.tight_layout(); fig.savefig("tracking_error.png"); plt.close(fig)


def main():
    t_start = time.perf_counter()

    # --- SANITY CHECK first: PD alone must roughly track, not diverge --------
    pd_sim = forward_simulate(np.zeros(N_FF_PARAMS))
    pd_rms = float(pd_sim["rms_err"][0])
    print("=== PD-only sanity check (tau_ff = 0) ===")
    print(f"  diverged = {bool(pd_sim['diverged'][0])}   "
          f"RMS tracking error = {pd_rms:.5f} rad ({np.rad2deg(pd_rms):.2f} deg)\n")
    assert not pd_sim["diverged"][0], "PD-only closed loop diverged -- retune gains!"

    results = run_experiments()

    df, best_known, threshold = build_dataframe(results)
    df.to_csv("results_30runs.csv", index=False)
    print("Saved: results_30runs.csv")

    summary = summarize(df)
    stat = run_statistics(df)

    print("Measuring uncontended per-run wall clock (one run per algorithm)...")
    timing = measure_uncontended_timing()

    report = write_report(summary, stat, best_known, threshold, pd_rms, timing)
    print("Saved: stats_summary.txt\n")
    print(report)

    plot_convergence(results)
    plot_boxplot(df)
    plot_torque_profiles(results)
    plot_tracking_error(results)
    print("Saved: convergence_curves.png, final_cost_boxplot.png, "
          "torque_profiles.png, tracking_error.png")
    print(f"\nTotal wall-clock: {time.perf_counter() - t_start:.1f} s")


if __name__ == "__main__":
    main()
