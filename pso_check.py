"""
pso_check.py
============

Control experiment: does a *tuned* PSO (pso_tuned.py) close the gap to GWO/ACOR,
or does the ranking survive? This addresses the reviewer concern that vanilla
PSO's poor showing is merely a tuning artifact.

What it does
------------
1. Runs PSO_tuned on the SAME 30 seeds (0..29) on the forward-dynamics tracking
   objective, with the same n_pop / n_iter as the other optimizers.
2. Reuses results_30runs.csv for PSO / GWO / ACOR (does NOT re-run them).
3. Reports summary statistics + infeasible counts for all four, and pairwise
   Wilcoxon signed-rank tests (Holm-Bonferroni corrected) of PSO_tuned vs each
   of the original three, with rank-biserial effect sizes.
4. Writes results_pso_check.csv (per-seed rows for all four algorithms) and
   pso_tuning_check.txt (table + tests + one-line verdict).

Original pso.py is left untouched; the paper reports both vanilla and tuned.
"""

from __future__ import annotations

# Pin BLAS/OpenMP to one thread per process BEFORE numpy is imported (the pool
# workers would otherwise oversubscribe the CPU). See compare.py for details.
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from scipy import stats

import pso_tuned
from exo_model import forward_simulate
from objective import PARAM_BOUNDS, RMS_THRESH, objective
# reuse the exact statistics helpers and protocol constants from compare.py
from compare import holm_bonferroni, rank_biserial, N_SEEDS, N_POP, N_ITER, SEEDS

NEW_ALGO = "PSO_tuned"
ORIG_ALGOS = ["PSO", "GWO", "ACOR"]
ALL_ALGOS = [NEW_ALGO] + ORIG_ALGOS
ORIG_CSV = "results_30runs.csv"


# =============================================================================
# RUN PSO_tuned (parallel; reproducible per seed)
# =============================================================================
def _run_tuned(seed):
    """Run PSO_tuned for one seed; return final cost + feasibility."""
    best_x, best_cost, _ = pso_tuned.optimize(
        objective, PARAM_BOUNDS, n_pop=N_POP, n_iter=N_ITER, seed=seed)
    sim = forward_simulate(best_x)
    infeasible = bool(sim["diverged"][0] or sim["rms_err"][0] > RMS_THRESH)
    return {"algorithm": NEW_ALGO, "seed": seed,
            "final_cost": float(best_cost), "constraint_active": infeasible}


def run_pso_tuned():
    n_workers = min(cpu_count(), N_SEEDS)
    print(f"Running {NEW_ALGO} on {N_SEEDS} seeds (0..{N_SEEDS-1}) "
          f"on {n_workers} workers (n_pop={N_POP}, n_iter={N_ITER})...")
    with Pool(processes=n_workers) as pool:
        rows = pool.map(_run_tuned, SEEDS)
    return pd.DataFrame(rows)


# =============================================================================
# LOAD ORIGINAL RESULTS
# =============================================================================
def load_original():
    """Load PSO/GWO/ACOR per-seed final cost + feasibility from results_30runs.csv."""
    df = pd.read_csv(ORIG_CSV)
    # coerce the boolean column robustly (CSV stores it as text)
    df["constraint_active"] = (df["constraint_active"].astype(str).str.strip()
                               .isin(["True", "true", "1"]))
    return df[["algorithm", "seed", "final_cost", "constraint_active"]].copy()


# =============================================================================
# STATISTICS + REPORT
# =============================================================================
def _cost_by_seed(df, algo):
    """Final costs aligned by ascending seed for one algorithm."""
    sub = df[df["algorithm"] == algo].set_index("seed").loc[SEEDS]
    return sub["final_cost"].to_numpy()


def build_report(df_all):
    lines = []
    lines.append("=" * 72)
    lines.append("PSO TUNING CHECK: does a well-tuned PSO close the gap?")
    lines.append("Forward-dynamics gait TRACKING objective J  |  lower cost is better")
    lines.append(f"{N_SEEDS} seeds (0..{N_SEEDS-1}), n_pop={N_POP}, n_iter={N_ITER}")
    lines.append("PSO_tuned: w 0.9->0.4, c1=c2=1.49445, |v|<=0.2*(hi-lo)  "
                 "(vanilla PSO: c1=c2=2.0, |v|<=1.0*(hi-lo))")
    lines.append("=" * 72)

    # --- summary table -------------------------------------------------------
    lines.append("\n[1] SUMMARY STATISTICS (final cost)")
    header = (f"  {'algo':<10} {'mean':>10} {'std':>10} {'best':>10} {'worst':>11} "
              f"{'median':>10} {'infeas':>7}")
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    order = sorted(ALL_ALGOS, key=lambda a: _cost_by_seed(df_all, a).mean())
    for a in order:
        c = _cost_by_seed(df_all, a)
        infeas = int(df_all[df_all["algorithm"] == a]["constraint_active"].sum())
        lines.append(f"  {a:<10} {c.mean():>10.3f} {c.std(ddof=1):>10.3f} {c.min():>10.3f} "
                     f"{c.max():>11.3f} {np.median(c):>10.3f} {infeas:>7d}")
    lines.append(f"\n  infeas = # of {N_SEEDS} runs violating tracking constraint "
                 f"(RMS > {RMS_THRESH} rad or diverged)")

    # --- pairwise Wilcoxon: PSO_tuned vs each original -----------------------
    pairs = [(NEW_ALGO, a) for a in ORIG_ALGOS]
    raw_p, wstats, effects = [], [], []
    tuned = _cost_by_seed(df_all, NEW_ALGO)
    for _, b in pairs:
        other = _cost_by_seed(df_all, b)
        w, p = stats.wilcoxon(tuned, other)
        wstats.append(float(w)); raw_p.append(float(p))
        effects.append(rank_biserial(tuned, other))
    corr_p = holm_bonferroni(raw_p)

    lines.append("\n[2] PAIRWISE WILCOXON SIGNED-RANK: PSO_tuned vs each original")
    lines.append("    (Holm-Bonferroni corrected over the 3 comparisons)")
    lines.append(f"  {'pair':<20} {'W':>9} {'raw p':>12} {'corr p':>12} "
                 f"{'r (rank-bis.)':>14} {'sig':>5}")
    lines.append("  " + "-" * 76)
    for (a, b), w, rp, cp, eff in zip(pairs, wstats, raw_p, corr_p, effects):
        sig = "yes" if cp < 0.05 else "no"
        lines.append(f"  {a+' vs '+b:<20} {w:>9.2f} {rp:>12.3e} {cp:>12.3e} "
                     f"{eff:>14.3f} {sig:>5}")
    lines.append("\n  r = rank-biserial effect size; r>0 => PSO_tuned has HIGHER "
                 "(worse) cost than the comparator.")

    # --- verdict -------------------------------------------------------------
    mean_tuned = tuned.mean()
    mean_van = _cost_by_seed(df_all, "PSO").mean()
    mean_gwo = _cost_by_seed(df_all, "GWO").mean()
    # index of the GWO comparison
    gwo_i = ORIG_ALGOS.index("GWO")
    tuned_vs_gwo_sig = corr_p[gwo_i] < 0.05
    tuned_worse_than_gwo = effects[gwo_i] > 0
    improved_vs_vanilla = mean_tuned < mean_van

    lines.append("\n[3] VERDICT")
    improve_pct = 100.0 * (mean_van - mean_tuned) / mean_van
    base = (f"Tuning {'improved' if improved_vs_vanilla else 'did not improve'} PSO "
            f"({mean_van:.0f} -> {mean_tuned:.0f}, "
            f"{improve_pct:+.1f}% vs vanilla).")
    if tuned_vs_gwo_sig and tuned_worse_than_gwo:
        verdict = (f"  {base} GWO still significantly beats PSO_tuned "
                   f"(mean {mean_gwo:.0f} vs {mean_tuned:.0f}, Holm p="
                   f"{corr_p[gwo_i]:.2e}): the ranking SURVIVES -- PSO's deficit is "
                   f"not merely a tuning artifact.")
    elif not tuned_vs_gwo_sig:
        verdict = (f"  {base} PSO_tuned is statistically indistinguishable from GWO "
                   f"(Holm p={corr_p[gwo_i]:.2e}): tuning CLOSES the gap -- PSO's "
                   f"deficit was largely a tuning artifact.")
    else:
        verdict = (f"  {base} PSO_tuned now significantly beats GWO "
                   f"(Holm p={corr_p[gwo_i]:.2e}): tuning MORE than closes the gap.")
    lines.append(verdict)

    return "\n".join(lines) + "\n"


def main():
    df_tuned = run_pso_tuned()
    df_orig = load_original()
    df_all = pd.concat([df_tuned, df_orig], ignore_index=True)

    # per-seed CSV for all four algorithms (PSO_tuned rows added to the reused three)
    df_all.sort_values(["algorithm", "seed"]).to_csv("results_pso_check.csv", index=False)
    print("Saved: results_pso_check.csv")

    report = build_report(df_all)
    with open("pso_tuning_check.txt", "w") as f:
        f.write(report)
    print("Saved: pso_tuning_check.txt\n")
    print(report)


if __name__ == "__main__":
    main()
