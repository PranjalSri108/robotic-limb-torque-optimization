"""
anthro_sensitivity.py
======================

Anthropometric sensitivity study (EVALUATION ONLY -- no optimization is re-run).

Question
--------
The feedforward torque tau_ff(t) (27 Fourier coefficients) was optimized for a
single nominal subject (Winter anthropometry, M_body = 70 kg, H_body = 1.75 m).
Does each optimizer's best solution still track well if the wearer's body mass
and stature differ? We hold the controller FIXED (same 27 coefficients, same PD
gains, same reference gait, same RK4/300 steps, same cost) and only change the
plant's anthropometry, then re-score.

How the plant is re-parameterized
---------------------------------
exo_model derives the rigid-body dynamics symbolically and, in the shipped
module, bakes in the nominal segment parameters. Its forward simulation depends
on anthropometry ONLY through the combined "terms" callable exo_model._TERMS_FUN
(everything else -- PD gains, reference gait, RK4 stage data -- is anthropometry
independent). So we:

  1. Re-derive the SAME symbolic dynamics ONCE, but keep the per-segment
     mass / length / CoM / inertia and gravity as free symbols and lambdify with
     them as arguments (identical expressions to exo_model, just not yet
     numerically substituted).
  2. For each subject variant, rebuild the segment parameters from the SAME
     Winter fractions used in exo_model (mass ~ M_body, length ~ H_body,
     CoM ~ length, inertia ~ mass*length^2), specialize the terms callable to
     those numbers, swap it into exo_model, and call the UNMODIFIED
     forward_simulate / objective. Nothing in the experimental code is edited.

Grid: M in {63, 66.5, 70, 73.5, 77} kg (+/-10%),
      H in {1.6625, 1.75, 1.8375} m  (+/-5%)  ->  15 subject variants.
Solutions: GWO-best, PSO-tuned-best, ACOR-best (recovered from the experiment).

Outputs: anthro_sensitivity.csv, anthro_summary.txt, and one IEEE-style figure
(figures/anthro_sensitivity.{pdf,png}).
"""

from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
import sympy as sp

import exo_model as exo
import objective as obj
import pso_tuned
from objective import PARAM_BOUNDS, RMS_THRESH, W_TRACK

import plotstyle
from plotstyle import savefig_both, new_fig

# --- perturbation grid -------------------------------------------------------
M_NOM, H_NOM = 70.0, 1.75
M_GRID = [63.0, 66.5, 70.0, 73.5, 77.0]          # +/-10 %
H_GRID = [1.6625, 1.75, 1.8375]                  # +/- 5 %
NOMINAL_GWO_J = 2410.4                            # sanity target (study's GWO best)

ALGO_ORDER = ["GWO", "PSO-t", "ACOR"]


# =============================================================================
# 1. RECOVER THE BEST SOLUTION VECTOR PER ALGORITHM (no optimization re-run
#    except the single PSO-tuned best seed, which is not cached)
# =============================================================================
def recover_solutions():
    """Return {algo: (best_x (27,), reported_cost)} for GWO, PSO-t, ACOR.

    GWO and ACOR best vectors are cached in figures/plot_arrays.npz. PSO-tuned's
    best vector was never cached, so we re-run ONLY its single best (seed, algo)
    run -- the deterministic pso_tuned on the seed that produced the best cost in
    results_pso_check.csv -- and verify the cost reproduces.
    """
    cache = np.load(os.path.join(plotstyle.FIGDIR, "plot_arrays.npz"))
    sols = {
        "GWO": (cache["bestx_GWO"].astype(float), 2410.4060),
        "ACOR": (cache["bestx_ACOR"].astype(float), 2633.3878),
    }

    # PSO-tuned: find its best seed in results_pso_check.csv, re-run just that one.
    df = pd.read_csv("results_pso_check.csv")
    df["algorithm"] = df["algorithm"].astype(str).str.strip()
    psot = df[df["algorithm"] == "PSO_tuned"]
    best_row = psot.loc[psot["final_cost"].idxmin()]
    best_seed = int(best_row["seed"])
    reported = float(best_row["final_cost"])
    print(f"Re-deriving PSO-tuned best (seed {best_seed}, reported cost "
          f"{reported:.4f}) by re-running that single deterministic run...")
    bx, bc, _ = pso_tuned.optimize(obj.objective, PARAM_BOUNDS,
                                   n_pop=30, n_iter=100, seed=best_seed)
    if not np.isclose(bc, reported, rtol=1e-9, atol=1e-4):
        raise AssertionError(
            f"PSO-tuned re-run cost {bc:.6f} != reported {reported:.6f}; "
            f"re-derivation is not faithful.")
    print(f"  verified: PSO-tuned re-run reproduces cost {bc:.4f} exactly.")
    sols["PSO-t"] = (np.asarray(bx, float), reported)
    return sols


# =============================================================================
# 2. SYMBOLICALLY PARAMETERIZED DYNAMICS (derived ONCE; params kept as symbols)
# =============================================================================
def build_param_terms():
    """Re-derive exo_model's dynamics but lambdify the combined 'terms' callable
    with the segment mass/length/CoM/inertia and gravity kept as ARGUMENTS.

    Returns terms_param(qs, qds, m, L, d, Iz, g) -> list of 12 entries
    (9 mass-matrix entries row-major, then the 3 bias entries h = C*qd + G),
    byte-for-byte the same expressions exo_model uses, just not yet substituted.
    """
    N = exo.N_DOF
    t = sp.symbols("t", real=True)
    q = [sp.Function(f"q{i}")(t) for i in range(N)]
    qd = [sp.diff(qi, t) for qi in q]
    m = sp.symbols("m0:3", positive=True)
    L = sp.symbols("L0:3", positive=True)
    d = sp.symbols("d0:3", positive=True)
    Iz = sp.symbols("I0:3", positive=True)
    g = sp.symbols("g", positive=True)

    a = [sum(q[: i + 1]) for i in range(N)]
    joint_pos = [(sp.Integer(0), sp.Integer(0))]
    for i in range(N):
        px, py = joint_pos[i]
        joint_pos.append((px + L[i] * sp.cos(a[i]), py + L[i] * sp.sin(a[i])))
    com_pos = []
    for i in range(N):
        px, py = joint_pos[i]
        com_pos.append((px + d[i] * sp.cos(a[i]), py + d[i] * sp.sin(a[i])))

    T_kin = sp.Integer(0)
    for i in range(N):
        vx = sp.diff(com_pos[i][0], t)
        vy = sp.diff(com_pos[i][1], t)
        omega = sp.diff(a[i], t)
        T_kin += sp.Rational(1, 2) * m[i] * (vx**2 + vy**2)
        T_kin += sp.Rational(1, 2) * Iz[i] * omega**2
    V_pot = sum(m[i] * g * com_pos[i][1] for i in range(N))

    T_exp = sp.expand(sp.trigsimp(T_kin))
    M = sp.zeros(N, N)
    for i in range(N):
        for j in range(N):
            M[i, j] = sp.simplify(sp.diff(T_exp, qd[i], qd[j]))
    G = sp.zeros(N, 1)
    for i in range(N):
        G[i] = sp.simplify(sp.diff(V_pot, q[i]))
    C = sp.zeros(N, N)
    for i in range(N):
        for j in range(N):
            c_ij = sp.Integer(0)
            for k in range(N):
                christoffel = sp.Rational(1, 2) * (
                    sp.diff(M[i, j], q[k]) + sp.diff(M[i, k], q[j])
                    - sp.diff(M[j, k], q[i]))
                c_ij += christoffel * qd[k]
            C[i, j] = sp.simplify(c_ij)

    qs = sp.symbols("qs0:3", real=True)
    qds = sp.symbols("qds0:3", real=True)
    subs_map = {qd[i]: qds[i] for i in range(N)}
    subs_map.update({q[i]: qs[i] for i in range(N)})
    M = M.subs(subs_map); C = C.subs(subs_map); G = G.subs(subs_map)

    h = C * sp.Matrix(qds) + G
    terms = list(M) + list(h)                        # 12 scalar expressions
    return sp.lambdify((qs, qds, m, L, d, Iz, g), terms, modules="numpy")


def segment_params(M_body, H_body):
    """Rebuild absolute segment params from the SAME Winter fractions as exo_model."""
    seg = exo._SEGMENTS
    mass = np.array([exo._MASS_FRAC[s] * M_body for s in seg])
    length = np.array([exo._LEN_FRAC[s] * H_body for s in seg])
    com = np.array([exo._COM_FRAC[s] * length[i] for i, s in enumerate(seg)])
    inertia = np.array([mass[i] * (exo._RGYR_FRAC[s] * length[i]) ** 2
                        for i, s in enumerate(seg)])
    return mass, length, com, inertia


def make_terms_fun(terms_param, M_body, H_body):
    """A drop-in replacement for exo_model._TERMS_FUN specialized to a subject."""
    m, L, d, I = segment_params(M_body, H_body)
    g = exo.GRAV

    def _terms(qs, qds):
        return terms_param(qs, qds, m, L, d, I, g)
    return _terms


# =============================================================================
# 3. EVALUATE ONE (solution, subject) CLOSED-LOOP SIMULATION
# =============================================================================
def evaluate(x, terms_param, M_body, H_body):
    """Forward-simulate the fixed controller `x` on the subject (M_body, H_body).

    Swaps the specialized terms callable into exo_model, runs the UNMODIFIED
    forward_simulate + objective cost, and restores the original callable.
    """
    saved = exo._TERMS_FUN
    try:
        exo._TERMS_FUN = make_terms_fun(terms_param, M_body, H_body)
        exo._stage_data.cache_clear()                 # (state-indep; clear to be safe)
        sim = exo.forward_simulate(np.asarray(x, float))
        J = float(obj._costs_from_sim(sim)[0])
        rms = float(sim["rms_err"][0])
        diverged = bool(sim["diverged"][0])
    finally:
        exo._TERMS_FUN = saved
        exo._stage_data.cache_clear()
    feasible = (not diverged) and (rms <= RMS_THRESH)
    return J, rms, feasible, diverged


# =============================================================================
# 4. FIGURE: RMS vs body mass for the three stature levels (GWO solution only)
# =============================================================================
def plot_gwo(df):
    sub = df[df["algorithm"] == "GWO"]
    # colourblind-safe, per-stature colour + marker + linestyle
    styles = {
        H_GRID[0]: ("#0072B2", "o", "-"),    # -5 %  stature
        H_GRID[1]: ("#000000", "s", "--"),   # nominal
        H_GRID[2]: ("#D55E00", "^", "-."),   # +5 %  stature
    }
    labels = {H_GRID[0]: r"$H=1.663$ m ($-5\%$)",
              H_GRID[1]: r"$H=1.750$ m (nom.)",
              H_GRID[2]: r"$H=1.838$ m ($+5\%$)"}

    fig, ax = new_fig(3.5, 2.62)
    for H in H_GRID:
        d = sub[np.isclose(sub["H"], H)].sort_values("M")
        c, mk, ls = styles[H]
        ax.plot(d["M"], np.rad2deg(d["rms"]), color=c, ls=ls, marker=mk,
                markersize=4, label=labels[H])
    ax.axhline(np.rad2deg(RMS_THRESH), color="#CC0000", ls=(0, (4, 2)), lw=1.0)
    ax.text(M_GRID[0], np.rad2deg(RMS_THRESH) + 0.15,
            r"feasibility 0.15 rad ($8.6^\circ$)", color="#CC0000",
            fontsize=7, va="bottom")
    ax.set_xlabel("Body mass $M$ [kg]")
    ax.set_ylabel("RMS tracking error [deg]")
    ax.set_xticks(M_GRID)
    ax.set_xlim(M_GRID[0] - 1, M_GRID[-1] + 1)
    ax.legend(loc="upper left", handlelength=1.8, fontsize=7)
    return savefig_both(fig, "anthro_sensitivity")


# =============================================================================
# 5. SUMMARY REPORT
# =============================================================================
def write_summary(df, path="anthro_summary.txt"):
    L = []
    L.append("=" * 74)
    L.append("ANTHROPOMETRIC SENSITIVITY OF THE OPTIMIZED FEEDFORWARD CONTROLLERS")
    L.append("Fixed 27-coeff tau_ff + fixed PD, re-scored across body-mass/stature grid")
    L.append(f"Grid: M in {M_GRID} kg (+/-10%), H in {H_GRID} m (+/-5%)  "
             f"=> {len(M_GRID)*len(H_GRID)} subjects/solution")
    L.append(f"Cost J = int(sum tau^2) dt + {W_TRACK:g}*int(sum err^2) dt "
             f"+ continuous RMS penalty; feasible = RMS <= {RMS_THRESH} rad "
             f"and not diverged.")
    L.append("=" * 74)

    L.append(f"\n{'algorithm':<8} {'nominal J':>11} {'worst J':>11} "
             f"{'%degrad':>9} {'infeasible':>11} {'worst-case (M,H)':>20}")
    L.append("-" * 74)
    verdict_bits = {}
    for a in ALGO_ORDER:
        s = df[df["algorithm"] == a]
        nom = s[np.isclose(s["M"], M_NOM) & np.isclose(s["H"], H_NOM)]["J"].iloc[0]
        wi = s["J"].idxmax()
        worst = s.loc[wi, "J"]
        worstMH = (s.loc[wi, "M"], s.loc[wi, "H"])
        degr = 100.0 * (worst - nom) / nom
        n_infeas = int((~s["feasible"]).sum())
        L.append(f"{a:<8} {nom:>11.1f} {worst:>11.1f} {degr:>8.1f}% "
                 f"{n_infeas:>4d}/{len(s):<6d} "
                 f"{'(%.1f kg, %.3f m)' % worstMH:>20}")
        verdict_bits[a] = (nom, worst, degr, n_infeas, worstMH,
                           float(np.rad2deg(s["rms"].max())))

    # worst offender is heavy+tall or light+short? report the RMS envelope for GWO
    L.append("\nPer-solution RMS tracking-error envelope over the grid [deg]:")
    for a in ALGO_ORDER:
        s = df[df["algorithm"] == a]
        L.append(f"  {a:<6}: min {np.rad2deg(s['rms'].min()):.2f}, "
                 f"nominal {np.rad2deg(s[np.isclose(s['M'],M_NOM)&np.isclose(s['H'],H_NOM)]['rms'].iloc[0]):.2f}, "
                 f"max {np.rad2deg(s['rms'].max()):.2f}  "
                 f"(threshold {np.rad2deg(RMS_THRESH):.2f})")

    # --- plain-English verdict ------------------------------------------------
    L.append("\nVERDICT")
    total = len(M_GRID) * len(H_GRID)
    for a in ALGO_ORDER:
        nom, worst, degr, n_infeas, worstMH, rmsmax = verdict_bits[a]
        if n_infeas == 0:
            v = (f"  {a}: TRANSFERS across the +/-10% mass / +/-5% stature range -- "
                 f"all {total} subjects stay feasible (worst RMS {rmsmax:.2f} deg "
                 f"< {np.rad2deg(RMS_THRESH):.2f}); cost rises at most {degr:.1f}% "
                 f"(worst at M={worstMH[0]:.1f} kg, H={worstMH[1]:.3f} m). The "
                 f"optimized feedforward is robust to subject size, not overfit to "
                 f"the nominal body.")
        else:
            v = (f"  {a}: PARTIAL transfer -- {n_infeas}/{total} subjects become "
                 f"INFEASIBLE (RMS > {np.rad2deg(RMS_THRESH):.2f} deg), worst-case "
                 f"cost +{degr:.1f}% at M={worstMH[0]:.1f} kg, H={worstMH[1]:.3f} m; "
                 f"the controller is sensitive to anthropometry at the grid extremes "
                 f"and would need re-tuning / gain-scheduling for those subjects.")
        L.append(v)

    text = "\n".join(L) + "\n"
    with open(path, "w") as f:
        f.write(text)
    return text


# =============================================================================
# MAIN
# =============================================================================
def main():
    sols = recover_solutions()

    print("Deriving parameterized symbolic dynamics once (params kept symbolic)...")
    terms_param = build_param_terms()

    # --- SANITY CHECK: GWO-best on the NOMINAL subject must reproduce ~2410.4 --
    Jn, rmsn, feas_n, div_n = evaluate(sols["GWO"][0], terms_param, M_NOM, H_NOM)
    print(f"\n=== SANITY CHECK: GWO-best on nominal (M=70, H=1.75) ===")
    print(f"  J = {Jn:.4f}  (target ~{NOMINAL_GWO_J})   RMS = {rmsn:.5f} rad   "
          f"feasible = {feas_n}")
    if abs(Jn - NOMINAL_GWO_J) > 0.5:
        raise SystemExit(
            f"STOP: nominal GWO J = {Jn:.4f} does not reproduce {NOMINAL_GWO_J} "
            f"(diff {Jn - NOMINAL_GWO_J:+.4f}). The re-parameterized plant does "
            f"not match exo_model at nominal -- investigate before sweeping.")
    print("  OK: nominal reproduces the study's GWO best; proceeding with sweep.\n")

    # --- SWEEP: 3 solutions x 15 subjects = 45 evaluations --------------------
    rows = []
    for a in ALGO_ORDER:
        x = sols[a][0]
        for M in M_GRID:
            for H in H_GRID:
                J, rms, feasible, diverged = evaluate(x, terms_param, M, H)
                rows.append({"algorithm": a, "M": M, "H": H, "J": J,
                             "rms": rms, "feasible": feasible, "diverged": diverged})
    df = pd.DataFrame(rows)
    df.to_csv("anthro_sensitivity.csv", index=False)
    print(f"Saved: anthro_sensitivity.csv ({len(df)} rows)")

    summary = write_summary(df)
    print("Saved: anthro_summary.txt")

    pdf, png = plot_gwo(df)
    print(f"Saved: {pdf}, {png}\n")

    print(summary)


if __name__ == "__main__":
    main()
