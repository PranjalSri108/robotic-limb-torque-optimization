# Robotic-Limb Torque Optimization

This repository contains the code accompanying the paper. It compares three population
metaheuristics — Grey Wolf Optimizer (GWO), Particle Swarm Optimization (PSO), and Ant Colony
Optimization for continuous domains (ACOR) — on the problem of optimizing a feedforward torque
profile for a 3-DOF sagittal-plane robotic limb (hip, knee, ankle). The limb dynamics are derived
symbolically from Winter (2009) anthropometry and integrated with fixed-step RK4; each candidate
solution is a 27-coefficient truncated Fourier feedforward torque added to a fixed PD base
controller, and is evaluated by forward-dynamics simulation of gait tracking against a reference
trajectory, scored by a cost that trades actuation effort against tracking error under a
continuous RMS feasibility penalty. The three optimizers are implemented from scratch and compared
over 30 independent seeds with non-parametric statistical testing (Friedman, pairwise Wilcoxon
signed-rank with Holm–Bonferroni correction, rank-biserial effect sizes). The optimized
feedforward controllers are benchmarked against a mixed-sensitivity H-infinity feedback
controller, and a further study re-scores the optimized solutions across a grid of body masses and
statures to test whether they transfer beyond the nominal subject.

<img width="414" height="378" alt="Screenshot from 2026-07-15 22-37-09" src="https://github.com/user-attachments/assets/52029857-a9bb-441f-aaac-e23b9b088421" />
<img width="837" height="1444" alt="Screenshot from 2026-07-14 17-41-17" src="https://github.com/user-attachments/assets/d338ebf2-d0ca-4d06-ab26-5388d2bdbc13" />

## Requirements

Python 3.10 or newer.

```bash
pip install -r requirements.txt
```

Pinned dependencies: `numpy`, `scipy`, `sympy`, `matplotlib`, `pandas`, `control`, `slycot`.

`slycot` supplies the SLICOT bindings that `control.mixsyn` needs; without it the H-infinity
synthesis in `hinf_baseline.py` cannot run. On x86-64 Linux with CPython 3.10 it installs from a
prebuilt manylinux wheel with no system packages required. If no wheel matches your platform, pip
falls back to building from source, which needs a Fortran compiler and BLAS/LAPACK development
headers:

```bash
# Debian / Ubuntu
sudo apt install gfortran libblas-dev liblapack-dev cmake

# or avoid the source build entirely
conda install -c conda-forge slycot
```

Figures use a Times-metric-compatible serif font. `plotstyle.py` selects the best available of
Nimbus Roman / Liberation Serif / DejaVu Serif and prints its choice at import, so a silent
fallback is always visible. On Debian/Ubuntu, `sudo apt install fonts-liberation` provides a
Times-compatible face if none is installed.

Note for ROS users: sourcing `/opt/ros/*/setup.bash` places ROS site-packages on `PYTHONPATH`,
which takes precedence over a virtual environment and can shadow the pinned versions above. Run in
a shell without ROS sourced, or clear the variable for the run
(`env -u PYTHONPATH python3 compare.py`).

## Reproducing the results

Run from the repository root, in this order. Each script writes its outputs to the working
directory; figures go to `figures/`, which can be redirected with the `FIGDIR` environment
variable. Steps 3–5 must precede steps 6–7, which consume the CSV files those steps write.

Runtimes below are for a 16-core machine.

| # | Command | Runtime | Produces |
|---|---------|---------|----------|
| 1 | `python3 exo_model.py` | seconds | Self-test of the dynamics and closed-loop simulation. Console output only; optional. |
| 2 | `python3 objective.py` | seconds | PD-only baseline check and batch-vs-per-row consistency check. Console output only; optional. |
| 3 | `python3 compare.py` | ~15 min | `results_30runs.csv` (per-seed final cost, iterations to within 5 % of best, wall time, feasibility flag) and `stats_summary.txt` (metric table, Friedman test, pairwise Wilcoxon with Holm–Bonferroni correction, rank-biserial effect sizes, ranking conclusion) |
| 4 | `python3 pso_check.py` | ~5 min | `results_pso_check.csv` (per-seed costs for PSO, tuned PSO, GWO, ACOR) and `pso_tuning_check.txt` (whether tuning closes PSO's gap to GWO/ACOR) |
| 5 | `python3 hinf_baseline.py` | ~5 min | `hinf_results.txt` (operating point, mixed-sensitivity design metrics γ / ‖S‖∞ / ‖T‖∞ / controller order, and the closed-loop comparison against the GWO-optimized feedforward and the PD-only baseline) |
| 6 | `python3 make_figures.py` | ~10 min first run, seconds after | `figures/convergence_curves`, `figures/final_cost_boxplot`, `figures/tracking_error`, `figures/torque_profiles`, `figures/hinf_tracking` — each as `.pdf` (vector, for LaTeX) and `.png` (300 dpi). Also writes the cache `figures/plot_arrays.npz`. |
| 7 | `python3 anthro_sensitivity.py` | <1 min | `anthro_sensitivity.csv` (45 rows: 3 solutions × 15 subject variants, with cost, RMS error, feasibility), `anthro_summary.txt` (per-algorithm nominal and worst-case cost, percent degradation, infeasible-corner count, verdict), and `figures/anthro_sensitivity.{pdf,png}` |

Notes on reproducibility:

* All optimizer runs are seeded (`SEEDS` in `compare.py`) and deterministic, so repeated runs
  reproduce the reported numbers exactly.
* `make_figures.py` does not run new experiments. The original study persisted only per-seed final
  costs, not convergence histories or best-solution vectors, so on first invocation the script
  re-executes the same seeded optimizer calls to recover those arrays and asserts that the
  recovered final costs match `results_30runs.csv` to within 1e-9 before plotting. The recovered
  arrays are cached in `figures/plot_arrays.npz`, so subsequent runs are immediate.
* `anthro_sensitivity.py` is evaluation-only and invokes no optimizer, apart from re-deriving the
  single tuned-PSO best run whose solution vector was not cached, whose cost it verifies against
  `results_pso_check.csv`. It re-derives the dynamics once with the segment parameters kept
  symbolic, then specializes them numerically per subject, so the perturbation study runs against
  the unmodified simulation and cost code.
* BLAS threading is pinned to one thread per worker process in the parallel scripts so that
  reported per-seed wall times are not distorted by thread contention.
* The committed CSV, TXT, and figure files are the outputs of the runs above, so the reported
  numbers can be inspected without re-running anything.

## File overview

| File | Purpose |
|------|---------|
| `exo_model.py` | Limb model: Winter anthropometry, symbolic M(q), C(q,q̇), G(q) lambdified to NumPy, reference gait, PD gains, vectorized RK4 `forward_simulate()` |
| `objective.py` | Cost function: effort integral, weighted tracking integral, continuous RMS feasibility penalty, divergence guard; defines the 27 decision variables and their bounds |
| `pso.py` | Particle Swarm Optimization, implemented from scratch |
| `pso_tuned.py` | PSO with recommended tuning constants, otherwise identical to `pso.py`; control experiment |
| `gwo.py` | Grey Wolf Optimizer, implemented from scratch |
| `acor.py` | Ant Colony Optimization for continuous domains (ACOR), implemented from scratch |
| `compare.py` | Experiment runner: 30 seeds × 3 optimizers, statistical tests, summary report |
| `pso_check.py` | Tuned-PSO control experiment; reuses `results_30runs.csv` rather than re-running the other optimizers |
| `hinf_baseline.py` | Mixed-sensitivity H-infinity feedback baseline: linearization, `control.mixsyn` synthesis, ZOH discretization, nonlinear closed-loop simulation |
| `anthro_sensitivity.py` | Anthropometric sweep over ±10 % body mass and ±5 % stature, re-scoring the optimized solutions |
| `make_figures.py` | Regenerates every figure in the paper |
| `plotstyle.py` | Shared figure style: font selection, rcParams, colorblind-safe palette, save helpers |
| `simple_pendulum.py` | Standalone 1-DOF PD cost-landscape illustration; not part of the reproduction pipeline |
| `ACO.m`, `GWO.m`, `PSO.m` | Early MATLAB prototypes of the optimizers, superseded by the Python implementations; not used for any reported result |
| `results.csv` | Summary table from an earlier, superseded objective formulation; retained for provenance and not used in the paper |

## Citation

```bibtex
@article{srivastav2026torque,
  author  = {Srivastav, P. and Banjeet, V. and Dwivedy, S. K.},
  title   = {TODO},
  journal = {TODO},
  year    = {2026},
  note    = {Under review}
}
```

## Acknowledgment

Developed at the Mechatronics Laboratory, IIT Guwahati, under the supervision of
Prof. S. K. Dwivedy.

## License

Released under the MIT License. See [LICENSE](LICENSE).
