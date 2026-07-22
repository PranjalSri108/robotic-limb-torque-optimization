"""
exo_model.py
============

Minimal 3-DOF sagittal-plane lower-limb exoskeleton model (hip, knee, ankle).

This module provides, for a 3-link planar serial chain:

    1. Winter (2009) anthropometry for a 70 kg / 1.75 m subject.
    2. Symbolically-derived rigid-body dynamics  M(q), C(q, q_dot), G(q)
       (via SymPy), lambdified to fast NumPy callables.
    3. A parametric reference gait trajectory  q_ref(t)  per joint, built from a
       base offset plus 3 sinusoidal harmonics. The trajectory shape parameters
       are the decision variables of the later optimization study.
    4. Inverse dynamics:  trajectory params -> q(t), q_dot(t), q_ddot(t)
       (by *analytic* differentiation) -> joint torques tau(t).

This is step 1 of 3: physics + trajectory + inverse dynamics only. No optimizer
lives here (see objective.py for the scalar cost J that an optimizer will call).

Coordinate / sign convention
----------------------------
The leg is modelled as a planar serial chain rooted at the hip. Generalized
coordinates q = [q_hip, q_knee, q_ankle] are *relative* joint angles (each link's
angle measured relative to the previous link), which is the standard manipulator
convention. This keeps the dynamics derivation clean; the numeric joint-limit /
neutral values below are chosen to be physically reasonable in this convention
rather than to match any specific clinical gait-lab sign definition.

References
----------
Winter, D. A. (2009). *Biomechanics and Motor Control of Human Movement*,
4th ed. Wiley. (Segment mass, length, CoM and radius-of-gyration tables.)
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import sympy as sp

# =============================================================================
# 1. ANTHROPOMETRY  --  Winter (2009), 70 kg / 1.75 m subject
# =============================================================================
# Winter's tables express each quantity as a fraction of a whole-body measure:
#   * segment mass          = fraction of total body mass M_body
#   * segment length        = fraction of total body height H_body
#   * CoM location          = fraction of segment length, measured from proximal
#   * radius of gyration    = fraction of segment length, about the segment CoM
#
# Values below are taken from Winter (2009), Table 4.1 ("Anthropometric Data").
# -----------------------------------------------------------------------------
M_BODY = 70.0    # [kg]  total body mass
H_BODY = 1.75    # [m]   total body height
GRAV = 9.81      # [m/s^2] gravitational acceleration

# --- mass fraction of total body mass (Winter 2009, Table 4.1) ---------------
_MASS_FRAC = {"thigh": 0.1000, "shank": 0.0465, "foot": 0.0145}
# --- segment length as fraction of body height -------------------------------
_LEN_FRAC = {"thigh": 0.245, "shank": 0.246, "foot": 0.152}
# --- CoM location as fraction of segment length, from the PROXIMAL joint ------
_COM_FRAC = {"thigh": 0.433, "shank": 0.433, "foot": 0.500}
# --- radius of gyration as fraction of segment length, about the segment CoM --
_RGYR_FRAC = {"thigh": 0.323, "shank": 0.302, "foot": 0.475}

# Absolute per-segment parameters, ordered proximal -> distal: [thigh, shank, foot]
_SEGMENTS = ("thigh", "shank", "foot")

LINK_MASS = np.array([_MASS_FRAC[s] * M_BODY for s in _SEGMENTS])      # m_i  [kg]
LINK_LEN = np.array([_LEN_FRAC[s] * H_BODY for s in _SEGMENTS])        # L_i  [m]
LINK_COM = np.array([_COM_FRAC[s] * L for s, L in zip(_SEGMENTS, LINK_LEN)])  # d_i [m]
# Inertia about segment CoM:  I_i = m_i * (rgyr_frac_i * L_i)^2
LINK_INERTIA = np.array(
    [LINK_MASS[i] * (_RGYR_FRAC[s] * LINK_LEN[i]) ** 2 for i, s in enumerate(_SEGMENTS)]
)  # I_i [kg m^2]

N_DOF = 3  # hip, knee, ankle

# -----------------------------------------------------------------------------
# Joint ranges of motion (relative-angle convention, radians).
# Used by objective.py for range-of-motion penalties and as the trajectory
# "neutral" reference for the boundary-condition penalty.
# -----------------------------------------------------------------------------
JOINT_NEUTRAL = np.array([0.0, 0.35, 0.0])                 # comfortable standing-ish pose [rad]
JOINT_MIN = np.deg2rad(np.array([-20.0, 0.0, -30.0]))      # hip, knee, ankle lower limit
JOINT_MAX = np.deg2rad(np.array([120.0, 140.0, 20.0]))     # hip, knee, ankle upper limit


# =============================================================================
# 2. SYMBOLIC DYNAMICS  ->  lambdified M(q), C(q, q_dot), G(q)
# =============================================================================
def _derive_dynamics():
    """Derive the 3-link planar-chain dynamics symbolically (Lagrangian method).

    Returns three NumPy-callable functions:
        M_fun(q)              -> (3, 3) inertia matrix
        C_fun(q, q_dot)       -> (3, 3) Coriolis/centrifugal matrix
        G_fun(q)              -> (3,)   gravity torque vector

    Equation of motion:  M(q) q_ddot + C(q, q_dot) q_dot + G(q) = tau
    """
    t = sp.symbols("t", real=True)

    # Relative joint angles as functions of time (needed for velocities).
    q = [sp.Function(f"q{i}")(t) for i in range(N_DOF)]
    qd = [sp.diff(qi, t) for qi in q]

    # Constant symbols for masses, lengths, CoM offsets, inertias, gravity.
    m = sp.symbols("m0:3", positive=True)
    L = sp.symbols("L0:3", positive=True)
    d = sp.symbols("d0:3", positive=True)
    Iz = sp.symbols("I0:3", positive=True)
    g = sp.symbols("g", positive=True)

    # Absolute link angles: a_i = q0 + q1 + ... + qi  (chain of relative angles).
    a = [sum(q[: i + 1]) for i in range(N_DOF)]

    # Forward kinematics of each link CoM (planar chain rooted at hip origin).
    # Joint i+1 sits at the distal end of link i.
    joint_pos = [(sp.Integer(0), sp.Integer(0))]  # base (hip) at origin
    for i in range(N_DOF):
        px, py = joint_pos[i]
        joint_pos.append((px + L[i] * sp.cos(a[i]), py + L[i] * sp.sin(a[i])))

    com_pos = []
    for i in range(N_DOF):
        px, py = joint_pos[i]
        com_pos.append((px + d[i] * sp.cos(a[i]), py + d[i] * sp.sin(a[i])))

    # Kinetic energy: translational (CoM) + rotational (about CoM).
    T_kin = sp.Integer(0)
    for i in range(N_DOF):
        vx = sp.diff(com_pos[i][0], t)
        vy = sp.diff(com_pos[i][1], t)
        omega = sp.diff(a[i], t)
        T_kin += sp.Rational(1, 2) * m[i] * (vx**2 + vy**2)
        T_kin += sp.Rational(1, 2) * Iz[i] * omega**2

    # Potential energy: gravity acts in -y; V = sum m_i g y_com_i.
    V_pot = sum(m[i] * g * com_pos[i][1] for i in range(N_DOF))

    # --- Inertia matrix:  M_ij = d^2 T / (d qd_i d qd_j) -------------------
    T_exp = sp.expand(sp.trigsimp(T_kin))
    M = sp.zeros(N_DOF, N_DOF)
    for i in range(N_DOF):
        for j in range(N_DOF):
            M[i, j] = sp.simplify(sp.diff(T_exp, qd[i], qd[j]))

    # --- Gravity vector:  G_i = d V / d q_i --------------------------------
    G = sp.zeros(N_DOF, 1)
    for i in range(N_DOF):
        G[i] = sp.simplify(sp.diff(V_pot, q[i]))

    # --- Coriolis/centrifugal matrix via Christoffel symbols ---------------
    #   C_ij = sum_k 1/2 (dM_ij/dq_k + dM_ik/dq_j - dM_jk/dq_i) * qd_k
    C = sp.zeros(N_DOF, N_DOF)
    for i in range(N_DOF):
        for j in range(N_DOF):
            c_ij = sp.Integer(0)
            for k in range(N_DOF):
                christoffel = sp.Rational(1, 2) * (
                    sp.diff(M[i, j], q[k])
                    + sp.diff(M[i, k], q[j])
                    - sp.diff(M[j, k], q[i])
                )
                c_ij += christoffel * qd[k]
            C[i, j] = sp.simplify(c_ij)

    # --- Substitute plain scalar symbols for the time-dependent functions --
    qs = sp.symbols("qs0:3", real=True)      # positions
    qds = sp.symbols("qds0:3", real=True)    # velocities
    subs_map = {}
    for i in range(N_DOF):
        subs_map[qd[i]] = qds[i]  # substitute velocities first (they contain q)
    for i in range(N_DOF):
        subs_map[q[i]] = qs[i]

    M = M.subs(subs_map)
    C = C.subs(subs_map)
    G = G.subs(subs_map)

    # Numeric parameter substitution (mass/length/inertia/gravity -> numbers).
    param_subs = {g: GRAV}
    for i in range(N_DOF):
        param_subs[m[i]] = float(LINK_MASS[i])
        param_subs[L[i]] = float(LINK_LEN[i])
        param_subs[d[i]] = float(LINK_COM[i])
        param_subs[Iz[i]] = float(LINK_INERTIA[i])

    M = M.subs(param_subs)
    C = C.subs(param_subs)
    G = G.subs(param_subs)

    # --- Lambdify ENTRY-WISE to NumPy callables ----------------------------
    # Lambdifying each scalar matrix entry (rather than the whole matrix) lets
    # us broadcast over a batch/time axis: calling an entry with (3, N) inputs
    # returns either a scalar (constant entries) or an (N,) array, which we then
    # broadcast and assemble. This turns the per-time-sample Python loop in the
    # inverse dynamics into a handful of vectorized NumPy ops (major speed-up),
    # while the numeric result is identical to evaluating sample-by-sample.
    M_ent = [[sp.lambdify((qs,), M[i, j], modules="numpy") for j in range(N_DOF)]
             for i in range(N_DOF)]
    C_ent = [[sp.lambdify((qs, qds), C[i, j], modules="numpy") for j in range(N_DOF)]
             for i in range(N_DOF)]
    G_ent = [sp.lambdify((qs,), G[i], modules="numpy") for i in range(N_DOF)]

    # Combined "terms" function for the forward-dynamics inner loop: returns the
    # 9 entries of M (row-major) followed by the 3-vector bias h = C*q_dot + G,
    # all from a SINGLE lambdified call. The forward simulation then only needs
    #   q_ddot = M^-1 (tau - h),
    # which is far fewer Python-level calls than evaluating M, C, G separately.
    h = C * sp.Matrix(qds) + G                 # (3, 1) bias torque
    terms = list(M) + list(h)                  # 12 scalar expressions
    terms_fun = sp.lambdify((qs, qds), terms, modules="numpy")

    return M_ent, C_ent, G_ent, terms_fun


# Derive once at import time (symbolic derivation is the expensive part; the
# resulting lambdified callables are fast and reused for every objective call).
_M_ENT, _C_ENT, _G_ENT, _TERMS_FUN = _derive_dynamics()


def _dyn_terms(Q, Qd):
    """Batched M and bias h = C q_dot + G for a population of states.

    Q, Qd : (n, 3). Returns M (n, 3, 3) and h (n, 3), via one lambdified call.
    """
    n = Q.shape[0]
    vals = _TERMS_FUN(Q.T, Qd.T)               # list of 12 (scalars or (n,) arrays)
    # Column assignment broadcasts scalar (constant) entries and copies array
    # entries without the per-entry broadcast_to/stack overhead of the profiler.
    stacked = np.empty((n, 12))
    for idx, v in enumerate(vals):
        stacked[:, idx] = v
    M = stacked[:, :9].reshape(n, N_DOF, N_DOF)
    h = stacked[:, 9:]
    return M, h


def mass_matrix(q: np.ndarray) -> np.ndarray:
    """Inertia matrix M(q).

    Accepts a single pose q of shape (3,) -> returns (3, 3), or a batch of
    poses of shape (3, N) -> returns (3, 3, N).
    """
    q = np.asarray(q, dtype=float)
    shape = () if q.ndim == 1 else (q.shape[1],)
    out = np.empty((N_DOF, N_DOF) + shape, dtype=float)
    for i in range(N_DOF):
        for j in range(N_DOF):
            out[i, j] = np.broadcast_to(_M_ENT[i][j](q), shape)
    return out


def coriolis_matrix(q: np.ndarray, q_dot: np.ndarray) -> np.ndarray:
    """Coriolis/centrifugal matrix C(q, q_dot).

    Single pose (3,) -> (3, 3); batch (3, N) -> (3, 3, N).
    """
    q = np.asarray(q, dtype=float)
    q_dot = np.asarray(q_dot, dtype=float)
    shape = () if q.ndim == 1 else (q.shape[1],)
    out = np.empty((N_DOF, N_DOF) + shape, dtype=float)
    for i in range(N_DOF):
        for j in range(N_DOF):
            out[i, j] = np.broadcast_to(_C_ENT[i][j](q, q_dot), shape)
    return out


def gravity_vector(q: np.ndarray) -> np.ndarray:
    """Gravity torque vector G(q).

    Single pose (3,) -> (3,); batch (3, N) -> (3, N).
    """
    q = np.asarray(q, dtype=float)
    shape = () if q.ndim == 1 else (q.shape[1],)
    out = np.empty((N_DOF,) + shape, dtype=float)
    for i in range(N_DOF):
        out[i] = np.broadcast_to(_G_ENT[i](q), shape)
    return out


# =============================================================================
# 3. PARAMETRIC REFERENCE GAIT TRAJECTORY
# =============================================================================
# One gait cycle has period T. Each joint j follows:
#
#     q_j(t) = offset_j + sum_{k=1..H} A_{j,k} * sin(2*pi*k*t/T + phi_{j,k})
#
# with H = 3 harmonics. The SHAPE of this trajectory (offset, amplitudes,
# phases) is what the optimizer will tune, so those are the decision variables.
#
# Per joint the decision variables are:
#     offset            (1)
#     A_1, A_2, A_3     (3 harmonic amplitudes)
#     phi_1, phi_2, phi_3 (3 harmonic phases)
#   => 7 parameters / joint  x  3 joints  =  21 decision variables total.
#
# Analytic derivatives give q_dot and q_ddot exactly (no finite differencing).
# =============================================================================
GAIT_PERIOD = 1.2          # [s]  T, one full gait cycle
N_HARMONICS = 3            # sinusoidal harmonics per joint
PARAMS_PER_JOINT = 1 + 2 * N_HARMONICS   # offset + amplitudes + phases = 7
N_PARAMS = N_DOF * PARAMS_PER_JOINT      # 21 decision variables


def unpack_params(params: np.ndarray):
    """Split the flat (21,) decision vector into per-joint (offset, A, phi).

    Layout is joint-major:
        [j0: off, A1, A2, A3, phi1, phi2, phi3,
         j1: off, A1, A2, A3, phi1, phi2, phi3,
         j2: off, A1, A2, A3, phi1, phi2, phi3]

    Returns (offset, amp, phase) with shapes (3,), (3, H), (3, H).
    """
    p = np.asarray(params, dtype=float).reshape(N_DOF, PARAMS_PER_JOINT)
    offset = p[:, 0]
    amp = p[:, 1 : 1 + N_HARMONICS]
    phase = p[:, 1 + N_HARMONICS :]
    return offset, amp, phase


def trajectory(params: np.ndarray, t: np.ndarray):
    """Evaluate q, q_dot, q_ddot for all joints at time(s) t.

    Parameters
    ----------
    params : (21,) decision vector.
    t      : scalar or (N,) array of times [s].

    Returns
    -------
    q, q_dot, q_ddot : each (N_DOF, N) arrays (N = number of time samples).
    """
    offset, amp, phase = unpack_params(params)
    t = np.atleast_1d(np.asarray(t, dtype=float))          # (N,)

    k = np.arange(1, N_HARMONICS + 1)                       # (H,)
    w = 2.0 * np.pi * k / GAIT_PERIOD                        # (H,) angular freq per harmonic

    # Broadcast to (N_DOF, H, N): joints x harmonics x time.
    phase_arg = w[None, :, None] * t[None, None, :] + phase[:, :, None]
    A = amp[:, :, None]
    W = w[None, :, None]

    q = offset[:, None] + np.sum(A * np.sin(phase_arg), axis=1)
    q_dot = np.sum(A * W * np.cos(phase_arg), axis=1)
    q_ddot = np.sum(-A * W**2 * np.sin(phase_arg), axis=1)

    return q, q_dot, q_ddot


# =============================================================================
# 4. INVERSE DYNAMICS
# =============================================================================
def inverse_dynamics(params: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Joint torques required to track the reference trajectory.

    Computes q, q_dot, q_ddot analytically from `params`, then applies the
    inverse-dynamics equation joint-wise:

        tau = M(q) q_ddot + C(q, q_dot) q_dot + G(q)

    Parameters
    ----------
    params : (21,) decision vector.
    t      : (N,) array of time samples over one gait cycle.

    Returns
    -------
    tau : (N_DOF, N) required torque at each joint and time sample [N m].
    """
    q, q_dot, q_ddot = trajectory(params, t)   # each (N_DOF, N)

    # Batched dynamics: M (3,3,N), C (3,3,N), G (3,N). The einsums contract the
    # joint index while keeping the time axis, i.e. per time sample n:
    #   tau[:, n] = M[:,:,n] @ q_ddot[:,n] + C[:,:,n] @ q_dot[:,n] + G[:,n]
    M = mass_matrix(q)
    C = coriolis_matrix(q, q_dot)
    G = gravity_vector(q)

    tau = (np.einsum("ijn,jn->in", M, q_ddot)
           + np.einsum("ijn,jn->in", C, q_dot)
           + G)
    return tau


# =============================================================================
# 5. CLOSED-LOOP FORWARD DYNAMICS  (PD base controller + feedforward torque)
# =============================================================================
# We now switch from inverse-dynamics effort minimization to *forward-dynamics
# tracking*. A fixed reference gait q_ref(t) is tracked by
#
#     tau(t) = tau_ff(t) + Kp (q_ref - q) + Kd (q_dot_ref - q_dot)
#
# and the closed loop is forward-simulated:  q_ddot = M^-1 (tau - C q_dot - G).
# The decision variables are now the FEEDFORWARD torque tau_ff(t) (see below);
# the PD gains are FIXED and hand-tuned for closed-loop stability.
# -----------------------------------------------------------------------------

# --- FIXED reference gait -----------------------------------------------------
# The reference trajectory q_ref(t) reuses the existing 3-harmonic parametric
# form, but its parameters are now CONSTANT (they are no longer optimized).
# Chosen to be a smooth, physiologically-plausible sagittal pattern that stays
# well inside every joint's range of motion. Layout per joint:
#   [offset, A1, A2, A3, phi1, phi2, phi3]  (see unpack_params / trajectory).
REF_GAIT_PARAMS = np.array([
    # hip:   swings about a slightly-flexed offset
    0.30, 0.40, 0.08, 0.03,  0.00, 0.50, -0.30,
    # knee:  larger flexion, phase-shifted from the hip
    0.50, 0.40, 0.10, 0.04, -1.20, 0.40,  0.20,
    # ankle: small dorsi/plantar-flexion oscillation
    0.00, 0.15, 0.05, 0.02,  1.00, -0.60, 0.30,
], dtype=float)


def reference(t):
    """Reference joint trajectory q_ref(t), q_dot_ref(t).

    t : scalar or (N,) times. Returns (q_ref, q_dot_ref), each (N_DOF,) for a
    scalar t or (N_DOF, N) for an array t.
    """
    q, qd, _ = trajectory(REF_GAIT_PARAMS, t)
    if np.isscalar(t) or np.ndim(t) == 0:
        return q[:, 0], qd[:, 0]
    return q, qd


# --- FIXED PD base-controller gains ------------------------------------------
# Hand-tuned per joint for a stable closed loop. Rationale (using the neutral-
# pose effective inertias M_ii ~ [2.62, 0.51, 0.034] kg m^2):
#   * hip  : wn = sqrt(Kp/I) ~ 10.7 rad/s, damping ratio ~0.5 (well damped).
#   * knee : ~critically damped (Kd ~ 2 sqrt(Kp I)).
#   * ankle: the light distal link is the fast mode; it is deliberately
#            OVER-damped (large Kd relative to its critical value ~2.9) so the
#            fast oscillation the caution warns about is suppressed.
KP = np.array([300.0, 200.0, 60.0])   # proportional gains [N m / rad]
KD = np.array([30.0, 20.0, 6.0])      # derivative gains  [N m s / rad]

# --- FEEDFORWARD torque parameterization (the DECISION VARIABLES) ------------
# Each joint's tau_ff(t) is a truncated Fourier series over the gait period:
#
#   tau_ff_j(t) = a0_j + sum_{k=1..4} [ a_{j,k} cos(k w t) + b_{j,k} sin(k w t) ]
#
# with w = 2*pi/T. Coefficients per joint: 1 offset + 4 cos + 4 sin = 9.
#   => 9 x 3 joints = 27 decision variables total.
FF_HARMONICS = 4
FF_PER_JOINT = 1 + 2 * FF_HARMONICS      # offset + cos + sin = 9
N_FF_PARAMS = N_DOF * FF_PER_JOINT       # 27 decision variables


def _ff_basis(t):
    """Fourier basis [1, cos(w t)..cos(4w t), sin(w t)..sin(4w t)] at scalar t.

    Returns a (FF_PER_JOINT,) vector.
    """
    w = 2.0 * np.pi / GAIT_PERIOD
    k = np.arange(1, FF_HARMONICS + 1)
    return np.concatenate(([1.0], np.cos(k * w * t), np.sin(k * w * t)))


def feedforward_torque(params, t):
    """Feedforward torque tau_ff(t) for a batch of parameter vectors.

    params : (n, 27) batch (or (27,) single) of Fourier coefficients.
    t      : scalar time.
    Returns (n, 3) torque (or (3,) for a single param vector).
    """
    P = np.atleast_2d(np.asarray(params, dtype=float))       # (n, 27)
    n = P.shape[0]
    coeffs = P.reshape(n, N_DOF, FF_PER_JOINT)               # (n, 3, 9)
    basis = _ff_basis(t)                                     # (9,)
    tau_ff = coeffs @ basis                                  # (n, 3)
    return tau_ff[0] if np.ndim(params) == 1 else tau_ff


@lru_cache(maxsize=8)
def _stage_data(n_steps):
    """Precompute the state-INDEPENDENT quantities the RK4 sweep needs.

    The reference trajectory q_ref(t), q_dot_ref(t) and the feedforward Fourier
    basis depend only on time, not on the decision variables or the state, so
    they are identical for every objective evaluation. We compute them once per
    n_steps (cached) at the RK4 node times (i*dt) and midpoint times
    ((i+0.5)*dt), eliminating ~1500 trigonometric trajectory/basis rebuilds per
    objective call.

    Returns dt and two tuples (nodes, mids), each = (q_ref, qd_ref, basis) with
    shapes ((S+1 or S, 3), (.,3), (.,9)).
    """
    T = GAIT_PERIOD
    dt = T / n_steps
    w = 2.0 * np.pi / T
    k = np.arange(1, FF_HARMONICS + 1)

    def _build(ts):
        q, qd, _ = trajectory(REF_GAIT_PARAMS, ts)           # (3, len)
        basis = np.column_stack(
            [np.ones_like(ts)]
            + [np.cos(kk * w * ts) for kk in k]
            + [np.sin(kk * w * ts) for kk in k]
        )                                                    # (len, 9)
        return q.T.copy(), qd.T.copy(), basis

    node_t = np.arange(n_steps + 1) * dt
    mid_t = (np.arange(n_steps) + 0.5) * dt
    return dt, _build(node_t), _build(mid_t)


def _control(coeffs, q_ref, qd_ref, basis, Q, Qd):
    """Commanded torque tau = tau_ff + PD, using precomputed stage data.

    coeffs : (n,3,9).  q_ref, qd_ref : (3,).  basis : (9,).  Q, Qd : (n,3).
    """
    tau_ff = coeffs @ basis                                  # (n, 3)
    return tau_ff + KP * (q_ref - Q) + KD * (qd_ref - Qd)


def _accel(coeffs, q_ref, qd_ref, basis, Q, Qd):
    """Closed-loop acceleration q_ddot (and the torque tau) for the population.

    Solves M(q) q_ddot = tau - (C q_dot + G); M is SPD so this is well-posed
    unless a state has diverged (NaN) -> solve yields NaN, caught downstream.
    """
    n = Q.shape[0]
    tau = _control(coeffs, q_ref, qd_ref, basis, Q, Qd)      # (n, 3)
    M, h = _dyn_terms(Q, Qd)                                 # (n,3,3), (n,3)
    rhs = (tau - h)[..., None]                               # (n, 3, 1)
    try:
        qddot = np.linalg.solve(M, rhs)[..., 0]              # (n, 3)
    except np.linalg.LinAlgError:
        qddot = np.full((n, N_DOF), np.nan)
    return qddot, tau


# Fixed-step RK4 resolution. The light ankle link is a fast mode: verified that
# 150 steps (dt = 8 ms) blows up numerically (RMS ~2.4 rad, unbounded torque),
# while 300 steps (dt = 4 ms) is stable and agrees with 600 steps to <1e-3 rad.
# So we use 300.
FWD_N_STEPS = 300
_Q_DIVERGE = 50.0          # |q| beyond this (rad) is treated as blow-up


def forward_simulate(params, n_steps=FWD_N_STEPS, return_history=False):
    """Forward-simulate the PD+feedforward closed loop over one gait cycle.

    Fixed-step RK4 integration of the 6-state system s = [q, q_dot] with
    q_ddot = M^-1 (tau - C q_dot - G). Vectorized over a population of
    parameter vectors for a large speed-up.

    Parameters
    ----------
    params        : (n, 27) batch or (27,) single feedforward-coefficient vector.
    n_steps       : number of fixed RK4 steps over [0, T].
    return_history: if True also return the full state/torque history.

    Returns
    -------
    dict with:
        integral_tau2 : (n,) time-integral of sum-of-squared torque per member,
        integral_err2 : (n,) time-integral of sum-of-squared tracking error,
        rms_err       : (n,) RMS tracking error (over time and joints) [rad],
        diverged      : (n,) bool, True if the member blew up (NaN or |q| large).
    plus, if return_history: t (S+1,), q (S+1,n,3), q_ref (S+1,3), tau (S+1,n,3).
    """
    single = np.ndim(params) == 1
    P = np.atleast_2d(np.asarray(params, dtype=float))       # (n, 27)
    n = P.shape[0]
    coeffs = P.reshape(n, N_DOF, FF_PER_JOINT)               # (n, 3, 9)

    T = GAIT_PERIOD
    dt, (nq, nqd, nB), (mq, mqd, mB) = _stage_data(n_steps)  # cached stage data

    # initial condition: start ON the reference (q(0)=q_ref(0), qd(0)=qd_ref(0))
    Q = np.tile(nq[0], (n, 1))                               # (n, 3)
    Qd = np.tile(nqd[0], (n, 1))                             # (n, 3)

    diverged = np.zeros(n, dtype=bool)

    # trapezoidal accumulators over the integration nodes
    integral_tau2 = np.zeros(n)
    integral_err2 = np.zeros(n)
    sum_err2_nodes = np.zeros(n)     # for RMS: sum over nodes of mean-joint err^2

    if return_history:
        q_hist = np.empty((n_steps + 1, n, N_DOF))
        tau_hist = np.empty((n_steps + 1, n, N_DOF))
        qref_hist = np.empty((n_steps + 1, N_DOF))

    def _node_terms(idx, Q, Qd):
        """sum-sq torque and sum-sq tracking error at node `idx` (per member).

        The commanded torque needs only the controller (feedforward + PD), not
        the rigid-body dynamics, so this is cheap and adds no matrix solves.
        """
        q_ref = nq[idx]
        tau = _control(coeffs, q_ref, nqd[idx], nB[idx], Q, Qd)
        err = Q - q_ref                                      # (n, 3)
        return np.sum(tau**2, axis=1), np.sum(err**2, axis=1), tau, q_ref

    # evaluate node 0
    s_tau0, s_err0, tau0, qref0 = _node_terms(0, Q, Qd)
    prev_tau2, prev_err2 = s_tau0, s_err0
    sum_err2_nodes += s_err0 / N_DOF
    if return_history:
        q_hist[0], tau_hist[0], qref_hist[0] = Q, tau0, qref0

    for i in range(n_steps):
        # --- classic RK4 on s = [Q, Qd]; leaders: node i, midpoint i, node i+1
        k1_q, (k1_qd, _) = Qd, _accel(coeffs, nq[i], nqd[i], nB[i], Q, Qd)
        k2_q = Qd + 0.5 * dt * k1_qd
        k2_qd, _ = _accel(coeffs, mq[i], mqd[i], mB[i], Q + 0.5 * dt * k1_q, k2_q)
        k3_q = Qd + 0.5 * dt * k2_qd
        k3_qd, _ = _accel(coeffs, mq[i], mqd[i], mB[i], Q + 0.5 * dt * k2_q, k3_q)
        k4_q = Qd + dt * k3_qd
        k4_qd, _ = _accel(coeffs, nq[i + 1], nqd[i + 1], nB[i + 1], Q + dt * k3_q, k4_q)

        Q = Q + (dt / 6.0) * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)
        Qd = Qd + (dt / 6.0) * (k1_qd + 2 * k2_qd + 2 * k3_qd + k4_qd)

        # --- divergence detection (NaN/inf or exploding angle) --------------
        bad = ~np.isfinite(Q).all(axis=1) | ~np.isfinite(Qd).all(axis=1) \
            | (np.abs(Q).max(axis=1) > _Q_DIVERGE)
        diverged |= bad
        # freeze diverged members on the reference so the batched solve stays
        # well-conditioned for everyone else (their cost is overwritten later).
        if diverged.any():
            Q[diverged] = nq[i + 1]
            Qd[diverged] = nqd[i + 1]

        # --- accumulate cost integrals (trapezoidal) ------------------------
        s_tau2, s_err2, tau_n, qref_n = _node_terms(i + 1, Q, Qd)
        integral_tau2 += 0.5 * (prev_tau2 + s_tau2) * dt
        integral_err2 += 0.5 * (prev_err2 + s_err2) * dt
        prev_tau2, prev_err2 = s_tau2, s_err2
        sum_err2_nodes += s_err2 / N_DOF

        if return_history:
            q_hist[i + 1], tau_hist[i + 1], qref_hist[i + 1] = Q, tau_n, qref_n

    rms_err = np.sqrt(sum_err2_nodes / (n_steps + 1))         # (n,)

    out = {
        "integral_tau2": integral_tau2 if not single else integral_tau2,
        "integral_err2": integral_err2,
        "rms_err": rms_err,
        "diverged": diverged,
    }
    if return_history:
        out["t"] = np.linspace(0.0, T, n_steps + 1)
        out["q"] = q_hist
        out["q_ref"] = qref_hist
        out["tau"] = tau_hist
    return out


if __name__ == "__main__":
    # Quick self-check: print anthropometry and evaluate dynamics at neutral.
    np.set_printoptions(precision=4, suppress=True)
    print("=== Winter (2009) anthropometry (70 kg / 1.75 m) ===")
    for i, s in enumerate(_SEGMENTS):
        print(
            f"  {s:6s}: mass={LINK_MASS[i]:6.3f} kg  len={LINK_LEN[i]:6.3f} m  "
            f"com={LINK_COM[i]:6.3f} m  I={LINK_INERTIA[i]:7.4f} kg m^2"
        )

    q0 = JOINT_NEUTRAL.copy()
    qd0 = np.zeros(N_DOF)
    print("\n=== Dynamics at neutral pose ===")
    print("M(q) =\n", mass_matrix(q0))
    print("C(q, q_dot) =\n", coriolis_matrix(q0, qd0))
    print("G(q) =", gravity_vector(q0))

    print(f"\nDecision variables: {N_PARAMS} "
          f"({PARAMS_PER_JOINT} per joint x {N_DOF} joints)")
