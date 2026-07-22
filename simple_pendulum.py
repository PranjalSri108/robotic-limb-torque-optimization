import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- Pendulum + cost ----------------
m, l, b, g = 1.0, 1.0, 0.2, 9.81
T, dt = 5.0, 0.01
N = int(T/dt)
th_ref = np.pi/4          # step reference from hanging position
lam = 0.001               # torque penalty weight

def cost(Kp, Kd):
    th, om = 0.0, 0.0
    J = 0.0
    for _ in range(N):
        # RK4 on [th, om]
        def f(th, om):
            e, ed = th_ref - th, -om
            tau = Kp*e + Kd*ed
            return om, (tau - b*om - m*g*l*np.sin(th))/(m*l*l), tau
        k1t, k1o, tau = f(th, om)
        k2t, k2o, _ = f(th+0.5*dt*k1t, om+0.5*dt*k1o)
        k3t, k3o, _ = f(th+0.5*dt*k2t, om+0.5*dt*k2o)
        k4t, k4o, _ = f(th+dt*k3t, om+dt*k3o)
        th += dt*(k1t+2*k2t+2*k3t+k4t)/6
        om += dt*(k1o+2*k2o+2*k3o+k4o)/6
        e = th_ref - th
        J += (e*e + lam*tau*tau)*dt
        if not np.isfinite(th) or abs(th) > 50:
            return 1e6
    return J

BOUNDS = np.array([[0.0, 20.0], [0.0, 10.0]])  # Kp, Kd
D = 2

def clip(x): return np.clip(x, BOUNDS[:,0], BOUNDS[:,1])
def evalpop(X): return np.array([cost(*x) for x in X])

# ---------------- PSO ----------------
def pso(seed, n=10, iters=40, w=0.7, c1=1.5, c2=1.5, trace=False):
    rng = np.random.default_rng(seed)
    X = rng.uniform(BOUNDS[:,0], BOUNDS[:,1], (n, D))
    V = np.zeros((n, D)); vmax = 0.2*(BOUNDS[:,1]-BOUNDS[:,0])
    F = evalpop(X); P, Pf = X.copy(), F.copy()
    gi = np.argmin(F); G, Gf = X[gi].copy(), F[gi]
    path = [G.copy()]; tr = []
    for t in range(iters):
        r1, r2 = rng.random((n,D)), rng.random((n,D))
        V = w*V + c1*r1*(P-X) + c2*r2*(G-X)
        V = np.clip(V, -vmax, vmax)
        X = clip(X + V)
        F = evalpop(X)
        if trace and t < 2:
            tr.append((t+1, X.copy(), V.copy(), F.copy(), G.copy(), Gf))
        imp = F < Pf; P[imp], Pf[imp] = X[imp], F[imp]
        gi = np.argmin(Pf)
        if Pf[gi] < Gf: G, Gf = P[gi].copy(), Pf[gi]
        path.append(G.copy())
    return G, Gf, np.array(path), tr

# ---------------- GWO ----------------
def gwo(seed, n=10, iters=40):
    rng = np.random.default_rng(seed)
    X = rng.uniform(BOUNDS[:,0], BOUNDS[:,1], (n, D))
    F = evalpop(X)
    order = np.argsort(F)
    A_, B_, De = X[order[0]].copy(), X[order[1]].copy(), X[order[2]].copy()
    fA = F[order[0]]
    path = [A_.copy()]
    for t in range(iters):
        a = 2 - 2*t/iters
        Xn = np.empty_like(X)
        for i in range(n):
            xs = []
            for L in (A_, B_, De):
                r1, r2 = rng.random(D), rng.random(D)
                Acoef, C = 2*a*r1 - a, 2*r2
                Dv = np.abs(C*L - X[i])
                xs.append(L - Acoef*Dv)
            Xn[i] = clip(np.mean(xs, axis=0))
        X = Xn; F = evalpop(X)
        order = np.argsort(F)
        if F[order[0]] < fA: A_, fA = X[order[0]].copy(), F[order[0]]
        # rebuild beta/delta from current population
        B_, De = X[order[1]].copy(), X[order[2]].copy()
        path.append(A_.copy())
    return A_, fA, np.array(path)

# ---------------- ACOR ----------------
def acor(seed, n=10, iters=40, k=10, q=0.5, xi=0.85):
    rng = np.random.default_rng(seed)
    A = rng.uniform(BOUNDS[:,0], BOUNDS[:,1], (k, D))
    Fa = evalpop(A)
    order = np.argsort(Fa); A, Fa = A[order], Fa[order]
    wgt = (1/(q*k*np.sqrt(2*np.pi)))*np.exp(-((np.arange(k))**2)/(2*(q*k)**2))
    prob = wgt/wgt.sum()
    path = [A[0].copy()]
    for t in range(iters):
        Xn = np.empty((n, D))
        for i in range(n):
            l_ = rng.choice(k, p=prob)
            for d in range(D):
                sig = xi*np.sum(np.abs(A[:,d]-A[l_,d]))/(k-1)
                Xn[i,d] = A[l_,d] + sig*rng.standard_normal()
        Xn = clip(Xn); Fn = evalpop(Xn)
        A = np.vstack([A, Xn]); Fa = np.concatenate([Fa, Fn])
        order = np.argsort(Fa)[:k]; A, Fa = A[order], Fa[order]
        path.append(A[0].copy())
    return A[0], Fa[0], np.array(path)

# ---------------- Run ----------------
gP, fP, pathP, trace = pso(0, trace=True)
gG, fG, pathG = gwo(0)
gA, fA2, pathA = acor(0)
print(f"PSO : Kp={gP[0]:.3f} Kd={gP[1]:.3f} J={fP:.5f}")
print(f"GWO : Kp={gG[0]:.3f} Kd={gG[1]:.3f} J={fG:.5f}")
print(f"ACOR: Kp={gA[0]:.3f} Kd={gA[1]:.3f} J={fA2:.5f}")

# worked PSO trace (first 3 particles, iterations 1-2)
print("\n--- PSO worked trace (3 particles) ---")
for (t, X, V, F, G, Gf) in trace:
    print(f"iter {t}: gbest=({G[0]:.2f},{G[1]:.2f}) Jg={Gf:.4f}")
    for i in range(3):
        print(f"  p{i+1}: x=({X[i,0]:6.3f},{X[i,1]:6.3f}) v=({V[i,0]:+.3f},{V[i,1]:+.3f}) J={F[i]:.4f}")

# initial state of those particles for the notes
rng = np.random.default_rng(0)
X0 = rng.uniform(BOUNDS[:,0], BOUNDS[:,1], (10, D))
F0 = evalpop(X0)
print("\ninit:", [(f"({X0[i,0]:.3f},{X0[i,1]:.3f}) J={F0[i]:.4f}") for i in range(3)])
print("init gbest:", X0[np.argmin(F0)], F0.min())

# ---------------- Landscape figure ----------------
kp = np.linspace(0.2, 20, 60); kd = np.linspace(0.05, 10, 55)
Z = np.array([[cost(a, c) for a in kp] for c in kd])
fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=200)
cs = ax.contourf(kp, kd, np.log10(Z), levels=30, cmap="cividis")
fig.colorbar(cs, label=r"$\log_{10} J(K_p, K_d)$")
for path, cl, name, mk in [(pathP,"#E74C3C","PSO","o"),(pathG,"#F1C40F","GWO","s"),(pathA,"#2ECC71","ACOR","^")]:
    ax.plot(path[:,0], path[:,1], color=cl, lw=1.8, marker=mk, ms=3.5, label=f"{name} best-so-far")
ax.plot(*pathG[-1], marker="*", ms=16, color="white", mec="black", zorder=5)
ax.set_xlabel(r"$K_p$"); ax.set_ylabel(r"$K_d$")
ax.set_title("Pendulum PD-tuning cost landscape and optimizer paths")
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout(); fig.savefig("pendulum_landscape.png", bbox_inches="tight")
print("\nfigure saved")

# response of best solution vs untuned, for a second figure
def simulate(Kp, Kd):
    th, om = 0.0, 0.0; TH=[]
    for _ in range(N):
        def f(th, om):
            e, ed = th_ref - th, -om
            tau = Kp*e + Kd*ed
            return om, (tau - b*om - m*g*l*np.sin(th))/(m*l*l)
        k1t,k1o=f(th,om); k2t,k2o=f(th+0.5*dt*k1t,om+0.5*dt*k1o)
        k3t,k3o=f(th+0.5*dt*k2t,om+0.5*dt*k2o); k4t,k4o=f(th+dt*k3t,om+dt*k3o)
        th += dt*(k1t+2*k2t+2*k3t+k4t)/6; om += dt*(k1o+2*k2o+2*k3o+k4o)/6
        TH.append(th)
    return np.array(TH)
t = np.arange(N)*dt
fig2, ax2 = plt.subplots(figsize=(7.2,3.4), dpi=200)
ax2.axhline(th_ref, color="k", ls="--", lw=1, label=r"$\theta_{ref}$")
ax2.plot(t, simulate(2,0.5), color="#95A5A6", lw=1.6, label="untuned (Kp=2, Kd=0.5)")
ax2.plot(t, simulate(*gG), color="#C8860D", lw=2.0, label=f"GWO-tuned (Kp={gG[0]:.1f}, Kd={gG[1]:.1f})")
ax2.set_xlabel("time [s]"); ax2.set_ylabel(r"$\theta$ [rad]")
ax2.set_title("Closed-loop step response: untuned vs. optimized gains")
ax2.legend(fontsize=9); fig2.tight_layout()
fig2.savefig("pendulum_response.png", bbox_inches="tight")
print("response figure saved")