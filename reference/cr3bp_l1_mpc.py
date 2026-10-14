"""
Planar CR3BP (Earth-Moon): Lagrange points, L1 Lyapunov orbit family via
differential correction, invariant manifolds, and MPC station-keeping.

Nondimensional units: distance = Earth-Moon distance, time = 1/mean motion.
State x = [x, y, vx, vy] in the rotating frame.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import cvxpy as cp

MU = 0.012150585          # Earth-Moon mass parameter
L_KM = 384400.0           # length unit [km]
T_S = 375190.0            # time unit [s] (~4.34 days)
V_MS = L_KM * 1e3 / T_S   # velocity unit [m/s]
A_MS2 = V_MS / T_S        # acceleration unit [m/s^2]
TOL = dict(rtol=1e-11, atol=1e-12)


# ---------------------------------------------------------------- dynamics
def r12(x, y):
    r1 = np.sqrt((x + MU) ** 2 + y ** 2)
    r2 = np.sqrt((x - 1 + MU) ** 2 + y ** 2)
    return r1, r2


def accel(x, y, vx, vy):
    r1, r2 = r12(x, y)
    ax = 2 * vy + x - (1 - MU) * (x + MU) / r1 ** 3 - MU * (x - 1 + MU) / r2 ** 3
    ay = -2 * vx + y - (1 - MU) * y / r1 ** 3 - MU * y / r2 ** 3
    return ax, ay


def f(t, s, u=(0.0, 0.0)):
    x, y, vx, vy = s
    ax, ay = accel(x, y, vx, vy)
    return [vx, vy, ax + u[0], ay + u[1]]


def jacobian(x, y):
    """A(t) of the linearized planar CR3BP."""
    r1, r2 = r12(x, y)
    a, b = x + MU, x - 1 + MU
    Uxx = 1 - (1 - MU) / r1**3 - MU / r2**3 + 3*(1 - MU)*a**2 / r1**5 + 3*MU*b**2 / r2**5
    Uyy = 1 - (1 - MU) / r1**3 - MU / r2**3 + 3*(1 - MU)*y**2 / r1**5 + 3*MU*y**2 / r2**5
    Uxy = 3*(1 - MU)*a*y / r1**5 + 3*MU*b*y / r2**5
    return np.array([[0, 0, 1, 0],
                     [0, 0, 0, 1],
                     [Uxx, Uxy, 0, 2],
                     [Uxy, Uyy, -2, 0]])


def f_stm(t, z):
    s, Phi = z[:4], z[4:].reshape(4, 4)
    return np.concatenate([f(t, s), (jacobian(s[0], s[1]) @ Phi).ravel()])


def propagate_stm(s0, tf, **kw):
    z0 = np.concatenate([s0, np.eye(4).ravel()])
    return solve_ivp(f_stm, (0, tf), z0, **TOL, **kw)


def jacobi(s):
    x, y, vx, vy = s
    r1, r2 = r12(x, y)
    return x**2 + y**2 + 2*(1 - MU)/r1 + 2*MU/r2 - vx**2 - vy**2


# ---------------------------------------------------------- Lagrange points
def collinear_points():
    g = lambda x: x - (1 - MU)*(x + MU)/abs(x + MU)**3 - MU*(x - 1 + MU)/abs(x - 1 + MU)**3
    L1 = brentq(g, 0.5, 1 - MU - 1e-3)
    L2 = brentq(g, 1 - MU + 1e-3, 1.5)
    L3 = brentq(g, -1.5, -MU - 1e-3)
    return L1, L2, L3


# -------------------------------------------- Lyapunov orbits (diff. corr.)
def y_cross(t, z):
    return z[1]
y_cross.terminal, y_cross.direction = True, -1   # first crossing from y>0


def correct_lyapunov(x0, vy0, iters=30):
    """Fix x0, correct vy0 so that vx = 0 at the half-period x-axis crossing."""
    for _ in range(iters):
        sol = propagate_stm([x0, 0, 0, vy0], 10, events=y_cross)
        zf = sol.y[:, -1]
        sf, Phi = zf[:4], zf[4:].reshape(4, 4)
        if abs(sf[2]) < 1e-11:
            return vy0, 2 * sol.t[-1]
        ax, _ = accel(*sf)
        dvy = -sf[2] / (Phi[2, 3] - ax / sf[3] * Phi[1, 3])
        vy0 += dvy
    raise RuntimeError("differential correction did not converge")


def lyapunov_family(xL1, n=8, step=0.003):
    A = jacobian(xL1, 0)
    w, V = np.linalg.eig(A)
    k = np.argmax(np.abs(w.imag))           # in-plane center mode
    v = V[:, k] / V[0, k]                   # x-component real
    eps = -1e-3
    x0, vy0 = xL1 + eps * v[0].real, eps * v[3].real
    fam = []
    for i in range(n):
        vy0, T = correct_lyapunov(x0, vy0)
        fam.append((x0, vy0, T))
        x_new = x0 - step                   # continuation in amplitude
        if len(fam) >= 2:                   # secant extrapolation of vy0(x0)
            (xa, va, _), (xb, vb, _) = fam[-2], fam[-1]
            vy0 = vb + (vb - va) / (xb - xa) * (x_new - xb)
        else:                               # linear scaling with amplitude
            vy0 = vy0 * (xL1 - x_new) / (xL1 - x0)
        x0 = x_new
    return fam


# ------------------------------------------------------------- manifolds
def manifolds(s0, T, n_pts=40, d_km=50.0, t_flow=3.5):
    sol = propagate_stm(s0, T, dense_output=True)
    M = sol.y[4:, -1].reshape(4, 4)
    lam, V = np.linalg.eig(M)
    iu, isb = np.argmax(np.abs(lam)), np.argmin(np.abs(lam))
    vu, vs = V[:, iu].real, V[:, isb].real
    d = d_km / L_KM
    unstable, stable = [], []
    for tau in np.linspace(0, T, n_pts, endpoint=False):
        z = sol.sol(tau)
        s, Phi = z[:4], z[4:].reshape(4, 4)
        for vec, sign_t, out in ((vu, 1, unstable), (vs, -1, stable)):
            v = Phi @ vec
            v /= np.linalg.norm(v[:2])
            for side in (+1, -1):
                tr = solve_ivp(f, (0, sign_t * t_flow), s + side * d * v,
                               rtol=1e-9, atol=1e-10, max_step=0.01)
                out.append(tr.y)
    return np.abs(lam[iu]), unstable, stable


# ------------------------------------------------------ MPC station-keeping
def station_keeping(s0, T, n_periods=3, steps_per_T=150, N=25,
                    inj_km=20.0, inj_cms=5.0, nav_km=1.0, nav_cms=1.0,
                    umax_ms2=2e-4, seed=0):
    rng = np.random.default_rng(seed)
    dt = T / steps_per_T
    K = n_periods * steps_per_T
    ref = solve_ivp(f, (0, T), s0, dense_output=True, **TOL)
    xref = lambda t: ref.sol(t % T)

    # discrete LTV model along the reference: Ad_k = STM over one step
    Ad, Bd = [], []
    Bc = np.vstack([np.zeros((2, 2)), np.eye(2)])
    for k in range(steps_per_T):
        z = solve_ivp(f_stm, (0, dt), np.concatenate([xref(k*dt), np.eye(4).ravel()]),
                      **TOL).y[:, -1]
        Phi = z[4:].reshape(4, 4)
        Ad.append(Phi)
        Bd.append(0.5 * (Phi + np.eye(4)) @ Bc * dt)   # trapezoidal ZOH approx

    umax = umax_ms2 / A_MS2
    Q = np.diag([1, 1, 0.1, 0.1]) * 1e4
    R = np.eye(2) * 1.0

    dx = cp.Variable((4, N + 1))
    du = cp.Variable((2, N))
    dx0 = cp.Parameter(4)
    A_par = [cp.Parameter((4, 4)) for _ in range(N)]
    B_par = [cp.Parameter((4, 2)) for _ in range(N)]
    cost, cons = 0, [dx[:, 0] == dx0]
    for j in range(N):
        cost += cp.quad_form(dx[:, j + 1], Q) + cp.quad_form(du[:, j], R)
        cons += [dx[:, j + 1] == A_par[j] @ dx[:, j] + B_par[j] @ du[:, j],
                 cp.norm(du[:, j], "inf") <= umax]
    prob = cp.Problem(cp.Minimize(cost), cons)

    unit = lambda v: v / np.linalg.norm(v)
    inj = np.r_[unit(rng.normal(size=2)) * inj_km / L_KM,      # fixed-size injection error
                unit(rng.normal(size=2)) * inj_cms / 100 / V_MS]
    nav_sd = np.r_[[nav_km / L_KM] * 2, [nav_cms / 100 / V_MS] * 2]

    def run(controlled):
        s = s0 + inj
        traj, us, err = [s.copy()], [], []
        for k in range(K):
            t = k * dt
            e_true = s - xref(t)
            err.append(np.linalg.norm(e_true[:2]) * L_KM)
            if controlled:
                dx0.value = e_true + rng.normal(size=4) * nav_sd   # noisy estimate
                for j in range(N):
                    A_par[j].value = Ad[(k + j) % steps_per_T]
                    B_par[j].value = Bd[(k + j) % steps_per_T]
                prob.solve(solver=cp.CLARABEL)
                u = du.value[:, 0]
            else:
                u = np.zeros(2)
            s = solve_ivp(f, (t, t + dt), s, args=(u,), **TOL).y[:, -1]
            traj.append(s.copy()); us.append(u)
            if np.linalg.norm(s[:2] - xref(t + dt)[:2]) > 0.3:     # escaped
                break
        return np.array(traj), np.array(us), np.array(err), dt

    return run(False), run(True), xref


# ------------------------------------------------------------------- main
def main():
    xL1, xL2, xL3 = collinear_points()
    print(f"L1 = {xL1:.6f}, L2 = {xL2:.6f}, L3 = {xL3:.6f} (nondim)")
    print(f"L1 is {(1 - MU - xL1) * L_KM:,.0f} km from the Moon")

    fam = lyapunov_family(xL1)
    for x0, vy0, T in fam:
        print(f"Lyapunov: x0={x0:.5f} vy0={vy0:+.5f} T={T:.4f} ({T*T_S/86400:.2f} d) "
              f"C={jacobi([x0, 0, 0, vy0]):.5f}")

    x0, vy0, T = fam[-1]
    s0 = np.array([x0, 0, 0, vy0])
    lam_u, Wu, Ws = manifolds(s0, T)
    print(f"Unstable monodromy eigenvalue: {lam_u:.1f} (deviation x{lam_u:.0f} per period)")

    (tr0, _, e0, dt), (tr1, u1, e1, _), xref = station_keeping(s0, T)
    dv = np.sum(np.linalg.norm(u1, axis=1)) * dt * V_MS
    days = len(u1) * dt * T_S / 86400
    print(f"Uncontrolled: escaped after {len(e0)*dt*T_S/86400:.1f} d")
    print(f"MPC: max error {e1.max():.1f} km, final {e1[-1]:.1f} km, "
          f"total dv {dv:.2f} m/s over {days:.1f} d")

    # ---------------------------------------------------------- figures
    plt.rcParams.update({"font.size": 10})
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))

    for x0_, vy0_, T_ in fam:
        o = solve_ivp(f, (0, T_), [x0_, 0, 0, vy0_], **TOL, max_step=0.01)
        ax[0].plot(o.y[0], o.y[1], lw=1)
    ax[0].plot(xL1, 0, "k+", ms=10); ax[0].text(xL1, 0.004, "L1", ha="center")
    ax[0].plot(1 - MU, 0, "o", color="gray", ms=8); ax[0].text(1 - MU, 0.01, "Moon", ha="center")
    ax[0].set_title("L1 Lyapunov orbit family (differential correction)")

    for w in Wu:
        ax[1].plot(w[0], w[1], color="tab:red", lw=0.4, alpha=0.6)
    for w in Ws:
        ax[1].plot(w[0], w[1], color="tab:green", lw=0.4, alpha=0.6)
    o = solve_ivp(f, (0, T), s0, **TOL, max_step=0.01)
    ax[1].plot(o.y[0], o.y[1], "k", lw=1.5)
    ax[1].plot(-MU, 0, "o", color="tab:blue", ms=10); ax[1].plot(1 - MU, 0, "o", color="gray", ms=6)
    ax[1].set_xlim(0.3, 1.3); ax[1].set_ylim(-0.35, 0.35)
    ax[1].set_title("Unstable (red) and stable (green) manifolds")

    t0 = np.arange(len(e0)) * dt * T_S / 86400
    t1 = np.arange(len(e1)) * dt * T_S / 86400
    ax[2].semilogy(t0, e0, label="No control")
    ax[2].semilogy(t1, e1, label="MPC (noisy navigation, |u| bounded)")
    ax[2].set_xlabel("time [days]"); ax[2].set_ylabel("position error [km]")
    ax[2].set_title(f"Station-keeping: total dv = {dv:.2f} m/s in {days:.0f} d")
    ax[2].legend(); ax[2].grid(True, which="both", alpha=0.3)

    for a in ax[:2]:
        a.set_aspect("equal"); a.set_xlabel("x [nondim]"); a.set_ylabel("y [nondim]")
    fig.tight_layout()
    fig.savefig("cr3bp_l1_mpc.png", dpi=150)
    print("saved cr3bp_l1_mpc.png")


if __name__ == "__main__":
    main()
