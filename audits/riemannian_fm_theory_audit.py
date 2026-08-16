"""
riemannian_fm_theory_audit.py
=============================

Numerical verification of the geometric identities that Chapter 12 of
``Texte/theory.tex`` (Riemannian flow matching) derives, plus the geometry
facts newly proved in Chapter 3.  Pure numpy/scipy, float64, no GPU, no
repository imports.

Conventions match the monograph:
  * generation time t in [0,1];  t=0 source, t=1 data
  * SO(3) metric  g(A,B) = 1/2 tr(A^T B),  so ||R w^||_g = ||w||_2
  * body-frame (left-trivialised) angular velocity:  w^  = R^T Rdot
  * geodesic interpolant           X_t = Exp_{x0}( t Log_{x0}(x1) )
  * conditional vector field       u_t(x|z) = Log_x(z) / (1-t)

Run:  python audits/riemannian_fm_theory_audit.py
"""

import numpy as np
from scipy.stats import ks_2samp

rng = np.random.default_rng(20260816)
FAILURES = []


def report(name, residual, tol, note=""):
    ok = residual <= tol
    if not ok:
        FAILURES.append(name)
    extra = f"   ({note})" if note else ""
    print(f"  [{'OK  ' if ok else 'FAIL'}] {name:<56s} residual = {residual:.3e}  tol = {tol:.1e}{extra}")


def section(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


# ============================================================================
#  SO(3) primitives (Rodrigues / matrix log, exactly as the monograph derives)
# ============================================================================

def hat(w):
    w = np.asarray(w, dtype=float)
    return np.array([[0.0, -w[2], w[1]],
                     [w[2], 0.0, -w[0]],
                     [-w[1], w[0], 0.0]])


def vee(A):
    return np.array([-A[1, 2], A[0, 2], -A[0, 1]])


def Exp(w):
    """Rodrigues, thm:rodrigues-scratch."""
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3) + hat(w)
    W = hat(w)
    return np.eye(3) + (np.sin(th) / th) * W + ((1 - np.cos(th)) / th ** 2) * (W @ W)


def angle_of(R):
    return np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))


def Log(R):
    """Principal log, thm:so3-log-scratch, with the two special regimes."""
    th = angle_of(R)
    if th < 1e-10:
        return vee(0.5 * (R - R.T))
    if abs(th - np.pi) < 1e-7:                      # near-pi branch, prop:log-nearpi
        M = 0.5 * (R + np.eye(3))
        i = int(np.argmax(np.diag(M)))
        n = M[:, i] / np.sqrt(max(M[i, i], 0.0))
        n = n / np.linalg.norm(n)
        return np.pi * n
    return vee(th / (2 * np.sin(th)) * (R - R.T))


def haar_rotation(n=1):
    """Sample Haar(SO(3)): axis uniform on S^2, angle ~ (1-cos th)/pi on [0,pi]."""
    grid = np.linspace(0.0, np.pi, 20001)
    cdf = np.cumsum(1.0 - np.cos(grid))
    cdf = cdf / cdf[-1]
    u = rng.uniform(size=n)
    th = np.interp(u, cdf, grid)
    ax = rng.normal(size=(n, 3))
    ax /= np.linalg.norm(ax, axis=1, keepdims=True)
    return np.stack([Exp(th[i] * ax[i]) for i in range(n)]), th


def g_so3(A, B):
    """Ambient/bi-invariant inner product 1/2 tr(A^T B)."""
    return 0.5 * np.trace(A.T @ B)


# ============================================================================
section("1.  SO(3): Exp and Log are mutual inverses")
# ============================================================================
Rs, _ = haar_rotation(400)
res_el, res_le = 0.0, 0.0
for R in Rs:
    res_el = max(res_el, np.max(np.abs(Exp(Log(R)) - R)))
for _ in range(400):
    w = rng.normal(size=3)
    w = w / np.linalg.norm(w) * rng.uniform(0.01, np.pi - 0.01)   # inside injectivity radius
    res_le = max(res_le, np.max(np.abs(Log(Exp(w)) - w)))
report("Exp(Log(R)) == R  over Haar samples", res_el, 1e-9)
report("Log(Exp(w)) == w  for ||w|| < pi", res_le, 1e-9)


# ============================================================================
section("2.  SO(3): the bi-invariant metric and the vee isometry")
# ============================================================================
# prop:so3-distance normalisation:  || R w^ ||_g = || w ||_2
res = 0.0
for _ in range(300):
    R, _ = haar_rotation(1); R = R[0]
    w = rng.normal(size=3)
    res = max(res, abs(np.sqrt(g_so3(R @ hat(w), R @ hat(w))) - np.linalg.norm(w)))
report("|| R w^ ||_g == || w ||_2  (the 1/2 in the metric)", res, 1e-12)

# bi-invariance: g is unchanged by left and right translation
res_L, res_R = 0.0, 0.0
for _ in range(300):
    Q, _ = haar_rotation(1); Q = Q[0]
    R, _ = haar_rotation(1); R = R[0]
    A, B = R @ hat(rng.normal(size=3)), R @ hat(rng.normal(size=3))
    res_L = max(res_L, abs(g_so3(Q @ A, Q @ B) - g_so3(A, B)))
    res_R = max(res_R, abs(g_so3(A @ Q, B @ Q) - g_so3(A, B)))
report("metric is left-invariant", res_L, 1e-12)
report("metric is right-invariant (bi-invariance)", res_R, 1e-12)

# prop:so3-distance:  d_g(R0,R1) = theta(R0^T R1) = ||Log(R0^T R1)||
res = 0.0
for _ in range(300):
    R0, _ = haar_rotation(1); R1, _ = haar_rotation(1)
    R0, R1 = R0[0], R1[0]
    res = max(res, abs(np.linalg.norm(Log(R0.T @ R1)) - angle_of(R0.T @ R1)))
report("|| Log(R0^T R1) || == theta  (distance)", res, 1e-9)


# ============================================================================
section("3.  SO(3): one-parameter subgroups really are geodesics")
# ============================================================================
# prop:so3-geodesic-proof:  Rddot = R Xi^2  is orthogonal to T_R SO(3).
res = 0.0
for _ in range(300):
    R0, _ = haar_rotation(1); R0 = R0[0]
    Xi = hat(rng.normal(size=3))
    t = rng.uniform(0, 1)
    R = R0 @ Exp(t * vee(Xi))
    Rddot = R @ (Xi @ Xi)
    for _ in range(5):                                # test against random tangents
        Om = hat(rng.normal(size=3))
        res = max(res, abs(g_so3(Rddot, R @ Om)))
report("Rddot _|_ T_R SO(3)  (geodesic equation)", res, 1e-12)

# and the resulting constant speed, prop:geodesic-constant-speed
res = 0.0
for _ in range(200):
    R0, _ = haar_rotation(1); R0 = R0[0]
    xi = rng.normal(size=3)
    sp = [np.sqrt(g_so3(R0 @ Exp(t * xi) @ hat(xi), R0 @ Exp(t * xi) @ hat(xi)))
          for t in np.linspace(0, 1, 7)]
    res = max(res, np.ptp(sp))
report("speed is constant along the geodesic", res, 1e-12)


# ============================================================================
section("4.  SO(3): the geodesic interpolant and its velocity")
# ============================================================================
res_ep0, res_ep1, res_fd, res_bd, res_cp = 0.0, 0.0, 0.0, 0.0, 0.0
h = 1e-6
for _ in range(300):
    R0, _ = haar_rotation(1); R1, _ = haar_rotation(1)
    R0, R1 = R0[0], R1[0]
    if angle_of(R0.T @ R1) > np.pi - 1e-3:            # stay off the cut locus
        continue
    xi = Log(R0.T @ R1)                                # Xi^vee
    Rt = lambda t: R0 @ Exp(t * xi)

    res_ep0 = max(res_ep0, np.max(np.abs(Rt(0.0) - R0)))
    res_ep1 = max(res_ep1, np.max(np.abs(Rt(1.0) - R1)))

    t = rng.uniform(0.05, 0.9)
    Rdot_fd = (Rt(t + h) - Rt(t - h)) / (2 * h)        # finite-difference velocity
    res_fd = max(res_fd, np.max(np.abs(Rdot_fd - Rt(t) @ hat(xi))))

    # body-frame velocity is the constant Xi
    res_bd = max(res_bd, np.max(np.abs(Rt(t).T @ Rdot_fd - hat(xi))))

    # current-point form:  Log(R_t^T R_1) / (1-t) == Xi^vee
    res_cp = max(res_cp, np.max(np.abs(Log(Rt(t).T @ R1) / (1 - t) - xi)))

report("interpolant endpoint R_{t=0} == R_0", res_ep0, 1e-12)
report("interpolant endpoint R_{t=1} == R_1", res_ep1, 1e-9)
report("finite-difference velocity == R_t Xi", res_fd, 1e-8, "central diff, h=1e-6")
report("body-frame velocity R_t^T Rdot_t == Xi (constant)", res_bd, 1e-8)
report("current-point form Log(R_t^T R_1)/(1-t) == Xi^vee", res_cp, 1e-9)


# ============================================================================
section("5.  SO(3): the 90-degree worked example from the text")
# ============================================================================
R0 = np.eye(3)
R1 = Exp(np.array([0.0, 0.0, np.pi / 2]))
xi = Log(R0.T @ R1)
print(f"  [INFO] Xi^vee = {np.round(xi, 6)}   (expected (0,0,pi/2) = (0,0,{np.pi/2:.6f}))")
report("worked example: Xi^vee == (0,0,pi/2)", np.max(np.abs(xi - np.array([0, 0, np.pi / 2]))), 1e-12)
res = 0.0
for t in (0.0, 0.25, 0.5, 0.75, 1.0):
    Rt = R0 @ Exp(t * xi)
    phi = np.pi * t / 2
    Rz = np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])
    res = max(res, np.max(np.abs(Rt - Rz)))
report("worked example: R_t == R_z(t*pi/2)", res, 1e-12)
t = 0.37
Rt = R0 @ Exp(t * xi)
Rdot = (R0 @ Exp((t + h) * xi) - R0 @ Exp((t - h) * xi)) / (2 * h)
report("worked example: R_t^T Rdot_t == (pi/2) e_z^",
       np.max(np.abs(Rt.T @ Rdot - hat(xi))), 1e-8)
print(f"  [INFO] ||omega_bd|| = {np.linalg.norm(vee(Rt.T @ Rdot)):.6f} = d_g(R0,R1) = {angle_of(R0.T@R1):.6f}")


# ============================================================================
section("6.  SO(3): Haar measure is bi-invariant (= Riemannian volume)")
# ============================================================================
# prop:so3-haar-is-volume relies on Haar being the unique bi-invariant measure.
# Numerically: the angle distribution of Haar samples is unchanged by left or
# right translation by a fixed rotation.
Rs, th = haar_rotation(20000)
Q, _ = haar_rotation(1); Q = Q[0]
thL = np.array([angle_of(Q @ R) for R in Rs])
thR = np.array([angle_of(R @ Q) for R in Rs])
pL = ks_2samp(th, thL).pvalue
pR = ks_2samp(th, thR).pvalue
print(f"  [INFO] KS p-values: left-translated {pL:.3f}, right-translated {pR:.3f}")
report("Haar invariant under left translation (KS)", 0.0 if pL > 0.01 else 1.0, 0.5,
       "reported as pass/fail of a KS test at level 0.01")
report("Haar invariant under right translation (KS)", 0.0 if pR > 0.01 else 1.0, 0.5)
# and the analytic density (1-cos)/pi
emp = np.mean(th > np.deg2rad(171.9))
ana = (np.pi - (np.deg2rad(171.9) - np.sin(np.deg2rad(171.9)))) / np.pi
print(f"  [INFO] P(theta > 171.9 deg): empirical {emp:.4f}, analytic {ana:.4f}")
report("angle density (1-cos)/pi reproduced", abs(emp - ana), 0.01, "MC, N=20000")


# ============================================================================
section("7.  S^2: Exp/Log, interpolant, and the general velocity formula")
# ============================================================================

def s2_exp(x, v):
    a = np.linalg.norm(v)
    if a < 1e-14:
        return x.copy()
    return np.cos(a) * x + np.sin(a) * v / a


def s2_log(x, y):
    a = np.arccos(np.clip(x @ y, -1.0, 1.0))
    w = y - (x @ y) * x
    nw = np.linalg.norm(w)
    if nw < 1e-14:
        return np.zeros(3)
    return a * w / nw


def rand_s2():
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


res_rt, res_tan, res_fd, res_cp, res_sp = 0.0, 0.0, 0.0, 0.0, 0.0
for _ in range(400):
    x, z = rand_s2(), rand_s2()
    if x @ z < -0.999:                                 # off the cut locus
        continue
    v = s2_log(x, z)
    res_rt = max(res_rt, np.max(np.abs(s2_exp(x, v) - z)))
    res_tan = max(res_tan, abs(x @ v))
    Xt = lambda t: s2_exp(x, t * v)
    t = rng.uniform(0.05, 0.9)
    fd = (Xt(t + h) - Xt(t - h)) / (2 * h)
    res_cp = max(res_cp, np.max(np.abs(s2_log(Xt(t), z) / (1 - t) - fd)))
    res_tan = max(res_tan, abs(Xt(t) @ fd))
    sp = [np.linalg.norm((Xt(tt + h) - Xt(tt - h)) / (2 * h)) for tt in (0.1, 0.4, 0.8)]
    res_sp = max(res_sp, np.ptp(sp))
report("S^2: Exp_x(Log_x(y)) == y", res_rt, 1e-12)
report("S^2: Log and velocity are tangent (x . v == 0)", res_tan, 1e-8)
report("S^2: Log_{X_t}(z)/(1-t) == Xdot_t   [prop:geodesic-vf-general]", res_cp, 1e-8,
       "central diff, h=1e-6")
report("S^2: constant speed along the interpolant", res_sp, 1e-8)

# the explicit quarter-equator example from the text
x, z = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
print(f"  [INFO] Log_x(z) = {np.round(s2_log(x, z), 6)}  (expected (0, pi/2, 0))")
report("S^2 worked example: Log_x(z) == (0,pi/2,0)",
       np.max(np.abs(s2_log(x, z) - np.array([0, np.pi / 2, 0]))), 1e-12)


# ============================================================================
section("8.  S^1: the wrap is the principal branch")
# ============================================================================

def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


th0, th1 = np.deg2rad(170.0), np.deg2rad(-170.0)
D = wrap(th1 - th0)
print(f"  [INFO] naive difference {np.rad2deg(th1-th0):+.1f} deg, wrapped {np.rad2deg(D):+.1f} deg")
report("S^1 worked example: wrapped log == +20 deg", abs(np.rad2deg(D) - 20.0), 1e-9)
t = 0.75
tht = th0 + t * D
report("S^1: Log_{theta_t}(theta_1)/(1-t) == Delta",
       abs(wrap(th1 - tht) / (1 - t) - D), 1e-12)
report("S^1: remaining log is (1-t)*Delta",
       abs(wrap(th1 - tht) - (1 - t) * D), 1e-12)


# ============================================================================
section("9.  Product manifold R^3 x SO(3): everything factorises")
# ============================================================================
lam = 1.0
r0 = np.zeros(3)
r1 = np.array([2.0, 1.0, 0.0])
P0, P1 = np.eye(3), Exp(np.array([0.0, 0.0, np.pi / 2]))
xi = Log(P0.T @ P1)

res_r, res_R, res_v, res_w = 0.0, 0.0, 0.0, 0.0
for t in np.linspace(0.0, 1.0, 11):
    rt = (1 - t) * r0 + t * r1
    Rt = P0 @ Exp(t * xi)
    res_r = max(res_r, np.max(np.abs(rt - t * r1)))
    fd_r = ((1 - (t + h)) * r0 + (t + h) * r1 - ((1 - (t - h)) * r0 + (t - h) * r1)) / (2 * h)
    res_v = max(res_v, np.max(np.abs(fd_r - (r1 - r0))))
    if 0 < t < 1:
        fd_R = (P0 @ Exp((t + h) * xi) - P0 @ Exp((t - h) * xi)) / (2 * h)
        res_w = max(res_w, np.max(np.abs(vee(Rt.T @ fd_R) - xi)))
res_R = max(np.max(np.abs(P0 @ Exp(0 * xi) - P0)), np.max(np.abs(P0 @ Exp(1 * xi) - P1)))
report("product: translation endpoints and constant velocity", max(res_r, res_v), 1e-8)
report("product: rotation endpoints", res_R, 1e-9)
report("product: rotational body velocity constant", res_w, 1e-8)

speed2 = np.linalg.norm(r1 - r0) ** 2 + lam * np.linalg.norm(xi) ** 2
dist2 = np.linalg.norm(r1 - r0) ** 2 + lam * angle_of(P0.T @ P1) ** 2
print(f"  [INFO] ||(v,omega)||_g = {np.sqrt(speed2):.6f},  d_g(pose0,pose1) = {np.sqrt(dist2):.6f}")
report("product: speed == distance (lambda = 1)", abs(speed2 - dist2), 1e-12)

# prop:lambda-effect(i): the SAME curve is a constant-speed geodesic for every
# lambda -- i.e. changing lambda rescales all speeds by one common factor and
# leaves the interpolant untouched.  This can fail: if lambda changed the
# geodesic, the speed would stop being constant in t.
res_const, res_dist = 0.0, 0.0
for lam in (0.25, 1.0, 4.0, 9.0):
    speeds = []
    for t in np.linspace(0.05, 0.95, 9):
        fd_r = (r1 - r0)                                   # translation velocity
        fd_R = (P0 @ Exp((t + h) * xi) - P0 @ Exp((t - h) * xi)) / (2 * h)
        Rt = P0 @ Exp(t * xi)
        om = vee(Rt.T @ fd_R)
        speeds.append(np.sqrt(np.sum(fd_r ** 2) + lam * np.sum(om ** 2)))
    res_const = max(res_const, np.ptp(speeds))             # constant in t?
    d_lam = np.sqrt(np.sum((r1 - r0) ** 2) + lam * angle_of(P0.T @ P1) ** 2)
    res_dist = max(res_dist, abs(np.mean(speeds) - d_lam))  # equals the g^lambda distance?
report("product: same curve is a g^lambda-geodesic for all lambda", res_const, 1e-8,
       "speed constant in t for lambda in {0.25,1,4,9}")
report("product: its speed equals d_{g^lambda}", res_dist, 1e-8)


# ============================================================================
section("10.  Euclidean consistency: the flat case reduces to Chapter 11")
# ============================================================================
res_i, res_u = 0.0, 0.0
for _ in range(400):
    d = 4
    x0, x1 = rng.normal(size=d), rng.normal(size=d)
    t = rng.uniform(0.02, 0.95)
    # Exp_x(v) = x+v, Log_x(y) = y-x
    xt_riem = x0 + t * (x1 - x0)
    xt_eucl = (1 - t) * x0 + t * x1
    res_i = max(res_i, np.max(np.abs(xt_riem - xt_eucl)))
    u_riem = (x1 - xt_riem) / (1 - t)
    res_u = max(res_u, np.max(np.abs(u_riem - (x1 - x0))))
report("flat case: geodesic interpolant == linear path", res_i, 1e-14)
report("flat case: Log_{x_t}(x_1)/(1-t) == x_1 - x_0", res_u, 1e-13)


# ============================================================================
section("11.  IGSO(3) heat-kernel limits (Chapter 7 worked example)")
# ============================================================================
# f_sigma(w) = sum_l (2l+1) chi_l(w) exp(-l(l+1) sigma^2 / 2),
# chi_l(w) = sin((l+1/2)w)/sin(w/2)   -- density w.r.t. Haar.
# Two limits:  sigma -> infinity gives f == 1 (uniform);
#              sigma -> 0 gives the pushforward of N(0, sigma^2 I_3) under Exp,
#              whose angle marginal is the chi_3 density  ~ w^2 exp(-w^2/2s^2).

def igso3_f(w, s, L=2000):
    l = np.arange(L + 1)[:, None]
    w = np.atleast_1d(w)[None, :]
    chi = np.where(np.abs(np.sin(w / 2)) < 1e-12, 2 * l + 1,
                   np.sin((l + 0.5) * w) / np.sin(w / 2))
    return ((2 * l + 1) * chi * np.exp(-0.5 * l * (l + 1) * s ** 2)).sum(0)


wg = np.linspace(1e-6, np.pi, 400)
print("  large-sigma limit: max|f_sigma - 1| over (0, pi]")
dev = {}
for s in (1.5, 3.0, 5.0):
    dev[s] = np.max(np.abs(igso3_f(wg, s) - 1.0))
    print(f"    sigma = {s:4.1f}   ->  {dev[s]:.3e}")
report("IGSO(3) -> uniform for large sigma", dev[5.0], 1e-8)
report("IGSO(3) NOT uniform at sigma_max = 1.5",
       0.0 if dev[1.5] > 0.5 else 1.0, 0.5,
       f"deviation {dev[1.5]:.2f} is order one, as the text claims")

print("  small-sigma limit: max|p_IGSO - p_chi3| (angle marginals, normalised)")
worst = 0.0
for s in (0.05, 0.10, 0.20):
    p1 = igso3_f(wg, s) * (1 - np.cos(wg)); p1 /= np.trapezoid(p1, wg)
    p2 = wg ** 2 * np.exp(-wg ** 2 / (2 * s ** 2)); p2 /= np.trapezoid(p2, wg)
    d = np.max(np.abs(p1 - p2))
    worst = max(worst, d)
    print(f"    sigma = {s:4.2f}   ->  {d:.3e}   (peak density {p1.max():.2f})")
report("IGSO(3) -> Exp-pushforward Gaussian for small sigma", worst, 1e-2,
       "deviation grows linearly in sigma, as curvature starts to matter")


# ============================================================================
section("SUMMARY")
# ============================================================================
if FAILURES:
    print(f"  {len(FAILURES)} CHECK(S) FAILED:")
    for f in FAILURES:
        print(f"    - {f}")
    raise SystemExit(1)
print("  All checks passed.")
