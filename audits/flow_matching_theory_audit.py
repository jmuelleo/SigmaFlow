"""
flow_matching_theory_audit.py
=============================

Numerical verification of every non-trivial identity that Part VI of
``Texte/theory.tex`` (Euclidean flow matching) derives.  No GPU, no
checkpoint, no repository imports -- pure numpy/scipy, so this can be
re-run by anyone.

Conventions (identical to the monograph, ``rem:time-convention-global``):

    t in [0,1] is GENERATION time.  t=0 is the source p_init, t=1 is p_data.
    s = 1 - t is NOISING time, used only for the diffusion side.

Gaussian probability path (monograph ``def:gaussian-path``):

    X_t = alpha_t * z + sigma_t * eps,   eps ~ N(0, I_d),  z ~ p_data
    alpha_0 = 0, alpha_1 = 1, sigma_0 = 1, sigma_1 = 0.

Every check prints a residual; all residuals should be at machine-precision
level except where a finite-difference or Monte-Carlo tolerance is stated
explicitly in the check's own message.

Run:  python audits/flow_matching_theory_audit.py
"""

import numpy as np
from scipy.integrate import solve_ivp

rng = np.random.default_rng(20260816)

FAILURES = []


def report(name, residual, tol, note=""):
    ok = residual <= tol
    flag = "OK  " if ok else "FAIL"
    if not ok:
        FAILURES.append(name)
    extra = f"   ({note})" if note else ""
    print(f"  [{flag}] {name:<58s} residual = {residual:.3e}  tol = {tol:.1e}{extra}")


def section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# ----------------------------------------------------------------------------
# Schedules.  A schedule is (alpha, sigma, alpha_dot, sigma_dot) as callables.
# ----------------------------------------------------------------------------

def sched_condot():
    """Linear / conditional-optimal-transport path: alpha_t = t, sigma_t = 1-t."""
    return (lambda t: t,
            lambda t: 1.0 - t,
            lambda t: np.ones_like(np.asarray(t, dtype=float)),
            lambda t: -np.ones_like(np.asarray(t, dtype=float)))


def sched_trig():
    """Variance-preserving trigonometric path: alpha=cos(phi), sigma=sin(phi),
    phi(t) = (pi/2)(1-t), so alpha^2 + sigma^2 = 1 and phi decreases in t."""
    phi = lambda t: 0.5 * np.pi * (1.0 - t)
    dphi = -0.5 * np.pi
    return (lambda t: np.cos(phi(t)),
            lambda t: np.sin(phi(t)),
            lambda t: -np.sin(phi(t)) * dphi,
            lambda t: np.cos(phi(t)) * dphi)


def sched_ve(sigma_max=3.0):
    """Variance-exploding path read in generation time: alpha = 1, and
    sigma_t = sigma_max * (1-t) (so sigma_1 = 0, sigma_0 = sigma_max).
    Note alpha_0 = 1 != 0, i.e. this is a VE path, whose source is
    N(z, sigma_max^2) rather than N(0, I) -- the usual VE caveat."""
    return (lambda t: np.ones_like(np.asarray(t, dtype=float)),
            lambda t: sigma_max * (1.0 - t),
            lambda t: np.zeros_like(np.asarray(t, dtype=float)),
            lambda t: -sigma_max * np.ones_like(np.asarray(t, dtype=float)))


SCHEDULES = {"CondOT (alpha=t, sigma=1-t)": sched_condot(),
             "trig VP (cos/sin)": sched_trig(),
             "VE (alpha=1)": sched_ve()}


def u_cond(x, z, t, sch):
    """Conditional vector field, monograph eq. (u_t(x|z)) =
       alpha_dot * z + (sigma_dot/sigma) * (x - alpha*z)."""
    a, s, ad, sd = sch
    return ad(t) * z + (sd(t) / s(t)) * (x - a(t) * z)


def score_cond(x, z, t, sch):
    """Conditional score of N(alpha_t z, sigma_t^2 I): -(x - alpha z)/sigma^2."""
    a, s, _, _ = sch
    return -(x - a(t) * z) / s(t) ** 2


def ab_coeffs(t, sch):
    """Monograph coefficients:  u_t(x) = a_t * score_t(x) + b_t * x, with
       a_t = sigma^2 * alpha_dot/alpha - sigma_dot*sigma,   b_t = alpha_dot/alpha."""
    a, s, ad, sd = sch
    b_t = ad(t) / a(t)
    a_t = s(t) ** 2 * ad(t) / a(t) - sd(t) * s(t)
    return a_t, b_t


# ----------------------------------------------------------------------------
section("1.  Conditional vector field really generates the conditional path")
# ----------------------------------------------------------------------------
# Claim: the flow of u_t(.|z) started at x_0 is exactly x_t = alpha_t z + sigma_t x_0.
# We integrate the ODE numerically and compare against the closed form.

for name, sch in SCHEDULES.items():
    a, s, _, _ = sch
    d = 3
    z = rng.normal(size=d)
    # start at t0 slightly above 0 so that VE (alpha_0=1) is handled uniformly:
    t0, t1 = 1e-6, 1.0 - 1e-6
    x_start = a(t0) * z + s(t0) * rng.normal(size=d)
    x0_noise = (x_start - a(t0) * z) / s(t0)          # recover the eps that generated it

    sol = solve_ivp(lambda t, x: u_cond(x, z, t, sch), (t0, t1), x_start,
                    rtol=1e-11, atol=1e-13, dense_output=True)
    closed = a(t1) * z + s(t1) * x0_noise
    report(f"flow of u(.|z) matches alpha*z+sigma*eps  [{name}]",
           np.max(np.abs(sol.y[:, -1] - closed)), 1e-8)


# ----------------------------------------------------------------------------
section("2.  Conversion formula, CONDITIONAL:  u_t(x|z) = a_t*score + b_t*x")
# ----------------------------------------------------------------------------

for name, sch in SCHEDULES.items():
    d = 4
    res = 0.0
    for _ in range(200):
        t = rng.uniform(0.02, 0.98)
        z = rng.normal(size=d)
        x = rng.normal(size=d) * 2.0
        a_t, b_t = ab_coeffs(t, sch)
        lhs = u_cond(x, z, t, sch)
        rhs = a_t * score_cond(x, z, t, sch) + b_t * x
        res = max(res, np.max(np.abs(lhs - rhs)))
    report(f"conditional conversion identity  [{name}]", res, 1e-11)


# ----------------------------------------------------------------------------
section("3.  Marginal quantities for a discrete p_data (exactly computable)")
# ----------------------------------------------------------------------------
# p_data = sum_k w_k delta_{z_k}.  Then p_t is an exact Gaussian mixture, and
# both the marginal score and the marginal field are available in closed form.

D = 2
Z = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.5], [-0.7, -1.2]])
W = np.array([0.4, 0.3, 0.2, 0.1])


def gauss_pdf(x, mu, var):
    dd = x.shape[-1]
    return np.exp(-np.sum((x - mu) ** 2, axis=-1) / (2 * var)) / (2 * np.pi * var) ** (dd / 2)


def posterior(x, t, sch):
    """w_k(x) = p_t(x|z_k) p_data(z_k) / p_t(x), shape (..., K).

    Computed in log-space so that the ratio stays well defined even when every
    component density underflows (which happens for small sigma_t far from all
    the data points); only the *differences* of the exponents matter."""
    a, s, _, _ = sch
    logs = np.stack([-np.sum((x - a(t) * Z[k]) ** 2, axis=-1) / (2 * s(t) ** 2) + np.log(W[k])
                     for k in range(len(W))], axis=-1)
    logs = logs - np.max(logs, axis=-1, keepdims=True)
    comp = np.exp(logs)
    return comp / np.sum(comp, axis=-1, keepdims=True)


def p_marg(x, t, sch):
    a, s, _, _ = sch
    comp = np.stack([gauss_pdf(x, a(t) * Z[k], s(t) ** 2) * W[k] for k in range(len(W))], axis=-1)
    return np.sum(comp, axis=-1)


def u_marg(x, t, sch):
    """Marginal field as the posterior average of the conditional fields."""
    wgt = posterior(x, t, sch)                       # (..., K)
    fields = np.stack([u_cond(x, Z[k], t, sch) for k in range(len(W))], axis=-2)  # (...,K,D)
    return np.sum(wgt[..., None] * fields, axis=-2)


def score_marg(x, t, sch):
    wgt = posterior(x, t, sch)
    scores = np.stack([score_cond(x, Z[k], t, sch) for k in range(len(W))], axis=-2)
    return np.sum(wgt[..., None] * scores, axis=-2)


# 3a. marginal score really is grad log p_t  (finite difference)
for name, sch in SCHEDULES.items():
    h = 1e-6
    res = 0.0
    for _ in range(50):
        t = rng.uniform(0.05, 0.95)
        x = rng.normal(size=D) * 1.5
        fd = np.zeros(D)
        for i in range(D):
            e = np.zeros(D); e[i] = h
            fd[i] = (np.log(p_marg(x + e, t, sch)) - np.log(p_marg(x - e, t, sch))) / (2 * h)
        res = max(res, np.max(np.abs(fd - score_marg(x, t, sch))))
    report(f"posterior-averaged score == grad log p_t  [{name}]", res, 1e-5,
           "central difference, h=1e-6")

# 3b. marginal conversion formula, same a_t, b_t as the conditional one
for name, sch in SCHEDULES.items():
    res = 0.0
    for _ in range(300):
        t = rng.uniform(0.02, 0.98)
        x = rng.normal(size=D) * 2.0
        a_t, b_t = ab_coeffs(t, sch)
        res = max(res, np.max(np.abs(u_marg(x, t, sch) - (a_t * score_marg(x, t, sch) + b_t * x))))
    report(f"MARGINAL conversion identity (same a_t,b_t)  [{name}]", res, 1e-10)


# ----------------------------------------------------------------------------
section("4.  Continuity equation for the marginal pair (p_t, u_t)")
# ----------------------------------------------------------------------------
# d/dt p_t + div(p_t u_t) = 0, checked by central finite differences.

for name, sch in SCHEDULES.items():
    ht, hx = 1e-5, 1e-4
    res, scale = 0.0, 0.0
    for _ in range(40):
        t = rng.uniform(0.15, 0.85)
        x = rng.normal(size=D) * 1.2
        dpdt = (p_marg(x, t + ht, sch) - p_marg(x, t - ht, sch)) / (2 * ht)
        div = 0.0
        for i in range(D):
            e = np.zeros(D); e[i] = hx
            fp = p_marg(x + e, t, sch) * u_marg(x + e, t, sch)[i]
            fm = p_marg(x - e, t, sch) * u_marg(x - e, t, sch)[i]
            div += (fp - fm) / (2 * hx)
        res = max(res, abs(dpdt + div))
        scale = max(scale, abs(dpdt))
    report(f"continuity equation residual  [{name}]", res / max(scale, 1e-12), 5e-5,
           "relative to |dp/dt|, central differences")


# ----------------------------------------------------------------------------
section("5.  Marginalisation theorem:  u_t(x) = E[u_t(x|Z) | X_t = x]")
# ----------------------------------------------------------------------------
# Independent Monte-Carlo check: draw (Z, X_t) from the joint law, keep the
# samples whose X_t lands in a small ball around a fixed x, and average the
# conditional targets over those.  This never uses the posterior formula.

sch = sched_condot()
a, s, _, _ = sch
t = 0.6
x_star = np.array([0.35, 0.25])
radius = 0.02
n_hits, acc, acc_sq = 0, np.zeros(D), np.zeros(D)
for _ in range(40):                                   # 40 x 2e6 = 8e7 draws, streamed
    n = 2_000_000
    idx = rng.choice(len(W), size=n, p=W)
    Xt = a(t) * Z[idx] + s(t) * rng.normal(size=(n, D))
    keep = np.sum((Xt - x_star) ** 2, axis=1) < radius ** 2
    tg = u_cond(Xt[keep], Z[idx][keep], t, sch)
    n_hits += tg.shape[0]
    acc += tg.sum(axis=0)
    acc_sq += (tg ** 2).sum(axis=0)
mc = acc / n_hits
var = acc_sq / n_hits - mc ** 2
stderr = np.sqrt(var / n_hits)
exact = u_marg(x_star, t, sch)
# The correct statistical statement: the discrepancy must be small *in units of
# the Monte-Carlo standard error*, not small in absolute terms.
zscore = np.max(np.abs(mc - exact) / stderr)
print(f"  [INFO] {n_hits} samples in a ball of radius {radius}; "
      f"MC estimate {np.round(mc, 4)} vs exact {np.round(exact, 4)}")
print(f"  [INFO] standard error per component = {np.round(stderr, 4)}")
report("Monte-Carlo E[u(x|Z)|X_t=x] == u_t(x)", zscore, 4.0,
       "residual reported as |discrepancy| / MC standard error")


# ----------------------------------------------------------------------------
section("6.  Closed form of the two-point 2D example used in the text")
# ----------------------------------------------------------------------------
# p_data = 1/2 delta_a + 1/2 delta_{-a}, a = e_1, CondOT path.
# Claim:  u_t(x) = ( tanh( t*x_1/(1-t)^2 ) e_1 - x ) / (1-t).

Z_bak, W_bak = Z.copy(), W.copy()
Z = np.array([[1.0, 0.0], [-1.0, 0.0]])
W = np.array([0.5, 0.5])
sch = sched_condot()
res = 0.0
for _ in range(400):
    t = rng.uniform(0.02, 0.95)
    x = rng.normal(size=D) * 1.5
    claimed = (np.tanh(t * x[0] / (1 - t) ** 2) * np.array([1.0, 0.0]) - x) / (1 - t)
    res = max(res, np.max(np.abs(u_marg(x, t, sch) - claimed)))
report("two-point tanh formula for u_t(x)", res, 1e-10)

# endpoint value u_0(x) = E[X_1] - x = -x
res0 = np.max(np.abs(u_marg(np.array([0.4, -0.9]), 1e-9, sch) - (-np.array([0.4, -0.9]))))
report("u_0(x) = E[X_1] - x  (here = -x)", res0, 1e-7)

# The table printed in the monograph (Example "two data points in two dimensions"):
# marginal vs conditional field at x = (0.5, 0.3), showing the marginal field
# converging onto the conditional one as t -> 1.
x_tab = np.array([0.5, 0.3])
a_pt = np.array([1.0, 0.0])
print("\n  Table reproduced in the text (x = (0.5, 0.3)):")
print("     t      tanh        u_marginal              u_conditional(.|a)")
for t_ in (0.2, 0.5, 0.8, 0.95):
    th = np.tanh(t_ * x_tab[0] / (1 - t_) ** 2)
    um = u_marg(x_tab, t_, sch)
    uc = u_cond(x_tab, a_pt, t_, sch)
    print(f"   {t_:.2f}   {th:7.4f}   ({um[0]:8.4f},{um[1]:8.4f})   ({uc[0]:8.4f},{uc[1]:8.4f})")
Z, W = Z_bak, W_bak


# ----------------------------------------------------------------------------
section("7.  Diffusion bridge: probability flow ODE == Gaussian-path velocity")
# ----------------------------------------------------------------------------
# Monograph thm:pf-ode-scratch, in GENERATION time:
#     u_t(x) = -f(x, 1-t) + 1/2 g(1-t)^2 * grad log p_t(x).
# Claim: for the VP and VE SDEs this equals a_t * score + b_t * x with the
# Gaussian-path coefficients of section 2, once the schedules are matched by
#     alpha_t = alpha_diff(1-t),  sigma_t = sigma_diff(1-t).

print("\n  VP-SDE:  f(x,s) = -1/2 beta(s) x,  g(s) = sqrt(beta(s)),")
print("           alpha_diff(s) = exp(-1/2 int_0^s beta),  sigma_diff = sqrt(1-alpha^2)\n")

beta_min, beta_max = 0.1, 20.0
beta_rate = lambda s: beta_min + s * (beta_max - beta_min)
int_beta = lambda s: beta_min * s + 0.5 * s ** 2 * (beta_max - beta_min)
alpha_diff = lambda s: np.exp(-0.5 * int_beta(s))
sigma_diff = lambda s: np.sqrt(np.maximum(1.0 - alpha_diff(s) ** 2, 0.0))

# the same path written in generation time
al = lambda t: alpha_diff(1.0 - t)
si = lambda t: sigma_diff(1.0 - t)
h = 1e-7
ald = lambda t: (al(t + h) - al(t - h)) / (2 * h)
sid = lambda t: (si(t + h) - si(t - h)) / (2 * h)
sch_vp = (al, si, ald, sid)

res_a, res_b = 0.0, 0.0
for _ in range(300):
    t = rng.uniform(0.02, 0.98)
    a_t, b_t = ab_coeffs(t, sch_vp)
    # probability flow ODE coefficients, read off eq:pf-ode-scratch
    pf_b = 0.5 * beta_rate(1 - t)          # from -f(x,1-t) = +1/2 beta(1-t) x
    pf_a = 0.5 * beta_rate(1 - t)          # from 1/2 g(1-t)^2 = 1/2 beta(1-t)
    res_b = max(res_b, abs(b_t - pf_b))
    res_a = max(res_a, abs(a_t - pf_a))
report("VP: b_t (drift coeff) == 1/2 beta(1-t)", res_b, 1e-6, "alpha_dot by finite diff")
report("VP: a_t (score coeff) == 1/2 g(1-t)^2", res_a, 1e-6, "alpha_dot by finite diff")

print("\n  VE-SDE:  f = 0,  g(s)^2 = d/ds sigma(s)^2,  alpha_diff = 1\n")
sig_ve = lambda s: 0.02 * (50.0 ** s)              # geometric schedule
al_ve = lambda t: np.ones_like(np.asarray(t, dtype=float))
si_ve = lambda t: sig_ve(1.0 - t)
ald_ve = lambda t: np.zeros_like(np.asarray(t, dtype=float))
sid_ve = lambda t: (si_ve(t + h) - si_ve(t - h)) / (2 * h)
sch_ve = (al_ve, si_ve, ald_ve, sid_ve)

res_a, res_b = 0.0, 0.0
for _ in range(300):
    t = rng.uniform(0.02, 0.98)
    a_t, b_t = ab_coeffs(t, sch_ve)
    s_ = 1.0 - t
    g2 = (sig_ve(s_ + h) ** 2 - sig_ve(s_ - h) ** 2) / (2 * h)
    res_b = max(res_b, abs(b_t - 0.0))
    res_a = max(res_a, abs(a_t - 0.5 * g2))
report("VE: b_t == 0 (no drift)", res_b, 1e-12)
report("VE: a_t == 1/2 g(1-t)^2", res_a, 1e-6, "sigma_dot by finite diff")


# ----------------------------------------------------------------------------
section("8.  Diffusion v-parameterisation is NOT the flow velocity")
# ----------------------------------------------------------------------------
# Salimans & Ho (2022) define, on the VP trigonometric path,
#     v = alpha * eps - sigma * z.
# Claim: on THAT path, u_t = phi_dot * v with phi_dot = -pi/2 (a constant,
# negative, scalar), and on a non-VP path no such scalar exists.

sch = sched_trig()
a, s, ad, sd = sch
phidot = -0.5 * np.pi
res = 0.0
for _ in range(300):
    t = rng.uniform(0.02, 0.98)
    z = rng.normal(size=3)
    eps = rng.normal(size=3)
    x = a(t) * z + s(t) * eps
    v = a(t) * eps - s(t) * z
    u = u_cond(x, z, t, sch)
    res = max(res, np.max(np.abs(u - phidot * v)))
report("trig VP path:  u_t(x|z) == phi_dot * v   (phi_dot = -pi/2)", res, 1e-10)

# on the CondOT path the two are genuinely different directions
sch = sched_condot()
a, s, _, _ = sch
worst_cos = 1.0
for _ in range(300):
    t = rng.uniform(0.1, 0.9)
    z = rng.normal(size=3)
    eps = rng.normal(size=3)
    x = a(t) * z + s(t) * eps
    v = a(t) * eps - s(t) * z
    u = u_cond(x, z, t, sch)
    c = abs(np.dot(u, v)) / (np.linalg.norm(u) * np.linalg.norm(v))
    worst_cos = min(worst_cos, c)
print(f"  [INFO] CondOT path: smallest |cos(angle(u,v))| over 300 draws = {worst_cos:.3f}")
print("         (a scalar multiple would force |cos| == 1 for every draw)")
report("CondOT path: u and v are NOT parallel", 0.0 if worst_cos < 0.99 else 1.0, 0.5)

# ...except at the single time t = 1/2, where 2t-1 = 0 and v = -u/2 exactly.
res = 0.0
for _ in range(200):
    z = rng.normal(size=3)
    eps = rng.normal(size=3)
    t = 0.5
    x = a(t) * z + s(t) * eps
    v = a(t) * eps - s(t) * z
    res = max(res, np.max(np.abs(u_cond(x, z, t, sch) + 2.0 * v)))
report("CondOT path: u == -2v exactly at t = 1/2 (the one exception)", res, 1e-13)


# ----------------------------------------------------------------------------
section("9.  Interconvertibility determinant and the SNR condition")
# ----------------------------------------------------------------------------
# Claim: alpha*sigma_dot - sigma*alpha_dot = -sigma^2 * d/dt (alpha/sigma),
# so the (z, eps) -> (x, target) map is invertible iff the SNR alpha/sigma is
# strictly monotone.

for name, sch in SCHEDULES.items():
    a, s, ad, sd = sch
    res = 0.0
    for _ in range(200):
        t = rng.uniform(0.02, 0.98)
        lhs = a(t) * sd(t) - s(t) * ad(t)
        snr = lambda tt: a(tt) / s(tt)
        dsnr = (snr(t + h) - snr(t - h)) / (2 * h)
        rhs = -s(t) ** 2 * dsnr
        res = max(res, abs(lhs - rhs) / max(abs(lhs), 1e-9))
    report(f"det = -sigma^2 d/dt(alpha/sigma)  [{name}]", res, 1e-5, "relative")


# ----------------------------------------------------------------------------
section("10.  Non-uniqueness of the vector field for a given probability path")
# ----------------------------------------------------------------------------
# p_t == N(0, I_2) constant in t.  Both u = 0 and u(x) = (-x_2, x_1) satisfy
# the continuity equation, but generate completely different trajectories.

hx = 1e-5
res = 0.0
for _ in range(200):
    x = rng.normal(size=2) * 1.5
    p = lambda y: np.exp(-np.sum(y ** 2) / 2) / (2 * np.pi)
    u_rot = lambda y: np.array([-y[1], y[0]])
    div = 0.0
    for i in range(2):
        e = np.zeros(2); e[i] = hx
        div += (p(x + e) * u_rot(x + e)[i] - p(x - e) * u_rot(x - e)[i]) / (2 * hx)
    res = max(res, abs(div))
report("div(p * rotation field) == 0 for p = N(0,I_2)", res, 1e-9,
       "so u=0 and u=rotation generate the same path")


# ----------------------------------------------------------------------------
section("11.  L2 projection identity and the irreducible CFM loss floor")
# ----------------------------------------------------------------------------
# E||f(X)-Y||^2 = E||f(X)-E[Y|X]||^2 + E||Y-E[Y|X]||^2, and hence
# L_CFM(theta*) = E||u_t(X_t|Z) - u_t(X_t)||^2 > 0 while L_FM(theta*) = 0.

sch = sched_condot()
a, s, _, _ = sch
N = 2_000_000
tt = rng.uniform(0.0, 1.0, size=N)
idx = rng.choice(len(W), size=N, p=W)
eps = rng.normal(size=(N, D))
Xt = a(tt)[:, None] * Z[idx] + s(tt)[:, None] * eps

targ = np.empty((N, D))
marg = np.empty((N, D))
for k in range(0, N, 250_000):
    sl = slice(k, min(k + 250_000, N))
    targ[sl] = u_cond(Xt[sl], Z[idx][sl], tt[sl][:, None], sch)
    # u_marg vectorised over a batch with per-sample t
    aa = a(tt[sl])[:, None]; ss = s(tt[sl])[:, None]
    comp = np.stack([np.exp(-np.sum((Xt[sl] - aa * Z[k2]) ** 2, axis=1) / (2 * ss[:, 0] ** 2))
                     / (2 * np.pi * ss[:, 0] ** 2) * W[k2] for k2 in range(len(W))], axis=-1)
    wgt = comp / np.sum(comp, axis=-1, keepdims=True)
    flds = np.stack([u_cond(Xt[sl], Z[k2], tt[sl][:, None], sch) for k2 in range(len(W))], axis=-2)
    marg[sl] = np.sum(wgt[..., None] * flds, axis=-2)

# a deliberately imperfect "network": the marginal field plus a fixed bias
bias = np.array([0.17, -0.23])
f_theta = marg + bias

lhs = np.mean(np.sum((f_theta - targ) ** 2, axis=1))
rhs = np.mean(np.sum((f_theta - marg) ** 2, axis=1)) + np.mean(np.sum((targ - marg) ** 2, axis=1))
report("Pythagoras: E||f-Y||^2 = E||f-E[Y|X]||^2 + E||Y-E[Y|X]||^2",
       abs(lhs - rhs) / abs(lhs), 3e-3, "Monte Carlo, N=2e6, relative")

floor = np.mean(np.sum((targ - marg) ** 2, axis=1))
l_cfm_star = np.mean(np.sum((marg - targ) ** 2, axis=1))
l_fm_star = np.mean(np.sum((marg - marg) ** 2, axis=1))
print(f"  [INFO] L_CFM at the optimum  = {l_cfm_star:.4f}   (irreducible floor = {floor:.4f})")
print(f"  [INFO] L_FM  at the optimum  = {l_fm_star:.2e}")
print("         => a non-zero CFM training loss is NOT evidence of underfitting.")
report("L_CFM(theta*) == conditional-variance floor > 0",
       abs(l_cfm_star - floor), 1e-12)


# ----------------------------------------------------------------------------
section("SUMMARY")
# ----------------------------------------------------------------------------
if FAILURES:
    print(f"  {len(FAILURES)} CHECK(S) FAILED:")
    for f in FAILURES:
        print(f"    - {f}")
    raise SystemExit(1)
print("  All checks passed.")
