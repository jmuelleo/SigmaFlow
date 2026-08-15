"""Diffusion gegen Flow Matching: Targets, Skalen, Quellverteilungen.

Prueft numerisch, was der Code als Zielgroesse konstruiert, ob die
Flow-Matching-Ziele wirklich die Ableitung ihres eigenen Pfades sind, und
ob die Verlustterme fuer Translation und Rotation vergleichbar skaliert sind.

Nichts hier braucht eine GPU oder einen Checkpoint.

    python audits/engine_swap_audit.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

MIN = Path(__file__).resolve().parents[1] / "SigmaFlow_Minimal"
sys.path.insert(0, str(MIN / "src"))
torch.set_default_dtype(torch.float64)

from sigmadock.diff import so3_utils                              # noqa: E402
from sigmadock.diff.r3_flow_matcher import R3_FlowMatcher          # noqa: E402
from sigmadock.diff.se3_flow_matcher import SE3_FlowMatcher        # noqa: E402
from sigmadock.diff.so3_flow_matcher import SO3_FlowMatcher        # noqa: E402


def rel(a, b):
    return (a - b).abs().max().item() / max(a.abs().max().item(),
                                            b.abs().max().item(), 1e-30)


def rand_rot(g, n):
    A = torch.randn(n, 3, 3, generator=g)
    Q, R = torch.linalg.qr(A)
    Q = Q * torch.sign(torch.diagonal(R, dim1=-2, dim2=-1)).unsqueeze(-2)
    flip = torch.det(Q) < 0
    Q[flip, :, 0] = -Q[flip, :, 0]
    return Q


# ====================================================================== 1
def targets_are_path_derivatives():
    print("=" * 74)
    print("1. SIND DIE FM-ZIELE WIRKLICH DIE ABLEITUNG IHRES PFADES?")
    print("   Analytisches u_t gegen zentralen Differenzenquotienten.")
    print("=" * 74)
    g = torch.Generator().manual_seed(3)
    n = 64
    x_1 = torch.randn(n, 3, generator=g)
    x_0 = torch.randn(n, 3, generator=g)
    R_1 = rand_rot(g, n)
    R_0 = rand_rot(g, n)
    L = so3_utils.log(R_0.transpose(-1, -2) @ R_1)          # [n,3,3]
    eps = 1e-6

    print(f"  {'t':>6} {'Translation':>14} {'Rotation':>14}")
    for t in (0.01, 0.25, 0.5, 0.75, 0.99):
        tt = torch.full((n,), t)
        # --- Translation: x_t = (1-t)x_0 + t x_1,  u_t = x_1 - x_0
        xp = (1 - (t + eps)) * x_0 + (t + eps) * x_1
        xm = (1 - (t - eps)) * x_0 + (t - eps) * x_1
        d_tr = rel((xp - xm) / (2 * eps), x_1 - x_0)
        # --- Rotation: R_t = R_0 exp(t L). Koerperkoordinaten R_t^T Rdot_t.
        Rp = R_0 @ so3_utils.exp((t + eps) * L)
        Rm = R_0 @ so3_utils.exp((t - eps) * L)
        Rt = R_0 @ so3_utils.exp(t * L)
        Rdot = (Rp - Rm) / (2 * eps)
        omega_fd = Rt.transpose(-1, -2) @ Rdot
        d_rot = rel(omega_fd, L)
        print(f"  {t:>6.2f} {d_tr:>14.2e} {d_rot:>14.2e}")

    print("\n  Beide Ziele sind zeitkonstant. Das ist kein Zufall:")
    print("  fuer die Geodaete ist R_t^T Rdot_t = log(R_0^T R_1) fuer alle t.")

    # Die zweite, im Code ebenfalls vorhandene Form muss dasselbe liefern.
    print("\n  Zwei Parametrisierungen desselben Ziels im Code:")
    fm = SE3_FlowMatcher(0.0)
    for t in (0.1, 0.5, 0.9):
        tt = torch.full((n,), t)
        x_t = (1 - t) * x_0 + t * x_1
        R_t = R_0 @ so3_utils.exp(t * L)
        v = fm.calc_vector_field(x_t, R_t, x_1, R_1, tt)
        print(f"    t={t:.1f}  (x_1-x_t)/(1-t) vs x_1-x_0 : "
              f"{rel(v['u_t_trans'], x_1 - x_0):.2e}   "
              f"log(R_t^T R_1)/(1-t) vs log(R_0^T R_1) : "
              f"{rel(v['u_t_R'], L):.2e}")


# ====================================================================== 2
def endpoints():
    print()
    print("=" * 74)
    print("2. ENDPUNKTE UND SINGULARITAETEN")
    print("=" * 74)
    g = torch.Generator().manual_seed(5)
    n = 32
    x_1 = torch.randn(n, 3, generator=g)
    x_0 = torch.randn(n, 3, generator=g)
    R_1 = rand_rot(g, n)
    R_0 = rand_rot(g, n)
    L = so3_utils.log(R_0.transpose(-1, -2) @ R_1)
    for t in (0.0, 1.0):
        x_t = (1 - t) * x_0 + t * x_1
        R_t = R_0 @ so3_utils.exp(t * L)
        tgt_x, tgt_R = (x_0, R_0) if t == 0.0 else (x_1, R_1)
        print(f"  t={t:.0f}:  x_t vs x_{int(t)} = {rel(x_t, tgt_x):.2e}   "
              f"R_t vs R_{int(t)} = {rel(R_t, tgt_R):.2e}")

    print("\n  Die im Code benutzte Zielform log(R_0^T R_1) bzw. x_1 - x_0 hat")
    print("  bei t=1 KEINE Singularitaet. Die Alternativform mit 1/(1-t) hat")
    print("  eine; sie wird im Training nicht benutzt, nur in sampling.py als")
    print("  Diagnose. epsilon_t = 0.01 begrenzt nur die untere Seite.")


# ====================================================================== 3
def source_distributions():
    print()
    print("=" * 74)
    print("3. QUELLVERTEILUNGEN: SigmaFlow gegen SigmaDocks Terminalverteilung")
    print("=" * 74)
    # --- Translation
    min_beta, max_beta = 0.1, 20.0
    int_beta_1 = min_beta + 0.5 * (max_beta - min_beta)
    alpha_1 = np.exp(-int_beta_1 / 2)
    var_1 = 1 - alpha_1 ** 2
    print("  Translation")
    print(f"    SigmaDock VP-SDE bei t=1: N(alpha_1 x_0, (1-alpha_1^2) I)")
    print(f"      alpha_1 = {alpha_1:.3e}   Varianz = {var_1:.10f}")
    print(f"    SigmaFlow Quelle:         N(0, I)")
    print(f"    -> Abweichung in der Standardabweichung: "
          f"{abs(1 - np.sqrt(var_1)):.2e}. Praktisch identisch.")

    # --- Rotation: IGSO(3) bei sigma_max = 1.5 gegen Haar
    def igso3_angle_density(om, sigma, L=500):
        l = np.arange(L)[:, None]
        f = ((2 * l + 1) * np.exp(-l * (l + 1) * sigma ** 2 / 2)
             * np.sin((l + 0.5) * om[None, :]) / np.sin(om[None, :] / 2)).sum(0)
        return f * (1 - np.cos(om)) / np.pi

    om = np.linspace(1e-6, np.pi, 4000)
    haar = (1 - np.cos(om)) / np.pi
    haar /= np.trapezoid(haar, om)
    ig = igso3_angle_density(om, 1.5)
    ig /= np.trapezoid(ig, om)
    tv = 0.5 * np.trapezoid(np.abs(ig - haar), om)
    print("\n  Rotation")
    print(f"    SigmaDock IGSO(3), sigma_max = 1.5")
    print(f"      Mittelwert Winkel = {np.degrees(np.trapezoid(om*ig, om)):.1f} Grad")
    print(f"    SigmaFlow Haar (uniform ueber SO(3))")
    print(f"      Mittelwert Winkel = {np.degrees(np.trapezoid(om*haar, om)):.1f} Grad")
    print(f"    -> Totalvariationsabstand = {tv:.4f}")
    print("    Nicht identisch. IGSO(3) bei sigma=1.5 ist noch nicht uniform;")
    print("    es fehlt Masse bei grossen Winkeln. Das ist ein echter, wenn")
    print("    auch kleiner Unterschied in der Startverteilung.")


# ====================================================================== 4
def loss_scales():
    print()
    print("=" * 74)
    print("4. SKALEN DER BEIDEN VERLUSTTERME")
    print("   SigmaFlow gewichtet 1.0 * loss_trans + 0.5 * loss_R.")
    print("   Diese Gewichte stammen aus SigmaDock, wo zusaetzlich")
    print("   lambda = 1/score_scaling^2 normierte. Diese Normierung fehlt.")
    print("=" * 74)
    g = torch.Generator().manual_seed(9)
    n = 20000
    # trans_1 sind Fragment-COMs in normierten Einheiten (durch 2.7 A geteilt).
    # Typische Groesse aus dem echten Datensatz waere besser; hier eine
    # konservative Bandbreite.
    for com_std in (0.5, 1.0, 2.0):
        x_1 = com_std * torch.randn(n, 3, generator=g)
        x_0 = torch.randn(n, 3, generator=g)
        u_tr = x_1 - x_0
        l_tr = (u_tr ** 2).sum(-1)
        R_1 = torch.eye(3).expand(n, 3, 3)
        R_0 = rand_rot(g, n)
        u_R = so3_utils.log(R_0.transpose(-1, -2) @ R_1)
        l_R = (u_R ** 2).sum(dim=(-1, -2))
        print(f"  COM-Streuung {com_std:.1f} (normiert):")
        print(f"    E||u_trans||^2      = {l_tr.mean():8.3f}")
        print(f"    E||u_R||_F^2        = {l_R.mean():8.3f}")
        print(f"    gewichtet (1 : 0.5) = {l_tr.mean():8.3f} : "
              f"{0.5*l_R.mean():.3f}   Verhaeltnis "
              f"{l_tr.mean()/(0.5*l_R.mean()):.2f}")
    tr = rand_rot(g, n).diagonal(dim1=-2, dim2=-1).sum(-1)
    theta = torch.arccos(((tr - 1) / 2).clamp(-1, 1))
    print(f"\n  ||log R||_F^2 = 2*theta^2 mit theta ~ Haar; "
          f"E[theta] = {np.degrees(theta.mean().item()):.1f} Grad "
          f"(analytisch 126.5)")
    print("  Die Rotationsgroesse ist also weitgehend datenunabhaengig, die")
    print("  Translationsgroesse haengt an dimensional_scale = 2.7 A.")


# ====================================================================== 5
def euler_consistency():
    print()
    print("=" * 74)
    print("5. INTEGRATOR: implementiert er die hergeleitete ODE?")
    print("=" * 74)
    g = torch.Generator().manual_seed(13)
    n = 16
    x_1 = torch.randn(n, 3, generator=g)
    x_0 = torch.randn(n, 3, generator=g)
    R_1 = rand_rot(g, n)
    R_0 = rand_rot(g, n)
    L = so3_utils.log(R_0.transpose(-1, -2) @ R_1)
    fm = SE3_FlowMatcher(0.0)

    # Mit dem EXAKTEN Vektorfeld muss ein Euler-Schritt entlang der Geodaete
    # exakt landen - fuer Rotation, weil exp(a L) exp(b L) = exp((a+b) L),
    # fuer Translation, weil der Pfad linear ist.
    for dt in (0.5, 0.25, 0.1):
        t = 0.0
        x_t, R_t = x_0.clone(), R_0.clone()
        steps = int(round(1.0 / dt))
        for _ in range(steps):
            s = fm.euler_step(x_t, R_t, x_1 - x_0, L, dt)
            x_t, R_t = s["trans_new"], s["R_new"]
        print(f"  dt={dt:.2f} ({steps} Schritte):  x_end vs x_1 = "
              f"{rel(x_t, x_1):.2e}   R_end vs R_1 = {rel(R_t, R_1):.2e}")
    print("  Exakt bei jeder Schrittweite. Der Integrator implementiert genau")
    print("  x + dt*v bzw. R @ exp(dt*v) und nichts sonst; keine geerbten")
    print("  sigma- oder Rauschfaktoren.")


# ====================================================================== 6
def dead_sigma_min():
    print()
    print("=" * 74)
    print("6. sigma_min: tot oder lebendig?")
    print("=" * 74)
    r3 = R3_FlowMatcher(sigma_min=999.0)
    g = torch.Generator().manual_seed(1)
    torch.manual_seed(0)
    a = r3.conditional_probability_path(torch.randn(8, 3, generator=g),
                                        torch.full((8,), 0.5))
    r3b = R3_FlowMatcher(sigma_min=0.0)
    torch.manual_seed(0)
    b = r3b.conditional_probability_path(torch.randn(8, 3, generator=g),
                                         torch.full((8,), 0.5))
    print(f"  sigma_min=999 gegen sigma_min=0, gleicher Seed: "
          f"x_t {rel(a[0], b[0]):.2e}, u_t {rel(a[1], b[1]):.2e}")
    print("  SO3_FlowMatcher.__init__ nimmt sigma_min gar nicht erst entgegen.")
    print("  -> Der Wert ist im R3-Pfad gespeichert und wird nirgends gelesen.")


# ====================================================================== 7
def log_map_precision():
    print()
    print("=" * 74)
    print("7. GENAUIGKEIT DER LOG-ABBILDUNG")
    print("   Das FM-Rotationsziel IST log(R_0^T R_1). Seine Genauigkeit ist")
    print("   damit direkt die Genauigkeit des Ziels.")
    print("=" * 74)
    g = torch.Generator().manual_seed(2)
    ax = torch.randn(4000, 3, generator=g)
    ax = ax / ax.norm(dim=-1, keepdim=True)
    print(f"  {'Winkel':>9} {'|exp(log R)-R|':>16} {'|Omega-theta|':>15}")
    for deg in (0.01, 1, 45, 90, 135, 170, 178, 179, 179.9):
        th = np.radians(deg)
        R = so3_utils.exp(so3_utils.hat(ax * th))
        err = (so3_utils.exp(so3_utils.log(R)) - R).abs().amax(dim=(-1, -2))
        om = so3_utils.Omega(R)
        print(f"  {deg:>9} {err.median():>16.2e} {(om-th).abs().median():>15.2e}")

    print("\n  Zwei getrennte Ursachen, beide aus SigmaDock geerbt:")
    print("  (a) Omega() multipliziert die Spur mit (1 - 1e-6), bevor arccos")
    print("      angewandt wird. Das verzerrt JEDEN Winkel systematisch, nicht")
    print("      nur die Randfaelle: bei 90 Grad exakt 5.0e-07 rad.")
    print("  (b) rotation_vector_from_matrix schaltet erst bei")
    print("      |theta - pi| < 1e-2 auf den pi-Zweig um. Dazwischen, also bei")
    print("      etwa 178 bis 179.4 Grad, rechnet es mit theta/(2 sin theta)")
    print("      und sin(theta) ist dort schon fast null.")

    g2 = torch.Generator().manual_seed(1)
    R = rand_rot(g2, 40000)
    err = (so3_utils.exp(so3_utils.log(R)) - R).abs().amax(dim=(-1, -2))
    q = torch.quantile(err, torch.tensor([0.5, 0.99, 0.999]))
    print(f"\n  Ueber 40000 Haar-Rotationen (= die Quellverteilung von")
    print(f"  SigmaFlow, da R_1 = I): Median {q[0]:.1e}, "
          f"99 % {q[1]:.1e}, 99.9 % {q[2]:.1e}, Max {err.max():.1e}")
    print("  Bezogen auf eine Zielgroesse von etwa 2 rad sind das relativ")
    print("  2e-7 im Median und rund 0.5 % im Extremfall.")
    print("\n  Einordnung: real, systematisch, geerbt - aber um Groessen-")
    print("  ordnungen zu klein, um den Rotationsausfall zu erklaeren")
    print("  (138 Grad gegen 132 Grad Zufall ist ein O(1)-Effekt).")
    print("  Die Notiz in so3_flow_matcher.py, der Omega-Clamp sei 'fixed',")
    print("  ist zu stark: der grobe Fehler wurde beseitigt, exakt ist es nicht.")


if __name__ == "__main__":
    targets_are_path_derivatives()
    endpoints()
    source_distributions()
    loss_scales()
    euler_consistency()
    log_map_precision()
    dead_sigma_min()
    print()
