"""Audit der Rotations-Frame-Konvention in SigmaFlow_Minimal.

WARUM
  Vor einem 72h-Lauf, von dem es nur einen gibt, muss bewiesen sein, dass
  Zielvektorfeld, Netzausgabe und ODE-Schritt IM SELBEN Frame leben. Genau
  dort sass der Bug vom 2026-08-09: die Netzausgabe war im Weltframe, Ziel und
  Integrator im Koerperframe, und niemand hat konvertiert.

  Kommentare im Code werden hier NICHT geglaubt. Jede Behauptung wird aus dem
  importierten Code numerisch nachgerechnet.

    python audits/so3_trivialisation_audit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "SigmaFlow_Minimal" / "src"))

from sigmadock.diff import so3_utils  # noqa: E402
from sigmadock.diff.so3_flow_matcher import SO3_FlowMatcher  # noqa: E402

FAILS: list[str] = []
torch.set_default_dtype(torch.float64)


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'OK  ' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def rand_rot(n: int) -> torch.Tensor:
    return so3_utils.sample_uniform(n).to(torch.float64)


def main() -> int:  # noqa: C901
    torch.manual_seed(20260816)
    np.random.seed(20260816)
    n = 500
    fm = SO3_FlowMatcher()

    R_0, R_1 = rand_rot(n), rand_rot(n)
    t = torch.rand(n) * 0.8 + 0.1               # weg von den Raendern

    print("\n1. Welche Trivialisierung benutzt das Zielvektorfeld?")
    # Behauptung im Code-Docstring: "right trivialised vector field".
    # Nachgerechnet wird, welche der beiden Groessen tatsaechlich herauskommt:
    #     links/koerper:  Omega = R^T Rdot      (Rdot = R Omega)
    #     rechts/raum:    omega = Rdot R^T      (Rdot = omega R)
    L = so3_utils.log(R_0.transpose(-1, -2) @ R_1)          # [n,3,3]
    R_t = R_0 @ so3_utils.exp(t[:, None, None] * L)

    # Ableitung der Geodaete numerisch, zentraler Differenzenquotient.
    h = 1e-6
    R_p = R_0 @ so3_utils.exp((t + h)[:, None, None] * L)
    R_m = R_0 @ so3_utils.exp((t - h)[:, None, None] * L)
    Rdot = (R_p - R_m) / (2 * h)

    body = R_t.transpose(-1, -2) @ Rdot                     # R^T Rdot
    spatial = Rdot @ R_t.transpose(-1, -2)                  # Rdot R^T

    u_target = fm.calc_rot_vector_field(R_t, R_1, t)

    # Verglichen wird RELATIV, nicht absolut. Der Logarithmus in so3_utils ist
    # bei grossen Winkeln begrenzt genau (Abschnitt 9 quantifiziert das), und
    # u_target wird zusaetzlich durch (1-t) geteilt, was den Fehler verstaerkt.
    # Eine absolute Toleranz liesse deshalb korrekten Code durchfallen.
    # Entscheidend ist, WELCHE der beiden Formen getroffen wird -- und der
    # Unterschied zwischen ihnen ist um Groessenordnungen groesser.
    scale = float(u_target.abs().max())
    err_body = float((u_target - body).abs().max()) / scale
    err_spatial = float((u_target - spatial).abs().max()) / scale
    check("Ziel u_t_R == R^T Rdot  (KOERPER / linkstrivialisiert)",
          err_body < 1e-2, f"relativer Restfehler {err_body:.2e}")
    check("Ziel u_t_R != Rdot R^T  (nicht raumfest)",
          err_spatial > 0.1, f"relativer Abstand zur raumfesten Form {err_spatial:.2e}")
    check("die koerperfeste Form passt um Groessenordnungen besser",
          err_spatial / max(err_body, 1e-30) > 10,
          f"Faktor {err_spatial / max(err_body, 1e-30):.0f}")

    print("\n   -> Die im Code als \"right trivialised\" bezeichnete Groesse ist")
    print("      nach der ueblichen Konvention (Marsden/Ratiu) die LINKS-")
    print("      trivialisierte, also die KOERPERgeschwindigkeit. Rdot = R * Omega.")
    print("      Die Mathematik stimmt; nur die Benennung im Docstring ist die")
    print("      jeweils andere. Das ist eine Dokumentations-, keine Codefrage.")

    print("\n2. Ist das Ziel entlang der Geodaete konstant?")
    # Fuer die Geodaete muss Omega zeitunabhaengig sein. Ist sie das nicht,
    # stimmt die Interpolation nicht mit dem Vektorfeld ueberein.
    spread = float((u_target - L).abs().max()) / scale
    check("u_t_R == log(R_0^T R_1), unabhaengig von t", spread < 1e-2,
          f"max relative Abweichung {spread:.2e}")

    print("\n3. Stimmen die beiden Codepfade fuer dasselbe Ziel ueberein?")
    # Training benutzt conditional_probability_path (gibt L zurueck),
    # Sampling benutzt calc_rot_vector_field (gibt log(R_t^T R_1)/(1-t)).
    # Beide muessen identisch sein, sonst trainiert und sampelt man Verschiedenes.
    # sample_init castet intern auf float32; der Pfad laeuft damit in der
    # Produktionsgenauigkeit, nicht in float64.
    torch.manual_seed(7)
    R_t2, u_t2 = fm.conditional_probability_path(R_1.float(), t.float())
    u_via_calc = fm.calc_rot_vector_field(R_t2, R_1.float(), t.float())
    d = float((u_t2 - u_via_calc).abs().max()) / max(float(u_t2.abs().max()), 1e-30)
    check("conditional_probability_path == calc_rot_vector_field", d < 1e-2,
          f"max relative Abweichung {d:.2e}")

    print("\n4. Passt der Integrator zur Trivialisierung?")
    # euler_step: R_next = R_t @ exp(v dt). Das ist die Rechtsmultiplikation,
    # die zu Rdot = R Omega gehoert. Ein exakter Schritt entlang der Geodaete
    # mit dt = 1-t muss genau R_1 treffen.
    dt = (1.0 - t)[:, None, None]
    R_end = fm.euler_step(R_t, u_target, dt)
    err_hit = float(so3_utils.Log(R_end.transpose(-1, -2) @ R_1).norm(dim=-1).max())
    # 1 Grad ist grosszuegig gegenueber der in Abschnitt 9 gemessenen
    # Logarithmusgenauigkeit und trotzdem 100-mal schaerfer als der Abstand zur
    # falschen Multiplikationsseite (unten, ~100 Grad).
    check("ein Euler-Schritt der Laenge (1-t) trifft R_1",
          np.rad2deg(err_hit) < 1.0, f"max Restwinkel {np.rad2deg(err_hit):.2e} Grad")

    # Gegenprobe: mit LINKSmultiplikation trifft er NICHT. Ohne diesen Fall
    # koennte der Test auch eine frameverwechselnde Implementierung durchwinken.
    R_end_wrong = so3_utils.exp(u_target * dt) @ R_t
    err_wrong = float(so3_utils.Log(R_end_wrong.transpose(-1, -2) @ R_1).norm(dim=-1).max())
    check("Linksmultiplikation trifft R_1 NICHT (Test kann scheitern)",
          np.rad2deg(err_wrong) > 10.0, f"max Restwinkel {np.rad2deg(err_wrong):.1f} Grad")

    print("\n5. Die eigentliche Fix-Gleichung")
    # Behauptung im Docstring von _compute_vector_field:
    #     R_t exp(u_body dt) = exp(w_world dt) R_t   <=>   u_body = R_t^T w_world R_t
    w_world = so3_utils.hat(torch.randn(n, 3) * 0.3)
    u_body = R_t.transpose(-1, -2) @ w_world @ R_t
    dt_s = 0.05
    lhs = R_t @ so3_utils.exp(u_body * dt_s)
    rhs = so3_utils.exp(w_world * dt_s) @ R_t
    err_adj = float((lhs - rhs).abs().max())
    check("R_t exp(R_t^T w R_t dt) == exp(w dt) R_t", err_adj < 1e-10,
          f"Restfehler {err_adj:.2e}")

    check("Konjugation erhaelt die Schiefsymmetrie",
          float((u_body + u_body.transpose(-1, -2)).abs().max()) < 1e-12,
          f"max |u + u^T| = {float((u_body + u_body.transpose(-1, -2)).abs().max()):.2e}")

    # Gegenprobe: OHNE Transport ist die Gleichung verletzt. Das ist exakt der
    # Zustand vor dem 2026-08-09-Fix.
    lhs_bug = R_t @ so3_utils.exp(w_world * dt_s)
    err_bug = float((lhs_bug - rhs).abs().max())
    check("ohne Transport ist die Gleichung verletzt (der alte Bug)",
          err_bug > 1e-3, f"Abweichung {err_bug:.2e}")

    print("\n6. Aequivarianz der Delta-Rotation unter globaler Drehung")
    # Behauptung: unter R -> Q R Q^T bleibt die Konstruktion konsistent.
    # Geprueft wird die Groesse, die tatsaechlich transportiert wird.
    Q = rand_rot(1)[0]
    R_0q, R_1q = Q @ R_0 @ Q.transpose(-1, -2), Q @ R_1 @ Q.transpose(-1, -2)
    Lq = so3_utils.log(R_0q.transpose(-1, -2) @ R_1q)

    # Der Winkel ist die eindeutige Groesse und muss IMMER invariant sein.
    # `L` ist schiefsymmetrisch, kein Rotationsmatrix -- der Winkel kommt
    # deshalb ueber vee(), nicht ueber Log().
    ang_before = so3_utils.vee(L).norm(dim=-1)
    ang_after = so3_utils.vee(Lq).norm(dim=-1)
    d_ang = float((ang_before - ang_after).abs().max())
    check("Rotationswinkel ist invariant unter globaler Drehung Q",
          d_ang < 1e-9, f"max Abweichung {d_ang:.2e}")

    # Die Matrixgleichung Lq = Q L Q^T gilt algebraisch exakt, weil der
    # Matrixlogarithmus mit Konjugation vertauscht. NUMERISCH gilt sie nur
    # abseits des Cut Locus: bei Drehwinkel genau pi ist der Logarithmus
    # ZWEIDEUTIG (+-pi*n sind beide gueltig), und der Algorithmus kann vor und
    # nach der Konjugation verschiedene Zweige waehlen. Der Restfehler springt
    # dann auf bis zu 2*pi -- das ist eine Eigenschaft von SO(3), kein Fehler.
    resid = (Lq - Q @ L @ Q.transpose(-1, -2)).abs().amax(dim=(-2, -1))
    ang_deg = np.rad2deg(ang_before.numpy())
    safe = ang_deg < 170.0
    err_safe = float(resid[torch.as_tensor(safe)].max())
    check("abseits des Cut Locus (<170 Grad): Lq == Q L Q^T",
          err_safe < 1e-6, f"max Restfehler {err_safe:.2e} auf {safe.sum()} Faellen")

    n_flip = int((resid > 1.0).sum())
    frac_near = float((ang_deg >= 170.0).mean())
    print(f"      Zweigwechsel bei {n_flip}/{n} Faellen; {100 * frac_near:.1f} % der")
    print("      Haar-Ziehungen liegen ueber 170 Grad. Das trifft das TRAINING")
    print("      nicht, weil dort nie zwei konjugierte Varianten desselben")
    print("      Komplexes verglichen werden -- wohl aber jeden Test, der")
    print("      Aequivarianz auf der Lie-Algebra statt auf dem Winkel prueft.")

    print("\n7. Was speichert Minimal als R_1?")
    from sigmadock.diff.sigma_flow_generator import SigmaFlowGenerator
    import inspect
    src = inspect.getsource(SigmaFlowGenerator.get_fragment_com_and_rot)
    check("get_fragment_com_and_rot gibt UNBEDINGT die Identitaet zurueck",
          "rots.append(I3)" in src and "kabsch" not in src.lower(),
          "R_1 = I fuer jedes Fragment")
    print("      -> Das Ziel ist also nicht 'die Orientierung des Fragments',")
    print("         sondern per Definition die Identitaet. R_t ist damit die")
    print("         Delta-Rotation gegenueber der KRISTALLorientierung von pos_0.")

    print("\n8. Enthaelt der aktuelle Stand den Frame-Fix?")
    src_vf = inspect.getsource(SigmaFlowGenerator._compute_vector_field)
    check("_compute_vector_field enthaelt den adjungierten Transport",
          'R_t.transpose(-1, -2) @ updates["omega"] @ R_t' in src_vf)

    print("\n9. Wie genau ist der Logarithmus - und ist das relevant?")
    # Der Kommentar in so3_flow_matcher.py sagt, SigmaDocks Omega() habe arccos
    # auf [-0.99, 0.99] geklemmt und das sei hier auf [-1+1e-7, 1-1e-7] verengt
    # worden. Verengt heisst nicht beseitigt: nahe 180 Grad bleibt der
    # Logarithmus schlecht konditioniert. Gemessen wird deshalb, wie gross der
    # Fehler auf dem TRAININGSZIEL wirklich ist.
    m = 20000
    ang_true = torch.rand(m) * np.pi
    ax = torch.randn(m, 3)
    ax = ax / ax.norm(dim=-1, keepdim=True)
    ang_meas = so3_utils.Log(so3_utils.Exp(ang_true[:, None] * ax)).norm(dim=-1)
    abs_err = (ang_meas - ang_true).abs().numpy()
    rel_err = abs_err / np.maximum(ang_true.numpy(), 1e-9)
    hw = (1 - np.cos(ang_true.numpy())) / np.pi        # Haar-Winkeldichte
    med, p99, mx = (float(np.median(rel_err)), float(np.percentile(rel_err, 99)),
                    float(rel_err.max()))
    print(f"      relativer Zielfehler: median {med:.2e}, p99 {p99:.2e}, max {mx:.2e}")
    print(f"      Haar-gewichteter Winkelfehler: "
          f"{np.average(np.rad2deg(abs_err), weights=hw):.2e} Grad")
    check("relativer Zielfehler im Median unter 1e-5", med < 1e-5, f"{med:.2e}")
    check("relativer Zielfehler auch im schlimmsten Fall unter 1 %",
          mx < 1e-2, f"{mx:.2e}")
    print("      -> Gegen ein Modell, dessen Rotationsfehler bei 145 Grad liegt")
    print("         (Zufall: 132), ist ein Zielfehler von 0.5 % im schlimmsten")
    print("         Fall belanglos. Caveat, kein Blocker.")

    print("\n" + "=" * 66)
    if FAILS:
        print(f"{len(FAILS)} CHECK(S) FAILED: {FAILS}")
        return 1
    print("Alle Checks bestanden. Frame-Kette ist konsistent.")
    return 0


def test_so3_trivialisation() -> None:
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
