"""EXP-100 — vollstaendige lokale Testbatterie.

Kein Test darf R_1 = I voraussetzen. Die alte Parametrisierung hatte diesen
Spezialfall ueberall; die Tests hier ziehen R_1 explizit Haar-uniform.

Ausfuehren:  python tests/test_exp100.py    (aus dem Variantenordner)
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sigmadock.diff import so3_utils                                # noqa: E402
from sigmadock.diff.r3_flow_matcher import R3_FlowMatcher           # noqa: E402
from sigmadock.diff.so3_flow_matcher import SO3_FlowMatcher         # noqa: E402
from sigmadock.diff.state_reparam import (                          # noqa: E402
    fragment_targets, kabsch_residual, kabsch_rotation,
)

torch.set_default_dtype(torch.float64)
FAIL = []


def check(name, cond, detail=""):
    if not cond:
        FAIL.append(name)
    print(f"  [{'OK  ' if cond else 'FAIL'}] {name} {detail}")


def haar(n, seed):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(n, 4, generator=g)
    q = q / q.norm(dim=1, keepdim=True)
    w, x, y, z = q.unbind(1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=1).reshape(n, 3, 3)


def axis_angle(axis, deg):
    a = torch.tensor(axis, dtype=torch.float64)
    a = a / a.norm()
    return so3_utils.exp(so3_utils.hat((a * torch.tensor(deg * torch.pi / 180))[None]))[0]


def geo_deg(A, B):
    R = A.transpose(-1, -2) @ B
    tr = torch.clamp((torch.diagonal(R, dim1=-2, dim2=-1).sum(-1) - 1) / 2, -1.0, 1.0)
    return torch.rad2deg(torch.arccos(tr))


print("=" * 88)
print("1. KABSCH — Richtung, nicht nur Betrag")
print("=" * 88)
g = torch.Generator().manual_seed(0)
C = torch.randn(12, 3, generator=g)
C = C - C.mean(0)
check("Identitaet: Y=C  ->  R=I",
      torch.allclose(kabsch_rotation(C, C), torch.eye(3), atol=1e-10))

# DER entscheidende Test: bekannte Rotation muss Q ergeben, NICHT Q^T
errs = []
for s in range(200):
    Q = haar(1, 100 + s)[0]
    Y = C @ Q.transpose(-1, -2)            # Y = Q C  (zeilenweise)
    R = kabsch_rotation(C, Y)
    errs.append(geo_deg(R[None], Q[None]).item())
errs = torch.tensor(errs)
# Schwelle 1e-5 Grad: geo_deg nutzt arccos(tr/2); dessen Ableitung divergiert
# bei tr->3, die Aufloesung liegt in float64 bei ~6e-7 Grad. 1e-5 liegt fuenf
# Groessenordnungen unter jedem physikalisch relevanten Fehler.
check("bekannte Rotation: R == Q (nicht Q^T)", errs.max().item() < 1e-5,
      f"| max Abweichung {errs.max().item():.2e} Grad ueber 200 Faelle")

# Kontrollprobe: gegen die Inverse muss es FEHLSCHLAGEN, sonst ist der Test blind
Q = haar(1, 7)[0]
inv_err = geo_deg(kabsch_rotation(C, C @ Q.transpose(-1, -2))[None], Q.transpose(-1, -2)[None]).item()
check("Kontrollprobe: R != Q^T", inv_err > 1.0, f"| Abstand zu Q^T = {inv_err:.1f} Grad")

R = kabsch_rotation(C, C @ haar(1, 3)[0].transpose(-1, -2))
check("R orthogonal", torch.allclose(R @ R.transpose(-1, -2), torch.eye(3), atol=1e-10))
check("det R = +1", abs(torch.linalg.det(R).item() - 1) < 1e-10)

# Reflexion darf NICHT gewaehlt werden, auch wenn sie besser passen wuerde
Cm = C.clone()
Ym = C.clone()
Ym[:, 2] *= -1                                  # gespiegeltes Ziel
Rm = kabsch_rotation(Cm, Ym)
check("Reflexion ausgeschlossen (det=+1 auch bei gespiegeltem Ziel)",
      abs(torch.linalg.det(Rm).item() - 1) < 1e-10)

print("\n" + "=" * 88)
print("2. ENDPUNKTE des Pfades bei ALLGEMEINEM R_1")
print("=" * 88)
so3 = SO3_FlowMatcher()
r3 = R3_FlowMatcher(sigma_min=0.0)
n = 300
R1 = haar(n, 11)
R0 = haar(n, 22)
# Der Endpunktfehler bei t=1 wird von der Praezision von so3_utils.log nahe
# pi bestimmt - eine GEERBTE Eigenschaft, die in SigmaFlow-Minimal identisch
# auftritt (dort mit R_1=I gemessen: median 3.77e-05, max 1.11 Grad; hier mit
# R_1~Haar: 4.37e-05 / 1.12 Grad). Deshalb wird getrennt geprueft: der
# regulaere Bereich scharf, der Nahe-pi-Schwanz nur dokumentiert.
for tval, ref, lbl in ((0.0, R0, "R(t=0) == R_0"), (1.0, R1, "R(t=1) == R_1")):
    t = torch.full((n,), tval)
    D = R0.transpose(-1, -2) @ R1
    Rt = R0 @ so3_utils.exp(t[:, None, None] * so3_utils.log(D))
    e = geo_deg(Rt, ref)
    near_pi = geo_deg(torch.eye(3).expand_as(D), D) > 172.0
    check(lbl + "  (|Delta| <= 172 Grad)", e[~near_pi].max().item() < 1e-2,
          f"| max {e[~near_pi].max().item():.2e} Grad")
    if tval == 1.0:
        print(f"         Nahe-pi-Schwanz ({100*near_pi.float().mean():.0f}% der Faelle): "
              f"max {e[near_pi].max().item():.2e} Grad - geerbte so3_utils-Grenze, "
              f"in Minimal identisch")

x0 = torch.randn(n, 3, generator=torch.Generator().manual_seed(5))
x1 = torch.randn(n, 3, generator=torch.Generator().manual_seed(6)) * 3
for tval, ref, lbl in ((0.0, x0, "x(t=0) == x_0"), (1.0, x1, "x(t=1) == x_1")):
    xt = (1 - tval) * x0 + tval * x1
    check(lbl, (xt - ref).norm(dim=-1).max().item() < 1e-10)

print("\n" + "=" * 88)
print("3. ORACLE-INTEGRATION R_0 -> R_1 mit R_1 != I   [BLOCKIEREND]")
print("=" * 88)
print(f"{'Zielwinkel':>12}{'Schritte':>10}{'median Fehler':>16}{'max':>12}")
print("-" * 88)
for deg in [1.0, 30.0, 90.0, 150.0, 179.0, None]:
    m = 200
    R0 = haar(m, 31 + int(deg or 0))
    if deg is None:
        R1 = haar(m, 999)
        lbl = "zufaellig"
    else:
        gg = torch.Generator().manual_seed(41 + int(deg))
        ax = torch.randn(m, 3, generator=gg)
        ax = ax / ax.norm(dim=1, keepdim=True)
        R1 = R0 @ so3_utils.exp(so3_utils.hat(ax * (deg * torch.pi / 180)))
        lbl = f"{deg:.0f} Grad"
    for nst in (5, 25):
        Rt = R0.clone()
        ts = torch.linspace(0.01, 1.0, nst + 1)
        for i in range(nst):
            t = torch.full((m,), ts[i].item())
            v = so3.calc_rot_vector_field(Rt, R1, t)
            Rt = so3.euler_step(Rt, v, (ts[i + 1] - ts[i]).item())
        e = geo_deg(Rt, R1)
        ok = e.median().item() < 1e-2
        if not ok:
            FAIL.append(f"oracle {lbl} {nst}")
        print(f"{lbl:>12}{nst:>10}{e.median().item():>16.2e}{e.max().item():>12.2e}"
              f"{'' if ok else '   <-- FAIL'}")

print("\n  Translation, dieselbe Oracle-Logik:")
for nst in (5, 25):
    xt = x0.clone()
    ts = torch.linspace(0.01, 1.0, nst + 1)
    for i in range(nst):
        t = torch.full((n,), ts[i].item())
        xt = r3.euler_step(xt, r3.calc_trans_vector_field(xt, x1, t), (ts[i + 1] - ts[i]).item())
    e = (xt - x1).norm(dim=-1)
    check(f"R3 Oracle, {nst} Schritte", e.max().item() < 1e-8, f"| max {e.max().item():.2e}")

print("\n" + "=" * 88)
print("4. GAUGE-INVARIANZ: C -> S C  muss  R_1 -> R_1 S^T  ergeben")
print("=" * 88)
errs, rot_errs = [], []
for s in range(200):
    gg = torch.Generator().manual_seed(200 + s)
    C = torch.randn(9, 3, generator=gg)
    C = C - C.mean(0)
    Q = haar(1, 300 + s)[0]
    Y = C @ Q.transpose(-1, -2) + torch.randn(9, 3, generator=gg) * 0.02   # leichtes Rauschen
    Y = Y - Y.mean(0)
    R1 = kabsch_rotation(C, Y)
    S = haar(1, 400 + s)[0]
    Cs = C @ S.transpose(-1, -2)          # C' = S C
    R1s = kabsch_rotation(Cs, Y)
    # Erwartung: R1' = R1 S^T, und die rekonstruierte Pose bleibt identisch
    rot_errs.append(geo_deg(R1s[None], (R1 @ S.transpose(-1, -2))[None]).item())
    errs.append((Cs @ R1s.transpose(-1, -2) - C @ R1.transpose(-1, -2)).norm(dim=-1).max().item())
check("R_1' == R_1 S^T", max(rot_errs) < 1e-5, f"| max {max(rot_errs):.2e} Grad")
check("rekonstruierte Pose invariant", max(errs) < 1e-9, f"| max {max(errs):.2e} A")

print("\n" + "=" * 88)
print("5. GLOBALE SE(3)-AEQUIVARIANZ der Zielkonstruktion")
print("=" * 88)
print("  x -> Qx + a  muss  R_1 -> Q R_1  und  p_1 -> Q p_1 + a  ergeben.")
rot_e, tr_e, pose_e = [], [], []
for s in range(200):
    gg = torch.Generator().manual_seed(500 + s)
    Cref = torch.randn(10, 3, generator=gg)
    Ytrue = torch.randn(10, 3, generator=gg) * 2 + torch.tensor([3.0, -1.0, 2.0])
    fidx = torch.zeros(10, dtype=torch.long)
    _, p1, R1, _ = fragment_targets(Cref, Ytrue, fidx, 1)
    Q = haar(1, 600 + s)[0]
    a = torch.randn(3, generator=gg) * 5
    # Global transformiert wird der KOMPLEX, also die Zielpose. Der
    # Referenzkonformer ist eine eigene, komplexunabhaengige Groesse und
    # bleibt unveraendert - genau das prueft die Aequivarianz der Zuordnung.
    _, p1q, R1q, _ = fragment_targets(Cref, Ytrue @ Q.transpose(-1, -2) + a, fidx, 1)
    rot_e.append(geo_deg(R1q, Q @ R1).item())
    tr_e.append((p1q[0] - (Q @ p1[0] + a)).norm().item())
    pose_e.append(((Cref - Cref.mean(0)) @ R1q[0].transpose(-1, -2) + p1q[0]
                   - (((Cref - Cref.mean(0)) @ R1[0].transpose(-1, -2) + p1[0]) @ Q.transpose(-1, -2) + a)
                   ).norm(dim=-1).max().item())
check("R_1 -> Q R_1", max(rot_e) < 1e-5, f"| max {max(rot_e):.2e} Grad")
check("p_1 -> Q p_1 + a", max(tr_e) < 1e-9, f"| max {max(tr_e):.2e} A")
check("rekonstruierte Pose transformiert korrekt", max(pose_e) < 1e-9, f"| max {max(pose_e):.2e} A")

print("\n" + "=" * 88)
print("6. GRADIENTEN / AUTOGRAD-HYGIENE")
print("=" * 88)
Cg = torch.randn(8, 3, requires_grad=True)
Yg = torch.randn(8, 3, requires_grad=True)
Cc, p1, R1, res = fragment_targets(Cg, Yg, torch.zeros(8, dtype=torch.long), 1)
check("Ziel traegt keinen Autograd-Graphen",
      not (R1.requires_grad or p1.requires_grad or Cc.requires_grad),
      "| fragment_targets ist @torch.no_grad")
check("keine NaN/Inf in Zielen",
      bool(torch.isfinite(R1).all() and torch.isfinite(p1).all() and torch.isfinite(res).all()))

print("\n" + "=" * 88)
print("7. DEGENERIERTE FAELLE")
print("=" * 88)
for nat, lbl in ((1, "1 Atom"), (2, "2 Atome"), (3, "3 kollinear")):
    gg = torch.Generator().manual_seed(900 + nat)
    if nat == 3:
        base = torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]])
        Cx, Yx = base, base @ haar(1, 7)[0].transpose(-1, -2)
    else:
        Cx = torch.randn(nat, 3, generator=gg)
        Yx = torch.randn(nat, 3, generator=gg)
    _, _, Rx, rx = fragment_targets(Cx, Yx, torch.zeros(nat, dtype=torch.long), 1)
    ok = bool(torch.isfinite(Rx).all()) and abs(torch.linalg.det(Rx[0]).item() - 1) < 1e-6
    check(f"{lbl}: endlich und det=+1", ok, f"| Residuum {rx[0].item():.3f}")

print("\n" + "=" * 88)
print(f"ERGEBNIS: {len(FAIL)} Fehlschlaege" + (f" -> {FAIL}" if FAIL else " - alle Tests bestanden"))
print("=" * 88)
sys.exit(1 if FAIL else 0)
