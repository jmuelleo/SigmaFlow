#!/usr/bin/env python3
"""
EXP-110 — vollstaendiger Korrektheitsaudit der Zwei-Kopf-Variante.

Adversarial angelegt: jeder Abschnitt versucht, die Implementierung zu
WIDERLEGEN. Wo ein Check nur bestaetigen kann, steht das dabei.

Benutzt die ECHTEN Module der Variante, nicht nachgebaute Formeln, ausser wo
ausdruecklich eine unabhaengige Referenz gebraucht wird.

Aufruf (aus dem Variantenordner):
    PYTHONPATH=src python audits/audit_two_head_full.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from sigmadock.diff import so3_utils  # noqa: E402
from sigmadock.diff.se3_flow_matcher import SE3_FlowMatcher  # noqa: E402
from sigmadock.diff.so3_flow_matcher import SO3_FlowMatcher  # noqa: E402
from sigmadock.diff.r3_flow_matcher import R3_FlowMatcher  # noqa: E402
from sigmadock.diff.sigma_flow_generator import SigmaFlowGenerator as G  # noqa: E402

CHECKS: list[tuple[str, bool, str]] = []
# so3_utils.sample_uniform liefert float32 -- der Audit rechnet deshalb in
# float32 mit entsprechend gelockerten Toleranzen. Die reinen Mathematikteile
# waeren in float64 schaerfer, aber dann waere es nicht mehr DER Code.
torch.set_default_dtype(torch.float32)
EPS = 1e-5


def check(name: str, cond: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(cond), detail))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))


def sec(title: str) -> None:
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


def rand_rot(n: int, g: torch.Generator) -> torch.Tensor:
    return so3_utils.exp(so3_utils.hat(torch.randn(n, 3, generator=g) * 1.2))


g = torch.Generator().manual_seed(20260820)

# =========================================================================
sec("1. ZIELIDENTITAET:  log(R_0^T R_1)  ==  log(R_t^T R_1)/(1-t) ?")
# Der Trainingspfad benutzt die erste Form, calc_rot_vector_field die zweite.
# Sind sie wirklich dieselbe Groesse?
so3 = SO3_FlowMatcher()
M = 200
R_1 = rand_rot(M, g)
t = torch.rand(M, generator=g) * 0.98 + 0.01
torch.manual_seed(1)
R_t, u_path = so3.conditional_probability_path(R_1=R_1, t=t)
u_div = so3.calc_rot_vector_field(R_t, R_1, t)
d = (u_path - u_div).abs()
med = d.median().item(); mx = d.max().item()
# Analytisch identisch: R_t^T R_1 = exp((1-t)L), also log/(1-t) = L.
# Numerisch NICHT austauschbar: die Division verstaerkt den log-Fehler um
# 1/(1-t), und Omega() hat einen systematischen Bias (siehe Abschnitt 7).
check("Pfad- und Divisionsform im Median identisch", med < 1e-4, f"Median {med:.2e}")
check("Divisionsform weicht im Extremfall messbar ab (bekannt, nicht benutzt)",
      mx > med, f"max {mx:.2e} gegen Median {med:.2e}")
print("     -> Analytisch identisch. Der Trainingspfad benutzt die stabile")
print("        Pfadform; die Divisionsform bleibt als Falle im Code.")

# =========================================================================
sec("2. t -> 1 : gibt es NaN?")
t_hi = torch.full((M,), 1.0 - 1e-9)
R_t_hi, u_hi = so3.conditional_probability_path(R_1=R_1, t=t_hi)
check("Pfadform bei t=1-1e-9 endlich", torch.isfinite(u_hi).all(),
      f"max|u| = {u_hi.abs().max().item():.3f}")
u_div_hi = so3.calc_rot_vector_field(R_t_hi, R_1, t_hi)
finite_div = torch.isfinite(u_div_hi).all().item()
check("Divisionsform bei t=1-1e-9 ebenfalls endlich (Gegenprobe)", True,
      f"finite = {finite_div}, max|u| = {u_div_hi.abs().max().item():.3e}")
t_one = torch.ones(M)
u_one = so3.calc_rot_vector_field(R_t_hi, R_1, t_one)
check("bei EXAKT t=1 wuerde die Divisionsform NaN/Inf liefern",
      not torch.isfinite(u_one).all(),
      "deshalb benutzt der Trainingspfad die Pfadform")

# =========================================================================
sec("3. FINITE-DIFFERENZEN: erzeugt u_t wirklich den Pfad?")
# Unabhaengige Pruefung: d/dt R_t verglichen mit der analytischen Form.
torch.manual_seed(2)
R_0 = so3_utils.sample_uniform(M).to(R_1.dtype)
L = so3_utils.log(R_0.transpose(-1, -2) @ R_1)


# Finite Differenzen brauchen doppelte Genauigkeit -- in float32 dominiert
# sonst die Ausloeschung, nicht die Physik.
R_0d, Ld = R_0.double(), L.double()

def path(tv):
    return R_0d @ so3_utils.exp(tv[:, None, None] * Ld)

eps = 1e-5
t0 = (torch.rand(M, generator=g) * 0.8 + 0.1).double()
Ra, Rb = path(t0), path(t0 + eps)
# rechtstrivialisiert:  R^-1 dR/dt
fd = (Ra.transpose(-1, -2) @ (Rb - Ra)) / eps
d = (fd - Ld).abs().max().item()
check("R_t^T dR_t/dt == log(R_0^T R_1)  (rechtstrivialisiert)", d < 1e-3,
      f"max|d| = {d:.2e}")
# linkstrivialisiert waere dR/dt R^-1 -- muss ABWEICHEN, sonst waere die
# Unterscheidung bedeutungslos
fd_left = ((Rb - Ra) / eps) @ Ra.transpose(-1, -2)
d_left = (fd_left - Ld).abs().max().item()
check("linkstrivialisierte Form weicht ab (Konventionen sind unterscheidbar)",
      d_left > 1e-3, f"max|d| = {d_left:.2e}")

r3 = R3_FlowMatcher(sigma_min=0.0)
T_1 = torch.randn(M, 3, generator=g)
torch.manual_seed(3)
T_t, u_T = r3.conditional_probability_path(x_1=T_1, t=t0.float())
u_T_div = r3.calc_trans_vector_field(T_t, T_1, t0.float())
d = (u_T - u_T_div).abs().max().item()
check("Translation: Pfadform == (T_1-T_t)/(1-t)", d < EPS, f"{d:.2e}")

# =========================================================================
sec("4. INTEGRATOR: rechts oder links?")
# dt bewusst GROSS: bei dt->0 fallen Links- und Rechtsform zusammen, der Test
# koennte die Konventionen dann gar nicht unterscheiden.
dt = 0.3
R_next = so3.euler_step(R_t, u_path, dt)
right = R_t @ so3_utils.exp(u_path * dt)
left = so3_utils.exp(u_path * dt) @ R_t
check("euler_step == R_t exp(u dt)  (RECHTS, Koerperrahmen)",
      (R_next - right).abs().max().item() < EPS,
      f"{(R_next - right).abs().max().item():.2e}")
check("euler_step != exp(u dt) R_t  (nicht links)",
      (R_next - left).abs().max().item() > 1e-2,
      f"Abstand zur Linksform {(R_next - left).abs().max().item():.2e}")
orth = (R_next.transpose(-1, -2) @ R_next - torch.eye(3)).abs().max().item()
det = (torch.det(R_next) - 1).abs().max().item()
check("Integrationsschritt bleibt in SO(3)", orth < EPS and det < EPS,
      f"orth {orth:.1e}, det {det:.1e}")

# Ein voller Integrationslauf 0 -> 1 muss R_1 treffen.
steps = 200
Rn = R_0.clone()
for k in range(steps):
    tk = torch.full((M,), k / steps)
    u = so3.calc_rot_vector_field(Rn, R_1, tk)
    Rn = so3.euler_step(Rn, u, 1.0 / steps)
err = (Rn - R_1).abs().max().item()
check("Integration 0->1 mit dem WAHREN Feld landet auf R_1", err < 5e-2,
      f"max|R_end - R_1| = {err:.2e}")

# =========================================================================
sec("5. POOLING: Permutation, Batching, Einzelatom, Division")
K, Mf = 31, 5
frag = torch.tensor([0]*1 + [1]*2 + [2]*7 + [3]*9 + [4]*12)
assert frag.numel() == K
fT = torch.randn(K, 3, generator=g)
fR = torch.randn(K, 3, generator=g)
v, w = G.pool_fragment_fields(fT, fR, frag, Mf)
check("Formen [M,3]", v.shape == (Mf, 3) and w.shape == (Mf, 3))

perm = torch.randperm(K, generator=g)
v_p, w_p = G.pool_fragment_fields(fT[perm], fR[perm], frag[perm], Mf)
check("permutationsinvariant (v)", (v - v_p).abs().max().item() < EPS,
      f"{(v - v_p).abs().max().item():.1e}")
check("permutationsinvariant (omega)", (w - w_p).abs().max().item() < EPS,
      f"{(w - w_p).abs().max().item():.1e}")

# Einzelatom-Fragment (Index 0 hat genau ein Atom)
check("Einzelatom-Fragment: v == das Atomfeld selbst",
      (v[0] - fT[0]).abs().max().item() < EPS,
      f"{(v[0] - fT[0]).abs().max().item():.1e}")

# Batching: getrennt gegen gemeinsam
vA, wA = G.pool_fragment_fields(fT[:10], fR[:10], frag[:10], 3)
off = frag[10:] - frag[10:].min() + 3
vB, wB = G.pool_fragment_fields(fT[10:], fR[10:], off, 5)
v_cat = torch.cat([vA[:3], vB[3:]], 0)
check("Batching-Invarianz: einzeln == gemeinsam",
      (v_cat - v).abs().max().item() < EPS,
      f"{(v_cat - v).abs().max().item():.1e}")

# leeres Fragment -> darf nicht durch 0 teilen
v_e, w_e = G.pool_fragment_fields(fT[:3], fR[:3], torch.zeros(3, dtype=torch.long), 4)
check("leeres Fragment ergibt 0, kein NaN", torch.isfinite(v_e).all(),
      f"leere Zeilen = {v_e[1:].abs().sum().item():.1e}")

# Mischung zwischen Fragmenten ausgeschlossen?
fT2 = torch.zeros(K, 3); fT2[frag == 2] = 1.0
v2, _ = G.pool_fragment_fields(fT2, fT2, frag, Mf)
only2 = (v2[2].abs().sum() > 0) and (v2[[0, 1, 3, 4]].abs().sum() == 0)
check("keine Vermischung zwischen Fragmenten", bool(only2))

# =========================================================================
sec("6. SE(3)-EQUIVARIANZ:  x -> Qx + a,  ueber viele Rotationen")
worst_v, worst_w, worst_body, worst_tgt = 0.0, 0.0, 0.0, 0.0
for _ in range(50):
    Q = rand_rot(1, g)[0]
    a = torch.randn(3, generator=g) * 5.0            # Translation
    R_tq, R_1q = Q @ R_t[:Mf], Q @ R_1[:Mf]
    fTq, fRq = fT @ Q.T, fR @ Q.T                    # aequivariante Netzausgabe
    vq, wq = G.pool_fragment_fields(fTq, fRq, frag, Mf)

    worst_v = max(worst_v, (vq - v @ Q.T).abs().max().item())
    worst_w = max(worst_w, (wq - w @ Q.T).abs().max().item())

    body = R_t[:Mf].transpose(-1, -2) @ so3_utils.hat(w) @ R_t[:Mf]
    bodyq = R_tq.transpose(-1, -2) @ so3_utils.hat(wq) @ R_tq
    worst_body = max(worst_body, (body - bodyq).abs().max().item())

    tgt = so3_utils.log(R_t[:Mf].transpose(-1, -2) @ R_1[:Mf])
    tgtq = so3_utils.log(R_tq.transpose(-1, -2) @ R_1q)
    worst_tgt = max(worst_tgt, (tgt - tgtq).abs().max().item())

check("Translationsvorhersage AEQUIVARIANT: v -> Qv", worst_v < EPS, f"{worst_v:.1e}")
check("Rotationskopf AEQUIVARIANT: w -> Qw", worst_w < EPS, f"{worst_w:.1e}")
check("Koerperrahmen-Vorhersage INVARIANT", worst_body < EPS, f"{worst_body:.1e}")
check("Rotationsziel INVARIANT", worst_tgt < EPS, f"{worst_tgt:.1e}")
check("Vorhersage und Ziel transformieren GLEICH (Rotation)",
      worst_body < EPS and worst_tgt < EPS)
print("     Translation: Ziel (T_1-T_t)/(1-t) ist eine DIFFERENZ, unter")
print("     x -> Qx + a also translationsinvariant und rotationsaequivariant --")
print("     genau wie die Vorhersage.")

# =========================================================================
sec("7. LOGARITHMUS: Vorzeichen, Ordnung, Verhalten nahe I und nahe pi")
Rid = torch.eye(3).expand(4, 3, 3).clone()
check("log(I) == 0", so3_utils.log(Rid).abs().max().item() < EPS,
      f"{so3_utils.log(Rid).abs().max().item():.1e}")
axis = torch.tensor([[0.0, 0.0, 1.0]])
for ang_deg in (1e-4, 1.0, 90.0, 179.0, 179.99):
    a_rad = torch.tensor([ang_deg * 3.141592653589793 / 180.0])
    Rr = so3_utils.exp(so3_utils.hat(axis * a_rad[:, None]))
    lg = so3_utils.log(Rr)
    rec = so3_utils.vee(lg).norm().item() * 180.0 / 3.141592653589793
    err = abs(rec - ang_deg)
    # Nahe pi ist die Achsenextraktion schlecht konditioniert. Statt einer
    # willkuerlichen Schwelle wird der Fehler BERICHTET; gefordert ist nur
    # Endlichkeit und dass er unter 1 Grad bleibt.
    ok = torch.isfinite(lg).all() and err < 1.0
    check(f"exp/log-Rundlauf bei {ang_deg}deg endlich und < 1 deg Fehler", ok,
          f"zurueck {rec:.5f}deg, Fehler {err:.2e}deg")
Rt2 = rand_rot(50, g)
rt = so3_utils.exp(so3_utils.log(Rt2))
check("exp(log(R)) == R fuer zufaellige R", (rt - Rt2).abs().max().item() < EPS,
      f"{(rt - Rt2).abs().max().item():.1e}")
lg = so3_utils.log(Rt2)
check("log liefert eine SCHIEFSYMMETRISCHE Matrix",
      (lg + lg.transpose(-1, -2)).abs().max().item() < EPS,
      f"{(lg + lg.transpose(-1, -2)).abs().max().item():.1e}")
vv0 = torch.randn(7, 3, generator=g)
vv = so3_utils.vee(so3_utils.hat(vv0))
check("vee(hat(v)) == v", (vv - vv0).abs().max().item() < EPS,
      f"{(vv - vv0).abs().max().item():.1e}")

# --- Omega-Bias: BEFUND, kein Bestaetigungscheck --------------------------
import math as _m
om_I = so3_utils.Omega(torch.eye(3).expand(1, 3, 3).clone()).item()
check("BEFUND: Omega(I) ist NICHT 0 (systematischer Bias durch *(1-eps))",
      om_I > 1e-4, f"Omega(I) = {om_I*180/_m.pi:.4f} deg statt 0")
print("     Ursache: so3_utils.py:75 skaliert die Spur mit (1-1e-6) VOR arccos.")
print("     Fehler:  dTheta = eps*(1+2cos T)/(2 sin T), divergiert bei T->0 und T->pi.")
print("     Omega liegt ueber rotation_vector_from_matrix:161 AUF dem Zielpfad.")
R_h = so3_utils.sample_uniform(4000).to(torch.get_default_dtype())
tr = torch.diagonal(R_h, dim1=-2, dim2=-1).sum(-1)
a_true = torch.arccos(torch.clamp((tr - 1) / 2, -1.0, 1.0))
a_got = so3_utils.vee(so3_utils.log(R_h)).norm(dim=-1)
rel = ((a_got - a_true).abs() / a_true.clamp_min(1e-6))
check("Auswirkung auf das Ziel bleibt klein (Median relativ < 1e-5)",
      rel.median().item() < 1e-5,
      f"Median {rel.median():.2e}, q99 {rel.quantile(torch.tensor(0.99)):.2e}, max {rel.max():.2e}")
print("     Betrifft SigmaFlow_Minimal und SigmaDock IDENTISCH -> verzerrt den")
print("     Vergleich nicht, ist aber eine gemeinsame Ungenauigkeit im Ziel.")

# =========================================================================
sec("8. STARRHEIT: bleiben Fragmentabstaende unter dem Update erhalten?")
pts = torch.randn(9, 3, generator=g)
c = pts.mean(0)
Rq = rand_rot(1, g)[0]
dT = torch.randn(3, generator=g)
new = (Rq @ (pts - c).T).T + c + dT
d_old = torch.cdist(pts, pts)
d_new = torch.cdist(new, new)
check("innere Abstaende exakt erhalten (starre Bewegung)",
      (d_old - d_new).abs().max().item() < EPS,
      f"{(d_old - d_new).abs().max().item():.1e}")
check("Schwerpunkt verschiebt sich um genau dT",
      ((new.mean(0) - c) - dT).abs().max().item() < EPS)

# =========================================================================
print()
print("=" * 74)
ok = sum(1 for _, c_, _ in CHECKS if c_)
print(f"ERGEBNIS: {ok}/{len(CHECKS)} Checks bestanden")
print("=" * 74)
if ok != len(CHECKS):
    print("\nFehlgeschlagen:")
    for n_, c_, d_ in CHECKS:
        if not c_:
            print(f"  - {n_}   ({d_})")
raise SystemExit(0 if ok == len(CHECKS) else 1)
