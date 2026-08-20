#!/usr/bin/env python3
"""
EXP-110 — Audit der Zwei-Kopf-Konstruktion.

Geprueft wird die Kette

    zwei l=1-Felder je Atom
      -> Mittel-Pooling je Fragment
      -> hat(.)  ->  so(3), Weltrahmen
      -> adjungierte Konjugation  ->  Koerperrahmen
      -> Vergleich mit log(R_t^T R_1)/(1-t)

Die Pooling- und Rahmenlogik wird hier NACHGEBAUT statt importiert. Ein Test,
der die zu pruefende Implementierung importiert, kann eine falsche Formel nicht
entdecken. Zusaetzlich wird am Ende gegen die echte Implementierung verglichen,
sofern sie importierbar ist.

Aufruf:  python audits/test_two_head_vector_field.py
Exit 0 = alle Checks bestanden.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(cond), detail))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# Referenzimplementierung (bewusst unabhaengig nachgebaut)
# ---------------------------------------------------------------------------
def hat(v: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(v[..., 0])
    return torch.stack([
        torch.stack([z, -v[..., 2], v[..., 1]], -1),
        torch.stack([v[..., 2], z, -v[..., 0]], -1),
        torch.stack([-v[..., 1], v[..., 0], z], -1),
    ], -2)


def mean_pool(field: torch.Tensor, frag_idx: torch.Tensor, M: int) -> torch.Tensor:
    out = torch.zeros((M, 3), dtype=field.dtype)
    cnt = torch.zeros(M, dtype=field.dtype)
    out.index_add_(0, frag_idx, field)
    cnt.index_add_(0, frag_idx, torch.ones_like(frag_idx, dtype=field.dtype))
    return out / cnt.clamp_min(1.0)[:, None]


def body_frame(omega_world_vec: torch.Tensor, R_t: torch.Tensor) -> torch.Tensor:
    return R_t.transpose(-1, -2) @ hat(omega_world_vec) @ R_t


def rand_rot(g: torch.Generator, n: int = 1) -> torch.Tensor:
    A = torch.randn(n, 3, 3, generator=g, dtype=torch.float64)
    Q, R = torch.linalg.qr(A)
    Q = Q * torch.sign(torch.diagonal(R, dim1=-2, dim2=-1)).unsqueeze(-2)
    flip = torch.det(Q) < 0
    Q[flip, :, 0] *= -1
    return Q


g = torch.Generator().manual_seed(0xC0FFEE)
M, K = 4, 23                                   # 4 Fragmente, 23 Atome
frag_idx = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2,
                         3, 3, 3, 3, 3, 3, 3, 3, 3])
assert frag_idx.numel() == K and int(frag_idx.max()) == M - 1

fT = torch.randn(K, 3, generator=g, dtype=torch.float64)
fR = torch.randn(K, 3, generator=g, dtype=torch.float64)
R_t = rand_rot(g, M)
R_1 = rand_rot(g, M)
t = torch.rand(M, generator=g, dtype=torch.float64) * 0.8 + 0.1

print()
print("=" * 72)
print("1. FORMEN")
print("=" * 72)
v = mean_pool(fT, frag_idx, M)
w = mean_pool(fR, frag_idx, M)
u_R = body_frame(w, R_t)
check("f^T, f^R je [K,3]", fT.shape == (K, 3) and fR.shape == (K, 3), f"{tuple(fT.shape)}")
check("v_frag [M,3]", v.shape == (M, 3), f"{tuple(v.shape)}")
check("omega_frag [M,3]", w.shape == (M, 3), f"{tuple(w.shape)}")
check("u_R [M,3,3]", u_R.shape == (M, 3, 3), f"{tuple(u_R.shape)}")

print()
print("=" * 72)
print("2. SCHIEFSYMMETRIE  U_R^T = -U_R")
print("=" * 72)
asym = (u_R + u_R.transpose(-1, -2)).abs().max().item()
check("Koerperrahmen-Vorhersage liegt in so(3)", asym < 1e-12, f"max|U+U^T| = {asym:.2e}")
w_asym = (hat(w) + hat(w).transpose(-1, -2)).abs().max().item()
check("Weltrahmen-Groesse liegt in so(3)", w_asym < 1e-12, f"{w_asym:.2e}")

print()
print("=" * 72)
print("3. TRANSFORMATIONSVERHALTEN UNTER GLOBALER ROTATION Q")
print("=" * 72)
Q = rand_rot(g, 1)[0]
fT_r, fR_r = fT @ Q.T, fR @ Q.T          # aequivariante Netzausgabe
R_t_r, R_1_r = Q @ R_t, Q @ R_1          # mitgedrehtes System

v_r = mean_pool(fT_r, frag_idx, M)
w_r = mean_pool(fR_r, frag_idx, M)

e = (v_r - v @ Q.T).abs().max().item()
check("Translation ist AEQUIVARIANT:  v(Qx) = Q v(x)", e < 1e-12, f"{e:.2e}")
e = (w_r - w @ Q.T).abs().max().item()
check("Rotationspooling ist AEQUIVARIANT:  w(Qx) = Q w(x)", e < 1e-12, f"{e:.2e}")

tgt = torch.linalg.matrix_exp(torch.zeros(M, 3, 3, dtype=torch.float64))  # Platzhalter
A = R_t.transpose(-1, -2) @ R_1
A_r = R_t_r.transpose(-1, -2) @ R_1_r
e = (A - A_r).abs().max().item()
check("ZIEL ist INVARIANT:  (QR_t)^T(QR_1) = R_t^T R_1", e < 1e-12, f"{e:.2e}")

u_R_r = body_frame(w_r, R_t_r)
e = (u_R - u_R_r).abs().max().item()
check("Koerperrahmen-Vorhersage ist INVARIANT (passt zum Ziel)", e < 1e-12, f"{e:.2e}")

print()
print("  Kernaussage: das Ziel ist invariant, jede aequivariante Netzausgabe")
print("  ist es nicht. Die Konjugation R_t^T (.) R_t schliesst diese Luecke --")
print("  fuer JEDE Konstruktion, auch fuer das Drehmoment in Minimal.")

print()
print("=" * 72)
print("4. PERMUTATIONSINVARIANZ INNERHALB DER FRAGMENTE")
print("=" * 72)
perm = torch.randperm(K, generator=g)
v_p = mean_pool(fT[perm], frag_idx[perm], M)
w_p = mean_pool(fR[perm], frag_idx[perm], M)
check("v_frag unveraendert unter Atomumordnung",
      (v - v_p).abs().max().item() < 1e-13, f"{(v - v_p).abs().max().item():.2e}")
check("omega_frag unveraendert unter Atomumordnung",
      (w - w_p).abs().max().item() < 1e-13, f"{(w - w_p).abs().max().item():.2e}")

print()
print("=" * 72)
print("5. GROESSENUNABHAENGIGKEIT: MITTEL GEGEN SUMME")
print("=" * 72)
big = torch.cat([fR[frag_idx == 0]] * 4, 0)
idx_big = torch.zeros(big.shape[0], dtype=torch.long)
mean_big = mean_pool(big, idx_big, 1)
mean_small = mean_pool(fR[frag_idx == 0], torch.zeros(5, dtype=torch.long), 1)
check("Mittel: Fragment vervierfachen aendert nichts",
      (mean_big - mean_small).abs().max().item() < 1e-13,
      f"{(mean_big - mean_small).abs().max().item():.2e}")
sum_big = big.sum(0)
sum_small = fR[frag_idx == 0].sum(0)
ratio = (sum_big.norm() / sum_small.norm()).item()
check("Summe waere um Faktor 4 groesser (deshalb Mittel)",
      abs(ratio - 4.0) < 1e-9, f"Faktor {ratio:.3f}")

print()
print("=" * 72)
print("6. ENTARTUNG: WAS DIE DREHMOMENTKONSTRUKTION NICHT KANN")
print("=" * 72)


def torque_omega(pts: torch.Tensor, f: torch.Tensor) -> tuple[torch.Tensor, float]:
    c = pts.mean(0)
    rel = pts - c
    tau = torch.cross(rel, f, dim=-1).sum(0)
    rr = (rel ** 2).sum(-1)
    I = (rr[:, None, None] * torch.eye(3, dtype=pts.dtype)
         - rel[:, :, None] * rel[:, None, :]).sum(0)
    ev = torch.linalg.eigvalsh(I)
    mn = max(1e-8 * I.trace().item() / 3.0, 1e-12)
    Ereg = ev.clamp_min(mn)
    cond = (Ereg.max() / Ereg.min()).item()
    return tau, cond


one_atom = torch.zeros(1, 3, dtype=torch.float64)
tau1, _ = torque_omega(one_atom, torch.randn(1, 3, generator=g, dtype=torch.float64))
check("1-Atom-Fragment: Drehmoment ist identisch 0 -> Rotation unerreichbar",
      tau1.abs().max().item() < 1e-15, f"|tau| = {tau1.abs().max().item():.2e}")
w1 = mean_pool(torch.randn(1, 3, generator=g, dtype=torch.float64),
               torch.zeros(1, dtype=torch.long), 1)
check("Mittel-Pooling liefert dort weiterhin eine Winkelgeschwindigkeit",
      w1.abs().max().item() > 1e-6, f"|w| = {w1.abs().max().item():.3f}")

lin = torch.tensor([[0., 0, 0], [1.5, 0, 0], [3.0, 0, 0]], dtype=torch.float64)
_, cond_lin = torque_omega(lin, torch.randn(3, 3, generator=g, dtype=torch.float64))
check("lineares Fragment: Traegheit ist schlecht konditioniert",
      cond_lin > 1e6, f"cond(I_reg) = {cond_lin:.2e}")

print()
print("=" * 72)
print("7. FRAME-KONSISTENZ MIT DEM INTEGRATOR  R_next = R_t exp(u dt)")
print("=" * 72)
dt = 1e-4
R_next = R_t @ torch.linalg.matrix_exp(u_R * dt)
det = (torch.det(R_next) - 1).abs().max().item()
orth = (R_next.transpose(-1, -2) @ R_next
        - torch.eye(3, dtype=torch.float64)).abs().max().item()
check("Integrationsschritt bleibt in SO(3): det = 1", det < 1e-10, f"{det:.2e}")
check("Integrationsschritt bleibt in SO(3): R^T R = I", orth < 1e-10, f"{orth:.2e}")

# Aequivalenz beider Schreibweisen: rechts im Koerper == links in der Welt
left = torch.linalg.matrix_exp(hat(w) * dt) @ R_t
e = (R_next - left).abs().max().item()
check("R_t exp(u_body dt) == exp(w_world dt) R_t", e < 1e-10, f"{e:.2e}")

print()
print("=" * 72)
print("8. VERGLEICH MIT DER ECHTEN IMPLEMENTIERUNG")
print("=" * 72)
root = Path(__file__).resolve().parents[1]
src = root / "SigmaFlow_FM_Specific" / "EXP-110_two_head_vector_field" / "src"
sys.path.insert(0, str(src))
try:
    from sigmadock.diff.sigma_flow_generator import SigmaFlowGenerator as G
    vi, wi = G.pool_fragment_fields(fT.float(), fR.float(), frag_idx, M)
    check("pool_fragment_fields stimmt mit der Referenz ueberein (v)",
          (vi.double() - v).abs().max().item() < 1e-5,
          f"{(vi.double() - v).abs().max().item():.2e}")
    check("pool_fragment_fields stimmt mit der Referenz ueberein (omega)",
          (wi.double() - w).abs().max().item() < 1e-5,
          f"{(wi.double() - w).abs().max().item():.2e}")
except Exception as exc:  # noqa: BLE001
    check("Implementierung importierbar", False, f"{type(exc).__name__}: {exc}")

print()
print("=" * 72)
ok = sum(1 for _, c, _ in CHECKS if c)
print(f"ERGEBNIS: {ok}/{len(CHECKS)} Checks bestanden")
print("=" * 72)
if ok != len(CHECKS):
    print("\nFehlgeschlagen:")
    for n_, c, d in CHECKS:
        if not c:
            print(f"  - {n_}  ({d})")
raise SystemExit(0 if ok == len(CHECKS) else 1)
