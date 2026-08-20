#!/usr/bin/env python3
"""
EXP-110 — Gradientenaudit der Zwei-Kopf-Variante.

Die geometrischen Eigenschaften stehen in test_two_head_vector_field.py. Hier
geht es um die Verdrahtung: bekommt jeder Kopf Gradienten aus SEINEM Loss, und
gibt es unbeabsichtigte Querverbindungen?

Das ist die Frage, die bei der Ein-Kopf-Variante nicht stellbar war -- dort
teilten sich beide Losse denselben Ausgang. Mit zwei Koepfen muss gelten:

    dL_trans / d(rot_block)   == 0     exakt, nicht nur klein
    dL_rot   / d(trans_block) == 0     exakt
    dL_trans / d(trans_block) != 0
    dL_rot   / d(rot_block)   != 0
    beide Losse erreichen den GETEILTEN Rumpf

Gearbeitet wird mit einem Stellvertreterrumpf statt EquiformerV2: geprueft
wird die Verdrahtung der Kopf- und Poolinglogik, nicht das Netz. Ein echter
EquiformerV2-Vorwaertslauf braeuchte einen vollstaendigen Graphen und wuerde
die Frage nicht schaerfer beantworten.

Aufruf: python audits/test_two_head_gradients.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "SigmaFlow_FM_Specific" / "EXP-110_two_head_vector_field" / "src"))

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(cond), detail))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))


from sigmadock.diff.sigma_flow_generator import SigmaFlowGenerator  # noqa: E402
from sigmadock.diff import so3_utils  # noqa: E402


class Stub(nn.Module):
    """Geteilter Rumpf, zwei getrennte lineare Koepfe -- dieselbe Topologie
    wie EquiformerV2 mit trans_block und rot_block."""

    def __init__(self, d: int = 16):
        super().__init__()
        self.trunk = nn.Linear(d, d)
        self.trans_block = nn.Linear(d, 3)
        self.rot_block = nn.Linear(d, 3)

    def forward(self, h):
        z = torch.tanh(self.trunk(h))
        return self.trans_block(z), self.rot_block(z)


torch.manual_seed(7)
K, M, D = 17, 3, 16
frag_idx = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2])
h = torch.randn(K, D)

net = Stub(D)
R_t = so3_utils.exp(so3_utils.hat(torch.randn(M, 3) * 0.4))

print()
print("=" * 70)
print("1. VORWAERTSLAUF")
print("=" * 70)
fT, fR = net(h)
v, w = SigmaFlowGenerator.pool_fragment_fields(fT, fR, frag_idx, M)
omega_world = so3_utils.hat(w)
u_R = R_t.transpose(-1, -2) @ omega_world @ R_t

check("f^T [K,3]", fT.shape == (K, 3), str(tuple(fT.shape)))
check("f^R [K,3]", fR.shape == (K, 3), str(tuple(fR.shape)))
check("v_frag [M,3]", v.shape == (M, 3), str(tuple(v.shape)))
check("u_R [M,3,3]", u_R.shape == (M, 3, 3), str(tuple(u_R.shape)))
check("u_R schiefsymmetrisch",
      (u_R + u_R.transpose(-1, -2)).abs().max().item() < 1e-6,
      f"{(u_R + u_R.transpose(-1, -2)).abs().max().item():.2e}")

print()
print("=" * 70)
print("2. GRADIENTENZUORDNUNG")
print("=" * 70)


def grads(loss):
    net.zero_grad(set_to_none=True)
    loss.backward(retain_graph=True)
    g = {}
    for name, mod in (("trans_block", net.trans_block),
                      ("rot_block", net.rot_block),
                      ("trunk", net.trunk)):
        tot = sum(p.grad.abs().sum().item() for p in mod.parameters() if p.grad is not None)
        g[name] = tot
    return g


tgt_v = torch.randn(M, 3)
tgt_R = so3_utils.hat(torch.randn(M, 3))

L_trans = ((v - tgt_v) ** 2).sum()
g = grads(L_trans)
print(f"    L_trans  ->  trans_block {g['trans_block']:.4e} | "
      f"rot_block {g['rot_block']:.4e} | trunk {g['trunk']:.4e}")
check("L_trans erreicht trans_block", g["trans_block"] > 1e-8, f"{g['trans_block']:.3e}")
check("L_trans erreicht rot_block NICHT (exakt 0)", g["rot_block"] == 0.0, f"{g['rot_block']:.3e}")
check("L_trans erreicht den geteilten Rumpf", g["trunk"] > 1e-8, f"{g['trunk']:.3e}")

fT, fR = net(h)
v, w = SigmaFlowGenerator.pool_fragment_fields(fT, fR, frag_idx, M)
u_R = R_t.transpose(-1, -2) @ so3_utils.hat(w) @ R_t
L_rot = ((u_R - tgt_R) ** 2).sum()
g = grads(L_rot)
print(f"    L_rot    ->  trans_block {g['trans_block']:.4e} | "
      f"rot_block {g['rot_block']:.4e} | trunk {g['trunk']:.4e}")
check("L_rot erreicht rot_block", g["rot_block"] > 1e-8, f"{g['rot_block']:.3e}")
check("L_rot erreicht trans_block NICHT (exakt 0)", g["trans_block"] == 0.0, f"{g['trans_block']:.3e}")
check("L_rot erreicht den geteilten Rumpf", g["trunk"] > 1e-8, f"{g['trunk']:.3e}")

print()
print("  Die exakten Nullen sind der eigentliche Befund: die Koepfe sind")
print("  strukturell entkoppelt. In der Ein-Kopf-Variante war das unmoeglich --")
print("  dort teilten sich beide Losse denselben Ausgang und der Audit konnte")
print("  nur zeigen, dass die Gradienten ORTHOGONAL sind (cos = 0.0000).")

print()
print("=" * 70)
print("3. ADJUNGIERTE KONJUGATION IST DIFFERENZIERBAR")
print("=" * 70)
net.zero_grad(set_to_none=True)
fT, fR = net(h)
v, w = SigmaFlowGenerator.pool_fragment_fields(fT, fR, frag_idx, M)
u_R = R_t.transpose(-1, -2) @ so3_utils.hat(w) @ R_t
u_R.pow(2).sum().backward()
gw = sum(p.grad.abs().sum().item() for p in net.rot_block.parameters())
check("Gradient fliesst durch R_t^T (.) R_t", gw > 1e-8, f"{gw:.3e}")
check("Gradienten endlich",
      all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None))

print()
print("=" * 70)
print("4. GEGENPROBE: kuenstliches detach muss den Gradienten toeten")
print("=" * 70)
# Mit detach haengt u_R an keinem Parameter mehr -- der Graph ist leer.
# Genau das ist der Nachweis, also wird es so geprueft und nicht per backward.
net.zero_grad(set_to_none=True)
fT, fR = net(h)
_, w_det = SigmaFlowGenerator.pool_fragment_fields(fT, fR.detach(), frag_idx, M)
u_R_det = R_t.transpose(-1, -2) @ so3_utils.hat(w_det) @ R_t
check("mit detach haengt u_R an keinem Parameter mehr",
      not u_R_det.requires_grad, f"requires_grad = {u_R_det.requires_grad}")

# Und die Translation muss davon voellig unberuehrt bleiben.
v_det, _ = SigmaFlowGenerator.pool_fragment_fields(fT, fR.detach(), frag_idx, M)
((v_det - tgt_v) ** 2).sum().backward()
gt = sum(p.grad.abs().sum().item() for p in net.trans_block.parameters() if p.grad is not None)
gr = sum(p.grad.abs().sum().item() for p in net.rot_block.parameters() if p.grad is not None)
check("Translationsgradient bleibt bei detachtem Rotationszweig erhalten", gt > 1e-8, f"{gt:.3e}")
check("Rotationskopf bekommt dabei nichts", gr == 0.0, f"{gr:.3e}")

print()
print("=" * 70)
ok = sum(1 for _, c, _ in CHECKS if c)
print(f"ERGEBNIS: {ok}/{len(CHECKS)} Checks bestanden")
print("=" * 70)
raise SystemExit(0 if ok == len(CHECKS) else 1)
