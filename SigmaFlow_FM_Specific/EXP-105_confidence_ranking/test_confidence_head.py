"""Tests des integrierten Confidence-Heads.

Der Head entscheidet spaeter, welche von K Posen berichtet wird. Zwei
Eigenschaften muessen deshalb bewiesen sein, bevor irgendein Compute
hineingeht:

  1. INVARIANZ. "Ist diese Pose gut" darf nicht davon abhaengen, in welchem
     globalen Bezugssystem der Komplex vorliegt.
  2. MASKIERUNG. Padding-Knoten duerfen das Ergebnis nicht beeinflussen, sonst
     lernt der Head Molekuelgroesse statt Posenqualitaet.

    python SigmaFlow_FM_Specific/EXP-105_confidence_ranking/test_confidence_head.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from confidence_head import (  # noqa: E402
    ConfidenceHead,
    InvariantPooling,
    confidence_loss,
    extract_invariant_features,
)

FAILS: list[str] = []
torch.set_default_dtype(torch.float64)


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'OK  ' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def main() -> int:  # noqa: C901
    torch.manual_seed(20260816)
    N, C, B = 60, 32, 4
    batch_idx = torch.arange(B).repeat_interleave(N // B)
    node_mask = torch.ones(N, dtype=torch.bool)

    print("\n1. Der l=0-Kanal ist der invariante")
    emb = torch.randn(N, 9, C)          # L=2 -> (L+1)^2 = 9
    h = extract_invariant_features(emb)
    check("Form [N, C]", h.shape == (N, C), str(tuple(h.shape)))
    check("es ist wirklich Index 0", torch.equal(h, emb[:, 0, :]))
    try:
        extract_invariant_features(torch.randn(N, C))
        check("falsche Form wird abgelehnt", False)
    except ValueError:
        check("falsche Form wird abgelehnt", True)

    print("\n2. Pooling")
    pool = InvariantPooling("mean")
    out = pool(h, batch_idx, node_mask, B)
    check("Form [B, C]", out.shape == (B, C), str(tuple(out.shape)))
    manual = torch.stack([h[batch_idx == i].mean(0) for i in range(B)])
    check("Mittel stimmt mit der Handrechnung", float((out - manual).abs().max()) < 1e-12)

    # DER entscheidende Fall: maskierte Knoten duerfen NICHTS beitragen.
    # Ohne diesen Test koennte eine Implementierung, die die Maske ignoriert,
    # unbemerkt durchgehen -- sie saehe bei voller Maske identisch aus.
    h_pad = torch.cat([h, torch.full((12, C), 1e3)], dim=0)
    idx_pad = torch.cat([batch_idx, torch.arange(B).repeat_interleave(3)])
    mask_pad = torch.cat([node_mask, torch.zeros(12, dtype=torch.bool)])
    out_pad = pool(h_pad, idx_pad, mask_pad, B)
    check("maskierte Knoten aendern das Ergebnis nicht",
          float((out_pad - out).abs().max()) < 1e-10,
          f"max Abweichung {float((out_pad - out).abs().max()):.2e}")
    # Gegenprobe: OHNE Maske waere der Effekt riesig.
    out_nomask = pool(h_pad, idx_pad, torch.ones(N + 12, dtype=torch.bool), B)
    check("ohne Maske waere der Effekt gross (Test kann scheitern)",
          float((out_nomask - out).abs().max()) > 1.0,
          f"{float((out_nomask - out).abs().max()):.1f}")

    check("max-Pooling ignoriert maskierte Knoten ebenfalls",
          float((InvariantPooling("max")(h_pad, idx_pad, mask_pad, B)
                 - InvariantPooling("max")(h, batch_idx, node_mask, B)).abs().max()) < 1e-10)
    # Komplex ganz ohne gueltige Knoten: 0, kein -inf und kein NaN.
    empty = pool(h, batch_idx, torch.zeros(N, dtype=torch.bool), B)
    check("Komplex ohne gueltige Knoten gibt 0, nicht NaN",
          bool(torch.isfinite(empty).all()) and float(empty.abs().max()) == 0.0)

    print("\n3. Der Head")
    head = ConfidenceHead(C, hidden=16).double()
    logits = head(h, batch_idx, node_mask, B)
    check("Form [B]", logits.shape == (B,), str(tuple(logits.shape)))
    check("Ausgabe ist endlich", bool(torch.isfinite(logits).all()))
    # Startwert: die Nullinitialisierung der letzten Schicht macht die Ausgabe
    # unabhaengig von der Eingabe und gleich dem Bias.
    check("startet bei der Basisrate ~10 %, nicht bei 50 %",
          abs(float(torch.sigmoid(logits).mean()) - 0.10) < 0.01,
          f"{float(torch.sigmoid(logits).mean()):.3f}")

    print("\n4. Invarianz unter globaler Drehung")
    # Der Head sieht ausschliesslich l=0-Kanaele. Eine globale Drehung des
    # Komplexes laesst diese per Konstruktion unveraendert und veraendert nur
    # l>=1. Simuliert wird das, indem NUR die hoeheren Kanaele ersetzt werden.
    emb_rot = emb.clone()
    emb_rot[:, 1:, :] = torch.randn_like(emb_rot[:, 1:, :])
    l_before = head(extract_invariant_features(emb), batch_idx, node_mask, B)
    l_after = head(extract_invariant_features(emb_rot), batch_idx, node_mask, B)
    check("Logits haengen NICHT von den l>=1-Kanaelen ab",
          float((l_before - l_after).abs().max()) < 1e-12)
    # Gegenprobe: aendert man den l=0-Kanal, MUSS sich etwas aendern -- sonst
    # ignoriert der Head seine Eingabe komplett.
    torch.manual_seed(5)
    head.mlp[-1].weight.data.normal_(0, 0.1)     # Nullinit aufheben
    emb2 = emb.clone()
    emb2[:, 0, :] += 1.0
    d = float((head(extract_invariant_features(emb), batch_idx, node_mask, B)
               - head(extract_invariant_features(emb2), batch_idx, node_mask, B)).abs().max())
    check("aendert sich der l=0-Kanal, aendert sich das Logit", d > 1e-6, f"{d:.2e}")

    print("\n5. Verlustfunktion")
    rmsd = torch.tensor([0.5, 1.5, 3.0, 8.0])
    lo = torch.tensor([5.0, 5.0, -5.0, -5.0])          # perfekt
    hi = torch.tensor([-5.0, -5.0, 5.0, 5.0])          # invertiert
    check("perfekte Vorhersage hat kleinen Verlust",
          float(confidence_loss(lo, rmsd)) < 0.01, f"{float(confidence_loss(lo, rmsd)):.4f}")
    check("invertierte Vorhersage hat grossen Verlust",
          float(confidence_loss(hi, rmsd)) > 4.0, f"{float(confidence_loss(hi, rmsd)):.2f}")
    check("die Schwelle wirkt: 1.5 A zaehlt als Erfolg, 3.0 A nicht",
          float(confidence_loss(torch.tensor([5.0, 5.0, -5.0, -5.0]), rmsd, 2.0)) <
          float(confidence_loss(torch.tensor([5.0, -5.0, -5.0, -5.0]), rmsd, 2.0)))
    # pos_weight muss das Ungleichgewicht wirklich verschieben.
    pw = torch.tensor(9.0)
    check("pos_weight erhoeht den Verlust verpasster Positiver",
          float(confidence_loss(hi, rmsd, pos_weight=pw)) >
          float(confidence_loss(hi, rmsd)),
          "sonst lernt der Head, immer 'schlecht' zu sagen")

    print("\n6. Gradienten fliessen")
    head.zero_grad()
    loss = confidence_loss(head(h, batch_idx, node_mask, B), torch.rand(B) * 5)
    loss.backward()
    grads = [p.grad for p in head.parameters() if p.grad is not None]
    check("alle Parameter bekommen einen Gradienten",
          len(grads) == len(list(head.parameters()))
          and all(bool(torch.isfinite(g).all()) for g in grads),
          f"{len(grads)} Tensoren")

    print("\n" + "=" * 66)
    if FAILS:
        print(f"{len(FAILS)} CHECK(S) FAILED: {FAILS}")
        return 1
    print("Alle Checks bestanden.")
    return 0


def test_confidence_head() -> None:
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
