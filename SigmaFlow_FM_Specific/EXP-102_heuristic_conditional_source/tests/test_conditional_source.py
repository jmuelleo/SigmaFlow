"""Tests der EXP-102-Quellverteilung.

Der wichtigste Test ist der KONTROLLTEST: `source_mode="haar"` muss EXP-100
bit-identisch reproduzieren. Ist das nicht der Fall, ist das Experiment keine
Ein-Faktor-Ablation mehr, und ein gemessener Unterschied waere nicht mehr der
Quelle zuzuschreiben.

    python SigmaFlow_FM_Specific/EXP-102_heuristic_conditional_source/tests/test_conditional_source.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_EXP = _HERE.parent
_ROOT = _EXP.parents[1]
sys.path.insert(0, str(_EXP / "src"))

from sigmadock.diff import so3_utils  # noqa: E402
from sigmadock.diff.conditional_source import (  # noqa: E402
    HAAR_MEDIAN_DEG,
    pocket_alignment_rotation,
    principal_axes,
    sample_conditional_init,
    sigma_from_median_deg,
)
from sigmadock.diff.so3_flow_matcher import SO3_FlowMatcher  # noqa: E402

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'OK  ' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def ang_deg(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    rel = A.transpose(-1, -2).to(torch.float64) @ B.to(torch.float64)
    tr = rel.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos_w = ((tr - 1.0) / 2.0).clamp(-1.0, 1.0)
    eye = torch.eye(3, dtype=rel.dtype).expand_as(rel)
    chord = torch.linalg.matrix_norm(rel - eye, ord="fro")
    w = torch.where(cos_w >= 0.0,
                    2.0 * torch.arcsin((chord / (2 * np.sqrt(2.0))).clamp(-1, 1)),
                    torch.arccos(cos_w))
    return torch.rad2deg(w)


def main() -> int:  # noqa: C901
    n = 20000
    fm = SO3_FlowMatcher()

    print("\n1. KONTROLLE: haar reproduziert EXP-100 bit-identisch")
    # Zweimal derselbe Seed, einmal ueber den alten Pfad (sample_uniform),
    # einmal ueber den neuen mit R_centre=None. Muss BITGLEICH sein.
    np.random.seed(4242)
    a = so3_utils.sample_uniform(500).to("cpu", dtype=torch.float32)
    np.random.seed(4242)
    b = fm.sample_init(500, "cpu")                       # R_centre default None
    check("sample_init(None) == so3_utils.sample_uniform, bitgleich",
          torch.equal(a, b), f"max |a-b| = {float((a - b).abs().max()):.1e}")
    np.random.seed(4242)
    c = sample_conditional_init(500, None, 90.0, "cpu")
    check("sample_conditional_init(R_centre=None) ebenfalls bitgleich",
          torch.equal(a, c))

    print("\n2. Der Medianwinkel ist der eingestellte")
    # sigma wird ueber die Chi-3-Median-Relation gesetzt. Geprueft wird, dass
    # der EMPIRISCHE Median die Vorgabe trifft -- nicht die Formel gegen sich
    # selbst, sondern gegen gezogene Stichproben.
    R_c = so3_utils.Exp(torch.tensor([[0.3, -0.7, 0.2]], dtype=torch.float64))[0].float()
    for target in (10.0, 30.0, 60.0, 90.0):
        torch.manual_seed(7)
        R0 = sample_conditional_init(n, R_c, target, "cpu")
        med = float(np.median(ang_deg(R_c.expand_as(R0), R0).numpy()))
        # Stichprobenfehler des Medians ~ 1.253*sigma/sqrt(n); grosszuegig 1 %.
        ok = abs(med - target) / target < 0.02
        check(f"Median {target:5.1f} Grad wird getroffen", ok,
              f"gemessen {med:.2f}")

    print("\n3. Grenzverhalten")
    check("sigma waechst monoton im Medianwinkel",
          sigma_from_median_deg(30) < sigma_from_median_deg(90) < sigma_from_median_deg(150))
    for bad in (0.0, -5.0, 180.0, 200.0):
        try:
            sigma_from_median_deg(bad)
            check(f"unzulaessiger Medianwinkel {bad} wird abgelehnt", False)
        except ValueError:
            check(f"unzulaessiger Medianwinkel {bad} wird abgelehnt", True)

    print("\n4. Die Quelle ist WIRKLICH konzentriert (gegen Haar)")
    torch.manual_seed(11)
    np.random.seed(11)
    R_conc = sample_conditional_init(n, R_c, 40.0, "cpu")
    R_haar = so3_utils.sample_uniform(n).float()
    m_conc = float(np.median(ang_deg(R_c.expand_as(R_conc), R_conc).numpy()))
    m_haar = float(np.median(ang_deg(R_c.expand_as(R_haar), R_haar).numpy()))
    check("konzentriert liegt klar unter Haar", m_conc < m_haar - 50,
          f"{m_conc:.1f} gegen {m_haar:.1f} (Haar-Referenz {HAAR_MEDIAN_DEG})")
    check("Haar-Kontrolle trifft die Referenz", abs(m_haar - HAAR_MEDIAN_DEG) < 2.0,
          f"{m_haar:.1f}")

    print("\n5. Hauptachsen: die Vorzeichenfixierung")
    X = torch.randn(400, 3, dtype=torch.float64) @ torch.diag(
        torch.tensor([5.0, 2.0, 0.7], dtype=torch.float64))
    V = principal_axes(X)
    check("orthonormal", float((V.T @ V - torch.eye(3, dtype=V.dtype)).abs().max()) < 1e-9)
    check("det = +1", abs(float(torch.linalg.det(V)) - 1.0) < 1e-9)
    check("deterministisch", torch.equal(V, principal_axes(X.clone())))
    # Aequivarianz: dreht die Wolke, dreht der Rahmen mit. Das ist die
    # Eigenschaft, ohne die die Heuristik keine Heuristik waere.
    Q = so3_utils.sample_uniform(1)[0]
    check("aequivariant: V(QX) == Q V(X)",
          float((principal_axes(X @ Q.T) - Q @ V).abs().max()) < 1e-6,
          f"{float((principal_axes(X @ Q.T) - Q @ V).abs().max()):.2e}")

    print("\n6. Taschenausrichtung")
    lig = torch.randn(20, 3, dtype=torch.float64) @ torch.diag(
        torch.tensor([4.0, 1.5, 0.6], dtype=torch.float64))
    Qtrue = so3_utils.sample_uniform(1)[0].to(torch.float64)
    pkt = lig @ Qtrue.T + torch.randn(20, 3, dtype=torch.float64) * 0.01
    Q_hat = pocket_alignment_rotation(lig, pkt)
    check("richtet den Liganden auf eine gedrehte Kopie aus",
          float(ang_deg(Q_hat[None], Qtrue[None])[0]) < 5.0,
          f"{float(ang_deg(Q_hat[None], Qtrue[None])[0]):.2f} Grad")
    check("Ergebnis ist eine echte Rotation",
          abs(float(torch.linalg.det(Q_hat)) - 1.0) < 1e-9)
    # Zu wenige Atome: Identitaet statt Absturz oder Zufall.
    check("weniger als 3 Atome -> Identitaet",
          float((pocket_alignment_rotation(lig[:2], pkt)
                 - torch.eye(3, dtype=lig.dtype)).abs().max()) < 1e-12)

    print("\n7. Der Pfad benutzt die Quelle tatsaechlich")
    # Ohne diesen Test koennte conditional_probability_path das Zentrum
    # stillschweigend ignorieren und alles obige waere folgenlos.
    R_1 = so3_utils.sample_uniform(4000).float()
    t = torch.full((4000,), 0.01)
    torch.manual_seed(3)
    R_t_haar, _ = fm.conditional_probability_path(R_1, t)
    torch.manual_seed(3)
    R_t_conc, _ = fm.conditional_probability_path(R_1, t, R_c, 20.0)
    d_haar = float(np.median(ang_deg(R_c.expand_as(R_t_haar), R_t_haar).numpy()))
    d_conc = float(np.median(ang_deg(R_c.expand_as(R_t_conc), R_t_conc).numpy()))
    check("bei t~0 liegt der Pfad mit Zentrum nahe am Zentrum",
          d_conc < 40 and d_haar > 100,
          f"konzentriert {d_conc:.1f} Grad, Haar {d_haar:.1f} Grad")

    print("\n" + "=" * 66)
    if FAILS:
        print(f"{len(FAILS)} CHECK(S) FAILED: {FAILS}")
        return 1
    print("Alle Checks bestanden.")
    return 0


def test_conditional_source() -> None:
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
