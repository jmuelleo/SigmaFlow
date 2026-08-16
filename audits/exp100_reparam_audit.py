"""Audit von EXP-100 (Zustandsreparametrisierung).

DIE LEITFRAGE
  Ist EXP-100 nur eine KOORDINATENWAHL fuer dasselbe generative Modell, oder
  ein ANDERES Modell? Davon haengt ab, ob es als Ablation gegen Minimal
  interpretierbar ist oder als eigenstaendige Methode gefuehrt werden muss.

  Die Antwort ist zweigeteilt, und der Test trennt die beiden Teile sauber:

  (1) Auf SO(3) ist es exakt eine Umkoordinatisierung. Aus
          pose_min(R)   = R * Y_c + p          (Y_c = zentrierte Kristallgeometrie)
          pose_100(R')  = R' * C_F + p         (C_F = zentrierter Referenzkonformer)
      und  Y_c ~= R_1F C_F  folgt  R' = R * R_1F: eine RECHTStranslation um eine
      fragmentabhaengige Konstante. Rechtstranslationen sind Isometrien der
      bi-invarianten Metrik, und Haar ist rechtsinvariant -- Quelle, Geodaeten
      und CFM-Ziel bleiben formgleich.

  (2) Auf der GEOMETRIE ist es KEINE Umkoordinatisierung. Die zu platzierende
      Punktwolke wechselt von der Kristallgeometrie des Fragments zum
      ETKDG-Konformer. Beide haben verschiedene Bindungslaengen, -winkel und
      Torsionen. Damit aendert sich, was das Modell ueberhaupt platzieren kann,
      und es entsteht ein RMSD-Boden, der auch bei perfekter Platzierung bleibt.

  Dieser Boden ist der eigentliche Preis von EXP-100 und wird unten gemessen.

    python audits/exp100_reparam_audit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
_EXP = _ROOT / "SigmaFlow_FM_Specific" / "EXP-100_state_reparam"
sys.path.insert(0, str(_EXP / "src"))

from sigmadock.diff import so3_utils  # noqa: E402
from sigmadock.diff.state_reparam import (  # noqa: E402
    kabsch_residual,
    kabsch_rotation,
)

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'OK  ' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def geo_deg(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Geodaetenwinkel in Grad, ausloeschungsfrei (siehe exp101_distance_audit)."""
    rel = (A.transpose(-1, -2).to(torch.float64) @ B.to(torch.float64))
    tr = rel.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos_w = ((tr - 1.0) / 2.0).clamp(-1.0, 1.0)
    eye = torch.eye(3, dtype=rel.dtype).expand_as(rel)
    chord = torch.linalg.matrix_norm(rel - eye, ord="fro")
    w = torch.where(cos_w >= 0.0,
                    2.0 * torch.arcsin((chord / (2 * np.sqrt(2.0))).clamp(-1, 1)),
                    torch.arccos(cos_w))
    return torch.rad2deg(w)


def main() -> int:  # noqa: C901
    torch.manual_seed(20260816)
    np.random.seed(20260816)

    print("\n1. Kabsch: Konvention und Korrektheit")
    n_frag, n_at = 400, 12
    C = torch.randn(n_frag, n_at, 3, dtype=torch.float64)
    C = C - C.mean(1, keepdim=True)
    R_true = so3_utils.sample_uniform(n_frag).to(torch.float64)
    Y = C @ R_true.transpose(-1, -2)                    # Y = R_true C, zentriert

    R_hat = kabsch_rotation(C, Y)
    err = float(geo_deg(R_hat, R_true).max())
    check("findet eine BEKANNTE Rotation zurueck", err < 1e-6,
          f"max {err:.2e} Grad")
    check("Ergebnis ist orthogonal",
          float((R_hat.transpose(-1, -2) @ R_hat
                 - torch.eye(3, dtype=torch.float64)).abs().max()) < 1e-10)
    check("Determinante ist +1 (keine Spiegelung)",
          float((torch.linalg.det(R_hat) - 1).abs().max()) < 1e-10)

    # Konventionsprobe: C @ R^T richtet auf Y aus, NICHT R @ C.
    # Genau dieser Richtungsfehler trat laut Docstring waehrend der Entwicklung
    # auf und lieferte ein Residuum in der Groesse des Fragments.
    res_right = float(kabsch_residual(C, Y, R_hat).max())
    res_wrong = float(kabsch_residual(C, Y, R_hat.transpose(-1, -2)).max())
    check("C @ R^T ist die richtige Richtung", res_right < 1e-9, f"{res_right:.2e}")
    check("die transponierte Richtung ist deutlich schlechter (Test kann scheitern)",
          res_wrong > 0.1, f"Residuum {res_wrong:.3f} vs {res_right:.2e}")

    # Spiegelung: Kabsch darf sie NIE zurueckgeben, auch wenn sie besser passt.
    Y_mirror = Y.clone()
    Y_mirror[..., 2] *= -1
    R_m = kabsch_rotation(C, Y_mirror)
    check("auch bei gespiegeltem Ziel bleibt det = +1",
          float((torch.linalg.det(R_m) - 1).abs().max()) < 1e-10)

    print("\n2. float64 ist noetig, nicht Kosmetik")
    # Behauptung im Docstring: float32 liefert bei C == Y bis zu 6.1e-2 Grad
    # statt 0. Nachgerechnet.
    Cd = C[:200]
    R64 = kabsch_rotation(Cd, Cd)
    e64 = float(geo_deg(R64, torch.eye(3, dtype=torch.float64).expand_as(R64)).max())
    H32 = Cd.float().transpose(-1, -2) @ Cd.float()
    U, _, Vh = torch.linalg.svd(H32)                  # bewusst in float32
    R32 = Vh.transpose(-1, -2) @ U.transpose(-1, -2)
    e32 = float(geo_deg(R32, torch.eye(3).expand_as(R32)).max())
    check("float64 liefert bei C == Y praktisch die Identitaet", e64 < 1e-4,
          f"{e64:.2e} Grad")
    check("float32 waere um Groessenordnungen schlechter (Begruendung belegt)",
          e32 > 20 * max(e64, 1e-12), f"float32 {e32:.2e} vs float64 {e64:.2e} Grad")

    print("\n3. Ist es auf SO(3) nur eine Umkoordinatisierung?")
    # Behauptung: R_100 = R_min * R_1F, also eine Rechtstranslation.
    R_1F = so3_utils.sample_uniform(n_frag).to(torch.float64)   # Kabsch-Ziele
    R_min = so3_utils.sample_uniform(n_frag).to(torch.float64)  # ein Zustand in Minimal
    R_100 = R_min @ R_1F

    # (a) Die Pose ist dieselbe:  R_min * (R_1F C) == R_100 * C
    Cc = C - C.mean(1, keepdim=True)
    pose_min = (Cc @ R_1F.transpose(-1, -2)) @ R_min.transpose(-1, -2)
    pose_100 = Cc @ R_100.transpose(-1, -2)
    check("beide Parametrisierungen erzeugen dieselbe Pose",
          float((pose_min - pose_100).abs().max()) < 1e-10,
          f"{float((pose_min - pose_100).abs().max()):.2e}")

    # (b) Rechtstranslation ist eine Isometrie: Abstaende bleiben erhalten.
    A, B = so3_utils.sample_uniform(300).to(torch.float64), so3_utils.sample_uniform(300).to(torch.float64)
    Q = so3_utils.sample_uniform(1)[0].to(torch.float64)
    d_before = geo_deg(A, B)
    d_after = geo_deg(A @ Q, B @ Q)
    check("Rechtstranslation erhaelt die Geodaetendistanz",
          float((d_before - d_after).abs().max()) < 1e-6,
          f"max Abweichung {float((d_before - d_after).abs().max()):.2e} Grad")

    # (c) Haar ist rechtsinvariant -> die QUELLE aendert sich nicht.
    # Kolmogorow-Smirnow auf der Winkelverteilung gegen die Haar-Referenz.
    from scipy import stats
    N = 20000
    H = so3_utils.sample_uniform(N).to(torch.float64)
    ang = np.deg2rad(geo_deg(torch.eye(3, dtype=torch.float64).expand_as(H), H).numpy())
    angQ = np.deg2rad(geo_deg(torch.eye(3, dtype=torch.float64).expand_as(H), H @ Q).numpy())
    cdf = lambda w: (w - np.sin(w)) / np.pi          # noqa: E731  Haar-Winkel-CDF
    p_h = stats.kstest(ang, cdf).pvalue
    p_q = stats.kstest(angQ, cdf).pvalue
    check("Haar bleibt unter Rechtstranslation Haar (KS-Test)",
          p_q > 0.01 and p_h > 0.01, f"p(Haar)={p_h:.3f}, p(Haar*Q)={p_q:.3f}")

    print("\n   -> Auf SO(3) ist EXP-100 exakt eine Umkoordinatisierung:")
    print("      dieselbe Pose, dieselbe Metrik, dieselbe Quellverteilung.")

    print("\n4. Und auf der GEOMETRIE? Der Preis von EXP-100")
    # Die zu platzierende Punktwolke wechselt von der Kristall- zur
    # Konformergeometrie. Selbst bei perfekter starrer Platzierung bleibt der
    # Kabsch-Restfehler als RMSD-Boden stehen. Gemessen an echten Molekuelen:
    # zwei ETKDG-Konformere desselben Molekuels als Stellvertreter fuer
    # "Konformer gegen Kristallpose".
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")

    smis = ["CCOc1ccccc1C(=O)NC", "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
            "COc1cc2c(cc1OC)CCN(C2)C(=O)c1ccccc1", "CN1CCC(CC1)Oc1ccc(Cl)cc1",
            "OC(=O)c1ccccc1Nc1ccccc1", "CCN(CC)CCNC(=O)c1ccc(N)cc1"]
    residuals, sizes = [], []
    for smi in smis:
        mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        ps = AllChem.ETKDGv3()
        ps.randomSeed = 0xBEEF
        if AllChem.EmbedMultipleConfs(mol, numConfs=6, params=ps) == 0:
            continue
        AllChem.MMFFOptimizeMoleculeConfs(mol)
        mol = Chem.RemoveHs(mol)
        if mol.GetNumConformers() < 2:
            continue
        ref = torch.tensor(mol.GetConformer(0).GetPositions(), dtype=torch.float64)
        for ci in range(1, mol.GetNumConformers()):
            tgt = torch.tensor(mol.GetConformer(ci).GetPositions(), dtype=torch.float64)
            # Fragmentgroesse: SigmaDock zerlegt an Torsionsbindungen, typische
            # Fragmente haben 5-10 Schweratome. Wir werten gleitende Fenster aus.
            for k in (5, 8):
                if mol.GetNumAtoms() < k:
                    continue
                for s in range(0, mol.GetNumAtoms() - k + 1, max(k // 2, 1)):
                    a, b = ref[s:s + k], tgt[s:s + k]
                    ac, bc = a - a.mean(0), b - b.mean(0)
                    R = kabsch_rotation(ac[None], bc[None])[0]
                    residuals.append(float(kabsch_residual(ac[None], bc[None], R[None])[0]))
                    sizes.append(k)
    res = np.array(residuals)
    print(f"      {len(res)} Fragmentfenster aus {len(smis)} Molekuelen")
    print(f"      Kabsch-Restfehler (A): median {np.median(res):.3f}  "
          f"p90 {np.percentile(res, 90):.3f}  max {res.max():.3f}")
    check("es gibt ueberhaupt einen messbaren Boden", np.median(res) > 1e-3,
          f"median {np.median(res):.3f} A")
    check("der Boden liegt klar unter der 2-A-Erfolgsschwelle",
          np.percentile(res, 90) < 2.0, f"p90 {np.percentile(res, 90):.3f} A")
    print("      -> Bei starrer Platzierung eines Konformerfragments statt des")
    print("         Kristallfragments bleibt dieser Fehler stehen. Er ist klein")
    print("         gegen die gemessenen ~5 A RMSD, aber er ist NICHT null und")
    print("         gehoert in jede Ergebnisdiskussion zu EXP-100.")

    print("\n5. Was EXP-100 NICHT anfasst")
    import inspect
    from sigmadock.diff.so3_flow_matcher import SO3_FlowMatcher as SO3_100
    from sigmadock.diff.r3_flow_matcher import R3_FlowMatcher as R3_100
    sys.path.insert(0, str(_ROOT / "SigmaFlow_Minimal" / "src"))
    # Vergleich auf Quelltextebene: die Flow-Matcher muessen identisch sein.
    min_so3 = (_ROOT / "SigmaFlow_Minimal/src/sigmadock/diff/so3_flow_matcher.py").read_text(encoding="utf-8")
    exp_so3 = (_EXP / "src/sigmadock/diff/so3_flow_matcher.py").read_text(encoding="utf-8")
    min_r3 = (_ROOT / "SigmaFlow_Minimal/src/sigmadock/diff/r3_flow_matcher.py").read_text(encoding="utf-8")
    exp_r3 = (_EXP / "src/sigmadock/diff/r3_flow_matcher.py").read_text(encoding="utf-8")
    check("SO(3)-Flow-Matcher byte-identisch zu Minimal", min_so3 == exp_so3)
    check("R3-Flow-Matcher byte-identisch zu Minimal", min_r3 == exp_r3)
    check("Quelle unveraendert: sample_init zieht weiter Haar",
          "sample_uniform" in inspect.getsource(SO3_100.sample_init))
    check("Quelle unveraendert: R3 zieht weiter N(0,I)",
          "randn" in inspect.getsource(R3_100.sample_init))

    n_net_min = len(list((_ROOT / "SigmaFlow_Minimal/src/sigmadock/net").glob("*.py")))
    n_net_exp = len(list((_EXP / "src/sigmadock/net").glob("*.py")))
    net_same = all(
        (_ROOT / "SigmaFlow_Minimal/src/sigmadock/net" / f.name).read_bytes() == f.read_bytes()
        for f in (_EXP / "src/sigmadock/net").glob("*.py"))
    check("alle net/-Dateien byte-identisch -> gleiche Architektur, gleiche "
          "Parameterzahl", net_same and n_net_min == n_net_exp,
          f"{n_net_exp} Dateien")

    print("\n" + "=" * 68)
    if FAILS:
        print(f"{len(FAILS)} CHECK(S) FAILED: {FAILS}")
        return 1
    print("Alle Checks bestanden.")
    return 0


def test_exp100_reparam() -> None:
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
