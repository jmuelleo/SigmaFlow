"""Test der reinen Rechenfunktionen aus arc/exp101_distance_audit.py.

Der Audit loest auf ARC eine Entscheidung aus ("EXP-102 bauen: ja/nein").
Ein Messinstrument, das diese Entscheidung traegt, muss vorher an Faellen mit
BEKANNTER Antwort geprueft sein.

Getestet wird NICHT die Datenladung (die braucht ARC-Daten und die
EXP-100-Codebasis), sondern die drei Funktionen, die das Ergebnis erzeugen:
`so3_angle_deg`, `karcher_mean`, `principal_axes`.

    python arc/test_exp101_distance_audit.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "SigmaFlow_Minimal" / "src"))

# Das Auditskript liegt in arc/ und ist kein Paket; es wird direkt geladen.
_spec = importlib.util.spec_from_file_location(
    "exp101_audit", _REPO / "arc" / "exp101_distance_audit.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
so3_angle_deg, karcher_mean, principal_axes = (
    _mod.so3_angle_deg, _mod.karcher_mean, _mod.principal_axes)

from sigmadock.diff import so3_utils  # noqa: E402

HAAR_MEDIAN_DEG = 132.3
FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'OK  ' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def main() -> int:
    torch.manual_seed(20260816)
    np.random.seed(20260816)
    M = 4000

    print("\n1. Winkelmass")
    # Gegen EXAKT bekannte Winkel ueber neun Groessenordnungen. Das trennt die
    # Formel von der Orthogonalitaetsabweichung zufaellig gezogener Matrizen:
    # hier ist die Eingabe konstruiert und der Sollwert analytisch bekannt.
    axis = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    worst, worst_th = 0.0, None
    for th in [1e-9, 1e-7, 1e-5, 1e-2, 0.5, 1.5, 3.0, np.pi - 1e-6]:
        got = float(so3_angle_deg(so3_utils.Exp(float(th) * axis)[0]))
        rel = abs(got - np.rad2deg(th)) / np.rad2deg(th)
        if rel > worst:
            worst, worst_th = rel, th
    check("relativ genau von 1e-9 rad bis pi", worst < 1e-7,
          f"groesster relativer Fehler {worst:.2e} bei theta={worst_th:.1e} rad")

    R_pi = torch.tensor([[-1.0, 0, 0], [0, -1.0, 0], [0, 0, 1.0]])
    check("180 Grad wird als 180 Grad gemessen",
          abs(float(so3_angle_deg(R_pi)) - 180.0) < 1e-3,
          f"{float(so3_angle_deg(R_pi)):.4f}")

    # Der float32-Selbstabstand war in einer frueheren Fassung 2.3e-2 Grad und
    # passte fast exakt zur Schranke sqrt(2*eps32) = 2.8e-2. Das sah nach einer
    # Grenze der Darstellung aus, war aber Ausloeschung in der Formel. Nach der
    # Umstellung auf ||R - I||_F ist der Fehler LINEAR in eps.
    R_a = so3_utils.sample_uniform(M).float()
    self32 = float(so3_angle_deg(R_a.transpose(-1, -2) @ R_a).abs().max())
    lin32 = float(np.rad2deg(3 * np.finfo(np.float32).eps / np.sqrt(2)))
    check("float32-Selbstabstand folgt der LINEAREN Schranke", self32 <= 5 * lin32,
          f"{self32:.2e} Grad, 3*eps32/sqrt(2) = {lin32:.2e} Grad")

    R_b = so3_utils.sample_uniform(M).float()
    med = float(np.median(so3_angle_deg(R_a.transpose(-1, -2) @ R_b).numpy()))
    # Stichprobenfehler des Medians ~ 1.253*sigma/sqrt(M), sigma ~ 33 Grad.
    # Die Toleranz MUSS daher kommen, nicht geraten sein.
    se = 1.253 * 33.0 / np.sqrt(M)
    check("Haar gegen Haar reproduziert den Median 132.3 Grad",
          abs(med - HAAR_MEDIAN_DEG) < 3 * se,
          f"{med:.2f} (erwartet {HAAR_MEDIAN_DEG}, 3*SE={3 * se:.2f})")

    print("\n2. Karcher-Mittel")
    R_star = so3_utils.Exp(torch.tensor([[0.4, -0.9, 0.2]])).float()[0]
    R_conc = R_star[None] @ so3_utils.Exp(0.15 * torch.randn(M, 3)).float()
    err = float(so3_angle_deg(karcher_mean(R_conc).transpose(-1, -2) @ R_star))
    check("findet ein bekanntes Zentrum", err < 1.0, f"{err:.3f} Grad Abweichung")

    # Gegenprobe: bei Haar-verteilten Daten ist das Funktional flach, der
    # Abstand zum Karcher-Mittel muss nahe der Haar-Referenz bleiben. Ohne
    # diesen Fall koennte ein Karcher-Mittel, das einfach R[0] zurueckgibt,
    # unbemerkt durchgehen.
    R_u = so3_utils.sample_uniform(M).float()
    med_u = float(np.median(so3_angle_deg(karcher_mean(R_u).transpose(-1, -2) @ R_u).numpy()))
    check("bringt bei Haar-Daten fast nichts", abs(med_u - HAAR_MEDIAN_DEG) < 12,
          f"{med_u:.1f} vs Haar {HAAR_MEDIAN_DEG}")

    print("\n3. Hauptachsen")
    X = np.random.randn(400, 3) @ np.diag([5.0, 2.0, 0.7])
    V = principal_axes(X)
    check("Spalten sind orthonormal",
          np.abs(V.T @ V - np.eye(3)).max() < 1e-9, f"{np.abs(V.T @ V - np.eye(3)).max():.2e}")
    check("rechtshaendig (det = +1)", abs(np.linalg.det(V) - 1.0) < 1e-9,
          f"det={np.linalg.det(V):.6f}")
    # Groesste Varianz zuerst: die Projektion auf Spalte 0 muss am breitesten sein.
    sd = (X - X.mean(0)) @ V
    check("nach Varianz absteigend sortiert",
          sd[:, 0].std() > sd[:, 1].std() > sd[:, 2].std(),
          f"{sd[:, 0].std():.2f} > {sd[:, 1].std():.2f} > {sd[:, 2].std():.2f}")
    # DER entscheidende Punkt: die Vorzeichenfixierung. Ohne sie waere die
    # "Heuristik" zufaellig zwischen vier gleichwertigen Rahmen - ein Fehler,
    # der sich als plausibles Ergebnis tarnt. Eine gespiegelte Punktwolke muss
    # deshalb denselben Rahmen liefern, nicht einen vorzeichenverdrehten.
    V2 = principal_axes(X.copy())
    check("deterministisch bei identischer Eingabe", np.abs(V - V2).max() < 1e-12)
    # Wird die Wolke starr gedreht, muss der Rahmen mitdrehen: V(QX) = Q V(X).
    Q = so3_utils.sample_uniform(1)[0].numpy()
    Vq = principal_axes(X @ Q.T)
    check("dreht mit der Punktwolke mit (Aequivarianz)",
          np.abs(Vq - Q @ V).max() < 1e-6, f"{np.abs(Vq - Q @ V).max():.2e}")

    print("\n" + "=" * 62)
    if FAILS:
        print(f"{len(FAILS)} CHECK(S) FAILED: {FAILS}")
        return 1
    print("Alle Checks bestanden.")
    return 0


def test_exp101_distance_audit() -> None:
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
