"""Die drei Arme, bewertet nach dem Protokoll des SigmaDock-Papers.

WAS DAS PAPER MACHT (Prat et al. 2026, Anhang F.2 und Abschnitt I.2)
    Ranker    s_i = -b_i * p_i^4 ueber genau fuenf stereochemische Checks.
    Kennzahl  "Top-1 (%) as our key performance metric", berichtet fuer
              (RMSD<2), PB-Validitaet und (RMSD<2 & PB-Valid).
    Aufbau    generierter Konformer, 20-30 Integrationsschritte, N_seeds
              als Parameter; im Paper vor allem N_seeds = 10 und 40.
    Oracle    wird als empirische Obergrenze mitberichtet.

WAS HIER ABWEICHT UND WARUM
    Unsere Laeufe haben 25 Integrationsschritte, das liegt in der vom Paper
    genannten Spanne. Alles andere ist identisch parametrisiert.

    Das Paper berichtet KEINEN Rankervergleich; deshalb entsteht dort keine
    Zirkularitaet. Hier steht die reine Vinardo-Energie zum Vergleich
    daneben, und die beiden Validitaetsziele sind als teilweise zirkulaer
    markiert: der Ranker enthaelt vier ligandenintrinsische Checks und einen
    Proteincheck, misst sich dort also teilweise selbst.

AUFRUF   python paper_protokoll.py
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, ".")
from paper_ranker import ARME, BETA, ZELLEN, lade, oracle, top1  # noqa: E402

# Das Paper-Setup ist die erste Zelle: generierter Konformer, 25 Schritte.
ZELLE, KONF, NS = ZELLEN[0]

# Die Seedzahlen, fuer die das Paper Zahlen nennt, plus unsere volle Zahl.
KS = (1, 5, 10, 20, 40, 80)

ZIELE = (
    ("rmsd", "RMSD < 2 A", ""),
    ("vm", "PB-Valid (volle Batterie)", "  teilweise zirkulaer"),
    ("jm", "RMSD < 2 A & PB-Valid", "  teilweise zirkulaer"),
)


def main() -> int:
    D = {a: lade(*KONF[a], NS) for a in ARME}
    keys = sorted(set.intersection(*[set(D[a][0]) & set(D[a][1]) & set(D[a][4])
                                     for a in ARME]))
    cids = sorted({c for c, _ in keys})
    sds = sorted({s for _, s in keys})
    K = len(sds)

    print("=" * 78)
    print(f"  PROTOKOLL DES PAPERS  --  {ZELLE}")
    print(f"  {len(cids)} Komplexe x {K} Seeds,  Ranker s_i = -b_i * p_i^{BETA:.0f}")
    print("=" * 78)

    M = {}
    for a in ARME:
        R, P5, VO, VM, B = D[a]
        M[a] = {
            "rmsd": np.array([[R[(c, s)] < 2.0 for s in sds] for c in cids]),
            "vm": np.array([[VM[(c, s)] for s in sds] for c in cids]),
            "b": np.array([[B[(c, s)] for s in sds] for c in cids]),
            "p5": np.array([[P5[(c, s)] for s in sds] for c in cids]),
        }
        M[a]["jm"] = M[a]["rmsd"] & M[a]["vm"]
        M[a]["paper"] = -M[a]["b"] * (M[a]["p5"] ** BETA)

    for ziel, lab, zirk in ZIELE:
        print(f"\n  Zielgroesse: {lab}{zirk}")
        kopf = "".join(f"{'k=' + str(k):>9}" for k in KS if k <= K)
        print(f"    {'Arm':<11}{kopf}")
        print("    " + "-" * (11 + 9 * len([k for k in KS if k <= K])))

        for a in ARME:
            H = M[a][ziel]
            zeile = "".join(f"{top1(M[a]['paper'], H, k, True):>8.2f}%"
                            for k in KS if k <= K)
            print(f"    {a:<11}{zeile}")

        # Oracle als Obergrenze, wie im Paper mitberichtet.
        print(f"    {'(Oracle)':<11}", end="")
        for k in (k for k in KS if k <= K):
            o = max(oracle(M[a][ziel], k) for a in ARME)
            print(f"{o:>8.2f}%", end="")
        print("   bester Arm")

    # Gegenprobe ohne Zirkularitaet: derselbe Ranker gegen die reine Energie,
    # gemessen an RMSD allein. Das Paper stellt diesen Vergleich nicht an.
    print("\n" + "=" * 78)
    print("  GEGENPROBE  --  Ranker gegen Ranker, nur gegen RMSD < 2 A")
    print("  (RMSD steckt in keinem Ranker, deshalb hier zirkularitaetsfrei)")
    print("=" * 78)
    print(f"    {'Arm':<11}{'k':>4}{'Zufall':>9}{'Vinardo':>10}{'Paper s_i':>11}"
          f"{'Oracle':>9}")
    for a in ARME:
        H = M[a]["rmsd"]
        for k in (10, 40, 80):
            if k > K:
                continue
            print(f"    {a if k == 10 else '':<11}{k:>4}{100*H.mean():>8.2f}%"
                  f"{top1(M[a]['b'], H, k, False):>9.2f}%"
                  f"{top1(M[a]['paper'], H, k, True):>10.2f}%"
                  f"{oracle(H, k):>8.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
