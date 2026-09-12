"""Oracle@K und Zufallsauswahl je Arm, Schrittzahl und Snapshot.

EXAKT STATT SIMULIERT
    Fuer einen Komplex mit n Posen, davon m erfolgreich, ist die
    Wahrscheinlichkeit, dass eine zufaellige K-Teilmenge mindestens einen
    Treffer enthaelt, 1 - C(n-m, K) / C(n, K). Das ist geschlossen
    berechenbar; ein Bootstrap ueber Ziehungen waere nur ungenauer.

    Oracle@K heisst hier: K Posen ziehen und die BESTE nehmen, gemessen an der
    wahren RMSD. Das ist die Obergrenze fuer jeden denkbaren Ranker. Die
    Zufallsauswahl (K=1) ist die Untergrenze. Jeder echte Ranker, auch GNINA,
    liegt dazwischen.

UNSICHERHEIT
    Cluster-Bootstrap ueber Komplexe, gleiche Konvention wie sonst.
"""
import math
import pathlib

import numpy as np
import pandas as pd

from arme import vorhanden
from kurve_daten import SEED, zelle_laden

HIER = pathlib.Path(__file__).resolve().parent
B = 4000


def oracle_je_komplex(m: pd.DataFrame, spalte: str, K: int) -> np.ndarray:
    """P(mindestens ein Treffer in K gezogenen Posen), je Komplex."""
    g = m.groupby("complex")[spalte].agg(["sum", "count"])
    out = np.empty(len(g))
    for i, (treffer, n) in enumerate(zip(g["sum"].to_numpy(int),
                                         g["count"].to_numpy(int))):
        k = min(K, n)
        misserfolge = n - treffer
        # C(misserfolge, k) / C(n, k); ist misserfolge < k, ist ein Treffer
        # unvermeidlich und die Wahrscheinlichkeit 1.
        p_kein = (math.comb(misserfolge, k) / math.comb(n, k)
                  if misserfolge >= k else 0.0)
        out[i] = 1.0 - p_kein
    return out


def ki(werte: np.ndarray, B=B, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(werte)
    z = werte[rng.integers(0, n, size=(B, n))].mean(axis=1)
    return 100 * np.percentile(z, 2.5), 100 * np.percentile(z, 97.5)


ZELLEN = {}
for arm, konf in vorhanden().items():
    letzte = max(konf["position"], key=lambda t: konf["position"][t][1])
    for nfe in (25, 5):
        treffer = [d.name for d in konf["redock"].iterdir()
                   if d.is_dir() and d.name.startswith(f"{letzte}__nfe{nfe}__")]
        ZELLEN[(arm, nfe)] = (zelle_laden(treffer[0], arm),
                              konf["position"][letzte][1])

for spalte, titel in (("u2", "RMSD < 2 A"),
                      ("u2_prot", "< 2 A und PB-valide mit Protein")):
    print(f"\n================ {titel} ================")
    print(f"{'Arm':<20}{'nfe':>4}{'Ep':>5}" + "".join(f"{'K='+str(k):>9}"
                                                      for k in (1,2,3,5,8,10)))
    for (arm, nfe), (m, ep) in ZELLEN.items():
        zeile = f"{arm:<20}{nfe:>4}{ep:>5}"
        for K in (1, 2, 3, 5, 8, 10):
            zeile += f"{100 * oracle_je_komplex(m, spalte, K).mean():8.2f}%"
        print(zeile)

# Die gematchte Frage: 8 Seeds bei 5 Schritten gegen 2 Seeds bei 25.
print("\n================ Gematchter Sampling-Aufwand ================")
print("8 Seeds x 5 Schritte = 73,0 min   gegen   2 Seeds x 25 Schritte = 71,2 min")
for spalte, titel in (("u2", "RMSD < 2 A"), ("u2_prot", "< 2 A und PB+Protein")):
    a = oracle_je_komplex(ZELLEN[("SigmaFlow-Minimal", 5)][0], spalte, 8)
    b = oracle_je_komplex(ZELLEN[("SigmaDock", 25)][0], spalte, 2)
    lo_a, hi_a = ki(a)
    lo_b, hi_b = ki(b)
    # gepaart: dieselben Komplexe
    gem = sorted(set(ZELLEN[("SigmaFlow-Minimal", 5)][0]["complex"]) &
                 set(ZELLEN[("SigmaDock", 25)][0]["complex"]))
    d = a - b
    rng = np.random.default_rng(SEED)
    zieh = rng.integers(0, len(d), size=(B, len(d)))
    boot = 100 * d[zieh].mean(axis=1)
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    print(f"\n{titel}")
    print(f"  Minimal, 5 Schritte, Oracle@8 : {100*a.mean():6.2f}%  [{lo_a:5.2f},{hi_a:5.2f}]")
    print(f"  SigmaDock, 25 Schritte, Oracle@2: {100*b.mean():6.2f}%  [{lo_b:5.2f},{hi_b:5.2f}]")
    print(f"  Differenz (Minimal - SigmaDock) : {100*d.mean():+6.2f} pp  "
          f"[{np.percentile(boot,2.5):+5.2f},{np.percentile(boot,97.5):+5.2f}]  "
          f"p = {max(p, 1/B):.4f}")
