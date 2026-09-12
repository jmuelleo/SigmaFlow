"""Konsens-Ranking: erst den besten Cluster waehlen, dann darin die beste Pose.

DAS PROTOKOLL
    1. K Posen ziehen (40 bei 25 Schritten, 200 bei 5).
    2. Optional filtern -- gar nicht, auf die FUENF Checks des Mixed Scores,
       oder auf die vollen 24 PoseBusters-Pruefungen.
    3. Clustern (vollstaendige Verkettung, 1 oder 2 Angstroem).
    4. Je Cluster den MITTLEREN gnina-Score bilden.
    5. Den Cluster mit dem hoechsten Mittel waehlen.
    6. Darin die Pose mit dem hoechsten gnina-Score nehmen.

DIE IDEE DAHINTER
    Der uebliche Ranker nimmt die eine bestbewertete Pose. Ist der Score
    verrauscht, gewinnt gelegentlich ein Ausreisser -- eine Pose, die zufaellig
    hoch punktet, ohne dass die Umgebung sie stuetzt. Das Konsensprotokoll
    verlangt stattdessen, dass die ganze GRUPPE gut bewertet ist, und nimmt
    erst dann deren besten Vertreter.

    Das ist ein Mittelwert-gegen-Maximum-Tausch: der Mittelwert ist robuster
    gegen Ausreisser, verschenkt aber die Information des Maximums. Ob sich
    das lohnt, haengt daran, wie verrauscht der Score ist -- und das ist eine
    empirische Frage, keine prinzipielle.

WARUM DER MITTLERE SCORE UND NICHT DER MEDIAN
    Wie vom Nutzer vorgegeben. Der Median waere robuster, aber der Mittelwert
    ist das, was "die Gruppe ist insgesamt gut" am direktesten ausdrueckt.
    Eine Variante mit Median ist ueber --mittel median zu haben.

ACHTUNG BEI GROSSEN CLUSTERN
    Ein Cluster mit zwei Posen hat leicht ein hohes Mittel; einer mit dreissig
    kaum. Das Protokoll bevorzugt also kleine Cluster. `--min-cluster` setzt
    eine Mindestgroesse dagegen.

Aufruf:
    python SigmaFlow_Variants/konsens.py --satz astex
    python SigmaFlow_Variants/konsens.py --satz pb308 --min-cluster 3
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

_HIER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HIER)
sys.path.insert(0, os.path.dirname(_HIER))
import posencache  # noqa: E402
from zellen import SAETZE, lade_zelle  # noqa: E402
from SigmaFlow_Evaluation.ranking.heuristic_score import PB_CHECKS  # noqa: E402

from scipy.cluster.hierarchy import fcluster, linkage  # noqa: E402
from scipy.spatial.distance import squareform  # noqa: E402

LANG = {"SigmaDock": "SigmaDock", "Minimal": "SF-Minimal",
        "Separate": "SF-Separate"}
ZELLEN = [("SigmaDock", 25), ("Minimal", 25), ("Separate", 25),
          ("Minimal", 5), ("Separate", 5), ("SigmaDock", 5)]
FILTER = [("keiner", None), ("5 Checks", "f5"), ("volle 24", "f24")]

p = argparse.ArgumentParser()
p.add_argument("--satz", choices=sorted(SAETZE), default="astex")
p.add_argument("--schwellen", default="1.0,2.0")
p.add_argument("--mittel", default="mean", choices=["mean", "median"])
p.add_argument("--min-cluster", type=int, default=1,
               help="Cluster unter dieser Groesse kommen nicht als Sieger infrage")
p.add_argument("--ziel", default="beides", choices=["acc", "beides"])
p.add_argument("--out", default=None)
a = p.parse_args()
SCHW = [float(x) for x in a.schwellen.split(",")]
AUS = a.out or os.path.join(_HIER, f"konsens_{a.satz}_{a.ziel}.csv")
agg = np.mean if a.mittel == "mean" else np.median


def clu(sub, S):
    n = len(sub)
    if n <= 1:
        return np.ones(n, int)
    return fcluster(linkage(squareform(sub, checks=False), method="complete"),
                    t=S, criterion="distance")


zeilen = []
for arm, nfe in ZELLEN:
    z = next((z for z in SAETZE[a.satz]["zellen"]
              if z["arm"] == arm and z["nfe"] == nfe), None)
    if z is None:
        continue
    tab = lade_zelle(z, leise=True).set_index(["complex", "seed"])
    daten = posencache.hole(a.satz, arm, nfe, leise=True)
    for code, v in daten.items():
        g = tab.loc[[(code, int(s)) for s in v["seed"]]]
        gn = -g["affinity"].to_numpy()          # hoeher ist besser
        heur = g["heur"].to_numpy()
        f5 = g[list(PB_CHECKS)].all(axis=1).to_numpy()
        f24 = np.asarray(v["valid"], bool)
        ziel = np.asarray(v[a.ziel], bool)
        d = v["d"].astype(float)
        n = len(gn)

        r = {"arm": arm, "nfe": nfe, "complex": code, "n": n,
             # Bezugswerte: der Score des Papers, und gnina nach hartem Filter.
             "basis_mixed": bool(ziel[int(np.argmax(heur))]),
             "basis_gnina_f24": bool(ziel[np.where(f24)[0][np.argmax(gn[f24])]])
             if f24.any() else False}
        for fname, fkey in FILTER:
            maske = (np.ones(n, bool) if fkey is None
                     else (f5 if fkey == "f5" else f24))
            idx = np.where(maske)[0]
            for S in SCHW:
                schl = f"{fname}@{S}"
                if len(idx) == 0:
                    r[schl] = False
                    continue
                lab = clu(d[np.ix_(idx, idx)], S)
                gruppen = [idx[lab == cl] for cl in np.unique(lab)]
                gross = [gr for gr in gruppen if len(gr) >= a.min_cluster]
                if not gross:
                    gross = gruppen          # sonst bliebe nichts uebrig
                mittel = [agg(gn[gr]) for gr in gross]
                sieger = gross[int(np.argmax(mittel))]
                beste = sieger[int(np.argmax(gn[sieger]))]
                r[schl] = bool(ziel[beste])
        zeilen.append(r)

t = pd.DataFrame(zeilen)
t.to_csv(AUS, index=False)

KRIT = "RMSD < 2 A" + (" und PB-valide" if a.ziel == "beides" else "")
print(f"########## {a.satz.upper()}, Ziel {KRIT}, "
      f"{a.mittel} je Cluster, Mindestgroesse {a.min_cluster} ##########\n")
spalten = [f"{f}@{S}" for f, _ in FILTER for S in SCHW]
kopf = (f"  {'Zelle':<20}{'K':>5}{'Mixed':>8}{'gnina|24':>10}   "
        + "".join(f"{c:>13}" for c in spalten))
print(kopf)
for arm, nfe in ZELLEN:
    d = t[(t.arm == arm) & (t.nfe == nfe)]
    if d.empty:
        continue
    print(f"  {LANG[arm] + ', ' + str(nfe):<20}{d.n.mean():5.0f}"
          f"{100 * d.basis_mixed.mean():7.1f}%{100 * d.basis_gnina_f24.mean():9.1f}%   "
          + "".join(f"{100 * d[c].mean():12.1f}%" for c in spalten))
print(f"\n  Spalten: Filter@Clusterschwelle. 'Mixed' ist der Score des Papers "
      f"(Argmax ueber alle Posen),\n  'gnina|24' ist gnina nach hartem "
      f"24-Pruefungen-Filter -- beides ohne Clustering.")
print(f"\n{len(t)} Zeilen nach {AUS}")
