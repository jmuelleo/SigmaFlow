"""Konsens-Ranking II: Cluster nach dem Mixed Score der besten Mitglieder.

DAS PROTOKOLL
    1. K Posen ziehen, OHNE Filter clustern (vollstaendige Verkettung).
    2. Je Cluster die Mitglieder nach Mixed Score sortieren.
    3. Vom oberen Anteil q den Mittelwert bilden.
    4. Den Cluster mit dem hoechsten Mittel waehlen.
    5. Darin die Pose mit dem hoechsten Mixed Score nehmen.

WARUM DAS ANDERS IST ALS KONSENS I
    Dort wurde nach mittlerem gnina-Score gewaehlt, und der sagt nichts ueber
    chemische Plausibilitaet -- ohne vorgeschalteten Filter gewannen kompakte
    Gruppen unmoeglicher Posen (35 bis 48 Prozent statt 64 bis 74). Der Mixed
    Score traegt die Validitaet als Faktor p^4 bereits in sich, ein separater
    Filter ist also nicht noetig.

WOZU DAS TRIMMEN
    q = 1,0 ist der volle Mittelwert: robust, aber ein Cluster mit wenigen
    guten und vielen schwachen Mitgliedern faellt durch.
    q = 0,25 ist fast das Maximum: empfindlich gegen Ausreisser.
    Dazwischen liegt der Kompromiss -- "die Gruppe hat mehrere gute
    Vertreter", ohne dass ein einzelner Zufallstreffer reicht.

    Bei kleinen Clustern rundet `ceil` auf, damit q = 0,25 in einem
    Zweier-Cluster nicht auf null Mitglieder faellt.

Aufruf:
    python SigmaFlow_Variants/konsens2.py --satz astex
    python SigmaFlow_Variants/konsens2.py --satz pb308 --min-cluster 3
"""
import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

_HIER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HIER)
import posencache  # noqa: E402
from zellen import SAETZE, lade_zelle  # noqa: E402

from scipy.cluster.hierarchy import fcluster, linkage  # noqa: E402
from scipy.spatial.distance import squareform  # noqa: E402

LANG = {"SigmaDock": "SigmaDock", "Minimal": "SF-Minimal",
        "Separate": "SF-Separate"}
ZELLEN = [("SigmaDock", 25), ("Minimal", 25), ("Separate", 25),
          ("Minimal", 5), ("Separate", 5)]

p = argparse.ArgumentParser()
p.add_argument("--satz", choices=sorted(SAETZE), default="pb308")
p.add_argument("--schwellen", default="1.0,2.0")
p.add_argument("--quantile", default="0.25,0.5,1.0")
p.add_argument("--min-cluster", type=int, default=3)
p.add_argument("--ziel", default="beides", choices=["acc", "beides"])
p.add_argument("--out", default=None)
a = p.parse_args()
SCHW = [float(x) for x in a.schwellen.split(",")]
QU = [float(x) for x in a.quantile.split(",")]
AUS = a.out or os.path.join(_HIER, f"konsens2_{a.satz}_{a.ziel}.csv")


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
    for code, v in posencache.hole(a.satz, arm, nfe, leise=True).items():
        g = tab.loc[[(code, int(s)) for s in v["seed"]]]
        heur = g["heur"].to_numpy()
        ziel = np.asarray(v[a.ziel], bool)
        d = v["d"].astype(float)
        r = {"arm": arm, "nfe": nfe, "complex": code, "n": len(heur),
             "basis": bool(ziel[int(np.argmax(heur))])}
        for S in SCHW:
            lab = clu(d, S)
            gruppen = [np.where(lab == cl)[0] for cl in np.unique(lab)]
            gross = [x for x in gruppen if len(x) >= a.min_cluster] or gruppen
            for q in QU:
                punkte = []
                for gr in gross:
                    s = np.sort(heur[gr])[::-1]
                    k = max(1, math.ceil(q * len(s)))
                    punkte.append(s[:k].mean())
                sieger = gross[int(np.argmax(punkte))]
                beste = sieger[int(np.argmax(heur[sieger]))]
                r[f"q{q}@{S}"] = bool(ziel[beste])
        zeilen.append(r)

t = pd.DataFrame(zeilen)
t.to_csv(AUS, index=False)

KRIT = "RMSD < 2 A" + (" und PB-valide" if a.ziel == "beides" else "")
print(f"########## {a.satz.upper()}, Ziel {KRIT}, ohne Filter, "
      f"Mixed Score, Mindestcluster {a.min_cluster} ##########\n")
sp = [f"q{q}@{S}" for S in SCHW for q in QU]
print(f"  {'Zelle':<20}{'K':>5}{'Basis':>9}   "
      + "".join(f"{c:>11}" for c in sp))
for arm, nfe in ZELLEN:
    d = t[(t.arm == arm) & (t.nfe == nfe)]
    if d.empty:
        continue
    beste = max(100 * d[c].mean() for c in sp)
    z = f"  {LANG[arm] + ', ' + str(nfe):<20}{d.n.mean():5.0f}{100 * d.basis.mean():8.1f}%   "
    for c in sp:
        w = 100 * d[c].mean()
        z += f"{w:10.1f}%" if abs(w - beste) > 1e-9 else f"{w:9.1f}%*"
    print(z)
print("\n  Basis = Argmax des Mixed Scores ueber alle Posen, ohne Clustering.")
print("  q = Anteil der besten Mitglieder, ueber den je Cluster gemittelt wird.")
print("  * markiert die beste Konsensvariante der Zeile.")
print(f"\n{len(t)} Zeilen nach {AUS}")
