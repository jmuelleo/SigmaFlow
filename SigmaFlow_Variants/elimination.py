"""Cluster-Elimination: nur vollstaendig valide Cluster bleiben im Rennen.

DAS PROTOKOLL
    1. Clustern (vollstaendige Verkettung, Schwelle S).
    2. Nur Cluster mit mindestens ZWEI Mitgliedern kommen infrage.
    3. Jeden Cluster verwerfen, der auch nur EINE nicht PB-valide Pose
       enthaelt. Uebrig bleiben Gruppen, die durchgehend valide sind.
    4. Auswahl:
       - genau ein Cluster uebrig  -> darin die beste Pose nach gnina
       - kein Cluster uebrig       -> beste gnina-Pose unter allen validen
                                      Posen (Rueckfall)
       - mehrere uebrig            -> den Cluster waehlen, dessen ZWEI beste
                                      gnina-Werte am hoechsten sind, darin
                                      die beste Pose

DIE IDEE
    Ein Cluster, in dem ALLE Posen valide sind, ist ein Ort, an dem das Modell
    wiederholt und zuverlaessig etwas Sinnvolles erzeugt hat -- nicht nur ein
    einzelner Glueckstreffer. Die Elimination ist strenger als ein Filter auf
    Posenebene: dort ueberlebt eine gute Pose in schlechter Nachbarschaft,
    hier nicht.

    Der Preis ist, dass viele Komplexe gar keinen Cluster behalten. Deshalb
    der Rueckfall, und deshalb wird unten zuerst die Landschaft berichtet.

WARUM DAS TOP-ZWEI-KRITERIUM UND NICHT DER MITTELWERT
    Ein Mittelwert bevorzugt kleine Cluster, weil dort keine schwachen
    Mitglieder mitziehen. Die zwei besten Werte sind groessenneutral: zwei
    gute Posen sind zwei gute Posen, ob der Cluster drei oder dreissig hat.

Aufruf:
    python SigmaFlow_Variants/elimination.py --satz pb308
    python SigmaFlow_Variants/elimination.py --satz astex --schwellen 0.5,1.0
"""
import argparse
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
from scipy.stats import binomtest  # noqa: E402

LANG = {"SigmaDock": "SigmaDock", "Minimal": "SF-Minimal",
        "Separate": "SF-Separate"}
ZELLEN = [("SigmaDock", 25), ("Minimal", 25), ("Separate", 25),
          ("Minimal", 5), ("Separate", 5)]

p = argparse.ArgumentParser()
p.add_argument("--satz", choices=sorted(SAETZE), default="pb308")
p.add_argument("--schwellen", default="0.5,1.0")
p.add_argument("--min-mitglieder", type=int, default=2)
p.add_argument("--ziel", default="beides", choices=["acc", "beides"])
p.add_argument("--out", default=None)
a = p.parse_args()
SCHW = [float(x) for x in a.schwellen.split(",")]
AUS = a.out or os.path.join(_HIER, f"elimination_{a.satz}.csv")

zeilen, cluster_zeilen = [], []
for arm, nfe in ZELLEN:
    z = next((z for z in SAETZE[a.satz]["zellen"]
              if z["arm"] == arm and z["nfe"] == nfe), None)
    if z is None:
        continue
    tab = lade_zelle(z, leise=True).set_index(["complex", "seed"])
    for code, v in posencache.hole(a.satz, arm, nfe, leise=True).items():
        g = tab.loc[[(code, int(s)) for s in v["seed"]]]
        gn = -g["affinity"].to_numpy()
        heur = g["heur"].to_numpy()
        valid = np.asarray(v["valid"], bool)
        ziel = np.asarray(v[a.ziel], bool)
        d = v["d"].astype(float)
        n = len(gn)

        r = {"arm": arm, "nfe": nfe, "complex": code, "n": n,
             "basis_mixed": bool(ziel[int(np.argmax(heur))]),
             "basis_f24": bool(ziel[np.where(valid)[0][np.argmax(gn[valid])]])
             if valid.any() else False}

        for S in SCHW:
            lab = (np.ones(n, int) if n <= 1 else
                   fcluster(linkage(squareform(d, checks=False),
                                    method="complete"),
                            t=S, criterion="distance"))
            gruppen = [np.where(lab == cl)[0] for cl in np.unique(lab)]
            kandidaten = [gr for gr in gruppen
                          if len(gr) >= a.min_mitglieder and valid[gr].all()]

            r[f"n_cl{S}"] = len(kandidaten)
            for gr in kandidaten:
                cluster_zeilen.append({"arm": arm, "nfe": nfe, "S": S,
                                       "groesse": len(gr),
                                       "anteil": float(ziel[gr].mean())})

            if len(kandidaten) == 0:
                r[f"elim{S}"] = r["basis_f24"]
                r[f"weg{S}"] = "Rueckfall"
            elif len(kandidaten) == 1:
                gr = kandidaten[0]
                r[f"elim{S}"] = bool(ziel[gr[int(np.argmax(gn[gr]))]])
                r[f"weg{S}"] = "ein Cluster"
            else:
                punkte = [np.sort(gn[gr])[::-1][:2].mean() for gr in kandidaten]
                gr = kandidaten[int(np.argmax(punkte))]
                r[f"elim{S}"] = bool(ziel[gr[int(np.argmax(gn[gr]))]])
                r[f"weg{S}"] = "mehrere"
        zeilen.append(r)

t = pd.DataFrame(zeilen)
ct = pd.DataFrame(cluster_zeilen)
t.to_csv(AUS, index=False)

KRIT = "RMSD < 2 A" + (" und PB-valide" if a.ziel == "beides" else "")
print(f"########## {a.satz.upper()}, Ziel {KRIT}, Cluster ab "
      f"{a.min_mitglieder} Mitgliedern, nur vollstaendig valide ##########")

for S in SCHW:
    print(f"\n=== Landschaft nach der Elimination, Schwelle {S} A ===")
    print(f"  {'Zelle':<20}{'Cluster/Komplex':>17}{'kein Cluster':>14}"
          f"{'genau einer':>13}{'mehrere':>10}   {'rein 100%':>11}{'rein 0%':>9}"
          f"{'Mittel':>8}")
    for arm, nfe in ZELLEN:
        d = t[(t.arm == arm) & (t.nfe == nfe)]
        c = ct[(ct.arm == arm) & (ct.nfe == nfe) & (ct.S == S)]
        if d.empty:
            continue
        k = d[f"n_cl{S}"]
        print(f"  {LANG[arm] + ', ' + str(nfe):<20}{k.mean():17.2f}"
              f"{100 * (k == 0).mean():13.1f}%{100 * (k == 1).mean():12.1f}%"
              f"{100 * (k >= 2).mean():9.1f}%   "
              + (f"{100 * (c.anteil == 1).mean():10.1f}%"
                 f"{100 * (c.anteil == 0).mean():8.1f}%{c.anteil.mean():8.3f}"
                 if len(c) else f"{'--':>10}{'--':>9}{'--':>8}"))

print(f"\n=== Leistung ===")
sp = ["basis_mixed", "basis_f24"] + [f"elim{S}" for S in SCHW]
print(f"  {'Zelle':<20}" + "".join(f"{c:>14}" for c in sp))
for arm, nfe in ZELLEN:
    d = t[(t.arm == arm) & (t.nfe == nfe)]
    if d.empty:
        continue
    print(f"  {LANG[arm] + ', ' + str(nfe):<20}"
          + "".join(f"{100 * d[c].mean():13.1f}%" for c in sp))

print(f"\n=== Gepaart gegen 'gnina nach 24er-Filter' (McNemar) ===")
for S in SCHW:
    print(f"  Schwelle {S} A:")
    for arm, nfe in ZELLEN:
        d = t[(t.arm == arm) & (t.nfe == nfe)]
        if d.empty:
            continue
        x, y = d.basis_f24.to_numpy(), d[f"elim{S}"].to_numpy()
        na, nb = int((x & ~y).sum()), int((y & ~x).sum())
        pv = binomtest(nb, na + nb, 0.5).pvalue if na + nb else 1.0
        print(f"    {LANG[arm] + ', ' + str(nfe):<20}{100 * y.mean():5.1f} gegen "
              f"{100 * x.mean():5.1f}   {nb}:{na}   p = {pv:.3f}")
    x, y = t.basis_f24.to_numpy(), t[f"elim{S}"].to_numpy()
    na, nb = int((x & ~y).sum()), int((y & ~x).sum())
    pv = binomtest(nb, na + nb, 0.5).pvalue if na + nb else 1.0
    print(f"    {'GEPOOLT':<20}{100 * y.mean():5.1f} gegen {100 * x.mean():5.1f}"
          f"   {nb}:{na}   p = {pv:.4f}")

print(f"\n{len(t)} Zeilen nach {AUS}")
