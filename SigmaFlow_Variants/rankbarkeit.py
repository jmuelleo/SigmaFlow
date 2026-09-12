"""Warum gewinnt SigmaFlow nach Ranking, obwohl es je Zug schlechter ist?

DIE BEOBACHTUNG, DIE ERKLAERT WERDEN SOLL   (Ziel RMSD < 2 A, NFE 25, K = 40)

    Arm         je Zug   Orakel@40   nach Ranking
    SigmaDock    50,53      93,16        69,71
    Minimal      44,81      93,49        74,59

    Die Orakel sind praktisch gleich: BEIDE Verteilungen enthalten die
    richtige Pose gleich oft. SigmaDock trifft je Zug haeufiger. Trotzdem
    liefert Minimal nach Auswahl das bessere Ergebnis.

    Der Unterschied kann also weder in der Abdeckung liegen noch in der
    Trefferquote, sondern nur darin, wie gut sich die richtige Pose INNERHALB
    der Verteilung erkennen laesst. Dieses Skript misst genau das.

DREI MASSE, ALLE INNERHALB EINES KOMPLEXES

    bedingte Trefferquote   Anteil der Komplexe, bei denen der Ranker die
                            richtige Pose findet -- GEGEBEN, dass unter den K
                            Ziehungen ueberhaupt eine richtige ist. Das ist
                            "nach Ranking / Orakel" und trennt die Faehigkeit
                            des Rankers von der Qualitaet des Samplers.

    AUC                     Wie gut trennt der Rankerwert richtige von
                            falschen Posen? Gerechnet je Komplex ueber dessen
                            K Ziehungen, dann gemittelt. 0,5 = zufaellig,
                            1,0 = perfekt trennbar. Nur Komplexe mit BEIDEN
                            Klassen gehen ein, sonst ist AUC undefiniert.

    Rang der besten Pose    Auf welchem Platz der Rankerliste steht die Pose
                            mit dem KLEINSTEN RMSD? Median ueber die Komplexe,
                            normiert auf K, damit K = 40 und K = 200
                            vergleichbar sind.

WARUM AUC UND NICHT KORRELATION
    Der Ranker muss keine RMSD vorhersagen, er muss nur ORDNEN. AUC misst
    genau das und ist unempfindlich gegen monotone Verzerrungen des Scores.

Aufruf:
    python SigmaFlow_Variants/rankbarkeit.py
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

_HIER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HIER)
from zellen import SAETZE, alle_zellen  # noqa: E402


def auc(score: np.ndarray, positiv: np.ndarray) -> float:
    """Mann-Whitney-U als AUC. Bindungen bekommen den mittleren Rang."""
    n1, n0 = int(positiv.sum()), int((~positiv).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(score).rank().to_numpy()
    return (r[positiv].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


REGELN = {"Mixed Score": "heur", "nur gnina": "negaff"}

p = argparse.ArgumentParser()
p.add_argument("--satz", choices=sorted(SAETZE), default="pb308")
p.add_argument("--out", default=os.path.join(_HIER, "rankbarkeit.csv"))
a = p.parse_args()

print("Zellen:")
daten = alle_zellen(satz=a.satz)

zeilen = []
for (arm, nfe), m in daten.items():
    k = m["seed"].nunique()
    m = m.copy()
    m["negaff"] = -m["affinity"]        # hoeher = besser, wie `heur`
    # Der harte Filter ist keine Score-Ordnung, sondern eine Vorauswahl.
    # Fuer AUC brauchen wir einen Score, deshalb hier nur die zwei Ordnungen.
    for ziel in ("acc", "beides"):
        for rname, sp in REGELN.items():
            orakel, gewaehlt, aucs, raenge = [], [], [], []
            for c, g in m.groupby("complex"):
                hat = bool(g[ziel].any())
                orakel.append(hat)
                i = g[sp].idxmax()
                gewaehlt.append(bool(g.loc[i, ziel]))
                v = auc(g[sp].to_numpy(float), g[ziel].to_numpy(bool))
                if not np.isnan(v):
                    aucs.append(v)
                if hat and sp == "heur" and ziel == "acc":
                    # Platz der Pose mit kleinstem RMSD in der Rankerliste.
                    # Die RMSD-Spalte ist bool (<2 A), also naehern wir mit
                    # der BESTEN richtigen Pose: dem hoechsten Rankerwert
                    # unter den richtigen -- das ist der Platz, an dem der
                    # Ranker die erste richtige Pose findet.
                    ordnung = g.sort_values(sp, ascending=False).reset_index(drop=True)
                    pos = int(ordnung.index[ordnung[ziel]][0]) + 1
                    raenge.append(pos / len(g))
            o = float(np.mean(orakel))
            w = float(np.mean(gewaehlt))
            je_zug = float(m[ziel].mean())
            # WAS EIN ZUFALLSRANKER ERREICHEN WUERDE
            #   Zieht man blind eine der K Posen, trifft man mit der
            #   Trefferquote je Zug. Bedingt auf "es gibt ueberhaupt eine
            #   richtige" sind das je_zug / orakel. Der HEBEL ist, um welchen
            #   Faktor der Ranker darueber hinauskommt.
            #
            #   Ohne diese Normierung vergleicht man Aepfel mit Birnen: ein
            #   Arm mit hoeherer Trefferquote je Zug hat es leichter, per
            #   Zufall eine richtige Pose zu erwischen, und sieht dadurch
            #   besser aus, ohne dass sein Ranker etwas leistet.
            zufall = je_zug / o if o > 0 else np.nan
            zeilen.append({
                "arm": arm, "nfe": nfe, "K": k, "ziel": ziel, "regel": rname,
                "je_zug": round(100 * je_zug, 2),
                "orakel": round(100 * o, 2),
                "nach_ranking": round(100 * w, 2),
                "bedingt": round(100 * w / o, 2) if o > 0 else np.nan,
                "zufall_bedingt": round(100 * zufall, 2),
                "hebel": round((w / o) / zufall, 3) if zufall and zufall > 0 else np.nan,
                "auc": round(float(np.mean(aucs)), 4) if aucs else np.nan,
                "n_auc": len(aucs),
                "rang_erste_richtige": (round(float(np.median(raenge)), 4)
                                        if raenge else np.nan),
            })

t = pd.DataFrame(zeilen)
t.to_csv(a.out, index=False)
print(f"\n{len(t)} Zeilen nach {a.out}")

for ziel, titel in (("acc", "RMSD < 2 A"), ("beides", "RMSD<2 UND PB-valide")):
    print(f"\n=== {titel} ===")
    d = t[t["ziel"] == ziel]
    print(f"  {'Zelle':<16}{'Regel':<13}{'jeZug':>7}{'Orakel':>8}{'Ranking':>9}"
          f"{'bedingt':>9}{'Zufall':>8}{'HEBEL':>7}{'AUC':>7}")
    for _, r in d.sort_values(["regel", "arm", "nfe"]).iterrows():
        print(f"  {r['arm'] + ' ' + str(r['nfe']):<16}{r['regel']:<13}"
              f"{r['je_zug']:7.2f}{r['orakel']:8.2f}{r['nach_ranking']:9.2f}"
              f"{r['bedingt']:9.2f}{r['zufall_bedingt']:8.2f}"
              f"{r['hebel']:7.3f}{r['auc']:7.3f}")

print("""
LESEHILFE
  Orakel   enthaelt die Verteilung ueberhaupt eine richtige Pose?
  bedingt  findet der Ranker sie, WENN sie da ist? (Ranking / Orakel)
  AUC      trennt der Rankerwert richtig von falsch, innerhalb des Komplexes?
  Zufall   was ein BLINDER Ranker bedingt erreichen wuerde (jeZug/Orakel)
  HEBEL    bedingt / Zufall -- um welchen Faktor der Ranker den Zufall
           schlaegt. DAS ist die Zahl, die Arme mit verschiedener
           Trefferquote je Zug vergleichbar macht.""")
