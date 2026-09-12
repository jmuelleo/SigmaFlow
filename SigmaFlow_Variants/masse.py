"""Welches Mass sagt am besten voraus, ob die gerankte Pose stimmt?

DIE FRAGE HINTER DER FRAGE
    "Konzentration im groessten Cluster" ist EIN Mass fuer die Sicherheit
    eines generativen Modells, und es ist gut (AUC um 0,80). Aber es steckt
    voller Entscheidungen: Mittelwert-Verkettung, Schnitt bei 2 Angstroem,
    RMSD ohne Symmetriekorrektur, harte Partition. Jede davon koennte die
    Antwort verzerren, und keine ist von der Sache erzwungen.

    Statt darueber zu argumentieren, werden hier ein Dutzend Alternativen
    gegen dasselbe Ziel gemessen. Die AUC entscheidet.

DIE KANDIDATEN, IN VIER GRUPPEN

  A. DIESELBE IDEE, ANDERE VERKETTUNG
     Mittelwert-, Vollstaendige- und Einzelverkettung. Einzelverkettung
     verkettet ueber duenne Bruecken ("chaining") und sollte Moden
     verschmelzen; vollstaendige Verkettung erzwingt kompakte Cluster. Wenn
     die Rangfolge der Komplexe davon abhaengt, ist der Befund ein Artefakt
     der Methode.

  B. DICHTEBASIERT STATT PARTITIONIEREND
     DBSCAN und HDBSCAN muessen NICHT jede Pose einem Cluster zuordnen --
     Ausreisser bleiben Rauschen. Das passt besser zum Begriff "Modus": ein
     Modus ist eine Dichtespitze, keine Zelle einer Zerlegung. Insbesondere
     blaehen einzelne Ausreisser die Modenzahl nicht mehr auf.

  C. GANZ OHNE CLUSTERING
     p_agree: der Anteil der PAARE mit RMSD unter der Schwelle. Das ist die
     Wahrscheinlichkeit, dass zwei unabhaengige Ziehungen uebereinstimmen --
     dieselbe Groesse, die ein Simpson-Index misst, aber ohne jede Partition
     und ohne Verkettungsregel. Ihr Erwartungswert haengt ausserdem NICHT von
     K ab, waehrend der Anteil im groessten Cluster mit K faellt. Damit ist
     sie das theoretisch sauberere Konzentrationsmass.

     Dazu der mittlere paarweise RMSD und der mediane Nachbarabstand, beide
     mit umgekehrtem Vorzeichen, damit "gross" ueberall "sicher" heisst.

  D. AUF DIE ANTWORT BEZOGEN STATT AUF DIE VERTEILUNG
     Die bisherigen Masse beschreiben die ganze Verteilung. Interessant ist
     aber die GEWAEHLTE Pose:

       lokal_dichte  Anteil der Ziehungen innerhalb der Schwelle um die
                     gerankte Pose. "Wie viel Masse stuetzt genau diese
                     Antwort?" -- nicht "wie gebuendelt ist das Modell?".

       stabil_halb   Zwei disjunkte Haelften der Ziehungen, jede rankt fuer
                     sich; stimmen die beiden Antworten ueberein? Ueber 1000
                     Teilungen gemittelt. Das misst die Reproduzierbarkeit
                     der Pipeline und braucht weder Cluster noch Schwelle
                     fuer die Verteilung, nur fuer den Vergleich zweier Posen.

     Und als Kontrolle das Score-Mass, das sich schon als wertlos erwiesen
     hat: die z-Marge des Rankers. Es soll in der Tabelle stehen, damit der
     Abstand sichtbar ist.

WAS "AUC" HIER BEDEUTET
    Die Wahrscheinlichkeit, dass ein Komplex mit richtiger Top-1-Pose einen
    hoeheren Masswert hat als einer mit falscher. 0,5 ist wertlos, 1,0 waere
    perfekt. Rangbasiert, also unabhaengig von Skala und Einheit -- nur
    deshalb sind Prozentanteile, Angstroem und Standardabweichungen in einer
    Tabelle vergleichbar.

Aufruf:
    python SigmaFlow_Variants/masse.py
    python SigmaFlow_Variants/masse.py --sym          (symmetriekorrigiert)
    python SigmaFlow_Variants/masse.py --schwelle 1.5
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

_HIER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HIER)
import posencache  # noqa: E402
from zellen import SAETZE  # noqa: E402

from scipy.cluster.hierarchy import fcluster, linkage  # noqa: E402
from scipy.spatial.distance import squareform  # noqa: E402
from sklearn.cluster import DBSCAN, HDBSCAN  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--satz", choices=sorted(SAETZE), default="pb308")
p.add_argument("--nfe", type=int, default=25)
p.add_argument("--schwelle", type=float, default=2.0)
p.add_argument("--ziel", default="acc", choices=["acc", "beides"])
p.add_argument("--teilungen", type=int, default=1000)
p.add_argument("--seed", type=int, default=20260908)
p.add_argument("--sym", action="store_true",
               help="symmetriekorrigierter RMSD (langsam, nur kleine K)")
p.add_argument("--out", default=None)
a = p.parse_args()
AUS = a.out or os.path.join(
    _HIER, f"masse_{a.satz}_nfe{a.nfe}{'_sym' if a.sym else ''}.csv")

ARME = ["SigmaDock", "Minimal", "Separate"]
S = a.schwelle


def groesster_anteil(lab, n):
    """Anteil im groessten Cluster; Rauschlabel -1 zaehlt nicht als Cluster."""
    echte = lab[lab >= 0]
    if not len(echte):
        return 1.0 / n
    return float(np.bincount(echte).max() / n)


def verkettung(d, methode):
    Z = linkage(squareform(d, checks=False), method=methode)
    return fcluster(Z, t=S, criterion="distance") - 1


def masse_eines(d, heur, rng):
    """Alle Kandidaten fuer EINEN Komplex."""
    n = len(heur)
    m = {}
    for name, meth in (("avg", "average"), ("complete", "complete"),
                       ("single", "single")):
        lab = verkettung(d, meth)
        m[f"konz_{name}"] = groesster_anteil(lab, n)
        if name == "avg":
            g = np.bincount(lab)
            pk = g[g > 0] / n
            # exp(Entropie) = "effektive Modenzahl". Anders als die rohe
            # Modenzahl zaehlt sie einen Cluster mit einer einzigen Pose
            # kaum mit. Negativ, damit gross = sicher.
            m["neg_eff_moden"] = -float(np.exp(-(pk * np.log(pk)).sum()))

    # Dichtebasiert: metric="precomputed" heisst, die Matrix IST der Abstand.
    lab = DBSCAN(eps=S, min_samples=3, metric="precomputed").fit_predict(d)
    m["konz_dbscan"] = groesster_anteil(lab, n)
    lab = HDBSCAN(min_cluster_size=3, metric="precomputed",
                          copy=True).fit_predict(d.astype(float))
    m["konz_hdbscan"] = groesster_anteil(lab, n)

    # Ohne Clustering. iu: nur das obere Dreieck, sonst zaehlt jede Paarung
    # doppelt und die Diagonale (RMSD 0) faelscht den Anteil nach oben.
    iu = np.triu_indices(n, 1)
    paare = d[iu]
    m["p_agree"] = float((paare < S).mean())
    m["neg_mittel_rmsd"] = -float(paare.mean())
    dn = d + np.eye(n) * 1e9
    m["neg_nn_median"] = -float(np.median(dn.min(axis=1)))

    # Auf die gewaehlte Antwort bezogen.
    sieger = int(np.argmax(heur))
    m["lokal_dichte"] = float((d[sieger] < S).mean())

    # Split-half: zwei disjunkte Haelften, jede rankt fuer sich.
    h = n // 2
    treffer = 0
    for _ in range(a.teilungen):
        perm = rng.permutation(n)
        i1, i2 = perm[:h], perm[h:2 * h]
        s1 = i1[np.argmax(heur[i1])]
        s2 = i2[np.argmax(heur[i2])]
        treffer += d[s1, s2] < S
    m["stabil_halb"] = treffer / a.teilungen

    # Kontrolle: Score-Marge.
    sd = heur.std(ddof=1)
    m["z_marge"] = float((heur.max() - heur.mean()) / sd) if sd > 0 else np.nan
    return m


def auc(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, bool)
    ok = ~np.isnan(x)
    x, y = x[ok], y[ok]
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(x).rank().to_numpy()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


rng = np.random.default_rng(a.seed)
zeilen = []
for arm in ARME:
    print(f"{arm}, {a.nfe} Schritte ...", flush=True)
    daten = posencache.hole(a.satz, arm, a.nfe, sym=a.sym)
    for code, v in daten.items():
        if a.ziel not in v:
            continue
        d = v["d"].astype(float)
        heur, ziel = v["heur"], v[a.ziel]
        zeilen.append({"arm": arm, "complex": code, "n": len(heur),
                       "trifft": bool(ziel[int(np.argmax(heur))]),
                       **masse_eines(d, heur, rng)})

t = pd.DataFrame(zeilen)
t.to_csv(AUS, index=False)

MASSE = [c for c in t.columns
         if c not in ("arm", "complex", "n", "trifft")]
NAMEN = {
    "konz_avg": "groesster Cluster (average)",
    "konz_complete": "groesster Cluster (complete)",
    "konz_single": "groesster Cluster (single)",
    "konz_dbscan": "groesster Cluster (DBSCAN)",
    "konz_hdbscan": "groesster Cluster (HDBSCAN)",
    "neg_eff_moden": "-exp(Entropie) = eff. Modenzahl",
    "p_agree": "P(zwei Ziehungen stimmen ueberein)",
    "neg_mittel_rmsd": "-mittlerer paarweiser RMSD",
    "neg_nn_median": "-medianer Nachbarabstand",
    "lokal_dichte": "Masse um die gerankte Pose",
    "stabil_halb": "Split-half-Uebereinstimmung",
    "z_marge": "z-Marge des Rankers (Kontrolle)",
}

print(f"\n########## {a.satz}, {a.nfe} Schritte, Ziel {a.ziel}, "
      f"Schwelle {S} A{', symmetriekorrigiert' if a.sym else ''} ##########")
print(f"{len(t) // len(ARME)} Komplexe je Arm, "
      f"{int(t['n'].median())} Posen je Komplex\n")
print(f"  {'Mass':<36}{'SigmaDock':>11}{'Minimal':>10}{'Separate':>10}"
      f"{'Mittel':>9}")
werte = {}
for m in MASSE:
    r = [auc(t[t.arm == arm][m], t[t.arm == arm]["trifft"]) for arm in ARME]
    werte[m] = float(np.nanmean(r))
    print(f"  {NAMEN.get(m, m):<36}" + "".join(f"{v:10.3f} " for v in r)
          + f"{werte[m]:8.3f}")

print("\n  Nach mittlerer AUC:")
for i, (m, v) in enumerate(sorted(werte.items(), key=lambda kv: -kv[1]), 1):
    print(f"    {i:2d}. {NAMEN.get(m, m):<38}{v:.3f}")
print(f"\n{len(t)} Zeilen nach {AUS}")
