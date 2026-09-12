"""Spricht die Umgebung fuer die Pose? Eine Batterie weicher Rankingregeln.

DIE HYPOTHESE
    Eine Pose, deren Nachbarschaft ebenfalls valide und gut bewertet ist, ist
    wahrscheinlicher richtig als eine gleich bewertete Pose in schlechter
    Gesellschaft. Diese Zusatzinformation muesste sich in ein Ranking einbauen
    lassen.

    Vier vorangegangene Versuche sind gescheitert (Konsens ueber gnina, Konsens
    ueber den Mixed Score, Groessenfilter, Cluster-Elimination). Alle vier
    hatten gemeinsam, dass sie ZUERST einen Cluster waehlten und die
    Poseninformation dabei wegwarfen. Diese Batterie macht es umgekehrt: die
    Pose bleibt die Einheit, die Umgebung wird nur als GEWICHT zugeschlagen.

DIE SECHS FAMILIEN

  A  Harte Posenvalidität mal weicher Clusterrueckhalt
       s = valid_i * gnina_i * anteil_valide_im_cluster^alpha
     Die Pose muss selbst valide sein -- das ist die bisher beste Regel --
     und wird zusaetzlich danach gewichtet, wie valide ihre Umgebung ist.

  B  Weicher Rueckhalt ohne harten Filter
       s = gnina_i * anteil_valide^alpha
     Falls der harte Filter zu streng ist und die weiche Gewichtung ihn
     ersetzen kann.

  C  Groesse statt Validitaet
       s = valid_i * gnina_i * clustergroesse^beta
     Reiner Rueckhalt durch Wiederholung, ohne Qualitaetsurteil.

  D  Clustervorfilter
     Nur Posen aus Clustern mit einem Validitaetsanteil ueber q kommen in
     Frage, darin dann normal Argmax.

  E  Schrumpfung zum Clustermittel
       s = valid_i * (w * gnina_i + (1-w) * mittel(gnina im Cluster))
     Die klassische Form, Einzelmessungen an ihrer Gruppe zu stabilisieren.

  F  Zahl der validen Nachbarn statt Anteil
       s = valid_i * gnina_i * (1 + n_valide_im_cluster)^gamma
     Anteil und Anzahl sind verschieden: ein Zweiercluster mit zwei validen
     Posen hat Anteil 1, aber wenig Rueckhalt.

WARUM DAS EHRLICH BLEIBEN MUSS
    Es werden viele Regeln auf denselben Daten geprueft. Der beste Wert einer
    solchen Suche ist nach oben verzerrt. Deshalb: Auswahl auf PB308,
    Bestaetigung auf ASTEX, und der gepaarte Test laeuft gegen die bisher
    beste Regel, nicht gegen den schwaechsten Bezug.

Aufruf:
    python SigmaFlow_Variants/umgebung.py --satz pb308
    python SigmaFlow_Variants/umgebung.py --satz astex
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

LANG = {"SigmaDock": "SigmaDock", "Minimal": "SF-Minimal",
        "Separate": "SF-Separate"}
ZELLEN = [("SigmaDock", 25), ("Minimal", 25), ("Separate", 25),
          ("Minimal", 5), ("Separate", 5)]

p = argparse.ArgumentParser()
p.add_argument("--satz", choices=sorted(SAETZE), default="pb308")
p.add_argument("--schwellen", default="0.5,1.0,2.0")
p.add_argument("--ziel", default="beides", choices=["acc", "beides"])
p.add_argument("--out", default=None)
a = p.parse_args()
SCHW = [float(x) for x in a.schwellen.split(",")]
AUS = a.out or os.path.join(_HIER, f"umgebung_{a.satz}.csv")


def argmax_sicher(s, rueckfall):
    """Argmax, aber wenn alles null ist, den Rueckfall nehmen.

    Bei multiplikativen Regeln wird jede nicht valide Pose zu null. Enthaelt
    ein Komplex ueberhaupt keine valide Pose, ist der ganze Vektor null und
    `argmax` gaebe stumm den Index 0 zurueck -- also die Pose mit der
    niedrigsten Seednummer. Das waere kein Ranking, sondern Zufall, und es
    fiele in keiner Kennzahl auf.
    """
    return int(np.argmax(s)) if np.any(s > 0) else rueckfall


def regeln(heur, gn, valid, d, S):
    """Alle Regeln fuer EINEN Komplex bei EINER Clusterschwelle."""
    n = len(gn)
    lab = (np.ones(n, int) if n <= 1 else
           fcluster(linkage(squareform(d, checks=False), method="complete"),
                    t=S, criterion="distance"))
    # Clustereigenschaften, auf die Posen zurueckgespiegelt
    cval = np.empty(n)      # Anteil valider Posen im eigenen Cluster
    cnval = np.empty(n)     # Anzahl valider Posen im eigenen Cluster
    csize = np.empty(n)
    cmean = np.empty(n)
    for cl in np.unique(lab):
        m = lab == cl
        cval[m] = valid[m].mean()
        cnval[m] = valid[m].sum()
        csize[m] = m.sum()
        cmean[m] = gn[m].mean()

    v = valid.astype(float)
    rf = (int(np.where(valid)[0][np.argmax(gn[valid])]) if valid.any()
          else int(np.argmax(gn)))
    aus = {}
    for al in (0.25, 0.5, 1.0, 2.0, 4.0):
        aus[f"A a{al}"] = argmax_sicher(v * gn * cval ** al, rf)
    for al in (0.5, 1.0, 2.0, 4.0):
        aus[f"B a{al}"] = argmax_sicher(gn * cval ** al, rf)
    for be in (0.1, 0.25, 0.5):
        aus[f"C b{be}"] = argmax_sicher(v * gn * csize ** be, rf)
    for q in (0.25, 0.5, 0.75, 1.0):
        m = valid & (cval >= q)
        aus[f"D q{q}"] = (int(np.where(m)[0][np.argmax(gn[m])]) if m.any() else rf)
    for w in (0.5, 0.75, 0.9):
        aus[f"E w{w}"] = argmax_sicher(v * (w * gn + (1 - w) * cmean), rf)
    for ga in (0.1, 0.25, 0.5):
        aus[f"F g{ga}"] = argmax_sicher(v * gn * (1 + cnval) ** ga, rf)
    # Kombination: Validitaetsanteil UND Groesse
    for al, be in ((0.5, 0.1), (1.0, 0.1), (0.5, 0.25)):
        aus[f"AC {al}/{be}"] = argmax_sicher(
            v * gn * cval ** al * csize ** be, rf)
    return aus


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
        gn = -g["affinity"].to_numpy()
        valid = np.asarray(v["valid"], bool)
        ziel = np.asarray(v[a.ziel], bool)
        d = v["d"].astype(float)
        rf = (int(np.where(valid)[0][np.argmax(gn[valid])]) if valid.any()
              else int(np.argmax(gn)))
        r = {"arm": arm, "nfe": nfe, "complex": code,
             "mixed": bool(ziel[int(np.argmax(heur))]),
             "f24+gnina": bool(ziel[rf])}
        for S in SCHW:
            for name, i in regeln(heur, gn, valid, d, S).items():
                r[f"{name}@{S}"] = bool(ziel[i])
        zeilen.append(r)

t = pd.DataFrame(zeilen)
t.to_csv(AUS, index=False)
STRAT = [c for c in t.columns if c not in ("arm", "nfe", "complex")]

print(f"########## {a.satz.upper()}, {len(t) // len(ZELLEN)} Komplexe, "
      f"Ziel RMSD<2 und PB-valide ##########\n")
werte = {}
for s_ in STRAT:
    zeile = [100 * t[(t.arm == arm) & (t.nfe == nfe)][s_].mean()
             for arm, nfe in ZELLEN]
    werte[s_] = (np.mean(zeile), zeile)

basis = werte["f24+gnina"][0]
print(f"  {'Regel':<16}" + "".join(
    f"{LANG[arm][:9] + '/' + str(nfe):>13}" for arm, nfe in ZELLEN)
    + f"{'Mittel':>9}{'vs Basis':>10}")
for s_ in ["mixed", "f24+gnina"]:
    m, zl = werte[s_]
    print(f"  {s_:<16}" + "".join(f"{x:12.1f}%" for x in zl)
          + f"{m:8.1f}%{m - basis:+9.1f}")
print("  " + "-" * 88)
rang = sorted((s for s in STRAT if s not in ("mixed", "f24+gnina")),
              key=lambda s: -werte[s][0])
for s_ in rang[:14]:
    m, zl = werte[s_]
    marke = " <--" if m > basis else ""
    print(f"  {s_:<16}" + "".join(f"{x:12.1f}%" for x in zl)
          + f"{m:8.1f}%{m - basis:+9.1f}{marke}")
print(f"\n  ... {len(rang) - 14} weitere Regeln, alle darunter.")
print(f"\n{len(t)} Zeilen, {len(STRAT)} Regeln nach {AUS}")
