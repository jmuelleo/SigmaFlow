"""Wer liegt richtig, wer falsch -- und woran liegt es strukturell?

DREI TEILE

  A  Wie sehen die Verteilungen aus? Modenzahl und Anteil im groessten Modus,
     je Arm.

  B  Wer trifft, wer nicht? Die vollstaendige Kreuztabelle ueber alle drei
     Arme -- acht Muster, von "alle drei richtig" bis "alle drei falsch".

  C  Fuer die Faelle, in denen die Arme sich UNEINIG sind: woran liegt es?

DIE ZERLEGUNG IN TEIL C
    Je Komplex und Arm werden vier Dinge festgehalten:

      hat_korrekte   Enthaelt die Verteilung ueberhaupt eine korrekte Pose?
                     Nein heisst: Abdeckungsproblem, der Ranker hatte keine
                     Chance.

      gross_richtig  Ist der GROESSTE Modus der richtige? Definiert als:
                     mehr als die Haelfte seiner Posen erfuellt das Kriterium.
                     Das ist die Frage, ob das Modell seine Dichte auf die
                     richtige Antwort legt.

      wahl_gross     Hat der Ranker aus dem groessten Modus gegriffen?

      trifft         War die gewaehlte Pose korrekt?

    Daraus ergeben sich fuer einen Arm, der FALSCH liegt, vier Ursachen:

      keine Abdeckung      hat_korrekte = nein. Nichts zu holen gewesen.
      Dichte falsch,       gross_richtig = nein UND wahl_gross = ja.
      Wahl gefolgt         Das Modell ist seiner eigenen Dichte in die
                           falsche Antwort gefolgt.
      Dichte richtig,      gross_richtig = ja UND wahl_gross = nein.
      Wahl daneben         Die Antwort lag im Hauptmodus, der Ranker hat
                           danebengegriffen.
      beides daneben       gross_richtig = nein UND wahl_gross = nein.

    Und fuer einen Arm, der RICHTIG liegt, zwei Wege:

      aus dem Hauptmodus   wahl_gross = ja. Die Dichte war richtig und der
                           Ranker ist ihr gefolgt.
      aus einem Nebenmodus wahl_gross = nein. Der Ranker hat die Antwort
                           gegen die Dichte des Modells gefunden.

    Genau diese letzte Zeile ist die Hypothese, um die es geht: gewinnt der
    Flow-Arm, WEIL er aus einem Nebenmodus zieht?

CLUSTERN WIE UEBERALL
    Paarweiser RMSD ohne Ueberlagerung, Mittelwert-Verkettung, Schnitt bei
    2 Angstroem, K = alle vorhandenen Ziehungen der Zelle.

Aufruf:
    python SigmaFlow_Variants/zerlegung.py --satz astex
    python SigmaFlow_Variants/zerlegung.py --satz pb308 --ziel acc
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

ARME = ["SigmaDock", "Minimal", "Separate"]
KURZ = {"SigmaDock": "SD", "Minimal": "MIN", "Separate": "SEP"}

p = argparse.ArgumentParser()
p.add_argument("--satz", choices=sorted(SAETZE), default="astex")
p.add_argument("--nfe", type=int, default=25)
p.add_argument("--ziel", default="beides", choices=["acc", "beides"])
p.add_argument("--schwelle", type=float, default=2.0)
p.add_argument("--out", default=None)
a = p.parse_args()
AUS = a.out or os.path.join(_HIER, f"zerlegung_{a.satz}_nfe{a.nfe}_{a.ziel}.csv")

zeilen = []
for arm in ARME:
    for code, v in posencache.hole(a.satz, arm, a.nfe, leise=True).items():
        d = v["d"].astype(float)
        n = len(d)
        ok = np.asarray(v[a.ziel], bool)
        lab = fcluster(linkage(squareform(d, checks=False), method="average"),
                       t=a.schwelle, criterion="distance")
        g = np.bincount(lab)[1:]
        gross = int(np.argmax(g)) + 1
        in_gross = lab == gross
        s = int(np.argmax(v["heur"]))
        zeilen.append({
            "complex": code, "arm": arm, "n": n,
            "moden": int(len(g)),
            "gross_anteil": float(g.max() / n),
            "hat_korrekte": bool(ok.any()),
            "n_korrekt": int(ok.sum()),
            # Mehr als die Haelfte des groessten Modus korrekt -> "die Dichte
            # liegt richtig". Der Median statt eines beliebigen Anteils, damit
            # ein Modus nicht schon durch eine einzige korrekte Pose als
            # richtig gilt.
            "gross_richtig": bool(ok[in_gross].mean() > 0.5),
            "wahl_gross": bool(lab[s] == gross),
            "trifft": bool(ok[s]),
        })
t = pd.DataFrame(zeilen)
t.to_csv(AUS, index=False)

W = {arm: t[t.arm == arm].set_index("complex") for arm in ARME}
idx = W["SigmaDock"].index
for arm in ARME[1:]:
    idx = idx.intersection(W[arm].index)
N = len(idx)
KRIT = "RMSD < 2 A" + (" und PB-valide" if a.ziel == "beides" else "")

print(f"########## {a.satz.upper()}, {a.nfe} Schritte, {N} Komplexe, "
      f"Kriterium {KRIT} ##########\n")

# --- A -------------------------------------------------------------------
print("=== A. Wie sehen die Verteilungen aus? ===")
print(f"  {'Arm':<11}{'Posen':>7}{'Moden':>8}{'Anteil im groessten':>21}"
      f"{'groesster ist richtig':>23}")
for arm in ARME:
    d = W[arm].loc[idx]
    print(f"  {arm:<11}{d.n.mean():7.0f}{d.moden.mean():8.1f}"
          f"{100 * d.gross_anteil.mean():20.1f}%{100 * d.gross_richtig.mean():22.1f}%")

# --- B -------------------------------------------------------------------
muster = pd.DataFrame({arm: W[arm].trifft.loc[idx] for arm in ARME})
print(f"\n=== B. Wer trifft? Alle acht Muster ===")
print(f"  {'SD':>4}{'MIN':>5}{'SEP':>5}{'n':>6}{'Anteil':>9}   Lesart")
lesart = {
    (1, 1, 1): "alle drei richtig",
    (0, 0, 0): "alle drei falsch",
    (0, 1, 1): "nur die Flow-Arme richtig",
    (1, 0, 0): "nur SigmaDock richtig",
    (0, 1, 0): "nur Minimal richtig",
    (0, 0, 1): "nur Separate richtig",
    (1, 1, 0): "SigmaDock und Minimal richtig",
    (1, 0, 1): "SigmaDock und Separate richtig",
}
zaehl = muster.groupby(ARME).size()
for schl in sorted(lesart, key=lambda k: -zaehl.get(k, 0)):
    # groupby liefert BOOLEAN-Schluessel; ein Nachschlagen mit (1,1,1)
    # trifft nichts und liefert still ueberall Null.
    n_ = int(zaehl.get(tuple(bool(x) for x in schl), 0))
    print(f"  {schl[0]:>4}{schl[1]:>5}{schl[2]:>5}{n_:6d}{100 * n_ / N:8.1f}%   "
          f"{lesart[schl]}")

# --- C -------------------------------------------------------------------
def ursache(r):
    """Warum liegt dieser Arm bei diesem Komplex richtig oder falsch?"""
    if r.trifft:
        return "richtig: aus dem Hauptmodus" if r.wahl_gross \
            else "richtig: aus einem NEBENMODUS"
    if not r.hat_korrekte:
        return "falsch: keine korrekte Pose vorhanden"
    if not r.gross_richtig and r.wahl_gross:
        return "falsch: der eigenen falschen Dichte gefolgt"
    if r.gross_richtig and not r.wahl_gross:
        return "falsch: Hauptmodus war richtig, Wahl daneben"
    if r.gross_richtig and r.wahl_gross:
        return "falsch: Hauptmodus richtig, Wahl darin trotzdem falsch"
    return "falsch: Dichte falsch UND Wahl ausserhalb"


REIHE = ["richtig: aus dem Hauptmodus", "richtig: aus einem NEBENMODUS",
         "falsch: keine korrekte Pose vorhanden",
         "falsch: der eigenen falschen Dichte gefolgt",
         "falsch: Hauptmodus war richtig, Wahl daneben",
         "falsch: Hauptmodus richtig, Wahl darin trotzdem falsch",
         "falsch: Dichte falsch UND Wahl ausserhalb"]

gruppen = [("nur die Flow-Arme richtig", (0, 1, 1)),
           ("nur SigmaDock richtig", (1, 0, 0)),
           ("nur Minimal richtig", (0, 1, 0)),
           ("nur Separate richtig", (0, 0, 1)),
           ("SigmaDock und Minimal richtig", (1, 1, 0)),
           ("SigmaDock und Separate richtig", (1, 0, 1)),
           ("alle drei falsch", (0, 0, 0)),
           ("alle drei richtig", (1, 1, 1))]

print(f"\n=== C. Woran liegt es? Zerlegung je uneiniger Gruppe ===")
for name, schl in gruppen:
    k = idx[(muster.SigmaDock == bool(schl[0]))
            & (muster.Minimal == bool(schl[1]))
            & (muster.Separate == bool(schl[2]))]
    if len(k) == 0:
        continue
    print(f"\n  --- {name}  (n = {len(k)}) ---")
    tab = {}
    for arm in ARME:
        d = W[arm].loc[k]
        u = d.apply(ursache, axis=1).value_counts()
        tab[KURZ[arm]] = u
    tab = pd.DataFrame(tab).reindex(REIHE).fillna(0).astype(int)
    tab = tab[tab.sum(axis=1) > 0]
    print(f"    {'':<48}" + "".join(f"{c:>6}" for c in tab.columns))
    for zeile, werte in tab.iterrows():
        print(f"    {zeile:<48}" + "".join(f"{int(v):6d}" for v in werte))
    for arm in ARME:
        d = W[arm].loc[k]
        print(f"    [{KURZ[arm]}] korrekte Posen im Pool: "
              f"{d.n_korrekt.mean():.1f}/{int(d.n.mean())}   "
              f"groesster Modus {100 * d.gross_anteil.mean():.0f} %   "
              f"davon richtig {100 * d.gross_richtig.mean():.0f} %")

print(f"\n{len(t)} Zeilen nach {AUS}")
