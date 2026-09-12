"""Strukturelle Analyse der Posenverteilung: Moden, und WO die Fehler liegen.

WOZU, NACH ALLEM BISHERIGEN
    `koeder.py` hat gezeigt: SigmaFlow hat eine scharfe Spitze (Praezision@1
    79,8 %, @10 nur noch 65,1 %), SigmaDock ein flaches Plateau (75,2 / 71,0).
    Die Score-Erklaerung ist der p^4-Faktor, der bei SigmaFlows niedrigerem
    p_pb wie ein hartes Tor wirkt.

    Offen ist die STRUKTURELLE Seite: Sieht ein flaches Plateau anders aus als
    eine scharfe Spitze -- in den Koordinaten, nicht in den Scores?

DREI MASSE

    1. WO LIEGEN DIE FEHLER
       Abstand des Ligandschwerpunkts zur Kristallpose, getrennt nach
       richtigen und falschen Posen. Ein Fehler IN der Tasche (nur verdreht)
       ist ein besserer Koeder als einer daneben.

       Das braucht KEINE Atomzuordnung: der Schwerpunkt ist der Mittelwert
       aller Schweratome, und der ist von der Reihenfolge unabhaengig,
       solange die Atommenge dieselbe ist. Deshalb ist dieses Mass robust,
       auch wenn RDKit die Atome anders sortiert als erwartet.

    2. MODENSTRUKTUR
       Die K Posen je Komplex werden bei SCHWELLE Angstroem paarweisem RMSD
       geclustert (mittlere Bindung, ohne Ueberlagerung -- die Posen liegen
       schon im selben Bezugssystem, und sie aufeinanderzulegen wuerde genau
       das wegrechnen, worum es geht).

       Berichtet werden: Zahl der Moden, Anteil des groessten Modus, und ob
       der groesste Modus ueberwiegend RICHTIG ist. Fuer die Paar-RMSD ist
       die Atomreihenfolge noetig; sie wird gegen die erste Pose geprueft und
       bei Abweichung wird abgebrochen.

    3. ABDECKUNG
       Anteil der Posen im selben Modus wie die beste richtige Pose. Das ist
       "Trefferquote je Zug", aber strukturell statt ueber die RMSD-Schwelle.

WAS NICHT GEMESSEN WIRD
    Symmetrie. Ein gedrehter Phenylring zaehlt hier als Unterschied. Fuer den
    VERGLEICH zwischen Armen ist das unschaedlich (es trifft alle gleich), als
    Absolutwert ist der RMSD eine Obergrenze.

Aufruf:
    python SigmaFlow_Variants/struktur.py
    python SigmaFlow_Variants/struktur.py --schwelle 1.5
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

_HIER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HIER)
from zellen import SAETZE, lade_zelle, posenordner  # noqa: E402

from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

try:
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
except ImportError:
    sys.exit("ABBRUCH: scipy fehlt. pip install scipy")

def lade(pfad: str, sanitize: bool = False):
    m = Chem.MolFromMolFile(pfad, sanitize=sanitize, removeHs=False)
    if m is None or m.GetNumConformers() == 0:
        return None, None
    return (m.GetConformer().GetPositions(),
            tuple(a.GetSymbol() for a in m.GetAtoms()))


p = argparse.ArgumentParser()
p.add_argument("--schwelle", type=float, default=2.0,
               help="Clusterschwelle in Angstroem")
p.add_argument("--satz", choices=sorted(SAETZE), default="pb308",
               help="pb308 oder astex -- verschiedene Verzeichnisse und Referenzen")
p.add_argument("--true", default=None,
               help="Referenzverzeichnis; Vorgabe kommt aus zellen.SAETZE")
p.add_argument("--out", default=os.path.join(_HIER, "struktur.csv"))
a = p.parse_args()

SATZ = SAETZE[a.satz]
REF = a.true or SATZ["referenz"]

verfuegbar = {}
for z in SATZ["zellen"]:
    w = posenordner(a.satz, z["arm"], z["nfe"])
    if os.path.isdir(w):
        verfuegbar[(z["arm"], z["nfe"])] = w
if not verfuegbar:
    sys.exit(f"ABBRUCH: keine Posenverzeichnisse fuer Satz '{a.satz}'. "
             f"Erwartet unter <lauf>/{SATZ['posen_unterordner']}/... "
             f"-- Archiv entpackt?")

mengen = [{os.path.basename(f).split("__")[0]
           for f in glob.glob(os.path.join(w, "seed_*", "*.sdf"))}
          for w in verfuegbar.values()]
codes = sorted(set.intersection(*mengen))
print(f"{len(verfuegbar)} Zellen, {len(codes)} Komplexe in allen\n")

print("Tabellen:")
tab = {}
for z in SATZ["zellen"]:
    if (z["arm"], z["nfe"]) in verfuegbar:
        m = lade_zelle(z)
        tab[(z["arm"], z["nfe"])] = m[m["complex"].isin(codes)]

# Fragmentzahlen sind je Satz eigen. Fehlt die Datei, bleibt n_frag leer --
# die Modenanalyse braucht sie nicht, nur die Stratifizierung.
frag = {}
fp = os.path.join(_HIER, f"fragmentzahl_{a.satz}.csv")
if not os.path.isfile(fp) and a.satz == "pb308":
    fp = os.path.join(_HIER, "fragmentzahl_pb308.csv")
if os.path.isfile(fp):
    f = pd.read_csv(fp)
    frag = dict(zip(f["complex"], f["n_frag"]))

zeilen = []
for code in codes:
    kris_datei = os.path.join(REF, code, SATZ["referenz_ligand"].format(code=code))
    kris, kris_el = lade(kris_datei, sanitize=True)
    if kris is None:
        print(f"  {code}: Kristallpose nicht ladbar -- uebersprungen")
        continue
    kris_zentrum = kris.mean(axis=0)

    for (arm, nfe), w in verfuegbar.items():
        g = tab[(arm, nfe)]
        g = g[g["complex"] == code]
        if g.empty:
            continue
        acc_von = dict(zip(g["seed"], g["acc"]))

        dateien = sorted(glob.glob(os.path.join(w, "seed_*", f"{code}__*.sdf")),
                         key=lambda f: int(re.search(r"seed_(\d+)", f).group(1)))
        x, acc, ref = [], [], None
        for f in dateien:
            s = int(re.search(r"seed_(\d+)", f).group(1))
            k, el = lade(f)
            if k is None:
                continue
            if ref is None:
                ref = el
            elif el != ref:
                sys.exit(f"ABBRUCH: {f} hat eine andere Atomfolge als die "
                         f"erste Pose. Paarweise RMSD waeren sinnlos.")
            x.append(k)
            acc.append(bool(acc_von.get(s, False)))
        if len(x) < 3:
            continue
        x = np.asarray(x, float)
        acc = np.asarray(acc, bool)

        # --- 1. Wo liegen die Fehler? ---------------------------------
        d_zent = np.linalg.norm(x.mean(axis=1) - kris_zentrum, axis=1)

        # --- 2. Moden --------------------------------------------------
        n = len(x)
        d = np.sqrt(((x[:, None, :, :] - x[None, :, :, :]) ** 2).sum(-1).mean(-1))
        np.fill_diagonal(d, 0.0)
        Z = linkage(squareform(d, checks=False), method="average")
        lab = fcluster(Z, t=a.schwelle, criterion="distance")
        groessen = np.bincount(lab)[1:]
        gross = int(np.argmax(groessen)) + 1
        in_gross = lab == gross
        anteil = float(groessen.max() / n)
        # Entropie der Modenverteilung, normiert auf log(n) -- 0 = ein
        # einziger Modus, 1 = jede Pose ein eigener.
        pk = groessen / n
        ent = float(-(pk * np.log(pk)).sum() / np.log(n)) if n > 1 else 0.0

        # --- 3. Abdeckung ---------------------------------------------
        if acc.any():
            # Modus, in dem die meisten richtigen Posen liegen
            richtige_moden = np.bincount(lab[acc], minlength=lab.max() + 1)
            richtig_modus = int(np.argmax(richtige_moden))
            abdeckung = float((lab == richtig_modus).mean())
        else:
            richtig_modus, abdeckung = -1, np.nan

        zeilen.append({
            "complex": code, "n_frag": frag.get(code, np.nan),
            "arm": arm, "nfe": nfe, "n_posen": n,
            "trefferquote": float(acc.mean()),
            "n_moden": int(len(groessen)),
            "groesster_modus": round(anteil, 4),
            "groesster_richtig": bool(acc[in_gross].mean() > 0.5),
            "entropie": round(ent, 4),
            "abdeckung_richtig": round(abdeckung, 4) if acc.any() else np.nan,
            "d_zentrum_richtig": round(float(d_zent[acc].mean()), 3) if acc.any() else np.nan,
            "d_zentrum_falsch": round(float(d_zent[~acc].mean()), 3) if (~acc).any() else np.nan,
        })
    print(f"  {code} fertig", flush=True)

t = pd.DataFrame(zeilen)
t.to_csv(a.out, index=False)
print(f"\n{len(t)} Zeilen nach {a.out}\n")

print(f"=== MODENSTRUKTUR (Clusterschwelle {a.schwelle} A), Mittel ueber "
      f"{t['complex'].nunique()} Komplexe ===")
print(f"  {'Zelle':<16}{'Posen':>7}{'Moden':>7}{'groesster':>11}"
      f"{'groesster richtig':>19}{'Entropie':>10}")
for (arm, nfe), g in t.groupby(["arm", "nfe"]):
    print(f"  {arm + ' ' + str(nfe):<16}{g['n_posen'].mean():7.0f}"
          f"{g['n_moden'].mean():7.1f}{100 * g['groesster_modus'].mean():10.1f}%"
          f"{100 * g['groesster_richtig'].mean():18.1f}%"
          f"{g['entropie'].mean():10.3f}")

print("\n=== WO LIEGEN DIE FEHLER? Schwerpunktabstand zur Kristallpose (A) ===")
print(f"  {'Zelle':<16}{'richtige Posen':>16}{'falsche Posen':>15}{'Differenz':>11}")
for (arm, nfe), g in t.groupby(["arm", "nfe"]):
    r, f_ = g["d_zentrum_richtig"].mean(), g["d_zentrum_falsch"].mean()
    print(f"  {arm + ' ' + str(nfe):<16}{r:16.2f}{f_:15.2f}{f_ - r:11.2f}")

print("\n=== ABDECKUNG: Anteil der Posen im Modus der richtigen ===")
print(f"  {'Zelle':<16}{'Abdeckung':>11}{'Trefferquote':>14}")
for (arm, nfe), g in t.groupby(["arm", "nfe"]):
    print(f"  {arm + ' ' + str(nfe):<16}{100 * g['abdeckung_richtig'].mean():10.1f}%"
          f"{100 * g['trefferquote'].mean():13.1f}%")

print("""
LESEHILFE
  Moden              Zahl der Cluster bei der Schwelle. Wenige = konzentriert.
  groesster          Anteil der Posen im groessten Modus.
  groesster richtig  Bei wie vielen Komplexen ist der groesste Modus der
                     RICHTIGE? Das ist die eigentliche Frage: konzentriert
                     sich der Arm auf die richtige Antwort oder auf eine
                     falsche?
  Differenz          Liegen falsche Posen weiter vom Kristallschwerpunkt weg?
                     Klein = Fehler sitzen IN der Tasche, nur verdreht --
                     also gute Koeder. Gross = Fehler liegen daneben.""")
