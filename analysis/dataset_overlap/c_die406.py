"""Sind die 406 Komplexe, die R1 gegenueber v2020-Original fehlen,
dem PB308-Satz aehnlicher als der Rest?

SigmaDock trainierte auf v2020-Original (19.443), unsere Laeufe auf
v2020.R1 (19.037). Die Differenz sind 406 Komplexe, die PDBbind selbst bis
v2024 aussortiert hat. Falls diese 406 ueberzufaellig nah an PB308 lagen,
waere ein Teil der Reproduktionsluecke schlicht fehlende Naehe zum
Testsatz -- also ein Datenargument statt eines Trainingsarguments.

Aufruf:
    python c_die406.py <r1_ids.txt>
wobei r1_ids.txt die 19.037 Verzeichnisnamen des R1-Bestands enthaelt
(ein Code je Zeile, Grossschreibung egal).
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HIER = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(0)

df = pd.read_csv(os.path.join(HIER, "aehnlichkeit_19443.csv"))
r1 = {z.strip().lower() for z in open(sys.argv[1]) if z.strip()}
df["in_r1"] = df.code.str.lower().isin(r1)

weg = df[~df.in_r1]
blieb = df[df.in_r1]
print(f"v2020-Original : {len(df)}")
print(f"davon in R1    : {len(blieb)}")
print(f"ENTFERNT       : {len(weg)}")
fehlt = r1 - set(df.code.str.lower())
if fehlt:
    print(f"WARNUNG: {len(fehlt)} R1-Codes stehen nicht im Original: "
          f"{sorted(fehlt)[:5]}")
print()

# --------------------------------------------------------- Kernvergleich
def block(name, spalte, schwellen):
    a = weg[spalte].dropna().to_numpy(float)
    b = blieb[spalte].dropna().to_numpy(float)
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    # Wahrscheinlichkeit, dass ein zufaellig gezogener Entfernter naeher
    # liegt als ein zufaellig gezogener Verbliebener (AUC / common language)
    auc = u / (len(a) * len(b))
    print(f"{name}   ({spalte})")
    print(f"  entfernt  n={len(a):5d}  Median {np.median(a):.4f}  "
          f"Mittel {a.mean():.4f}")
    print(f"  verblieben n={len(b):5d}  Median {np.median(b):.4f}  "
          f"Mittel {b.mean():.4f}")
    print(f"  Mann-Whitney p = {p:.4g},  P(entfernt naeher) = {auc:.3f}")
    print(f"  {'Schwelle':>10} {'entfernt':>12} {'verblieben':>12} "
          f"{'Verhaeltnis':>12} {'p (Fisher)':>12}")
    for t in schwellen:
        na, nb = int((a >= t).sum()), int((b >= t).sum())
        pa, pb = 100 * na / len(a), 100 * nb / len(b)
        odds, pf = stats.fisher_exact([[na, len(a) - na], [nb, len(b) - nb]])
        v = pa / pb if pb > 0 else np.inf
        print(f"  {t:>10.2f} {pa:>10.2f} % {pb:>10.2f} % "
              f"{v:>11.2f}x {pf:>12.4g}")
    print()


block("PROTEIN  k-mer-Containment", "prot_cont", [0.5, 0.9, 0.95, 0.99])
block("LIGAND   max. Tanimoto", "lig_tanimoto", [0.4, 0.6, 0.7, 0.9, 1.0])

# -------------------------------------------------- gemeinsame Bedingung
for pt, lt in [(0.9, 0.7), (0.95, 0.9), (0.99, 1.0)]:
    na = int(((weg.prot_cont >= pt) & (weg.lig_tanimoto >= lt)).sum())
    nb = int(((blieb.prot_cont >= pt) & (blieb.lig_tanimoto >= lt)).sum())
    odds, pf = stats.fisher_exact([[na, len(weg) - na], [nb, len(blieb) - nb]])
    print(f"prot>={pt} UND lig>={lt}: entfernt {na}/{len(weg)} "
          f"({100*na/len(weg):.2f} %), verblieben {nb}/{len(blieb)} "
          f"({100*nb/len(blieb):.2f} %), p = {pf:.4g}")
print()

# ----------------------------------------------- Kontrolle: Stoerfaktoren
print("Kontrolle -- unterscheiden sich die Gruppen sonst?")
for s in ["jahr", "n_res"]:
    a, b = weg[s].to_numpy(float), blieb[s].to_numpy(float)
    p = stats.mannwhitneyu(a, b, alternative="two-sided")[1]
    print(f"  {s:8s} entfernt {np.median(a):8.1f}   verblieben "
          f"{np.median(b):8.1f}   p = {p:.4g}")

# Wie viele PB308-Ziele verlieren ueberhaupt einen nahen Trainingsnachbarn?
print("\nWirkung auf den Testsatz: PB308-Ziele, deren naechster Nachbar")
print("unter den ENTFERNTEN lag (prot_cont >= 0.9):")
nah = weg[weg.prot_cont >= 0.9]
print(f"  betroffene PB308-Ziele: {nah.prot_partner.nunique()} von 308")
if len(nah):
    print(nah.groupby("prot_partner").size().sort_values(ascending=False)
          .head(15).to_string())
