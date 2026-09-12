"""SCHADENSPROFIL erzeugter Posen -- nicht "gueltig ja/nein", sondern wie viel.

PB-Validitaet ist eine Null-oder-Eins-Groesse und sagt nichts darueber, ob eine
Pose um ein Haar oder um Welten scheitert. `full_report=True` liefert die
Zaehler und Betraege dahinter. Dieses Skript stellt sie zusammen:

    Wieviele Bindungen sind Ausreisser -- von wievielen?
    Wieviele Winkel? Wieviele ligandinterne Atompaare?
    Wieviele Kontakte zum Protein -- von wievielen geprueften?
    Und wie weit daneben, in Angstroem beziehungsweise relativ?

Alles wird auf die Groesse des Molekuels bezogen. Zwei Ausreisserbindungen sind
bei einem Molekuel mit 15 Bindungen etwas anderes als bei einem mit 60.

LESEART DER RELATIVWERTE
    `*_relative_length`, `*_relative_angle`, `*_relative_distance` sind
    Verhaeltnisse zum Sollwert: 1,00 ist perfekt. 0,80 heisst 20 % zu kurz,
    1,25 heisst 25 % zu lang. Die Schwellen von PoseBusters liegen bei etwa
    0,75 fuer Abstaende und 0,25 relative Abweichung bei Bindungen und Winkeln.

WARUM DIE KRISTALLPOSEN IMMER MITLAUFEN
    Eine Zahl wie "Median 2 kollidierende Atompaare" ist ohne Bezugsgroesse
    bedeutungslos. Die Kristallposen sind die bestmoegliche Antwort, die es
    gibt -- alles, was das Modell erzeugt, wird daran gemessen.

Aufruf:
    $ARC_SF_ENV/bin/python arc/analyse_posen_schaden.py <verzeichnis> [...]
    (Verzeichnisse mit rd_*seed*.csv aus FULL_REPORT=1)
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

KRISTALL = "/data/stat-cadd/shug8458/arc_runs/crystal_pb/pdbbind_pocket"
WAHR = {"true", "1", "1.0"}


def bool_sp(s):
    return s.astype(str).str.strip().str.lower().isin(WAHR)


def z(t, c):
    if c not in t.columns:
        return pd.Series(np.nan, index=t.index, dtype=float)
    return pd.to_numeric(t[c], errors="coerce")


def zeile(name, v, einheit=""):
    v = pd.Series(v).dropna()
    if not len(v):
        print(f"  {name:<44}   (keine Werte)")
        return
    q = np.percentile(v, [50, 75, 90, 99])
    print(f"  {name:<44} {q[0]:8.2f} {q[1]:8.2f} {q[2]:8.2f} {q[3]:8.2f} "
          f"{v.max():9.2f} {einheit}")


def kopf():
    print(f"  {'':<44} {'Median':>8} {'75 %':>8} {'90 %':>8} {'99 %':>8} {'max':>9}")


def profil(name, t):
    print()
    print("=" * 94)
    print(f"  {name}   ({len(t)} Posen)")
    print("=" * 94)

    nb, na, nn = z(t, "number_bonds"), z(t, "number_angles"), z(t, "number_noncov_pairs")
    print(f"  Molekuelgroesse: Median {nb.median():.0f} Bindungen, "
          f"{na.median():.0f} Winkel, {nn.median():.0f} ligandinterne Atompaare")

    aus_kurz = z(t, "number_short_outlier_bonds").fillna(0)
    aus_lang = z(t, "number_long_outlier_bonds").fillna(0)
    aus_b = aus_kurz + aus_lang
    aus_w = z(t, "number_outlier_angles").fillna(0)
    kl = z(t, "number_clashes").fillna(0)

    print()
    print("  LIGANDINTERN -- Anteil der Posen mit mindestens einem Fehler:")
    print(f"    Bindungslaenge {100 * (aus_b > 0).mean():6.2f} %"
          f"     Winkel {100 * (aus_w > 0).mean():6.2f} %"
          f"     innerer Clash {100 * (kl > 0).mean():6.2f} %")

    print()
    print("  Wieviele sind betroffen, WENN etwas betroffen ist?")
    kopf()
    zeile("Ausreisser-Bindungen (Anzahl)", aus_b[aus_b > 0])
    zeile("  Anteil aller Bindungen", (100 * aus_b / nb)[aus_b > 0], "%")
    zeile("Ausreisser-Winkel (Anzahl)", aus_w[aus_w > 0])
    zeile("  Anteil aller Winkel", (100 * aus_w / na)[aus_w > 0], "%")
    zeile("innere Clashs (Anzahl Atompaare)", kl[kl > 0])
    zeile("  Anteil aller inneren Paare", (100 * kl / nn)[kl > 0], "%")

    print()
    print("  WIE WEIT daneben?  (1,00 = Sollwert)")
    kopf()
    zeile("laengste Bindung, relativ", z(t, "longest_bond_relative_length")[aus_lang > 0])
    zeile("kuerzeste Bindung, relativ", z(t, "shortest_bond_relative_length")[aus_kurz > 0])
    zeile("extremster Winkel, relativ", z(t, "most_extreme_relative_angle")[aus_w > 0])
    zeile("kuerzester innerer Abstand, relativ",
          z(t, "shortest_noncovalent_relative_distance")[kl > 0])
    zeile("aromatischer Ring, Abstand zur Ebene",
          z(t, "aromatic_ring_maximum_distance_from_plane"), "A")
    zeile("nicht-aromatischer Ring, zur Ebene",
          z(t, "non-aromatic_ring_maximum_distance_from_plane"), "A")
    zeile("Doppelbindung, Abstand zur Ebene",
          z(t, "double_bond_maximum_distance_from_plane"), "A")
    zeile("Energieverhaeltnis (Schwelle 100)", z(t, "energy_ratio"))

    if "num_pairwise_clashes_protein" not in t.columns:
        print()
        print("  (keine Rezeptorspalten -- ohne mol_cond gebustet?)")
        return

    npr = z(t, "num_pairwise_clashes_protein")
    rel = z(t, "most_extreme_relative_distance_protein")
    d = z(t, "most_extreme_distance_protein")
    soll = z(t, "most_extreme_sum_radii_scaled_protein")
    if "minimum_distance_to_protein" in t.columns:
        ok = bool_sp(t["minimum_distance_to_protein"])
    else:
        ok = pd.Series(True, index=t.index)

    print()
    print(f"  PROTEIN -- {100 * (~ok).mean():.2f} % der Posen mit mindestens einer Kollision")
    kopf()
    zeile("kollidierende Atompaare (Anzahl)", npr[~ok])
    zeile("  Anteil aller geprueften Kontakte", (100 * npr / nn)[~ok], "%")
    zeile("extremster Kontakt, relativ (Schwelle 0,75)", rel[~ok])
    zeile("Fehlmenge Soll minus Ist", (soll - d)[~ok], "A")
    zeile("Volumenueberlapp (Schwelle 0,075)", z(t, "volume_overlap_protein")[~ok])

    sl = "most_extreme_ligand_element_protein"
    spr = "most_extreme_protein_element_protein"
    if sl in t.columns and spr in t.columns:
        paar = (t[sl].astype(str) + "-" + t[spr].astype(str))[~ok]
        dd = d[~ok]
        print()
        print("  WELCHE Atome kollidieren (extremster Kontakt):")
        for nm, an in paar.value_counts().head(6).items():
            print(f"    {nm:<8} {an:7d}  ({100 * an / len(paar):5.1f} %)"
                  f"   Median {dd[paar == nm].median():.3f} A")
        eng = int(((dd >= 1.0) & (dd < 1.9)).sum())
        print(f"    im kovalenten Fenster 1,0-1,9 A: {eng} "
              f"({100 * eng / max(len(paar), 1):.1f} %)")
        print("      Beim MODELL ist das immer ein echter Fehler -- es knuepft")
        print("      keine Bindungen. Bei den Kristallposen war es korrekte Chemie.")


def lade(d, muster):
    f = [x for x in sorted(glob.glob(os.path.join(d, muster)))
         if not x.endswith("crystal_alle.csv")]
    if not f:
        return None
    return pd.concat([pd.read_csv(x) for x in f], ignore_index=True)


k = lade(KRISTALL, "crystal_*.csv")
if k is not None:
    sp = "file" if "file" in k.columns else k.columns[0]
    if "komplex" not in k.columns:
        k["komplex"] = k[sp].map(lambda p: os.path.basename(os.path.dirname(str(p))))
    profil("BEZUGSGROESSE: PDBBind-Kristallposen (Kopie 0)",
           k.groupby("komplex", sort=False).first())

for d in sys.argv[1:]:
    t = lade(d, "rd_*seed*.csv")
    if t is None:
        print(f"\n{d}: keine rd_*seed*.csv gefunden")
        continue
    profil(os.path.basename(d.rstrip("/")), t)
