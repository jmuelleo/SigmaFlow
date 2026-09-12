"""WIE schlimm sind die ungueltigen Kristallposen -- knapp daneben oder grob?

Die Booleschen Pruefungen sagen nur ja/nein. `full_report=True` liefert
daneben die Zahlenwerte, aus denen sie entstehen. Dieses Skript liest sie und
beantwortet: liegen die Durchgefallenen dicht an der Schwelle oder weit
darunter?

WIE DIE SCHWELLE BESTIMMT WIRD
    Nicht aus der Dokumentation, sondern EMPIRISCH aus den Daten: der
    kleinste Wert unter den Bestandenen und der groesste unter den
    Durchgefallenen klammern sie ein. So kann keine Versionsaenderung von
    PoseBusters unbemerkt eine falsche Schwelle unterschieben.

Aufruf:
    $ARC_SF_ENV/bin/python arc/analyse_schwere.py [wurzel] [satz]
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

_HIER = os.path.dirname(os.path.abspath(__file__))
quelle = open(os.path.join(_HIER, "analyse_crystal_pb.py"), encoding="utf-8").read()
ns = {"__name__": "_defs", "__file__": os.path.join(_HIER, "analyse_crystal_pb.py")}
sys.argv = [sys.argv[0]]
exec(compile(quelle[:quelle.index("saetze = sorted")], "analyse_crystal_pb.py", "exec"), ns)  # noqa: S102
als_bool = ns["als_bool"]

argv = sys.modules["__main__"].__dict__.get("_argv", [])
WURZEL = "/data/stat-cadd/shug8458/arc_runs/crystal_pb"
SATZ = "pdbbind_pocket"

# (boolesche Pruefung, Zahlenspalte, Richtung: "klein_schlecht" | "gross_schlecht")
PAARE = [
    ("minimum_distance_to_protein", "most_extreme_relative_distance_protein", "klein_schlecht"),
    ("minimum_distance_to_waters", "most_extreme_relative_distance_waters", "klein_schlecht"),
    ("minimum_distance_to_organic_cofactors",
     "most_extreme_relative_distance_organic_cofactors", "klein_schlecht"),
    ("minimum_distance_to_inorganic_cofactors",
     "most_extreme_relative_distance_inorganic_cofactors", "klein_schlecht"),
    ("volume_overlap_with_protein", "volume_overlap_protein", "gross_schlecht"),
    ("internal_energy", "energy_ratio", "gross_schlecht"),
]

teile = [f for f in sorted(glob.glob(os.path.join(WURZEL, SATZ, "crystal_*.csv")))
         if not f.endswith("crystal_alle.csv")]
if not teile:
    sys.exit(f"keine Tabellen unter {os.path.join(WURZEL, SATZ)}")
t = pd.concat([pd.read_csv(f) for f in teile], ignore_index=True)
sp = "file" if "file" in t.columns else t.columns[0]
if "komplex" not in t.columns:
    t["komplex"] = t[sp].map(lambda p: os.path.basename(os.path.dirname(str(p))))
t = t.groupby("komplex", sort=False).first()
print(f"{SATZ}: {len(t)} Komplexe (Kopie 0)\n")

for pruef, wert, richtung in PAARE:
    if pruef not in t.columns or wert not in t.columns:
        print(f"{pruef}: Spalte fehlt, uebersprungen")
        continue
    ok = als_bool(t[pruef])
    v = pd.to_numeric(t[wert], errors="coerce")
    schlecht = (~ok) & v.notna()
    n = int(schlecht.sum())
    print(f"=== {pruef}")
    print(f"    durchgefallen: {n} von {len(t)}  ({100 * n / len(t):.2f} %)"
          f"{'   davon ohne Zahlenwert: ' + str(int((~ok & v.isna()).sum())) if (~ok & v.isna()).any() else ''}")
    if n == 0:
        print()
        continue

    gut = v[ok & v.notna()]
    if richtung == "klein_schlecht":
        print(f"    Schwelle liegt zwischen {v[schlecht].max():.4f} (groesster "
              f"Durchgefallener) und {gut.min():.4f} (kleinster Bestandener)")
        q = np.percentile(v[schlecht], [5, 25, 50, 75, 95])
        print(f"    Verteilung der Durchgefallenen (relativer Abstand, 1,0 = Beruehrung):")
    else:
        print(f"    Schwelle liegt zwischen {gut.max():.4f} (groesster "
              f"Bestandener) und {v[schlecht].min():.4f} (kleinster Durchgefallener)")
        q = np.percentile(v[schlecht], [5, 25, 50, 75, 95])
        print(f"    Verteilung der Durchgefallenen:")
    print(f"      5 %   {q[0]:8.4f}")
    print(f"     25 %   {q[1]:8.4f}")
    print(f"     Median {q[2]:8.4f}")
    print(f"     75 %   {q[3]:8.4f}")
    print(f"     95 %   {q[4]:8.4f}")

    # Absolute Fehlmenge in Angstroem, wo es sie gibt.
    a, b = f"most_extreme_sum_radii_scaled_{pruef.split('_to_')[-1]}", \
           f"most_extreme_distance_{pruef.split('_to_')[-1]}"
    if a in t.columns and b in t.columns:
        d = pd.to_numeric(t.loc[schlecht, a], errors="coerce") - \
            pd.to_numeric(t.loc[schlecht, b], errors="coerce")
        d = d.dropna()
        if len(d):
            qq = np.percentile(d, [25, 50, 75, 95])
            print(f"    Fehlmenge in Angstroem (Sollabstand minus Istabstand):")
            print(f"      25 % {qq[0]:.3f}   Median {qq[1]:.3f}   "
                  f"75 % {qq[2]:.3f}   95 % {qq[3]:.3f}   max {d.max():.3f}")
    print()

# Wie knapp ist "knapp"? Anteil der Durchgefallenen innerhalb enger Baender.
pruef, wert = "minimum_distance_to_protein", "most_extreme_relative_distance_protein"
if pruef in t.columns and wert in t.columns:
    ok = als_bool(t[pruef])
    v = pd.to_numeric(t[wert], errors="coerce")
    s = v[(~ok) & v.notna()]
    schwelle = v[ok & v.notna()].min()
    print(f"=== Wie knapp scheitern die {len(s)} Protein-Kollisionen? "
          f"(Schwelle rund {schwelle:.3f})")
    for rand in (0.01, 0.02, 0.05, 0.10, 0.20):
        anteil = 100 * (s > schwelle - rand).mean()
        print(f"    innerhalb von {rand:.2f} unter der Schwelle: {anteil:5.1f} %")
    print(f"    weiter als 0,20 darunter:                  "
          f"{100 * (s <= schwelle - 0.20).mean():5.1f} %")
