"""Ligandinterne gegen rezeptorseitige Gueltigkeit -- erzeugte Posen.

Dieselbe Aufteilung wie bei den Kristallposen:
    INTERN  Chemie, Bindungen, Planaritaeten, Energie -- haengt NICHT vom
            Rezeptor ab, misst allein die Ligandgeometrie.
    LAGE    Abstaende und Ueberlappungen zu Protein, Kofaktoren, Wasser.
So laesst sich sagen, ob eine ungueltige Pose ein chemisch kaputtes Molekuel
ist oder ein chemisch sauberes an der falschen Stelle.
"""
import glob
import pathlib
import sys

import pandas as pd

quelle = pathlib.Path("vergleich_rezept.py").read_text(encoding="utf-8")
ns = {"__name__": "_defs", "__file__": "vergleich_rezept.py"}
exec(compile(quelle[:quelle.index("ZELLEN = {")], "vergleich_rezept.py", "exec"), ns)  # noqa: S102
zelle, wahl = ns["zelle"], ns["wahl"]

LAGE = ["minimum_distance_to_protein", "minimum_distance_to_organic_cofactors",
        "minimum_distance_to_inorganic_cofactors", "minimum_distance_to_waters",
        "volume_overlap_with_protein", "volume_overlap_with_organic_cofactors",
        "volume_overlap_with_inorganic_cofactors", "volume_overlap_with_waters",
        "protein-ligand_maximum_distance"]
INTERN = ["sanitization", "inchi_convertible", "all_atoms_connected", "no_radicals",
          "molecular_formula", "molecular_bonds", "double_bond_stereochemistry",
          "tetrahedral_chirality", "bond_lengths", "bond_angles",
          "internal_steric_clash", "aromatic_ring_flatness",
          "non-aromatic_ring_non-flatness", "double_bond_flatness", "internal_energy"]


def g1(m):
    t = sorted(glob.glob(m, recursive=True))
    if not t:
        sys.exit(f"nicht gefunden: {m}")
    return t[0]


def rezept(tag):
    return (f"_recipe_all/posebusters_redock/{tag}/rd_*seed*.csv",
            g1(f"_recipe_all/GNINA-SCORE-{tag}_*/gnina_scores_{tag}.csv"))


def h72(basis, nfe):
    return (g1(f"{basis}/**/posebusters_redock_curve/*__nfe{nfe}__sampled") + "/rd_*seed*.csv",
            g1(f"{basis}/**/learning_curve_cpu/*__nfe{nfe}__sampled/gnina_scores.csv"))


ZELLEN = {
    ("Rezept", "Minimal", 25): rezept("sigmaflow_minimal__recipe__confsampled"),
    ("Rezept", "Separate", 25): rezept("exp110__recipe__confsampled"),
    ("Rezept", "Minimal", 5): rezept("sigmaflow_minimal__nfe5__recipe__confsampled"),
    ("Rezept", "Separate", 5): rezept("exp110__nfe5__recipe__confsampled"),
    ("72 h", "SigmaDock", 25): h72("_cmp/sd_endpunkt_40seeds", 25),
    ("72 h", "Minimal", 25): h72("_cmp/endpunkt_min_nfe25", 25),
    ("72 h", "Separate", 25): h72("_cmp/endpunkt_sep_nfe25", 25),
}

daten = {}
for k, (rg, gn) in ZELLEN.items():
    m = zelle(rg, gn)
    m = m[m["seed"] < 40]
    for c in LAGE + INTERN:
        if c not in m.columns:
            sys.exit(f"Spalte fehlt: {c}")
    m["intern"] = m[INTERN].all(axis=1)
    m["lage"] = m[LAGE].all(axis=1)
    daten[k] = m

print("=== JE ZUG (alle 40 Ziehungen einzeln) ===")
print(f"{'Zelle':<26} {'intern ok':>10} {'Lage ok':>9} {'beides':>8} "
      f"{'nur Lage kaputt':>16} {'nur intern kaputt':>18}")
for k, m in daten.items():
    n = f"{k[0]}, {k[1]}, {k[2]} Schr."
    i, l = m["intern"], m["lage"]
    print(f"{n:<26} {100*i.mean():9.2f}% {100*l.mean():8.2f}% {100*(i&l).mean():7.2f}% "
          f"{100*(i&~l).mean():15.2f}% {100*(~i&l).mean():17.2f}%")

print("\n=== TOP-1 NACH MIXED SCORE, K = 40 ===")
print(f"{'Zelle':<26} {'intern ok':>10} {'Lage ok':>9} {'beides':>8}")
for k, m in daten.items():
    n = f"{k[0]}, {k[1]}, {k[2]} Schr."
    z = [100 * wahl(m, c, "heuristik").mean() for c in ("intern", "lage", "valid")]
    print(f"{n:<26} {z[0]:9.2f}% {z[1]:8.2f}% {z[2]:7.2f}%")

print("\n=== Wenn eine Pose ungueltig ist, WORAN liegt es? (je Zug) ===")
print(f"{'Zelle':<26} {'nur Lage':>10} {'nur intern':>11} {'beides':>8}")
for k, m in daten.items():
    n = f"{k[0]}, {k[1]}, {k[2]} Schr."
    i, l = m["intern"], m["lage"]
    schlecht = ~(i & l)
    s = schlecht.sum()
    print(f"{n:<26} {100*(i&~l).sum()/s:9.1f}% {100*(~i&l).sum()/s:10.1f}% "
          f"{100*(~i&~l).sum()/s:7.1f}%")
