"""Ligandinterne gegen rezeptorseitige Pruefungen, je Zug.

Die Aussage der Arbeit ("Flow Matching gewinnt die Ligandenchemie") betrifft
die intrinsische Geometrie des Liganden. Die Spalte `valid` verlangt dagegen
ALLE Pruefungen, auch Abstaende und Volumenueberlappung zum Rezeptor. Hier
werden beide Gruppen getrennt ausgewiesen.
"""
import glob
import pathlib

quelle = pathlib.Path("vergleich_rezept.py").read_text(encoding="utf-8")
ns = {"__name__": "_defs", "__file__": "vergleich_rezept.py"}
exec(compile(quelle[:quelle.index("ZELLEN = {")], "vergleich_rezept.py", "exec"), ns)  # noqa: S102
zelle = ns["zelle"]

LIGAND = ["sanitization", "inchi_convertible", "all_atoms_connected", "no_radicals",
          "molecular_formula", "molecular_bonds", "double_bond_stereochemistry",
          "tetrahedral_chirality", "bond_lengths", "bond_angles",
          "internal_steric_clash", "aromatic_ring_flatness",
          "non-aromatic_ring_non-flatness", "double_bond_flatness", "internal_energy"]
REZEPTOR = ["protein-ligand_maximum_distance", "minimum_distance_to_protein",
            "minimum_distance_to_organic_cofactors",
            "minimum_distance_to_inorganic_cofactors", "minimum_distance_to_waters",
            "volume_overlap_with_protein", "volume_overlap_with_organic_cofactors",
            "volume_overlap_with_inorganic_cofactors", "volume_overlap_with_waters"]


def g1(m):
    t = glob.glob(m)
    if not t:
        raise SystemExit(m)
    return t[0]


C = "_cmp"
ZELLEN = {
    "SigmaDock 72h": (f"{C}/sd_endpunkt_40seeds/*/posebusters_redock_curve/*nfe25*/rd_*_seed*.csv",
                      g1(f"{C}/sd_endpunkt_40seeds/*/learning_curve_cpu/*nfe25*/gnina_scores.csv")),
    "Minimal 72h":   (f"{C}/endpunkt_min_nfe25/*/posebusters_redock_curve/*/rd_*_seed*.csv",
                      g1(f"{C}/endpunkt_min_nfe25/*/learning_curve_cpu/*/gnina_scores.csv")),
    "Minimal Rezept": ("_recipe/posebusters_redock/sigmaflow_minimal__recipe__confsampled/rd_*_seed*.csv",
                       g1("_recipe/GNINA-SCORE-*/gnina_scores_*.csv")),
}

print(f"{'Modell':<16} {'Ligand-Chemie':>14} {'Rezeptorseite':>14} {'alles':>8}")
for name, (rg, gc) in ZELLEN.items():
    m = zelle(rg, gc)
    lig = [c for c in LIGAND if c in m.columns]
    rez = [c for c in REZEPTOR if c in m.columns]
    print(f"{name:<16} {100*m[lig].all(axis=1).mean():14.2f} "
          f"{100*m[rez].all(axis=1).mean():14.2f} {100*m['valid'].mean():8.2f}")
