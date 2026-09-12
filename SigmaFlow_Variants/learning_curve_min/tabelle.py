"""Vergleichstabelle: die vier Snapshots des 72-h-Laufs gegen den 12-h-Lauf.

Berichtet werden fuenf Groessen je Zug, alle auf denselben Posen:
    <2 A                RMSD symmetriekorrigiert, aus per_pose.csv
    PB Ligand           alle 15 ligandinternen Pruefungen bestanden
    PB mit Protein      zusaetzlich alle 9 proteinseitigen
    <2 A & PB Ligand    Konjunktion
    <2 A & PB Protein   Konjunktion, das Hauptkriterium der Arbeit

WARUM DIE RMSD AUS per_pose.csv UND NICHT AUS PoseBusters
    PoseBusters bringt eine eigene Spalte rmsd_<=_2A mit. Die Arbeit
    berichtet aber durchgehend die symmetriekorrigierte RMSD aus
    evaluate_run (spyrmsd), und die 12-h-Zeile ist damit gerechnet. Zwei
    Konventionen in einer Tabelle waeren die schlechteste Variante.
"""
import csv
import math
import pathlib

import pandas as pd

from pb_bool import als_bool

HIER = pathlib.Path(__file__).resolve().parent
VERGLEICH = (HIER.parent / "posebusters_full_comparison")

LIGAND = ["sanitization", "inchi_convertible", "all_atoms_connected",
          "no_radicals", "molecular_formula", "molecular_bonds",
          "double_bond_stereochemistry", "tetrahedral_chirality",
          "bond_lengths", "bond_angles", "internal_steric_clash",
          "aromatic_ring_flatness", "non-aromatic_ring_non-flatness",
          "double_bond_flatness", "internal_energy"]
PROTEIN = ["protein-ligand_maximum_distance", "minimum_distance_to_protein",
           "minimum_distance_to_organic_cofactors",
           "minimum_distance_to_inorganic_cofactors",
           "minimum_distance_to_waters", "volume_overlap_with_protein",
           "volume_overlap_with_organic_cofactors",
           "volume_overlap_with_inorganic_cofactors",
           "volume_overlap_with_waters"]

POSITION = {"sched255ep_at_006h": (6.0, 20), "sched255ep_at_012h": (12.0, 41),
            "sched255ep_at_018h": (18.0, 63), "sched255ep_final": (22.6, 78)}


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p, d = k / n, 1 + z * z / n
    m = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (m - h), 100 * (m + h)


def kennzahlen(pb: pd.DataFrame, pose: pd.DataFrame):
    # NICHT durch == "true" ersetzen: bei ausgefallenen Modulen
    # steht True als 1.0 in der Datei. Siehe pb_bool.py.
    for c in LIGAND + PROTEIN:
        pb[c] = als_bool(pb[c])
    pb["pb_lig"] = pb[LIGAND].all(axis=1)
    pb["pb_prot"] = pb[LIGAND + PROTEIN].all(axis=1)
    pb["seed"] = pb["seed"].astype(int)
    pose = pose.copy()
    pose["seed"] = pose["seed"].astype(int)
    d = pb[["complex", "seed", "pb_lig", "pb_prot"]].merge(
        pose[["complex", "seed", "rmsd"]], on=["complex", "seed"], how="inner")
    d["u2"] = d["rmsd"] < 2.0
    n = len(d)
    werte = {"n": n,
             "u2": 100 * d["u2"].mean(),
             "pb_lig": 100 * d["pb_lig"].mean(),
             "pb_prot": 100 * d["pb_prot"].mean(),
             "u2_lig": 100 * (d["u2"] & d["pb_lig"]).mean(),
             "u2_prot": 100 * (d["u2"] & d["pb_prot"]).mean()}
    werte["ci"] = wilson(int((d["u2"] & d["pb_prot"]).sum()), n)
    return werte


zeilen = []
for zelle in sorted((HIER / "poses").iterdir()):
    if not zelle.is_dir() or not (zelle / "pb_redock.csv").exists():
        continue
    tag, rest = zelle.name.split("__nfe", 1)
    nfe = int(rest.split("__")[0])
    w = kennzahlen(pd.read_csv(zelle / "pb_redock.csv"),
                   pd.read_csv(zelle / "per_pose.csv"))
    h, ep = POSITION[tag]
    zeilen.append({"arm": "Minimal 72h", "h": h, "epoch": ep, "nfe": nfe, **w})

# 12-h-Referenz aus den vorhandenen Rohdaten, gleicher Code
import glob  # noqa: E402
import re  # noqa: E402

for name, pbm, posef in (("Minimal 12h", "rd_minimalcs2_seed*.csv",
                          "pose_minimalcs80.csv"),
                         ("Separate 12h", "rd_exp110cs2_seed*.csv",
                          "pose_exp110cs80.csv"),
                         ("SigmaDock 12h", "rd_sigmadockcs2_seed*.csv",
                          "pose_sigmadockcs80.csv")):
    teile = []
    for f in sorted(glob.glob(str(VERGLEICH / pbm))):
        t = pd.read_csv(f)
        t["seed"] = int(re.search(r"seed(\d+)\.csv$", f).group(1))
        b = t["file"].str.replace("\\", "/", regex=False)
        t["complex"] = b.str.rsplit("/", n=1).str[-1].str.split("__").str[0]
        teile.append(t)
    if not teile:
        continue
    w = kennzahlen(pd.concat(teile, ignore_index=True),
                   pd.read_csv(VERGLEICH / posef))
    zeilen.append({"arm": name, "h": 11.1, "epoch": 6, "nfe": 25, **w})

kopf = (f"{'Arm':<14}{'h':>6}{'Ep':>5}{'nfe':>5}{'Posen':>7}"
        f"{'<2A':>8}{'PB Lig':>8}{'PB Prot':>9}"
        f"{'<2A&Lig':>9}{'<2A&Prot':>10}{'95%-KI':>16}")
for nfe in (25, 5):
    print(f"\n=== {nfe} Integrationsschritte ===")
    print(kopf)
    for z in [z for z in zeilen if z["nfe"] == nfe]:
        lo, hi = z["ci"]
        print(f"{z['arm']:<14}{z['h']:6.1f}{z['epoch']:5d}{z['nfe']:5d}"
              f"{z['n']:7d}{z['u2']:7.2f}%{z['pb_lig']:7.2f}%"
              f"{z['pb_prot']:8.2f}%{z['u2_lig']:8.2f}%{z['u2_prot']:9.2f}%"
              f"  [{lo:5.2f},{hi:5.2f}]")

with (HIER / "tabelle_pb.csv").open("w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["arm", "walltime_h", "epoch", "nfe", "n_poses", "rmsd_lt_2A",
                "pb_valid_ligand", "pb_valid_protein", "lt2A_and_pb_ligand",
                "lt2A_and_pb_protein", "ci_lo", "ci_hi"])
    for z in zeilen:
        w.writerow([z["arm"], z["h"], z["epoch"], z["nfe"], z["n"],
                    round(z["u2"], 3), round(z["pb_lig"], 3),
                    round(z["pb_prot"], 3), round(z["u2_lig"], 3),
                    round(z["u2_prot"], 3), round(z["ci"][0], 3),
                    round(z["ci"][1], 3)])
print(f"\ngeschrieben: {HIER / 'tabelle_pb.csv'}")
