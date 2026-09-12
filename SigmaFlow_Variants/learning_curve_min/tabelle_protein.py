"""Lernkurve MIT Protein: alle 24 PoseBusters-Pruefungen.

QUELLE
    Die Redock-CSVs kommen von ARC (arc/posebusters_redock.slurm), eine Datei
    je Seed, entpackt nach posebusters_redock_curve/<Zelle>/.

POSENSCHLUESSEL
    Die Redock-CSVs tragen den vollen Posenpfad in der Spalte `file`. Daraus
    laesst sich `rep` exakt bilden -- die laufende Nummer innerhalb einer
    (complex, seed)-Gruppe, sortiert nach Dateiname, also genau die Reihenfolge
    von evaluate_run (`sorted(glob("*.sdf"))`). Ohne `rep` bildet der Merge mit
    per_pose.csv in zwoelf Faellen das Kreuzprodukt; siehe run_pb_lig.py.

RMSD
    Aus per_pose.csv (spyrmsd, symmetriekorrigiert), NICHT aus der Spalte
    rmsd_<=_2A von PoseBusters. Die Arbeit berichtet durchgehend spyrmsd, und
    die 12-h-Zeile ist damit gerechnet.

GEGENPROBE GEGEN DIE LOKALE RECHNUNG
    Dieselben 15 ligandinternen Pruefungen wurden lokal schon gerechnet
    (pb_ligand.csv). Beide Wege muessen je Pose dasselbe sagen. Tun sie es
    nicht, unterscheiden sich die Umgebungen, und keine der beiden Tabellen
    ist verwendbar. Deshalb laeuft die Probe hier automatisch mit.
"""
import math
import pathlib
import re

import pandas as pd

from pb_bool import als_bool

HIER = pathlib.Path(__file__).resolve().parent
ROOT = HIER / "posebusters_redock_curve"
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
    p, d = k / n, 1 + z * z / n
    m = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (m - h), 100 * (m + h)


def lade_redock(zelle: pathlib.Path) -> pd.DataFrame:
    teile = []
    for f in sorted(zelle.glob("rd_*_seed*.csv")):
        d = pd.read_csv(f)
        d["seed"] = int(re.search(r"seed(\d+)\.csv$", f.name).group(1))
        teile.append(d)
    d = pd.concat(teile, ignore_index=True)
    p = d["file"].str.replace("\\", "/", regex=False)
    d["complex"] = p.str.rsplit("/", n=1).str[-1].str.split("__").str[0]
    d["datei"] = p.str.rsplit("/", n=1).str[-1]
    # rep: nach Dateiname sortiert, damit _seed0 vor _seed1 kommt -- die
    # Reihenfolge, in der evaluate_run die Posen anhaengt.
    d = d.sort_values(["complex", "seed", "datei"], kind="stable")
    d["rep"] = d.groupby(["complex", "seed"]).cumcount()
    for c in LIGAND + PROTEIN:
        d[c] = als_bool(d[c])
    return d.reset_index(drop=True)


def gegenprobe(zelle_name: str, rd: pd.DataFrame) -> str:
    """Sagen ARC und lokal je Pose dasselbe ueber die 15 Ligandpruefungen?"""
    lokal = HIER / "poses" / zelle_name / "pb_ligand.csv"
    if not lokal.exists():
        return "keine lokale Datei"
    lo = pd.read_csv(lokal)
    for c in LIGAND:
        lo[c] = als_bool(lo[c])
    lo["seed"] = lo["seed"].astype(int)
    m = rd[["complex", "seed", "rep"] + LIGAND].merge(
        lo[["complex", "seed", "rep"] + LIGAND], on=["complex", "seed", "rep"],
        suffixes=("_arc", "_lok"))
    if len(m) != len(rd):
        return f"MERGE {len(m)} statt {len(rd)}"
    ab = {c: int((m[f"{c}_arc"] != m[f"{c}_lok"]).sum()) for c in LIGAND}
    ab = {k: v for k, v in ab.items() if v}
    return "identisch" if not ab else f"ABWEICHUNG {ab}"


zeilen = []
for zelle in sorted(ROOT.iterdir()):
    if not zelle.is_dir():
        continue
    tag, rest = zelle.name.split("__nfe", 1)
    nfe = int(rest.split("__")[0])
    rd = lade_redock(zelle)
    rd["pb_lig"] = rd[LIGAND].all(axis=1)
    rd["pb_prot"] = rd[LIGAND + PROTEIN].all(axis=1)

    pose = pd.read_csv(HIER / "poses" / zelle.name / "per_pose.csv")
    pose["seed"] = pose["seed"].astype(int)
    pose["rep"] = pose.groupby(["complex", "seed"]).cumcount()
    d = rd[["complex", "seed", "rep", "pb_lig", "pb_prot"] + PROTEIN].merge(
        pose[["complex", "seed", "rep", "rmsd"]],
        on=["complex", "seed", "rep"], how="inner")
    if len(d) != len(rd):
        raise SystemExit(f"{zelle.name}: {len(rd)} -> {len(d)} beim Merge")
    d["u2"] = d["rmsd"] < 2.0
    n = len(d)
    lo, hi = wilson(int((d["u2"] & d["pb_prot"]).sum()), n)
    h, ep = POSITION[tag]
    ausf = (100 * (~d[PROTEIN])).mean().sort_values(ascending=False)
    zeilen.append({"h": h, "ep": ep, "nfe": nfe, "n": n,
                   "u2": 100 * d["u2"].mean(),
                   "lig": 100 * d["pb_lig"].mean(),
                   "prot": 100 * d["pb_prot"].mean(),
                   "u2lig": 100 * (d["u2"] & d["pb_lig"]).mean(),
                   "u2prot": 100 * (d["u2"] & d["pb_prot"]).mean(),
                   "lo": lo, "hi": hi,
                   "top": ", ".join(f"{k} {v:.1f}%" for k, v in ausf.head(2).items()),
                   "probe": gegenprobe(zelle.name, rd)})

print("=== Gegenprobe ARC gegen lokal, 15 Ligandpruefungen je Pose ===")
for z in zeilen:
    print(f"  nfe{z['nfe']:<2d} {z['h']:5.1f} h : {z['probe']}")

kopf = (f"{'h':>6}{'Ep':>5}{'Posen':>7}{'<2A':>8}{'PB Lig':>8}{'PB Prot':>9}"
        f"{'<2A&Lig':>9}{'<2A&Prot':>10}{'95%-KI':>16}   Proteinausfaelle")
for nfe in (25, 5):
    print(f"\n=== {nfe} Integrationsschritte, MIT Protein ===")
    print(kopf)
    for z in [z for z in zeilen if z["nfe"] == nfe]:
        print(f"{z['h']:6.1f}{z['ep']:5d}{z['n']:7d}{z['u2']:7.2f}%"
              f"{z['lig']:7.2f}%{z['prot']:8.2f}%{z['u2lig']:8.2f}%"
              f"{z['u2prot']:9.2f}%  [{z['lo']:5.2f},{z['hi']:5.2f}]   {z['top']}")

pd.DataFrame(zeilen).to_csv(HIER / "tabelle_protein.csv", index=False)
print(f"\ngeschrieben: {HIER / 'tabelle_protein.csv'}")
