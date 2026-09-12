"""PoseBusters ohne Protein: die fuenfzehn ligandinternen Pruefungen.

Berichtet je Zelle:
    <2 A            symmetriekorrigierte RMSD aus per_pose.csv (spyrmsd),
                    NICHT die Spalte rmsd_<=_2A von PoseBusters -- die Arbeit
                    berichtet durchgehend die spyrmsd-Konvention.
    PB Ligand       alle 15 Pruefungen bestanden
    <2 A & PB Lig   Konjunktion, das ligandseitige Hauptkriterium
Dazu die drei haeufigsten Einzelausfaelle, damit sichtbar ist, woran es
scheitert, statt nur dass es scheitert.
"""
import math
import pathlib

import pandas as pd

from pb_bool import als_bool

HIER = pathlib.Path(__file__).resolve().parent
LIGAND = ["sanitization", "inchi_convertible", "all_atoms_connected",
          "no_radicals", "molecular_formula", "molecular_bonds",
          "double_bond_stereochemistry", "tetrahedral_chirality",
          "bond_lengths", "bond_angles", "internal_steric_clash",
          "aromatic_ring_flatness", "non-aromatic_ring_non-flatness",
          "double_bond_flatness", "internal_energy"]
POSITION = {"sched255ep_at_006h": (6.0, 20), "sched255ep_at_012h": (12.0, 41),
            "sched255ep_at_018h": (18.0, 63), "sched255ep_final": (22.6, 78)}


def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    m = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (m - h), 100 * (m + h)


zeilen, ausfaelle = [], {}
for zelle in sorted((HIER / "poses").iterdir()):
    if not zelle.is_dir() or not (zelle / "pb_ligand.csv").exists():
        continue
    tag, rest = zelle.name.split("__nfe", 1)
    nfe = int(rest.split("__")[0])
    pb = pd.read_csv(zelle / "pb_ligand.csv")
    fehlt = [c for c in LIGAND if c not in pb.columns]
    if fehlt:
        raise SystemExit(f"{zelle.name}: fehlende Spalten {fehlt}")
    # NICHT durch == "true" ersetzen: bei ausgefallenen Modulen
    # steht True als 1.0 in der Datei. Siehe pb_bool.py.
    for c in LIGAND:
        pb[c] = als_bool(pb[c])
    pb["pb_lig"] = pb[LIGAND].all(axis=1)
    pb["seed"] = pb["seed"].astype(int)
    pose = pd.read_csv(zelle / "per_pose.csv")
    pose["seed"] = pose["seed"].astype(int)
    # rep = laufende Nummer in der (complex, seed)-Gruppe. Ohne sie ist der
    # Schluessel in zwoelf Faellen doppelt belegt und der Merge bildet das
    # Kreuzprodukt; siehe run_pb_lig.py.
    pose["rep"] = pose.groupby(["complex", "seed"]).cumcount()
    d = pb[["complex", "seed", "rep", "pb_lig"] + LIGAND].merge(
        pose[["complex", "seed", "rep", "rmsd"]],
        on=["complex", "seed", "rep"], how="inner")
    if len(d) != len(pb):
        raise SystemExit(f"{zelle.name}: {len(pb)} PB-Zeilen, {len(d)} nach "
                         f"Merge -- Schluessel stimmt nicht")
    d["u2"] = d["rmsd"] < 2.0
    n = len(d)
    lo, hi = wilson(int((d["u2"] & d["pb_lig"]).sum()), n)
    zeilen.append({"h": POSITION[tag][0], "ep": POSITION[tag][1], "nfe": nfe,
                   "n": n, "u2": 100 * d["u2"].mean(),
                   "pb": 100 * d["pb_lig"].mean(),
                   "beides": 100 * (d["u2"] & d["pb_lig"]).mean(),
                   "lo": lo, "hi": hi})
    ausfaelle[(nfe, POSITION[tag][0])] = (100 * (~d[LIGAND]).mean()
                                          ).sort_values(ascending=False).head(3)

kopf = (f"{'h':>6}{'Ep':>5}{'Posen':>7}{'<2A':>8}{'PB Lig':>9}"
        f"{'<2A&Lig':>10}{'95%-KI':>16}   haeufigste Ausfaelle")
for nfe in (25, 5):
    r = [z for z in zeilen if z["nfe"] == nfe]
    if not r:
        continue
    print(f"\n=== {nfe} Integrationsschritte ===")
    print(kopf)
    for z in r:
        a = ausfaelle[(nfe, z["h"])]
        txt = ", ".join(f"{k} {v:.1f}%" for k, v in a.items() if v > 0) or "-"
        print(f"{z['h']:6.1f}{z['ep']:5d}{z['n']:7d}{z['u2']:7.2f}%"
              f"{z['pb']:8.2f}%{z['beides']:9.2f}%"
              f"  [{z['lo']:5.2f},{z['hi']:5.2f}]   {txt}")

pd.DataFrame(zeilen).to_csv(HIER / "tabelle_pb_lig.csv", index=False)
print(f"\ngeschrieben: {HIER / 'tabelle_pb_lig.csv'}  ({len(zeilen)}/8 Zellen)")
