"""Tidy-Tabelle der Lernkurven, mit Cluster-Bootstrap ueber Komplexe.

WARUM NICHT WILSON
    Ein Wilson-Intervall behandelt die 3080 Posen einer Zelle als unabhaengig.
    Sie sind es nicht: je Komplex liegen 8-11 Posen, die denselben Liganden und
    dieselbe Tasche teilen. Das effektive n ist kleiner als 3080, das Intervall
    also zu eng. Kapitel 6 der Arbeit rechnet durchgehend einen Bootstrap ueber
    Komplexe; diese Tabelle folgt derselben Konvention.

    Gemessen: von 2 auf 10 Seeds schrumpft das Intervall nur um 22 %, weil die
    Streuung zwischen den 307 Komplexen dominiert. Mehr Seeds kaufen fuer die
    Raten fast nichts; sie zahlen sich nur bei Oracle@K aus.

VERFAHREN
    B Ziehungen mit Zuruecklegen aus den Komplexen. Je Ziehung wird die Rate
    ueber alle Posen der gezogenen Komplexe gebildet, Komplexe mit mehr Posen
    wiegen also schwerer -- genau wie in der Punktschaetzung. Fester Seed.

WELCHE ZELLEN
    Kommt aus arme.py. Ein Arm kann ueber mehrere Laeufe verteilt sein; die
    Zuordnung Zelle -> (Walltime, Epoche) haengt deshalb am Lauf, nicht am
    Snapshot-Namen.
"""
import pathlib
import re

import numpy as np
import pandas as pd

from arme import arme_vorhanden, slug, zellen
from pb_bool import als_bool

HIER = pathlib.Path(__file__).resolve().parent
B = 8000
SEED = 20260827
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
METRIKEN = ["u2", "pb_lig", "pb_prot", "u2_lig", "u2_prot"]


def zelle_laden(z: dict) -> pd.DataFrame:
    """Eine Zelle als Posen-Tabelle: complex, seed, rep, rmsd, pb_*.

    Der Schluessel ist (complex, seed, rep), nicht (complex, seed): in zwoelf
    Faellen liegen zwei Dateien desselben Komplexes in einem seed_<k>, und
    evaluate_run.py:239 bildet pose_id allein aus dem Verzeichnisnamen. `rep`
    wird auf beiden Seiten in derselben Reihenfolge gebildet -- Redock aus dem
    Dateinamen der Spalte `file`, per_pose.csv aus der Zeilenreihenfolge, die
    evaluate_run mit einem stabilen Sort erzeugt.
    """
    teile = []
    for f in sorted(z["redock_dir"].glob("rd_*_seed*.csv")):
        d = pd.read_csv(f)
        d["seed"] = int(re.search(r"seed(\d+)\.csv$", f.name).group(1))
        teile.append(d)
    if not teile:
        raise SystemExit(f"{z['arm']}/{z['zelle']}: keine rd_*_seed*.csv")
    d = pd.concat(teile, ignore_index=True)
    p = d["file"].str.replace("\\", "/", regex=False)
    d["complex"] = p.str.rsplit("/", n=1).str[-1].str.split("__").str[0]
    d["datei"] = p.str.rsplit("/", n=1).str[-1]
    d = d.sort_values(["complex", "seed", "datei"], kind="stable")
    d["rep"] = d.groupby(["complex", "seed"]).cumcount()
    for c in LIGAND + PROTEIN:
        d[c] = als_bool(d[c])
    # Das Erfolgsetikett kommt aus dem Redock, nicht aus per_pose.csv:
    # per_pose fuehrt den ROHEN RMSD des Samplers, der Redock den
    # symmetriekorrigierten. Die Korrektur kann einen RMSD nur senken, die
    # alte Fassung zaehlte also systematisch zu wenige Posen als richtig
    # (gemessen rund ein Punkt je Zelle). Siehe plot_ranking_lib._finish.
    _rc = next(c for c in d.columns if c.startswith("rmsd"))
    d["u2_redock"] = als_bool(d[_rc])
    d["pb_lig"] = d[LIGAND].all(axis=1)
    d["pb_prot"] = d[LIGAND + PROTEIN].all(axis=1)

    pose = pd.read_csv(z["per_pose"])
    pose["seed"] = pose["seed"].astype(int)
    pose["rep"] = pose.groupby(["complex", "seed"]).cumcount()
    m = d[["complex", "seed", "rep", "pb_lig", "pb_prot", "u2_redock"]].merge(
        pose[["complex", "seed", "rep", "rmsd"]],
        on=["complex", "seed", "rep"], how="inner")
    if len(m) != len(d):
        raise SystemExit(f"{z['arm']}/{z['zelle']}: {len(d)} -> {len(m)} beim Merge")
    m["u2"] = m["u2_redock"]
    m["u2_lig"] = m["u2"] & m["pb_lig"]
    m["u2_prot"] = m["u2"] & m["pb_prot"]
    return m


def bootstrap(m: pd.DataFrame, spalten, B=B, seed=SEED):
    """Cluster-Bootstrap ueber Komplexe. Gibt {spalte: (lo, hi)} in Prozent."""
    codes, _ = pd.factorize(m["complex"])
    n_cx = codes.max() + 1
    ordnung = np.argsort(codes, kind="stable")
    grenzen = np.searchsorted(codes[ordnung], np.arange(n_cx + 1))
    gruppen = [ordnung[grenzen[i]:grenzen[i + 1]] for i in range(n_cx)]
    n_je_cx = np.array([len(g) for g in gruppen])

    rng = np.random.default_rng(seed)
    out = {}
    for c in spalten:
        v = m[c].to_numpy(dtype=bool)
        s_je_cx = np.array([v[g].sum() for g in gruppen], dtype=float)
        zieh = rng.integers(0, n_cx, size=(B, n_cx))
        raten = 100 * s_je_cx[zieh].sum(axis=1) / n_je_cx[zieh].sum(axis=1)
        out[c] = (float(np.percentile(raten, 2.5)),
                  float(np.percentile(raten, 97.5)))
    return out


def main():
    alle = zellen()
    for arm in arme_vorhanden():
        zeilen = []
        for z in [z for z in alle if z["arm"] == arm]:
            m = zelle_laden(z)
            ki = bootstrap(m, METRIKEN)
            r = {"arm": arm, "lauf": z["lauf"], "snapshot": z["zelle"],
                 "walltime_h": z["walltime_h"], "epoch": z["epoch"],
                 "nfe": z["nfe"], "n_poses": len(m),
                 "n_complexes": m["complex"].nunique()}
            for k in METRIKEN:
                r[k] = 100 * m[k].mean()
                r[f"{k}_lo"], r[f"{k}_hi"] = ki[k]
            zeilen.append(r)
        df = (pd.DataFrame(zeilen)
              .sort_values(["nfe", "epoch"], ascending=[False, True]))
        ziel = HIER / f"kurve_{slug(arm)}_tidy.csv"
        df.to_csv(ziel, index=False)
        print(f"\n### {arm}   ({len(df)} Zellen)")
        print(df[["nfe", "epoch", "walltime_h", "u2", "pb_lig", "pb_prot",
                  "u2_prot", "u2_prot_lo", "u2_prot_hi"]].to_string(
                      index=False, float_format=lambda x: f"{x:7.2f}"))
        print(f"geschrieben: {ziel.name}  (B={B}, Seed={SEED})")


if __name__ == "__main__":
    main()
