"""PoseBusters (redock) ueber alle acht Zellen der Minimal-Lernkurve.

WARUM PARALLEL
    Gemessen 0,97 s je Pose einkernig, bei 24 640 Posen also 6,6 Stunden.
    Die Pruefungen sind zwischen Posen unabhaengig, deshalb ueber Prozesse.

WARUM NACH KOMPLEX SORTIERT
    PoseBusters liest `mol_cond` je Zeile neu ein. Liegen die zehn Seeds
    desselben Komplexes beieinander, trifft ein Arbeitsprozess dieselbe
    PDB mehrfach hintereinander an, was dem Dateisystem-Cache entgegenkommt.

AUSGABE
    Je Zelle eine `pb_redock.csv` mit Komplex, Seed und den 24 Pruefspalten.
"""
import multiprocessing as mp
import sys
import time
from pathlib import Path

import pandas as pd

HIER = Path(__file__).resolve().parent
TRUE = HIER / "true308"
N_PROC = 8
CHUNK = 150


def aufgaben(zelle: Path):
    """Eine Zeile je Pose, nach Komplex und Seed sortiert."""
    zeilen = []
    for f in zelle.rglob("*.sdf"):
        cid = f.name.split("__")[0]
        lig = TRUE / cid / f"{cid}_ligands.sdf"
        prot = TRUE / cid / f"{cid}_protein.pdb"
        if not (lig.exists() and prot.exists()):
            continue
        seed = f.parent.name.replace("seed_", "")
        zeilen.append({"complex": cid, "seed": seed, "mol_pred": str(f),
                       "mol_true": str(lig), "mol_cond": str(prot)})
    zeilen.sort(key=lambda r: (r["complex"], int(r["seed"])))
    return zeilen


def pruefe(block):
    """Ein Block Zeilen in einem Arbeitsprozess. PoseBusters wird hier und
    nicht global erzeugt, weil das Objekt nicht ueber Prozessgrenzen soll."""
    from posebusters import PoseBusters
    df = PoseBusters(config="redock").bust_table(
        pd.DataFrame(block)[["mol_pred", "mol_true", "mol_cond"]],
        full_report=False).reset_index(drop=True)
    df.insert(0, "seed", [r["seed"] for r in block])
    df.insert(0, "complex", [r["complex"] for r in block])
    return df


def main():
    zellen = sorted(d for d in (HIER / "poses").iterdir() if d.is_dir())
    t0 = time.time()
    for zelle in zellen:
        ziel = zelle / "pb_redock.csv"
        if ziel.exists():
            print(f"[pb] {zelle.name}: liegt schon vor, uebersprungen")
            continue
        zeilen = aufgaben(zelle)
        bloecke = [zeilen[i:i + CHUNK] for i in range(0, len(zeilen), CHUNK)]
        t = time.time()
        with mp.Pool(N_PROC) as pool:
            teile = []
            for n, df in enumerate(pool.imap_unordered(pruefe, bloecke), 1):
                teile.append(df)
                if n % 5 == 0 or n == len(bloecke):
                    print(f"[pb] {zelle.name}: {n}/{len(bloecke)} Bloecke, "
                          f"{time.time() - t:.0f} s", flush=True)
        pd.concat(teile, ignore_index=True).to_csv(ziel, index=False,
                                                   encoding="utf-8")
        print(f"[pb] {zelle.name}: {len(zeilen)} Posen in "
              f"{time.time() - t:.0f} s -> {ziel.name}", flush=True)
    print(f"[pb] fertig, gesamt {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    sys.exit(main())
