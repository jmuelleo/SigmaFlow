"""PoseBusters, nur die fuenfzehn ligandinternen Pruefungen, lokal.

WARUM EINE EIGENE KONFIGURATION
    Keine der mitgelieferten passt. `mol` laesst das Modul `identity` weg und
    verliert damit molecular_formula, molecular_bonds, Stereochemie und
    Chiralitaet -- vier der fuenfzehn. `dock` bringt umgekehrt die
    proteinseitigen Pruefungen mit und braucht deshalb mol_cond. Genommen wird
    also `redock` ohne intermolecular_distance, volume_overlap und rmsd; das
    ergibt exakt 15 Pruefspalten plus drei Ladespalten.

REIHENFOLGE
    Die Zellen werden nach absteigender Trainingszeit abgearbeitet, 25 Schritte
    vor 5. Der aussagekraeftigste Punkt liegt damit nach der ersten Zelle vor
    und nicht nach der letzten.

EINDEUTIGER POSENSCHLUESSEL
    `(complex, seed)` reicht NICHT. In zwoelf Faellen liegen in einem einzigen
    seed_<k>-Verzeichnis zwei Dateien desselben Komplexes (`..._seed0.sdf` und
    `..._seed1.sdf`), und evaluate_run.py:239 leitet pose_id allein aus dem
    Verzeichnisnamen ab. Beide Seiten haben dann zwei Zeilen mit gleichem
    Schluessel, und ein Merge bildet daraus das Kreuzprodukt -- die Konjunktion
    "<2 A und PB-valide" paart dabei teils die falsche RMSD mit dem falschen
    PB-Ergebnis. Deshalb wird hier zusaetzlich `rep` gefuehrt: die laufende
    Nummer innerhalb einer (complex, seed)-Gruppe.

    Damit `rep` auf beiden Seiten dasselbe bedeutet, muss die Reihenfolge
    exakt die von evaluate_run sein: seed_*-Verzeichnisse sortiert, darin
    `sorted(glob("*.sdf"))`. per_pose.csv wird mit einem STABILEN Sort nach
    (complex, pose_id) geschrieben, die Gruppenreihenfolge ist dort also die
    Anhaengereihenfolge -- eben diese Glob-Reihenfolge. `datei` wird
    mitgeschrieben, damit die Zuordnung nachpruefbar bleibt statt geglaubt.

Gemessen 0,81 s je Pose einkernig.
"""
import multiprocessing as mp
import pathlib
import sys
import time

import pandas as pd
import yaml

HIER = pathlib.Path(__file__).resolve().parent
TRUE = HIER / "true308"
N_PROC = 4
CHUNK = 150
RANG = {"sched255ep_final": 0, "sched255ep_at_018h": 1,
        "sched255ep_at_012h": 2, "sched255ep_at_006h": 3}


def ligand_config():
    import posebusters
    d = pathlib.Path(posebusters.__file__).parent / "config"
    cfg = yaml.safe_load((d / "redock.yml").read_text())
    raus = {"intermolecular_distance", "volume_overlap", "rmsd"}
    cfg["modules"] = [m for m in cfg["modules"]
                      if m.get("function") not in raus]
    return cfg


def aufgaben(zelle: pathlib.Path):
    """Alle Posen einer Zelle in exakt der Reihenfolge von evaluate_run."""
    seed_dirs = sorted(d for d in zelle.rglob("seed_*") if d.is_dir())
    if not seed_dirs:
        raise SystemExit(f"keine seed_*-Verzeichnisse unter {zelle}")
    zaehler, zeilen = {}, []
    for sd in seed_dirs:
        seed = sd.name.replace("seed_", "")
        for f in sorted(sd.glob("*.sdf")):
            cid = f.name.split("__")[0]
            lig = TRUE / cid / f"{cid}_ligands.sdf"
            if not lig.exists():
                continue
            rep = zaehler.get((cid, seed), 0)
            zaehler[(cid, seed)] = rep + 1
            zeilen.append({"complex": cid, "seed": seed, "rep": rep,
                           "datei": f.name,
                           "mol_pred": str(f), "mol_true": str(lig)})
    zeilen.sort(key=lambda r: (r["complex"], int(r["seed"]), r["rep"]))
    return zeilen


def pruefe(block):
    from posebusters import PoseBusters
    df = PoseBusters(config=ligand_config()).bust_table(
        pd.DataFrame(block)[["mol_pred", "mol_true"]],
        full_report=False).reset_index(drop=True)
    df.insert(0, "datei", [r["datei"] for r in block])
    df.insert(0, "rep", [r["rep"] for r in block])
    df.insert(0, "seed", [r["seed"] for r in block])
    df.insert(0, "complex", [r["complex"] for r in block])
    return df


def main():
    zellen = [d for d in (HIER / "poses").iterdir() if d.is_dir()]
    zellen.sort(key=lambda d: (0 if "__nfe25__" in d.name else 1,
                               RANG[d.name.split("__nfe")[0]]))
    t0 = time.time()
    for zelle in zellen:
        ziel = zelle / "pb_ligand.csv"
        if ziel.exists():
            print(f"[pb] {zelle.name}: liegt vor", flush=True)
            continue
        zeilen = aufgaben(zelle)
        bloecke = [zeilen[i:i + CHUNK] for i in range(0, len(zeilen), CHUNK)]
        t = time.time()
        with mp.Pool(N_PROC) as pool:
            teile = []
            for n, df in enumerate(pool.imap_unordered(pruefe, bloecke), 1):
                teile.append(df)
                print(f"[pb] {zelle.name}: {n}/{len(bloecke)} Bloecke, "
                      f"{time.time() - t:.0f} s", flush=True)
        alles = pd.concat(teile, ignore_index=True)
        if len(alles) != len(zeilen):
            raise SystemExit(f"{zelle.name}: {len(zeilen)} Posen rein, "
                             f"{len(alles)} raus")
        if alles.duplicated(["complex", "seed", "rep"]).any():
            raise SystemExit(f"{zelle.name}: Schluessel nicht eindeutig")
        alles.to_csv(ziel, index=False, encoding="utf-8")
        print(f"[pb] FERTIG {zelle.name}: {len(zeilen)} Posen in "
              f"{(time.time() - t) / 60:.1f} min", flush=True)
    print(f"[pb] alles fertig, {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    sys.exit(main())
