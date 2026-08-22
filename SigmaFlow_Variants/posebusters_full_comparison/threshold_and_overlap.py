#!/usr/bin/env python3
"""
Drei Auswertungen auf den bereits erzeugten Tabellen. Erzeugt selbst nichts.

VORAUSSETZUNGEN (alle im selben Verzeichnis)
    pose_<arm>.csv     complex,seed,rmsd        aus evaluate_run.py --per_pose_csv
    rd_<arm>_seed*.csv PoseBusters "redock"     aus run_redock.py
    budget_<arm>.csv   Fehlerzerlegung          aus error_budget.py

(1) KOMPLEMENTARITAET
    Welche Komplexe loest welcher Arm ueberhaupt? Der Vergleich von
    Vereinigung und bestem Einzelarm sagt, wieviel ein Ensemble holen wuerde.

(2) SCHWELLENWIRKUNG
    Die PB-Validitaet selbst haengt NICHT von der RMSD-Schwelle ab -- sie
    prueft Geometrie und kennt den RMSD nicht. Es aendern sich Trefferquote,
    gemeinsame Metrik und die BEDINGTE Validitaet P(valid | RMSD < X).
    Letztere beantwortet, ob genauere Posen auch physikalisch sauberer sind.

    VORBEHALT: die Kopplung ist teilweise eingebaut. Bei EINEM Fragment ist
    die Pose eine starre Bewegung der wahren Struktur, ihre innere Geometrie
    also per Konstruktion korrekt, und kleine RMSDs treten ueberproportional
    bei einfragmentigen Liganden auf. Ein Teil des Anstiegs ist Fragmentzahl,
    nicht Genauigkeit.

(3) ORACLE@k
    Erwartungstreu ueber zufaellige Auswahl von k der 10 Seeds, nicht nur
    "die ersten k" -- sonst haengt das Ergebnis an der Seed-Reihenfolge.

Aufruf:  python threshold_and_overlap.py
"""
from __future__ import annotations

import csv
import glob
import re
import sys

import numpy as np

ARMS = [("Minimal", "minimal"), ("Separate", "exp110"), ("SigmaDock", "sigmadock")]
THRESHOLDS = [1.0, 2.0, 2.5, 3.0]
N_SEEDS = 10


def tf(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "1.0", "yes")


def load_validity(path: str) -> dict[str, tuple[bool, bool]]:
    """complex -> (valide ohne Protein, valide mit Protein)."""
    rows = list(csv.DictReader(open(path, encoding="utf-8", errors="replace")))
    cols = list(rows[0])
    rmsd_col = next(c for c in cols if c.startswith("rmsd"))
    loaded = {"mol_pred_loaded", "mol_true_loaded", "mol_cond_loaded"}
    checks = [c for c in cols
              if c not in ({"file", "molecule", "position", rmsd_col, ""} | loaded)]
    prot = [c for c in checks
            if any(t in c for t in ("protein", "cofactor", "water", "volume", "distance"))]
    intra = [c for c in checks if c not in prot]
    out = {}
    for r in rows:
        cid = r["file"].replace("\\", "/").split("/")[-1].split("__")[0]
        out[cid] = (all(tf(r[c]) for c in intra), all(tf(r[c]) for c in checks))
    return out


def main() -> int:
    V, RM, NF = {}, {}, {}
    for lab, key in ARMS:
        files = glob.glob(f"rd_{key}_seed*.csv")
        if not files:
            print(f"FEHLT: rd_{key}_seed*.csv -- erst run_redock.py laufen lassen")
            return 2
        V[lab] = {}
        for f in files:
            seed = int(re.search(r"seed(\d+)\.csv$", f).group(1))
            for cid, t in load_validity(f).items():
                V[lab][(cid, seed)] = t
        RM[lab] = {(r["complex"], int(r["seed"])): float(r["rmsd"])
                   for r in csv.DictReader(open(f"pose_{key}.csv", encoding="utf-8"))}
        try:
            NF[lab] = {(r["complex"], int(r["seed"])): int(r["n_frag"])
                       for r in csv.DictReader(open(f"budget_{key}.csv", encoding="utf-8"))}
        except FileNotFoundError:
            NF[lab] = {}

    keys = sorted(set.intersection(*[set(V[l]) & set(RM[l]) for l, _ in ARMS]))
    cids = sorted({c for c, _ in keys})
    print(f"\n  {len(keys)} gepaarte Posen aus {len(cids)} Komplexen")

    # --- (1) Komplementaritaet ---------------------------------------------
    print("\n" + "=" * 74)
    print("  KOMPLEMENTARITAET: welche Komplexe loest welcher Arm? (>=1 von 10, <2 A)")
    print("=" * 74 + "\n")
    solved = {l: {c for c in cids
                  if any(RM[l].get((c, s), 9e9) < 2.0 for s in range(N_SEEDS))}
              for l, _ in ARMS}
    for l, _ in ARMS:
        print(f"  {l:12}{len(solved[l]):>4}  ({100*len(solved[l])/len(cids):.1f} %)")
    union = set().union(*solved.values())
    inter = set.intersection(*solved.values())
    best = max(len(s) for s in solved.values())
    print(f"\n  Vereinigung {len(union):>4}  ({100*len(union)/len(cids):.1f} %)")
    print(f"  Schnitt     {len(inter):>4}")
    print(f"  Ein perfekter Ensemble-Ranker ueber alle drei Arme kaeme auf "
          f"{100*len(union)/len(cids):.1f} %, der beste Einzelarm auf {100*best/len(cids):.1f} %.")

    # --- (2) Schwellenwirkung ----------------------------------------------
    print("\n" + "=" * 74)
    print("  SCHWELLENWIRKUNG   (die PB-Validitaet selbst haengt NICHT davon ab)")
    print("=" * 74)
    for name, idx in [("OHNE Protein", 0), ("MIT Protein", 1)]:
        print(f"\n  P(valid {name} | RMSD < X)")
        print(f"  {'Schwelle':>10}" + "".join(f"{l:>13}" for l, _ in ARMS) + f"{'n':>20}")
        print("  " + "-" * 69)
        for t in THRESHOLDS:
            row, ns = f"  {t:>8.1f} A", []
            for l, _ in ARMS:
                sel = [k for k in keys if RM[l][k] < t]
                ns.append(len(sel))
                row += (f"{100*np.mean([V[l][k][idx] for k in sel]):>12.1f}%"
                        if sel else f"{'--':>13}")
            print(row + f"{'/'.join(map(str, ns)):>20}")
        allv = [100 * np.mean([V[l][k][idx] for k in keys]) for l, _ in ARMS]
        print(f"  {'alle':>8}  " + "".join(f"{v:>12.1f}%" for v in allv)
              + f"{len(keys):>19}")

    # --- (3) Oracle@k -------------------------------------------------------
    print("\n" + "=" * 74)
    print("  ORACLE@k, erwartungstreu ueber zufaellige Seed-Auswahl")
    print("=" * 74)
    rng = np.random.default_rng(0)
    for name, ok in [("RMSD < 2 A", lambda l, c, s: RM[l].get((c, s), 9e9) < 2.0),
                     ("RMSD < 2 A UND valid mit Protein",
                      lambda l, c, s: RM[l].get((c, s), 9e9) < 2.0 and V[l][(c, s)][1])]:
        print(f"\n  {name}")
        print(f"  {'k':>4}" + "".join(f"{l:>13}" for l, _ in ARMS))
        print("  " + "-" * 44)
        M = {l: np.array([[ok(l, c, s) for s in range(N_SEEDS)] for c in cids])
             for l, _ in ARMS}
        for k in (1, 2, 3, 5, 10):
            row = f"  {k:>4}"
            for l, _ in ARMS:
                reps = [M[l][:, rng.permutation(N_SEEDS)[:k]].any(axis=1).mean()
                        for _ in range(400)]
                row += f"{100*np.mean(reps):>12.1f}%"
            print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
