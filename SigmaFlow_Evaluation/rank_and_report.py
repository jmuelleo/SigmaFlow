#!/usr/bin/env python3
"""
Ranking der gesampelten Posen nach SigmaDocks Heuristik, und der Vergleich
gegen die Obergrenze.

DIE FRAGE
    Oracle@K ist, was ein PERFEKTER Ranker aus K Ziehungen holen wuerde.
    Top-1 ist, was ein ECHTER Ranker holt. Die Differenz ist genau das, was
    ein Confidence-Modell zurueckholen koennte.

DIE DREI MODI, nachgebaut aus SigmaDocks compute_heuristic()
    vinardo    sortiere nach Vinardo-Affinitaet (kleiner ist besser)
    pb         sortiere nach dem Mittel der PoseBusters-Checks OHNE den
               RMSD-Check (groesser ist besser)
    heuristic  sortiere nach  (-Affinitaet) * (score_bias + PB_Mittel^pb_exponent)

    Der RMSD-Check MUSS aus dem PB-Mittel heraus, sonst kennt der Ranker die
    Zielgroesse und die Auswertung ist wertlos. SigmaDock macht das mit
    df.iloc[0][:-1] in compute_pb_checks().

WARUM DIE HEURISTIK NUR ALS GITTER
    score_bias und pb_exponent stehen NIRGENDS im SigmaDock-Repository. Der
    Modus ist implementiert, aber nicht parametrisiert; der ausfuehrbare
    Standardpfad ist scoring="vinardo". Statt Zahlen zu raten faehrt dieses
    Skript ein Gitter ueber beide Konstanten und zeigt, wie stark das Ergebnis
    davon abhaengt.

Aufruf:
    python rank_and_report.py --rmsd eval_exp110.csv --gnina gnina_scores.csv \\
        --pb pb_exp110_seed{}.csv --seeds 10 --label EXP-110
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def tf(v) -> bool | None:
    s = str(v).strip().lower()
    if s in ("true", "1", "1.0", "yes"):
        return True
    if s in ("false", "0", "0.0", "no"):
        return False
    return None


def load_pb(pattern: str, seeds: int) -> dict[tuple[str, int], float]:
    """(complex, seed) -> Mittel der PB-Checks ohne den RMSD-Check."""
    out: dict[tuple[str, int], float] = {}
    for s in range(seeds):
        p = Path(pattern.format(s))
        if not p.is_file():
            continue
        rows = list(csv.DictReader(open(p, encoding="utf-8", errors="replace")))
        if not rows:
            continue
        cols = list(rows[0])
        rmsd_col = next((c for c in cols if c.startswith("rmsd")), None)
        checks = [c for c in cols
                  if c not in ("file", "molecule", "position", "mol_cond_loaded", rmsd_col)]
        for r in rows:
            cid = r["file"].replace("\\", "/").split("/")[-1].split("__")[0]
            vals = [tf(r[c]) for c in checks]
            vals = [v for v in vals if v is not None]
            if vals:
                out[(cid, s)] = sum(vals) / len(vals)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rmsd", required=True, type=Path,
                    help="Per-Komplex-CSV aus evaluate_run.py (fuer die Wahrheit)")
    ap.add_argument("--per_pose_rmsd", type=Path, default=None,
                    help="optional: CSV mit complex,seed,rmsd je Pose")
    ap.add_argument("--gnina", type=Path, default=None)
    ap.add_argument("--pb", type=str, default=None,
                    help="Muster mit {} fuer den Seed, z.B. pb_exp110_seed{}.csv")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--label", default="run")
    ap.add_argument("--threshold", type=float, default=2.0)
    a = ap.parse_args()

    # --- Wahrheit je Pose ---------------------------------------------------
    per_pose: dict[str, dict[int, float]] = defaultdict(dict)
    if a.per_pose_rmsd and a.per_pose_rmsd.is_file():
        for r in csv.DictReader(open(a.per_pose_rmsd, encoding="utf-8")):
            per_pose[r["complex"]][int(r["seed"])] = float(r["rmsd"])
    else:
        print("HINWEIS: keine Per-Pose-RMSDs uebergeben. Ohne sie laesst sich")
        print("         Top-1 eines Rankers NICHT berechnen, nur Oracle.")
        print("         evaluate_run.py mit --per_pose_csv erzeugen.")

    # --- Bestandteile des Rankers -------------------------------------------
    aff: dict[tuple[str, int], float] = {}
    if a.gnina and a.gnina.is_file():
        for r in csv.DictReader(open(a.gnina, encoding="utf-8")):
            aff[(r["complex"], int(r["seed"]))] = float(r["affinity"])
        print(f"Vinardo-Affinitaeten: {len(aff)} Posen")
    pb = load_pb(a.pb, a.seeds) if a.pb else {}
    if pb:
        print(f"PB-Mittel:            {len(pb)} Posen")
    print()

    if not per_pose:
        return 0

    complexes = sorted(per_pose)
    n = len(complexes)

    def rate(chooser) -> float:
        hit = 0
        for c in complexes:
            seeds = sorted(per_pose[c])
            if not seeds:
                continue
            pick = chooser(c, seeds)
            if pick is not None and per_pose[c][pick] < a.threshold:
                hit += 1
        return 100.0 * hit / n

    def oracle(c, seeds):
        return min(seeds, key=lambda s: per_pose[c][s])

    def first(c, seeds):
        return seeds[0]

    def by_vinardo(c, seeds):
        cand = [s for s in seeds if (c, s) in aff]
        return min(cand, key=lambda s: aff[(c, s)]) if cand else None

    def by_pb(c, seeds):
        cand = [s for s in seeds if (c, s) in pb]
        return max(cand, key=lambda s: pb[(c, s)]) if cand else None

    def by_heuristic(bias, expo):
        def f(c, seeds):
            cand = [s for s in seeds if (c, s) in aff and (c, s) in pb]
            if not cand:
                return None
            return max(cand, key=lambda s: (-aff[(c, s)]) * (bias + pb[(c, s)] ** expo))
        return f

    print(f"=== {a.label}: {n} Komplexe, bis zu {a.seeds} Seeds, "
          f"Schwelle {a.threshold} A ===")
    print()
    print(f"  {'Auswahl':38}{'Erfolg':>9}")
    print("  " + "-" * 47)
    print(f"  {'Oracle (perfekter Ranker)':38}{rate(oracle):8.1f}%")
    print(f"  {'erster Seed (kein Ranker)':38}{rate(first):8.1f}%")
    if aff:
        print(f"  {'Vinardo-Affinitaet':38}{rate(by_vinardo):8.1f}%")
    if pb:
        print(f"  {'PB-Mittel':38}{rate(by_pb):8.1f}%")
    print()

    if aff and pb:
        print("  === gemischte Heuristik, Gitter ueber die unbekannten Konstanten ===")
        print(f"  {'':10}" + "".join(f"{'expo=' + str(e):>10}" for e in (0.5, 1, 2, 4)))
        for bias in (0.0, 0.25, 0.5, 1.0):
            row = f"  bias={bias:<5}"
            for expo in (0.5, 1, 2, 4):
                row += f"{rate(by_heuristic(bias, expo)):9.1f}%"
            print(row)
        print()
        print("  score_bias und pb_exponent stehen nicht im SigmaDock-Repo.")
        print("  Die Spannweite dieser Tabelle ist die Unsicherheit, die daraus folgt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
