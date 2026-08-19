#!/usr/bin/env python3
"""
Fragmentzahl je Ligand plus Verteilung ueber den Datensatz.

WOZU
    Die Zustandsdimension der Generierung ist D = 6 * F (drei Rotations- und
    drei Translationsfreiheitsgrade je Fragment). F bestimmt damit die Kosten
    der exakten ODE-Likelihood, die Auswahl der hochfragmentierten Faelle fuer
    die PyMOL-Abbildungen und die Frage, wie stark eine fragmentweise
    Quellverteilung ueberhaupt wirken kann.

    Bisher existierten nur drei Ordnungsstatistiken (Median D 24, q90 42,
    Max 66). Dieses Skript erzeugt die vollstaendige Liste und die Verteilung.

ZWEI DINGE, DIE ES NEBENBEI PRUEFT

    1. VERZEICHNISSE GEGEN PAARE
       DataFront._setup() zaehlt `num_visible` in derselben Schleife hoch, in
       der es auch anhaengt -- die Warnung "N directories found, but only M
       valid pairs" kann deshalb NIE ausloesen (datafronts.py:102-107).
       Ein Ordner ohne passende Dateien faellt vollstaendig lautlos heraus.
       Hier wird beides getrennt gezaehlt und die Differenz benannt.

    2. DETERMINISMUS DER FRAGMENTZAHL
       Die Trainingsstrategie ist `fragmentation_strategy="random"`, zieht aber
       aus den MINIMALEN Cut-Sets. Die Zahl der Fragmente ist damit pro Ligand
       fest, auch wenn variiert, WELCHE Bindungen geschnitten werden. Das wird
       je Molekuel nachgeprueft und in der Spalte `frag_varies` ausgewiesen.

LAUFZEIT
    enumerate_valid_fragmentations() ist kombinatorisch. Molekuele mit vielen
    Torsionen koennen Minuten kosten; ein lokaler Test blieb bei 23 Torsionen
    ueber 20 Minuten haengen. Deshalb ein HARTES Zeitlimit je Molekuel
    (--timeout, Default 120 s). Zeitueberschreitungen werden als solche
    protokolliert und NICHT geschaetzt -- eine erfundene Zahl in einer
    Verteilung waere schaedlicher als eine Luecke.

Aufruf (aus dem Code-Verzeichnis eines Arms, damit `sigmadock` aufloest):
    python ../arc/count_fragments.py --data-dir /data/stat-cadd/shug8458/data \
        --experiment posebusters --out counts_posebusters.csv
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import signal
import statistics
import sys
import time
from pathlib import Path


class Timeout(Exception):
    pass


def _alarm(signum, frame):  # noqa: ANN001, ARG001
    raise Timeout()


def fragment_count(mol, enumerate_fn, chem, timeout_s: int) -> dict:
    """Minimale Cut-Sets -> Fragmentzahl. Wirft Timeout bei Ueberschreitung."""
    have_alarm = hasattr(signal, "SIGALRM")
    if have_alarm and timeout_s > 0:
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(timeout_s)
    try:
        cutsets = enumerate_fn(mol, ignore_conjugated=False)
        kmin = min(len(c) for c in cutsets)
        minimal = [c for c in cutsets if len(c) == kmin]

        sizes = set()
        for cs in minimal:
            em = chem.RWMol(mol)
            for b in cs:
                bd = mol.GetBondWithIdx(b)
                em.RemoveBond(bd.GetBeginAtomIdx(), bd.GetEndAtomIdx())
            sizes.add(len(chem.GetMolFrags(em.GetMol())))
        return {
            "min_cuts": kmin,
            "n_minimal_cutsets": len(minimal),
            "fragments": min(sizes),
            "frag_varies": len(sizes) > 1,
        }
    finally:
        if have_alarm and timeout_s > 0:
            signal.alarm(0)


def quantile(sorted_vals: list[int], p: float) -> int:
    if not sorted_vals:
        return 0
    idx = min(len(sorted_vals) - 1, int(math.ceil(p * len(sorted_vals))) - 1)
    return sorted_vals[max(0, idx)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--experiment", default="posebusters",
                    help="Name aus conf/experiments/ -- Pfad und Regexe kommen von dort")
    ap.add_argument("--out", default="fragment_counts.csv")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--timeout", type=int, default=120, help="Sekunden je Molekuel")
    ap.add_argument("--max-complexes", type=int, default=0, help="0 = alle")
    args = ap.parse_args()

    try:
        from rdkit import Chem, RDLogger
        from sigmadock.chem.fragmentation import (
            detect_torsional_bonds,
            enumerate_valid_fragmentations,
        )
        from sigmadock.config import get_experiment_config
        from sigmadock.datafronts import DataFront
    except Exception as exc:  # noqa: BLE001
        print(f"FEHLER: Import fehlgeschlagen ({type(exc).__name__}: {exc})", file=sys.stderr)
        print("  Aus dem Code-Verzeichnis eines Arms aufrufen, passende Umgebung aktivieren.",
              file=sys.stderr)
        return 2
    RDLogger.DisableLog("rdApp.*")

    # Pfad UND Regexe aus der Config. Nicht selbst konstruieren -- genau daran
    # sind in dieser Sitzung bereits vier Skripte gescheitert.
    ec = get_experiment_config(args.experiment, root_dir=Path(args.data_dir))
    print(f"Experiment '{args.experiment}' -> {ec.dataset}")
    print(f"  pdb_regex = {ec.pdb_regex}")
    print(f"  sdf_regex = {ec.sdf_regex}")

    # --- Verzeichnisse gegen Paare -------------------------------------------
    root = Path(ec.dataset)
    subdirs = sorted(d for d in root.iterdir() if d.is_dir())
    front = DataFront(ec.dataset, pdb_regex=ec.pdb_regex, sdf_regex=ec.sdf_regex,
                      ref_sdf_regex=getattr(ec, "ref_sdf_regex", None))
    n_pairs = len(front)

    print()
    print(f"  Unterverzeichnisse im Datensatz : {len(subdirs)}")
    print(f"  Von der Datafront erkannte Paare: {n_pairs}")
    if len(subdirs) != n_pairs:
        print(f"  ACHTUNG: {len(subdirs) - n_pairs} Verzeichnis(se) liefern KEIN Paar.")
        print("           DataFront verwirft sie lautlos -- die Warnung in")
        print("           datafronts.py:105 kann nie ausloesen, weil num_visible")
        print("           in derselben Schleife hochgezaehlt wird wie pairs.")
        matched = {Path(p[0]).parent.name for p in front.pairs}
        missing = [d.name for d in subdirs if d.name not in matched]
        print(f"           Betroffen (erste 10): {missing[:10]}")
    else:
        print("  -> jedes Verzeichnis liefert genau ein Paar.")
    print()

    # --- Fragmente zaehlen ----------------------------------------------------
    rows, failures = [], collections.Counter()
    pairs = list(front.pairs)
    if args.max_complexes:
        pairs = pairs[: args.max_complexes]

    t_start = time.time()
    for i, (sdf_path, _pdb_path, _ref) in enumerate(pairs, 1):
        cid = Path(sdf_path).parent.name
        rec = {"complex": cid, "sdf": str(sdf_path)}
        t0 = time.time()
        try:
            mol = Chem.MolFromMolFile(str(sdf_path), sanitize=True, removeHs=True)
            if mol is None:
                failures["SDF nicht ladbar"] += 1
                rec["error"] = "parse"
            else:
                rec["atoms"] = mol.GetNumAtoms()
                rec["torsions"] = len(set(detect_torsional_bonds(mol, False)))
                rec.update(fragment_count(mol, enumerate_valid_fragmentations, Chem, args.timeout))
                rec["state_dimension"] = 6 * rec["fragments"]
        except Timeout:
            failures[f"Zeitlimit {args.timeout}s ueberschritten"] += 1
            rec["error"] = "timeout"
        except Exception as exc:  # noqa: BLE001
            failures[f"{type(exc).__name__}"] += 1
            rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["seconds"] = round(time.time() - t0, 2)
        rows.append(rec)
        if i % 25 == 0 or i == len(pairs):
            ok = sum(1 for r in rows if "fragments" in r)
            print(f"  {i}/{len(pairs)}  auswertbar={ok}  "
                  f"({time.time() - t_start:.0f}s)", flush=True)

    # --- Ausgabe --------------------------------------------------------------
    fields = ["complex", "atoms", "torsions", "min_cuts", "n_minimal_cutsets",
              "fragments", "state_dimension", "frag_varies", "seconds", "error"]
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\ngeschrieben: {args.out}")

    ok_rows = [r for r in rows if "fragments" in r]
    frags = sorted(r["fragments"] for r in ok_rows)
    if not frags:
        print("Keine auswertbaren Molekuele -- keine Verteilung.")
        return 1

    n = len(frags)
    hist = collections.Counter(frags)
    print()
    print("=" * 62)
    print(f"VERTEILUNG DER FRAGMENTZAHL  ({n} von {len(pairs)} Liganden)")
    print("=" * 62)
    print(" Fragmente | Liganden | Anteil | kumuliert")
    cum = 0
    for k in sorted(hist):
        cum += hist[k]
        print(f"    {k:3d}    |   {hist[k]:4d}   | {100 * hist[k] / n:5.1f}% | "
              f"{100 * cum / n:5.1f}%  {'#' * min(60, hist[k])}")

    print()
    print(f"  Mittel {statistics.mean(frags):.2f}   Median {statistics.median(frags):.1f}   "
          f"q25 {quantile(frags, .25)}   q75 {quantile(frags, .75)}   "
          f"q90 {quantile(frags, .90)}   q95 {quantile(frags, .95)}")
    print(f"  Min {frags[0]}   Max {frags[-1]}")
    print(f"  Zustandsdimension D = 6F:  Median {6 * int(statistics.median(frags))}   "
          f"q90 {6 * quantile(frags, .90)}   Max {6 * frags[-1]}")

    varies = sum(1 for r in ok_rows if r.get("frag_varies"))
    print()
    print(f"  Fragmentzahl haengt vom gewaehlten minimalen Cut-Set ab: {varies} Ligand(en)")
    if varies == 0:
        print("  -> die Zahl ist trotz fragmentation_strategy='random' deterministisch.")

    if failures:
        print()
        print("  Nicht ausgewertet:")
        for reason, cnt in failures.most_common():
            print(f"    {cnt:4d} x  {reason}")
        print("  Diese Liganden bleiben in der CSV mit leerer Fragmentspalte stehen.")
        print("  Es wird NICHTS geschaetzt.")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "experiment": args.experiment,
            "dataset": str(ec.dataset),
            "n_subdirs": len(subdirs),
            "n_pairs": n_pairs,
            "n_evaluated": n,
            "histogram": {str(k): v for k, v in sorted(hist.items())},
            "mean": statistics.mean(frags),
            "median": statistics.median(frags),
            "q90": quantile(frags, .90),
            "max": frags[-1],
            "failures": dict(failures),
        }, indent=2), encoding="utf-8")
        print(f"\ngeschrieben: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
