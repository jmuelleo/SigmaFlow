#!/usr/bin/env python3
"""
Misst die ECHTE Groesse der Trainings-, Validierungs- und Testdatensaetze.

WARUM
    Bisher wurde an mehreren Stellen mit 19443 gerechnet -- der Zahl aus dem
    SigmaDock-Paper (PDBBind v2020). Unsere Datafront entsteht aber aus einem
    Verzeichnis-Scan mit Regex ueber das, was auf ARC tatsaechlich liegt.
    Beides kann auseinanderlaufen, und die Differenz wandert direkt in
    steps_per_epoch, in den Scheduler-Horizont und in val_check_interval.

    Diese Zahl darf deshalb nicht geerbt, sondern muss gemessen werden.

BILLIG GENUG FUER DEN LOGIN-KNOTEN
    DataFront._setup() (datafronts.py:71-108) durchlaeuft nur Verzeichnisse und
    prueft Dateinamen gegen zwei regulaere Ausdruecke. Es werden KEINE Molekuele
    geparst, kein Torch geladen, keine GPU benutzt. Die Kosten sind reine
    Dateisystem-Latenz.

Aufruf (aus dem jeweiligen Code-Verzeichnis, damit `sigmadock` aufloest):
    python arc/probe_datafront_size.py --data-dir /data/stat-cadd/shug8458/data \
        --train pdbbind-general --val posebusters --test posebusters \
        --json-out n_train.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--train", nargs="+", default=["pdbbind-general"])
    p.add_argument("--val", nargs="+", default=["posebusters"])
    p.add_argument("--test", nargs="+", default=["posebusters"])
    p.add_argument("--json-out", default=None)
    p.add_argument("--env-out", default=None,
                   help="Schreibt 'export FINAL_N_TRAIN=...' zum Sourcen.")
    args = p.parse_args()

    try:
        from sigmadock.config import get_experiment_config
        from sigmadock.datafronts import MetaFront
    except Exception as exc:
        print(f"FEHLER: sigmadock nicht importierbar ({type(exc).__name__}: {exc})", file=sys.stderr)
        print("  Aus dem Code-Verzeichnis des jeweiligen Arms aufrufen und die", file=sys.stderr)
        print("  passende Umgebung aktivieren.", file=sys.stderr)
        return 2

    # Aus welchem Baum wurde geladen? Bei zwei parallelen sigmadock-Installationen
    # ist das die einzige Absicherung gegen eine Messung am falschen Code.
    import sigmadock
    print(f"sigmadock geladen aus: {Path(sigmadock.__file__).resolve()}")
    print(f"data_dir            : {args.data_dir}")
    print()

    result: dict[str, object] = {"data_dir": args.data_dir,
                                 "sigmadock_path": str(Path(sigmadock.__file__).resolve())}

    for split, names in (("train", args.train), ("val", args.val), ("test", args.test)):
        cfgs = [get_experiment_config(n, root_dir=args.data_dir) for n in names]
        front = MetaFront(cfgs)
        n = len(front)
        result[f"n_{split}"] = n
        result[f"{split}_exps"] = list(names)
        # Aufschluesselung je Teil-Datensatz, damit eine unerwartete Gesamtzahl
        # sofort einem Verzeichnis zuzuordnen ist.
        parts = {str(df.dataroot): len(df) for df in front.fronts}
        result[f"{split}_parts"] = parts
        print(f"{split:5s}  {'+'.join(names):30s}  n = {n:,}")
        for root, cnt in parts.items():
            print(f"         {root}: {cnt:,}")

    n_train = int(result["n_train"])  # type: ignore[arg-type]
    print()
    print(f"GEMESSEN: n_train = {n_train:,}")
    paper = 19443
    if n_train != paper:
        delta = 100.0 * (n_train - paper) / paper
        print(f"HINWEIS : Paper nennt {paper:,} -- Abweichung {delta:+.2f} %.")
        print("          Fuer alle Rechnungen gilt die GEMESSENE Zahl.")
    else:
        print(f"HINWEIS : stimmt exakt mit der Paper-Zahl {paper:,} ueberein.")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"geschrieben: {args.json_out}")
    if args.env_out:
        Path(args.env_out).write_text(f"export FINAL_N_TRAIN={n_train}\n", encoding="utf-8", newline="\n")
        print(f"geschrieben: {args.env_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
