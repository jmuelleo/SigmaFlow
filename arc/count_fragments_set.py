#!/usr/bin/env python3
r"""Fragmentzahl-Verteilung ueber eine EXPLIZITE Komplexliste.

WOZU
    `count_fragments.py` zaehlt ueber ein Experiment aus `conf/experiments/`,
    also ueber das, was die Datafront sieht -- bei uns die 209 entpackten
    Komplexe. Fuer die offiziellen 308 der Zeitschriftenfassung gibt es keine
    solche Config, wohl aber die vollen Daten unter `posebusters_full/` und
    die geprueften IDs in `SigmaFlow_Evaluation/reference/`.

    Dieses Skript zaehlt deshalb ueber eine uebergebene ID-Liste. Es braucht
    NUR das Liganden-SDF: die Fragmentzahl folgt aus der Konnektivitaet, nicht
    aus einer Pose. Also keine GPU, kein Sampling, kein Checkpoint.

WAS ES NICHT TUT
    Es konstruiert keinen Datensatzpfad. `--root` wird uebergeben, auf
    Existenz und Nichtleere geprueft, und die Komplexordner werden darunter
    GESUCHT statt angenommen -- die Verschachtelungstiefe von
    `posebusters_full/` ist nicht vorausgesetzt.

    Fehlende IDs werden benannt und gezaehlt, nicht stillschweigend
    uebersprungen. Eine Verteilung ueber 291 von 308 Liganden, die als "308"
    beschriftet ist, waere schlimmer als gar keine.

Aufruf (aus dem Code-Verzeichnis eines Arms, damit `sigmadock` aufloest):
    python ../arc/count_fragments_set.py \
        --root /data/stat-cadd/shug8458/data/posebusters_full \
        --ids  ../SigmaFlow_Evaluation/reference/posebusters_v2_308_ids.csv \
        --out  fragment_counts_308.csv --json-out fragment_counts_308.json
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from count_fragments import Timeout, fragment_count, quantile  # noqa: E402

# Ordnernamen der Form 5SAK_ZRY. Bewusst streng, damit Hilfsordner
# ("raw", "processed", "__MACOSX") nicht als Komplexe durchgehen.
KOMPLEX = re.compile(r"^[0-9A-Za-z]{4}_[0-9A-Za-z]{1,3}$")


def finde_komplexordner(root: Path, max_tiefe: int = 3) -> dict:
    """Sucht Komplexordner unterhalb von root, ohne die Tiefe anzunehmen."""
    gefunden = {}
    ebene = [root]
    for _ in range(max_tiefe):
        naechste = []
        for d in ebene:
            try:
                kinder = [k for k in d.iterdir() if k.is_dir()]
            except OSError:
                continue
            for k in kinder:
                if KOMPLEX.match(k.name):
                    gefunden.setdefault(k.name.upper(), k)
                else:
                    naechste.append(k)
        if gefunden or not naechste:
            break
        ebene = naechste
    return gefunden


def lies_ids(pfad: Path) -> list:
    """CSV mit pdb_id,ccd_id -- oder eine Zeile je ID."""
    text = pfad.read_text(encoding="utf-8").strip()
    if not text:
        raise SystemExit("FEHLER: {} ist leer.".format(pfad))
    erste = text.splitlines()[0]
    if "," in erste:
        rows = list(csv.DictReader(text.splitlines()))
        fehlt = {"pdb_id", "ccd_id"} - set(rows[0])
        if fehlt:
            raise SystemExit("FEHLER: {}: Spalten fehlen: {}".format(pfad, sorted(fehlt)))
        return [(r["pdb_id"].upper() + "_" + r["ccd_id"].upper()) for r in rows]
    return [z.strip().upper() for z in text.splitlines() if z.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="Verzeichnis mit den Komplexordnern")
    ap.add_argument("--ids", default=None,
                    help="CSV (pdb_id,ccd_id) oder Zeilenliste; ohne Angabe: alle gefundenen")
    ap.add_argument("--sdf-regex", default=r".*ligands\.sdf$",
                    help="wie in conf/experiments/posebusters.yaml")
    ap.add_argument("--out", default="fragment_counts_set.csv")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--label", default="set", help="Name der Menge, nur fuer die Ausgabe")
    ap.add_argument("--timeout", type=int, default=120, help="Sekunden je Molekuel")
    ap.add_argument("--allow-missing", action="store_true",
                    help="ohne dies bricht das Skript ab, wenn IDs fehlen")
    args = ap.parse_args()

    try:
        from rdkit import Chem, RDLogger
        from sigmadock.chem.fragmentation import (
            detect_torsional_bonds,
            enumerate_valid_fragmentations,
        )
    except Exception as exc:  # noqa: BLE001
        print("FEHLER: Import fehlgeschlagen ({}: {})".format(type(exc).__name__, exc),
              file=sys.stderr)
        print("  Aus dem Code-Verzeichnis eines Arms aufrufen (PYTHONPATH=<arm>/src).",
              file=sys.stderr)
        return 2
    RDLogger.DisableLog("rdApp.*")

    root = Path(args.root)
    if not root.is_dir():
        print("FEHLER: --root existiert nicht: {}".format(root), file=sys.stderr)
        return 2

    ordner = finde_komplexordner(root)
    print("root = {}".format(root))
    print("  Komplexordner gefunden: {}".format(len(ordner)))
    if not ordner:
        print("FEHLER: kein Ordner der Form XXXX_YYY unterhalb von --root.", file=sys.stderr)
        return 2
    print("  Beispiel: {}".format(ordner[sorted(ordner)[0]]))

    if args.ids:
        wunsch = lies_ids(Path(args.ids))
        print("  IDs aus {}: {}".format(args.ids, len(wunsch)))
    else:
        wunsch = sorted(ordner)
        print("  keine --ids -> alle gefundenen Ordner")

    fehlend = [i for i in wunsch if i not in ordner]
    if fehlend:
        print("  NICHT gefunden: {} von {}".format(len(fehlend), len(wunsch)))
        print("    erste 10: {}".format(fehlend[:10]))
        if not args.allow_missing:
            print("FEHLER: Abbruch. Mit --allow-missing bewusst fortsetzen.", file=sys.stderr)
            return 2
    vorhanden = [i for i in wunsch if i in ordner]

    sdf_re = re.compile(args.sdf_regex)
    rows, failures = [], collections.Counter()
    t_start = time.time()
    for i, cid in enumerate(vorhanden, 1):
        d = ordner[cid]
        rec = {"complex": cid}
        t0 = time.time()
        treffer = sorted(p for p in d.iterdir() if p.is_file() and sdf_re.match(p.name))
        if not treffer:
            failures["kein SDF passend zu --sdf-regex"] += 1
            rec["error"] = "no_sdf"
        else:
            rec["sdf"] = str(treffer[0])
            try:
                mol = Chem.MolFromMolFile(str(treffer[0]), sanitize=True, removeHs=True)
                if mol is None:
                    failures["SDF nicht ladbar"] += 1
                    rec["error"] = "parse"
                else:
                    rec["atoms"] = mol.GetNumAtoms()
                    rec["torsions"] = len(set(detect_torsional_bonds(mol, False)))
                    rec.update(fragment_count(mol, enumerate_valid_fragmentations,
                                              Chem, args.timeout))
                    rec["state_dimension"] = 6 * rec["fragments"]
            except Timeout:
                failures["Zeitlimit {}s ueberschritten".format(args.timeout)] += 1
                rec["error"] = "timeout"
            except Exception as exc:  # noqa: BLE001
                failures[type(exc).__name__] += 1
                rec["error"] = "{}: {}".format(type(exc).__name__, exc)
        rec["seconds"] = round(time.time() - t0, 2)
        rows.append(rec)
        if i % 25 == 0 or i == len(vorhanden):
            ok = sum(1 for r in rows if "fragments" in r)
            print("  {}/{}  auswertbar={}  ({:.0f}s)".format(
                i, len(vorhanden), ok, time.time() - t_start), flush=True)

    fields = ["complex", "atoms", "torsions", "min_cuts", "n_minimal_cutsets",
              "fragments", "state_dimension", "frag_varies", "seconds", "error", "sdf"]
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print("\ngeschrieben: {}".format(args.out))

    ok_rows = [r for r in rows if "fragments" in r]
    frags = sorted(r["fragments"] for r in ok_rows)
    if not frags:
        print("Keine auswertbaren Molekuele -- keine Verteilung.")
        return 1

    n = len(frags)
    hist = collections.Counter(frags)
    print()
    print("=" * 66)
    print("VERTEILUNG DER FRAGMENTZAHL '{}'  ({} auswertbar von {} angefordert)".format(
        args.label, n, len(wunsch)))
    print("=" * 66)
    print(" Fragmente |  D=6N | Liganden | Anteil | kumuliert")
    cum = 0
    for k in sorted(hist):
        cum += hist[k]
        print("    {:3d}    | {:5d} |   {:4d}   | {:5.1f}% | {:5.1f}%  {}".format(
            k, 6 * k, hist[k], 100 * hist[k] / n, 100 * cum / n, "#" * min(50, hist[k])))
    print()
    print("  Mittel {:.2f}   Median {:.1f}   q25 {}   q75 {}   q90 {}   Max {}".format(
        statistics.mean(frags), statistics.median(frags),
        quantile(frags, .25), quantile(frags, .75), quantile(frags, .90), frags[-1]))
    print("  D = 6N:  Mittel {:.1f}   Median {}   Max {}".format(
        6 * statistics.mean(frags), 6 * int(statistics.median(frags)), 6 * frags[-1]))

    varies = sum(1 for r in ok_rows if r.get("frag_varies"))
    print("  Fragmentzahl haengt vom minimalen Cut-Set ab: {} Ligand(en)".format(varies))

    if failures:
        print("\n  Nicht ausgewertet:")
        for reason, cnt in failures.most_common():
            print("    {:4d} x  {}".format(cnt, reason))
        print("  Es wird NICHTS geschaetzt.")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "label": args.label,
            "root": str(root),
            "ids_file": args.ids,
            "n_requested": len(wunsch),
            "n_found": len(vorhanden),
            "n_missing": len(fehlend),
            "missing": fehlend,
            "n_evaluated": n,
            "histogram": {str(k): v for k, v in sorted(hist.items())},
            "mean": statistics.mean(frags),
            "median": statistics.median(frags),
            "q90": quantile(frags, .90),
            "max": frags[-1],
            "failures": dict(failures),
        }, indent=2), encoding="utf-8")
        print("\ngeschrieben: {}".format(args.json_out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
