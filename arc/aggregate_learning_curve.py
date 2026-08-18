#!/usr/bin/env python3
"""
Fuehrt die Snapshot-Auswertungen beider Arme zu EINER Lernkurve zusammen.

Beantwortet: performance = f(Trainingszeit), fuer SigmaFlow und SigmaDock
nebeneinander, auf demselben festen Auswertungssubset.

EINGABEN (beliebig kombinierbar)
    --curve-csv <arm> <pfad>     die von arc/eval_snapshots.slurm erzeugte
                                 curve_<model>.csv
    --report-json <arm> <pfad>   ein snapshot_report.json aus
                                 arc/evaluate_snapshot.py

DIE TRENNUNG, DIE DIESES SKRIPT ERZWINGT
    Ein 12h-SNAPSHOT aus dem 72h-Lauf und der alte, eigenstaendige 12h-LAUF
    sind verschiedene Objekte: der eine sitzt mitten in einem langen Schedule
    auf hoher Lernrate, der andere ist vollstaendig ausannealt. In eine
    gemeinsame Kurve gehoeren sie nicht.

    Deshalb traegt jede Zeile eine `series`-Spalte, und Zeilen mit
    unterschiedlicher `series` werden nie zu einer Linie verbunden.
    `--annealed-endpoint` markiert eine Eingabe ausdruecklich als Endpunkt.

ES WIRD NICHTS ERFUNDEN
    Fehlende Metriken bleiben leer. Eine interpolierte Zahl in einer
    Lernkurve waere schaedlicher als eine Luecke.

Aufruf:
    python arc/aggregate_learning_curve.py \
        --curve-csv sigmaflow_minimal <run>/learning_curve/curve_sigmaflow_minimal.csv \
        --curve-csv sigmadock         <run>/learning_curve/curve_sigmadock.csv \
        --out-csv learning_curve.csv --out-md learning_curve.md
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

SERIES_TRAJECTORY = "trajectory_snapshot_of_72h"
SERIES_ANNEALED = "annealed_endpoint"

# Spalten der Ausgabetabelle. Bewusst fest: eine Kurve, deren Spalten je nach
# Eingabe wandern, laesst sich nicht ueber Sitzungen hinweg vergleichen.
FIELDS = [
    "series", "arm", "snapshot", "walltime_h", "epoch", "global_step",
    "examples_seen", "schedule_progress", "lr", "n_poses",
    "rmsd_median", "success_2A", "success_5A",
    "pb_valid_frac", "rot_error_median_deg", "trans_error_median_A",
    "source_json",
]


def hours_from_tag(tag: str) -> str:
    """'sched070ep_at_012h' -> '12'. Nur lesen, nicht raten."""
    m = re.search(r"_at_(\d+)h", tag)
    if m:
        return str(int(m.group(1)))
    m = re.fullmatch(r"(\d+)h", tag)
    return str(int(m.group(1))) if m else ""


def rows_from_curve_csv(arm: str, path: Path, series: str) -> list[dict]:
    out = []
    with open(path, newline="", encoding="utf-8") as fh:
        for rec in csv.DictReader(fh):
            snap = (rec.get("snapshot") or "").strip()
            row = {k: "" for k in FIELDS}
            row.update({
                "series": series,
                "arm": arm,
                "snapshot": snap,
                "walltime_h": (rec.get("walltime_h") or "").strip() or hours_from_tag(snap),
                "epoch": (rec.get("epoch") or "").strip(),
                "global_step": (rec.get("global_step") or "").strip(),
                "examples_seen": (rec.get("examples_seen") or "").strip(),
                "lr": (rec.get("lr") or "").strip(),
                "n_poses": (rec.get("n_poses") or "").strip(),
                "rmsd_median": (rec.get("rmsd_median") or "").strip(),
                "success_2A": (rec.get("success_2A") or "").strip(),
                "success_5A": (rec.get("success_5A") or "").strip(),
                "source_json": str(path),
            })
            out.append(row)
    return out


def rows_from_report_json(arm: str, path: Path, series_override: str | None) -> list[dict]:
    rep = json.loads(path.read_text(encoding="utf-8"))
    tp = rep.get("training_position") or {}
    kind = rep.get("snapshot_kind")
    series = series_override or (SERIES_ANNEALED if kind == SERIES_ANNEALED else SERIES_TRAJECTORY)

    row = {k: "" for k in FIELDS}
    snap = (rep.get("checkpoint") or {}).get("name", "")
    row.update({
        "series": series,
        "arm": rep.get("arm", arm),
        "snapshot": snap,
        "walltime_h": str(tp.get("walltime_h") or "") or hours_from_tag(snap),
        "epoch": str(tp.get("epoch") or ""),
        "global_step": str(tp.get("global_step") or ""),
        "examples_seen": str(tp.get("examples_seen") or ""),
        "schedule_progress": str(tp.get("schedule_progress") or ""),
        "lr": str(tp.get("lr") or ""),
        "source_json": str(path),
    })

    m = rep.get("metrics") or {}
    basics = m.get("basics") or {}
    single = basics.get("single_sample") or {}
    if basics:
        row["n_poses"] = str(basics.get("n_poses", ""))
        row["rmsd_median"] = str(basics.get("rmsd_median_all", ""))
        row["success_2A"] = str(single.get("success_below_2.0A", ""))
        row["success_5A"] = str(single.get("success_below_5.0A", ""))
    for src, dst in (("posebusters_valid_frac", "pb_valid_frac"),
                     ("rotation_error_median_deg", "rot_error_median_deg"),
                     ("translation_error_median_A", "trans_error_median_A")):
        if src in m:
            row[dst] = str(m[src])
    return [row]


def sort_key(row: dict) -> tuple:
    try:
        h = float(row["walltime_h"])
    except (TypeError, ValueError):
        h = float("inf")
    return (row["series"], h, row["arm"])


def to_markdown(rows: list[dict]) -> str:
    """Eine Tabelle je series -- sie duerfen nicht vermischt werden."""
    chunks = []
    for series in sorted({r["series"] for r in rows}):
        sub = [r for r in rows if r["series"] == series]
        arms = sorted({r["arm"] for r in sub})
        hours = sorted({r["walltime_h"] for r in sub if r["walltime_h"]},
                       key=lambda x: float(x))

        title = ("Zwischenstaende EINES 72h-Schedules" if series == SERIES_TRAJECTORY
                 else "Eigenstaendig ausannealte Endpunkte")
        chunks.append(f"### {series}\n\n*{title}*\n")

        cols = ["RMSD<2A", "RMSD median", "PB valid", "Rot-Fehler"]
        keys = ["success_2A", "rmsd_median", "pb_valid_frac", "rot_error_median_deg"]
        header = "| Trainingszeit |" + "".join(f" {a} {c} |" for c in cols for a in arms)
        sep = "|---|" + "---|" * (len(cols) * len(arms))
        chunks.append(header)
        chunks.append(sep)

        for h in hours:
            cells = []
            for key in keys:
                for arm in arms:
                    hit = [r for r in sub if r["arm"] == arm and r["walltime_h"] == h]
                    cells.append(hit[0].get(key, "") if hit else "")
            chunks.append(f"| {h} h |" + "".join(f" {c or '—'} |" for c in cells))
        chunks.append("")

    if any(r["series"] == SERIES_ANNEALED for r in rows) and \
       any(r["series"] == SERIES_TRAJECTORY for r in rows):
        chunks.append("> **Nicht zusammenfuehren.** Die beiden Tabellen beantworten")
        chunks.append("> verschiedene Fragen: ein Zwischenstand sitzt auf hoher Lernrate")
        chunks.append("> mitten im Schedule, ein Endpunkt ist vollstaendig ausannealt.")
    return "\n".join(chunks)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--curve-csv", nargs=2, action="append", metavar=("ARM", "PATH"), default=[])
    p.add_argument("--report-json", nargs=2, action="append", metavar=("ARM", "PATH"), default=[])
    p.add_argument("--annealed-endpoint", action="store_true",
                   help="Markiert ALLE Eingaben dieses Aufrufs als ausannealte Endpunkte.")
    p.add_argument("--out-csv", type=Path, default=None)
    p.add_argument("--out-md", type=Path, default=None)
    args = p.parse_args()

    series = SERIES_ANNEALED if args.annealed_endpoint else SERIES_TRAJECTORY
    rows: list[dict] = []
    for arm, path in args.curve_csv:
        pth = Path(path)
        if not pth.exists():
            print(f"WARNUNG: uebersprungen, fehlt: {pth}")
            continue
        rows += rows_from_curve_csv(arm, pth, series)
    for arm, path in args.report_json:
        pth = Path(path)
        if not pth.exists():
            print(f"WARNUNG: uebersprungen, fehlt: {pth}")
            continue
        rows += rows_from_report_json(arm, pth, series if args.annealed_endpoint else None)

    if not rows:
        print("Keine Eingaben gefunden. Nichts geschrieben.")
        return 1

    rows.sort(key=sort_key)

    print()
    print("=" * 70)
    print(f"LERNKURVE: {len(rows)} Zeilen, "
          f"{len({r['arm'] for r in rows})} Arm(e), "
          f"{len({r['series'] for r in rows})} Serie(n)")
    print("=" * 70)
    missing = sum(1 for r in rows if not r["success_2A"])
    if missing:
        print(f"  {missing} Zeile(n) ohne Metriken -- bleiben leer, werden nicht geschaetzt.")
    print()
    print(to_markdown(rows))

    if args.out_csv:
        with open(args.out_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"\ngeschrieben: {args.out_csv}")
    if args.out_md:
        args.out_md.write_text(to_markdown(rows), encoding="utf-8")
        print(f"geschrieben: {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
