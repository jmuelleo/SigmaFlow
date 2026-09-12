#!/usr/bin/env python3
"""
Test der Lernkurven-Aggregation gegen FIXTURES.

Die Zahlen hier sind offensichtlich synthetisch (0.11, 0.22, ...) und dienen
ausschliesslich dazu, die Mechanik zu pruefen. Sie duerfen NIE in einen
Ergebnisbericht wandern -- deshalb tragen alle Fixture-Dateien 'FIXTURE' im
Namen und werden in einem temporaeren Verzeichnis erzeugt, nicht im Repo.

Geprueft wird:
  1. Zeilen aus curve_*.csv werden korrekt gelesen
  2. snapshot_report.json wird korrekt gelesen
  3. Zwischenstaende und ausannealte Endpunkte landen in GETRENNTEN Serien
  4. die Stundenzahl wird aus dem Snapshot-Tag gelesen
  5. fehlende Metriken bleiben leer und werden nicht interpoliert
  6. die Sortierung ist nach Serie, dann Zeit, dann Arm

Aufruf: python arc/test_aggregate_learning_curve.py
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(cond), detail))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))


def write_curve_fixture(path: Path, rows: list[dict]) -> None:
    cols = ["snapshot", "walltime_h", "epoch", "global_step", "examples_seen",
            "lr", "n_poses", "rmsd_median", "success_2A", "success_5A"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="lc_fixture_"))
    print(f"\nFixture-Verzeichnis: {tmp}\n")

    # --- Fixture 1+2: Kurven beider Arme, ein Snapshot ohne Metriken --------
    write_curve_fixture(tmp / "FIXTURE_curve_sigmaflow_minimal.csv", [
        {"snapshot": "sched070ep_at_006h", "walltime_h": "6.0", "epoch": "2",
         "global_step": "3600", "examples_seen": "115200", "lr": "9.1e-05",
         "n_poses": "50", "rmsd_median": "5.10", "success_2A": "0.11", "success_5A": "0.44"},
        {"snapshot": "sched070ep_at_012h", "walltime_h": "12.0", "epoch": "5",
         "global_step": "7300", "examples_seen": "233600", "lr": "8.7e-05",
         "n_poses": "50", "rmsd_median": "4.40", "success_2A": "0.22", "success_5A": "0.55"},
        # Snapshot, dessen Sampling fehlgeschlagen ist -> Metriken leer
        {"snapshot": "sched070ep_at_018h", "walltime_h": "18.0", "epoch": "8",
         "global_step": "11000", "examples_seen": "352000", "lr": "8.2e-05",
         "n_poses": "", "rmsd_median": "", "success_2A": "", "success_5A": ""},
    ])
    write_curve_fixture(tmp / "FIXTURE_curve_sigmadock.csv", [
        {"snapshot": "sched070ep_at_006h", "walltime_h": "6.0", "epoch": "2",
         "global_step": "3550", "examples_seen": "113600", "lr": "9.1e-05",
         "n_poses": "50", "rmsd_median": "4.90", "success_2A": "0.13", "success_5A": "0.46"},
        {"snapshot": "sched070ep_at_012h", "walltime_h": "12.0", "epoch": "5",
         "global_step": "7200", "examples_seen": "230400", "lr": "8.7e-05",
         "n_poses": "50", "rmsd_median": "4.20", "success_2A": "0.25", "success_5A": "0.58"},
    ])

    # --- Fixture 3: ein ausannealter Endpunkt (der alte 12h-Lauf) -----------
    (tmp / "FIXTURE_report_annealed12h.json").write_text(json.dumps({
        "arm": "sigmaflow_minimal",
        "checkpoint": {"name": "annealed_endpoint_12h.ckpt", "sha256": "deadbeef"},
        "snapshot_kind": "annealed_endpoint",
        "training_position": {"walltime_h": "12.0", "epoch": "5", "global_step": "13750",
                              "examples_seen": "110000", "schedule_progress": "1.0"},
        "metrics": {"basics": {"n_poses": 50, "rmsd_median_all": 5.43,
                               "single_sample": {"success_below_2.0A": 0.029,
                                                 "success_below_5.0A": 0.488}}},
    }), encoding="utf-8")

    out_csv = tmp / "out.csv"

    # --- Lauf 1: die Zwischenstaende ----------------------------------------
    rc1 = subprocess.call([
        sys.executable, str(HERE / "aggregate_learning_curve.py"),
        "--curve-csv", "sigmaflow_minimal", str(tmp / "FIXTURE_curve_sigmaflow_minimal.csv"),
        "--curve-csv", "sigmadock", str(tmp / "FIXTURE_curve_sigmadock.csv"),
        "--out-csv", str(out_csv),
    ], stdout=subprocess.DEVNULL)
    check("Aggregation der Kurven laeuft durch", rc1 == 0, f"rc={rc1}")

    rows = list(csv.DictReader(open(out_csv, encoding="utf-8")))
    check("5 Zeilen aus zwei Kurven gelesen", len(rows) == 5, f"{len(rows)}")
    check("beide Arme vertreten",
          {r["arm"] for r in rows} == {"sigmaflow_minimal", "sigmadock"},
          str(sorted({r["arm"] for r in rows})))
    check("alle Zeilen in der Trajektorien-Serie",
          all(r["series"] == "trajectory_snapshot_of_72h" for r in rows))
    empty = [r for r in rows if not r["success_2A"]]
    check("fehlende Metrik bleibt leer, wird nicht interpoliert",
          len(empty) == 1 and empty[0]["walltime_h"] == "18.0",
          f"{len(empty)} leere Zeile(n)")
    hours = [r["walltime_h"] for r in rows]
    check("nach Zeit sortiert", hours == sorted(hours, key=float), str(hours))

    # --- Lauf 2: Endpunkt getrennt halten -----------------------------------
    out_csv2 = tmp / "out2.csv"
    rc2 = subprocess.call([
        sys.executable, str(HERE / "aggregate_learning_curve.py"),
        "--curve-csv", "sigmaflow_minimal", str(tmp / "FIXTURE_curve_sigmaflow_minimal.csv"),
        "--report-json", "sigmaflow_minimal", str(tmp / "FIXTURE_report_annealed12h.json"),
        "--out-csv", str(out_csv2),
    ], stdout=subprocess.DEVNULL)
    check("gemischte Eingaben laufen durch", rc2 == 0, f"rc={rc2}")

    rows2 = list(csv.DictReader(open(out_csv2, encoding="utf-8")))
    series = {r["series"] for r in rows2}
    check("zwei getrennte Serien entstanden",
          series == {"trajectory_snapshot_of_72h", "annealed_endpoint"}, str(sorted(series)))

    at12 = [r for r in rows2 if r["walltime_h"] == "12.0"]
    check("die beiden 12h-Objekte bleiben getrennte Zeilen", len(at12) == 2, f"{len(at12)}")
    check("sie liegen in verschiedenen Serien",
          len({r["series"] for r in at12}) == 2)
    check("der Endpunkt ist als solcher erkannt",
          any(r["series"] == "annealed_endpoint" and r["schedule_progress"] == "1.0" for r in at12))

    # --- Tag-Parser ----------------------------------------------------------
    sys.path.insert(0, str(HERE))
    from aggregate_learning_curve import hours_from_tag
    check("hours_from_tag('sched070ep_at_012h') == '12'", hours_from_tag("sched070ep_at_012h") == "12")
    check("hours_from_tag('sched070ep_at_072h') == '72'", hours_from_tag("sched070ep_at_072h") == "72")
    check("hours_from_tag('final') == ''", hours_from_tag("final") == "")

    print()
    passed = sum(1 for _, ok, _ in CHECKS if ok)
    print("=" * 70)
    print(f"ERGEBNIS: {passed}/{len(CHECKS)} Checks bestanden")
    print("=" * 70)
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
