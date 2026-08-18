"""Waehlt die hochfragmentierten Komplexe fuer die qualitative Abbildung.

ZWEI BETRIEBSARTEN

  --counts-csv   lokal. Liest eine bereits erzeugte Tabelle
                 (complex_id, fragments, atoms, rotatable_bonds).
                 Die Auswahllogik ist damit ohne ARC testbar.

  --datafront    auf ARC. Berechnet die Fragmentzahlen aus dem
                 vorverarbeiteten Datensatz und schreibt die Tabelle,
                 die dann lokal weiterverwendet werden kann.

  Diese Trennung ist Absicht: die Zaehlung braucht den Datensatz, die
  Auswahl nicht. Nur der erste Schritt muss auf dem Cluster laufen.

BEKANNTE GROESSENORDNUNG (aus EXP-203, ueber 209 Komplexe verifiziert)

  Zustandsdimension D = 6 x Fragmente; Median D 24, q90 42, Maximum 66
  -> Median 4, q90 7, Maximum 11 Fragmente.

  Liganden mit 15+ Fragmenten existieren in diesem Benchmark NICHT. Wer 20
  oder 30 sucht, sucht vergebens; das Skript sagt das statt eine leere
  Auswahl zurueckzugeben.

    python -m visualization.select_cases --counts-csv counts.csv \\
        --results-dir SigmaFlow_Variants/posebusters_full_comparison
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

TARGET_BANDS = (7, 8, 9, 10)


def read_counts(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        try:
            out.append({
                "complex_id": r["complex_id"].strip(),
                "fragments": int(r["fragments"]),
                "state_dimension": int(r.get("state_dimension") or 6 * int(r["fragments"])),
                "ligand_atoms": int(r.get("ligand_atoms") or 0),
                "rotatable_bonds": int(r.get("rotatable_bonds") or -1),
            })
        except (KeyError, ValueError):
            continue
    return out


def read_success(results_dir: Path) -> dict[str, dict]:
    """RMSD<=2A je Komplex aus den PoseBusters-CSVs.

    ACHTUNG: fuer SigmaDock 12h ist `..._lrfix.csv` die gueltige Datei.
    Die Variante ohne `_lrfix` enthaelt 0 Treffer und ist ein Artefakt --
    sie wird hier bewusst nicht gelesen.
    """
    files = {
        "sigmaflow_12h": "posebusters_ligandonly_SigmaFlow_12h_framefix.csv",
        "sigmadock_12h": "posebusters_ligandonly_SigmaDock_12h_lrfix.csv",
    }
    out: dict[str, dict] = {}
    for label, fn in files.items():
        p = results_dir / fn
        if not p.exists():
            print(f"[warn] fehlt, wird uebersprungen: {p}", file=sys.stderr)
            continue
        with p.open(encoding="utf-8", errors="replace", newline="") as fh:
            for r in csv.DictReader(fh):
                fname = (r.get("file") or "").replace("\\", "/").split("/")[-1]
                cid = fname.split("__")[0]
                rk = [c for c in r if c.lower().startswith("rmsd")]
                if not cid or not rk:
                    continue
                out.setdefault(cid, {})[label] = \
                    str(r[rk[0]]).strip().lower() == "true"
    return out


def distribution(rows: list[dict]) -> dict:
    fr = sorted(r["fragments"] for r in rows)
    n = len(fr)

    def q(p):
        return fr[min(n - 1, int(p * n))] if n else 0

    return {
        "n": n, "min": fr[0] if n else 0, "median": q(0.50), "mean":
        round(sum(fr) / n, 2) if n else 0, "q75": q(0.75), "q90": q(0.90),
        "q95": q(0.95), "max": fr[-1] if n else 0,
        "ge10": sum(f >= 10 for f in fr), "ge15": sum(f >= 15 for f in fr),
        "ge20": sum(f >= 20 for f in fr), "ge30": sum(f >= 30 for f in fr),
        "ge40": sum(f >= 40 for f in fr),
    }


def select(rows: list[dict], success: dict[str, dict]) -> list[dict]:
    """Ein Fall je Band 7,8,9,10 plus das Maximum.

    Innerhalb eines Bandes wird nach wissenschaftlichem Interesse sortiert:
    Faelle, in denen sich die beiden Modelle UNTERSCHEIDEN, zuerst -- ein
    Komplex, den beide verfehlen, zeigt in der Abbildung nur zweimal
    dasselbe Scheitern.
    """
    for r in rows:
        s = success.get(r["complex_id"], {})
        r["sigmaflow_12h_below2A"] = s.get("sigmaflow_12h")
        r["sigmadock_12h_below2A"] = s.get("sigmadock_12h")
        sf, sd = r["sigmaflow_12h_below2A"], r["sigmadock_12h_below2A"]
        if sf is None or sd is None:
            r["_interest"], r["winner"] = 0, "unbekannt"
        elif sf != sd:
            r["_interest"] = 3
            r["winner"] = "SigmaFlow" if sf else "SigmaDock"
        elif sf:
            r["_interest"], r["winner"] = 2, "beide"
        else:
            r["_interest"], r["winner"] = 1, "keiner"

    chosen: list[dict] = []
    used: set[str] = set()
    by_band = {b: [r for r in rows if r["fragments"] == b] for b in TARGET_BANDS}
    for band in TARGET_BANDS:
        cands = sorted(by_band[band],
                       key=lambda r: (-r["_interest"], -r["ligand_atoms"]))
        if not cands:                      # Band leer -> naechstbestes nehmen
            cands = sorted(rows, key=lambda r: (abs(r["fragments"] - band),
                                                -r["_interest"]))
        for c in cands:
            if c["complex_id"] not in used:
                c["role"] = f"F={band}"
                chosen.append(c)
                used.add(c["complex_id"])
                break
    mx = max(rows, key=lambda r: (r["fragments"], r["_interest"]), default=None)
    if mx and mx["complex_id"] not in used:
        mx["role"] = f"Maximum (F={mx['fragments']})"
        chosen.append(mx)
    return chosen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts-csv", type=Path,
                    help="complex_id,fragments[,state_dimension,ligand_atoms,rotatable_bonds]")
    ap.add_argument("--datafront", type=Path,
                    help="ARC: vorverarbeiteter Datensatz (schreibt die Tabelle)")
    ap.add_argument("--results-dir", type=Path,
                    default=Path("SigmaFlow_Variants/posebusters_full_comparison"))
    ap.add_argument("--out", type=Path, default=Path("visualisations/selection.csv"))
    a = ap.parse_args()

    if a.datafront:
        print("[datafront] Diese Betriebsart braucht den vorverarbeiteten\n"
              "            Datensatz und laeuft nur auf ARC. Sie ist hier\n"
              "            bewusst nicht implementiert, weil sie ohne die\n"
              "            echten Daten nicht getestet werden koennte.\n"
              "            Auf ARC: Fragmentzahlen je Komplex aus\n"
              "            batch.frag_idx_map zaehlen und als --counts-csv\n"
              "            schreiben (siehe ARC_RETURN_RUNBOOK.md, Phase A).",
              file=sys.stderr)
        return 2
    if not a.counts_csv or not a.counts_csv.exists():
        print("--counts-csv fehlt oder existiert nicht", file=sys.stderr)
        return 2

    rows = read_counts(a.counts_csv)
    if not rows:
        print("keine verwertbaren Zeilen", file=sys.stderr)
        return 1
    d = distribution(rows)
    print("=== Fragmentzahl-Verteilung ===")
    for k in ("n", "min", "median", "mean", "q75", "q90", "q95", "max"):
        print(f"  {k:8s} {d[k]}")
    print("  Komplexe mit  >=10 / >=15 / >=20 / >=30 / >=40 Fragmenten: "
          f"{d['ge10']} / {d['ge15']} / {d['ge20']} / {d['ge30']} / {d['ge40']}")
    if d["ge15"] == 0:
        print("  -> Liganden mit 15+ Fragmenten existieren in diesem "
              "Benchmark nicht. Das Zielband ist 7..%d." % d["max"])

    success = read_success(a.results_dir) if a.results_dir.exists() else {}
    chosen = select(rows, success)

    print("\n=== Auswahl ===")
    hdr = f"{'Rolle':<18}{'Komplex':<12}{'F':>3}{'Atome':>7}{'RotBnd':>8}  Gewinner"
    print(hdr); print("-" * len(hdr))
    for c in chosen:
        print(f"{c['role']:<18}{c['complex_id']:<12}{c['fragments']:>3}"
              f"{c['ligand_atoms']:>7}{c['rotatable_bonds']:>8}  {c['winner']}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[k for k in chosen[0] if not k.startswith("_")])
        w.writeheader()
        for c in chosen:
            w.writerow({k: v for k, v in c.items() if not k.startswith("_")})
    print(f"\ngeschrieben: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
