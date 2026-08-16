"""SigmaDocks feste Ranking-Heuristik, paper-konform reproduziert.

DIE FORMEL (Prat et al. 2026, Appendix F.2, S. 33)

    s_i = - b_i * p_i^beta ,      beta = 4

  b_i  Vinardo-Bindungsenergie der Probe i (negativ; kleiner = besser)
  p_i  Mittel der PoseBusters-Checks ueber genau die im Paper genannten
       stereochemischen Eigenschaften, in [0,1]
  s_i  hoeher = besser

WARUM DAS HIER NEU GESCHRIEBEN WIRD, OBWOHL ES `compute_heuristic` GIBT
  `chem/statistics.py::compute_heuristic` rechnet
      b * (score_bias + p^pb_exponent)
  also eine ADDITIVE Variante. Mit score_bias = 0 und pb_exponent = 4 ist sie
  identisch zur Paperformel -- aber KEINER der beiden Parameter hat einen
  Default in `conf/`, und die Funktion sitzt tief in einer Pipeline, die
  gnina-Aufrufe, Dateisystemlayout und Top-k-Sortierung vermischt.

  Fuer den Vergleich "feste Heuristik gegen gelernte Confidence" wird aber
  genau eine Sache gebraucht: eine Zahl je Pose, auf DEMSELBEN Posensatz,
  in demselben Format wie jeder andere Ranker. Das ist diese Datei. Sie ruft
  gnina nicht selbst auf, sondern liest die bereits berechneten Werte.

WICHTIG - Vorzeichen
  Vinardo-Energien sind negativ und "kleiner ist besser". `-b_i` dreht das um.
  Die Umdrehung passiert HIER, an der Stelle, an der die Konvention bekannt
  ist -- nicht in `evaluate_run.attach_scores`, wo sie geraten werden muesste.

    python -m SigmaFlow_Evaluation.ranking.heuristic_score \\
        --pb_csv <...>/posebusters_seed0.csv --out scores.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# Genau die fuenf Eigenschaften, die Appendix F.2 nennt:
# "bond lengths, bond angles, tetrahedral chirality mismatch, internal steric
#  clash, and minimum distance to the protein".
PB_CHECKS = (
    "bond_lengths",
    "bond_angles",
    "tetrahedral_chirality",
    "internal_steric_clash",
    "minimum_distance_to_protein",
)
BETA = 4.0


def mixed_score(binding_energy: float, pb_fraction: float,
                beta: float = BETA) -> float:
    """s = -b * p^beta. Hoeher ist besser.

    `pb_fraction` muss in [0,1] liegen. Ein Wert ausserhalb deutet auf eine
    falsch aggregierte Checkliste hin und wird nicht stillschweigend geklemmt --
    ein zu 1.2 gemitteltes p wuerde den Score um 2 Punkte verschieben, ohne
    dass irgendetwas auffiele.
    """
    if not 0.0 <= pb_fraction <= 1.0:
        raise ValueError(f"pb_fraction muss in [0,1] liegen, war {pb_fraction}")
    if not math.isfinite(binding_energy):
        raise ValueError(f"binding_energy ist nicht endlich: {binding_energy}")
    return -binding_energy * (pb_fraction ** beta)


def pb_fraction_from_row(row: dict, checks: tuple[str, ...] = PB_CHECKS
                         ) -> tuple[float, list[str]]:
    """Mittel der vorhandenen PB-Checks; meldet die fehlenden mit.

    Fehlende Checks werden NICHT als bestanden gewertet. Das waere die
    gefaehrliche Variante: eine Datei, in der ein Check nicht berechnet wurde,
    bekaeme so einen besseren Score als eine, in der er berechnet und nicht
    bestanden wurde.
    """
    vals, missing = [], []
    for c in checks:
        if c in row and row[c] is not None and str(row[c]).strip() != "":
            v = str(row[c]).strip().lower()
            if v in ("true", "1", "1.0", "yes"):
                vals.append(1.0)
            elif v in ("false", "0", "0.0", "no"):
                vals.append(0.0)
            else:
                try:
                    vals.append(float(v))
                except ValueError:
                    missing.append(c)
        else:
            missing.append(c)
    if not vals:
        return float("nan"), missing
    return sum(vals) / len(vals), missing


def score_table(rows: list[dict], energy_key: str = "vinardo",
                beta: float = BETA) -> tuple[dict, dict]:
    """rows -> ({complex_id: {pose_id: score}}, Diagnose).

    Erwartet je Zeile mindestens `complex_id`, `pose_id` und den Energieschluessel.
    """
    out: dict[str, dict[str, float]] = {}
    diag = {"n_rows": len(rows), "n_scored": 0, "n_no_energy": 0,
            "n_no_pb": 0, "missing_checks": {}}
    for row in rows:
        cid, pid = row.get("complex_id"), row.get("pose_id")
        if cid is None or pid is None:
            continue
        e = row.get(energy_key)
        if e is None or str(e).strip() == "":
            diag["n_no_energy"] += 1
            continue
        p, missing = pb_fraction_from_row(row)
        for m in missing:
            diag["missing_checks"][m] = diag["missing_checks"].get(m, 0) + 1
        if not math.isfinite(p):
            diag["n_no_pb"] += 1
            continue
        try:
            s = mixed_score(float(e), p, beta)
        except ValueError:
            continue
        out.setdefault(str(cid), {})[str(pid)] = s
        diag["n_scored"] += 1
    diag["coverage"] = round(diag["n_scored"] / max(len(rows), 1), 4)
    return out, diag


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pb_csv", required=True, type=Path,
                    help="CSV mit complex_id, pose_id, vinardo und den PB-Checks.")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--energy_key", default="vinardo")
    ap.add_argument("--beta", type=float, default=BETA)
    args = ap.parse_args()

    import csv
    with args.pb_csv.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    scores, diag = score_table(rows, args.energy_key, args.beta)
    args.out.write_text(json.dumps(scores, indent=2), encoding="utf-8")

    print(f"[heuristic] beta = {args.beta}")
    print(f"[heuristic] Zeilen {diag['n_rows']}, bewertet {diag['n_scored']} "
          f"({100 * diag['coverage']:.1f} %)")
    if diag["n_no_energy"]:
        print(f"[heuristic] ohne Energie: {diag['n_no_energy']}  "
              "(gnina nicht gelaufen?)")
    if diag["n_no_pb"]:
        print(f"[heuristic] ohne PB-Checks: {diag['n_no_pb']}")
    for k, v in sorted(diag["missing_checks"].items()):
        print(f"[heuristic] Check fehlt in {v} Zeilen: {k}")
    if diag["coverage"] < 1.0:
        print("[heuristic] WARNUNG: unvollstaendige Abdeckung. Top-1@K vergleicht "
              "dann gerankte mit ungerankten Posen.")
    print(f"[heuristic] geschrieben: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
