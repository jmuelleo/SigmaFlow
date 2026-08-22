"""Einheitliche Auswertung eines Sampling-Laufs.

WAS ES ERSETZT
  `SigmaFlow_Variants/posebusters_full_comparison/full_metrics.py` war ein
  Ad-hoc-Skript fuer den 12h-Vergleich: fest verdrahtete Laufliste, fester
  `_seed0`-Glob, kein TFD, kein Oracle@K. Fuer die finalen Laeufe braucht es
  einen Einstiegspunkt, der das Standard-Ausgabelayout von
  `arc/sample_pb_seeds.slurm` liest und ALLE Metriken in einem Durchgang
  liefert -- inklusive der Kette

      Single Sample  ->  Ranked Top-1@K  ->  Oracle@K

  die fuer die Confidence-Hypothese gebraucht wird.

EINGABELAYOUT (wie von sample_pb_seeds.slurm erzeugt)
    <sampling_root>/results/posebusters/<model>/seed_<k>/<complex>__*.sdf

ROBUSTHEITSPRINZIP
  Ein einzelner kaputter Komplex darf die Auswertung von 209 Komplexen nicht
  abbrechen. Jeder Fehler wird gezaehlt und mit Grund berichtet, nie still
  verschluckt. TFD-Abdeckung wird als eigene Zahl ausgewiesen, damit ein
  Deckungsverlust nicht als Qualitaetsaussage missverstanden wird.

KRISTALLKOPIEN
  84 von 209 PoseBusters-Dateien `<id>_ligands.sdf` enthalten mehrere
  kristallographische Kopien desselben Liganden. Naiv die erste zu nehmen
  erzeugte in einer frueheren Auswertung 42 Phantom-Ausreisser. Hier wird
  immer die naechstgelegene Kopie gewertet.

    python -m SigmaFlow_Evaluation.evaluate_run \\
        --sampling_root <...>/sampling/sigmaflow_minimal__controlled__raw__nfe25 \\
        --true_dir <...>/data/posebusters \\
        --label SF_MIN_72H
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from rdkit import Chem, RDLogger  # noqa: E402

from SigmaFlow_Evaluation.metrics import tfd as tfd_mod  # noqa: E402
from SigmaFlow_Evaluation.metrics.ranking import (  # noqa: E402
    PoseRecord,
    evaluate_ranking,
    format_report,
)

RDLogger.DisableLog("rdApp.*")

SUCCESS_THRESHOLDS = (2.0, 5.0)


@dataclass
class Failures:
    """Warum ein Komplex/eine Pose nicht gewertet wurde. Nie stillschweigend."""
    counts: dict = field(default_factory=lambda: defaultdict(int))

    def add(self, reason: str) -> None:
        self.counts[reason] += 1

    def report(self) -> str:
        if not self.counts:
            return "  (keine)"
        return "\n".join(f"  {r:<40s} {n}" for r, n in sorted(self.counts.items()))


def load_mols(path: str, sanitize: bool) -> list[Chem.Mol]:
    try:
        supp = Chem.SDMolSupplier(path, sanitize=sanitize, removeHs=True)
        return [m for m in supp if m is not None]
    except OSError:
        return []


def symmetry_rmsd(pos_pred: np.ndarray, mol_true: Chem.Mol) -> float:
    """RMSD gegen die wahre Pose, minimiert ueber Graphautomorphismen.

    53 % der Molekuele haben mehr als einen Automorphismus. Ohne diese
    Minimierung zaehlen korrekte Posen als falsch, weil die Atomnummerierung
    eine andere aequivalente Zuordnung gewaehlt hat.
    """
    pos_true = mol_true.GetConformer().GetPositions()
    if pos_pred.shape != pos_true.shape:
        return float("nan")
    try:
        matches = mol_true.GetSubstructMatches(mol_true, uniquify=False,
                                               useChirality=False, maxMatches=1000)
    except Exception:
        matches = ()
    if not matches:
        return float(np.sqrt(((pos_pred - pos_true) ** 2).sum(axis=1).mean()))
    best = np.inf
    for m in matches:
        d = pos_pred - pos_true[list(m)]
        best = min(best, float(np.sqrt((d ** 2).sum(axis=1).mean())))
    return best


def best_copy(pos_pred: np.ndarray, copies: list[Chem.Mol]) -> tuple[Chem.Mol | None, float]:
    """Naechstgelegene kristallographische Kopie und der zugehoerige RMSD."""
    best_m, best_r = None, np.inf
    for m in copies:
        r = symmetry_rmsd(pos_pred, m)
        if np.isfinite(r) and r < best_r:
            best_m, best_r = m, r
    return best_m, best_r


def collect(sampling_root: Path, true_dir: Path, model: str | None,
            fail: Failures) -> tuple[list[PoseRecord], dict[tuple[str, str], float]]:
    """Liest alle Posen aller Seeds und baut PoseRecords.

    Rueckgabe: (records, tfd_by_pose). PoseRecord ist frozen=True -- der
    TFD-Wert wird deshalb NICHT an das Objekt geheftet, sondern in einer
    Seitentabelle gehalten, die ueber (complex_id, pose_id) verknuepft."""
    base = sampling_root / "results" / "posebusters"
    if model:
        seed_dirs = sorted((base / model).glob("seed_*"))
    else:
        seed_dirs = sorted(base.glob("*/seed_*"))
    if not seed_dirs:
        raise SystemExit(f"Keine seed_*-Verzeichnisse unter {base}")

    records: list[PoseRecord] = []
    tfd_by_pose: dict[tuple[str, str], float] = {}
    tfd_ok, tfd_fail = 0, 0

    true_cache: dict[str, list[Chem.Mol]] = {}
    for sd in seed_dirs:
        seed = sd.name.replace("seed_", "")
        for sdf in sorted(sd.glob("*.sdf")):
            cid = sdf.name.split("__")[0]
            if cid not in true_cache:
                # Zwei Layouts sind zulaessig, weil beide real vorkommen:
                #   FLACH        <true_dir>/<cid>_ligands.sdf
                #                (so legt es SigmaFlow_Variants/.../true_ligands an)
                #   VERSCHACHTELT <true_dir>/<cid>/<cid>_ligands.sdf
                #                (so liegt der kanonische posebusters_benchmark_set)
                # Frueher wurde nur das flache Layout gesucht. Zeigte true_dir
                # auf den kanonischen Satz, fand die Auswertung KEINE einzige
                # Referenz -- und uebersprang jede Pose still (siehe unten).
                for cand in (true_dir / f"{cid}_ligands.sdf",
                             true_dir / f"{cid}_ligand.sdf",
                             true_dir / cid / f"{cid}_ligands.sdf",
                             true_dir / cid / f"{cid}_ligand.sdf"):
                    if cand.exists():
                        tp = cand
                        break
                else:
                    tp = true_dir / f"{cid}_ligands.sdf"   # existiert nicht; fuer die Meldung
                true_cache[cid] = load_mols(str(tp), sanitize=True) if tp.exists() else []
            copies = true_cache[cid]
            if not copies:
                fail.add("wahre Pose fehlt oder nicht ladbar")
                continue

            preds = load_mols(str(sdf), sanitize=True)
            if not preds:
                preds = load_mols(str(sdf), sanitize=False)
                if not preds:
                    fail.add("Vorhersage nicht ladbar")
                    continue
                fail.add("Vorhersage nur mit sanitize=False ladbar")
            mp = preds[0]

            pos_pred = mp.GetConformer().GetPositions()
            mol_true, rmsd = best_copy(pos_pred, copies)
            if mol_true is None or not np.isfinite(rmsd):
                fail.add("RMSD nicht berechenbar (Atomzahl?)")
                continue

            # TFD: darf ausfallen, ohne die Pose zu verwerfen.
            tfd_val = None
            try:
                res = tfd_mod.tfd(mol_true, mp, graft=True)
                if getattr(res, "ok", False) and np.isfinite(getattr(res, "value", np.nan)):
                    tfd_val = float(res.value)
                    tfd_ok += 1
                else:
                    tfd_fail += 1
                    fail.add(f"TFD: {getattr(res, 'reason', 'unbekannt')}")
            except Exception as exc:                      # noqa: BLE001
                tfd_fail += 1
                fail.add(f"TFD-Ausnahme: {type(exc).__name__}")

            pose_id = f"seed{seed}"
            records.append(PoseRecord(
                complex_id=cid,
                pose_id=pose_id,
                rmsd=rmsd,
                score=None,                # gesetzt, sobald ein Ranker existiert
                model_id=model or sd.parent.name,
            ))
            if tfd_val is not None:
                tfd_by_pose[(cid, pose_id)] = tfd_val
    print(f"[tfd] Abdeckung: {tfd_ok}/{tfd_ok + tfd_fail} "
          f"({100 * tfd_ok / max(tfd_ok + tfd_fail, 1):.1f} %)")
    return records, tfd_by_pose


def attach_scores(records: list[PoseRecord], score_path: Path,
                  fail: Failures) -> tuple[list[PoseRecord], dict]:
    """Haengt externe Ranking-Scores an die Posen.

    WOZU
      `evaluate_ranking` liefert Oracle@K immer, aber Top-1@K und die
      Kalibrierung nur, wenn jede Pose einen Score hat. Der Score kommt NICHT
      aus dieser Datei: SigmaDocks bestehende Heuristik in
      `chem/statistics.py::compute_heuristic` (Vinardo mal Mittel der
      PoseBusters-Checks) und ein spaeterer gelernter Ranker C_psi liefern
      beide dasselbe Format. Diese Funktion ist die einzige Nahtstelle.

    FORMAT (JSON)
        {"AAAA": {"seed0": -6.31, "seed1": -5.02}, "BBBB": {...}}
      Konvention: GROESSER IST BESSER. Vinardo-Bindungsenergien sind negativ
      und kleiner-ist-besser, muessen also beim Erzeugen negiert werden -- das
      passiert bewusst dort, wo der Score entsteht, nicht hier, damit an dieser
      Stelle keine Vorzeichenkonvention geraten werden muss.

    PoseRecord ist frozen=True, deshalb wird jeder Datensatz ersetzt statt
    veraendert (dataclasses.replace legt eine Kopie mit geaendertem Feld an).
    """
    raw = json.loads(Path(score_path).read_text(encoding="utf-8"))
    out, n_hit = [], 0
    for r in records:
        val = raw.get(r.complex_id, {}).get(r.pose_id)
        if val is None or not np.isfinite(float(val)):
            fail.add("kein Score fuer diese Pose")
            out.append(r)
            continue
        out.append(replace(r, score=float(val)))
        n_hit += 1
    cov = n_hit / max(len(records), 1)
    print(f"[score] Abdeckung: {n_hit}/{len(records)} ({100 * cov:.1f} %)")
    # Teilabdeckung macht Top-1@K unvergleichbar, weil dann fuer manche Posen
    # geraten und fuer andere gerankt wird. Deshalb explizit ausweisen.
    if 0 < cov < 1.0:
        print("[score] WARNUNG: unvollstaendige Abdeckung. Top-1@K vergleicht "
              "dann gerankte mit ungerankten Posen und ist nicht aussagekraeftig.")
    return out, {"n_scored": n_hit, "coverage": round(cov, 4),
                 "source": str(score_path)}


def basic_metrics(records: list[PoseRecord],
                  tfd_by_pose: dict[tuple[str, str], float]) -> dict:
    by_c = defaultdict(list)
    for r in records:
        by_c[r.complex_id].append(r)

    single = [rs[0].rmsd for rs in by_c.values()]        # ein Zug je Komplex
    allr = [r.rmsd for r in records]
    tfds = [tfd_by_pose[(r.complex_id, r.pose_id)] for r in records
            if (r.complex_id, r.pose_id) in tfd_by_pose]

    out = {
        "n_complexes": len(by_c),
        "n_poses": len(records),
        "rmsd_median_all": float(np.median(allr)) if allr else None,
        "rmsd_mean_all": float(np.mean(allr)) if allr else None,
        "single_sample": {},
        "tfd": {
            "n_valid": len(tfds),
            "coverage": round(len(tfds) / max(len(records), 1), 4),
            "median": float(np.median(tfds)) if tfds else None,
        },
    }
    for thr in SUCCESS_THRESHOLDS:
        out["single_sample"][f"success_below_{thr}A"] = (
            float(np.mean([r < thr for r in single])) if single else None
        )
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sampling_root", required=True, type=Path)
    p.add_argument("--true_dir", required=True, type=Path)
    p.add_argument("--model", default=None)
    p.add_argument("--label", default="run")
    p.add_argument("--out_json", default=None, type=Path)
    p.add_argument("--per_pose_csv", default=None, type=Path,
                   help="RMSD je EINZELNER Pose (complex,seed,rmsd). Ohne das "
                        "laesst sich Top-1 eines Rankers nicht berechnen, nur "
                        "die Obergrenze Oracle@K.")
    p.add_argument("--per_complex_csv", default=None, type=Path,
                   help="Eine Zeile je Komplex: RMSD (Seed 0, best, Median, worst) "
                        "und Erfolgsflags. Zum Verknuepfen mit Ligandeigenschaften.")
    p.add_argument("--scores", default=None, type=Path,
                   help="JSON {complex_id: {pose_id: score}}, groesser ist besser. "
                        "Ohne diese Datei bleibt Top-1@K undefiniert; Oracle@K "
                        "wird immer berechnet.")
    args = p.parse_args()

    fail = Failures()
    records, tfd_by_pose = collect(args.sampling_root, args.true_dir, args.model, fail)
    if not records:
        # Diagnostisch statt lakonisch. Der haeufigste Grund ist ein true_dir,
        # das zwar existiert, aber die Referenzliganden in einem anderen Layout
        # haelt -- dann wird JEDE Pose uebersprungen und das Ergebnis waere
        # stillschweigend leer. Bei einer Lernkurve ueber zwoelf Snapshots
        # faellt so etwas erst ganz am Ende auf.
        msg = [
            "Keine auswertbaren Posen gefunden.",
            "",
            f"  sampling_root : {args.sampling_root}",
            f"  true_dir      : {args.true_dir}",
            "",
            "  Erwartete Layouts fuer die Referenzliganden (beide zulaessig):",
            f"    flach         {args.true_dir}/<cid>_ligands.sdf",
            f"    verschachtelt {args.true_dir}/<cid>/<cid>_ligands.sdf",
            "",
            "  Auf ARC liegt der kanonische Satz verschachtelt unter",
            "    <data>/posebusters_paper/posebusters_benchmark_set/",
            "  waehrend <data>/posebusters/ nur raw/ und processed/ enthaelt.",
        ]
        if fail.counts:
            msg += ["", "  Gesammelte Gruende:"]
            msg += [f"    {n:>6} x  {reason}" for reason, n in sorted(fail.counts.items())]
        raise SystemExit("\n".join(msg))

    score_info = None
    if args.scores is not None:
        records, score_info = attach_scores(records, args.scores, fail)

    basics = basic_metrics(records, tfd_by_pose)
    ranking = evaluate_ranking(records)

    print(f"\n===== {args.label} =====")
    print(f"Komplexe: {basics['n_complexes']}   Posen: {basics['n_poses']}")
    print(f"RMSD Median (alle Posen): {basics['rmsd_median_all']:.3f} A")
    for k, v in basics["single_sample"].items():
        print(f"Single-Sample {k}: {100 * v:.1f} %")
    t = basics["tfd"]
    print(f"TFD: {t['n_valid']} gueltig ({100 * t['coverage']:.1f} %), "
          f"Median {t['median'] if t['median'] is None else round(t['median'], 4)}")
    print()
    print(format_report(ranking, label=args.label))
    print("\nNicht gewertet:")
    print(fail.report())

    # --- Per-Komplex-Zeilen -------------------------------------------------
    # Die Aggregate oben beantworten "wie gut insgesamt". Sie beantworten NICHT
    # "bei welchen Liganden geht es schief". Genau dafuer braucht es eine Zeile
    # je Komplex -- erst damit laesst sich Erfolg gegen Ligandeigenschaften
    # auftragen, etwa gegen die Fragmentzahl aus arc/count_fragments.py.
    # Je EINZELNE Pose. Der Per-Komplex-Export unten aggregiert bereits ueber
    # die Seeds und verliert damit genau die Information, die ein Ranker
    # braucht: welcher Seed welchen RMSD hatte.
    if args.per_pose_csv:
        with open(args.per_pose_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["complex", "seed", "rmsd"])
            for r in sorted(records, key=lambda x: (x.complex_id, x.pose_id)):
                seed = r.pose_id.replace("seed", "")
                w.writerow([r.complex_id, seed, f"{r.rmsd:.4f}"])
        print(f"[eval] Per-Pose-Tabelle: {args.per_pose_csv}")

    if args.per_complex_csv:
        per_c = defaultdict(list)
        for r in records:
            per_c[r.complex_id].append(r)
        with open(args.per_complex_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["complex", "n_poses", "rmsd_seed0", "rmsd_best",
                        "rmsd_median", "rmsd_worst",
                        "success_2A_seed0", "success_2A_oracle",
                        "success_5A_seed0", "success_5A_oracle"])
            for cid in sorted(per_c):
                rs = per_c[cid]
                vals = sorted(r.rmsd for r in rs)
                first = rs[0].rmsd          # Seed 0, dieselbe Konvention wie basics
                w.writerow([
                    cid, len(rs),
                    f"{first:.4f}", f"{vals[0]:.4f}",
                    f"{float(np.median(vals)):.4f}", f"{vals[-1]:.4f}",
                    int(first < 2.0), int(vals[0] < 2.0),
                    int(first < 5.0), int(vals[0] < 5.0),
                ])
        print(f"[eval] Per-Komplex-Tabelle: {args.per_complex_csv}")

    out = {"label": args.label, "basics": basics, "ranking": ranking,
           "scores": score_info, "failures": dict(fail.counts)}
    dest = args.out_json or (args.sampling_root / f"evaluation_{args.label}.json")
    Path(dest).write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\n[eval] geschrieben: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
