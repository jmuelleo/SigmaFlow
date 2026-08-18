"""Schreibt ein Reproduzierbarkeits-Manifest fuer einen langen Trainingslauf.

WARUM
  Nach 72 Stunden muss ohne Ratespiel rekonstruierbar sein, was gelaufen ist:
  welcher Commit, welche Hardware, wie viele Epochen und Schritte tatsaechlich
  erreicht wurden, wie viele Beispiele das Modell gesehen hat. Genau diese
  Zahlen braucht die Thesis fuer die Compute-Einordnung gegen das Original
  (4 Tage auf 4 A100, Batch 32, max 256 Epochen).

ENTWURFSENTSCHEIDUNG
  Das Skript liest AUSSCHLIESSLICH Dateien, die der Lauf ohnehin schreibt
  (run_metadata.txt, train_stdout.log, environment.txt, das Experimentverzeichnis).
  Es importiert weder sigmadock noch torch und aendert nichts am Training.
  Damit funktioniert es identisch fuer SigmaDock (dessen Verzeichnis read-only
  ist) und fuer jede SigmaFlow-Variante.

  Werte, die sich nicht sicher aus dem Log ablesen lassen, werden als null
  geschrieben -- nicht geschaetzt. Ein fehlender Wert ist ehrlicher als ein
  plausibel aussehender falscher.

    python arc/write_manifest.py --run_dir <RUN_DIR> --run_id ... --model ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

# Die Paper-Zahl. Sie steht hier NUR noch als Vergleichswert fuer den Report;
# als Default fuer Rechnungen ist sie bewusst entfernt worden (siehe --n_train).
N_TRAIN_PAPER = 19443   # Paper, Abschnitt "Datasets": PDBBind v2020


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _kv_from_metadata(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip()
    return out


def _last_int(pattern: str, text: str) -> int | None:
    """Letzter Treffer einer Gruppe als int, sonst None."""
    hits = re.findall(pattern, text)
    if not hits:
        return None
    try:
        return int(float(hits[-1]))
    except (TypeError, ValueError):
        return None


def _last_float(pattern: str, text: str) -> float | None:
    hits = re.findall(pattern, text)
    if not hits:
        return None
    try:
        return float(hits[-1])
    except (TypeError, ValueError):
        return None


def parse_progress(log: str) -> dict:
    """Zieht Epochen-/Schrittzahl aus Lightnings Ausgabe.

    Lightning schreibt die Fortschrittszeile als 'Epoch N: ... it/s'. Je nach
    Version und Terminal ist das mit \\r ueberschrieben; wir suchen deshalb
    grosszuegig und nehmen den letzten Treffer. Findet sich nichts, bleibt der
    Wert null statt einer Schaetzung.
    """
    return {
        "epochs_completed": _last_int(r"Epoch\s+(\d+)", log),
        "optimizer_steps": _last_int(r"(?:global_step|Step)[\s=:]+(\d+)", log),
        "best_val_loss": _last_float(r"loss_val/total[^0-9\-]*(-?\d+\.\d+)", log),
        "final_lr": _last_float(r"lr-AdamW[^0-9\-]*(-?[\d.eE+-]+)", log),
        "nan_mentions": len(re.findall(r"\bnan\b", log, flags=re.I)),
    }


def parse_gpu(env_text: str) -> dict:
    model = re.search(r"\|\s+\d+\s+(NVIDIA [^|]+?)\s{2,}", env_text)
    return {
        "gpu_model": model.group(1).strip() if model else None,
        "gpu_count": len(re.findall(r"^\|\s+\d+\s+NVIDIA", env_text, flags=re.M)) or None,
    }


def snapshot_records(rd: Path) -> list[dict]:
    """Ein Datensatz je Snapshot: Identitaet, Trainingsposition, EMA-Status.

    WARUM SHA256
        Ein Checkpoint ohne Pruefsumme laesst sich spaeter nicht mehr
        zweifelsfrei einem Lauf zuordnen. Genau das war der Kern der
        J1-Provenienzfrage.

    WARUM NICHT torch.load
        Das Manifest wird direkt nach dem Training geschrieben, wo unter
        Umstaenden noch GPU-Speicher belegt ist, und es soll auch ohne
        Torch-Umgebung laufen. Step/Epoche kommen deshalb aus den
        .meta.txt-Dateien, die das Jobskript neben jeden Snapshot legt.
    """
    snap_dir = rd / "snapshots"
    if not snap_dir.is_dir():
        return []

    records = []
    for ckpt in sorted(snap_dir.glob("*.ckpt")):
        h = hashlib.sha256()
        try:
            with open(ckpt, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            digest = h.hexdigest()
        except OSError:
            digest = None

        rec = {
            "name": ckpt.name,
            "path": str(ckpt),
            "size_bytes": ckpt.stat().st_size if ckpt.exists() else None,
            "sha256": digest,
            # Ein Snapshot aus diesem Lauf ist ein Zwischenstand eines langen
            # Schedules, kein ausannealter Endpunkt. Die Unterscheidung steht
            # explizit hier, damit sie eine spaetere Auswertung nicht raten muss.
            "snapshot_kind": "mid_schedule_snapshot",
            "is_annealed_endpoint": False,
        }

        meta_file = ckpt.with_suffix("").with_suffix(".meta.txt")
        if not meta_file.exists():
            meta_file = snap_dir / (ckpt.stem + ".meta.txt")
        if meta_file.exists():
            for line in meta_file.read_text(encoding="utf-8", errors="replace").splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    rec[k.strip()] = v.strip()
        records.append(rec)
    return records


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True, type=Path)
    p.add_argument("--run_id", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--code_dir", required=True)
    p.add_argument("--batch_size", type=int, required=True)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--precision", default="32")
    p.add_argument("--max_epochs", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train_rc", type=int, default=None)
    # PFLICHTANGABE. Frueher fiel das Skript stillschweigend auf die Paper-Zahl
    # zurueck; eine abweichende reale Datafront waere damit unbemerkt in
    # examples_seen und in jede abgeleitete Groesse eingegangen.
    p.add_argument("--n_train", type=int, required=True,
                   help="GEMESSENE Laenge der Trainings-Datafront (arc/probe_datafront_size.py)")
    p.add_argument("--max_steps", type=int, default=None, help="Scheduler-Horizont")
    p.add_argument("--train_status", default=None,
                   help="COMPLETED | CRASHED_rcN | WALLTIME_INTERRUPTED | KILLED_SIGNAL_N")
    p.add_argument("--world_size", type=int, default=1)
    p.add_argument("--cuda_precision", default=None)
    p.add_argument("--partition", default=None)
    p.add_argument("--requested_walltime", default=None)
    p.add_argument("--horizon_source", default=None)
    p.add_argument("--samples_per_s", type=float, default=None)
    args = p.parse_args()

    rd: Path = args.run_dir
    meta = _kv_from_metadata(_read(rd / "run_metadata.txt"))
    log = _read(rd / "train_stdout.log")
    envt = _read(rd / "environment.txt")
    git = _read(rd / "git_commit.txt")

    progress = parse_progress(log)
    eff_batch = args.batch_size * args.accum

    # Beispiele gesehen: nur berechnen, wenn eine der beiden Groessen bekannt ist.
    examples_seen = None
    if progress["optimizer_steps"] is not None:
        examples_seen = progress["optimizer_steps"] * eff_batch
    elif progress["epochs_completed"] is not None:
        examples_seen = progress["epochs_completed"] * args.n_train

    walltime_s = None
    start, end = meta.get("start_utc"), meta.get("end_utc")
    if start and end:
        try:
            fmt = "%Y-%m-%dT%H:%M:%SZ"
            walltime_s = int((datetime.strptime(end, fmt) - datetime.strptime(start, fmt)).total_seconds())
        except ValueError:
            pass

    snaps = sorted(str(q.name) for q in (rd / "snapshots").glob("*.ckpt")) \
        if (rd / "snapshots").is_dir() else []

    commit = None
    for line in git.splitlines():
        if re.fullmatch(r"[0-9a-f]{7,40}", line.strip()):
            commit = line.strip()
            break

    manifest = {
        "run_id": args.run_id,
        "model": args.model,
        "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "slurm_job_id": meta.get("slurm_job_id"),
        "hostname": meta.get("node"),
        "partition": meta.get("partition"),
        "code_dir": args.code_dir,
        "git_commit": commit,
        "python": meta.get("python_version"),
        "dataset": {
            "train": "pdbbind-general (PDBBind v2020)",
            "val": "posebusters",
            "n_train_measured": args.n_train,
            "n_train_paper": N_TRAIN_PAPER,
            "n_train_matches_paper": args.n_train == N_TRAIN_PAPER,
        },
        "config": {
            "batch_size_per_step": args.batch_size,
            "accum_grad_batches": args.accum,
            "effective_batch": eff_batch,
            "precision": args.precision,
            "max_epochs": args.max_epochs,
            "max_steps_scheduler_horizon": args.max_steps,
            "steps_per_epoch_safe": args.n_train // eff_batch if eff_batch else None,
            "microbatches_per_epoch": -(-args.n_train // args.batch_size) if args.batch_size else None,
            "world_size": args.world_size,
            "cuda_precision": args.cuda_precision,
            "tf32_enabled": (args.cuda_precision in ("high", "medium")) if args.cuda_precision else None,
            "horizon_source": args.horizon_source,
            "measured_samples_per_s": args.samples_per_s,
            "seed": args.seed,
            "optimizer": "AdamW",
            "scheduler": "LinearLR warmup (1/16 of max_steps) + cosine annealing, stepwise",
        },
        "hardware": parse_gpu(envt),
        "result": {
            "train_return_code": args.train_rc,
            "train_status": args.train_status,
            "anneal_completed": (
                progress["optimizer_steps"] >= args.max_steps
                if progress["optimizer_steps"] is not None and args.max_steps else None
            ),
            **progress,
            "examples_seen": examples_seen,
            "walltime_s": walltime_s,
            "gpu_hours": round(walltime_s / 3600, 2) if walltime_s else None,
        },
        "budget_vs_paper": {
            "paper_max_epochs": 256,
            "paper_effective_batch": 32,
            "paper_gpu_hours": 384,
            "fraction_of_paper_epochs": (
                round(progress["epochs_completed"] / 256, 4)
                if progress["epochs_completed"] is not None else None
            ),
        },
        "runtime": {
            "partition": args.partition or meta.get("partition"),
            "requested_walltime": args.requested_walltime,
            "start_utc": meta.get("start_utc"),
            "end_utc": meta.get("end_utc"),
        },
        "checkpoints": {
            "experiment_dir": _read(rd / "experiment_dir.txt").strip() or None,
            "snapshots": snapshot_records(rd),
            "snapshot_names": snaps,
        },
        "notes": [
            "Werte, die sich nicht eindeutig aus dem Log lesen liessen, stehen als null.",
            "epochs_completed/optimizer_steps stammen aus Lightnings stdout, nicht aus dem Checkpoint.",
            "n_train ist GEMESSEN (probe_datafront_size.py), nicht aus dem Paper uebernommen.",
            "Snapshots sind Zwischenstaende EINES langen Schedules, keine eigenstaendig "
            "ausannealten Endpunkte. Nicht mit den alten 6h/12h-Laeufen in eine Kurve mischen.",
        ],
    }

    out = rd / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[manifest] geschrieben: {out}")
    for k in ("epochs_completed", "optimizer_steps", "examples_seen", "gpu_hours"):
        print(f"[manifest]   {k:<18} = {manifest['result'][k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
