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
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

N_TRAIN_PDBBIND_GENERAL = 19443   # Paper, Abschnitt "Datasets": PDBBind v2020


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
    p.add_argument("--n_train", type=int, default=N_TRAIN_PDBBIND_GENERAL)
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
            "n_train_assumed": args.n_train,
        },
        "config": {
            "batch_size_per_step": args.batch_size,
            "accum_grad_batches": args.accum,
            "effective_batch": eff_batch,
            "precision": args.precision,
            "max_epochs": args.max_epochs,
            "seed": args.seed,
            "optimizer": "AdamW",
            "scheduler": "LinearLR warmup (1/16 of max_steps) + cosine annealing, stepwise",
        },
        "hardware": parse_gpu(envt),
        "result": {
            "train_return_code": args.train_rc,
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
        "checkpoints": {
            "experiment_dir": _read(rd / "experiment_dir.txt").strip() or None,
            "snapshots": snaps,
        },
        "notes": [
            "Werte, die sich nicht eindeutig aus dem Log lesen liessen, stehen als null.",
            "epochs_completed/optimizer_steps stammen aus Lightnings stdout, nicht aus dem Checkpoint.",
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
