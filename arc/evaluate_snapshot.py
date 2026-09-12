#!/usr/bin/env python3
"""
Auswertung EINES Snapshots -- Provenienz plus Metriken in einem Bundle.

ABGRENZUNG ZU arc/eval_snapshots.slurm
    Das Slurm-Skript faehrt die BILLIGE Kurve ueber ALLE Snapshots eines Laufs.
    Dieses Skript ist der Einzelzugriff: "wie gut war Arm X nach Y Stunden?"
    Es dupliziert die Sampling-Kette nicht, sondern ruft dieselbe auf
    (scripts/sample.py -> SigmaFlow_Evaluation.evaluate_run) und legt darum
    herum das, was bisher fehlte: die Herkunft des Checkpoints, direkt aus der
    Datei gelesen statt aus Logs rekonstruiert.

WARUM DIE PROVENIENZ AUS DEM CHECKPOINT KOMMT
    .meta.txt entsteht waehrend des Laufs aus stdout und kann Luecken haben.
    global_step, epoch und die Existenz der EMA-Gewichte stehen aber IM
    Checkpoint. Das ist die belastbarere Quelle und schliesst dieselbe Frage,
    die im J1-Provenienzaudit offen war.

WICHTIG -- WAS SO EIN SNAPSHOT IST
    Ein Snapshot bei 12 h aus dem 72h-Lauf ist ein ZWISCHENSTAND eines auf
    E Epochen kalibrierten Schedules. Er steht noch auf hoher Lernrate.
    Der alte, eigenstaendige 12h-Lauf war dagegen ein VOLLSTAENDIG
    ausannealtes Modell. Beide beantworten verschiedene Fragen und duerfen
    nie in dieselbe Kurve. Das Feld `snapshot_kind` haelt das fest.

Aufruf:
    # nur Provenienz, laeuft ueberall, braucht keine GPU
    python arc/evaluate_snapshot.py --arm sigmaflow_minimal --checkpoint <pfad>

    # mit Sampling und Metriken (auf ARC, im Code-Verzeichnis des Arms)
    python arc/evaluate_snapshot.py --arm sigmaflow_minimal --checkpoint <pfad> \
        --run --data-dir /data/stat-cadd/shug8458/data \
        --subset arc/eval_subset.txt --out-dir <ziel>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ANNEALED_ENDPOINT = "annealed_endpoint"
TRAJECTORY_SNAPSHOT = "trajectory_snapshot_of_72h"


def sha256_of(path: Path) -> str | None:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def read_meta_sidecar(ckpt: Path) -> dict[str, str]:
    """Die .meta.txt, die das Jobskript neben jeden Snapshot legt."""
    cand = ckpt.parent / (ckpt.stem + ".meta.txt")
    if not cand.exists():
        return {}
    out = {}
    for line in cand.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def read_checkpoint_provenance(ckpt: Path) -> dict:
    """Was der Checkpoint SELBST ueber sich weiss."""
    try:
        import torch
    except ImportError:
        return {"error": "torch nicht verfuegbar -- Provenienz nicht gelesen"}

    try:
        ck = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}

    hp = ck.get("hyper_parameters") or {}
    prov = {
        "global_step": ck.get("global_step"),
        "epoch": ck.get("epoch"),
        "has_ema_state_dict": "ema_state_dict" in ck,
        "has_optimizer_state": bool(ck.get("optimizer_states")),
        "has_lr_scheduler_state": bool(ck.get("lr_schedulers")),
        "pytorch_lightning_version": ck.get("pytorch-lightning_version"),
        "scheduler_max_steps": hp.get("max_steps"),
        "num_warmup_steps": hp.get("num_warmup_steps"),
    }
    for k in ("max_lr_start", "max_lr_end", "min_lr_start", "min_lr_end",
              "num_lr_cycles", "cycle_warmup_frac", "trans_score_weight",
              "rot_score_weight"):
        if k in hp:
            prov[k] = hp[k]

    # Wie weit ist der Anneal an dieser Stelle? Das ist die Zahl, die einen
    # Zwischenstand von einem Endpunkt unterscheidet.
    gs, ms = prov.get("global_step"), prov.get("scheduler_max_steps")
    if isinstance(gs, int) and isinstance(ms, int) and ms > 0:
        prov["schedule_progress"] = round(gs / ms, 4)
        prov["is_annealed_endpoint"] = gs >= ms
    return prov


def build_sampling_commands(args: argparse.Namespace, out_dir: Path) -> list[list[str]]:
    """Exakt die Kette aus arc/eval_snapshots.slurm, fuer EINEN Checkpoint.

    Bewusst identisch gehalten: eine abweichende Sampling-Konfiguration wuerde
    den Einzelwert unvergleichbar zur Kurve machen.
    """
    sample = [
        sys.executable, "scripts/sample.py",
        f"ckpt={args.checkpoint}",
        f"data_dir={args.data_dir}",
        "experiment=posebusters",
        f"output_dir={out_dir}",
        f"run_tag={args.arm}_{Path(args.checkpoint).stem}",
        f"seed={args.seed}",
        "num_seeds=1",
        f"ode.num_steps={args.nfe}",
        "graph.sample_conformer=false",
        "postprocessing.scoring=null",
        "postprocessing.bust_config=null",
        f"hydra.run.dir={out_dir}/hydra_out",
    ]
    if args.subset:
        sample.insert(-1, f"data.blacklist={args.subset}")

    evaluate = [
        sys.executable, "-m", "SigmaFlow_Evaluation.evaluate_run",
        "--sampling_root", str(out_dir),
        "--true_dir", args.true_dir or f"{args.data_dir}/posebusters_paper/posebusters_benchmark_set",
        "--label", f"{args.arm}_{Path(args.checkpoint).stem}",
        "--out_json", str(out_dir / "evaluation.json"),
    ]
    return [sample, evaluate]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True,
                   choices=["sigmaflow_minimal", "sigmadock", "sigmaflow_source", "sigmaflow_conf"])
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--subset", default=None, help="fester Auswertungssubset (arc/eval_subset.txt)")
    p.add_argument("--true-dir", dest="true_dir", default=None,
                   help="Referenzliganden. Default: posebusters_paper/posebusters_benchmark_set "
                        "(verschachtelt). <data>/posebusters ist LEER.")
    p.add_argument("--nfe", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run", action="store_true",
                   help="Sampling und Auswertung wirklich ausfuehren (braucht GPU und die Arm-Umgebung)")
    p.add_argument("--kind", choices=[ANNEALED_ENDPOINT, TRAJECTORY_SNAPSHOT], default=None,
                   help="ueberschreibt die automatische Einordnung")
    args = p.parse_args()

    ckpt: Path = args.checkpoint
    if not ckpt.exists():
        print(f"FEHLER: Checkpoint fehlt: {ckpt}", file=sys.stderr)
        return 2

    out_dir = args.out_dir or (ckpt.parent / f"eval_{ckpt.stem}")
    out_dir.mkdir(parents=True, exist_ok=True)

    sidecar = read_meta_sidecar(ckpt)
    prov = read_checkpoint_provenance(ckpt)

    # Einordnung: ein Snapshot ist genau dann ein Endpunkt, wenn der Anneal
    # durch ist. Ohne diese Information wird NICHT geraten.
    if args.kind:
        kind = args.kind
    elif prov.get("is_annealed_endpoint") is True:
        kind = ANNEALED_ENDPOINT
    elif prov.get("is_annealed_endpoint") is False:
        kind = TRAJECTORY_SNAPSHOT
    else:
        kind = "unknown"

    report = {
        "arm": args.arm,
        "checkpoint": {
            "path": str(ckpt.resolve()),
            "name": ckpt.name,
            "size_bytes": ckpt.stat().st_size,
            "sha256": sha256_of(ckpt),
        },
        "snapshot_kind": kind,
        "provenance": prov,
        "sidecar_meta": sidecar,
        "training_position": {
            "walltime_h": sidecar.get("walltime_h"),
            "epoch": prov.get("epoch", sidecar.get("epoch")),
            "global_step": prov.get("global_step", sidecar.get("global_step")),
            "examples_seen": sidecar.get("examples_seen"),
            "schedule_progress": prov.get("schedule_progress", sidecar.get("schedule_progress")),
            "lr": sidecar.get("lr"),
        },
        "weights_used_for_sampling": "EMA" if prov.get("has_ema_state_dict") else "raw_or_unknown",
        "sampling": {"nfe": args.nfe, "seed": args.seed, "subset": args.subset, "executed": False},
        "metrics": None,
    }

    cmds = None
    if args.data_dir:
        cmds = build_sampling_commands(args, out_dir)
        report["sampling"]["commands"] = [" ".join(c) for c in cmds]

    if args.run:
        if not args.data_dir:
            print("FEHLER: --run braucht --data-dir", file=sys.stderr)
            return 2
        for cmd in cmds or []:
            print(f"[eval] {' '.join(cmd[:3])} ...")
            rc = subprocess.call(cmd)
            if rc != 0:
                print(f"[eval] FEHLGESCHLAGEN (rc={rc}): {' '.join(cmd)}", file=sys.stderr)
                report["sampling"]["error"] = f"rc={rc}"
                break
        else:
            report["sampling"]["executed"] = True
            ev = out_dir / "evaluation.json"
            if ev.exists():
                report["metrics"] = json.loads(ev.read_text(encoding="utf-8"))

    out_json = out_dir / "snapshot_report.json"
    out_json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print()
    print("=" * 70)
    print(f"SNAPSHOT-BERICHT  {args.arm}")
    print("=" * 70)
    print(f"  Checkpoint     : {ckpt.name}")
    print(f"  SHA256         : {(report['checkpoint']['sha256'] or '?')[:16]}...")
    print(f"  Art            : {kind}")
    tp = report["training_position"]
    print(f"  Walltime       : {tp['walltime_h'] or '?'} h")
    print(f"  Epoche / Step  : {tp['epoch']} / {tp['global_step']}")
    print(f"  Anneal-Stand   : {tp['schedule_progress'] or '?'}")
    print(f"  EMA vorhanden  : {prov.get('has_ema_state_dict')}")
    print(f"  Optimizer/Sched: {prov.get('has_optimizer_state')} / {prov.get('has_lr_scheduler_state')}")
    if kind == TRAJECTORY_SNAPSHOT:
        print()
        print("  HINWEIS: Zwischenstand eines langen Schedules, KEIN ausannealtes")
        print("           Modell. Nicht mit den eigenstaendigen 6h/12h-Laeufen")
        print("           in dieselbe Kurve legen.")
    if not args.run and cmds:
        print()
        print("  Sampling nicht ausgefuehrt (--run fehlt). Befehle stehen im Bericht.")
    print()
    print(f"  geschrieben: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
