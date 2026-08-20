#!/usr/bin/env python3
"""
Schreibt RUN_STATUS.json in den Experimentordner eines EXP-110-Laufs.

Warum ein eigenes Skript und kein Heredoc in der Shell: der erste Entwurf
baute das JSON mit `cat > ... <<JSON` zusammen und erzeugte ungueltiges JSON,
sobald ein Pfad einen Backslash enthielt. json.dump kann das nicht passieren.

Alle Werte kommen ueber die Kommandozeile, damit das Skript auch aus einer
Signalfalle heraus laeuft. Es darf nie hart scheitern.

    python slurm/write_run_status.py <exp_dir> <status> <exit_code> \
        <start_unix> <ende_unix> <job_id> <git_commit>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _ensure_sigmadock_importable(hint: Path) -> None:
    """Den src-Baum der Variante selbst in sys.path haengen.

    Der Checkpoint enthaelt gepickelte Projektklassen (sigmadock.oracle.HParams
    und aehnliche); ohne importierbares sigmadock scheitert torch.load mit
    ModuleNotFoundError. Sich hier auf ein exportiertes PYTHONPATH zu
    verlassen waere unnoetig fragil -- der Pfad steht relativ zum Checkpoint
    fest, also wird er hergeleitet.
    """
    for base in (hint, *hint.parents):
        cand = base / "src"
        if (cand / "sigmadock" / "__init__.py").is_file():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            return


def read_ckpt_meta(path: Path) -> dict[str, object]:
    """Kopfdaten des Checkpoints; gibt im Fehlerfall eine Diagnose zurueck."""
    out: dict[str, object] = {"pfad": str(path)}
    if not path.is_file():
        out["status"] = "FEHLT"
        return out
    out["size_mb"] = round(path.stat().st_size / 1e6, 1)
    _ensure_sigmadock_importable(path.resolve())
    try:
        import torch

        ck = torch.load(path, map_location="cpu", weights_only=False)
        hp = ck.get("hyper_parameters", {}) or {}
        sd = ck.get("state_dict", {}) or {}
        gs, ms = ck.get("global_step"), hp.get("max_steps")
        out.update(
            status="OK",
            epoch=ck.get("epoch"),
            global_step=gs,
            max_steps_ziel=ms,
            hat_trans_block=any(".trans_block." in k for k in sd),
            hat_rot_block=any(".rot_block." in k for k in sd),
            hat_force_block=any(".force_block." in k for k in sd),
            hat_ema="ema_state_dict" in ck,
            hat_optimizer=bool(ck.get("optimizer_states")),
            hat_scheduler=bool(ck.get("lr_schedulers")),
        )
        if isinstance(gs, int) and isinstance(ms, int) and ms > 0:
            out["fortschritt_prozent"] = round(100.0 * gs / ms, 2)
    except Exception as e:  # noqa: BLE001 - darf nie hart scheitern
        out["status"] = "LESEFEHLER"
        out["fehler"] = f"{type(e).__name__}: {e}"[:200]
    return out


def main() -> int:
    if len(sys.argv) < 8:
        print("zu wenige Argumente", file=sys.stderr)
        return 2
    exp_dir = Path(sys.argv[1]).resolve()
    status, rc, t0, t1, job_id, commit = sys.argv[2:8]

    try:
        secs = int(t1) - int(t0)
    except ValueError:
        secs = -1
    hh, mm, ss = secs // 3600, secs % 3600 // 60, secs % 60

    doc = {
        "experiment": "EXP-110",
        "variante": "Zwei-Kopf-Vektorfeld",
        "label": "EXP-110 / Two-Head / 12h",
        "status": status,
        "vollstaendig": status == "COMPLETED",
        "exit_code": int(rc) if rc.lstrip("-").isdigit() else rc,
        "walltime_limit": "12:00:00",
        "max_epochs_ziel": 6,
        "slurm_job_id": job_id,
        "git_commit": commit,
        "code_dir": str(exp_dir.parent.parent.parent),
        "exp_dir": str(exp_dir),
        "start_unix": t0,
        "ende_unix": t1,
        "laufzeit_sekunden": secs,
        "laufzeit_lesbar": f"{hh:02d}:{mm:02d}:{ss:02d}",
        "vergleichslauf": "8541310 (SigmaFlow Minimal Changes, 12h)",
        "checkpoint": read_ckpt_meta(exp_dir / "checkpoints" / "last.ckpt"),
    }

    target = exp_dir / "RUN_STATUS.json"
    try:
        target.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except OSError as e:
        print(f"RUN_STATUS.json nicht schreibbar: {e}", file=sys.stderr)
        return 1

    ck = doc["checkpoint"]
    assert isinstance(ck, dict)
    print(f"[STATUS] {status}  (rc={rc}, Laufzeit {doc['laufzeit_lesbar']})")
    print(f"[STATUS] -> {target}")
    print(
        f"[STATUS] Checkpoint: {ck.get('status')}"
        + (
            f", Epoche {ck.get('epoch')}, Schritt {ck.get('global_step')}"
            f"/{ck.get('max_steps_ziel')} ({ck.get('fortschritt_prozent')} %)"
            if ck.get("status") == "OK"
            else ""
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
