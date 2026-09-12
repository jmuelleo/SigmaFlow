#!/usr/bin/env python3
"""
Liest die Kopfdaten eines Lightning-Checkpoints -- Epoche, Schritt, Zielhorizont.

Bewusst winzig und ohne Projektimporte: das Skript laeuft auch aus einer
Signalfalle heraus, wenn SLURM den Job gerade abraeumt und nur noch Sekunden
bleiben. Es darf nie scheitern; im Zweifel gibt es "unbekannt" aus.

    python slurm/read_ckpt_meta.py <pfad/zur/last.ckpt> [--json]
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


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    as_json = "--json" in sys.argv
    if not args:
        print("Aufruf: read_ckpt_meta.py <last.ckpt> [--json]", file=sys.stderr)
        return 2

    path = Path(args[0])
    out: dict[str, object] = {"checkpoint": str(path)}

    if not path.is_file():
        out["status"] = "FEHLT"
        print(json.dumps(out) if as_json else "checkpoint: FEHLT")
        return 0

    out["size_mb"] = round(path.stat().st_size / 1e6, 1)
    _ensure_sigmadock_importable(path.resolve())
    try:
        import torch

        ck = torch.load(path, map_location="cpu", weights_only=False)
        hp = ck.get("hyper_parameters", {}) or {}
        out["status"] = "OK"
        out["epoch"] = ck.get("epoch")
        out["global_step"] = ck.get("global_step")
        out["max_steps_ziel"] = hp.get("max_steps")
        gs, ms = ck.get("global_step"), hp.get("max_steps")
        if isinstance(gs, int) and isinstance(ms, int) and ms > 0:
            out["fortschritt_prozent"] = round(100.0 * gs / ms, 2)
        sd = ck.get("state_dict", {}) or {}
        out["hat_trans_block"] = any(".trans_block." in k for k in sd)
        out["hat_rot_block"] = any(".rot_block." in k for k in sd)
        out["hat_force_block"] = any(".force_block." in k for k in sd)
        out["hat_ema"] = "ema_state_dict" in ck
        out["hat_optimizer"] = bool(ck.get("optimizer_states"))
        out["hat_scheduler"] = bool(ck.get("lr_schedulers"))
    except Exception as e:  # noqa: BLE001 - darf nie hart scheitern
        out["status"] = "LESEFEHLER"
        out["fehler"] = f"{type(e).__name__}: {e}"[:200]

    if as_json:
        print(json.dumps(out, ensure_ascii=False))
    else:
        for k, v in out.items():
            print(f"  {k:22s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
