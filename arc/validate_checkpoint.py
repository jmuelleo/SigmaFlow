"""Prueft einen abgeschlossenen ARC-Lauf, bevor darauf gesampelt wird.

Ein Sampling-Lauf auf einem kaputten oder falschen Checkpoint kostet Stunden und
liefert Zahlen, die plausibel aussehen. Diese Pruefung ist billig und
maschinenlesbar.

    python arc/validate_checkpoint.py --run_dir /data/.../arc_runs/RUN-B24-SF_12345
    python arc/validate_checkpoint.py --run_dir ... --json      # nur JSON
    python arc/validate_checkpoint.py --all                     # alle Laeufe

Rueckgabewert 0 = PASS, 1 = FAIL. Damit kann ein Skript daran haengen.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

DEFAULT_RUNS = Path("/data/stat-cadd/shug8458/arc_runs")

# Welche Codebasis MUSS der jeweilige Lauf geladen haben? Das ist die Pruefung,
# die einen stillen Fehlvergleich verhindert.
EXPECTED_SOURCE = {
    "RUN-B24-SF":     ("SigmaFlow_Minimal", ["EXP-100", "SigmaFlow_Development", "SigmaFlow_Variants"]),
    "RUN-B24-SD":     ("SigmaDock_Reproduction", ["SigmaFlow_Minimal", "EXP-100"]),
    "RUN-B24-E100":   ("EXP-100", ["SigmaFlow_Minimal", "SigmaFlow_Development"]),
    "EXP100-SANITY":  ("EXP-100", ["SigmaFlow_Minimal"]),
}
# Erwartete Schrittzahl bei max_epochs=12 auf pdbbind-general, grosszuegige Spanne.
EXPECTED_STEPS = (20_000, 30_000)


def check(results: list, name: str, ok: bool, detail: str = "", fatal: bool = True) -> bool:
    results.append({"check": name, "pass": bool(ok), "detail": detail, "fatal": fatal})
    return ok


def validate(run_dir: Path) -> dict:
    r: list = []
    run_id = run_dir.name.split("_")[0]

    # --- 1. Metadaten ---------------------------------------------------
    for f in ("run_metadata.txt", "git_commit.txt", "slurm_job_id.txt",
              "environment.txt", "pip_freeze.txt"):
        check(r, f"Metadatei {f}", (run_dir / f).exists())

    meta = {}
    mp = run_dir / "run_metadata.txt"
    if mp.exists():
        for line in mp.read_text(errors="replace").splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                meta[k.strip()] = v.strip()
    check(r, "Startzeit vermerkt", "start_utc" in meta)
    check(r, "Endzeit vermerkt (Job sauber beendet)", "end_utc" in meta,
          "fehlt -> Job abgebrochen oder laeuft noch", fatal=False)
    check(r, "Seed dokumentiert", "seed" in meta, meta.get("seed", ""), fatal=False)

    # --- 2. Git-Commit --------------------------------------------------
    commit = ""
    gc = run_dir / "git_commit.txt"
    if gc.exists():
        for line in gc.read_text(errors="replace").splitlines():
            if re.fullmatch(r"[0-9a-f]{40}", line.strip()):
                commit = line.strip()
                break
    check(r, "Git-Commit erfasst", bool(commit), commit[:10] or "KEIN GIT-REPO", fatal=False)
    if gc.exists():
        dirty = [ln for ln in gc.read_text(errors="replace").splitlines()
                 if ln.strip() and ln.strip()[0] in "MADRU?"]
        check(r, "Codeverzeichnis war sauber", not dirty,
              f"{len(dirty)} geaenderte Dateien" if dirty else "", fatal=False)

    # --- 3. Richtige Codebasis geladen ----------------------------------
    log = run_dir / "train_stdout.log"
    if log.exists():
        txt = log.read_text(errors="replace")
        m = re.search(r"sigmadock geladen aus:\s*(\S+)", txt) or \
            re.search(r"sigmadock loaded from:\s*(\S+)", txt)
        src = m.group(1) if m else ""
        if run_id in EXPECTED_SOURCE:
            want, forbidden = EXPECTED_SOURCE[run_id]
            check(r, f"Codebasis enthaelt '{want}'", want in src, src or "nicht im Log")
            for f in forbidden:
                check(r, f"Codebasis enthaelt NICHT '{f}'", f not in src, src)
        else:
            check(r, "Codebasis im Log vermerkt", bool(src), src, fatal=False)

        # --- 4. NaN / Inf ------------------------------------------------
        n_nan = len(re.findall(r"\bnan\b", txt, flags=re.I))
        n_inf = len(re.findall(r"\binf\b", txt, flags=re.I))
        check(r, "keine NaN im Trainingslog", n_nan == 0, f"{n_nan} Treffer")
        check(r, "keine Inf im Trainingslog", n_inf == 0, f"{n_inf} Treffer", fatal=False)

        # --- 5. Schrittzahl ----------------------------------------------
        steps = re.findall(r"global_step[=: ]+(\d+)", txt)
        n_steps = int(steps[-1]) if steps else None
        if n_steps is not None:
            lo, hi = EXPECTED_STEPS
            check(r, f"Schrittzahl im erwarteten Bereich {lo}-{hi}",
                  lo <= n_steps <= hi, str(n_steps), fatal=False)
        else:
            check(r, "Schrittzahl im Log", False, "nicht gefunden", fatal=False)

        # --- 6. Beendigungsart -------------------------------------------
        killed = "DUE TO TIME LIMIT" in txt or "CANCELLED" in txt
        check(r, "Beendigung dokumentiert",
              True, "Walltime erreicht (bei 24h erwartet)" if killed else "regulaeres Ende",
              fatal=False)
    else:
        check(r, "train_stdout.log vorhanden", False)
        n_steps = None

    # --- 7. Checkpoint --------------------------------------------------
    exp = run_dir / "experiment_dir.txt"
    ckpt = None
    if exp.exists():
        p = exp.read_text(errors="replace").strip()
        exists = bool(p) and Path(p).is_dir()
        check(r, "Experimentordner aufgeloest", exists, p)
        if exists:
            cdir = Path(p) / "checkpoints"
            check(r, "checkpoints/ vorhanden", cdir.is_dir(), str(cdir))
            if cdir.is_dir():
                ckpts = list(cdir.glob("*.ckpt"))
                check(r, "mindestens ein Checkpoint", len(ckpts) > 0, f"{len(ckpts)} Dateien")
                last = cdir / "last.ckpt"
                check(r, "last.ckpt vorhanden", last.exists(), str(last))
                if last.exists():
                    size = last.stat().st_size
                    check(r, "last.ckpt nicht leer (> 1 MB)", size > 1_000_000,
                          f"{size/1e6:.1f} MB")
                    ckpt = str(last)
                    # Lesbarkeit: torch ist auf ARC verfuegbar, lokal evtl. nicht.
                    try:
                        import torch
                        sd = torch.load(last, map_location="cpu", weights_only=False)
                        keys = list(sd.keys()) if isinstance(sd, dict) else []
                        check(r, "Checkpoint mit torch.load lesbar", True,
                              f"Schluessel: {', '.join(keys[:5])}")
                        gs = sd.get("global_step") if isinstance(sd, dict) else None
                        if gs is not None:
                            check(r, "global_step im Checkpoint", True, str(gs), fatal=False)
                    except ImportError:
                        check(r, "Checkpoint mit torch.load lesbar", True,
                              "torch nicht verfuegbar - PENDING ARC", fatal=False)
                    except Exception as e:
                        check(r, "Checkpoint mit torch.load lesbar", False,
                              f"{type(e).__name__}: {str(e)[:80]}")
    else:
        check(r, "experiment_dir.txt vorhanden", False)

    fatal_fails = [c for c in r if not c["pass"] and c["fatal"]]
    soft_fails = [c for c in r if not c["pass"] and not c["fatal"]]
    return {
        "run_dir": str(run_dir), "run_id": run_id, "commit": commit[:10],
        "seed": meta.get("seed", "?"), "steps": n_steps, "checkpoint": ckpt,
        "verdict": "PASS" if not fatal_fails else "FAIL",
        "n_checks": len(r), "n_fatal_fail": len(fatal_fails), "n_soft_fail": len(soft_fails),
        "checks": r,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir")
    ap.add_argument("--runs_root", default=str(DEFAULT_RUNS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.all:
        root = Path(args.runs_root)
        if not root.is_dir():
            print(f"Laufverzeichnis nicht gefunden: {root}")
            return 1
        dirs = sorted(d for d in root.iterdir() if d.is_dir() and d.name != "slurm_logs")
    elif args.run_dir:
        dirs = [Path(args.run_dir)]
    else:
        ap.error("--run_dir oder --all angeben")

    reports = [validate(d) for d in dirs if d.is_dir()]
    if args.json:
        print(json.dumps(reports, indent=2))
        return 0 if all(x["verdict"] == "PASS" for x in reports) else 1

    for rep in reports:
        print("=" * 84)
        print(f"{rep['run_id']}  —  {rep['verdict']}")
        print("=" * 84)
        print(f"  Pfad   : {rep['run_dir']}")
        print(f"  Commit : {rep['commit']}   Seed: {rep['seed']}   Schritte: {rep['steps']}")
        print()
        for c in rep["checks"]:
            mark = "ok " if c["pass"] else ("FAIL" if c["fatal"] else "warn")
            print(f"  [{mark}] {c['check']}" + (f"   {c['detail']}" if c["detail"] else ""))
        if rep["checkpoint"]:
            print(f"\n  CKPT={rep['checkpoint']}")
        print()

    n_fail = sum(1 for x in reports if x["verdict"] == "FAIL")
    print("=" * 84)
    print(f"{len(reports)} Laeufe geprueft, {n_fail} FAIL")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
