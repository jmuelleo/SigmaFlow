"""Sammelt ein, was auf ARC entstanden ist. Interpretiert nichts.

Der Zweck ist bewusst eng: nach mehreren Laeufen soll EIN Aufruf zeigen, was
existiert, was fehlt und ob die Metadaten vollstaendig sind. Wissenschaftliche
Schluesse zieht dieses Skript nicht - die stehen in den Auswertungslogs.

    python arc/summarize_arc_results.py
    python arc/summarize_arc_results.py --runs_root /data/stat-cadd/shug8458/arc_runs
"""

import argparse
import os
import re
from pathlib import Path

REQUIRED = ["run_metadata.txt", "git_commit.txt", "slurm_job_id.txt",
            "environment.txt", "pip_freeze.txt"]


def read_kv(path):
    out = {}
    try:
        for line in Path(path).read_text(errors="replace").splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    except Exception:
        pass
    return out


def summarize_run(d: Path) -> dict:
    meta = read_kv(d / "run_metadata.txt")
    missing = [f for f in REQUIRED if not (d / f).exists()]

    commit = "?"
    gc = d / "git_commit.txt"
    if gc.exists():
        for line in gc.read_text(errors="replace").splitlines():
            if re.fullmatch(r"[0-9a-f]{40}", line.strip()):
                commit = line.strip()[:10]
                break

    ckpt_dir, n_ckpt, has_last = None, 0, False
    exp = d / "experiment_dir.txt"
    if exp.exists():
        p = exp.read_text(errors="replace").strip()
        if p and Path(p).is_dir():
            ckpt_dir = Path(p) / "checkpoints"
            if ckpt_dir.is_dir():
                n_ckpt = len(list(ckpt_dir.glob("*.ckpt")))
                has_last = (ckpt_dir / "last.ckpt").exists()

    log = d / "train_stdout.log"
    n_nan = steps = None
    if log.exists():
        try:
            txt = log.read_text(errors="replace")
            n_nan = len(re.findall(r"\bnan\b", txt, flags=re.I))
            m = re.findall(r"global_step[=: ]+(\d+)", txt)
            steps = int(m[-1]) if m else None
        except Exception:
            pass

    return {
        "run": d.name, "run_id": meta.get("run_id", "?"),
        "job": meta.get("slurm_job_id", "?"), "commit": commit,
        "start": meta.get("start_utc", "?")[:16], "end": meta.get("end_utc", "-")[:16],
        "node": meta.get("node", "?"), "seed": meta.get("seed", "?"),
        "budget": meta.get("budget", "-"),
        "n_ckpt": n_ckpt, "last": has_last, "nan": n_nan, "steps": steps,
        "missing": missing,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", default="/data/stat-cadd/shug8458/arc_runs")
    args = ap.parse_args()

    root = Path(args.runs_root)
    if not root.is_dir():
        print(f"Laufverzeichnis nicht gefunden: {root}")
        return

    runs = sorted([d for d in root.iterdir() if d.is_dir() and d.name != "slurm_logs"])
    if not runs:
        print(f"Keine Laeufe unter {root}")
        return

    rows = [summarize_run(d) for d in runs]

    print("=" * 118)
    print(f"ARC-LAEUFE unter {root}")
    print("=" * 118)
    hdr = f"{'run_id':<16}{'job':>10}{'commit':>12}{'start':>18}{'seed':>6}{'ckpt':>6}{'last':>6}{'steps':>9}{'NaN':>6}"
    print(hdr)
    print("-" * 118)
    for r in rows:
        print(f"{r['run_id']:<16}{r['job']:>10}{r['commit']:>12}{r['start']:>18}"
              f"{r['seed']:>6}{r['n_ckpt']:>6}{'ja' if r['last'] else 'NEIN':>6}"
              f"{str(r['steps'] or '-'):>9}{str(r['nan'] if r['nan'] is not None else '-'):>6}")

    print()
    print("VOLLSTAENDIGKEIT DER METADATEN")
    any_missing = False
    for r in rows:
        if r["missing"]:
            any_missing = True
            print(f"  {r['run_id']:<16} fehlt: {', '.join(r['missing'])}")
    if not any_missing:
        print("  alle Laeufe vollstaendig")

    print()
    print("WARNUNGEN")
    warn = False
    for r in rows:
        if not r["last"] and r["n_ckpt"] == 0:
            print(f"  {r['run_id']:<16} KEIN Checkpoint - Lauf unbrauchbar")
            warn = True
        if r["nan"]:
            print(f"  {r['run_id']:<16} {r['nan']} NaN-Treffer im Trainingslog - pruefen")
            warn = True
        if r["end"] == "-":
            print(f"  {r['run_id']:<16} kein end_utc - Job vermutlich abgebrochen oder laeuft noch")
            warn = True
    if not warn:
        print("  keine")

    print()
    print("BUDGETVERGLEICH (Fairness der 24h-Laeufe)")
    b24 = [r for r in rows if r["run_id"].startswith("RUN-B24")]
    if len(b24) < 2:
        print("  weniger als zwei B24-Laeufe vorhanden - Vergleich noch nicht moeglich")
    else:
        for r in b24:
            print(f"  {r['run_id']:<16} budget={r['budget']:<45} steps={r['steps']} seed={r['seed']}")
        budgets = {r["budget"] for r in b24}
        seeds = {r["seed"] for r in b24}
        if len(budgets) > 1:
            print("  [!] UNTERSCHIEDLICHE BUDGETS - der Vergleich waere unfair")
        if len(seeds) > 1:
            print("  [!] UNTERSCHIEDLICHE SEEDS - bewusst so gewollt? sonst unfair")
        if len(budgets) == 1 and len(seeds) == 1:
            print("  Budget und Seed stimmen ueberein.")

    print()
    print("CHECKPOINTS ZUM SAMPELN")
    for r in rows:
        d = root / r["run"]
        exp = d / "experiment_dir.txt"
        if exp.exists():
            p = exp.read_text(errors="replace").strip()
            if p and (Path(p) / "checkpoints" / "last.ckpt").exists():
                print(f"  {r['run_id']:<16} CKPT={p}/checkpoints/last.ckpt")


if __name__ == "__main__":
    main()
