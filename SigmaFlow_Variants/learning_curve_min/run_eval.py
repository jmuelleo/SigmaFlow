"""Wertet alle Zellen der Minimal-Lernkurve aus, eine je (Snapshot, Schrittzahl).

Ruft SigmaFlow_Evaluation.evaluate_run als Unterprozess auf, damit exakt
derselbe Code laeuft wie auf ARC -- keine zweite, abweichende Implementierung.
"""
import subprocess
import sys
from pathlib import Path

HIER = Path(__file__).resolve().parent
REPO = HIER.parents[1]
TRUE = HIER / "true308"
ZELLEN = sorted((HIER / "poses").iterdir())

for d in ZELLEN:
    if not d.is_dir():
        continue
    cmd = [sys.executable, "-m", "SigmaFlow_Evaluation.evaluate_run",
           "--sampling_root", str(d),
           "--true_dir", str(TRUE),
           "--label", f"min_{d.name}",
           "--out_json", str(d / "evaluation.json"),
           "--per_complex_csv", str(d / "per_complex.csv"),
           "--per_pose_csv", str(d / "per_pose.csv")]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    log = (d / "evaluate.log")
    log.write_text(r.stdout + "\n--- stderr ---\n" + r.stderr, encoding="utf-8")
    status = "ok" if r.returncode == 0 else f"FEHLER rc={r.returncode}"
    print(f"{status:14s} {d.name}")
    if r.returncode != 0:
        print("   " + "\n   ".join(r.stderr.strip().splitlines()[-6:]))
