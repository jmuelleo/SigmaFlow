"""Recompute RMSD for the four 200-draw cells.

The redock tables and the Vinardo scores already cover 200 seeds; the RMSD
values did not, because they come from evaluate_run over the poses and the
poses for seeds 140 to 199 arrived later. Running the same entry point that
produced every other RMSD in this thesis keeps the convention identical.
"""
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
BASE = HERE / "final200"
CELL = "sched255ep_emergency__nfe5__sampled"

TRUE = {"astex": HERE / "astex" / "astex_diverse_set",
        "pb308": HERE / "learning_curve_min" / "true308"}

CELLS = [
    ("SF_MIN_72H_s0_8653824", "learning_curve_cpu_astex", "astex", "min_ax"),
    ("SF_2H_72H_s0_8668713", "learning_curve_cpu_astex", "astex", "sep_ax"),
    ("SF_MIN_72H_s0_8653824", "learning_curve_cpu", "pb308", "min_pb"),
    ("SF_2H_72H_s0_8668713", "learning_curve_cpu", "pb308", "sep_pb"),
]

for run, tree, ref, label in CELLS:
    d = BASE / run / tree / CELL
    t0 = time.time()
    cmd = [sys.executable, "-m", "SigmaFlow_Evaluation.evaluate_run",
           "--sampling_root", str(d),
           "--true_dir", str(TRUE[ref]),
           "--label", f"{label}_200",
           "--out_json", str(d / "evaluation_200.json"),
           "--per_complex_csv", str(d / "per_complex_200.csv"),
           "--per_pose_csv", str(d / "per_pose_200.csv")]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    (d / "evaluate_200.log").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr,
                                        encoding="utf-8")
    status = "ok" if r.returncode == 0 else f"FAILED rc={r.returncode}"
    print(f"{status:12s} {label:7s} {time.time() - t0:6.0f} s")
    if r.returncode != 0:
        print("   " + "\n   ".join(r.stderr.strip().splitlines()[-6:]))
