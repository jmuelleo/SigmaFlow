"""Per-draw rates for every arm, which is also Top-1 at K = 1."""
import glob
import importlib
import pathlib

import pandas as pd

import final_tables as ft

pr = importlib.import_module("plot_ranking_lib")
CELL = "sched255ep_emergency__nfe5__sampled"


def maybe200(run, tree, redock_glob, gnina, fallback):
    """The 200-draw cell if its RMSD has been recomputed, else the old one."""
    pp = ft.FIN / run / tree / CELL / "per_pose_200.csv"
    if pp.exists():
        return ft.cell200(run, tree, redock_glob, gnina), True
    return fallback(), False


CASES = []
for bench in ("PB308", "AX85"):
    for arm in ("SigmaDock", "SigmaFlow-Minimal", "SigmaFlow-Separate"):
        for nfe in (25, 5):
            CASES.append((bench, arm, nfe))

rows = []
for bench, arm, nfe in CASES:
    fresh = False
    if nfe == 5 and arm != "SigmaDock":
        run = ("SF_MIN_72H_s0_8653824" if "Minimal" in arm
               else "SF_2H_72H_s0_8668713")
        if bench == "AX85":
            rd = ("sigmaflow_minimal__nfe5" if "Minimal" in arm
                  else "exp110__nfe5")
            gn = glob.glob(str(ft.FIN / ("GNINA-SCORE-" + (
                "sigmaflow_minimal" if "Minimal" in arm else "exp110")
                + "__nfe5__astex_*") / "gnina_scores_*.csv"))[0]
            m, fresh = maybe200(run, "learning_curve_cpu_astex",
                                str(ft.FIN / "astex_redock" / rd / "rd_*_seed*.csv"),
                                gn,
                                lambda: pr.load_ax(*pr.AX_CELLS[(bench, arm, nfe)]))
        else:
            m, fresh = maybe200(
                run, "learning_curve_cpu",
                str(ft.FIN / run / "posebusters_redock_curve" / CELL / "rd_*_seed*.csv"),
                ft.FIN / run / "learning_curve_cpu" / CELL / "gnina_scores_200.csv",
                lambda: pr.load_pb(*pr.CELLS[(bench, arm, nfe)]))
    else:
        m = (pr.load_pb(*pr.CELLS[(bench, arm, nfe)]) if bench == "PB308"
             else pr.load_ax(*pr.AX_CELLS[(bench, arm, nfe)]))
    both = 100 * m.groupby("complex")["both"].mean().mean()
    acc = 100 * m.groupby("complex")["acc"].mean().mean()
    rows.append((bench, arm, nfe, m.seed.nunique(), both, acc, fresh))

print(f"{'benchmark':<8}{'arm':<20}{'steps':>6}{'pool':>6}"
      f"{'both':>9}{'RMSD<2':>9}   source")
for bench, arm, nfe, pool, both, acc, fresh in rows:
    tag = "200-draw pool" if fresh else ("40 draws" if nfe == 25 or arm == "SigmaDock"
                                         else "140-draw pool, pending")
    print(f"{bench:<8}{arm:<20}{nfe:>6}{pool:>6}{both:>9.2f}{acc:>9.2f}   {tag}")
