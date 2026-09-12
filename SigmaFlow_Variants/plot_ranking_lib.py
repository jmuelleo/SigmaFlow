"""Top-1 after ranking against the number of drawn poses, both benchmarks.

WHAT THE FIGURE IS FOR
    The thesis compares mechanisms at equal sampling wall-clock, not at equal
    sample count, because the flow arms need five network evaluations per pose
    and the diffusion arm twenty-five. That argument is hard to read off a
    table. Here it is one panel: every arm gets its own curve in the number of
    drawn poses, and the point on each curve that costs the same wall-clock as
    SigmaDock's forty draws at twenty-five steps is marked.

WHY THE CURVES ARE EXACT AND NOT RESAMPLED
    Top-1 at K draws is an expectation over K-subsets, and it has a closed
    form. Rank the n poses of a complex by the mixed score, best first. The
    pose at rank i is the maximum of a uniformly drawn K-subset exactly when i
    is in the subset and none of the i better ones are, which happens with
    probability C(n-1-i, K-1) / C(n, K). Summing that against the success
    indicator gives Top-1(K) for the complex with no sampling error. The
    oracle is 1 - C(n-s, K)/C(n, K) with s successes. Averaging over complexes
    then gives the curve. A Monte Carlo estimate would need thousands of
    subsets per point to reach the same smoothness and would still wobble.

ENCODING
    Colour is the arm, in the fixed order used everywhere else in the thesis;
    line style is the number of integration steps. Identity therefore never
    rests on colour alone. Rows are the two success criteria, columns the two
    benchmarks, and the y-axis is shared within a row so that the two
    benchmarks may be compared directly.
"""
import glob
import math
import pathlib
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = pathlib.Path(".").resolve()
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE / "learning_curve_min"))
from pb_bool import als_bool
from SigmaFlow_Evaluation.ranking.heuristic_score import PB_CHECKS, BETA

TARGET = HERE.parent / "Thesis_Draft_Proposal" / "26.08.2026" / "draft3"
LOADED = {"mol_pred_loaded", "mol_true_loaded", "mol_cond_loaded"}

COLOUR = {"SigmaFlow-Minimal": "#2a78d6",
          "SigmaFlow-Separate": "#eb6834",
          "SigmaDock": "#1baf7a"}
STYLE = {25: "-", 5: "--"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

# Published SigmaDock, Nseeds = 40 at 25 steps (Prat et al., Fig. 4).
PUBLISHED = {("PB308", "both"): 79.9, ("PB308", "acc"): 80.5,
             ("AX85", "both"): 90.6, ("AX85", "acc"): 90.6}
# Draws that cost the same sampling wall-clock as SigmaDock's 40 at 25 steps.
MATCHED = {"PB308": 140, "AX85": 101}

PB = HERE / "pb308_endpoints"
AX = HERE / "astex"

CELLS = {
 ("PB308", "SigmaDock", 25): (PB / "sd_endpunkt_40seeds", "sched239ep_emergency__nfe25__sampled"),
 ("PB308", "SigmaDock", 5): (PB / "sd_endpunkt_40seeds", "sched239ep_emergency__nfe5__sampled"),
 ("PB308", "SigmaFlow-Minimal", 25): (PB / "endpunkt_min_nfe25", "sched255ep_emergency__nfe25__sampled"),
 ("PB308", "SigmaFlow-Minimal", 5): (PB / "endpunkt_min_nfe5", "sched255ep_emergency__nfe5__sampled"),
 ("PB308", "SigmaFlow-Separate", 25): (PB / "endpunkt_sep_nfe25", "sched255ep_emergency__nfe25__sampled"),
 ("PB308", "SigmaFlow-Separate", 5): (PB / "endpunkt_sep_nfe5", "sched255ep_emergency__nfe5__sampled"),
}

AX_CELLS = {
 ("AX85", "SigmaDock", 25): ("sigmadock",
    "GNINA-SCORE-sigmadock__astex_8680368/gnina_scores_sigmadock__astex.csv",
    "SD_BASE_72H_s0_8648493/learning_curve_cpu_astex/sched239ep_emergency__nfe25__sampled"),
 ("AX85", "SigmaDock", 5): ("sigmadock__nfe5",
    "GNINA-SCORE-sigmadock__nfe5__astex_8681056/gnina_scores_sigmadock__nfe5__astex.csv",
    "SD_BASE_72H_s0_8648493/learning_curve_cpu_astex/sched239ep_emergency__nfe5__sampled"),
 ("AX85", "SigmaFlow-Minimal", 25): ("sigmaflow_minimal",
    "GNINA-SCORE-sigmaflow_minimal__astex_8680369/gnina_scores_sigmaflow_minimal__astex.csv",
    "SF_MIN_72H_s0_8653824/learning_curve_cpu_astex/sched255ep_emergency__nfe25__sampled"),
 ("AX85", "SigmaFlow-Minimal", 5): ("sigmaflow_minimal__nfe5",
    "GNINA-SCORE-sigmaflow_minimal__nfe5__astex_8685385/gnina_scores_sigmaflow_minimal__nfe5__astex.csv",
    "SF_MIN_72H_s0_8653824/learning_curve_cpu_astex/sched255ep_emergency__nfe5__sampled"),
 ("AX85", "SigmaFlow-Separate", 25): ("exp110",
    "GNINA-SCORE-exp110__astex_8680370/gnina_scores_exp110__astex.csv",
    "SF_2H_72H_s0_8668713/learning_curve_cpu_astex/sched255ep_emergency__nfe25__sampled"),
 ("AX85", "SigmaFlow-Separate", 5): ("exp110__nfe5",
    "GNINA-SCORE-exp110__nfe5__astex_8681055/gnina_scores_exp110__nfe5__astex.csv",
    "SF_2H_72H_s0_8668713/learning_curve_cpu_astex/sched255ep_emergency__nfe5__sampled"),
}


def _finish(d, rmsd_df, gnina_df):
    """Attach RMSD and affinity, form the two success flags and the score."""
    for t in (d, rmsd_df, gnina_df):
        t.drop_duplicates(["complex", "seed"], keep="first", inplace=True)
    m = d.merge(rmsd_df, on=["complex", "seed"]).merge(
        gnina_df[["complex", "seed", "affinity"]], on=["complex", "seed"])
    # rmsd bleibt als ZAHL erhalten (Schwellenvergleiche brauchen sie),
    # das Erfolgsetikett kommt aber aus dem symmetriekorrigierten Redock.
    m["acc"] = m["acc_redock"] if "acc_redock" in m.columns else m["rmsd"] < 2.0
    m["both"] = m["acc"] & m["valid"]
    m["heur"] = -m["affinity"] * (m["p_pb"] ** BETA)
    return m


def _redock(files):
    parts = []
    for f in sorted(files):
        t = pd.read_csv(f)
        t["seed"] = int(re.search(r"seed(\d+)\.csv$", f).group(1))
        parts.append(t)
    d = pd.concat(parts, ignore_index=True)
    d["complex"] = d["file"].map(lambda p: pathlib.Path(p).name.split("__")[0])
    rmsd_col = next(c for c in d.columns if c.startswith("rmsd"))
    checks = [c for c in d.columns
              if c not in LOADED | {"file", "molecule", "position", "seed",
                                    "complex", rmsd_col}]
    for c in checks:
        d[c] = als_bool(d[c])
    d["valid"] = d[checks].all(axis=1)
    d["p_pb"] = d[list(PB_CHECKS)].mean(axis=1)
    # Der RMSD MUSS von hier kommen, nicht aus per_pose.csv.
    # per_pose.csv fuehrt den rohen RMSD des Samplers, OHNE
    # Symmetriekorrektur; der Redock fuehrt den korrigierten. Weil die
    # Korrektur einen RMSD nur senken kann, zaehlte die alte Fassung
    # systematisch zu wenige Posen als richtig -- gemessen 709 von 61150
    # Posen einer Zelle, alle 709 in dieselbe Richtung. Gegen den
    # unabhaengig gerechneten rmsd_streng.py stimmt diese Spalte auf
    # 172480 Posen zu 100,000 % ueberein.
    d["acc_redock"] = als_bool(d[rmsd_col])
    # Ein Komplex steht je Seed DOPPELT in der Redock-Tabelle. Frueher warf
    # _finish beide Kopien weg (keep=False) und verlor damit echte Posen;
    # zellen.py behaelt die erste nach ("seed", "file") und ist die Kette,
    # die in ZIEL 1 die publizierten Zahlen reproduziert hat. Gleich ziehen.
    d = d.sort_values(["seed", "file"]).drop_duplicates(["complex", "seed"],
                                                        keep="first")
    return d[["complex", "seed", "valid", "p_pb", "acc_redock"]]


def load_pb(root, cell):
    rd = [p for p in root.rglob("posebusters_redock_curve/" + cell)][0]
    cd = [p for p in root.rglob("learning_curve_cpu/" + cell)][0]
    return _finish(_redock(glob.glob(str(rd / "rd_*_seed*.csv"))),
                   pd.read_csv(cd / "per_pose.csv"),
                   pd.read_csv(cd / "gnina_scores.csv"))


def load_ax(redock_dir, gnina_rel, pose_rel):
    return _finish(_redock(glob.glob(str(AX / "astex_redock" / redock_dir / "rd_*_seed*.csv"))),
                   pd.read_csv(AX / pose_rel / "per_pose.csv"),
                   pd.read_csv(AX / gnina_rel))


def logC(a, b):
    if b < 0 or b > a:
        return -math.inf
    return (math.lgamma(a + 1) - math.lgamma(b + 1) - math.lgamma(a - b + 1))


# ACHTUNG: GESCHLOSSENE FORM und UNBENUTZT. Die Kurven rechnet
# curves() mit 1000 Teilmengen je Komplex; in diesem Projekt ist der
# gezogene Schaetzer festgelegt. Diese Funktion nicht wiederbeleben,
# ohne sie auf zieh.top1_je_komplex umzustellen.
def per_complex(m, target, K):
    """Expected Top-1 success probability of each complex at K draws.

    A complex with fewer than K poses contributes its whole pool, which is
    what drawing K times from it would give. Averaging the returned series
    gives the curve value; keeping it per complex allows a paired test.
    """
    out = {}
    for cid, g in m.groupby("complex"):
        g = g.sort_values("heur", ascending=False)
        y = g[target].to_numpy().astype(float)
        n = len(y)
        k = min(K, n)
        den = logC(n, k)
        w = np.array([math.exp(logC(n - 1 - i, k - 1) - den)
                      for i in range(n - k + 1)])
        out[cid] = float(y[:len(w)] @ w)
    return pd.Series(out)


def curves(m, target, ks):
    """Exact Top-1 and oracle against K, averaged over complexes."""
    top, orc = [], []
    groups = list(m.groupby("complex"))
    for K in ks:
        t = o = 0.0
        for _, g in groups:
            g = g.sort_values("heur", ascending=False)
            y = g[target].to_numpy().astype(float)
            n, s = len(y), int(y.sum())
            k = min(K, n)
            den = logC(n, k)
            w = np.array([math.exp(logC(n - 1 - i, k - 1) - den)
                          for i in range(n - k + 1)])
            t += float(y[:len(w)] @ w)
            o += 1.0 - (math.exp(logC(n - s, k) - den) if n - s >= k else 0.0)
        top.append(100 * t / len(groups))
        orc.append(100 * o / len(groups))
    return np.array(top), np.array(orc)


