"""Top-1 after ranking against the number of drawn poses, both benchmarks.

WHAT THE FIGURE IS FOR
    The thesis compares mechanisms at equal sampling wall-clock, not at equal
    sample count, because the flow arms need five network evaluations per pose
    and the diffusion arm twenty-five. That argument is hard to read off a
    table. Here it is one panel: every arm gets its own curve in the number of
    drawn poses, and the point on each curve that costs the same wall-clock as
    SigmaDock's forty draws at twenty-five steps is marked.

WIE DIE KURVEN GESCHAETZT WERDEN
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
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE / "learning_curve_min"))
from pb_bool import als_bool
from SigmaFlow_Evaluation.ranking.heuristic_score import PB_CHECKS, BETA

# Ausgabeverzeichnis ueber die Umgebung setzbar, damit die Abbildung nicht
# zwingend in draft3 landet -- der Entwurf wandert.
import os
TARGET = pathlib.Path(os.environ.get(
    "PLOT_ZIEL",
    HERE.parent / "Thesis_Draft_Proposal" / "26.08.2026" / "draft3"))
LOADED = {"mol_pred_loaded", "mol_true_loaded", "mol_cond_loaded"}

# Anzeigenamen der Arbeit. Die SCHLUESSEL bleiben "SigmaFlow-Minimal"
# und "SigmaFlow-Separate" -- sie adressieren Zellen, Pfade und Farben.
# Nur die Beschriftung wechselt, damit Abbildung und Text dieselben
# Namen fuehren.
ANZEIGE = {"SigmaFlow-Minimal": "SigmaFlow-NE",
           "SigmaFlow-Separate": "SigmaFlow-TR",
           "SigmaDock": "SigmaDock"}
def anz(a):
    return ANZEIGE.get(a, a)

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
        t.drop_duplicates(["complex", "seed"], keep=False, inplace=True)
    m = d.merge(rmsd_df, on=["complex", "seed"]).merge(
        gnina_df[["complex", "seed", "affinity"]], on=["complex", "seed"])
    # TOTER CODE -- die Zellen kommen aus final_tables.build(). Hier nur
    # mitgezogen, damit diese Kopie nicht spaeter still den unkorrigierten
    # RMSD aus per_pose.csv wiederbelebt; siehe plot_ranking_lib._finish.
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
    return d[["complex", "seed", "valid", "p_pb"]]


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


MC, MC_SEED = 1000, 20260830


def curves(m, target, ks):
    """Top-1 und Orakel gegen K aus den EINGEFROREN Zahlen.

    Frueher zog diese Funktion ihre eigenen MC-Teilmengen mit eigenem Seed.
    Jede Zahl der Arbeit zu den 72-h-Zellen kommt jetzt aus
    ergebnisse_72h_*.csv (B = 10.000, Seed 20260910), damit Abbildung, Tabelle
    und Test nicht um den Monte-Carlo-Fehler auseinanderliegen.

    Fehlt ein K in der Datei, wird ABGEBROCHEN statt hier nachzurechnen --
    dann gehoert das K in die KS-Liste von ergebnisse_72h.py.
    """
    import ergebnisse
    import numpy as np
    schl = m.attrs["zelle"]
    top = np.array([ergebnisse.wert(*schl, K, target, "mixed") for K in ks])
    orc = np.array([ergebnisse.wert(*schl, K, target, "orakel") for K in ks])
    return top, orc


print("loading cells")
import final_tables as ft
data = ft.build()
for k in sorted(data):
    print(f"  {k}  {len(data[k])} poses, {data[k].seed.nunique()} seeds")

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.labelcolor": INK, "text.color": INK,
})
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex=True)
ROWS = [("both", r"RMSD $<$ 2 Å and PB-valid"), ("acc", r"RMSD $<$ 2 Å")]
COLS = [("PB308", "PoseBusters, 308 complexes"), ("AX85", "Astex, 85 complexes")]

for r, (target, rowname) in enumerate(ROWS):
    for c, (bench, colname) in enumerate(COLS):
        ax = axes[r][c]
        for arm in COLOUR:
            for nfe in (25, 5):
                m = data[(bench, arm, nfe)]
                nmax = int(m.seed.nunique())
                ks = [k for k in (1, 2, 3, 5, 8, 12, 20, 30, 40, 60, 80, 100,
                                  101, 120, 140, 170, 200) if k <= nmax]
                top, _ = curves(m, target, ks)
                ax.plot(ks, top, STYLE[nfe], color=COLOUR[arm], linewidth=1.5,
                        zorder=3)
                # the draw count that costs what SigmaDock's 40 at 25 steps cost
                mk = 40 if nfe == 25 else MATCHED[bench]
                if mk in ks:
                    ax.plot([mk], [top[ks.index(mk)]], "o", color=COLOUR[arm],
                            markersize=5.5, markeredgecolor="white",
                            markeredgewidth=1.2, zorder=5)
        ref = PUBLISHED[(bench, target)]
        ax.axhline(ref, color=INK2, linewidth=0.7, linestyle=(0, (1, 2)),
                   zorder=1)
        ax.annotate(f"published SigmaDock  {ref}", (1.05, ref), fontsize=6.5,
                    color=INK2, va="bottom", ha="left")
        ax.set_xscale("log")
        # Beschriftet werden nur Ziehungszahlen, die etwas bedeuten. Die
        # logarithmische Achse setzt sonst unbeschriftete Zwischenstriche,
        # die nichts bezeichnen.
        #
        # 100, 140 und 200 liegen logarithmisch so dicht, dass ihre
        # Beschriftungen aneinanderstossen. Die 140, der walltime-gematchte
        # Punkt auf PB308, kommt deshalb in eine zweite Zeile darunter,
        # statt eine der Marken zu opfern.
        OBEN, UNTEN = [1, 5, 10, 40, 100, 200], [140]
        ax.xaxis.set_major_locator(mticker.FixedLocator(OBEN))
        ax.xaxis.set_major_formatter(
            mticker.FixedFormatter([str(k) for k in OBEN]))
        ax.xaxis.set_minor_locator(mticker.FixedLocator(UNTEN))
        ax.xaxis.set_minor_formatter(
            mticker.FixedFormatter([str(k) for k in UNTEN]))
        ax.tick_params(axis="x", which="minor", pad=11, labelsize=7.0,
                       length=3)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if r == 0:
            ax.set_title(colname, fontsize=8.5, color=INK, pad=7)
        if r == 1:
            ax.set_xlabel("Poses drawn per complex")
        if c == 0:
            ax.set_ylabel(rowname + "\nTop-1 (%)")
    lo = min(axes[r][c].get_ylim()[0] for c in (0, 1))
    hi = max(axes[r][c].get_ylim()[1] for c in (0, 1))
    for c in (0, 1):
        axes[r][c].set_ylim(lo, hi)

hand = [plt.Line2D([], [], color=COLOUR[a], linestyle=STYLE[n], linewidth=1.5,
                   label=f"{anz(a)}, {n} steps")
        for a in COLOUR for n in (25, 5)]
hand.append(plt.Line2D([], [], color=INK2, marker="o", linestyle="none",
                       markersize=5.5, markeredgecolor="white",
                       markeredgewidth=1.2, label="equal sampling wall-clock"))
fig.legend(handles=hand, loc="lower center", ncol=4, frameon=False,
           fontsize=7.2, bbox_to_anchor=(0.5, -0.10), handlelength=2.4)
fig.tight_layout(rect=(0, 0.07, 1, 1))
for ext in ("pdf", "png"):
    fig.savefig(TARGET / f"ranking_curves.{ext}", dpi=300, bbox_inches="tight")
print("written:", TARGET / "ranking_curves.pdf")
