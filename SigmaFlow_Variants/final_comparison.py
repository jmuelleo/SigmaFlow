"""The headline comparison table, the trajectory over K, and two bar charts.

ESTIMATOR
    Every reported number is estimated from B = 1000 random subsets. For one
    complex, draw a uniform K-subset of its generated poses, let the ranker
    pick the best pose in that subset, record whether it satisfies the
    criterion. Average over the 1000 subsets to get the complex's rate, then
    average over complexes. K = 1 uses the same procedure.

    One permutation matrix per complex serves every K: the first k columns of
    a uniform random permutation are a uniform k-subset, so each column block
    is a valid 1000-subset estimate for its K. Estimates for different K
    within a complex are therefore correlated, which is harmless for point
    estimates, makes the trajectory over K read smoothly, and avoids sorting
    the same array once per K.

    The Monte Carlo error is reported alongside, since these numbers carry one
    and the closed form does not. --exact switches to the closed form.

THE CLAMP TRAP
    ft.rule silently uses k = min(K, n). Asking for K = 200 of a 40-seed cell
    would return the K = 40 number and the table would show a fabricated
    value. Every cell is guarded by an explicit seed check and prints an em
    dash where the draws were never generated.
"""
import argparse
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import final_tables as ft

HERE = pathlib.Path(".").resolve()
TARGET = HERE.parent / "Thesis_Draft_Proposal" / "26.08.2026" / "draft3"

# Anzeigenamen der Arbeit. Die SCHLUESSEL bleiben "SigmaFlow-Minimal"
# und "SigmaFlow-Separate" -- sie adressieren Zellen, Pfade und Farben.
# Nur die Beschriftung wechselt, damit Abbildung und Text dieselben
# Namen fuehren.
ANZEIGE = {"SigmaFlow-Minimal": "SigmaFlow-NE",
           "SigmaFlow-Separate": "SigmaFlow-TR",
           "SigmaDock": "SigmaDock"}
def anz(a):
    return ANZEIGE.get(a, a)

ARMS = ["SigmaDock", "SigmaFlow-Minimal", "SigmaFlow-Separate"]
CRIT = [("acc", "RMSD < 2"), ("both", "RMSD < 2 & PB-valid")]
MATCHED = {"PB308": 140, "AX85": 101}
BENCH_LABEL = {"PB308": "PoseBusters, 307 complexes",
               "AX85": "Astex Diverse Set, 85 complexes"}
# Ueber dem Bild steht der Satzname, die Zaehlangaben stehen in der
# Bildunterschrift: die publizierten Werte gelten fuer 308 Komplexe,
# unsere Zellen decken 307 ab.
BILD_TITEL = {"PB308": "PoseBusters, 308-complex set",
              "AX85": "Astex Diverse Set"}
KS = {25: [1, 5, 10, 20, 40], 5: [1, 5, 10, 20, 40, 100, 140, 200]}
B = 1000
SEED = 20260830

COLOUR = {"SigmaFlow-Minimal": "#2a78d6", "SigmaFlow-Separate": "#eb6834",
          "SigmaDock": "#1baf7a"}
LIT = "#8d8b86"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

# Prat et al. (2026), Figure 4. Nseeds = 40, 25 integration steps.
PAPER = {
    "TankBind":  {"PB308": (15.9, 3.2),  "AX85": (59.0, 5.9)},
    "DiffDock":  {"PB308": (38.0, 12.7), "AX85": (72.0, 47.0)},
    "UniMol":    {"PB308": (21.8, 1.9),  "AX85": (45.0, 12.0)},
    "DeepDock":  {"PB308": (19.5, 5.2),  "AX85": (35.0, 11.0)},
    "Re-Dock":   {"PB308": (50.7, 32.8), "AX85": None},
    "Gold":      {"PB308": (58.1, 54.5), "AX85": (67.0, 64.0)},
    "Vina":      {"PB308": (59.7, 58.1), "AX85": (58.0, 56.0)},
    "SigmaDock (paper)": {"PB308": (80.5, 79.9), "AX85": (90.6, 90.6)},
}
PAPER_ORDER = list(PAPER)
HOLO = {"TankBind", "DiffDock", "UniMol"}


def estimate_cell(m, ks, rng):
    """Liest die EINGEFRORENEN Zahlen (B = 10.000) statt selbst zu ziehen.

    Frueher zog diese Funktion ihre eigenen 1000 Teilmengen mit eigenem Seed.
    Damit standen dieselben Groessen in Tabelle und Abbildung minimal
    verschieden. Jetzt kommt jede Zahl aus ergebnisse_72h_*.csv. Der zweite
    Rueckgabewert war der Monte-Carlo-Fehler; er ist hier null, weil nicht
    mehr gezogen wird.
    """
    import ergebnisse
    schl = m.attrs["zelle"]
    n_seeds = m["seed"].nunique()
    return {K: {t: (ergebnisse.wert(*schl, K, t), 0.0) for t, _ in CRIT}
            for K in ks if K <= n_seeds}


def exact_cell(m, ks):
    n_seeds = m["seed"].nunique()
    return {K: {t: (100.0 * ft.rule(m, t, K, "heuristic", exakt=True).mean(), 0.0)
                for t, _ in CRIT}
            for K in ks if K <= n_seeds}


def compute(data, exact):
    rng = np.random.default_rng(SEED)
    res = {}
    for bench in ("PB308", "AX85"):
        for arm in ARMS:
            for nfe in (25, 5):
                m = data[(bench, arm, nfe)]
                ks = sorted(set(KS[nfe]) | {40, MATCHED[bench], 200})
                res[(bench, arm, nfe)] = (exact_cell(m, ks) if exact
                                          else estimate_cell(m, ks, rng))
    return res


def get(res, bench, arm, nfe, K, target):
    cell = res[(bench, arm, nfe)]
    return cell[K][target][0] if K in cell else None


def err(res, bench, arm, nfe, K, target):
    cell = res[(bench, arm, nfe)]
    return cell[K][target][1] if K in cell else None


def fmt(v):
    return "  --  " if v is None else f"{v:6.1f}"


def headline_rows(res):
    out = []
    for nfe, ks in ((25, {"PB308": 40, "AX85": 40}),
                    (5, {"PB308": 40, "AX85": 40}),
                    (5, dict(MATCHED)),
                    (5, {"PB308": 200, "AX85": 200})):
        for arm in ARMS:
            cells = {b: (K, {t: get(res, b, arm, nfe, K, t) for t, _ in CRIT})
                     for b, K in ks.items()}
            if all(v is None for _, vs in cells.values() for v in vs.values()):
                continue
            out.append((arm, nfe, cells))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exact", action="store_true",
                    help="closed form instead of the 1000 random subsets")
    args = ap.parse_args()

    data = ft.build()
    res = compute(data, args.exact)
    how = ("closed form, no sampling error" if args.exact
           else "frozen values from ergebnisse_72h_*.csv (B = 10000)")

    print("=" * 98)
    print(f"TABLE 1  headline comparison. Top-1 (%). Estimator: {how}.")
    print("=" * 98)
    print(f"{'method':<26}{'steps':>6}{'draws':>9}"
          f"{'PB RMSD<2':>11}{'PB both':>9}{'AX RMSD<2':>11}{'AX both':>9}")
    for name in PAPER_ORDER:
        pb, ax = PAPER[name]["PB308"], PAPER[name]["AX85"]
        tag = name + (" (holo)" if name in HOLO else "")
        print(f"{tag:<26}{25:>6}{40:>9}{pb[0]:>11.1f}{pb[1]:>9.1f}"
              + (f"{ax[0]:>11.1f}{ax[1]:>9.1f}" if ax else f"{'--':>11}{'--':>9}"))
    print("-" * 98)
    for arm, nfe, cells in headline_rows(res):
        kpb, vpb = cells["PB308"]
        kax, vax = cells["AX85"]
        draws = f"{kpb}" if kpb == kax else f"{kpb}/{kax}"
        print(f"{arm + ' (ours)':<26}{nfe:>6}{draws:>9}"
              f"{fmt(vpb['acc']):>11}{fmt(vpb['both']):>9}"
              f"{fmt(vax['acc']):>11}{fmt(vax['both']):>9}")

    print()
    print("=" * 98)
    print("TABLE 2  trajectory over K, the three 72 h runs.")
    print("=" * 98)
    for bench in ("PB308", "AX85"):
        print(f"\n{BENCH_LABEL[bench]}")
        for target, cname in CRIT:
            print(f"\n  {cname}")
            for nfe in (25, 5):
                ks = KS[nfe]
                print(f"    {'K =':<30}" + "".join(f"{k:>8}" for k in ks))
                for arm in ARMS:
                    vals = [get(res, bench, arm, nfe, k, target) for k in ks]
                    print(f"    {arm + f', {nfe} steps':<30}"
                          + "".join(f"{fmt(v):>8}" for v in vals))

    if not args.exact:
        ses = [v[1] for cell in res.values() for ks in cell.values()
               for v in ks.values()]
        print(f"\nMonte Carlo error over all {len(ses)} estimated cells: "
              f"largest {max(ses):.3f}, median {np.median(ses):.3f} points.")
        print("At K equal to the number of seeds there is only one subset, so "
              "those cells carry no sampling error.")

    bars(res)


def bars(res):
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8.5,
        "axes.edgecolor": INK2, "axes.linewidth": 0.6,
        "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.labelcolor": INK, "text.color": INK,
    })
    for bench in ("PB308", "AX85"):
        entries = []
        for name in PAPER_ORDER:
            p = PAPER[name][bench]
            if p is None:
                continue
            entries.append((name + ("*" if name in HOLO else ""), p[0], p[1],
                            COLOUR["SigmaDock"] if "paper" in name else LIT,
                            "paper" in name))
        for arm, nfe, cells in headline_rows(res):
            K, v = cells[bench]
            if v["acc"] is None:
                continue
            entries.append((f"{anz(arm)}, {nfe} steps, K={K}", v["acc"], v["both"],
                            COLOUR[arm], False))
        entries.sort(key=lambda e: e[2])

        fig, ax = plt.subplots(# Auf die DRUCKGROESSE ausgelegt: 	extwidth der Arbeit ist 449,55 pt
        # = 6,22 in. Bei figsize 6,2 in und Einbindung mit width=\linewidth
        # ist die Skalierung 1:1, die Punktgroessen hier sind also die auf
        # dem Papier. Vorher: 7,2 in bei 0,6\linewidth, also Faktor 0,52 --
        # aus 7,2 pt Beschriftung wurden 3,8 pt.
        figsize=(6.2, 0.30 * len(entries) + 1.15))
        for i, (name, acc, both, col, is_paper) in enumerate(entries):
            ax.barh(i, acc, height=0.66, color=col, alpha=0.30,
                    edgecolor=col, linewidth=0.8)
            ax.barh(i, both, height=0.66, color=col,
                    edgecolor=INK if is_paper else "white",
                    linewidth=1.3 if is_paper else 0.8,
                    linestyle="--" if is_paper else "-")
            ax.text(acc + 1.2, i, f"{acc:.1f}", va="center", ha="left",
                    fontsize=7.5, color=INK2)
            if both >= 14:
                ax.text(both - 1.2, i, f"{both:.1f}", va="center", ha="right",
                        fontsize=7.5, color="white")
            else:
                ax.text(acc + 7.5, i, f"({both:.1f})", va="center", ha="left",
                        fontsize=7.5, color=INK2)
        ax.set_yticks(np.arange(len(entries)))
        ax.set_yticklabels([e[0] for e in entries], fontsize=8.0)
        ax.set_xlim(0, 108)
        ax.set_xlabel("Top-1 (%)", fontsize=8.5)
        ax.set_title(BILD_TITEL[bench], fontsize=9.5, color=INK, pad=8)
        ax.tick_params(axis="x", labelsize=8.0)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        hand = [plt.Rectangle((0, 0), 1, 1, facecolor=INK2, alpha=0.30,
                              edgecolor=INK2, linewidth=0.8),
                plt.Rectangle((0, 0), 1, 1, facecolor=INK2,
                              edgecolor="white", linewidth=0.8)]
        ax.legend(hand, ["RMSD < 2 Å", "RMSD < 2 Å and PB-valid"],
                  loc="upper left", bbox_to_anchor=(0.0, -0.09),
                  ncol=2, frameon=False, fontsize=8.0)
        stem = "final_bars_" + bench.lower()
        for ext in ("pdf", "png"):
            fig.savefig(TARGET / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
            fig.savefig(HERE / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {stem}.pdf and .png ({len(entries)} bars)")


if __name__ == "__main__":
    main()
