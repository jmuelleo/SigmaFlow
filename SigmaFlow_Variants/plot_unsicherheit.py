"""Wie sich Unsicherheit und ihre Aussagekraft mit der Ziehungszahl entwickeln.

ZWEI PANELS, ZWEI VERSCHIEDENE FRAGEN
    Links das NIVEAU: wie stark buendelt das Modell seine Ziehungen, getrennt
    danach, ob die spaetere Auswahl richtig lag. Das Band zwischen den beiden
    Kurven ist die Trennung -- je breiter, desto mehr sagt die Konzentration
    ueber den Ausgang. Die durchgezogene Linie in der Mitte ist das Mittel
    ueber alle Faelle.

    Rechts die AUC: dieselbe Trennung, aber rangbasiert. Sie ist die einzige
    der beiden Groessen, die man ueber K hinweg vergleichen darf -- das Niveau
    faellt schon deshalb mit K, weil mehr Ziehungen mehr Nebenmoden finden.
    Dazu, gestrichelt auf der rechten Achse, die Genauigkeit selbst.

WARUM DIESE GEGENUEBERSTELLUNG DER PUNKT DER ABBILDUNG IST
    Die beiden Kurven rechts laufen auseinander: die Genauigkeit steigt bis
    zum Rand des Messbereichs weiter, die Guete des Vertrauensmasses steht
    ab etwa zwanzig Ziehungen. Man braucht also wenige Ziehungen, um zu
    WISSEN, ob man dem Ergebnis trauen kann, und viele, um das Ergebnis
    besser zu machen.

X-ACHSE LOGARITHMISCH
    Die Stuetzstellen sind 2 bis 200 und multiplikativ gewaehlt. Linear
    aufgetragen draengten sich die ersten sechs auf dem linken Zehntel --
    genau dort, wo sich alles entscheidet.

Aufruf:
    python SigmaFlow_Variants/plot_unsicherheit.py
    python SigmaFlow_Variants/plot_unsicherheit.py --ziel "Pfad/zum/draft"
"""
import argparse
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent

COLOUR = {"Minimal": "#2a78d6", "Separate": "#eb6834"}
LANG = {"Minimal": "SigmaFlow-NE", "Separate": "SigmaFlow-TR"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

p = argparse.ArgumentParser()
p.add_argument("--ziel", default=str(HERE))
p.add_argument("--csv", default=str(HERE / "unsicherheit_k.csv"))
a = p.parse_args()

pfad = pathlib.Path(a.csv)
if not pfad.is_file():
    sys.exit(f"ABBRUCH: {pfad} fehlt. Erst unsicherheit_k.py.")
t = pd.read_csv(pfad)

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.labelcolor": INK, "text.color": INK,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.1))
KS = sorted(t["K"].unique())
# Alle Stuetzstellen bekommen einen kleinen Strich, nur diese eine Ziffer.
BESCHRIFTET = [2, 5, 10, 20, 40, 100, 200]

# --- links: Niveau, getrennt nach Ausgang --------------------------------
for arm in ("Minimal", "Separate"):
    d = t[t["arm"] == arm].sort_values("K")
    c = COLOUR[arm]
    ax1.fill_between(d["K"], 100 * d["konz_daneben"], 100 * d["konz_trifft"],
                     color=c, alpha=0.09, linewidth=0, zorder=1)
    ax1.plot(d["K"], 100 * d["konz_trifft"], "-", color=c, lw=1.3,
             marker="o", ms=3.0, markeredgecolor="white", markeredgewidth=0.6,
             zorder=3)
    ax1.plot(d["K"], 100 * d["konz_daneben"], "-", color=c, lw=1.3,
             marker="o", ms=3.0, markeredgecolor="white", markeredgewidth=0.6,
             zorder=3)
    ax1.plot(d["K"], 100 * d["konz"], ":", color=c, lw=1.0, zorder=2)

ax1.set_ylabel("Draws in the largest cluster (%)")
ax1.set_title("Concentration, split by outcome", fontsize=8.5, pad=6)
ax1.set_ylim(15, 86)
# Die Beschriftung gehoert ans rechte Ende: dort ist der Abstand zwischen
# den beiden Kurven am groessten, links laufen sie zusammen.
ax1.annotate("ranked pose correct", xy=(22, 68), fontsize=7, color=INK2)
ax1.annotate("ranked pose wrong", xy=(22, 20), fontsize=7, color=INK2)
ax1.annotate("all", xy=(215, 47.5), fontsize=7, color=INK2, va="center")

# --- rechts: AUC und Genauigkeit -----------------------------------------
for arm in ("Minimal", "Separate"):
    d = t[t["arm"] == arm].sort_values("K")
    c = COLOUR[arm]
    lo = d["auc"] - d["auc_sd"].fillna(0)
    hi = d["auc"] + d["auc_sd"].fillna(0)
    ax2.fill_between(d["K"], lo, hi, color=c, alpha=0.13, linewidth=0, zorder=1)
    ax2.plot(d["K"], d["auc"], "-", color=c, lw=1.4, marker="o", ms=3.0,
             markeredgecolor="white", markeredgewidth=0.6, zorder=3,
             label=LANG[arm])
ax2.axhline(0.5, color=INK2, lw=0.7, ls="-", alpha=0.5, zorder=0)
ax2.text(26, 0.52, "no information", fontsize=6.8, color=INK2)
ax2.set_ylabel("AUC of concentration vs. correctness")
ax2.set_ylim(0.45, 1.0)
ax2.set_title("Value of the confidence signal", fontsize=8.5, pad=6)

r = ax2.twinx()
for arm in ("Minimal", "Separate"):
    d = t[t["arm"] == arm].sort_values("K")
    r.plot(d["K"], d["top1"], "--", color=COLOUR[arm], lw=1.1, alpha=0.75,
           zorder=2)
r.set_ylabel("Top-1 after ranking (%)", fontsize=8, color=INK2)
r.set_ylim(45, 100)
r.grid(False)
r.spines["top"].set_visible(False)
r.spines["right"].set_visible(True)
r.spines["right"].set_color(INK2)
r.spines["right"].set_linewidth(0.6)
r.tick_params(labelsize=7)

for ax in (ax1, ax2):
    ax.set_xscale("log")
    ax.set_xticks(BESCHRIFTET)
    ax.xaxis.set_major_formatter(
        mticker.FixedFormatter([str(k) for k in BESCHRIFTET]))
    ax.set_xticks(KS, minor=True)
    # Auf einer Logachse beschriftet matplotlib auch die Nebenstriche und
    # schreibt "1.4 x 10^2" quer ueber die 100 und die 200. Abschalten.
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlim(1.8, 230)
    ax.set_xlabel("Draws per complex  $K$")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

hand = [plt.Line2D([], [], color=COLOUR[k], lw=1.4, marker="o", ms=4,
                   markeredgecolor="white", label=LANG[k])
        for k in ("Minimal", "Separate")]
hand += [plt.Line2D([], [], color=INK2, lw=1.1, ls="--",
                    label="Top-1 after ranking (right axis)")]
fig.legend(handles=hand, loc="lower center", ncol=3, frameon=False,
           fontsize=7.2, bbox_to_anchor=(0.5, -0.10), handlelength=2.4)
fig.suptitle("PoseBusters, 67 complexes with poses  ·  5 integration steps",
             fontsize=9, color=INK, y=1.02)
fig.tight_layout(rect=(0, 0.02, 1, 1))

ZIEL = pathlib.Path(a.ziel)
ZIEL.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(ZIEL / f"uncertainty_vs_draws.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"uncertainty_vs_draws.pdf / .png  ->  {ZIEL}")
