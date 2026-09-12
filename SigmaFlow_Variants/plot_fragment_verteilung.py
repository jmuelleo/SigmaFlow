#!/usr/bin/env python3
"""Histogramm der Fragmentzahl -- fuer PoseBusters UND Astex, nebeneinander lesbar.

WAS SICH GEGENUEBER "Thesis Visualisierungen/plot_fragment_histogram.py" AENDERT
    1. Die Zahlen kommen aus der LOKALEN Messung (fragmentzahl_<satz>.csv,
       erzeugt von fragmentzahl.py), nicht mehr aus der aggregierten CSV der
       ARC-Laeufe. Beide stimmen fuer N = 1 bis 9 exakt ueberein -- das sind
       299 der 308 Komplexe -- und unterscheiden sich nur im Schwanz um je +-1
       bei N = 10, 11, 13, 14.

       Das ist kein Fehler in einer der Messungen: `fragmentation_strategy`
       steht auf "random", und 14 von 308 Komplexen liefern ueber acht
       Ziehungen nicht dieselbe Zahl. Genau diese liegen im Schwanz. Hier wird
       der Median aus acht Ziehungen genommen.

       ACHTUNG: die alte Fassung vermerkte, die Klasse N = 13 sei leer und das
       sei "eine echte Luecke in der Verteilung". Unter dieser Messung liegt
       dort ein Komplex. Wer den Satz in einer Bildunterschrift stehen hat,
       muss ihn streichen.

    2. Es gibt ihn fuer beide Benchmarks, mit GEMEINSAMER x-Achse. Astex reicht
       nur bis acht Fragmente; die leeren Klassen bis vierzehn sind kein
       Schoenheitsfehler, sondern die Aussage: der Satz enthaelt keine grossen
       Liganden. Nebeneinander gestellt ist das der auffaelligste Unterschied
       zwischen den beiden Benchmarks.

    3. Die y-Achse ist NICHT geteilt. PoseBusters hat 61 Komplexe in der
       groessten Klasse, Astex 20; eine gemeinsame Skala drueckte Astex zu
       einer flachen Zeile zusammen. Die Saeulenzahlen stehen ohnehin darueber.

Aufruf:
    python SigmaFlow_Variants/plot_fragment_verteilung.py
    python SigmaFlow_Variants/plot_fragment_verteilung.py --ziel "Pfad/zum/draft"
    python SigmaFlow_Variants/plot_fragment_verteilung.py --eigene-achse
"""
import argparse
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent

# Palette und Typografie wie in Thesis Visualisierungen/plot_fragment_histogram.py,
# damit die neue Abbildung neben der alten nicht auffaellt.
C_BAR, C_BAR_EDGE = "#7BA7C7", "#3E6C8E"
C_LINE, C_GRID = "#333333", "#DDDDDD"

SAETZE = {
    "pb308": ("fragmentzahl_pb308.csv", "PoseBusters, 308 complexes",
              "A_fragment_distribution_308"),
    "astex": ("fragmentzahl_astex.csv", "Astex, 85 complexes",
              "A_fragment_distribution_astex85"),
}

p = argparse.ArgumentParser()
p.add_argument("--ziel", default=str(HERE))
p.add_argument("--eigene-achse", action="store_true",
               help="jede Abbildung nur bis zur eigenen groessten Klasse")
p.add_argument("--titel", action="store_true",
               help="Ueberschrift ins Bild statt in die Bildunterschrift")
a = p.parse_args()
ZIEL = pathlib.Path(a.ziel)

matplotlib.rcParams.update({
    "font.family": "sans", "mathtext.fontset": "dejavusans",
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": C_GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True,
})

# Erst alle Verteilungen einlesen, damit die gemeinsame Achse bekannt ist,
# bevor die erste Abbildung gezeichnet wird.
verteilungen = {}
for satz, (csv, titel, name) in SAETZE.items():
    f = HERE / csv
    if not f.is_file():
        sys.exit(f"ABBRUCH: {f} fehlt. Erst fragmentzahl.py fuer '{satz}'.")
    t = pd.read_csv(f)
    ohne = int(t["n_frag"].isna().sum())
    t = t[t["n_frag"].notna()]
    verteilungen[satz] = (t["n_frag"].astype(int).value_counts().sort_index(),
                          ohne, int((~t["stabil"]).sum()))

x_max_global = max(int(v.index.max()) for v, _, _ in verteilungen.values())

for satz, (csv, titel, name) in SAETZE.items():
    hist, ohne, instabil = verteilungen[satz]
    x_max = int(hist.index.max()) if a.eigene_achse else x_max_global
    ks = np.arange(1, x_max + 1)
    counts = np.array([float(hist.get(int(k), 0)) for k in ks])
    n = int(counts.sum())

    # Kennzahlen aus der Verteilung, nicht abgetippt.
    werte = np.repeat(ks, counts.astype(int))
    mittel, median = float(werte.mean()), float(np.median(werte))

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.bar(ks, counts, width=0.82, color=C_BAR, edgecolor=C_BAR_EDGE,
           linewidth=0.8, zorder=2)

    # Fallzahl ueber jeder Saeule: die y-Achse allein unterscheidet 15 und 18
    # nicht, und genau dort liegt der interessante Teil der Verteilung.
    for k, c in zip(ks, counts):
        if c:
            ax.text(k, c + counts.max() * 0.035, f"{int(c)}", ha="center",
                    va="bottom", fontsize=7.5, color="#444444", zorder=6)

    # ymax stoppt die Linien UNTER der Zeile mit den Saeulenzahlen -- sonst
    # laufen sie mitten durch die Ziffer ueber der hoechsten Saeule.
    # Und die Legende gehoert nach links: rechts oben liegt bei beiden
    # Benchmarks das Plateau der kumulativen Kurve.
    ax.axvline(median, color=C_LINE, lw=1.1, ls="-", alpha=.55, zorder=1,
               ymax=.80, label=f"Median {median:.0f}")
    ax.axvline(mittel, color=C_LINE, lw=1.1, ls=":", alpha=.75, zorder=1,
               ymax=.80, label=f"Mean {mittel:.2f}")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")

    ax.set_ylim(0, counts.max() * 1.22)
    ax.set_ylabel("Ligands")
    ax.set_xlabel(r"Rigid fragments per ligand  $N$")
    ax.set_xticks(ks)
    ax.set_xlim(0.4, x_max + 0.6)
    if a.titel:
        ax.set_title(f"{titel}   (n = {n})", loc="left", fontsize=10)

    # Zustandsdimension nach OBEN -- unten kollidiert sie mit der
    # Achsenbeschriftung.
    oben = ax.secondary_xaxis("top", functions=(lambda x: 6 * x, lambda x: x / 6))
    oben.set_xticks([6 * int(k) for k in ks])
    oben.set_xlabel(r"State dimension  $D = 6N$", fontsize=8, labelpad=4)
    oben.tick_params(labelsize=7)

    rechts = ax.twinx()
    rechts.plot(ks, 100 * np.cumsum(counts) / n, color=C_LINE, lw=1.3,
                marker="o", ms=3, zorder=5)
    rechts.set_ylabel("Cumulative  [%]", fontsize=8)
    rechts.set_ylim(0, 105)
    rechts.set_yticks([0, 25, 50, 75, 100])
    rechts.grid(False)
    rechts.spines["right"].set_visible(True)
    rechts.spines["top"].set_visible(False)

    ZIEL.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(ZIEL / f"{name}.{ext}")
    plt.close(fig)

    print(f"{name}.png / .pdf  ->  {ZIEL}")
    print(f"  n = {n}   Median {median:.0f}   Mittel {mittel:.2f}   "
          f"groesste Klasse {int(hist.index.max())}")
    print(f"  D = 6N:  Median {6*median:.0f}   Mittel {6*mittel:.1f}   "
          f"Max {6*int(hist.index.max())}")
    print(f"  instabile Fragmentzahl bei {instabil} Komplexen"
          + (f", {ohne} ohne Messwert" if ohne else ""))
    print("  Verteilung: " + "  ".join(
        f"{int(k)}:{int(c)}" for k, c in zip(ks, counts) if c))
    print()
