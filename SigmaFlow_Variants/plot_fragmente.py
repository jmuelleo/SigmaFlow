"""Top-1 nach Ranking gegen die Fragmentzahl des Liganden, je Benchmark.

WOZU DIE ABBILDUNG
    Die Fragmentzahl bestimmt die Dimension des generativen Zustandsraums:
    je Fragment eine Rotation und eine Translation. Ob die Arme sich mit
    wachsendem Zustandsraum unterschiedlich verhalten, ist aus einer Tabelle
    ueber vierzehn Klassen nicht abzulesen -- hier ist es eine Kurve.

WELCHE ZELLEN
    Fuenf: die drei Arme bei fuenfundzwanzig Schritten, dazu die beiden
    Flow-Arme bei fuenf. SigmaDock bei fuenf Schritten FEHLT bewusst -- diese
    Zelle ist zusammengebrochen (6,5 % auf PB308, 8,2 % auf Astex) und wuerde
    die y-Achse so stauchen, dass die uebrigen fuenf Kurven ununterscheidbar
    werden. Das gehoert in die Bildunterschrift, nicht in die Abbildung.

WARUM DIE FALLZAHL MITGEZEICHNET WIRD
    Die Klassen sind sehr ungleich besetzt: auf PB308 von 61 Komplexen bei
    fuenf Fragmenten bis zu einem einzigen bei dreizehn. Eine Kurve, die das
    verschweigt, laedt dazu ein, aus einer Klasse mit n = 1 einen Trend zu
    lesen. Deshalb skaliert die Markergroesse mit der Wurzel aus n, und die
    Fallzahl steht zusaetzlich als Zeile unter der Kurve.

ENCODING WIE IN ranking_curves
    Farbe ist der Arm, Linienart die Schrittzahl. Die Zuordnung ist dieselbe
    wie in der uebrigen Arbeit, damit Identitaet nie allein an der Farbe
    haengt -- auch nicht im Schwarzweissdruck.

Aufruf:
    python SigmaFlow_Variants/plot_fragmente.py
    python SigmaFlow_Variants/plot_fragmente.py --ziel "Pfad/zum/draft"
"""
import argparse
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent

# Dieselben Werte wie in plot_ranking.py, damit die Abbildungen zusammenpassen.
COLOUR = {"SigmaFlow-Minimal": "#2a78d6",
          "SigmaFlow-Separate": "#eb6834",
          "SigmaDock": "#1baf7a"}
STYLE = {25: "-", 5: "--"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

# In den CSV heissen die Arme kurz; hier die Zuordnung auf die Anzeigenamen.
# LANG ist der SCHLUESSEL fuer COLOUR und darf nicht umbenannt werden;
# die Beschriftung steht in ANZEIGE.
LANG = {"Minimal": "SigmaFlow-Minimal", "Separate": "SigmaFlow-Separate",
        "SigmaDock": "SigmaDock"}
ANZEIGE = {"SigmaFlow-Minimal": "SigmaFlow-NE",
           "SigmaFlow-Separate": "SigmaFlow-TR",
           "SigmaDock": "SigmaDock"}
ZELLEN = [("SigmaDock", 25), ("Minimal", 25), ("Separate", 25),
          ("Minimal", 5), ("Separate", 5)]

SAETZE = {
    "pb308": ("nach_fragmenten.csv", "PoseBusters, 308 complexes", "fragments_pb308"),
    "astex": ("nach_fragmenten_astex.csv", "Astex, 85 complexes", "fragments_astex"),
}
ZIELE = [("RMSD<2 & PB", r"RMSD $<$ 2 Å and PB-valid"),
         ("RMSD<2", r"RMSD $<$ 2 Å")]

p = argparse.ArgumentParser()
p.add_argument("--ziel", default=str(HERE), help="Ausgabeverzeichnis")
p.add_argument("--regel", default="Mixed Score (Paper)")
# Vorgabe 1: ALLE Fragmentklassen werden gezeichnet, auf PB308 also bis
# vierzehn. Zuvor stand hier 5, und die duenn besetzten Klassen fehlten --
# das machte die Achse unvollstaendig. Wer sie unterdruecken will, setzt
# --min-n; die Fallzahlen stehen im Protokoll und gehoeren in die
# Bildunterschrift, damit ein Sprung von 0 auf 100 bei n = 1 nicht als
# Trend gelesen wird.
p.add_argument("--min-n", type=int, default=1,
               help="Klassen mit weniger Komplexen werden nicht gezeichnet")
a = p.parse_args()
ZIELDIR = pathlib.Path(a.ziel)

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.labelcolor": INK, "text.color": INK,
})

for satz, (csv, titel, dateiname) in SAETZE.items():
    pfad = HERE / csv
    if not pfad.is_file():
        sys.exit(f"ABBRUCH: {pfad} fehlt. Erst nach_fragmenten.py --satz {satz}.")
    t = pd.read_csv(pfad)
    t = t[(t["regel"] == a.regel) & (t["gruppe"] != "alle")].copy()
    if t.empty:
        sys.exit(f"ABBRUCH: keine Zeilen fuer Regel '{a.regel}' in {csv}.")
    t["frag"] = t["gruppe"].str.split("=").str[1].astype(int)

    n_je_klasse = t.groupby("frag")["n_komplexe"].first()
    behalten = sorted(n_je_klasse[n_je_klasse >= a.min_n].index)
    weg = sorted(n_je_klasse[n_je_klasse < a.min_n].index)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharex=True, sharey=True)
    for c, (ziel, zielname) in enumerate(ZIELE):
        ax = axes[c]
        d = t[t["ziel"] == ziel]
        # SigmaDock zuletzt und mit hoeherem zorder: es ist die Bezugslinie,
        # und in der ersten Fassung lag sie unter den anderen und war nicht
        # mehr zu sehen.
        for arm, nfe in ZELLEN:
            g = d[(d["arm"] == arm) & (d["nfe"] == nfe)].set_index("frag")
            g = g.reindex(behalten).dropna(subset=["anteil_prozent"])
            if g.empty:
                continue
            # Einheitliche, kleine Marker. Eine Groessenkodierung der Fallzahl
            # war die erste Idee und war falsch: bei n = 61 wurde der Punkt so
            # gross, dass er die Nachbarkurven verdeckte. Die Fallzahl steht
            # ohnehin als eigene Zeile unter der Achse -- doppelt kodiert, aber
            # nur einmal lesbar.
            z = 6 if arm == "SigmaDock" else 3
            ax.plot(g.index, g["anteil_prozent"], STYLE[nfe], marker="o",
                    markersize=3.4, markeredgecolor="white",
                    markeredgewidth=0.7, color=COLOUR[LANG[arm]],
                    linewidth=1.4, zorder=z)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.xaxis.set_major_locator(mticker.FixedLocator(behalten))
        ax.set_title(zielname, fontsize=8.5, color=INK, pad=6)
        if c == 0:
            ax.set_ylabel("Top-1 after ranking (%)")

    # Die x-Achse traegt NUR die Fragmentzahlen. Eine zweite Zeile mit der
    # Fallzahl darunter stand hier erst und war ein Fehler: sie las sich wie
    # ein zweistelliger Stapel und machte die Achse unleserlich. Die Fallzahlen
    # stehen stattdessen im Konsolenprotokoll und gehoeren in die
    # Bildunterschrift.
    for ax in axes:
        ax.set_xlabel("Fragments in the ligand")

    hand = [plt.Line2D([], [], color=COLOUR[LANG[arm]], linestyle=STYLE[nfe],
                       marker="o", markersize=4.5, markeredgecolor="white",
                       markeredgewidth=0.8, linewidth=1.4,
                       label=f"{ANZEIGE[LANG[arm]]}, {nfe} steps")
            for arm, nfe in ZELLEN]
    fig.legend(handles=hand, loc="lower center", ncol=3, frameon=False,
               fontsize=7.2, bbox_to_anchor=(0.5, -0.16), handlelength=2.4)
    fig.suptitle(titel, fontsize=9, color=INK, y=1.03)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(ZIELDIR / f"{dateiname}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"{dateiname}.pdf / .png  ->  {ZIELDIR}")
    print("  Fallzahlen : " + "  ".join(
        f"{f}:{int(n_je_klasse[f])}" for f in behalten))
    if weg:
        print(f"  ausgelassen: {weg}  (jeweils n < {a.min_n}: "
              + ", ".join(f"{f}->{int(n_je_klasse[f])}" for f in weg) + ")")
    print()
