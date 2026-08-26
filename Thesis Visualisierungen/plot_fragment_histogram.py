#!/usr/bin/env python3
"""Histogramm der Fragmentzahl ueber den PoseBusters-v2-Satz.

EINFACH IN VS CODE MIT F5 / PLAY STARTEN. Keine Argumente noetig.
Alles, was man aendern koennte, steht in der EINSTELLUNGEN-Sektion unten.

WAS DIE ABBILDUNG ZEIGT
    Saeulen  : wie viele Liganden in N starre Fragmente zerfallen
    obere Achse: die daraus folgende Zustandsdimension D = 6N
    Kurve    : kumulativer Anteil, also "wie viele haben hoechstens N"
    Linien   : Median und Mittel
    Hinweis  : die Liganden mit N = 1, die gar nicht zerlegt werden

WOHER DIE ZAHLEN
    data/fragment_distribution_308.csv, erzeugt aus zwei ARC-Laeufen von
    arc/count_fragments_set.py ueber posebusters_v2_308_ids.csv:

      Job 8652888  305 Liganden, Zeitlimit 120 s je Molekuel
      Job 8652975  die drei Nachzuegler mit 22 bzw. 23 Torsionen
                   (7PK0_BYC, 7PT3_3KK, 8F4J_PHO), Zeitlimit 3600 s
                   -> alle drei zerfallen in 14 Fragmente, D = 84

    Damit sind es 308 von 308. Die Klasse N = 13 ist leer; das ist kein
    fehlender Messwert, sondern eine echte Luecke in der Verteilung.

    Falls doch einmal Liganden fehlen, weist der Titel die Zahl aus
    (N_ANGEFORDERT minus Summe der Balken). Es wird NICHTS geschaetzt.
"""

import csv
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ===========================================================================
#  EINSTELLUNGEN
# ===========================================================================

HIER = Path(__file__).resolve().parent

CSV_DATEI = HIER / "data" / "fragment_distribution_308.csv"
AUSGABE = HIER / "figures_en" / "A_fragment_distribution_308.png"

N_ANGEFORDERT = 308        # Groesse des Satzes; Differenz = nicht auswertbar
SPRACHE = "en"             # "en" fuer die Thesis, "de" fuer Arbeitslogs
KUMULATIV = "twin"         # "twin" = zweite y-Achse, "panel" = eigenes Feld
AUCH_PDF = True            # Vektorfassung fuer den Satz mitschreiben

# Was IN der Abbildung steht. Alles Uebrige gehoert in die LaTeX-Bildunterschrift:
# dort ist es im Fliesstext gesetzt, in der Schrift des Dokuments, und es
# skaliert nicht mit, wenn die Abbildung kleiner eingebunden wird.
TITEL = False              # Ueberschrift im Bild (sonst caption)
FUSSZEILE = False          # Hinweiszeile unter der Achse (sonst caption)
SAEULENZAHLEN = True       # Fallzahl ueber jeder Saeule
LINIEN = True              # Median und Mittel als Linien mit Legende
SCHRIFT = "sans"           # "serif" naehert die Schrift eines LaTeX-Dokuments an

# Zurueckhaltende, druckbare Palette; identisch zu visualization/plot_fragments.py,
# damit die Abbildungen der Thesis zusammenpassen.
C_BAR = "#7BA7C7"
C_BAR_EDGE = "#3E6C8E"
C_LINE = "#333333"
C_GRID = "#DDDDDD"

TEXTE = {
    "en": {
        "ligands": "Ligands",
        "frags": "Rigid fragments per ligand  $N$",
        "statedim": "State dimension  $D = 6N$",
        "cum": "Cumulative  [%]",
        "atleast": "Fraction $\\geq N$  [%]",
        "median": "Median",
        "mean": "Mean",
        "title": "Rigid fragments per ligand — PoseBusters v2",
        "nomeas": "not measured",
        "rigid": ("{n} of the {N} evaluated ligands ({p:.1f}%) consist of a single "
                  "rigid fragment and are placed as one body ($D = 6$)."),
    },
    "de": {
        "ligands": "Liganden",
        "frags": "Starre Fragmente je Ligand  $N$",
        "statedim": "Zustandsdimension  $D = 6N$",
        "cum": "Kumuliert  [%]",
        "atleast": "Anteil $\\geq N$  [%]",
        "median": "Median",
        "mean": "Mittel",
        "title": "Starre Fragmente je Ligand — PoseBusters v2",
        "nomeas": "ohne Messwert",
        "rigid": ("{n} der {N} ausgewerteten Liganden ({p:.1f}%) bestehen aus einem "
                  "einzigen starren Fragment und werden als ein Koerper platziert "
                  "($D = 6$)."),
    },
}

# ===========================================================================
#  AB HIER MUSS NICHTS MEHR ANGEPASST WERDEN
# ===========================================================================

matplotlib.rcParams.update({
    "font.family": SCHRIFT,
    "mathtext.fontset": "cm" if SCHRIFT == "serif" else "dejavusans",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": C_GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
})

T = TEXTE[SPRACHE]


def lies_verteilung(pfad):
    """Liest die aggregierte Verteilung: eine Zeile je Fragmentklasse."""
    if not pfad.exists():
        raise SystemExit(
            f"FEHLER: {pfad} fehlt.\n"
            "  Die Datei entsteht aus dem ARC-Lauf; siehe Kopf dieses Skripts."
        )
    hist = {}
    with open(pfad, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            hist[int(r["fragments"])] = int(r["n_ligands"])
    if not hist:
        raise SystemExit(f"FEHLER: {pfad} enthaelt keine Zeilen.")
    return hist


hist = lies_verteilung(CSV_DATEI)
lo, hi = min(hist), max(hist)
ks = np.arange(lo, hi + 1)
counts = np.array([hist.get(int(k), 0) for k in ks], dtype=float)
n = int(counts.sum())
fehlend = N_ANGEFORDERT - n

# Kennzahlen aus der Verteilung, nicht abgetippt. Der Median wird aus der
# expandierten Liste bestimmt, damit gerade Fallzahlen korrekt gemittelt werden.
werte = np.repeat(ks, counts.astype(int))
mittel = float(werte.mean())
median = float(np.median(werte))

print(f"{n} Liganden ausgewertet, {fehlend} ohne Messwert")
print(f"Mittel {mittel:.2f}   Median {median:.0f}   Max {hi}")
print(f"D = 6N:  Mittel {6*mittel:.1f}   Median {6*median:.0f}   Max {6*hi}")

# --- Aufbau ----------------------------------------------------------------
if KUMULATIV == "twin":
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax_unten = None
else:
    fig, (ax, ax_unten) = plt.subplots(
        2, 1, figsize=(6.4, 4.8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12})

balken = ax.bar(ks, counts, width=0.82, color=C_BAR,
                edgecolor=C_BAR_EDGE, linewidth=0.8, zorder=2)

# Fallzahl ueber jede Saeule. Bei zwoelf Klassen ist die y-Achse allein
# zu grob, um 15 von 18 zu unterscheiden.
if SAEULENZAHLEN:
    for k, c in zip(ks, counts):
        if c:
            ax.text(k, c + counts.max() * 0.035, f"{int(c)}", ha="center",
                    va="bottom", fontsize=7.5, color="#444444", zorder=6)

# Median und Mittel als Linien mit Legende. Inline gesetzte Beschriftungen
# ueberlagern bei diesem Wertebereich die Saeulen.
if LINIEN:
    ax.axvline(median, color=C_LINE, lw=1.1, ls="-", alpha=0.55, zorder=1,
               label=f'{T["median"]} {median:.0f}')
    ax.axvline(mittel, color=C_LINE, lw=1.1, ls=":", alpha=0.75, zorder=1,
               label=f'{T["mean"]} {mittel:.2f}')
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")

ax.set_ylim(0, counts.max() * 1.22)
ax.set_ylabel(T["ligands"])
ax.set_xticks(ks)

if TITEL:
    titel = f'{T["title"]}  (n = {n})'
    if fehlend:
        titel += f'   [{fehlend} {T["nomeas"]}]'
    ax.set_title(titel, loc="left", fontsize=10)

# Einzelfragment-Liganden: der aussagekraeftigste Rand der Verteilung, weil
# dort D = 6 ist und die Zerlegung nichts zu tun hat. Ohne Hinweis verschwindet
# die Saeule neben den grossen.
FUSSZEILE_TEXT = (T["rigid"].format(n=int(counts[0]), N=n, p=100 * counts[0] / n)
                  if (lo == 1 and counts[0]) else "")

# Die Zustandsdimension gehoert nach OBEN: unten kollidiert sie mit der
# Achsenbeschriftung.
oben = ax.secondary_xaxis("top", functions=(lambda x: 6 * x, lambda x: x / 6))
oben.set_xticks([6 * int(k) for k in ks])
oben.set_xlabel(T["statedim"], fontsize=8, labelpad=4)
oben.tick_params(labelsize=7)

# --- Kumulative Kurve ------------------------------------------------------
if KUMULATIV == "twin":
    rechts = ax.twinx()
    kum = 100 * np.cumsum(counts) / n
    rechts.plot(ks, kum, color=C_LINE, lw=1.3, marker="o", ms=3, zorder=5)
    rechts.set_ylabel(T["cum"], fontsize=8)
    rechts.set_ylim(0, 105)
    rechts.set_yticks([0, 25, 50, 75, 100])
    rechts.grid(False)
    rechts.spines["right"].set_visible(True)
    rechts.spines["top"].set_visible(False)
    ax.set_xlabel(T["frags"])
else:
    anteil_ab = 100 * (1 - np.cumsum(counts) / n + counts / n)
    ax_unten.step(ks, anteil_ab, where="mid", color=C_BAR_EDGE, lw=1.3)
    ax_unten.set_ylabel(T["atleast"], fontsize=8)
    ax_unten.set_xlabel(T["frags"])
    ax_unten.set_xticks(ks)
    ax_unten.set_ylim(0, 100)
    ax_unten.set_yticks([0, 50, 100])

if FUSSZEILE and FUSSZEILE_TEXT:
    # Unterhalb der Achsenbeschriftung. Bei -0.05 laege sie auf den Ziffern
    # der x-Achse; die Beschriftung selbst sitzt bei etwa -0.13.
    fig.text(0.0, -0.24, FUSSZEILE_TEXT, ha="left", va="top",
             fontsize=7.5, color="#555555", transform=ax.transAxes)

AUSGABE.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(AUSGABE)
print(f"geschrieben: {AUSGABE}")
if AUCH_PDF:
    pdf = AUSGABE.with_suffix(".pdf")
    fig.savefig(pdf)
    print(f"geschrieben: {pdf}")
plt.close(fig)
