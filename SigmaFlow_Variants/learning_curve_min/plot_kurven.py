"""Lernkurven ueber die Snapshots, eine Tafel je Metrik.

FORM
    Drei kleine Vielfache statt einer Achse mit sechs Linien. Die drei
    Metriken haben verschiedene Wertebereiche (bis 28 %, bis 19 %, bis 9 %);
    in einer Tafel waere die unterste Kurve flachgedrueckt. Zwei y-Achsen in
    einer Tafel kaeme nicht in Frage -- das ist der haeufigste Diagrammfehler
    ueberhaupt.

KODIERUNG
    Farbe = Arm (feste Reihenfolge, nie durchgewechselt), Linienart = Anzahl
    der Integrationsschritte. Die Identitaet haengt damit nicht an der Farbe
    allein, was fuer Druck und Farbfehlsichtigkeit noetig ist. Zusaetzlich
    werden die Endpunkte direkt beschriftet.

X-ACHSE
    Epochen, nicht Walltime. Die Arme haben verschiedenen Durchsatz (Separate
    ist rund 18 % langsamer) und zwei liefen ueber zwei Jobs mit je neu
    beginnender Stundenzaehlung; ueber Epochen entfaellt beides. Eine zweite
    Stundenachse gaebe es nicht, weil sie bei drei Armen fuer hoechstens einen
    richtig waere.

LUECKEN WERDEN NICHT UEBERMALT
    Bei SigmaFlow-Separate fehlen die Epochen 34 bis 139, weil der
    Snapshot-Mechanismus des Fortsetzungslaufs ausfiel. Eine durchgezogene
    Linie ueber diese Spanne wuerde einen gemessenen Verlauf behaupten, den es
    nicht gibt. Segmente ueber mehr als LUECKE Epochen werden deshalb blass
    gepunktet gezeichnet.

UNSICHERHEIT
    Band = 95 % aus dem Cluster-Bootstrap ueber Komplexe, nicht Wilson ueber
    Posen. Siehe kurve_daten.py.
"""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HIER = pathlib.Path(__file__).resolve().parent
ZIEL = HIER.parents[1] / "Thesis_Draft_Proposal" / "26.08.2026" / "draft3"

# Kategorische Reihenfolge, validiert (Slots 1-3): blau, orange, aqua.
# Anzeigenamen der Arbeit. Die SCHLUESSEL bleiben unveraendert -- sie
# adressieren Zellen, Pfade und Farben. Nur die Beschriftung wechselt.
ANZEIGE = {"SigmaFlow-Minimal": "SigmaFlow-NE",
           "SigmaFlow-Separate": "SigmaFlow-TR",
           "Minimal": "SigmaFlow-NE", "Separate": "SigmaFlow-TR",
           "SigmaDock": "SigmaDock"}
def anz(a):
    return ANZEIGE.get(a, a)

FARBE = {"SigmaFlow-Minimal": "#2a78d6",
         "SigmaFlow-Separate": "#eb6834",
         "SigmaDock": "#1baf7a"}
STIL = {25: "-", 5: "--"}
TAFELN = [("u2", r"RMSD $<$ 2 Å"),
          ("pb_prot", "PB-valid, with protein"),
          ("u2_prot", r"$<$ 2 Å and PB-valid")]
INK, INK2, GITTER = "#0b0b0b", "#52514e", "#d9d8d4"
# Ab dieser Epochendifferenz gilt ein Segment als nicht belegt.
LUECKE = 40

df = pd.concat([pd.read_csv(f) for f in sorted(HIER.glob("kurve_*_tidy.csv"))],
               ignore_index=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.labelcolor": INK, "text.color": INK,
})
fig, achsen = plt.subplots(1, 3, figsize=(7.2, 2.55), sharex=True)

for ax, (spalte, titel) in zip(achsen, TAFELN):
    for arm in [a for a in FARBE if a in set(df["arm"])]:
        for nfe in (25, 5):
            d = df[(df["arm"] == arm) & (df["nfe"] == nfe)].sort_values("epoch")
            if d.empty:
                continue
            f = FARBE[arm]
            ax.fill_between(d["epoch"], d[f"{spalte}_lo"], d[f"{spalte}_hi"],
                            color=f, alpha=0.13, linewidth=0)
            # Zusammenhaengende Stuecke voll, Luecken blass gepunktet.
            e = d["epoch"].to_numpy()
            bruch = [0] + [i for i in range(1, len(e))
                           if e[i] - e[i - 1] > LUECKE] + [len(e)]
            for a0, a1 in zip(bruch[:-1], bruch[1:]):
                t = d.iloc[a0:a1]
                ax.plot(t["epoch"], t[spalte], STIL[nfe], color=f,
                        linewidth=1.6, marker="o" if nfe == 25 else "s",
                        markersize=4.2, markerfacecolor="white",
                        markeredgewidth=1.4, zorder=3)
            for a1 in bruch[1:-1]:
                ax.plot(e[a1 - 1:a1 + 1],
                        d[spalte].to_numpy()[a1 - 1:a1 + 1], ":", color=f,
                        linewidth=1.0, alpha=0.45, zorder=2)
            # Endpunkt direkt beschriften: erfuellt die Reliefregel fuer
            # kontrastschwache Farben und spart einen Blick zur Legende.
            # Nach unten nur, solange darunter Platz ist. SigmaDock bei fuenf
            # Schritten endet knapp ueber null; eine Beschriftung darunter
            # laege auf der Achse.
            letzte = d.iloc[-1]
            hoch = nfe == 25 or letzte[spalte] < 0.12 * df[spalte].max()
            ax.annotate(f"{letzte[spalte]:.1f}",
                        (letzte["epoch"], letzte[spalte]),
                        textcoords="offset points",
                        xytext=(6, 4 if hoch else -4),
                        fontsize=7, color=INK2,
                        va="bottom" if hoch else "top")
    ax.set_title(titel, fontsize=8.5, color=INK, pad=8)
    ax.grid(axis="y", color=GITTER, linewidth=0.6)
    ax.set_axisbelow(True)
    for rand in ("top", "right"):
        ax.spines[rand].set_visible(False)
    ax.set_xlabel("Training epoch")
    ax.set_xlim(10, df['epoch'].max() + 12)
    ax.set_ylim(bottom=0)

achsen[0].set_ylabel("Poses (%)")

# KEINE zweite Stundenachse mehr.
#   Mit drei Armen ist sie mehrdeutig: dieselbe Epoche faellt bei jedem Arm auf
#   eine andere Walltime (Separate ist rund 18 % langsamer), und zwei Arme
#   liefen ueber zwei Jobs mit je neu beginnender Stundenzaehlung. Eine einzige
#   Stundenskala waere fuer hoechstens einen Arm richtig. Die Walltime steht
#   statt dessen in der Tidy-Tabelle und gehoert in die Bildunterschrift.
hand = [plt.Line2D([], [], color=FARBE[a], linestyle=STIL[n], linewidth=1.6,
                   marker="o" if n == 25 else "s", markersize=4.2,
                   markerfacecolor="white", markeredgewidth=1.4,
                   label=f"{anz(a)}, {n} steps")
        for a in FARBE if a in set(df["arm"]) for n in (25, 5)]
fig.legend(handles=hand, loc="lower center", ncol=3, frameon=False,
           fontsize=7.5, bbox_to_anchor=(0.5, -0.13), handlelength=2.4)
fig.tight_layout(rect=(0, 0.06, 1, 1))
for endung in ("pdf", "png"):
    fig.savefig(ZIEL / f"kurve_snapshots.{endung}", dpi=300,
                bbox_inches="tight")
print("geschrieben:", ZIEL / "kurve_snapshots.pdf", "und .png")
