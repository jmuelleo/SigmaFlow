#!/usr/bin/env python3
"""Schema: Geodaete, Exponential- und Logarithmusabbildung.

EINFACH IN VS CODE MIT PLAY STARTEN. Keine Argumente noetig.

WARUM 2D UND NICHT 3D
    Ein erster Entwurf zeichnete eine Kugel mit einer Tangentialebene in
    Matplotlibs 3D-Backend. Das scheitert: `plot_surface` und
    `Poly3DCollection` werden nicht gegeneinander tiefensortiert, die Ebene
    schneidet sichtbar durch die Kugel, und Beschriftungen lassen sich nicht
    zuverlaessig platzieren. Das ist eine Grenze des Backends, keine
    Einstellungsfrage.

    Ein Schema braucht ohnehin keine Perspektive. Gezeichnet wird deshalb ein
    gekruemmtes Flaechenstueck in der Ebene -- so, wie solche Abbildungen in
    der Differentialgeometrie ueblicherweise gesetzt werden.

WAS DIE ABBILDUNG ZEIGT
    gekruemmtes Flaechenstueck : Platzhalter fuer eine Riemannsche Mannigfaltigkeit
    R_a, R_b                   : zwei Punkte darauf
    durchgezogene Kurve        : die Geodaete zwischen ihnen
    schraege Ebene             : der Tangentialraum in R_a
    gerader Pfeil v            : Anfangsrichtung der Geodaete, auf ihre Laenge
                                 gebracht und geradlinig in der Ebene aufgetragen
    zwei gestrichelte Bogen    : Exp bringt v von der Ebene nach R_b,
                                 Log bringt R_b zurueck in die Ebene.
                                 Die beiden sind zueinander invers.

EINSCHRAENKUNG, DIE IN DIE BILDUNTERSCHRIFT GEHOERT
    SO(3) ist dreidimensional und nicht zeichenbar. Die Flaeche steht
    stellvertretend fuer eine gekruemmte Mannigfaltigkeit, sie ist NICHT SO(3).

    Ausserdem sind die gezeichneten Kurven KEINE berechneten Geodaeten. Sie
    sind Bilder von Geraden im Parameterraum und damit Kurven auf der Flaeche,
    nicht kuerzeste Verbindungen bezueglich der induzierten Metrik. Fuer ein
    Schema ist das angemessen, fuer eine quantitative Aussage waere es falsch.

    Die Abbildung erklaert eine Konstruktion. Sie belegt nichts.
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon

# ===========================================================================
#  EINSTELLUNGEN
# ===========================================================================

HIER = Path(__file__).resolve().parent
AUSGABE = HIER / "figures_en" / "M_manifold_schema.png"

AUCH_PDF = True
SCHRIFT = "serif"           # "serif" fuer LaTeX-nahe Mathematik, sonst "sans"
BESCHRIFTE_FLAECHE = True   # das Symbol M am Rand der Flaeche
ZEIGE_SCHNITTORT = False    # Punkt im Abstand pi samt den zwei kuerzesten Wegen
ZEIGE_THETA = False         # die Laenge der Geodaete beschriften
BEIDE_RICHTUNGEN = True     # Exp UND Log als getrennte Pfeile

C_FLAECHE = "#DCE7F0"
C_GITTER = "#A8BECF"
C_RAND = "#6E8CA6"
C_GEO = "#1F4E79"
C_TAN = "#F0C070"
C_TAN_RAND = "#C08A20"
C_VEK = "#B5451B"
C_CUT = "#707070"

# ===========================================================================
#  AB HIER MUSS NICHTS MEHR ANGEPASST WERDEN
# ===========================================================================

matplotlib.rcParams.update({
    "font.family": SCHRIFT,
    "mathtext.fontset": "cm" if SCHRIFT == "serif" else "dejavusans",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 10,
})


def flaeche(s, t):
    """Parametrisierung des Flaechenstuecks: (s,t) in [0,1]^2 -> Zeichenebene.

    Der Sinusterm woelbt die Flaeche nach oben, staerker am vorderen Rand
    (t = 0) als am hinteren. Ohne genug Woelbung faellt der geradlinige
    Tangentialvektor mit der Geodaete zusammen -- und genau der Unterschied
    zwischen beiden ist das, was die Abbildung zeigen soll.
    """
    s = np.asarray(s, float)
    t = np.asarray(t, float)
    x = 0.9 + 8.4 * s + 1.35 * t
    y = 1.15 + 2.45 * t + 1.55 * np.sin(np.pi * s) * (1.0 - 0.42 * t)
    return np.stack([x, y], axis=-1)


def tangenten(s0, t0, h=1e-4):
    """Die beiden Koordinatenrichtungen der Flaeche im Punkt (s0,t0)."""
    ds = (flaeche(s0 + h, t0) - flaeche(s0 - h, t0)) / (2 * h)
    dt = (flaeche(s0, t0 + h) - flaeche(s0, t0 - h)) / (2 * h)
    return ds, dt


def kurve(p0, p1, n=200):
    """Bild einer Geraden im Parameterraum -- auf der Flaeche gekruemmt."""
    lam = np.linspace(0, 1, n)[:, None]
    st = (1 - lam) * np.asarray(p0) + lam * np.asarray(p1)
    return flaeche(st[:, 0], st[:, 1])


fig, ax = plt.subplots(figsize=(7.2, 4.3))
ax.set_aspect("equal")
ax.axis("off")

# --- Flaechenstueck ---------------------------------------------------------
rand = np.concatenate([
    kurve((0, 0), (1, 0)), kurve((1, 0), (1, 1)),
    kurve((1, 1), (0, 1)), kurve((0, 1), (0, 0))])
ax.add_patch(Polygon(rand, closed=True, facecolor=C_FLAECHE,
                     edgecolor=C_RAND, linewidth=1.1, zorder=1))
for s0 in np.linspace(0, 1, 11)[1:-1]:
    ax.plot(*kurve((s0, 0), (s0, 1)).T, color=C_GITTER, lw=0.5, zorder=2)
for t0 in np.linspace(0, 1, 5)[1:-1]:
    ax.plot(*kurve((0, t0), (1, t0)).T, color=C_GITTER, lw=0.5, zorder=2)

# --- die beiden Punkte und die Geodaete -------------------------------------
# R_b liegt bewusst weit von R_a entfernt: erst dann trennen sich der
# geradlinige Vektor in der Ebene und die gekruemmte Geodaete sichtbar.
PA, PB = (0.17, 0.48), (0.80, 0.72)
A = flaeche(*PA)
B = flaeche(*PB)
geo = kurve(PA, PB)
ax.plot(*geo.T, color=C_GEO, lw=2.6, solid_capstyle="round", zorder=6)

# --- der Vektor v im Tangentialraum -----------------------------------------
# Richtung = Anfangsrichtung der Geodaete, Laenge = Laenge der Geodaete.
richtung = geo[1] - geo[0]
richtung = richtung / np.linalg.norm(richtung)
laenge = float(np.sum(np.linalg.norm(np.diff(geo, axis=0), axis=1)))
spitze = A + richtung * laenge

# --- Tangentialraum in R_a --------------------------------------------------
# Aufgespannt von den echten Flaechentangenten in R_a, damit die Ebene
# tatsaechlich anliegt. Die Ausdehnung folgt aus der Vektorlaenge, statt fest
# verdrahtet zu sein -- sonst raegt die Pfeilspitze heraus, sobald R_b weiter
# entfernt gesetzt wird.
ds, dt = tangenten(*PA)
dt = dt / np.linalg.norm(dt)
# Aufgespannt entlang des Vektors selbst und der zweiten Flaechenrichtung.
# Ein erster Versuch nahm ds und dt und bemass die Ebene ueber die Projektion
# von v darauf -- weil v nicht entlang ds zeigt, wuchs die Querausdehnung mit
# und die Ebene ueberdeckte die ganze Abbildung.
e1, e2 = richtung, dt
ecken = np.array([A + p * e1 + q * e2 for p, q in
                  [(-0.9, -1.15), (laenge + 1.0, -1.15),
                   (laenge + 1.0, 1.15), (-0.9, 1.15)]])
ax.add_patch(Polygon(ecken, closed=True, facecolor=C_TAN, alpha=0.30,
                     edgecolor=C_TAN_RAND, linewidth=1.1, zorder=4))

ax.add_patch(FancyArrowPatch(A, spitze, arrowstyle="-|>", mutation_scale=15,
                             color=C_VEK, lw=2.1, shrinkA=0, shrinkB=0,
                             zorder=7))

# --- Exp und Log: die beiden Richtungen zwischen Ebene und Flaeche ----------
# EIN gestrichelter Bogen mit Spitzen an beiden Enden. Zwei getrennte Bogen
# lagen bei diesem Abstand als Linse uebereinander und liessen sich nicht mehr
# eindeutig beschriften. Die Leserichtung steckt in den Spitzen: die Spitze
# an R_b gehoert zu Exp (Ebene -> Flaeche), die an v gehoert zu Log
# (Flaeche -> Ebene).
BOGEN_RAD = 0.30
stil = "<|-|>" if BEIDE_RICHTUNGEN else "-|>"
ax.add_patch(FancyArrowPatch(spitze, B, arrowstyle=stil, mutation_scale=12,
                             color=C_VEK, lw=1.3, ls=(0, (4, 3)),
                             shrinkA=6, shrinkB=9,
                             connectionstyle=f"arc3,rad={BOGEN_RAD}", zorder=7))


def auf_bogen(p0, p1, rad, t):
    """Punkt auf demselben Bogen, den `arc3` zeichnet.

    Matplotlib legt fuer arc3 eine quadratische Bezierkurve durch p0 und p1;
    der Kontrollpunkt sitzt im Mittelpunkt, um rad * Segmentlaenge senkrecht
    versetzt (siehe ConnectionStyle.Arc3.connect). Die Beschriftungen werden
    darauf gesetzt statt auf die Endpunkte -- nur so sitzen sie wirklich am
    Pfeilkopf und nicht irgendwo daneben.
    """
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    m = 0.5 * (p0 + p1)
    d = p1 - p0
    c = m + rad * np.array([d[1], -d[0]])
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * c + t ** 2 * p1

# --- Schnittort, standardmaessig aus ----------------------------------------
if ZEIGE_SCHNITTORT:
    PC = (0.965, 0.26)
    C = flaeche(*PC)
    for stuetze in ((0.80, 0.99), (0.50, 0.02)):
        lam = np.linspace(0, 1, 200)[:, None]
        st = ((1 - lam) ** 2 * np.array(PA)
              + 2 * (1 - lam) * lam * np.array(stuetze)
              + lam ** 2 * np.array(PC))
        ax.plot(*flaeche(st[:, 0], st[:, 1]).T, color=C_CUT, lw=1.1,
                ls=(0, (2.5, 2.5)), zorder=5)
    ax.plot(*C, marker="o", markerfacecolor="white", markeredgecolor=C_CUT,
            markersize=8, markeredgewidth=1.4, zorder=8)

for P in (A, B):
    ax.plot(*P, marker="o", color=C_GEO, markersize=7, zorder=8)

# --- Beschriftung: nur Symbole; alles Uebrige steht in der caption -----------
# Der gerade Pfeil ist der VEKTOR v, die beiden Bogen sind die ABBILDUNGEN.
# Ohne diese Trennung stuende "Log" zweimal im Bild und meinte zweierlei.
ax.annotate(r"$R_a$", A, textcoords="offset points", xytext=(-6, -17),
            color=C_GEO, fontsize=12, ha="center")
ax.annotate(r"$R_b$", B, textcoords="offset points", xytext=(6, -17),
            color=C_GEO, fontsize=12, ha="center")
ax.annotate(r"$v$", A + richtung * laenge * 0.55, textcoords="offset points",
            xytext=(-7, 13), color=C_VEK, fontsize=13, ha="center")
# Die beiden Namen sitzen an IHREM Ende des Doppelpfeils, nicht in der Mitte:
# so ist ohne Legende klar, welche Spitze zu welcher Abbildung gehoert.
ax.annotate(r"$\mathrm{Exp}_{R_a}$", auf_bogen(spitze, B, BOGEN_RAD, 0.86),
            textcoords="offset points", xytext=(12, -2), color=C_VEK,
            fontsize=11, ha="left", va="center")
if BEIDE_RICHTUNGEN:
    ax.annotate(r"$\mathrm{Log}_{R_a}$", auf_bogen(spitze, B, BOGEN_RAD, 0.12),
                textcoords="offset points", xytext=(12, 2), color=C_VEK,
                fontsize=11, ha="left", va="center")
ax.annotate(r"$T_{R_a}\mathcal{M}$", ecken[2], textcoords="offset points",
            xytext=(12, 6), color=C_TAN_RAND, fontsize=11, ha="left")
if ZEIGE_THETA:
    ax.annotate(r"$\theta$", geo[len(geo) // 2], textcoords="offset points",
                xytext=(-2, -16), color=C_GEO, fontsize=12, ha="center")
if BESCHRIFTE_FLAECHE:
    ax.annotate(r"$\mathcal{M}$", flaeche(0.05, 0.05),
                textcoords="offset points", xytext=(-4, 8), color=C_RAND,
                fontsize=13, ha="center")

ax.autoscale_view()
ax.margins(0.04)

AUSGABE.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(AUSGABE)
print(f"geschrieben: {AUSGABE}")
if AUCH_PDF:
    fig.savefig(AUSGABE.with_suffix(".pdf"))
    print(f"geschrieben: {AUSGABE.with_suffix('.pdf')}")
plt.close(fig)
