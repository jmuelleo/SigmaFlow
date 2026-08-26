#!/usr/bin/env python3
"""SigmaDock-Pipeline: Eingaben, EquiformerV2, Newton-Euler-Auslese, Scores.

EINFACH IN VS CODE MIT PLAY STARTEN. Keine Argumente noetig.

WOHER DIE INHALTE STAMMEN
    Alles unten ist aus dem SigmaDock-Referenzcode abgelesen, nicht aus dem
    Gedaechtnis. Fundstellen (SigmaDock/src_sigmadock/):

    Knoten- und Kantentypen        oracle.py:200-300  (HPARAMS)
    Architekturgroessen            config.py:89-108
    Zeit- und Knoteneinbettung     src_sigmadock_net/model#Der Equiformer.py:442
    Kanteneinbettung               ebenda:572
    Bloecke, Norm, force_block     ebenda:664-722
    f_i = l=1-Anteil der Ausgabe   ebenda:715-718  (narrow(1, 1, 3))
    f_i = -epsilon_theta           src_sigmadock_diff/denoiser:815
    Kraft und Drehmoment           ebenda:825-859  (linear_mechanics)
    dT, domega                     ebenda:861-915  (newton_maruyama)
    Skalierung und Scores          ebenda:976-1090 (_get_scalings, _compute_scores)
    Reihenfolge der Schritte       ebenda:1120-1213 (forward)

ZUM AUFBAU DES SKRIPTS
    Die Kastenhoehen werden aus dem Inhalt BERECHNET, nicht von Hand gesetzt.
    Ein erster Entwurf hatte sie fest verdrahtet, und der Text lief in der
    Haelfte der Kaesten unten heraus. Wer eine Zeile ergaenzt, muss hier
    nichts nachrechnen.

WAS DIE ABBILDUNG BEWUSST NICHT ZEIGT
    Trainingsverlust und Rueckwaerts-SDE nur als Schlusszeile. Der Fokus liegt
    auf dem Weg von der Eingabe ueber f_i zu den beiden Scores.

    Zahlen in Klammern sind die Vorgabewerte aus config.py und ueber die
    Kommandozeile ueberschreibbar.
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ===========================================================================
#  EINSTELLUNGEN
# ===========================================================================

HIER = Path(__file__).resolve().parent
AUSGABE = HIER / "figures_en" / "P_sigmadock_pipeline.png"

AUCH_PDF = True
SCHRIFT = "serif"
ZEIGE_GROESSEN = True      # die Vorgabewerte aus config.py mitschreiben

C_EIN, C_EIN_R = "#DCE7F0", "#6E8CA6"      # Eingaben
C_NETZ, C_NETZ_R = "#F7E9CD", "#C08A20"    # das Netz
C_INNEN = "#FDF7EC"                        # Kaesten im Netz
C_LESE, C_LESE_R = "#E6DCEF", "#7B5EA7"    # Auslese
C_SCORE, C_SCORE_R = "#F3D7CE", "#B5451B"  # Scores
C_GRAU, C_PFEIL = "#555555", "#8A8A8A"

# Setzmasse in Zeichenkoordinaten
H_TITEL, H_ZEILE, H_RAND = 4.4, 3.0, 2.0
FS_ABSCHNITT, FS_TITEL, FS_ZEILE = 11.0, 9.2, 7.4

# ===========================================================================
#  AB HIER MUSS NICHTS MEHR ANGEPASST WERDEN
# ===========================================================================

matplotlib.rcParams.update({
    "font.family": SCHRIFT,
    "mathtext.fontset": "cm" if SCHRIFT == "serif" else "dejavusans",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

fig, ax = plt.subplots(figsize=(8.2, 11.0))
ax.set_xlim(0, 100)
ax.set_ylim(0, 152)
ax.axis("off")

G = ZEIGE_GROESSEN


def hoehe(titel, zeilen):
    """Die Hoehe, die ein Kasten fuer seinen Inhalt braucht."""
    return (H_TITEL if titel else H_RAND) + len(zeilen) * H_ZEILE + H_RAND


def kasten(x, oben, w, titel, zeilen=(), fc=C_EIN, ec=C_EIN_R,
           ts=FS_TITEL, bs=FS_ZEILE, h=None, lw=1.1):
    """Zeichnet einen Kasten von `oben` nach unten. Gibt die Unterkante zurueck."""
    zeilen = [z for z in zeilen if z]
    h = hoehe(titel, zeilen) if h is None else h
    ax.add_patch(FancyBboxPatch(
        (x, oben - h), w, h, boxstyle="round,pad=0,rounding_size=1.4",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2))
    xm = x + w / 2
    y = oben - H_RAND
    if titel:
        ax.text(xm, y, titel, ha="center", va="top", fontsize=ts,
                color="#222222", zorder=3)
        y -= H_TITEL - 0.4
    for z in zeilen:
        ax.text(xm, y, z, ha="center", va="top", fontsize=bs, color=C_GRAU,
                zorder=3)
        y -= H_ZEILE
    return oben - h


def pfeil(x, y0, y1, lw=1.4):
    ax.add_patch(FancyArrowPatch(
        (x, y0), (x, y1), arrowstyle="-|>", mutation_scale=13,
        color=C_PFEIL, lw=lw, shrinkA=0, shrinkB=0, zorder=1))


def abschnitt(y, text):
    ax.text(0, y, text, fontsize=FS_ABSCHNITT, color="#222222", va="top")
    return y - 5.0


LUECKE = 4.6          # Pfeillaenge zwischen zwei Stufen
y = 151.0

# --------------------------------------------------------------- 1. Eingaben
y = abschnitt(y, "1   Inputs")

ein = [
    ("Ligand", ["fragmented into $N$ rigid parts",
                "nodes: ligand_atom, _anchor,",
                "_dummy, _virtual",
                "bonds, torsional bonds,",
                "fragment triangulation"]),
    ("Protein pocket", ["nodes: protein_atom,",
                        "protein_virtual",
                        "residue-type embedding",
                        "+ ESM positional embedding",
                        "pocket cutoff 6 / 5 $\\mathrm{\\AA}$" if G else ""]),
    ("Diffused state", ["$T_t,\\ R_t$ per fragment",
                        "$\\mathbf{x}^t_i = R_F(\\mathbf{x}^0_i-T^0_F)+T^t_F$",
                        "time $t \\sim \\mathcal{U}(0,1)$",
                        "sinusoidal embedding (32)" if G else ""]),
]
h_ein = max(hoehe(t, [z for z in zl if z]) for t, zl in ein)
for (titel, zeilen), x in zip(ein, (0, 34.5, 69)):
    kasten(x, y, 31, titel, zeilen, h=h_ein)
y_unten = y - h_ein
for x in (15.5, 50, 84.5):
    pfeil(x, y_unten, y_unten - LUECKE)
y = y_unten - LUECKE

# ------------------------------------------------------- 2. Heterogener Graph
y = kasten(0, y, 100, "Heterogeneous graph  $\\mathcal{G} = (V, E)$", [
    "6 node types  $\\cdot$  12 edge types: ligand_bonds, ligand_v2a, "
    "ligand_v2v, ligand_torsional_bond,",
    "fragment_triangulation, ligand_anchor_dummy, protein_bonds, "
    "protein_v2v, complex_lv2pv,",
    "inter_complex, inter_fragments  —  the last three rebuilt at every step "
    "from $\\mathbf{x}^t$ (cutoffs 4 / 15 / 18 $\\mathrm{\\AA}$)",
], bs=7.1)
pfeil(50, y, y - LUECKE)
y = y - LUECKE

# ------------------------------------------------------------ 3. EquiformerV2
y = abschnitt(y, "2   EquiformerV2")

innen = [
    ("Node embedding  ($\\ell = 0$)",
     ["one encoder per node type,",
      "time features concatenated $\\to$ 128 ch."]),
    ("Edge embedding",
     ["edge type + radial basis (Bessel/Fourier, 32),",
      "relative distance to $\\mathbf{x}^0$"]),
    ("Edge frames",
     ["one rotation per edge onto the $z$-axis, then Wigner-$D$ matrices, so "
      "the $SO(3)$ convolution",
      "reduces to $SO(2)$ operations — this is what makes "
      "$\\ell_{\\max} = 3$ affordable"]),
    ("$6\\times$  EquiformerV2 block",
     ["equivariant $SO(2)$ attention (4 heads, 64 hidden, 32 alpha, 16 value)"
      "  $\\cdot$  feed-forward (128)  $\\cdot$  layer norm",
      "node features carry degrees $\\ell = 0, 1, 2, 3$ throughout"]),
    ("Final layer norm  $\\to$  force block",
     ["the output is an $SO(3)$ embedding; only its $\\ell = 1$ part is kept"]),
]
kopf = 2.0 + (H_ZEILE if G else 0.0) + 4.4
h_paar = max(hoehe(*innen[0]), hoehe(*innen[1]))
h_netz = (kopf + h_paar + 1.6
          + sum(hoehe(*b) + 1.6 for b in innen[2:]) + 1.8)

ax.add_patch(FancyBboxPatch(
    (0, y - h_netz), 100, h_netz, boxstyle="round,pad=0,rounding_size=2.0",
    facecolor=C_NETZ, edgecolor=C_NETZ_R, linewidth=1.5, zorder=2))
yy = y - 2.2
ax.text(50, yy, "EquiformerV2  —  an $E(3)$-equivariant graph transformer",
        ha="center", va="top", fontsize=9.6, color="#222222", zorder=3)
yy -= 4.4
if G:
    ax.text(50, yy, "$\\ell_{\\max} = 3$,  $m_{\\max} = 2$,  6 layers,  "
                    "4 heads,  128 sphere channels,  32 edge channels",
            ha="center", va="top", fontsize=7.2, color="#8A6212", zorder=3)
    yy -= H_ZEILE

for (titel, zeilen), x in zip(innen[:2], (3.0, 51.5)):
    kasten(x, yy, 45.5, titel, zeilen, fc=C_INNEN, ec=C_NETZ_R,
           ts=8.4, bs=7.0, h=h_paar, lw=1.0)
yy -= h_paar
for titel, zeilen in innen[2:]:
    yy -= 1.6
    yy = kasten(3.0, yy, 94, titel, zeilen, fc=C_INNEN, ec=C_NETZ_R,
                ts=8.4, bs=7.0, lw=1.0)

y = y - h_netz
pfeil(50, y, y - LUECKE)
y = y - LUECKE

# --------------------------------------------------------------- 4. Ausgabe
y = kasten(14, y, 72, "Per-atom vector output   $f_i \\in \\mathbb{R}^3$", [
    "$f_i = -\\,\\epsilon_\\theta(\\mathcal{G}, t)_i$  at the unmasked ligand "
    "atoms  $\\cdot$  shape $[N_L, 3]$",
    "equivariant by construction: rotating the complex rotates every $f_i$",
], bs=7.2)
pfeil(50, y, y - LUECKE)
y = y - LUECKE

# --------------------------------------------- 5. Newton-Euler-Auslese
y = abschnitt(y, "3   Newton–Euler readout   (per fragment $F$)")

zeilen_lese = [
    "The zeroth moment of the field over a fragment is a force, the first "
    "moment a torque. $m_F$ is the atom count,",
    "$I_F(t)$ the inertia tensor at time $t$, and $\\hat{\\omega}_F = "
    "\\mathrm{hat}(\\Delta\\omega_F)$. This is the step that turns $3N_a$ "
    "atomic vectors into $6N$ rigid-body numbers.",
]
# 12.5 statt 9.5: die Summenzeichen mit Grenzen sind zwei Zeilen hoch.
# +2.4: mit va="top" ragt die letzte Textzeile unter ihren Ankerpunkt.
h_lese = 12.5 + len(zeilen_lese) * H_ZEILE + H_RAND + 2.4
ax.add_patch(FancyBboxPatch(
    (0, y - h_lese), 100, h_lese, boxstyle="round,pad=0,rounding_size=1.8",
    facecolor=C_LESE, edgecolor=C_LESE_R, linewidth=1.3, zorder=2))
yy = y - 2.6
for x, formel in ((3.5, "$F_F=\\sum_{i \\in F} f_i$"),
                  (26, "$\\tau_F=\\sum_{i \\in F}(\\mathbf{x}^t_i-T^t_F)"
                       "\\times f_i$"),
                  (62, "$\\Delta T_F = F_F / m_F$"),
                  (81, "$\\Delta\\omega_F = I_F(t)^{-1}\\tau_F$")):
    ax.text(x, yy, formel, fontsize=10.2, color="#3B2B57", va="top", zorder=3)
yy -= 12.5
for z in zeilen_lese:
    ax.text(50, yy, z, ha="center", va="top", fontsize=7.1, color=C_GRAU,
            zorder=3)
    yy -= H_ZEILE
y = y - h_lese
pfeil(50, y, y - LUECKE)
y = y - LUECKE

# ------------------------------------------------------------------ 6. Scores
zeilen_score = [
    "$\\lambda_{\\mathbb{R}^3}(t) = 1/\\sigma_t$;  $\\alpha$ comes from the "
    "$\\mathrm{IGSO}(3)$ density and is tabulated on a grid.",
    "Both are regressed against the true forward-process scores, each with "
    "its own $\\sigma$-dependent weight.",
]
# +2.4 statt H_RAND: mit va="top" ragt die letzte Zeile unter ihren Ankerpunkt.
h_score = 8.0 + len(zeilen_score) * H_ZEILE + H_RAND + 2.4
ax.add_patch(FancyBboxPatch(
    (0, y - h_score), 100, h_score, boxstyle="round,pad=0,rounding_size=1.8",
    facecolor=C_SCORE, edgecolor=C_SCORE_R, linewidth=1.3, zorder=2))
yy = y - 2.6
ax.text(10, yy, "$s_T = \\lambda_{\\mathbb{R}^3}(t)\\,\\Delta T_F$",
        fontsize=10.2, color="#7A2E10", va="top", zorder=3)
ax.text(48, yy, "$s_R = -\\,\\alpha(t, \\|\\Delta\\omega_F\\|)\\;"
                "\\hat{\\omega}_F\\,R_t$",
        fontsize=10.2, color="#7A2E10", va="top", zorder=3)
yy -= 8.0
for z in zeilen_score:
    ax.text(50, yy, z, ha="center", va="top", fontsize=7.1, color=C_GRAU,
            zorder=3)
    yy -= H_ZEILE
y = y - h_score

ax.text(100, y - 1.4, "Sizes are the defaults from config.py.",
        fontsize=6.6, color="#909090", ha="right", va="top")
ax.set_ylim(y - 6, 152)

AUSGABE.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(AUSGABE)
print(f"geschrieben: {AUSGABE}")
if AUCH_PDF:
    fig.savefig(AUSGABE.with_suffix(".pdf"))
    print(f"geschrieben: {AUSGABE.with_suffix('.pdf')}")
print(f"Unterkante bei y = {y:.1f}")
plt.close(fig)
