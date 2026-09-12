"""Zwei Modelle nebeneinander, Posen nach Bindungsmodus eingefaerbt.

WOZU DIESE ZWEITE SZENE
    `pymol_moden.py` legt EINEN Modus je Kachel und faerbt nach Erfolg. Das
    beantwortet "welcher Modus ist richtig". Die Frage hier ist eine andere:
    WIE VIELE Modi erzeugt ein Modell ueberhaupt. Dafuer braucht es beide
    Modelle im selben Bild und die Farbe am Modus statt am Erfolg.

WARUM GENAU ZWEI KACHELN
    PyMOL waehlt die Rasterform selbst aus der Zahl der Plaetze und laesst
    sich darin nicht steuern. Bei zwei Plaetzen ist das Ergebnis eindeutig
    nebeneinander; bei sechs oder zwoelf haengt es von PyMOLs Wurzelrechnung
    ab, und ein Bild, dessen Anordnung man nicht garantieren kann, taugt
    nicht fuer eine Abbildung.

DIE FARBEN SIND DIE AUSSAGE
    Ein Modus, eine Farbe, nach Groesse vergeben -- Modus 1 ist in beiden
    Kacheln dieselbe Farbe. Mehr verschiedene Farben rechts heisst mehr
    Modi. Die Kristallpose bleibt blau und dick, in beiden Kacheln.

VORAUSSETZUNG
    `pymol_moden.py` muss fuer denselben Komplex schon gelaufen sein; dieses
    Skript benutzt dessen exportierte SDF-Dateien weiter.
"""
import argparse
import io
import os
import re

_HIER = os.path.dirname(os.path.abspath(__file__))

# Weit auseinanderliegende Farben; Blau fehlt, es gehoert der Kristallpose.
FARBEN = ["firebrick", "orange", "forest", "purple", "magenta", "teal",
          "olive", "chocolate", "salmon", "deepteal", "wheat", "slate"]

p = argparse.ArgumentParser()
p.add_argument("--complex", required=True)
p.add_argument("--nfe", type=int, default=25)
p.add_argument("--arme", default="SigmaDock,Minimal")
p.add_argument("--zoom", type=float, default=10.0)
p.add_argument("--out", default=None)
a = p.parse_args()

WURZEL = os.path.join(_HIER, f"pymol_{a.complex}_nfe{a.nfe}")
if not os.path.isdir(WURZEL):
    raise SystemExit(f"ABBRUCH: {WURZEL} fehlt -- erst pymol_moden.py laufen "
                     f"lassen:\n  python pymol_moden.py --complex "
                     f"{a.complex} --nfe {a.nfe}")
ARME = [s.strip() for s in a.arme.split(",")]

z = ["# " + f"{a.complex}, {a.nfe} Schritte -- {' gegen '.join(ARME)}",
     "# Farbe = Bindungsmodus (Modus 1 in beiden Kacheln gleich gefaerbt).",
     "# blau: Kristallpose   dick: die vom Ranker gewaehlte Pose",
     "reinitialize", "bg_color white", "set ray_opaque_background, 1",
     "load protein.pdb, protein", "load kristall.sdf, kristall", ""]
raster, zaehl = [], {}

for slot, arm in enumerate(ARME, start=1):
    unter = os.path.join(WURZEL, arm)
    if not os.path.isdir(unter):
        raise SystemExit(f"ABBRUCH: {unter} fehlt.")
    # Modusdateien in Groessenreihenfolge; 'rest' sind die Einzelgaenger.
    moden = sorted({re.match(r"(m\d+|rest)_", f).group(1)
                    for f in os.listdir(unter)
                    if re.match(r"(m\d+|rest)_.*\.sdf$", f)},
                   key=lambda x: (x == "rest", x))
    zaehl[arm] = sum(1 for m in moden if m != "rest")
    z.append(f"# ---- Kachel {slot}: {arm} ----")
    for i, m in enumerate(moden):
        farbe = "grey60" if m == "rest" else FARBEN[i % len(FARBEN)]
        for teil in ("ja", "nein"):
            datei = os.path.join(unter, f"{m}_{teil}.sdf")
            if not os.path.isfile(datei):
                continue
            obj = f"{arm}_{m}_{teil}"
            z.append(f"load {arm}/{m}_{teil}.sdf, {obj}")
            z.append(f"color {farbe}, {obj}")
            raster.append(f"set grid_slot, {slot}, {obj}")
    gew = os.path.join(unter, "gewaehlt.sdf")
    if os.path.isfile(gew):
        z.append(f"load {arm}/gewaehlt.sdf, {arm}_gewaehlt")
        z.append(f"color black, {arm}_gewaehlt")
        raster.append(f"set grid_slot, {slot}, {arm}_gewaehlt")
    z.append("")

z += ["remove hydrogens", "hide everything", "set all_states, on", "",
      "# Protein als blasser Umriss, in BEIDEN Kacheln.",
      "show cartoon, protein", "color grey90, protein",
      "set cartoon_transparency, 0.85", "",
      "# Kristallpose: dick, blau, in beiden Kacheln dieselbe.",
      "show sticks, kristall", "set_bond stick_radius, 0.18, kristall",
      "color marine, kristall", "",
      "# Die Ziehungen duenn, damit die Zahl der Farben die Aussage traegt.",
      "set stick_radius, 0.055"]
for arm in ARME:
    z.append(f"show sticks, {arm}_*")
    z.append(f"set_bond stick_radius, 0.16, {arm}_gewaehlt")
z += ["", "set grid_mode, 1", f"set grid_max, {len(ARME)}",
      "set grid_slot, -2, protein", "set grid_slot, -2, kristall"] + raster
z += ["", f"orient kristall", f"zoom kristall, {a.zoom}",
      "set ray_shadows, 0", "set specular, 0.2", "set depth_cue, 0",
      "set opaque_background, 1", "",
      "# Drehen, dann @druck_vergleich.pml.",
      "# Modi je Kachel: " + ", ".join(f"{k} {v}" for k, v in zaehl.items())]

ziel = a.out or os.path.join(WURZEL, "vergleich.pml")
io.open(ziel, "w", encoding="utf-8").write("\n".join(z) + "\n")

druck = os.path.join(WURZEL, "druck_vergleich.pml")
io.open(druck, "w", encoding="utf-8").write(
    "# Rendert die AKTUELL im Fenster stehende Ansicht.\n"
    "get_view\nset ray_trace_mode, 0\nset antialias, 2\n"
    f"png {a.complex}_vergleich_nfe{a.nfe}.png, width=2200, height=1200, "
    "dpi=300, ray=1\n")

print(f"geschrieben: {ziel}")
print(f"             {druck}")
print(f"Modi: " + ", ".join(f"{k} {v}" for k, v in zaehl.items()))
print(f"\nIn PyMOL:\n  cd {WURZEL.replace(os.sep, '/')}\n  @vergleich.pml")
