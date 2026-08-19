# =====================================================================
#  Kristallpose gegen SigmaFlow und SigmaDock -- ein Komplex, alle Seeds.
#
#  ERWARTETE ORDNERSTRUKTUR (so, wie sie das scp-Block-Kommando anlegt):
#
#      <dieses Verzeichnis>/
#          6YRV_PJ8_protein.pdb
#          6YRV_PJ8_ligands.sdf          Kristall, ggf. mehrere Kopien
#          sigmaflow/*.sdf               zehn Seeds
#          sigmadock/*.sdf               zehn Seeds
#
#  Aufruf:   pymol view_complex.pml
#  Oder in einer laufenden Sitzung:   @view_complex.pml
#
#  WARUM ALLE ZEHN SEEDS UND NICHT NUR EINER
#      Das Ziehungsrauschen ist rund zehnmal so gross wie der
#      Methodenunterschied. Eine einzelne Pose zu zeigen hiesse, Rauschen
#      abzubilden. Die Wolke aus zehn Posen ist die ehrliche Darstellung:
#      sie zeigt, WIE BREIT das Modell streut, nicht nur wohin es einmal traf.
# =====================================================================

python
import glob, os
from pymol import cmd

# Der Komplex wird aus der Protein-Datei im Arbeitsverzeichnis abgeleitet.
# So laesst sich dasselbe Skript ohne Aenderung in jeden vis_<CID>-Ordner
# kopieren -- ein manuell gepflegter Name waere genau die Sorte Fehlerquelle,
# die still eine falsche Datei laedt.
_prot = sorted(glob.glob("*_protein.pdb"))
if not _prot:
    raise SystemExit("Keine <CID>_protein.pdb im Arbeitsverzeichnis gefunden. "
                     "Erst 'cd' in den vis_<CID>-Ordner.")
CID = os.path.basename(_prot[0])[:-len("_protein.pdb")]
print(f"[info] Komplex aus Dateinamen abgeleitet: {CID}")

cmd.reinitialize()
cmd.set("retain_order", 1)
cmd.set("pdb_conect_all", 1)

# ---- Protein: nur Kontext, bewusst zurueckhaltend ------------------
prot = f"{CID}_protein.pdb"
if os.path.exists(prot):
    cmd.load(prot, "PROT")
    cmd.hide("everything", "PROT")
    cmd.show("cartoon", "PROT")
    cmd.color("gray70", "PROT")
    cmd.set("cartoon_transparency", 0.72, "PROT")
else:
    print(f"[warn] {prot} nicht gefunden -- ohne Proteinkontext")

# ---- Kristallpose: die Referenz ------------------------------------
xtal = f"{CID}_ligands.sdf"
if os.path.exists(xtal):
    cmd.load(xtal, "XTAL")
    n = cmd.count_states("XTAL")
    if n > 1:
        print(f"[info] {n} kristallographische Kopien -- alle geladen, "
              f"gewertet wird gegen die naechstgelegene")
else:
    print(f"[FEHLER] {xtal} fehlt -- ohne Kristallpose ist der Vergleich sinnlos")

# ---- Vorhersagen: je ein Objekt pro Seed ---------------------------
def load_arm(folder, prefix):
    files = sorted(glob.glob(os.path.join(folder, "*.sdf")))
    for i, f in enumerate(files):
        cmd.load(f, f"{prefix}_{i:02d}")
    print(f"[info] {prefix}: {len(files)} Posen aus {folder}/")
    return len(files)

n_sf = load_arm("sigmaflow", "SF")
n_sd = load_arm("sigmadock", "SD")

# ---- Darstellung ----------------------------------------------------
cmd.hide("everything", "XTAL SF_* SD_*")
cmd.show("sticks", "XTAL SF_* SD_*")

# Palette identisch zu den Thesis-Abbildungen.
cmd.set_color("c_xtal", [0.85, 0.68, 0.22])   # Gold  = Wahrheit
cmd.set_color("c_sf",   [0.757, 0.400, 0.420])  # #C1666B
cmd.set_color("c_sd",   [0.310, 0.541, 0.545])  # #4F8A8B

cmd.color("c_xtal", "XTAL")
cmd.color("c_sf", "SF_*")
cmd.color("c_sd", "SD_*")
cmd.util.cnc("XTAL")                            # Heteroatome behalten ihre Farbe

# Die Referenz dick, die Vorhersagen duenn -- sonst erschlaegt die Wolke sie.
cmd.set("stick_radius", 0.17, "XTAL")
cmd.set("stick_radius", 0.055, "SF_*")
cmd.set("stick_radius", 0.055, "SD_*")
cmd.set("stick_transparency", 0.45, "SF_*")
cmd.set("stick_transparency", 0.45, "SD_*")

cmd.group("SigmaFlow", "SF_*")
cmd.group("SigmaDock", "SD_*")

# ---- Kamera ---------------------------------------------------------
cmd.orient("XTAL")
cmd.zoom("XTAL", 5)
cmd.set("ray_opaque_background", 0)
cmd.set("orthoscopic", 1)
cmd.set("antialias", 2)
cmd.set("ambient", 0.28)
cmd.set("specular", 0.15)
cmd.bg_color("white")

# ---- Szenen ---------------------------------------------------------
def scene(name, shown):
    cmd.disable("SigmaFlow SigmaDock XTAL PROT")
    for o in shown:
        cmd.enable(o)
    cmd.scene(name, "store")

scene("01_kristall",   ["PROT", "XTAL"])
scene("02_sigmaflow",  ["PROT", "XTAL", "SigmaFlow"])
scene("03_sigmadock",  ["PROT", "XTAL", "SigmaDock"])
scene("04_beide",      ["PROT", "XTAL", "SigmaFlow", "SigmaDock"])
cmd.scene("04_beide", "recall")

print("")
print("=" * 62)
print(f"  {CID}   Kristall + {n_sf} SigmaFlow + {n_sd} SigmaDock")
print("=" * 62)
print("  Szenen durchblaettern:   Bild auf / Bild ab")
print("  scene 01_kristall | 02_sigmaflow | 03_sigmadock | 04_beide")
print("")
print("  best              nur die beste Pose je Arm zeigen")
print("  seed 3            nur Seed 3 beider Arme")
print("  frag 2            Fragment 2 hervorheben (nach Kabsch-Reihenfolge)")
print("  spread            Streuung als Abstandslinien zur Kristallpose")
print("=" * 62)
python end


# ---------------------------------------------------------------------
#  Hilfsbefehle
# ---------------------------------------------------------------------
python
from pymol import cmd, stored

def best(cid=None):
    """Zeigt je Arm die Pose mit dem kleinsten RMSD zur Kristallpose.

    Gerechnet wird mit rms_cur ohne Ausrichtung -- die Posen liegen bereits
    im selben Bezugssystem, ein Fit wuerde genau den Fehler wegrechnen, den
    wir sehen wollen.
    """
    for prefix, label in (("SF", "SigmaFlow"), ("SD", "SigmaDock")):
        objs = sorted(o for o in cmd.get_object_list() if o.startswith(prefix + "_"))
        vals = []
        for o in objs:
            try:
                vals.append((cmd.rms_cur(o, "XTAL", matchmaker=-1), o))
            except Exception:
                pass
        if not vals:
            continue
        vals.sort()
        keep = vals[0][1]
        for o in objs:
            cmd.disable(o)
        cmd.enable(keep)
        cmd.set("stick_radius", 0.11, keep)
        cmd.set("stick_transparency", 0.0, keep)
        print(f"  {label:10s} beste Pose: {keep}   RMSD {vals[0][0]:.2f} A")
    cmd.enable("XTAL")
cmd.extend("best", best)

def seed(n):
    """Nur einen bestimmten Seed beider Arme zeigen."""
    n = int(n)
    for prefix in ("SF", "SD"):
        for o in cmd.get_object_list():
            if o.startswith(prefix + "_"):
                cmd.disable(o)
        tgt = f"{prefix}_{n:02d}"
        if tgt in cmd.get_object_list():
            cmd.enable(tgt)
            cmd.set("stick_radius", 0.11, tgt)
            cmd.set("stick_transparency", 0.0, tgt)
    cmd.enable("XTAL")
    print(f"  Seed {n} beider Arme")
cmd.extend("seed", seed)

def spread(prefix="SF"):
    """Abstandslinien von jeder Pose zur Kristallpose -- macht die Streuung sichtbar."""
    cmd.delete("spread_*")
    objs = sorted(o for o in cmd.get_object_list() if o.startswith(prefix + "_"))
    for o in objs:
        cmd.distance(f"spread_{o}", f"{o} and name C*", "XTAL and name C*", mode=4)
    cmd.hide("labels", "spread_*")
    cmd.set("dash_radius", 0.012)
    cmd.color("gray50", "spread_*")
    print(f"  Streuung von {len(objs)} {prefix}-Posen zur Kristallpose")
cmd.extend("spread", spread)

def frag(n):
    """Ein Fragment hervorheben. Die Nummerierung folgt der Atomreihenfolge
    des Liganden, nicht der Fragmentierung des Modells -- als grobe Orientierung
    fuer 'welcher Teil sitzt richtig' ist sie trotzdem brauchbar."""
    n = int(n)
    cmd.color("gray80", "SF_* SD_*")
    sel = f"(SF_* or SD_*) and rank {n*3}-{n*3+2}"
    cmd.color("magenta", sel)
    cmd.show("spheres", sel)
    cmd.set("sphere_scale", 0.22, sel)
    print(f"  Fragment {n} hervorgehoben (Atomraenge {n*3}-{n*3+2})")
cmd.extend("frag", frag)
python end
