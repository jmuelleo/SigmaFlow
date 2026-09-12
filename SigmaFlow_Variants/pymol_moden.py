"""PyMOL-Szenen: die Ziehungen eines Komplexes, nach Modus aufgeteilt.

ZWEI SZENEN JE ARM

    raster.pml   Ein Modus je KACHEL (`grid_mode`). Die Kachel kodiert den
                 Modus, also ist die FARBE frei fuer die eigentliche Aussage:
                 gruen erfuellt das Kriterium, rot nicht. Das ist die
                 Abbildung fuer die Arbeit.

    szene.pml    Alle Moden uebereinander in einem Bild, nach Modus
                 eingefaerbt. Gut, um die Gesamtstreuung zu sehen, aber bei
                 mehr als vier Moden schnell unleserlich.

WARUM DIE POSEN JE MODUS AUF ZWEI OBJEKTE VERTEILT WERDEN
    PyMOL laedt eine mehrteilige SDF als ZUSTAENDE eines Objekts. Farbe ist
    aber eine Eigenschaft der Atome, nicht des Zustands -- alle Zustaende
    eines Objekts haben zwangslaeufig dieselbe Farbe. Um innerhalb eines
    Modus zwischen richtig und falsch zu unterscheiden, braucht es deshalb
    zwei Objekte, die derselben Kachel zugewiesen werden.

    Das ist kein Umweg, sondern die einzige Moeglichkeit. Wer es nicht weiss,
    sucht lange nach einem `color ..., state=3`, das es nicht gibt.

DIE DREI ENTSCHEIDUNGEN, DIE DAS BILD LESBAR MACHEN

    1. IDENTISCHE KAMERA. Im Rastermodus gilt die Kamera GLOBAL: ein
       `orient`, ein Massstab fuer alle Kacheln. Genau deshalb ist das Raster
       dem Nebeneinanderlegen einzeln gerenderter Bilder vorzuziehen -- dort
       waere der Zoom je Bild neu gesetzt, und ein Groessenunterschied waere
       ein Kameraartefakt statt eines Befunds. Zwischen den ARMEN wird
       derselbe Zoomradius eingetragen, berechnet aus der Ausdehnung aller
       Arme zusammen.

    2. AUSREISSER IN EINE KACHEL. Ein Cluster aus einer einzigen Ziehung ist
       kein Modus. Bekaeme jeder eine eigene Kachel, saehe ein Arm mit vielen
       Einzelgaengern vielfaeltiger aus, als er ist. Alles unter --min-modus
       landet zusammen in der letzten Kachel.

    3. KACHELN NACH GROESSE. Kachel 1 ist immer der groesste Modus. So steht
       in jedem Bild dasselbe an derselben Stelle.

DAS CLUSTERN MUSS ZU DEN TABELLEN PASSEN
    Paarweiser RMSD ohne Ueberlagerung, VOLLSTAENDIGE Verkettung, Schnitt bei
    2 Angstroem -- wie in a_datensatz.py und auf der Modenseite. Ein Bild,
    das anders clustert als die Tabelle daneben, waere schlimmer als kein
    Bild; genau das war bis 2026-09-09 der Fall, weil hier `average` stand
    und in den Tabellen `complete`. Average verschmilzt grosszuegiger und
    zeigt deshalb zu wenige Modi. Umschaltbar ueber --verkettung.

Aufruf:
    python SigmaFlow_Variants/pymol_moden.py --complex 7TE8_P0T
    python SigmaFlow_Variants/pymol_moden.py --complex 7TE8_P0T --ziel beides
"""
import argparse
import glob
import io
import os
import re
import shutil
import sys

import numpy as np

_HIER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HIER)
import posencache  # noqa: E402
from zellen import SAETZE, posenordner  # noqa: E402

from scipy.cluster.hierarchy import fcluster, linkage  # noqa: E402
from scipy.spatial.distance import squareform  # noqa: E402

# Der SDF-Datensatztrenner, zusammengesetzt statt als Literal geschrieben --
# vier Dollarzeichen in einer Quelldatei sind eine Einladung an jede Shell und
# jedes Patch-Werkzeug, sie zu ersetzen.
TRENNER = "$" * 4

GRUEN, ROT = "forest", "firebrick"          # duenne Posen
GRUEN_HELL, ROT_HELL = "green", "red"       # die gewaehlte Pose, dick
KRISTALL = "marine"
# Nur fuer die Ueberlagerungsszene, wo die Farbe den Modus tragen muss.
MODUSFARBEN = ["firebrick", "orange", "forest", "purple", "teal", "hotpink",
               "olive", "slate", "chocolate", "deepteal", "raspberry", "limon"]
GRAU = "grey60"


def ein_datensatz(pfad):
    """SDF-Text mit GENAU EINEM abschliessenden Trenner.

    Die Quelldateien enden bereits auf der Trennerzeile. Haengt man blind eine
    weitere an, stehen zwei hintereinander, und PyMOL laedt dazwischen einen
    leeren Zustand -- ein Objekt haette dann doppelt so viele Zustaende wie
    Posen. In der Datei faellt das nicht auf, im Bild erst, wenn die Abbildung
    schon in der Arbeit steht.
    """
    t = io.open(pfad, encoding="utf-8", errors="replace").read().rstrip()
    while t.endswith(TRENNER):
        t = t[:-len(TRENNER)].rstrip()
    return t + "\n" + TRENNER + "\n"


p = argparse.ArgumentParser()
p.add_argument("--complex", required=True, help="z.B. 7TE8_P0T")
p.add_argument("--satz", choices=sorted(SAETZE), default="pb308")
p.add_argument("--nfe", type=int, default=25)
p.add_argument("--arme", default="SigmaDock,Minimal,Separate")
p.add_argument("--schwelle", type=float, default=2.0,
               help="Clusterschwelle in Angstroem")
p.add_argument("--ziel", default="beides", choices=["acc", "beides"],
               help="beides = RMSD unter 2 A UND PB-valide (Vorgabe, weil "
                    "das das Zielkriterium der Arbeit ist und zugleich das, "
                    "was der Mixed Score des Rankers optimiert); acc = nur "
                    "RMSD, wenn eine Abbildung allein ueber die LAGE "
                    "gemeint ist.")
p.add_argument("--verkettung", default="complete",
               choices=["complete", "average"],
               help="MUSS zu den Tabellen passen. a_datensatz.py und die "
                    "Modenseite clustern mit complete; struktur.py benutzte "
                    "frueher average, was groessere und weniger Cluster gibt.")
p.add_argument("--min-modus", type=int, default=2,
               help="Cluster darunter kommen zusammen in die Ausreisserkachel")
p.add_argument("--perzentil", type=float, default=98.0,
               help="Anteil der Atome, die ins Bild passen muessen; 100 = alle")
p.add_argument("--zoom", type=float, default=None,
               help="Zoomradius in A fest vorgeben")
p.add_argument("--puffer", type=float, default=3.0,
               help="zusaetzlicher Rand um die weiteste Pose, in Angstroem")
p.add_argument("--breite", type=int, default=1800)
p.add_argument("--hoehe", type=int, default=1300)
p.add_argument("--labels", action="store_true",
               help="Kachelbeschriftung ins 3D-Bild statt in die "
                    "Bildunterschrift (haengt an Kamera und Zoom)")
p.add_argument("--out", default=None)
a = p.parse_args()

CODE = a.complex
ARME = [s.strip() for s in a.arme.split(",")]
ZIEL = a.out or os.path.join(_HIER, f"pymol_{CODE}_nfe{a.nfe}")
SATZ = SAETZE[a.satz]

kristall = os.path.join(SATZ["referenz"], CODE, f"{CODE}_ligand.sdf")
protein = os.path.join(SATZ["referenz"], CODE, f"{CODE}_protein.pdb")
for f in (kristall, protein):
    if not os.path.isfile(f):
        sys.exit(f"ABBRUCH: {f} fehlt.")

os.makedirs(ZIEL, exist_ok=True)
shutil.copy(kristall, os.path.join(ZIEL, "kristall.sdf"))
shutil.copy(protein, os.path.join(ZIEL, "protein.pdb"))


def dateien_von(arm):
    """Dieselbe Reihenfolge, in der posencache die Posen abgelegt hat."""
    w = posenordner(a.satz, arm, a.nfe)
    return sorted(glob.glob(os.path.join(w, "seed_*", f"{CODE}__*.sdf")),
                  key=lambda f: int(re.search(r"seed_(\d+)", f).group(1)))


# --- erst alles einlesen, damit die gemeinsame Kamera bekannt ist ---------
szenen, alle_xyz = {}, []
for arm in ARME:
    daten = posencache.hole(a.satz, arm, a.nfe, leise=True)
    if CODE not in daten:
        sys.exit(f"ABBRUCH: {CODE} fehlt in {arm}/nfe{a.nfe}.")
    v = daten[CODE]
    if a.ziel not in v:
        sys.exit(f"ABBRUCH: Spalte '{a.ziel}' fehlt im Cache fuer {arm}.")
    d, x = v["d"].astype(float), v["xyz"].astype(float)
    n = len(d)
    lab = fcluster(linkage(squareform(d, checks=False), method=a.verkettung),
                   t=a.schwelle, criterion="distance")
    groessen = np.bincount(lab)[1:]
    dateien = dateien_von(arm)
    if len(dateien) != n:
        sys.exit(f"ABBRUCH: {arm}: {len(dateien)} Dateien, aber {n} Posen im "
                 f"Cache. Reihenfolge nicht verlaesslich -- Cache loeschen.")
    sieger = int(np.argmax(v["heur"]))
    szenen[arm] = dict(lab=lab, groessen=groessen, dateien=dateien, n=n,
                       reihenfolge=np.argsort(-groessen) + 1,
                       ok=np.asarray(v[a.ziel], bool), sieger=sieger)
    alle_xyz.append(x.reshape(-1, 3))

from rdkit import Chem, RDLogger  # noqa: E402
RDLogger.DisableLog("rdApp.*")
kmol = Chem.MolFromMolFile(kristall, sanitize=False, removeHs=False)
zentrum = kmol.GetConformer().GetPositions().mean(axis=0)
# NICHT das Maximum, sondern ein Perzentil: bei fuenf Integrationsschritten
# wirft der Diffusionssampler einzelne Posen zehnfach aus der Tasche, und die
# Kamera wuerde sich nach diesen Ausreissern richten -- gemessen 84,9 A statt
# der 12 A, in denen sich alles Interessante abspielt. Das Bild waere leer mit
# einem Punkt in der Mitte. Wer wirklich alles sehen will, setzt --perzentil 100.
d_alle = np.linalg.norm(np.vstack(alle_xyz) - zentrum, axis=1)
radius = float(np.percentile(d_alle, a.perzentil))
ZOOM = (a.zoom if a.zoom else radius + a.puffer)
n_ausserhalb = int((d_alle > radius).sum())
ZT = ", ".join(f"{v:.3f}" for v in zentrum)

KRIT = ("RMSD < 2 A" if a.ziel == "acc" else "RMSD < 2 A und PB-valide")
LEGENDEN = {}
print(f"{CODE}, {a.nfe} Schritte, Kriterium {KRIT}  ->  {ZIEL}")
print(f"  gemeinsamer Zoomradius {ZOOM:.1f} A um den Kristallschwerpunkt\n")

for arm in ARME:
    s = szenen[arm]
    unter = os.path.join(ZIEL, arm)
    os.makedirs(unter, exist_ok=True)
    ok, n = s["ok"], s["n"]

    # --- Kacheln bilden und je Kachel nach richtig/falsch trennen --------
    kacheln, rest_idx = [], []
    for cl in s["reihenfolge"]:
        idx = np.where(s["lab"] == cl)[0]
        if len(idx) < a.min_modus:
            rest_idx.extend(idx.tolist())
            continue
        kacheln.append((f"m{len(kacheln) + 1:02d}", idx))
    if rest_idx:
        kacheln.append(("rest", np.asarray(sorted(rest_idx))))

    objekte = []          # (objektname, slot, ist_richtig, anzahl)
    for slot, (basis, idx) in enumerate(kacheln, 1):
        for richtig, suffix in ((True, "ja"), (False, "nein")):
            teil = [int(i) for i in idx if bool(ok[i]) == richtig]
            if not teil:
                continue
            name = f"{basis}_{suffix}"
            with io.open(os.path.join(unter, f"{name}.sdf"), "w",
                         encoding="utf-8") as aus:
                for i in teil:
                    aus.write(ein_datensatz(s["dateien"][i]))
            objekte.append((name, slot, richtig, len(teil)))

    with io.open(os.path.join(unter, "gewaehlt.sdf"), "w",
                 encoding="utf-8") as aus:
        aus.write(ein_datensatz(s["dateien"][s["sieger"]]))
    sieger_slot = next((slot for slot, (_, idx) in enumerate(kacheln, 1)
                        if s["sieger"] in set(idx.tolist())), 1)
    sieger_ok = bool(ok[s["sieger"]])

    kopf = [
        "reinitialize",
        "bg_color white",
        "set ray_opaque_background, 1",
        "load ../protein.pdb, protein",
        "load ../kristall.sdf, kristall",
        "load gewaehlt.sdf, gewaehlt",
    ] + [f"load {name}.sdf, {name}" for name, _, _, _ in objekte] + [
        "",
        "# Wasserstoffe raus: sie verdoppeln die Linien und tragen nichts bei.",
        "remove hydrogens",
        "hide everything",
        "set all_states, on",
        "",
        "# Protein nur als blasser Umriss.",
        "show cartoon, protein",
        "color grey90, protein",
        "set cartoon_transparency, 0.80",
        "",
        "# Die Kristallpose: dick und blau, in jedem Bild dieselbe.",
        "show sticks, kristall",
        "set_bond stick_radius, 0.18, kristall",
        f"color {KRISTALL}, kristall",
    ]
    fuss = [
        "",
        "orient kristall",
        f"zoom kristall, {ZOOM - 1:.1f}",
        "set ray_shadows, 0",
        "set specular, 0.2",
        "set depth_cue, 0",
    ]

    # ------------------------------------------------------------------
    # raster.pml -- ein Modus je Kachel, Farbe = Kriterium
    # ------------------------------------------------------------------
    ras = [f"# {CODE}, {arm}, {a.nfe} Schritte -- ein Modus je Kachel",
           f"# green: {KRIT}   red: fails   blue: crystal pose   "
           f"thick: the ranked pose"] + kopf + [
        "",
        "# Die Ziehungen: duenn, gruen wenn das Kriterium erfuellt ist.",
        "set stick_radius, 0.06",
    ]
    for name, _, richtig, _ in objekte:
        ras.append(f"show sticks, {name}")
        ras.append(f"color {GRUEN if richtig else ROT}, {name}")
    ras += [
        "",
        "# Die gewaehlte Pose: dick, in der Farbe ihres eigenen Ausgangs.",
        "show sticks, gewaehlt",
        "set_bond stick_radius, 0.16, gewaehlt",
        f"color {GRUEN_HELL if sieger_ok else ROT_HELL}, gewaehlt",
        "",
        "# Raster. -2 heisst: in ALLEN Kacheln zeigen.",
        "set grid_mode, 1",
        f"set grid_max, {len(kacheln)}",
        "set grid_slot, -2, protein",
        "set grid_slot, -2, kristall",
    ]
    for name, slot, _, _ in objekte:
        ras.append(f"set grid_slot, {slot}, {name}")
    ras.append(f"set grid_slot, {sieger_slot}, gewaehlt")
    # Beschriftung im 3D-Bild ist standardmaessig AUS.
    #
    # Sie haengt an Kamera, Kachelgroesse und Versatz zugleich: ein Versatz,
    # der bei zwei Kacheln ueber dem Molekuel sitzt, liegt bei drei Kacheln
    # ausserhalb des Feldes, und beim Zoomradius von 27,9 A der 5-Schritt-
    # Tafeln stimmt wieder ein anderer. Dazu wandert sie bei jeder Drehung mit.
    #
    # Die Kachelreihenfolge ist ohnehin festgelegt -- Kachel 1 ist immer der
    # groesste Modus --, also gehoert die Legende in die Bildunterschrift.
    # Das Skript gibt sie am Ende als fertigen Text aus. Wer sie doch im Bild
    # will, nimmt --labels.
    ras += ([
        "",
        "# Beschriftung je Kachel ueber ein Pseudoatom im Schwerpunkt der",
        "# Kristallpose; label_position schiebt die Schrift nach oben.",
    ] if a.labels else [])
    # Beschriftung ENGLISCH -- die Abbildung geht in eine englische Arbeit,
    # und eine deutsche Kachelbeschriftung waere das Erste, was auffiele.
    legende = []
    for slot, (basis, idx) in enumerate(kacheln, 1):
        titel = "Outliers" if basis == "rest" else f"Mode {slot}"
        n_ok = int(ok[idx].sum())
        text = (f"{titel}: {len(idx)}/{n} ({100 * len(idx) / n:.0f}%), "
                f"{n_ok} correct")
        if slot == sieger_slot:
            text += "   <- ranked"
        legende.append(text)
        if not a.labels:
            continue
        # UEBER DIE PYTHON-SCHNITTSTELLE, nicht als PyMOL-Kommando.
        # `pseudoatom name, pos=..., label="a, b"` trennt seine Argumente am
        # KOMMA -- die Beschriftung bricht dort ab, und im Bild fehlt alles
        # hinter dem ersten Komma. Mit cmd.pseudoatom() gibt es dieses
        # Parsing nicht.
        ras.append(f'python')
        ras.append(f'cmd.pseudoatom("titel{slot:02d}", pos=[{ZT}], '
                   f'label={text!r})')
        ras.append(f'python end')
        ras.append(f"set grid_slot, {slot}, titel{slot:02d}")
    ras += (["set label_size, 15", "set label_color, black",
             "set label_bg_color, white", "set label_bg_transparency, 0.2",
             f"set label_position, (0, {ZOOM * 0.45:.1f}, 0)"]
            if a.labels else []) + fuss + [
        "",
        "# Gerendert wird NICHT hier. Erst von Hand drehen, dann @druck.pml.",
    ]
    with io.open(os.path.join(unter, "raster.pml"), "w",
                 encoding="utf-8") as f:
        f.write("\n".join(ras) + "\n")

    # ------------------------------------------------------------------
    # szene.pml -- alles uebereinander, Farbe = Modus
    # ------------------------------------------------------------------
    sz = [f"# {CODE}, {arm}, {a.nfe} Schritte -- alle Moden uebereinander"
          ] + kopf + ["", "set stick_radius, 0.05"]
    for name, slot, _, _ in objekte:
        farbe = (GRAU if name.startswith("rest")
                 else MODUSFARBEN[(slot - 1) % len(MODUSFARBEN)])
        sz.append(f"show sticks, {name}")
        sz.append(f"color {farbe}, {name}")
    sz += ["", "show sticks, gewaehlt",
           "set_bond stick_radius, 0.14, gewaehlt",
           "color black, gewaehlt"] + fuss
    with io.open(os.path.join(unter, "szene.pml"), "w", encoding="utf-8") as f:
        f.write("\n".join(sz) + "\n")

    # ------------------------------------------------------------------
    # druck.pml -- rendert die AKTUELLE Ansicht, wie sie gerade im Fenster
    # steht. Bewusst getrennt von der Szene: `orient` und `zoom` setzen die
    # Kamera zurueck, ein Rendern am Ende der Szene wuerde also jede von Hand
    # gedrehte Ansicht wieder verwerfen.
    #
    # `get_view` schreibt die Ansicht zusaetzlich ins Protokoll. Wer die
    # Bilder der drei Arme deckungsgleich haben will, kopiert die ausgegebene
    # set_view-Klammer und fuegt sie in den anderen Armen ein, BEVOR er dort
    # druckt -- sonst ist der Unterschied zwischen den Bildern teilweise
    # Kamera und nicht Modell.
    # ------------------------------------------------------------------
    dr = [
        f"# {CODE}, {arm}, {a.nfe} Schritte -- rendert die aktuelle Ansicht",
        "set ray_shadows, 0",
        "set antialias, 2",
        "set ray_opaque_background, 1",
        "get_view",
        f"ray {a.breite}, {a.hoehe}",
        f"png {CODE}_{arm}_nfe{a.nfe}.png, dpi=300",
        f'print "geschrieben: {CODE}_{arm}_nfe{a.nfe}.png"',
    ]
    with io.open(os.path.join(unter, "druck.pml"), "w", encoding="utf-8") as f:
        f.write("\n".join(dr) + "\n")

    LEGENDEN[arm] = legende
    n_moden = len(kacheln) - (1 if rest_idx else 0)
    print(f"  {arm:<11} {n} Posen, {n_moden} Moden + "
          f"{len(rest_idx)} Ausreisser, {int(ok.sum())} korrekt")
    print(f"    groesster Modus {100 * s['groessen'].max() / n:.0f} %, "
          f"Wahl in Kachel {sieger_slot}, "
          f"{'korrekt' if sieger_ok else 'falsch'}")

print("\nIn PyMOL eingeben -- aufbauen, drehen, dann drucken:")
for arm in ARME:
    print(f"  cd {os.path.join(ZIEL, arm).replace(os.sep, '/')}")
    print("  @raster.pml        # Kacheln aufbauen")
    print("  @druck.pml         # rendert die aktuelle Ansicht")
