"""Baut die Daten fuer die interaktive Modenseite.

WAS ERZEUGT WIRD
    Eine JSON-Datei mit allen Posen der ausgewaehlten Komplexe, ihrer
    Kristallpose, der Bindungsliste, dem RMSD jeder Pose zur Wahrheit, ihrer
    PoseBusters-Validitaet und -- fuer jede angebotene Clusterschwelle -- der
    Modusnummer jeder Pose.

ZWEI SCHWELLEN, DIE BEIDE "2 ANGSTROEM" HEISSEN UND NICHTS MITEINANDER ZU TUN
HABEN
    CLUSTERSCHWELLE   Wie aehnlich muessen zwei Posen sein, um als derselbe
                      Bindungsmodus zu gelten? Das ist unsere Wahl.
    ERFOLGSKRITERIUM  Wie nah muss eine Pose der Kristallstruktur sein, um als
                      richtig zu zaehlen? Das ist der Feldstandard.

    Die Seite laesst beide getrennt einstellen. Deshalb liegen hier die
    Modusnummern fuer JEDE Clusterschwelle vor und der RMSD als ZAHL statt als
    Boolean -- das Erfolgskriterium wertet der Browser dann selbst aus.

WARUM DER RMSD HIER NEU GERECHNET WIRD
    Die Auswertungstabellen fuehren nur `rmsd <= 2 A` als Boolean. Der
    Zahlenwert wird deshalb aus den GECACHTEN Koordinaten gerechnet -- nicht
    aus der CSV von `rmsd_streng.py`, weil deren Zeilen ueber (Komplex, Seed)
    zugeordnet werden muessten und der Cache die Seednummern nicht fuehrt.
    Eine stille Fehlzuordnung waere hier besonders unangenehm, weil sie im
    Bild nicht auffiele.

    Symmetriekorrigiert ueber `CalcRMS`: dafuer wird die Kristallpose kopiert
    und ihr Konformer auf die Posenkoordinaten gesetzt. So ist die Zuordnung
    zum Cache exakt und die Symmetrie trotzdem beruecksichtigt.

Aufruf:
    python SigmaFlow_Variants/baue_moden_seite.py
    python SigmaFlow_Variants/baue_moden_seite.py --komplexe 6XM9_V55,7PGX_FMN
"""
import argparse
import json
import os
import sys

import numpy as np

_HIER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HIER)
import posencache  # noqa: E402
from zellen import SAETZE  # noqa: E402

from rdkit import Chem, RDLogger  # noqa: E402
RDLogger.DisableLog("rdApp.*")
from rdkit.Chem import rdMolAlign  # noqa: E402
from rdkit.Geometry import Point3D  # noqa: E402

from scipy.cluster.hierarchy import fcluster, linkage  # noqa: E402
from scipy.spatial.distance import squareform  # noqa: E402

ARME = [("SigmaDock", "SigmaDock"),
        ("Minimal", "SigmaFlow-Minimal"),
        ("Separate", "SigmaFlow-Separate")]

p = argparse.ArgumentParser()
p.add_argument("--komplexe",
               default="7PIH_7QW,6XM9_V55,7SFO_98L,7OFK_VCH,7PGX_FMN,"
                       "7TE8_P0T,7L00_XCJ")
p.add_argument("--satz", default="pb308")
p.add_argument("--nfe", type=int, default=25)
p.add_argument("--schwellen", default="1.0,1.5,2.0",
               help="angebotene Clusterschwellen")
p.add_argument("--methode", default="complete")
p.add_argument("--out", default=os.path.join(_HIER, "moden_daten.json"))
a = p.parse_args()
CODES = [c.strip() for c in a.komplexe.split(",")]
SCHW = [float(x) for x in a.schwellen.split(",")]
SATZ = SAETZE[a.satz]

H = {arm: posencache.hole(a.satz, arm, a.nfe, leise=True) for arm, _ in ARME}


def rmsd_zur_wahrheit(kris, xyz):
    """Symmetriekorrigierter RMSD jeder Pose zur Kristallpose.

    Die Kristallpose wird kopiert und ihr Konformer auf die Posenkoordinaten
    gesetzt. Damit ist der Molekuelgraph identisch -- CalcRMS kann seine
    Automorphismen durchprobieren -- und die Zuordnung zum Cache exakt.
    """
    aus = []
    for pos in xyz:
        m = Chem.Mol(kris)
        c = m.GetConformer()
        for i, q in enumerate(pos):
            c.SetAtomPosition(i, Point3D(float(q[0]), float(q[1]), float(q[2])))
        try:
            aus.append(round(float(rdMolAlign.CalcRMS(m, kris)), 3))
        except Exception:
            aus.append(round(float(np.sqrt(((pos - kris.GetConformer()
                                             .GetPositions()) ** 2)
                                           .sum(-1).mean())), 3))
    return aus


daten = {"nfe": a.nfe, "schwellen": SCHW, "methode": a.methode, "komplexe": []}
for code in CODES:
    kris_p = os.path.join(SATZ["referenz"], code, f"{code}_ligand.sdf")
    kris = Chem.MolFromMolFile(kris_p, sanitize=True, removeHs=True)
    if kris is None:
        print(f"  {code}: Kristallpose nicht ladbar -- uebersprungen")
        continue
    elemente = [at.GetSymbol() for at in kris.GetAtoms()]
    bindungen = [[b.GetBeginAtomIdx(), b.GetEndAtomIdx(),
                  int(b.GetBondTypeAsDouble()) or 1] for b in kris.GetBonds()]
    eintrag = {"code": code, "elemente": elemente, "bindungen": bindungen,
               "kristall": np.round(kris.GetConformer().GetPositions(),
                                    2).tolist(),
               "arme": []}

    for arm, lang in ARME:
        if code not in H[arm]:
            continue
        v = H[arm][code]
        d = v["d"].astype(float)
        xyz = v["xyz"].astype(float)
        if xyz.shape[1] != len(elemente):
            print(f"  {code}/{arm}: {xyz.shape[1]} Atome gegen "
                  f"{len(elemente)} -- uebersprungen")
            continue
        n = len(d)
        Z = linkage(squareform(d, checks=False), method=a.methode)
        moden = {}
        for s in SCHW:
            lab = fcluster(Z, t=s, criterion="distance")
            g = np.bincount(lab)[1:]
            # Nach Groesse umnummerieren, damit Modus 0 immer der groesste ist.
            rang = {int(cl): k for k, cl in enumerate(np.argsort(-g) + 1)}
            moden[f"{s:.1f}"] = [rang[int(x)] for x in lab]
        eintrag["arme"].append({
            "kurz": arm, "name": lang,
            "posen": np.round(xyz, 2).tolist(),
            "rmsd": rmsd_zur_wahrheit(kris, xyz),
            "valid": [int(x) for x in np.asarray(v["valid"], bool)],
            "sieger": int(np.argmax(v["heur"])),
            "moden": moden,
        })
    if eintrag["arme"]:
        daten["komplexe"].append(eintrag)
        s2 = f"{SCHW[-1]:.1f}"
        print(f"  {code}: {len(elemente)} Atome, "
              + ", ".join(f"{x['kurz']} {len(set(x['moden'][s2]))} Cl@{s2}"
                          for x in eintrag["arme"]))

with open(a.out, "w", encoding="utf-8") as f:
    json.dump(daten, f, separators=(",", ":"))
print(f"\n{len(daten['komplexe'])} Komplexe, Schwellen {SCHW}, "
      f"nach {a.out} ({os.path.getsize(a.out) / 1024:.0f} kB)")
