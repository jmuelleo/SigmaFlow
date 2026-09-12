"""RMSD jeder Pose zur Kristallpose -- fuer strengere Schwellen als 2 Angstroem.

WOZU
    Die Auswertungstabellen fuehren nur den Boolean `rmsd_<=_2A`. Fuer die
    Frage "wie streng muesste man sein, damit falsch herum gedrehte Fragmente
    nicht mehr durchgehen" braucht es den Zahlenwert.

SYMMETRIEKORRIGIERT
    `rdMolAlign.CalcRMS` probiert die Automorphismen des Molekuelgraphen durch
    und rechnet OHNE Ueberlagerung -- die Posen liegen bereits im Bezugssystem
    des Proteins, eine Ueberlagerung wuerde genau die Information zerstoeren,
    um die es geht. Ohne Symmetriekorrektur zaehlte ein um 180 Grad gedrehter
    Phenylring als Fehler, und die strengen Schwellen saehen schlechter aus,
    als sie sind.

    Das ist auch die Definition, die PoseBusters benutzt. Als Gegenprobe
    vergleicht das Skript den eigenen Wert gegen den Boolean der Tabelle; bei
    mehr als ein paar Promille Abweichung stimmt etwas nicht.

WARUM EIN EIGENER CACHE
    Die RMSD haengen nicht von der Fragestellung ab, das Einlesen der SDF
    dauert aber Minuten. Ergebnis als CSV daneben, damit jede spaetere Frage
    nach Schwellen Sekunden kostet.

Aufruf:
    python SigmaFlow_Variants/rmsd_streng.py --satz pb308 --nfe 25
    python SigmaFlow_Variants/rmsd_streng.py --satz astex --nfe 5
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

_HIER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HIER)
from zellen import SAETZE, lade_zelle, posenordner  # noqa: E402

from rdkit import Chem, RDLogger  # noqa: E402
RDLogger.DisableLog("rdApp.*")
from rdkit.Chem import rdMolAlign  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--satz", choices=sorted(SAETZE), default="pb308")
p.add_argument("--nfe", type=int, default=25)
p.add_argument("--arme", default="SigmaDock,Minimal,Separate")
p.add_argument("--out", default=None)
a = p.parse_args()
ARME = [s.strip() for s in a.arme.split(",")]
AUS = a.out or os.path.join(_HIER, f"rmsd_{a.satz}_nfe{a.nfe}.csv")
SATZ = SAETZE[a.satz]

zeilen = []
for arm in ARME:
    z = next(z for z in SATZ["zellen"] if z["arm"] == arm and z["nfe"] == a.nfe)
    w = posenordner(a.satz, arm, a.nfe)
    tab = lade_zelle(z, leise=True)
    codes = sorted(tab["complex"].unique())
    print(f"{arm}, {a.nfe} Schritte: {len(codes)} Komplexe", flush=True)
    for i, code in enumerate(codes, 1):
        kris_p = os.path.join(SATZ["referenz"], code, f"{code}_ligand.sdf")
        kris = Chem.MolFromMolFile(kris_p, sanitize=True, removeHs=True)
        if kris is None:
            continue
        g = tab[tab["complex"] == code]
        flag = dict(zip(g["seed"], g["acc"]))
        for f in glob.glob(os.path.join(w, "seed_*", f"{code}__*.sdf")):
            s = int(re.search(r"seed_(\d+)", f).group(1))
            if s not in flag:
                continue
            m = Chem.MolFromMolFile(f, sanitize=True, removeHs=True)
            if m is None:
                continue
            try:
                r = float(rdMolAlign.CalcRMS(m, kris))
            except Exception:
                continue
            zeilen.append({"arm": arm, "complex": code, "seed": s,
                           "rmsd": r, "flag2a": bool(flag[s])})
        if i % 50 == 0:
            print(f"  {i}/{len(codes)}", flush=True)

t = pd.DataFrame(zeilen)
t.to_csv(AUS, index=False)

# --- Gegenprobe gegen den Boolean der Tabelle ---------------------------
eigen = t["rmsd"] < 2.0
stimmt = (eigen == t["flag2a"]).mean()
print(f"\n{len(t)} Posen nach {AUS}")
print(f"Gegenprobe: eigener RMSD < 2 stimmt mit dem Tabellen-Boolean in "
      f"{100 * stimmt:.2f} % der Faelle ueberein.")
if stimmt < 0.98:
    print("  ACHTUNG: das ist zu wenig. Die RMSD-Definition weicht ab --")
    print("  pruefen, bevor irgendeine Zahl daraus benutzt wird.")

print(f"\n=== Verteilung der RMSD je Arm ===")
print(f"  {'Arm':<11}{'Median':>9}{'< 0,5':>9}{'< 1,0':>9}{'< 1,5':>9}{'< 2,0':>9}")
for arm in ARME:
    d = t[t.arm == arm]["rmsd"]
    print(f"  {arm:<11}{d.median():9.2f}" + "".join(
        f"{100 * (d < s).mean():8.1f}%" for s in (0.5, 1.0, 1.5, 2.0)))
