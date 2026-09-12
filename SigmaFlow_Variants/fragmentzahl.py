"""Fragmentzahl je PB308-Komplex, mit derselben Fragmentierung wie im Training.

WARUM NICHT EINFACH "ANZAHL DREHBARER BINDUNGEN + 1"
    SigmaDock schneidet nicht an jeder drehbaren Bindung. `fragment_molecule`
    waehlt eine MINIMALE Schnittmenge und verschmilzt danach noch Fragmente.
    Die Zahl, die das Modell sieht, ist `len(get_fragments_as_mols(frag_mol))`
    aus `data.py:345` -- und nur die zaehlt hier.

DIE FRAGMENTZAHL IST NICHT VOLLSTAENDIG DETERMINISTISCH
    `fragmentation_strategy` steht auf "random". Gemessen an 20 Komplexen:
    19 liefern ueber 8 Ziehungen immer dieselbe Zahl, einer schwankt
    (9/10/11). Die minimalen Schnittmengen haben zwar alle gleich viele
    Schnitte, aber das anschliessende Verschmelzen haengt von der gewaehlten
    Menge ab.

    Konsequenz: je Komplex werden ZIEHUNGEN Fragmentierungen gerechnet und der
    MEDIAN genommen. Die Spalte `stabil` sagt, ob alle Ziehungen
    uebereinstimmten. Wer nach Fragmentzahl stratifiziert, muss die instabilen
    Komplexe kennen -- ihre Zuordnung ist auf +-1 genau, nicht exakt.

    Fuer die einzelne gesampelte Pose ist die tatsaechlich benutzte Zahl nicht
    mehr rekonstruierbar (sie wurde nirgends protokolliert). Die Stratifizierung
    ist daher fuer diese Komplexe eine Naeherung, keine Ablesung.

Aufruf:
    python SigmaFlow_Variants/fragmentzahl.py
    python SigmaFlow_Variants/fragmentzahl.py --ziehungen 12 --out frag.csv
"""
import argparse
import glob
import os
import random
import sys
from collections import Counter

import numpy as np
import pandas as pd

_HIER = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HIER)
sys.path.insert(0, os.path.join(_REPO, "SigmaFlow_Minimal", "src"))

from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

from sigmadock.chem.fragmentation import (  # noqa: E402
    fragment_molecule,
    get_fragments_as_mols,
)

p = argparse.ArgumentParser()
p.add_argument("--wurzel",
               default=os.path.join(_HIER, "learning_curve_min", "true308"),
               help="Verzeichnis mit einem Ordner je Komplex")
p.add_argument("--ziehungen", type=int, default=8)
p.add_argument("--out", default=os.path.join(_HIER, "fragmentzahl_pb308.csv"))
a = p.parse_args()

ordner = sorted(d for d in glob.glob(os.path.join(a.wurzel, "*")) if os.path.isdir(d))
if not ordner:
    sys.exit(f"ABBRUCH: keine Komplexordner unter {a.wurzel}")
print(f"{len(ordner)} Komplexe, {a.ziehungen} Ziehungen je Komplex", flush=True)

zeilen = []
for n, d in enumerate(ordner):
    name = os.path.basename(d)
    treffer = sorted(glob.glob(os.path.join(d, "*_ligands.sdf")))
    if not treffer:
        zeilen.append({"complex": name, "n_frag": np.nan, "stabil": False,
                       "n_atome": np.nan, "bemerkung": "keine _ligands.sdf"})
        continue
    mol = Chem.MolFromMolFile(treffer[0], sanitize=True, removeHs=True)
    if mol is None:
        zeilen.append({"complex": name, "n_frag": np.nan, "stabil": False,
                       "n_atome": np.nan, "bemerkung": "RDKit-Ladefehler"})
        continue
    zahlen = []
    fehler = ""
    for z in range(a.ziehungen):
        # Beide Generatoren setzen: fragment_molecule zieht ueber `random`,
        # nachgelagerter Code teils ueber numpy.
        random.seed(z)
        np.random.seed(z)
        try:
            fm = fragment_molecule(mol, selection="random", ignore_conjugated=False)
            zahlen.append(len(get_fragments_as_mols(fm, asMols=False)))
        except Exception as e:  # noqa: BLE001
            fehler = f"{type(e).__name__}"
    if not zahlen:
        zeilen.append({"complex": name, "n_frag": np.nan, "stabil": False,
                       "n_atome": mol.GetNumAtoms(),
                       "bemerkung": f"Fragmentierung scheiterte: {fehler}"})
        continue
    c = Counter(zahlen)
    zeilen.append({
        "complex": name,
        "n_frag": int(np.median(zahlen)),
        "stabil": len(c) == 1,
        "n_atome": mol.GetNumAtoms(),
        "bemerkung": "" if len(c) == 1 else f"schwankt {dict(sorted(c.items()))}",
    })
    if (n + 1) % 50 == 0:
        print(f"  {n + 1}/{len(ordner)} ...", flush=True)

t = pd.DataFrame(zeilen)
t.to_csv(a.out, index=False)
print(f"\n{len(t)} Zeilen nach {a.out}")

fehlend = t["n_frag"].isna().sum()
instabil = (~t["stabil"] & t["n_frag"].notna()).sum()
print(f"ohne Fragmentzahl : {fehlend}")
print(f"instabil          : {instabil}  ({100 * instabil / max(len(t), 1):.1f} %)")
if fehlend:
    print("\nOhne Zahl:")
    print(t[t["n_frag"].isna()][["complex", "bemerkung"]].to_string(index=False))

print("\nVerteilung der Fragmentzahl:")
v = t["n_frag"].dropna().astype(int).value_counts().sort_index()
for k, n in v.items():
    print(f"  {k:2d} Fragmente : {n:4d} Komplexe  {'#' * max(1, n // 3)}")
print(f"\n  Median {t['n_frag'].median():.0f}, "
      f"Spanne {t['n_frag'].min():.0f} bis {t['n_frag'].max():.0f}")
