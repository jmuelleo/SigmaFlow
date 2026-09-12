"""Posen einlesen, paarweise RMSD rechnen, Ergebnis auf Platte legen.

WOZU EIN EIGENES MODUL
    Drei Skripte brauchen dieselbe teure Vorstufe: die Koordinaten aller
    Posen eines Komplexes und die paarweisen Abstaende daraus. Das Einlesen
    von 13 400 SDF und eine 200x200-Matrix je Komplex dauern Minuten; jede
    Auswertung darauf dauert Sekunden.

    Wer die Vorstufe in jedes Skript hineinkopiert, hat sie drei Mal --
    und aendert eines Tages zwei davon. Deshalb liegt sie hier, einmal, und
    die Skripte rufen `hole(...)`.

    Das ist das allgemeine Muster: die Grenze zwischen "teuer und selten
    geaendert" und "billig und oft geaendert" ist die richtige Stelle fuer
    einen Cache UND fuer eine Modulgrenze. Beides faellt hier zusammen.

DER DATEINAME DES CACHES
    `cache_unsicherheit_<satz>_<arm>_nfe<n>.npz` -- bewusst derselbe Name,
    den unsicherheit_k.py schon benutzt hat, damit die dort bereits
    gerechneten Matrizen der Flow-Arme wiederverwendet werden, statt noch
    einmal eine halbe Stunde zu kosten.

ZWEI ABSTANDSBEGRIFFE
    `d`      Paarweiser RMSD ueber die Atomreihenfolge, ohne Ueberlagerung.
             Schnell, weil eine einzige numpy-Operation.
    `d_sym`  Dasselbe, aber symmetriekorrigiert: RDKits `CalcRMS` probiert
             die Automorphismen des Molekuelgraphen durch, sodass ein um
             180 Grad gedrehter Phenylring nicht mehr als Unterschied zaehlt.
             Korrekt, aber O(n^2) RDKit-Aufrufe -- deshalb nur auf Anforderung
             und nur fuer kleine n sinnvoll.

    Ohne Symmetriekorrektur ist der RMSD eine OBERGRENZE. Fuer den Vergleich
    zwischen Armen ist das unschaedlich, weil es alle gleich trifft; fuer eine
    Absolutaussage ueber Modenzahlen ist es das nicht.
"""
import glob
import os
import re
import sys

import numpy as np

_HIER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HIER)
from zellen import SAETZE, lade_zelle, posenordner  # noqa: E402

from rdkit import Chem, RDLogger  # noqa: E402
RDLogger.DisableLog("rdApp.*")
from rdkit.Chem import rdMolAlign  # noqa: E402


def _lade(pfad, sanitize=False):
    m = Chem.MolFromMolFile(pfad, sanitize=sanitize, removeHs=False)
    if m is None or m.GetNumConformers() == 0:
        return None, None, None
    return (m.GetConformer().GetPositions(),
            tuple(a.GetSymbol() for a in m.GetAtoms()), m)


def _pfad(satz, arm, nfe, sym=False):
    n = f"cache_unsicherheit_{satz}_{arm}_nfe{nfe}"
    return os.path.join(_HIER, f"{n}{'_sym' if sym else ''}.npz")


def bauen(satz, arm, nfe, sym=False, leise=False):
    """Cache erzeugen, falls er fehlt; Pfad zurueckgeben."""
    p = _pfad(satz, arm, nfe, sym)
    if os.path.isfile(p):
        return p
    z = next((z for z in SAETZE[satz]["zellen"]
              if z["arm"] == arm and z["nfe"] == nfe), None)
    if z is None:
        sys.exit(f"ABBRUCH: keine Zelle {arm}/nfe{nfe} in Satz '{satz}'.")
    w = posenordner(satz, arm, nfe)
    if not os.path.isdir(w):
        sys.exit(f"ABBRUCH: Posenverzeichnis fehlt: {w}")
    tab = lade_zelle(z, leise=True)
    spalten = [c for c in ("acc", "beides", "valid") if c in tab.columns]

    ablage, codes = {}, sorted(tab["complex"].unique())
    for i, code in enumerate(codes, 1):
        g = tab[tab["complex"] == code]
        von = {c: dict(zip(g["seed"], g[c])) for c in spalten}
        heur_von = dict(zip(g["seed"], g["heur"]))
        dateien = sorted(
            glob.glob(os.path.join(w, "seed_*", f"{code}__*.sdf")),
            key=lambda f: int(re.search(r"seed_(\d+)", f).group(1)))
        x, mole, heur, seeds, ref = [], [], [], [], None
        werte = {c: [] for c in spalten}
        gesehen = set()
        for f in dateien:
            s = int(re.search(r"seed_(\d+)", f).group(1))
            if s not in heur_von:
                continue
            # EIN Pose je (Komplex, Seed). Bei 41 der 307 PB308-Komplexe liegt
            # in einem Seed-Verzeichnis eine zweite SDF-Datei, die in der
            # Auswertungstabelle keine Zeile hat. Ohne diese Sperre haette der
            # Cache dort eine Pose mehr als die Tabelle Zeilen, und jede
            # spaetere Verknuepfung mit den Scores waere um eins verschoben --
            # still, denn beide Listen sind ja "fast" gleich lang.
            if s in gesehen:
                continue
            gesehen.add(s)
            k, el, mol = _lade(f, sanitize=sym)
            if k is None:
                continue
            if ref is None:
                ref = el
            elif el != ref:
                sys.exit(f"ABBRUCH: {f} hat eine andere Atomfolge.")
            x.append(k)
            mole.append(mol)
            heur.append(float(heur_von[s]))
            seeds.append(s)
            for c in spalten:
                werte[c].append(bool(von[c].get(s, False)))
        if len(x) < 3:
            continue
        x = np.asarray(x, float)
        n = len(x)
        if sym:
            # O(n^2) RDKit-Aufrufe. CalcRMS rechnet "in place", also ohne die
            # Molekuele aufeinander zu legen -- genau richtig, weil die Posen
            # schon im Bezugssystem des Proteins liegen und eine Ueberlagerung
            # die raeumliche Information zerstoeren wuerde, um die es geht.
            d = np.zeros((n, n))
            for j in range(n):
                for k2 in range(j + 1, n):
                    try:
                        v = rdMolAlign.CalcRMS(mole[j], mole[k2])
                    except Exception:
                        v = float(np.sqrt(((x[j] - x[k2]) ** 2).sum(-1).mean()))
                    d[j, k2] = d[k2, j] = v
        else:
            d = np.sqrt(((x[:, None, :, :] - x[None, :, :, :]) ** 2)
                        .sum(-1).mean(-1))
            np.fill_diagonal(d, 0.0)
        ablage[f"{code}__d"] = d.astype(np.float32)
        # Die Koordinaten selbst: 40 Posen x ~35 Atome x 3 in float32 sind
        # 17 kB je Komplex. Aus der Abstandsmatrix allein liesse sich die
        # Varianz zwar rekonstruieren, aber nicht ZERLEGEN -- fuer die
        # Trennung von Lage, Orientierung und innerer Konformation braucht
        # es die Punkte, nicht nur ihre Abstaende.
        ablage[f"{code}__xyz"] = x.astype(np.float32)
        ablage[f"{code}__heur"] = np.asarray(heur, float)
        # Die Seednummern mitspeichern: nur damit laesst sich der
        # Cache spaeter exakt gegen die Score-Tabellen verknuepfen.
        ablage[f"{code}__seed"] = np.asarray(seeds, int)
        for c in spalten:
            ablage[f"{code}__{c}"] = np.asarray(werte[c], bool)
        if not leise and (i % 25 == 0 or i == len(codes)):
            print(f"    {arm} nfe{nfe}{' sym' if sym else ''}: "
                  f"{i}/{len(codes)}", flush=True)
    np.savez_compressed(p, **ablage)
    return p


def hole(satz, arm, nfe, sym=False, leise=False):
    """{code: {'d':..., 'heur':..., 'acc':..., 'beides':...}} zurueckgeben."""
    z = np.load(bauen(satz, arm, nfe, sym, leise))
    aus = {}
    for k in z.files:
        code, feld = k.split("__")
        aus.setdefault(code, {})[feld] = z[k]
    return aus
