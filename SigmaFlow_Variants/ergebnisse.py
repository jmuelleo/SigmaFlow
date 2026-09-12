"""Lesezugriff auf die EINGEFRORENEN 72-h-Zahlen aus ergebnisse_72h.py.

Jede Zahl der Arbeit zu den 72-h-Endpunkten kommt aus dieser Quelle:
B = 10.000 Ziehungen je Komplex, Seed 20260910, einmal gerechnet. Wer hier
nichts findet, soll ABBRECHEN statt still selbst zu rechnen -- genau daran
sind vorher drei Schaetzer nebeneinander entstanden.
"""
import os

import pandas as pd

_HIER = os.path.dirname(os.path.abspath(__file__))
_AGG = _KOM = None


def _laden():
    global _AGG, _KOM
    if _AGG is None:
        _AGG = pd.read_csv(os.path.join(_HIER, "ergebnisse_72h_aggregat.csv"))
        _AGG = _AGG.set_index(["satz", "arm", "nfe", "K", "ziel", "regel"])
    if _KOM is None:
        _KOM = pd.read_csv(os.path.join(_HIER, "ergebnisse_72h_je_komplex.csv"))
        _KOM = _KOM.set_index(["satz", "arm", "nfe", "K", "ziel", "regel"])
    return _AGG, _KOM


ALIAS = {"heuristic": "mixed", "vinardo": "gnina", "draw": "zug",
         "oracle": "orakel", "both": "both", "acc": "acc", "valid": "valid"}


def wert(satz, arm, nfe, K, ziel, regel="mixed"):
    """Top-1 in Prozent, gemittelt ueber die Komplexe."""
    agg, _ = _laden()
    schl = (satz, arm, nfe, K, ALIAS.get(ziel, ziel), ALIAS.get(regel, regel))
    if schl not in agg.index:
        raise KeyError(f"Nicht in ergebnisse_72h_aggregat.csv: {schl}. "
                       f"K in ergebnisse_72h.py ergaenzen und neu rechnen -- "
                       f"NICHT hier nachrechnen.")
    return float(agg.loc[schl, "prozent"])


def je_komplex(satz, arm, nfe, K, ziel, regel="mixed"):
    """Wert je Komplex in [0, 1] als Series, fuer gepaarte Tests."""
    _, kom = _laden()
    schl = (satz, arm, nfe, K, ALIAS.get(ziel, ziel), ALIAS.get(regel, regel))
    if schl not in kom.index:
        raise KeyError(f"Nicht in ergebnisse_72h_je_komplex.csv: {schl}")
    t = kom.loc[schl]
    return t.set_index("complex")["wert"]
