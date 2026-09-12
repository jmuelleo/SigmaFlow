"""Gepaarter Vergleich der Arme an epochennahen Snapshots.

WARUM DAS GEPAART GEHT
    Alle Arme werden auf DERSELBEN Komplexmenge ausgewertet (307 Komplexe,
    dieselben Referenzliganden, dieselbe PoseBusters-Konfiguration). Die
    Differenz laesst sich deshalb je Komplex bilden, und der Bootstrap laeuft
    ueber Komplexe -- dasselbe Verfahren wie tab:paired in Kapitel 6.

WAS DABEI NICHT GEPAART IST
    Die Posen. Jeder Arm zieht seine zehn Seeds unabhaengig; nur die Komplexe
    sind gemeinsam. Der Bootstrap zieht deshalb Komplexe und wertet beide Arme
    auf denselben gezogenen Komplexen aus.

PAARUNG NACH EPOCHE, MIT AUSGEWIESENER TOLERANZ
    Die Arme treffen sich nicht exakt: Minimal 20/41/63/78/99/118,
    SigmaDock 20/40/60/80/99/118/137/157, Separate 17/33/140. Gepaart wird der
    jeweils naechste Snapshot innerhalb von TOLERANZ Epochen, und die
    tatsaechliche Epochendifferenz steht in jeder Ausgabezeile. Wo sie zugunsten
    eines Arms ausfaellt, ist das im Ergebnis zu nennen und nicht wegzurunden.
"""
import itertools
import pathlib

import numpy as np
import pandas as pd

from arme import arme_vorhanden, zellen
from kurve_daten import SEED, zelle_laden
from kurve_paare import gepaart

HIER = pathlib.Path(__file__).resolve().parent
TOLERANZ = 8          # Epochen
GEZEIGT = ("u2", "pb_lig", "pb_prot", "u2_prot")
NAME = {"u2": "RMSD<2A", "pb_lig": "PB Lig", "pb_prot": "PB+Protein",
        "u2_prot": "<2A & PB"}


def main():
    alle = zellen()
    cache = {}

    def lade(z):
        if z["zelle"] + z["lauf"] not in cache:
            cache[z["zelle"] + z["lauf"]] = zelle_laden(z)
        return cache[z["zelle"] + z["lauf"]]

    zeilen = []
    for a, b in itertools.combinations(arme_vorhanden(), 2):
        for nfe in (25, 5):
            la = sorted([z for z in alle if z["arm"] == a and z["nfe"] == nfe],
                        key=lambda z: z["epoch"])
            lb = sorted([z for z in alle if z["arm"] == b and z["nfe"] == nfe],
                        key=lambda z: z["epoch"])
            if not la or not lb:
                continue
            paare = []
            for za in la:
                nah = min(lb, key=lambda z: abs(z["epoch"] - za["epoch"]))
                if abs(nah["epoch"] - za["epoch"]) <= TOLERANZ:
                    paare.append((za, nah))
            if not paare:
                continue
            print(f"\n### {b} minus {a}, {nfe} Schritte  (Prozentpunkte)")
            print(f"{'Epochen':>14}{'dEp':>5} {'Metrik':>11} {'Delta':>8} "
                  f"{'95%-KI':>18} {'p':>10}")
            for za, zb in paare:
                res, n = gepaart(lade(za), lade(zb), GEZEIGT)
                dep = zb["epoch"] - za["epoch"]
                lbl = f"{za['epoch']}/{zb['epoch']}"
                for k, (d, lo, hi, p) in res.items():
                    stern = "*" if (lo > 0 or hi < 0) else " "
                    ptxt = f"<{1/8000:.1e}" if p <= 1 / 8000 else f"{p:.4f}"
                    print(f"{lbl:>14}{dep:+5d} {NAME[k]:>11} {d:+7.2f}{stern} "
                          f"[{lo:+6.2f},{hi:+6.2f}] {ptxt:>10}")
                    zeilen.append({"arm_a": a, "arm_b": b, "nfe": nfe,
                                   "epoch_a": za["epoch"], "epoch_b": zb["epoch"],
                                   "delta_epoch": dep, "metrik": k,
                                   "delta_pp": round(d, 3), "lo": round(lo, 3),
                                   "hi": round(hi, 3), "p": p, "n_complexes": n})
    pd.DataFrame(zeilen).to_csv(HIER / "kurve_vergleich.csv", index=False)
    print(f"\ngeschrieben: kurve_vergleich.csv")
    print("Positiv = arm_b besser. delta_epoch ausweisen: eine positive "
          "Differenz gibt arm_b zusaetzliches Training.")


if __name__ == "__main__":
    main()
