"""Gepaarte Differenzen zwischen aufeinanderfolgenden Snapshots, je Arm.

WARUM GEPAART UND NICHT UEBER GETRENNTE INTERVALLE
    Zwei Snapshots werden auf DENSELBEN Komplexen ausgewertet. Die Frage "ist
    der Zuwachs echt" ist deshalb eine Frage ueber die Differenz je Komplex,
    nicht ueber zwei unabhaengige Raten. Getrennte Intervalle koennen
    ueberlappen, waehrend die Differenz klar von null verschieden ist -- bei
    gemeinsamer Komplexmenge ist der gepaarte Test deutlich schaerfer, und er
    ist das Verfahren, das Kapitel 6 der Arbeit ohnehin benutzt (tab:paired).

    Am 2026-08-27 war das entscheidend: die Intervalle der letzten beiden
    Minimal-Snapshots ueberlappen, der gepaarte Test weist den Zuwachs
    trotzdem mit p = 0,008 nach.

VERFAHREN
    B Ziehungen aus den gemeinsamen Komplexen. EINE Ziehung, beide Snapshots
    darauf ausgewertet, dann die Differenz -- sonst waere die Paarung verloren.
    p zweiseitig aus dem Vorzeichenanteil; Aufloesungsgrenze 1/B.
"""
import pathlib

import numpy as np
import pandas as pd

from arme import arme_vorhanden, slug, zellen
from kurve_daten import SEED, zelle_laden

HIER = pathlib.Path(__file__).resolve().parent
B = 8000
GEZEIGT = ("u2", "pb_lig", "pb_prot", "u2_prot")


def gepaart(a: pd.DataFrame, b: pd.DataFrame, spalten, B=B, seed=SEED):
    """b - a in Prozentpunkten, Bootstrap ueber die gemeinsamen Komplexe."""
    gemein = sorted(set(a["complex"]) & set(b["complex"]))
    n = len(gemein)
    rng = np.random.default_rng(seed)
    zieh = rng.integers(0, n, size=(B, n))
    out = {}
    for spalte in spalten:
        paare = []
        for d in (a, b):
            g = d[d["complex"].isin(gemein)].groupby("complex")[spalte]
            g = g.agg(["sum", "count"]).reindex(gemein)
            paare.append((g["sum"].to_numpy(float), g["count"].to_numpy(float)))
        (sa, na), (sb, nb) = paare
        d_boot = (100 * sb[zieh].sum(1) / nb[zieh].sum(1)
                  - 100 * sa[zieh].sum(1) / na[zieh].sum(1))
        punkt = 100 * sb.sum() / nb.sum() - 100 * sa.sum() / na.sum()
        p = 2 * min((d_boot <= 0).mean(), (d_boot >= 0).mean())
        out[spalte] = (punkt, float(np.percentile(d_boot, 2.5)),
                       float(np.percentile(d_boot, 97.5)), max(p, 1 / B))
    return out, n


def main():
    alle = zellen()
    zeilen = []
    for arm in arme_vorhanden():
        for nfe in (25, 5):
            folge = sorted([z for z in alle if z["arm"] == arm and z["nfe"] == nfe],
                           key=lambda z: z["epoch"])
            if len(folge) < 2:
                print(f"\n### {arm}, {nfe} Schritte: nur {len(folge)} Zelle(n), "
                      f"kein Uebergang")
                continue
            geladen = [zelle_laden(z) for z in folge]
            print(f"\n### {arm}, {nfe} Integrationsschritte")
            print(f"{'Uebergang':>16} {'Metrik':>10} {'Delta':>8} "
                  f"{'95%-KI':>18} {'p':>10}")
            for i in range(len(folge) - 1):
                res, n = gepaart(geladen[i], geladen[i + 1], GEZEIGT)
                lbl = f"Ep {folge[i]['epoch']} -> {folge[i+1]['epoch']}"
                for k, (d, lo, hi, p) in res.items():
                    stern = "*" if (lo > 0 or hi < 0) else " "
                    ptxt = f"<{1 / B:.1e}" if p <= 1 / B else f"{p:.4f}"
                    print(f"{lbl:>16} {k:>10} {d:+7.2f}{stern} "
                          f"[{lo:+6.2f},{hi:+6.2f}] {ptxt:>10}")
                    zeilen.append({"arm": arm, "nfe": nfe,
                                   "von_epoch": folge[i]["epoch"],
                                   "bis_epoch": folge[i + 1]["epoch"],
                                   "metrik": k, "delta_pp": round(d, 3),
                                   "lo": round(lo, 3), "hi": round(hi, 3),
                                   "p": p, "n_complexes": n})
    pd.DataFrame(zeilen).to_csv(HIER / "kurve_paare.csv", index=False)
    print(f"\ngeschrieben: kurve_paare.csv   (* = 95%-KI schliesst 0 aus)")


if __name__ == "__main__":
    main()
