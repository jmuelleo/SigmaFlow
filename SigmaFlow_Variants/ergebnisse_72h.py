"""EINMAL rechnen, dann ueberall dieselben Zahlen: die 72-h-Endpunktzellen.

WARUM DIESE DATEI
    Es gab drei Wege, dieselbe Groesse zu schaetzen -- 1000 gezogene
    Teilmengen (zellen.top1), eine geschlossene Form (final_tables.rule) und
    eine eigene MC-Fassung in plot_ranking.curves, jede mit eigenem Seed. Die
    Zahlen wichen dadurch um bis zu zwei Zehntel voneinander ab, je nachdem
    welches Skript sie erzeugt hatte. Ab jetzt gibt es EINE Quelle:
    diese Datei rechnet mit B = 10.000 Ziehungen und schreibt die Werte JE
    KOMPLEX weg. Tabellen, Abbildungen und Tests lesen nur noch das Ergebnis.

METHODIK
    Je Komplex eine Zufallspermutation der Posen, B-mal. Die ersten k Spalten
    einer Permutation sind eine gleichverteilte k-Teilmenge ohne Zuruecklegen.
    Damit bedient EINE Matrix alle K, alle Ziele und alle Regeln:

      pos[i]   Rang der Pose i in der Regelordnung, 0 = beste
      r        = pos[perm], also die Raenge in Ziehreihenfolge
      min-akk  np.minimum.accumulate(r, axis=1) liefert fuer JEDES k in einem
               Durchgang den Rang der Gewinnerin der ersten k Ziehungen

    Fuer das Orakel entsprechend das laufende Maximum ueber die Etiketten.
    Bei k >= n gibt es nur eine Teilmenge, der Wert ist dann exakt.

    Erst im Komplex mitteln, dann ueber die Komplexe -- sonst bekaemen
    Komplexe mit mehr Posen mehr Gewicht.
"""
import numpy as np
import pandas as pd

import final_tables as ft

B, SEED = 10000, 20260910
# Alle K, die irgendein Verbraucher braucht: Tabellen (1,5,10,20,40,100,
# 140,200), die walltime-gematchten Zellen (101, 140), und die dichte
# Kurve in plot_ranking.py. Lieber einmal zu viel rechnen als spaeter
# einen Punkt aus einer zweiten Quelle nachziehen.
# ZWEI RASTER.
#   KS_DICHT  jedes K bis zur Zahl der Seeds. Daraus entsteht die
#             aggregierte Kurve; sie kostet fast nichts, weil eine
#             Permutationsmatrix je Komplex alle K bedient.
#   KS_TEST   die Stuetzstellen, fuer die auch die Werte JE KOMPLEX
#             weggeschrieben werden. Nur die braucht der gepaarte Test, und
#             nur sie bestimmen die Dateigroesse -- alle 200 K je Komplex
#             waeren rund 3,3 Millionen Zeilen.
KS_DICHT = list(range(1, 201))
KS_TEST = sorted(set(range(1, 41)) | {45, 50, 60, 70, 80, 90, 100, 101, 120,
                                      140, 160, 170, 180, 200})
ZIELE = ["acc", "valid", "both"]
# Regel -> (Spalte, aufsteigend sortieren?)  None = Sonderfall
REGELN = {"mixed": ("heur", False), "gnina": ("affinity", True),
          "filter": (None, None), "zug": (None, None), "orakel": (None, None)}
ZELLEN = [("PB308", "SigmaDock", 25), ("PB308", "SigmaFlow-Minimal", 25),
          ("PB308", "SigmaFlow-Separate", 25), ("PB308", "SigmaDock", 5),
          ("PB308", "SigmaFlow-Minimal", 5), ("PB308", "SigmaFlow-Separate", 5),
          ("AX85", "SigmaDock", 25), ("AX85", "SigmaFlow-Minimal", 25),
          ("AX85", "SigmaFlow-Separate", 25), ("AX85", "SigmaDock", 5),
          ("AX85", "SigmaFlow-Minimal", 5), ("AX85", "SigmaFlow-Separate", 5)]


def ordnung(g, regel):
    """Positionsindex je Pose in der Regelordnung, 0 = beste."""
    if regel == "filter":
        h = g.sort_values(["valid", "affinity"], ascending=[False, True])
    else:
        sp, auf = REGELN[regel]
        h = g.sort_values(sp, ascending=auf)
    pos = np.empty(len(g), int)
    pos[h.reset_index().index.to_numpy()] = np.arange(len(g))
    # h.index sind die Originalzeilen; wir brauchen Position je urspruenglicher Zeile
    rang = np.empty(len(g), int)
    rang[np.argsort(np.argsort(-np.arange(len(g))))] = 0  # Platzhalter, s.u.
    lage = {ix: r for r, ix in enumerate(h.index)}
    return np.array([lage[ix] for ix in g.index]), h


def merke(zeilen, summe, satz, arm, nfe, K, ziel, regel, code, w):
    """Aggregat immer mitfuehren, Werte je Komplex nur an den Teststellen."""
    schl = (satz, arm, nfe, K, ziel, regel)
    s0, n0 = summe.get(schl, (0.0, 0))
    summe[schl] = (s0 + w, n0 + 1)
    if K in KS_TEST:
        zeilen.append((satz, arm, nfe, K, ziel, regel, code, w))


def main():
    rng = np.random.default_rng(SEED)
    zeilen, summe = [], {}
    d = ft.build()
    for satz, arm, nfe in ZELLEN:
        if (satz, arm, nfe) not in d:
            continue
        m = d[(satz, arm, nfe)]
        n_seeds = m["seed"].nunique()
        ks = [K for K in KS_DICHT if K <= n_seeds]
        print(f"{satz:6s} {arm:20s} nfe {nfe:2d}  n_seeds={n_seeds:3d}  "
              f"K bis {max(ks)}", flush=True)
        for code, g in m.groupby("complex"):
            n = len(g)
            y = {t: g[t].to_numpy().astype(float) for t in ZIELE}
            perm = np.argsort(rng.random((B, n)), axis=1)
            for regel in REGELN:
                if regel == "zug":
                    for t in ZIELE:
                        w = float(y[t].mean())
                        for K in ks:
                            merke(zeilen, summe, satz, arm, nfe, K, t,
                                  regel, code, w)
                    continue
                if regel == "orakel":
                    for t in ZIELE:
                        lauf = np.maximum.accumulate(y[t][perm], axis=1)
                        for K in ks:
                            k = min(K, n)
                            w = 1.0 if k >= n else float(lauf[:, k - 1].mean())
                            if k >= n:
                                w = float(y[t].any())
                            merke(zeilen, summe, satz, arm, nfe, K, t,
                                  regel, code, w)
                    continue
                pos, h = ordnung(g, regel)
                ysort = {t: h[t].to_numpy().astype(float) for t in ZIELE}
                bester = np.minimum.accumulate(pos[perm], axis=1)
                for t in ZIELE:
                    for K in ks:
                        k = min(K, n)
                        w = (float(ysort[t][0]) if k >= n
                             else float(ysort[t][bester[:, k - 1]].mean()))
                        merke(zeilen, summe, satz, arm, nfe, K, t,
                              regel, code, w)

    df = pd.DataFrame(zeilen, columns=["satz", "arm", "nfe", "K", "ziel",
                                       "regel", "complex", "wert"])
    df.to_csv("ergebnisse_72h_je_komplex.csv", index=False)
    agg = pd.DataFrame(
        [(*k, v[0] / v[1], v[1]) for k, v in summe.items()],
        columns=["satz", "arm", "nfe", "K", "ziel", "regel", "mean", "count"])
    agg["prozent"] = 100 * agg["mean"]
    agg.to_csv("ergebnisse_72h_aggregat.csv", index=False)
    print(f"\n{len(df)} Zeilen je Komplex, {len(agg)} Zellen.")
    print(f"B = {B}, Seed = {SEED}")


if __name__ == "__main__":
    main()
