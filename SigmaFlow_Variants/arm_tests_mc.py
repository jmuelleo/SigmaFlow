"""Dieselben Tests, aber mit 1000 gezogenen Teilmengen statt der Formel.

Die Frage ist, ob der Schaetzer der Komplexwerte den Test verschiebt. Der
Ablauf ist sonst identisch: Wert je Komplex, gepaarte Differenz, Bootstrap
ueber 8000 Neuziehungen der Komplexmenge, zweiseitiger p-Wert.

ZWEI RAUSCHQUELLEN STATT EINER
    Die Formel liefert den Komplexwert exakt. Mit 1000 Teilmengen kommt je
    Komplex ein Schaetzfehler hinzu, und der geht ungepaart in die Differenz
    ein: die beiden Arme haben verschiedene Posenmengen, ihre Teilmengen
    lassen sich also nicht koppeln. Erwartet wird deshalb, dass die p-Werte
    leicht groesser ausfallen -- der Test sieht Rauschen, das gar nicht von
    den Armen kommt.

    Wie stark, sagt der Vergleich unten. Die Monte-Carlo-Streuung eines
    Komplexwertes ist hoechstens 0.5/sqrt(1000), rund 1,6 Prozentpunkte; ueber
    307 Komplexe gemittelt bleibt davon ein Bruchteil.
"""
import numpy as np
import pandas as pd

import final_tables as ft
from arm_tests import MATCHED, KRIT, B, holm, test

MC = 1000
SEED = 20260830
_cache = {}


def mc_reihen(data, bench, arm, nfe, ks, rng):
    """{K: Serie ueber Komplexe}, alle K aus EINER Permutationsmatrix."""
    schluessel = (bench, arm, nfe, tuple(sorted(ks)))
    if schluessel in _cache:
        return _cache[schluessel]
    m = data[(bench, arm, nfe)]
    ziel = {t: {K: {} for K in ks} for t, _ in KRIT}
    for cid, g in m.groupby("complex"):
        h = g["heur"].to_numpy()
        ys = {t: g[t].to_numpy().astype(bool) for t, _ in KRIT}
        n = len(h)
        perm = np.argsort(rng.random((MC, n)), axis=1)
        zeilen = np.arange(MC)
        for K in ks:
            idx = perm[:, :min(K, n)]
            sieger = idx[zeilen, np.argmax(h[idx], axis=1)]
            for t, _ in KRIT:
                ziel[t][K][cid] = ys[t][sieger].mean()
    out = {(t, K): pd.Series(ziel[t][K]) for t, _ in KRIT for K in ks}
    _cache[schluessel] = out
    return out


def main():
    data = ft.build()

    for bench in ("PB308", "AX85"):
        mk = MATCHED[bench]
        rng = np.random.default_rng(SEED)
        paare = []
        for arm in ("SigmaFlow-Minimal", "SigmaFlow-Separate"):
            paare.append((f"{arm} 5/{mk} gegen SigmaDock 25/40  [gematcht]",
                          ("SigmaDock", 25, 40), (arm, 5, mk)))
        for arm in ("SigmaFlow-Minimal", "SigmaFlow-Separate"):
            paare.append((f"{arm} 25/40 gegen SigmaDock 25/40",
                          ("SigmaDock", 25, 40), (arm, 25, 40)))
        for arm in ("SigmaFlow-Minimal", "SigmaFlow-Separate"):
            paare.append((f"{arm} 5/200 gegen SigmaDock 25/40",
                          ("SigmaDock", 25, 40), (arm, 5, 200)))
        for nfe, K in ((25, 40), (5, mk), (5, 200)):
            paare.append((f"Separate gegen Minimal, {nfe}/{K}",
                          ("SigmaFlow-Minimal", nfe, K),
                          ("SigmaFlow-Separate", nfe, K)))

        # alle benoetigten K je Zelle einsammeln
        noetig = {}
        for _, a, b in paare:
            for arm, nfe, K in (a, b):
                noetig.setdefault((arm, nfe), set()).add(K)
        reihen = {}
        for (arm, nfe), ks in noetig.items():
            reihen[(arm, nfe)] = mc_reihen(data, bench, arm, nfe, sorted(ks), rng)

        for target, kname in KRIT:
            print(f"\n{'=' * 100}\n{bench}, {kname}   "
                  f"(Formel gegen {MC} Teilmengen)\n{'=' * 100}")
            print(f"{'Vergleich':<52}{'Diff':>7}{'Diff':>7}"
                  f"{'p Formel':>11}{'p MC':>11}  Holm MC")
            zeilen, ps = [], []
            for name, (a1, n1, k1), (a2, n2, k2) in paare:
                ea = ft.rule(data[(bench, a1, n1)], target, k1, "heuristic")
                eb = ft.rule(data[(bench, a2, n2)], target, k2, "heuristic")
                de, _, _, pe, _ = test(ea, eb, np.random.default_rng(11))
                ma = reihen[(a1, n1)][(target, k1)]
                mb = reihen[(a2, n2)][(target, k2)]
                dm, _, _, pm, _ = test(ma, mb, np.random.default_rng(11))
                zeilen.append((name, de, dm, pe, pm))
                ps.append(pm)
            for (name, de, dm, pe, pm), h in zip(zeilen, holm(ps)):
                print(f"{name:<52}{de:>+7.2f}{dm:>+7.2f}{pe:>11.4g}{pm:>11.4g}"
                      f"  {'ja' if h else 'nein'}")


if __name__ == "__main__":
    main()
