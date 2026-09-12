"""Top-1 nach Ranking gegen die Ziehungszahl -- 1000 Zufallsteilmengen.

DIE METHODIK, EINHEITLICH
    Je Komplex werden 1000 Teilmengen der Groesse K aus den vorhandenen Posen
    gezogen. In jeder waehlt die Regel eine Pose; der Mittelwert ueber die
    1000 Ziehungen ist der Wert des Komplexes, und erst darueber wird ueber
    die Komplexe gemittelt.

    Umgesetzt in `zellen.top1`. Es gibt in diesen Skripten keine zweite
    Rechenweise mehr -- weder eine geschlossene Form noch "die ersten K
    Seeds". Bei K gleich der Zahl vorhandener Posen gibt es nur eine
    Teilmenge; das erkennt die Funktion und rechnet direkt.

WARUM ALLES DURCH zellen.py LAEUFT
    `final_tables.py` und `zellen.py` behandeln die Sampler-Macke
    unterschiedlich: je Seed liegen 308 Redock-Zeilen fuer 307 Komplexe vor,
    einer ist doppelt. `zellen.py` dedupliziert und behaelt den ersten
    Eintrag, `final_tables.py` verliert beim Merge beide. Das sind 45 Posen
    bei 41 Komplexen und rund ein Prozentpunkt Unterschied im Ergebnis.

    Beides ist vertretbar, aber in einer Arbeit darf nur eines vorkommen.
    Hier ist es `zellen.py`, damit die Ziehungstabellen, die Fragmenttabellen,
    die Modenanalyse und die Praezisionszahlen aus derselben Quelle stammen.

WARUM K NICHT UEBER DIE VORHANDENEN SEEDS HINAUSGEHT
    Es wird aus den GEZOGENEN Posen gezogen. Fuer K groesser als deren Zahl
    gibt es keine Teilmengen; die Zeile endet dort, wo die Zelle endet.

Aufruf:
    python SigmaFlow_Variants/tabelle_seeds.py
    python SigmaFlow_Variants/tabelle_seeds.py --regel filter
    python SigmaFlow_Variants/tabelle_seeds.py --regel mixed --latex
"""
import argparse
import os
import sys

import pandas as pd

_HIER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HIER)
from zellen import SAETZE, alle_zellen, top1  # noqa: E402

KS = [1, 2, 3, 5, 10, 20, 40, 60, 100, 140, 200]
ZELLEN = [("SigmaDock", 25), ("Minimal", 25), ("Separate", 25),
          ("SigmaDock", 5), ("Minimal", 5), ("Separate", 5)]
ZIELE = [("beides", "RMSD<2 and PB-valid"), ("acc", "RMSD<2")]
BENCH = [("pb308", "PoseBusters, 308 complexes"),
         ("astex", "Astex, 85 complexes")]
LANG = {"Minimal": "SigmaFlow-Minimal", "Separate": "SigmaFlow-Separate",
        "SigmaDock": "SigmaDock"}

p = argparse.ArgumentParser()
p.add_argument("--regel", default="mixed",
               choices=["mixed", "gnina", "filter", "orakel", "zug"],
               help="mixed = Score des Papers, filter = harter PB-Filter dann gnina")
p.add_argument("--ziehungen", type=int, default=1000)
p.add_argument("--latex", action="store_true")
p.add_argument("--out", default=None)
a = p.parse_args()
AUS = a.out or os.path.join(_HIER, f"tabelle_seeds_{a.regel}.csv")

zeilen = []
for satz, btitel in BENCH:
    print(f"{btitel} ...")
    d = alle_zellen(leise=True, satz=satz)
    for arm, nfe in ZELLEN:
        m = d[(arm, nfe)]
        n_max = int(m["seed"].nunique())
        for K in KS:
            if K > n_max:
                continue
            zeilen.append({
                "bench": satz, "arm": arm, "nfe": nfe, "K": K,
                **{ziel: round(top1(m, ziel, K, a.regel, a.ziehungen), 2)
                   for ziel, _ in ZIELE}})
        print(f"  {LANG[arm]:20s} {nfe:2d} Schritte, {n_max:3d} Seeds")

t = pd.DataFrame(zeilen)
t.to_csv(AUS, index=False)
print(f"\n{len(t)} Zeilen nach {AUS}")
print(f"Regel: {a.regel}   Ziehungen je Komplex und K: {a.ziehungen}\n")

for satz, btitel in BENCH:
    for ziel, ztitel in ZIELE:
        d = t[t["bench"] == satz].copy()
        d["zelle"] = d["arm"].map(LANG) + ", " + d["nfe"].astype(str) + " steps"
        piv = d.pivot_table(index="zelle", columns="K", values=ziel)
        piv = piv.reindex([f"{LANG[arm]}, {nfe} steps" for arm, nfe in ZELLEN])
        print(f"=== {btitel} -- {ztitel} -- {a.regel} ===")
        print(piv.to_string(float_format=lambda x: f"{x:6.2f}", na_rep="   ---"))
        print()

if a.latex:
    for satz, btitel in BENCH:
        d = t[t["bench"] == satz].copy()
        d["zelle"] = d["arm"].map(LANG) + ", " + d["nfe"].astype(str) + " steps"
        piv = d.pivot_table(index="zelle", columns="K", values="beides")
        piv = piv.reindex([f"{LANG[arm]}, {nfe} steps" for arm, nfe in ZELLEN])
        print(f"% {btitel} -- RMSD<2 and PB-valid -- {a.regel}")
        print(r"\begin{tabular}{@{}l" + "r" * len(piv.columns) + r"@{}}")
        print(r"\toprule")
        print("Model & " + " & ".join(f"${k}$" for k in piv.columns) + r" \\")
        print(r"\midrule")
        for name, row in piv.iterrows():
            print(f"{name} & "
                  + " & ".join("---" if pd.isna(v) else f"{v:.1f}" for v in row)
                  + r" \\")
        print(r"\bottomrule")
        print(r"\end{tabular}")
        print()
