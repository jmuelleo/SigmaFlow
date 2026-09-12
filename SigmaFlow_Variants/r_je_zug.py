"""Je Zug, Ranking-Hebel und Saettigung -- fuer jede Zelle.

Die drei Groessen, aus denen die Kernerzaehlung besteht:
  ZUG        Mittel ueber alle Einzelposen: wie gut ist EINE Ziehung?
  TOP-1(K)   nach Mixed Score aus K Ziehungen, 1000 Zufallsteilmengen
  ORAKEL(K)  enthaelt die Teilmenge ueberhaupt eine gute Pose?
Der Ranking-Hebel ist TOP-1(K) - ZUG, die Praezision@1 ist TOP-1/ORAKEL.
"""
import argparse

import pandas as pd

from zellen import SAETZE, lade_zelle, top1

p = argparse.ArgumentParser()
p.add_argument("--satz", choices=sorted(SAETZE), default="pb308")
p.add_argument("--ziel", default="beides", choices=["acc", "valid", "beides"])
a = p.parse_args()

KS = [1, 2, 3, 5, 10, 20, 40, 100, 200]

for z in SAETZE[a.satz]["zellen"]:
    m = lade_zelle(z, leise=True)
    name = f"{z['arm']} {z['nfe']} Schritte, K={z['k']}"
    print(f"\n{'='*78}\n{name}   ({m['complex'].nunique()} Komplexe, "
          f"{len(m)} Posen)\n{'='*78}")

    print("  JE ZUG (eine einzige Ziehung):")
    for ziel, tag in [("acc", "RMSD < 2 A"), ("valid", "PB-valide"),
                      ("beides", "beides")]:
        print(f"    {tag:12s} {top1(m, ziel, 1, 'zug'):6.2f} %")

    print(f"\n  KURVE, Ziel '{a.ziel}':")
    print(f"    {'K':>5} {'Top-1':>8} {'Orakel':>8} {'Praez@1':>9} "
          f"{'Hebel':>8}")
    zug = top1(m, a.ziel, 1, "zug")
    for K in KS:
        if K > z["k"]:
            continue
        t = top1(m, a.ziel, K, "mixed")
        o = top1(m, a.ziel, K, "orakel")
        print(f"    {K:5d} {t:7.2f} % {o:7.2f} % {100*t/o:8.2f} % "
              f"{t-zug:+7.2f}")
    print(f"\n    Ranking-Hebel bei K = {z['k']}: "
          f"{top1(m, a.ziel, z['k'], 'mixed') - zug:+.2f} Punkte "
          f"(von {zug:.2f} auf {top1(m, a.ziel, z['k'], 'mixed'):.2f})")
