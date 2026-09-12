"""Gegeben Fehlgriff und vorhandene richtige Pose: in welchem Cluster liegt sie?

Nenner ist immer "Top-1 falsch UND mindestens eine richtige Pose im Pool",
also die rettbaren Faelle. Gefragt ist, ob die richtige Pose im GROESSTEN
Cluster steckt oder in einem kleineren.
"""
import numpy as np
import pandas as pd

ZELLEN = [("SigmaDock", 25), ("Minimal", 25), ("Separate", 25),
          ("Minimal", 5), ("Separate", 5)]
LAB = {"SigmaDock": "SigmaDock", "Minimal": "SigmaFlow-NE",
       "Separate": "SigmaFlow-TR"}
SP = ["arm", "nfe", "S", "complex", "cluster", "g_gr", "contains_correct",
      "selected_by_ranker", "complex_hat_correct", "ranker_richtig"]

for satz in ("pb308", "astex"):
    df = pd.read_csv(f"cluster_{satz}.csv", usecols=SP)
    df = df[np.isclose(df.S, 2.0)]
    print(f"\n{'='*94}\n{satz.upper()}, Clusterschwelle 2,0 A\n{'='*94}")
    print(f"  {'Modell':<14}{'K':>5}{'rettbar':>9}{'im groessten':>14}"
          f"{'nur kleiner':>13}{'Rang der besten':>17}{'Ranker nahm groessten':>23}")
    for arm, nfe in ZELLEN:
        z = df[(df.arm == arm) & (df.nfe == nfe)]
        if z.empty:
            continue
        gross, klein, raenge, nahm_gross = 0, 0, [], 0
        for code, g in z.groupby("complex"):
            if g.ranker_richtig.iloc[0] or not g.complex_hat_correct.iloc[0]:
                continue
            g = g.sort_values("g_gr", ascending=False).reset_index(drop=True)
            mit = g.index[g.contains_correct].tolist()
            if not mit:
                continue
            if 0 in mit:
                gross += 1
            else:
                klein += 1
            raenge.append(min(mit) + 1)
            if bool(g.loc[0, "selected_by_ranker"]):
                nahm_gross += 1
        n = gross + klein
        if not n:
            continue
        print(f"  {LAB[arm]:<14}{40 if nfe==25 else (40 if arm=='SigmaDock' else 200):>5}"
              f"{n:>9}{100*gross/n:>12.1f} %{100*klein/n:>11.1f} %"
              f"{np.median(raenge):>13.0f} (Md){100*nahm_gross/n:>21.1f} %")
    print("\n  'Rang der besten' ist der Groessenrang des groessten Clusters,")
    print("  der eine richtige Pose enthaelt. 1 = der groesste Cluster.")
