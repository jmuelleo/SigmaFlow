"""Minimal against Separate, with intervals, on both benchmarks.

Also shows how far the realised value of one particular set of 140 draws sits
from the expectation over 140-subsets of the 200 available, which is what made
the ordering look settled before.
"""
import importlib

import final_tables as ft

pr = importlib.import_module("plot_ranking_lib")

data = ft.build()

for bench, draws in (("PB308", (140, 200)), ("AX85", (101, 200))):
    mn = data[(bench, "SigmaFlow-Minimal", 5)]
    sp = data[(bench, "SigmaFlow-Separate", 5)]
    for target, name in (("both", "RMSD<2 and PB-valid"), ("acc", "RMSD<2")):
        print(f"\n=== {bench}, {name}, Separate minus Minimal ===")
        for K in draws:
            a = ft.rule(mn, target, K, "heuristic")
            b = ft.rule(sp, target, K, "heuristic").reindex(a.index)
            diff, ci, p = ft.boot(a, b)
            print(f"  K={K:3}   Minimal {100 * a.mean():6.2f}   "
                  f"Separate {100 * b.mean():6.2f}   {100 * diff:+6.2f} pp   "
                  f"[{100 * ci[0]:+6.2f},{100 * ci[1]:+6.2f}]   p={p:.3g}")

print("\n=== what one particular set of 140 draws gave, against the "
      "expectation over all of them ===")
for arm in ("SigmaFlow-Minimal", "SigmaFlow-Separate"):
    m = data[("PB308", arm, 5)]
    first = m[m.seed < 140]
    realised = 100 * ft.rule(first, "both", 140, "heuristic").mean()
    expected = 100 * ft.rule(m, "both", 140, "heuristic").mean()
    print(f"  {arm:19} first 140 seeds {realised:6.2f}   "
          f"expectation {expected:6.2f}   {realised - expected:+6.2f}")
