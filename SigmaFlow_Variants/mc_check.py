"""Monte Carlo cross-check of the closed-form Top-1 curve.

For each complex, draw B random subsets of size K without replacement from its
generated poses, rank each subset by the mixed score, take the top pose and
record whether it satisfies the label. The mean over the B subsets is that
complex's estimate; the mean over complexes is the reported rate. That is the
same order of averaging the closed form uses, so the two must agree up to
Monte Carlo error.
"""
import importlib
import sys

import numpy as np

import final_tables as ft

pr = importlib.import_module("plot_ranking_lib")

B = int(sys.argv[1]) if len(sys.argv) > 1 else 100
RNG = np.random.default_rng(1)


def monte_carlo(m, target, K, B):
    """B subsets per complex, drawn at once.

    Sorting a row of uniform noise and keeping the first k columns is a
    uniform k-subset without replacement, which lets all B draws for a
    complex be built as one array rather than in a Python loop.
    """
    per = []
    for _, g in m.groupby("complex"):
        h = g["heur"].to_numpy()
        y = g[target].to_numpy().astype(bool)
        n = len(y)
        k = min(K, n)
        idx = np.argsort(RNG.random((B, n)), axis=1)[:, :k]
        chosen = idx[np.arange(B), np.argmax(h[idx], axis=1)]
        per.append(y[chosen].mean())
    per = np.array(per)
    # Monte Carlo error of the comparison, not the spread across complexes
    mc = 100 * np.sqrt(np.mean(per * (1 - per)) / B / len(per))
    return 100 * per.mean(), mc


data = ft.build()
CASES = [("PB308", "SigmaFlow-Minimal", 140), ("PB308", "SigmaFlow-Separate", 140),
         ("AX85", "SigmaFlow-Separate", 101), ("AX85", "SigmaFlow-Minimal", 101)]

print(f"{B} random subsets per complex\n")
print(f"{'cell':<34}{'target':>8}{'Monte Carlo':>16}{'closed form':>13}"
      f"{'diff':>8}")
for bench, arm, K in CASES:
    m = data[(bench, arm, 5)]
    for target in ("acc", "both"):
        mc, se = monte_carlo(m, target, K, B)
        exact = 100 * ft.rule(m, target, K, "heuristic").mean()
        print(f"{bench + ' ' + arm + ' K=' + str(K):<34}{target:>8}"
              f"{mc:>11.2f}±{se:.2f}{exact:>13.2f}{mc - exact:>+8.2f}")
