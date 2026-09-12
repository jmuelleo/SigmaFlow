"""Revised table rows: every K-dependent quantity as an expectation over
K-subsets rather than as the first K seeds.

At K equal to the number of available draws the two coincide, so only the
Astex rows at 101 draws change. The paired comparisons use the bootstrap over
complexes that the thesis already declares for rates.
"""
import importlib
import math

import numpy as np
import pandas as pd

pr = importlib.import_module("plot_ranking_lib")
RNG = np.random.default_rng(0)


def per_complex_rule(m, target, K, rule):
    """Expected success of a given selection rule at K draws, per complex."""
    out = {}
    for cid, g in m.groupby("complex"):
        y_all = g[target].to_numpy().astype(float)
        n = len(y_all)
        k = min(K, n)
        if rule == "draw":
            out[cid] = y_all.mean()
            continue
        if rule == "oracle":
            s = int(y_all.sum())
            den = pr.logC(n, k)
            out[cid] = 1.0 - (math.exp(pr.logC(n - s, k) - den)
                              if n - s >= k else 0.0)
            continue
        col = "affinity" if rule == "vinardo" else "heur"
        asc = rule == "vinardo"          # Vinardo: lower is better
        h = g.sort_values(col, ascending=asc)
        y = h[target].to_numpy().astype(float)
        den = pr.logC(n, k)
        w = np.array([math.exp(pr.logC(n - 1 - i, k - 1) - den)
                      for i in range(n - k + 1)])
        out[cid] = float(y[:len(w)] @ w)
    return pd.Series(out)


def boot(a, b, n=8000):
    d = (b - a).to_numpy()
    idx = RNG.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return d.mean(), max(p, 1 / n)


def load(bench, arm, nfe):
    if bench == "PB308":
        return pr.load_pb(*pr.CELLS[(bench, arm, nfe)])
    return pr.load_ax(*pr.AX_CELLS[(bench, arm, nfe)])


ROWS = {
    "PB308": [("SigmaDock", 25, 40), ("SigmaFlow-Minimal", 25, 40),
              ("SigmaFlow-Separate", 25, 40), ("SigmaDock", 5, 40),
              ("SigmaFlow-Minimal", 5, 140), ("SigmaFlow-Separate", 5, 140)],
    "AX85": [("SigmaDock", 25, 40), ("SigmaFlow-Minimal", 25, 40),
             ("SigmaFlow-Separate", 25, 40), ("SigmaDock", 5, 40),
             ("SigmaFlow-Minimal", 5, 101), ("SigmaFlow-Separate", 5, 101),
             ("SigmaFlow-Minimal", 5, 140), ("SigmaFlow-Separate", 5, 140)],
}

cache = {}
for bench, rows in ROWS.items():
    for target, name in (("both", "RMSD<2 and PB-valid with protein"),
                         ("acc", "RMSD<2")):
        print(f"\n=== {bench}, {name} ===")
        print(f"{'row':<34}{'per draw':>10}{'Vinardo':>10}"
              f"{'mixed':>10}{'oracle':>10}")
        for arm, nfe, K in rows:
            m = cache.setdefault((bench, arm, nfe), load(bench, arm, nfe))
            vals = [100 * per_complex_rule(m, target, K, r).mean()
                    for r in ("draw", "vinardo", "heuristic", "oracle")]
            print(f"{arm + ', ' + str(nfe) + '/' + str(K):<34}"
                  + "".join(f"{v:10.2f}" for v in vals))

        base = cache[(bench, "SigmaDock", 25)]
        a = per_complex_rule(base, target, 40, "heuristic")
        print("  -- mixed score, paired bootstrap against SigmaDock 25/40 --")
        for arm, nfe, K in rows:
            if (arm, nfe) == ("SigmaDock", 25):
                continue
            m = cache[(bench, arm, nfe)]
            b = per_complex_rule(m, target, K, "heuristic").reindex(a.index)
            d, p = boot(a, b)
            print(f"     {arm + ', ' + str(nfe) + '/' + str(K):<32}"
                  f"{100 * d:+7.2f} pp   p={p:.3g}")

        if bench == "AX85":
            mn = per_complex_rule(cache[(bench, "SigmaFlow-Minimal", 5)],
                                  target, 101, "heuristic")
            sp = per_complex_rule(cache[(bench, "SigmaFlow-Separate", 5)],
                                  target, 101, "heuristic").reindex(mn.index)
            d, p = boot(mn, sp)
            print(f"     Separate minus Minimal, 5/101   {100 * d:+7.2f} pp"
                  f"   p={p:.3g}")
