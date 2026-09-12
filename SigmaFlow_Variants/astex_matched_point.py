"""The Astex compute-matched point, as an expectation rather than one subset.

The tables previously read the first 101 of the 140 available seeds. That is a
single realisation, and on this benchmark it is a noisy one. Averaging over all
101-subsets is the quantity 101 draws actually deliver, and it is available in
closed form, so it is what should be reported.

Because the per-complex quantity is then a probability rather than a 0/1
outcome, the paired comparison is a Wilcoxon signed-rank test over complexes
rather than McNemar on discordant pairs.
"""
import importlib

import numpy as np

pr = importlib.import_module("plot_ranking_lib")

RNG = np.random.default_rng(0)


def paired_bootstrap(a, b, n=8000):
    """Two-sided p and interval for the mean paired difference over complexes.

    This is the convention the thesis uses for rates. A signed-rank test is
    not appropriate here, because the per-complex differences are expectations
    rather than outcomes: almost every complex carries a small non-zero
    difference, and a rank test is then driven by the many near-ties instead
    of by the complexes that actually change hands.
    """
    d = (b - a).to_numpy()
    idx = RNG.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return d.mean(), np.percentile(means, [2.5, 97.5]), max(p, 1 / n)

BASE = ("AX85", "SigmaDock", 25)
ARMS = [("SigmaFlow-Minimal", 5), ("SigmaFlow-Separate", 5)]


def cell(key):
    rd, gn, po = pr.AX_CELLS[key]
    return pr.load_ax(rd, gn, po)


def report(bench, base_key, arm_keys, draws):
    if bench == "PB308":
        root, c = pr.CELLS[base_key]
        base = pr.load_pb(root, c)
        load = lambda k: pr.load_pb(*pr.CELLS[k])
    else:
        base = cell(base_key)
        load = cell
    for target, name in (("both", "RMSD<2 and PB-valid"), ("acc", "RMSD<2")):
        print(f"\n=== {bench}, {name} ===")
        a = pr.per_complex(base, target, 40)
        print(f"  SigmaDock 25 steps, 40 draws           {100 * a.mean():6.2f}")
        for arm, nfe in arm_keys:
            m = load((bench, arm, nfe))
            for K in draws:
                b = pr.per_complex(m, target, K).reindex(a.index)
                diff, ci, p = paired_bootstrap(a, b)
                print(f"  {arm:19} {nfe} steps, {K:3} draws  "
                      f"{100 * b.mean():6.2f}   {100 * diff:+6.2f} pp   "
                      f"[{100 * ci[0]:+5.1f}, {100 * ci[1]:+5.1f}]   p={p:.3g}")


report("AX85", BASE, ARMS, (101, 140))
report("PB308", ("PB308", "SigmaDock", 25),
       [("SigmaFlow-Minimal", 5), ("SigmaFlow-Separate", 5)], (140,))
