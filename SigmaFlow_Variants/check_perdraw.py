"""Two ways to read a per-draw rate, and how far apart they are.

    pooled     fraction of all generated poses that satisfy the criterion
    per complex  fraction within each complex, then averaged over complexes

They agree only when every complex contributes the same number of poses.
"""
import importlib

pr = importlib.import_module("plot_ranking_lib")

for label, key in (("PB308 SigmaDock 25", ("PB308", "SigmaDock", 25)),
                   ("PB308 Minimal 25", ("PB308", "SigmaFlow-Minimal", 25)),
                   ("PB308 Separate 25", ("PB308", "SigmaFlow-Separate", 25)),
                   ("AX85  SigmaDock 25", ("AX85", "SigmaDock", 25))):
    m = pr.load_pb(*pr.CELLS[key]) if key[0] == "PB308" \
        else pr.load_ax(*pr.AX_CELLS[key])
    n = m.groupby("complex").size()
    for target in ("both", "acc"):
        pooled = 100 * m[target].mean()
        percx = 100 * m.groupby("complex")[target].mean().mean()
        print(f"{label:20} {target:5} pooled {pooled:6.3f}   "
              f"per complex {percx:6.3f}   diff {percx - pooled:+.3f}")
    print(f"{'':20} poses per complex: {n.min()} to {n.max()}, "
          f"{(n != n.max()).sum()} complexes short of the maximum\n")
