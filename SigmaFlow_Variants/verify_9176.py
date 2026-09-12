"""Check the two values that came out identical, by counting complexes.

At K equal to the pool size there is one subset, so the selected pose of each
complex is determined and the rate is a plain count over complexes. Anything
identical to two decimals should therefore be an identical count, and the
accurate set must be a superset of the accurate-and-valid set.
"""
import importlib

import final_tables as ft

pr = importlib.import_module("plot_ranking_lib")

data = ft.build()


def selected(m, K):
    """The pose each complex's ranker picks out of its whole pool."""
    x = m[m.seed < K]
    return x.loc[x.groupby("complex")["heur"].idxmax()].set_index("complex")


for bench, arm, K, n_cx in (("AX85", "SigmaFlow-Separate", 200, 85),
                            ("AX85", "SigmaFlow-Minimal", 200, 85)):
    s = selected(data[(bench, arm, 5)], K)
    acc, both = int(s["acc"].sum()), int(s["both"].sum())
    print(f"{bench} {arm} K={K}: {len(s)} complexes")
    print(f"   accurate            {acc:3d}/{len(s)} = {100*acc/len(s):.2f}")
    print(f"   accurate and valid  {both:3d}/{len(s)} = {100*both/len(s):.2f}")
    diff = s[s["acc"] & ~s["both"]]
    print(f"   accurate but invalid: {len(diff)}"
          + (f"  {list(diff.index)[:6]}" if len(diff) else ""))
    assert (s["both"] <= s["acc"]).all(), "both must imply acc"
    print()

# the other coincidence: two arms with the same PB308 oracle on RMSD
for arm in ("SigmaFlow-Minimal", "SigmaFlow-Separate"):
    m = data[("PB308", arm, 5)]
    o = m.groupby("complex")["acc"].any()
    print(f"PB308 {arm} oracle at 200 draws, RMSD: "
          f"{int(o.sum())}/{len(o)} = {100*o.mean():.4f}")
