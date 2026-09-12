"""Choose the complex for the trajectory figure.

The figure must show, in one glance, that coarse integration breaks the
diffusion sampler and not the flow ones. So the complex has to satisfy, on
SEED 0 specifically, since seed 0 is the trajectory that gets drawn:

    four rigid fragments,
    solved by all three arms at twenty-five steps,
    still solved by both flow arms at five steps,
    and NOT solved by the diffusion arm at five steps.

Fragment counts come from fragments_vs_performance.csv, which is a structural
property and therefore independent of the run it was measured in. Everything
else is read from the 72-hour endpoint cells.
"""
import pathlib

import pandas as pd

import final_tables as ft

HERE = pathlib.Path(".").resolve()
FRAGS = HERE.parent / "fragments_vs_performance.csv"
ARMS = ["SigmaDock", "SigmaFlow-Minimal", "SigmaFlow-Separate"]
SEED = 0

frag = pd.read_csv(FRAGS).set_index("complex")["fragments"]
data = ft.build()

# seed-0 outcome per complex for each of the six cells
rows = {}
for arm in ARMS:
    for nfe in (25, 5):
        m = data[("PB308", arm, nfe)]
        s = m[m.seed == SEED].set_index("complex")
        rows[(arm, nfe, "acc")] = s["acc"]
        rows[(arm, nfe, "both")] = s["both"]
        rows[(arm, nfe, "rmsd")] = s["rmsd"] if "rmsd" in s else None

tab = pd.DataFrame({k: v for k, v in rows.items() if v is not None})
# a complex missing from a cell counts as not solved, never as solved
for c in tab.columns:
    tab[c] = tab[c].fillna(False).astype(bool)
tab["fragments"] = frag.reindex(tab.index)

print(f"{len(tab)} complexes, of which {int((tab.fragments == 4).sum())} "
      f"have four fragments\n")

ok = tab[
    (tab["fragments"] == 4)
    & tab[("SigmaDock", 25, "acc")] & tab[("SigmaFlow-Minimal", 25, "acc")]
    & tab[("SigmaFlow-Separate", 25, "acc")]
    & tab[("SigmaFlow-Minimal", 5, "acc")] & tab[("SigmaFlow-Separate", 5, "acc")]
    & ~tab[("SigmaDock", 5, "acc")]
]

print(f"{len(ok)} candidates meet every condition on seed {SEED}:\n")
cols = [(a, n, "both") for a in ARMS for n in (25, 5)]
if len(ok):
    show = ok[cols].copy()
    show.columns = [f"{a.replace('SigmaFlow-', 'SF-')} {n}" for a, n, _ in cols]
    print(show.to_string())
    print("\ncolumns are the combined criterion; the accuracy condition is "
          "already enforced above")
else:
    print("none. Relaxing: drop the requirement that SigmaDock succeeds at 25 "
          "steps.\n")
    loose = tab[
        (tab["fragments"] == 4)
        & tab[("SigmaFlow-Minimal", 25, "acc")]
        & tab[("SigmaFlow-Separate", 25, "acc")]
        & tab[("SigmaFlow-Minimal", 5, "acc")]
        & tab[("SigmaFlow-Separate", 5, "acc")]
        & ~tab[("SigmaDock", 5, "acc")]
    ]
    print(f"{len(loose)} candidates without that condition:")
    print(loose[[("SigmaDock", 25, "acc")] + cols].to_string())
