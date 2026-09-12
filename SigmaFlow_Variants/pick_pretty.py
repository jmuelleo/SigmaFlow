"""Kandidaten fuer ein schoenes Trajektorienbild aus dem 72-h-Lauf.

Kriterien, alle auf SEED 0, weil seed 0 die gezeichnete Bahn ist:

    vier oder fuenf starre Fragmente   -- weniger wirkt leer, mehr wird dicht
    22 bis 34 Schweratome              -- darunter mager, darueber unlesbar
    von allen drei Armen bei 25 Schritten geloest
    von beiden Flow-Armen auch bei 5 Schritten geloest
    vom Diffusionsarm bei 5 Schritten NICHT geloest

Das letzte Kriterium erhaelt die Aussage der sechsteiligen Abbildung. Wie weit
die Fragmente tatsaechlich wandern, steht hier noch nicht drin: das ergibt sich
erst aus der Trajektorie und wird auf ARC gemessen.
"""
import pathlib

import pandas as pd

import final_tables as ft

HERE = pathlib.Path(".").resolve()
ARMS = ["SigmaDock", "SigmaFlow-Minimal", "SigmaFlow-Separate"]
SEED = 0

meta = pd.read_csv(HERE.parent / "fragments_vs_performance.csv").set_index("complex")
data = ft.build()

sp = {}
for arm in ARMS:
    for nfe in (25, 5):
        m = data[("PB308", arm, nfe)]
        s = m[m.seed == SEED].set_index("complex")
        sp[(arm, nfe, "acc")] = s["acc"]
        sp[(arm, nfe, "rmsd")] = s["rmsd"]

tab = pd.DataFrame(sp)
for c in [c for c in tab.columns if c[2] == "acc"]:
    tab[c] = tab[c].fillna(False).astype(bool)
tab["fragments"] = meta["fragments"].reindex(tab.index)
tab["atoms"] = meta["atoms"].reindex(tab.index)

ok = tab[
    tab["fragments"].isin([4, 5])
    & tab["atoms"].between(22, 34)
    & tab[("SigmaDock", 25, "acc")]
    & tab[("SigmaFlow-Minimal", 25, "acc")]
    & tab[("SigmaFlow-Separate", 25, "acc")]
    & tab[("SigmaFlow-Minimal", 5, "acc")]
    & tab[("SigmaFlow-Separate", 5, "acc")]
    & ~tab[("SigmaDock", 5, "acc")]
].copy()

ok["mittel_rmsd25"] = ok[[(a, 25, "rmsd") for a in ARMS]].mean(axis=1)
ok = ok.sort_values("mittel_rmsd25")

print(f"{len(ok)} Kandidaten\n")
print(f"{'Komplex':<12}{'Frag':>5}{'Atome':>7}{'SD25':>7}{'SD5':>8}"
      f"{'Min25':>7}{'Min5':>7}{'Sep25':>7}{'Sep5':>7}")
for cid, r in ok.iterrows():
    print(f"{cid:<12}{int(r['fragments']):>5}{int(r['atoms']):>7}"
          f"{r[('SigmaDock', 25, 'rmsd')]:>7.2f}{r[('SigmaDock', 5, 'rmsd')]:>8.1f}"
          f"{r[('SigmaFlow-Minimal', 25, 'rmsd')]:>7.2f}"
          f"{r[('SigmaFlow-Minimal', 5, 'rmsd')]:>7.2f}"
          f"{r[('SigmaFlow-Separate', 25, 'rmsd')]:>7.2f}"
          f"{r[('SigmaFlow-Separate', 5, 'rmsd')]:>7.2f}")

print("\nnach mittlerem RMSD bei 25 Schritten sortiert, bester zuerst")
print("Die Wanderstrecke der Fragmente entscheidet sich erst an der "
      "Trajektorie und wird auf ARC gemessen.")
