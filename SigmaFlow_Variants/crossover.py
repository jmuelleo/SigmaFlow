"""Where the flow arms overtake the diffusion arm at equal integration steps.

Only the twenty-five-step cells are compared here, so neither arm gets credit
for a cheaper sample; the only thing that varies is how many poses are drawn
and ranked. Every cell holds forty draws, so the comparison runs to K = 40.
"""
import importlib

import numpy as np

import final_tables as ft

pr = importlib.import_module("plot_ranking_lib")

KS = list(range(1, 41))


def load(bench, arm):
    key = (bench, arm, 25)
    return (pr.load_pb(*pr.CELLS[key]) if bench == "PB308"
            else pr.load_ax(*pr.AX_CELLS[key]))


for bench in ("PB308", "AX85"):
    sd = load(bench, "SigmaDock")
    arms = {a: load(bench, a) for a in ("SigmaFlow-Minimal",
                                        "SigmaFlow-Separate")}
    for target, name in (("both", "RMSD<2 and PB-valid"), ("acc", "RMSD<2")):
        base = np.array([100 * ft.rule(sd, target, K, "heuristic").mean()
                         for K in KS])
        print(f"\n=== {bench}, {name}, twenty-five steps, mixed score ===")
        print("  K    SigmaDock   Minimal  (diff)   Separate  (diff)")
        curves = {a: np.array([100 * ft.rule(m, target, K, "heuristic").mean()
                               for K in KS]) for a, m in arms.items()}
        for i, K in enumerate(KS):
            if K in (1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 40):
                mn, sp = curves["SigmaFlow-Minimal"][i], curves["SigmaFlow-Separate"][i]
                print(f"{K:4}{base[i]:11.2f}{mn:10.2f} ({mn - base[i]:+5.2f})"
                      f"{sp:11.2f} ({sp - base[i]:+5.2f})")
        for a, c in curves.items():
            ahead = np.where(c > base)[0]
            if len(ahead) == 0:
                print(f"  {a}: never ahead up to K = 40")
                continue
            # first K from which it stays ahead
            stable = [i for i in ahead if all(c[j] > base[j]
                                              for j in range(i, len(KS)))]
            first = KS[ahead[0]]
            keeps = KS[stable[0]] if stable else None
            print(f"  {a}: first ahead at K = {first}, "
                  + (f"ahead from K = {keeps} onwards" if keeps
                     else "never ahead for the rest of the range"))
