"""Cross-check the ranking curves against the numbers printed in the thesis.

At K equal to the number of available draws there is exactly one subset, so
the exact curve must reproduce the table value to the last digit. Any drift
would mean the curve and the table disagree about what is being plotted.
"""
import importlib

pr = importlib.import_module("plot_ranking_lib")

CHECKS = [
    ("PB308", "SigmaDock", 25, 40, "both", 64.17),
    ("PB308", "SigmaFlow-Minimal", 5, 140, "both", 68.08),
    ("PB308", "SigmaFlow-Separate", 5, 140, "both", 69.38),
    ("PB308", "SigmaFlow-Separate", 5, 140, "acc", 75.90),
    ("PB308", "SigmaDock", 25, 40, "acc", 69.06),
    ("AX85", "SigmaDock", 25, 40, "both", 74.12),
    ("AX85", "SigmaFlow-Separate", 5, 101, "both", 88.24),
    ("AX85", "SigmaFlow-Separate", 5, 140, "both", 91.76),
    ("AX85", "SigmaFlow-Minimal", 5, 101, "both", 80.00),
]

for bench, arm, nfe, K, target, expected in CHECKS:
    if bench == "PB308":
        root, cell = pr.CELLS[(bench, arm, nfe)]
        m = pr.load_pb(root, cell)
    else:
        rd, gn, po = pr.AX_CELLS[(bench, arm, nfe)]
        m = pr.load_ax(rd, gn, po)
    top, _ = pr.curves(m, target, [K])
    ok = "ok" if abs(top[0] - expected) < 0.02 else "MISMATCH"
    print(f"{bench:6} {arm:19} {nfe:2} steps  K={K:3}  {target:5}  "
          f"curve {top[0]:6.2f}  table {expected:6.2f}  {ok}")
