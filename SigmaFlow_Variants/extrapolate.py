"""What the per-draw rate might reach at epoch 390.

The snapshot curves measure generation alone, with scoring and ranking off, so
this extrapolates the per-draw rate and nothing else. A saturating form is
fitted because every measured curve bends; a straight line through the last
two points would be an obvious over-estimate.

    y(e) = A * (1 - exp(-(e - e0) / tau))

The fitted asymptote A is the quantity to distrust most: it is determined by
the curvature at the end of the observed range, where the data are sparsest.
"""
import glob
import pathlib

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

HERE = pathlib.Path(".").resolve()
CURVES = HERE / "learning_curve_min"
TARGET_EPOCH = 390


def model(e, A, tau, e0):
    return A * (1.0 - np.exp(-(e - e0) / tau))


d = pd.concat([pd.read_csv(f) for f in sorted(CURVES.glob("kurve_*_tidy.csv"))],
              ignore_index=True)
d = d[d.arm != "arm"].copy()
for c in ("epoch", "nfe", "u2", "pb_prot", "u2_prot"):
    d[c] = pd.to_numeric(d[c])

print(f"{'arm':<20}{'steps':>6}{'metric':>10}{'last obs':>12}"
      f"{'fit @390':>10}{'asymptote':>11}{'gain':>8}")
for arm in ("SigmaFlow-Minimal", "SigmaFlow-Separate", "SigmaDock"):
    for nfe in (25, 5):
        for col, name in (("u2", "RMSD<2"), ("u2_prot", "combined")):
            g = d[(d.arm == arm) & (d.nfe == nfe)].sort_values("epoch")
            if len(g) < 4:
                print(f"{arm:<20}{nfe:>6}{name:>10}"
                      f"{'too few points':>12}")
                continue
            e, y = g["epoch"].to_numpy(float), g[col].to_numpy(float)
            try:
                p, _ = curve_fit(model, e, y, p0=[y[-1] * 1.3, 100.0, 5.0],
                                 maxfev=20000)
            except RuntimeError:
                print(f"{arm:<20}{nfe:>6}{name:>10}{'no fit':>12}")
                continue
            at390 = model(TARGET_EPOCH, *p)
            print(f"{arm:<20}{nfe:>6}{name:>10}{y[-1]:12.2f}"
                  f"{at390:10.2f}{p[0]:11.2f}{at390 - y[-1]:+8.2f}")

print("\nlast observed slope, percentage points per epoch")
for arm in ("SigmaFlow-Minimal", "SigmaFlow-Separate", "SigmaDock"):
    g = d[(d.arm == arm) & (d.nfe == 25)].sort_values("epoch")
    e, y = g["epoch"].to_numpy(float), g["u2"].to_numpy(float)
    print(f"  {arm:<20} epochs {int(e[-2])}->{int(e[-1])}: "
          f"{(y[-1] - y[-2]) / (e[-1] - e[-2]):+.3f}  "
          f"(linear to 390 would add {(y[-1] - y[-2]) / (e[-1] - e[-2]) * (390 - e[-1]):+.1f})")
