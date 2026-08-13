"""
Paired significance tests for the PoseBusters validity checks.

WHY THIS EXISTS
The raw pass rates invite over-reading. SigmaFlow passes the bond-length
check on 55.5% of complexes and SigmaDock on 48.8% - a 6.7 point gap that
looks decisive and is not. Both methods are evaluated on the SAME 209
complexes, so the comparison is PAIRED, and the only informative quantity is
how many complexes the two methods DISAGREE on.

THE TEST (McNemar, exact)
For a binary check, split the complexes into four cells:
    both pass | both fail | only A passes (n01) | only B passes (n10)
The concordant cells carry no information about which method is better - if
both fail a bond-angle check, that says nothing about their difference. Under
the null hypothesis "the methods are equally good", each DISCORDANT complex
is an independent coin flip, so n01 ~ Binomial(n01+n10, 0.5). The exact
two-sided p-value is the binomial tail.

This is why the test can return "not distinguishable" for a 6.7-point gap:
37 vs 23 discordant complexes is well within what a fair coin produces in 60
tosses.

CAVEAT THAT NO TEST CAN FIX
Each method is represented by ONE training run and ONE sampling seed. The
test covers complex-to-complex variation only, not run-to-run variation. A
"significant" result here would still not prove a method difference - it
could be one lucky training run. That is what the 10-seed experiment
(sample_pb_seeds10.sh) is for.

Usage: python validity_significance.py [csv_a] [csv_b]
Defaults to the two 12h runs. Run from posebusters_full_comparison/.
"""

import sys
from math import comb

import pandas as pd

DEFAULT_A = "posebusters_ligandonly_SigmaFlow_12h_framefix.csv"
DEFAULT_B = "posebusters_ligandonly_SigmaDock_12h_lrfix.csv"

# Protein-dependent checks cannot run without the receptor PDBs, and the
# loaded/rmsd columns are not validity checks - excluded from "all checks".
EXCLUDE_PREFIXES = ("position", "mol_cond", "molecule", "rmsd",
                    "mol_pred_loaded", "mol_true_loaded")


def bool_columns(df_a: pd.DataFrame, df_b: pd.DataFrame) -> list[str]:
    return [
        c for c in df_a.columns
        if c in df_b.columns
        and set(df_a[c].dropna().unique()) <= {True, False}
        and not any(c.startswith(p) for p in EXCLUDE_PREFIXES)
        and "rmsd" not in c.lower()
    ]


def mcnemar_exact(x, y) -> tuple[int, int, float]:
    """x, y: boolean arrays of per-complex pass/fail. Returns (n01, n10, p)."""
    n01 = int((x & ~y).sum())   # only x passes
    n10 = int((~x & y).sum())   # only y passes
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n
    return n01, n10, min(1.0, p)


def main() -> None:
    fa = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_A
    fb = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_B
    a, b = pd.read_csv(fa), pd.read_csv(fb)
    cols = bool_columns(a, b)

    name_a = fa.replace("posebusters_ligandonly_", "").replace(".csv", "")
    name_b = fb.replace("posebusters_ligandonly_", "").replace(".csv", "")
    print(f"A = {name_a}\nB = {name_b}\nKomplexe: {len(a)}, Checks: {len(cols)}\n")
    print(f"{'Check':<26}{'A %':>7}{'B %':>7}{'nur A':>7}{'nur B':>7}{'p':>9}   Urteil")

    for c in cols + ["ALLE CHECKS"]:
        if c == "ALLE CHECKS":
            x, y = a[cols].all(axis=1).values, b[cols].all(axis=1).values
        else:
            x = a[c].fillna(False).values.astype(bool)
            y = b[c].fillna(False).values.astype(bool)
        if x.all() and y.all():
            continue  # both perfect - nothing to test
        n01, n10, p = mcnemar_exact(x, y)
        if p >= 0.05:
            verdict = "nicht unterscheidbar"
        else:
            verdict = "A besser" if n01 > n10 else "B besser"
        print(f"{c:<26}{100*x.mean():>6.1f}{100*y.mean():>7.1f}"
              f"{n01:>7}{n10:>7}{p:>9.4f}   {verdict}")

    rmsd_cols = [c for c in a.columns if "rmsd" in c.lower()]
    if rmsd_cols:
        c = rmsd_cols[0]
        x = a[c].fillna(False).values.astype(bool)
        y = b[c].fillna(False).values.astype(bool)
        n01, n10, p = mcnemar_exact(x, y)
        verdict = ("nicht unterscheidbar" if p >= 0.05
                   else ("A besser" if n01 > n10 else "B besser"))
        print(f"\n{c + ' (symmetriekorr.)':<26}{100*x.mean():>6.1f}{100*y.mean():>7.1f}"
              f"{n01:>7}{n10:>7}{p:>9.4f}   {verdict}")


if __name__ == "__main__":
    main()
