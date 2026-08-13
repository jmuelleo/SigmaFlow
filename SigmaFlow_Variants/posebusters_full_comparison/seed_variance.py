"""
How much of the SigmaFlow-vs-SigmaDock difference is method, and how much is
sampling noise?

WHY THIS EXISTS
Every comparison in this project so far rests on ONE sampling draw per
method. Both methods start from a RANDOMLY drawn initial pose, so a single
draw is a single sample from an unmeasured distribution. Without knowing that
distribution's width we cannot say whether e.g. "1.9% vs 4.8% under 2 A" is a
method difference or a coin flip. This script measures the width, using 10
independent seeds per method on the same 209 complexes.

WHAT IT REPORTS
  1. Per-seed aggregates. If the ten values for one method scatter as widely
     as the gap between the methods, the gap means nothing.
  2. Within-complex spread: how much does the SAME complex move between
     seeds?
  3. Paired method comparison on per-complex seed MEANS - averaging over
     seeds removes sampling noise, leaving the method effect.
  4. Outlier consistency. SigmaDock put 42 complexes beyond 20 A in the
     single-draw comparison. Are those the SAME complexes every time
     (systematic failure) or different ones (bad luck)? This is the question
     the robustness claim stands or falls on.
  5. Best-of-10 ("oracle top-1"). The paper ranks N samples with a trained
     confidence model; we have no such model, so best-of-10 is the upper
     bound that ranking could reach. Reported for both methods equally.

CAVEAT: this measures SAMPLING variance only. Both methods are still a single
TRAINING run - run-to-run variance of training remains unmeasured.

Usage: python seed_variance.py
Run from SigmaFlow_Variants/posebusters_full_comparison/.
"""

import glob
import os

import numpy as np

from ligand_reference import best_copy, load_copies

TRUE_DIR = "true_ligands"
METHODS = {
    "SigmaFlow 12h": "seeds10_sigmaflow/last",
    "SigmaDock 12h": "seeds10_sigmadock/last",
}
N_SEEDS = 10
OUTLIER_A = 20.0


def rmsds(P: np.ndarray, Q: np.ndarray) -> tuple[float, float]:
    """(raw, after optimal rigid alignment). P=pred, Q=true, both [n,3]."""
    raw = float(np.sqrt(((P - Q) ** 2).sum(1).mean()))
    cP, cQ = P.mean(0), Q.mean(0)
    Pc, Qc = P - cP, Q - cQ
    H = Qc.T @ Pc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    aligned = float(np.sqrt(((Pc - Qc @ R.T) ** 2).sum(1).mean()))
    return raw, aligned


def collect(root: str, cids: list[str], true_copies: dict) -> np.ndarray:
    """Returns [n_complexes, N_SEEDS] raw RMSD, NaN where a file is missing.

    Scored against the crystallographic copy the pose is closest to - see
    ligand_reference.py for why taking copy 0 is wrong on 84 of 209 complexes.
    """
    out = np.full((len(cids), N_SEEDS), np.nan)
    for s in range(N_SEEDS):
        for i, cid in enumerate(cids):
            hits = glob.glob(os.path.join(root, f"seed_{s}", f"{cid}__*_seed0.sdf"))
            if not hits:
                continue
            copies = load_copies(hits[0])
            if not copies:
                continue
            P = copies[0].GetConformer().GetPositions()
            _, r, _ = best_copy(P, true_copies[cid])
            if np.isfinite(r):
                out[i, s] = r
    return out


def main() -> None:
    true_paths = sorted(glob.glob(os.path.join(TRUE_DIR, "*_ligands.sdf")))
    cids, true_copies = [], {}
    for tp in true_paths:
        cid = os.path.basename(tp).replace("_ligands.sdf", "")
        copies = load_copies(tp)
        if not copies:
            continue
        cids.append(cid)
        true_copies[cid] = copies
    multi = sum(1 for c in cids if len(true_copies[c]) > 1)
    print(f"Referenzliganden: {len(cids)}, davon mit mehreren Kristallkopien: "
          f"{multi} (gewertet wird die naechstgelegene Kopie)")

    data = {}
    for label, root in METHODS.items():
        if not os.path.isdir(root):
            print(f"SKIPPED (fehlt): {label} -> {root}")
            continue
        data[label] = collect(root, cids, true_copies)
        print(f"{label}: geladen, {np.isfinite(data[label]).sum()} von "
              f"{len(cids)*N_SEEDS} Posen")

    # ---- 1. per-seed aggregates -------------------------------------------
    print("\n" + "=" * 74)
    print("1. AGGREGATE PRO SEED  (streuen sie so stark wie der Methodenunterschied?)")
    print("=" * 74)
    for label, M in data.items():
        med = np.nanmedian(M, axis=0)
        lt2 = 100 * np.nanmean(M < 2.0, axis=0)
        out = (M > OUTLIER_A).sum(axis=0)
        print(f"\n{label}")
        print(f"  Median-RMSD je Seed : min={med.min():.2f}  max={med.max():.2f}  "
              f"Spanne={med.max()-med.min():.2f} A   (SD={med.std(ddof=1):.3f})")
        print(f"  Anteil <2 A je Seed : min={lt2.min():.1f}%  max={lt2.max():.1f}%  "
              f"Mittel={lt2.mean():.1f}%   (SD={lt2.std(ddof=1):.2f})")
        print(f"  Komplexe >{OUTLIER_A:.0f} A je Seed: min={out.min()}  max={out.max()}  "
              f"Mittel={out.mean():.1f}")

    # ---- 2. within-complex spread -----------------------------------------
    print("\n" + "=" * 74)
    print("2. STREUUNG DESSELBEN KOMPLEXES ZWISCHEN SEEDS")
    print("=" * 74)
    for label, M in data.items():
        sd = np.nanstd(M, axis=1, ddof=1)
        rng = np.nanmax(M, axis=1) - np.nanmin(M, axis=1)
        print(f"  {label:<16} SD  Median={np.nanmedian(sd):.2f} A  Mittel={np.nanmean(sd):.2f} A"
              f"   |  Spannweite Median={np.nanmedian(rng):.2f} A")

    # ---- 3. paired comparison on seed means -------------------------------
    if len(data) == 2:
        (la, A), (lb, B) = data.items()
        ma, mb = np.nanmean(A, axis=1), np.nanmean(B, axis=1)
        ok = np.isfinite(ma) & np.isfinite(mb)
        d = ma[ok] - mb[ok]
        rng_ = np.random.default_rng(0)
        boot = rng_.choice(d, size=(10000, d.size), replace=True).mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print("\n" + "=" * 74)
        print(f"3. GEPAART AUF SEED-MITTELWERTEN  ({la} minus {lb})")
        print("=" * 74)
        print(f"  Median der Differenz : {np.median(d):+.2f} A")
        print(f"  Mittel der Differenz : {d.mean():+.2f} A   CI [{lo:+.2f}, {hi:+.2f}]")
        print(f"  {la} besser bei      : {100*(d<0).mean():.0f}% der Komplexe")

    # ---- 4. outlier consistency -------------------------------------------
    print("\n" + "=" * 74)
    print(f"4. SIND DIE AUSREISSER (>{OUTLIER_A:.0f} A) SYSTEMATISCH ODER PECH?")
    print("=" * 74)
    for label, M in data.items():
        cnt = (M > OUTLIER_A).sum(axis=1)          # in wie vielen der 10 Seeds
        ever = int((cnt > 0).sum())
        always = int((cnt == N_SEEDS).sum())
        print(f"\n{label}")
        print(f"  Komplexe, die in MINDESTENS einem Seed ausreissen : {ever}")
        print(f"  davon in ALLEN 10 Seeds                           : {always}")
        if ever:
            print(f"  -> {100*always/ever:.0f}% der betroffenen Komplexe reissen IMMER aus"
                  f"  (systematisch, nicht Pech)")

    # ---- 5. best-of-10 ----------------------------------------------------
    print("\n" + "=" * 74)
    print("5. BEST-OF-10 (Obergrenze dessen, was ein Ranking erreichen koennte)")
    print("=" * 74)
    for label, M in data.items():
        best = np.nanmin(M, axis=1)
        single = M[:, 0]
        print(f"  {label:<16} <2 A: ein Seed {100*np.nanmean(single<2):.1f}%"
              f"  ->  best-of-10 {100*np.nanmean(best<2):.1f}%"
              f"   |  Median-RMSD {np.nanmedian(single):.2f} -> {np.nanmedian(best):.2f} A")


if __name__ == "__main__":
    main()
