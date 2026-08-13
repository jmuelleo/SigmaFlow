"""How many ODE integration steps does SigmaFlow actually need?

THE QUESTION
Every SigmaFlow run so far sampled with ode.num_steps=25 - a value inherited
from SigmaDock, where the paper tuned it for a DIFFUSION reverse process
("diminishing returns with more than 20-30 steps", Prat et al. 2026, App. F).
Nothing says a flow-matching ODE has the same optimum. Two outcomes, both
useful:

  flat from ~10 steps    -> 25 is fine, sampling is not the bottleneck, and
                            we could sample several times cheaper
  still improving at 200 -> we have been UNDER-integrating all along, and
                            every past comparison understated SigmaFlow

DESIGN
7 step counts x 3 seeds (job array 8554148). Three seeds per point because
seed_variance.py established that a single draw scatters by ~4.5 A per
complex - far more than any plausible step effect. Everything below is
therefore reported as mean +/- SD ACROSS SEEDS, and the step-to-step
comparison is done PAIRED per complex, which removes complex difficulty and
leaves only the step effect.

WHAT IS MEASURED
  raw RMSD      - the docking number: placement AND internal geometry
  aligned RMSD  - after optimal rigid superposition, i.e. internal geometry
                  alone. Splitting them says WHICH part more integration
                  helps: an ODE that is under-integrated should mostly
                  improve placement.

Scored against the nearest crystallographic copy - see ligand_reference.py.

CAVEAT on cost: the wall-clock times of this job array are NOT a cost
measurement. With --array=0-20%5 the early tasks ran five at a time and
contended for CPU and filesystem; only the last-running tasks scale cleanly
(~1.3 min fixed + ~2.4 s per step). For real sampling cost, rerun with %1.

Usage: python stepsweep_curve.py
Run from SigmaFlow_Variants/posebusters_full_comparison/.
"""

import glob
import os

import numpy as np

from ligand_reference import best_copy, load_copies

TRUE_DIR = "true_ligands"
ROOT = os.path.join("stepsweep", "sampling_output_pb_stepsweep")
STEPS = [5, 10, 15, 25, 50, 100, 200]
SEEDS = [0, 1, 2]
DEFAULT_STEPS = 25  # the inherited value everything else was measured at


def aligned_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """RMSD after optimal rigid superposition (Kabsch). P=pred, Q=true."""
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    U, _, Vt = np.linalg.svd(Qc.T @ Pc)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return float(np.sqrt(((Pc - Qc @ R.T) ** 2).sum(1).mean()))


def main() -> None:
    # ---- reference ---------------------------------------------------------
    cids, true_copies = [], {}
    for tp in sorted(glob.glob(os.path.join(TRUE_DIR, "*_ligands.sdf"))):
        cid = os.path.basename(tp).replace("_ligands.sdf", "")
        copies = load_copies(tp)
        if copies:
            cids.append(cid)
            true_copies[cid] = copies
    print(f"Referenzliganden: {len(cids)}  "
          f"(davon {sum(1 for c in cids if len(true_copies[c]) > 1)} mit mehreren "
          f"Kristallkopien)")

    # ---- collect: [complex, step, seed] ------------------------------------
    # Two separate cubes so placement and internal geometry can be split.
    raw = np.full((len(cids), len(STEPS), len(SEEDS)), np.nan)
    ali = np.full((len(cids), len(STEPS), len(SEEDS)), np.nan)
    for j, n in enumerate(STEPS):
        for k, s in enumerate(SEEDS):
            d = os.path.join(ROOT, f"steps_{n}", "results", "posebusters",
                             "last", f"seed_{s}")
            if not os.path.isdir(d):
                print(f"FEHLT: {d}")
                continue
            for i, cid in enumerate(cids):
                hits = glob.glob(os.path.join(d, f"{cid}__*_seed0.sdf"))
                if not hits:
                    continue
                pc = load_copies(hits[0])
                if not pc:
                    continue
                P = pc[0].GetConformer().GetPositions()
                mol_true, r, _ = best_copy(P, true_copies[cid])
                if mol_true is None:
                    continue
                raw[i, j, k] = r
                ali[i, j, k] = aligned_rmsd(P, mol_true.GetConformer().GetPositions())
    got = int(np.isfinite(raw).sum())
    print(f"Posen geladen: {got} von {len(cids)*len(STEPS)*len(SEEDS)}\n")

    # ---- 1. the curve ------------------------------------------------------
    # Aggregate WITHIN each seed first, then mean/SD ACROSS seeds. Doing it the
    # other way round (pooling all seeds, then one median) would hide exactly
    # the seed spread this run exists to expose.
    print("=" * 78)
    print("1. KURVE: METRIK UEBER SCHRITTZAHL   (Mittel +/- SD ueber 3 Seeds)")
    print("=" * 78)
    print(f"{'Schritte':>9}{'Median roh':>18}{'<2 A':>16}{'<5 A':>16}{'ausgerichtet':>17}")
    print("-" * 78)
    curve = {}
    for j, n in enumerate(STEPS):
        med = np.nanmedian(raw[:, j, :], axis=0)      # one value per seed
        lt2 = 100 * np.nanmean(raw[:, j, :] < 2.0, axis=0)
        lt5 = 100 * np.nanmean(raw[:, j, :] < 5.0, axis=0)
        alm = np.nanmedian(ali[:, j, :], axis=0)
        curve[n] = (med, lt2, lt5, alm)
        print(f"{n:>9}"
              f"{med.mean():>12.2f} +-{med.std(ddof=1):<4.2f}"
              f"{lt2.mean():>10.1f}% +-{lt2.std(ddof=1):<4.1f}"
              f"{lt5.mean():>10.1f}% +-{lt5.std(ddof=1):<4.1f}"
              f"{alm.mean():>12.2f} +-{alm.std(ddof=1):<4.2f}")

    # ---- 2. is any difference bigger than the seed noise? ------------------
    print("\n" + "=" * 78)
    print(f"2. GEPAART GEGEN DEN GEERBTEN DEFAULT ({DEFAULT_STEPS} SCHRITTE)")
    print("=" * 78)
    print("Pro Komplex ueber die 3 Seeds gemittelt, dann gepaarte Differenz.")
    print("Bootstrap-CI ueber Komplexe; schliesst es 0 ein, ist der Effekt nicht belegt.\n")
    ref = np.nanmean(raw[:, STEPS.index(DEFAULT_STEPS), :], axis=1)
    rng = np.random.default_rng(0)
    print(f"{'Schritte':>9}{'Diff. Median':>15}{'Diff. Mittel':>15}{'95%-CI':>22}"
          f"{'besser bei':>12}")
    print("-" * 78)
    for j, n in enumerate(STEPS):
        cur = np.nanmean(raw[:, j, :], axis=1)
        ok = np.isfinite(cur) & np.isfinite(ref)
        d = cur[ok] - ref[ok]
        if n == DEFAULT_STEPS or d.size == 0:
            continue
        boot = rng.choice(d, size=(10000, d.size), replace=True).mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        mark = "" if lo < 0 < hi else "  <- signifikant"
        print(f"{n:>9}{np.median(d):>+15.3f}{d.mean():>+15.3f}"
              f"   [{lo:+.3f}, {hi:+.3f}]{100*(d<0).mean():>11.0f}%{mark}")

    # ---- 3. how big is the step effect next to the seed effect? ------------
    print("\n" + "=" * 78)
    print("3. GROESSENORDNUNGEN: SCHRITTZAHL GEGEN SEED-RAUSCHEN")
    print("=" * 78)
    per_complex_seed_sd = np.nanstd(raw[:, STEPS.index(DEFAULT_STEPS), :],
                                    axis=1, ddof=1)
    means = np.array([np.nanmean(raw[:, j, :], axis=1) for j in range(len(STEPS))])
    span = np.nanmax(means, axis=0) - np.nanmin(means, axis=0)
    print(f"  Streuung EINES Komplexes ueber die 3 Seeds (bei {DEFAULT_STEPS} Schritten):")
    print(f"      SD Median {np.nanmedian(per_complex_seed_sd):.2f} A")
    print(f"  Spanne EINES Komplexes ueber ALLE 7 Schrittzahlen (seed-gemittelt):")
    print(f"      Median {np.nanmedian(span):.2f} A")
    print("\n  Ist die zweite Zahl deutlich kleiner als die erste, ist die Schrittzahl")
    print("  im getesteten Bereich irrelevant gegenueber der Ziehungsstreuung.")


if __name__ == "__main__":
    main()
