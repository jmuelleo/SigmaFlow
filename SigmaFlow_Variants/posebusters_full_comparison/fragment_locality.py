"""
Locality criterion: are BONDED fragment pairs oriented more consistently
with each other than NON-BONDED ones?

This is the success criterion for the frame fix (STATUS.md, "ROOT CAUSE
GEFUNDEN"). It deliberately replaces the transition-bond-length metric,
which misled the analysis three times.

WHY RELATIVE ROTATIONS
----------------------
Per fragment f, Kabsch gives R_f, the rotation mapping the TRUE fragment
onto the PREDICTED one - i.e. that fragment's orientation error. Its
absolute value is dominated by whatever global orientation error the whole
molecule has, which tells us nothing about internal consistency.

For a pair (f, g) we therefore look at the RELATIVE error

    R_rel = R_f^T R_g ,     angle = arccos((tr(R_rel) - 1) / 2)

A rotation error shared by both fragments cancels exactly: if R_f = R_g,
then R_rel = I and the angle is 0, no matter how wrong the global
orientation is. The angle measures only whether the two fragments are
mis-oriented RELATIVE TO EACH OTHER.

Splitting these pairs into bonded vs. non-bonded gives a built-in control:
- bonded pairs   -> should be strongly coupled if the model learned local
                    geometry (a bond constrains relative orientation);
- non-bonded     -> no direct constraint, expected near the random baseline.

The GAP between the two is the signal. Both near 126.5 deg means the model
has learned no locality at all - which is what the control run showed.

Reference values on the full 209-complex PoseBusters set (STATUS.md):
    SigmaFlow control (job 8512798)  bonded 121.8 | non-bonded 124.9
    SigmaDock         (job 8512922)  bonded 102.9 | non-bonded 126.6
    random baseline                  126.5 for both

Usage: python fragment_locality.py
Run from SigmaFlow_Variants/posebusters_full_comparison/.
"""

import glob
import os

import numpy as np
from rdkit import Chem

TRUE_DIR = "true_ligands"
PRED_DIRS = {
    # --- 6h stage ---------------------------------------------------------
    "SigmaFlow 6h, no fix (8512798)": "sigmaflow_lrfix_pred",
    "SigmaFlow 6h FRAME FIX (8530243)": "sigmaflow_framefix_pred",
    "SigmaDock 6h (8512922)": "sigmadock_lrfix_pred",
    # --- 12h stage, compute-matched pair ----------------------------------
    "SigmaFlow 12h FRAME FIX (8541310)": "sigmaflow_12h_pred",
    "SigmaDock 12h (8541439)": "sigmadock_12h_pred",
    # --- loss variants, all on top of the frame fix, 6h -------------------
    # Control for these is the 6h FRAME FIX run, NOT the old 8512798 -
    # otherwise the loss change and the frame fix are confounded.
    "  + variant A time-weight (8540758)": "sigmaflow_a_pred",
    "  + variant B rot-data-space (8534746)": "sigmaflow_framefix_b_pred",
    "  + variant C anchor-dist (8534747)": "sigmaflow_framefix_c_pred",
}
# NOTE: this metric needs no cross-run matching. The bonded/non-bonded split
# is made WITHIN each run's own recovered fragmentation, so every run is
# self-consistent. (full_metrics.py is different: its cut-bond comparison is
# across runs and therefore restricted to a matched subset.)
TOL = 0.02  # Angstrom; same threshold as bond_length_check.py
MIN_FRAG_ATOMS = 3  # fewer than 3 atoms -> rotation is not well defined
RANDOM_BASELINE_DEG = 126.5


from ligand_reference import best_copy, load_copies, load_first as load_mol


def find_fragments(mol_true, mol_pred) -> np.ndarray:
    """Assign each atom a fragment id via connected components over bonds
    whose length is preserved between true and predicted pose.

    Fragments are placed as rigid bodies, so intra-fragment bonds keep their
    length exactly; bonds whose length changed are exactly the cut bonds.
    """
    n = mol_true.GetNumAtoms()
    ct, cp = mol_true.GetConformer(), mol_pred.GetConformer()
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for bond in mol_true.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        dt = ct.GetAtomPosition(i).Distance(ct.GetAtomPosition(j))
        dp = cp.GetAtomPosition(i).Distance(cp.GetAtomPosition(j))
        if abs(dp - dt) <= TOL:
            ra, rb = find(i), find(j)
            if ra != rb:
                parent[rb] = ra

    roots, frag_id = {}, np.empty(n, dtype=int)
    for a in range(n):
        r = find(a)
        if r not in roots:
            roots[r] = len(roots)
        frag_id[a] = roots[r]
    return frag_id


def kabsch_rotation(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """P=predicted, Q=true, both [n,3]. Returns R [3,3] with Q_centred @ R.T ~ P_centred.

    Same convention as kabsch_errors() in placement_vs_bond_error.py, but
    returns the matrix instead of only its angle, because relative errors
    need the matrices themselves.
    """
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    H = Qc.T @ Pc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


def rotation_angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


def summarise(name: str, values: list[float]) -> str:
    if not values:
        return f"  {name:<26}: (no pairs)"
    a = np.array(values)
    return (
        f"  {name:<26}: mean={a.mean():6.1f} deg  median={np.median(a):6.1f} deg  "
        f"(n={len(a)})"
    )


def main() -> None:
    true_paths = sorted(glob.glob(os.path.join(TRUE_DIR, "*_ligands.sdf")))
    print(f"True ligands found: {len(true_paths)}")
    print(f"Random baseline: {RANDOM_BASELINE_DEG} deg\n")

    for label, pred_dir in PRED_DIRS.items():
        if not os.path.isdir(pred_dir):
            print(f"=== {label}: SKIPPED (no {pred_dir}/) ===\n")
            continue

        bonded, nonbonded = [], []
        per_complex_gap = []  # paired within complex, immune to composition effects
        n_ok = 0

        for true_path in true_paths:
            cid = os.path.basename(true_path).replace("_ligands.sdf", "")
            cands = glob.glob(os.path.join(pred_dir, f"{cid}__*_seed0.sdf"))
            if not cands:
                continue
            mol_pred = load_mol(cands[0])
            if mol_pred is None:
                continue
            # Score against the crystallographic copy this pose is closest to.
            # 84 of 209 files hold several copies; taking copy 0 measures a
            # correct pose against the wrong site. See ligand_reference.py.
            mol_true, _, _ = best_copy(
                mol_pred.GetConformer().GetPositions(), load_copies(true_path))
            if mol_true is None:
                continue
            if mol_true.GetNumBonds() != mol_pred.GetNumBonds():
                continue

            pos_true = mol_true.GetConformer().GetPositions()
            pos_pred = mol_pred.GetConformer().GetPositions()
            frag_id = find_fragments(mol_true, mol_pred)

            # Per-fragment rotation error, only where it is well defined.
            rots = {}
            for f in np.unique(frag_id):
                sel = frag_id == f
                if sel.sum() < MIN_FRAG_ATOMS:
                    continue
                rots[int(f)] = kabsch_rotation(pos_pred[sel], pos_true[sel])
            if len(rots) < 2:
                continue
            n_ok += 1

            # Which fragment pairs are joined by a (cut) bond in the TRUE molecule.
            bonded_pairs = set()
            for bond in mol_true.GetBonds():
                fi = int(frag_id[bond.GetBeginAtomIdx()])
                fj = int(frag_id[bond.GetEndAtomIdx()])
                if fi != fj and fi in rots and fj in rots:
                    bonded_pairs.add((min(fi, fj), max(fi, fj)))

            ids = sorted(rots)
            here_b, here_n = [], []
            for a_i in range(len(ids)):
                for b_i in range(a_i + 1, len(ids)):
                    f, g = ids[a_i], ids[b_i]
                    ang = rotation_angle_deg(rots[f].T @ rots[g])
                    if (f, g) in bonded_pairs:
                        bonded.append(ang)
                        here_b.append(ang)
                    else:
                        nonbonded.append(ang)
                        here_n.append(ang)

            if here_b and here_n:
                per_complex_gap.append(np.mean(here_b) - np.mean(here_n))

        print(f"=== {label} ({n_ok} complexes) ===")
        print(summarise("bonded pairs", bonded))
        print(summarise("non-bonded pairs", nonbonded))
        if bonded and nonbonded:
            gap = np.mean(bonded) - np.mean(nonbonded)
            print(f"  {'gap (bonded - nonbonded)':<26}: {gap:+6.1f} deg   (negative = locality)")
        if per_complex_gap:
            g = np.array(per_complex_gap)
            print(
                f"  {'gap, paired per complex':<26}: {g.mean():+6.1f} deg  "
                f"median={np.median(g):+6.1f}  (n={len(g)} complexes with both)"
            )
            # Bootstrap CI over COMPLEXES (the independent unit - fragment
            # pairs within one complex share a molecule and are not
            # independent, so resampling pairs would understate the spread).
            rng = np.random.default_rng(0)
            boot = rng.choice(g, size=(10000, len(g)), replace=True).mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            frac = float((g < 0).mean())
            print(f"  {'95% CI (bootstrap)':<26}: [{lo:+.1f}, {hi:+.1f}] deg")
            print(f"  {'complexes with gap < 0':<26}: {frac:.0%}")
        print()


if __name__ == "__main__":
    main()
