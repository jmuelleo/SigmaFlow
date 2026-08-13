"""
Tests the hypothesis formulated after three failed loss interventions
(STATUS.md, "Variante C ausgewertet"): is the long transition bond an
independent defect, or merely a readout of general per-fragment placement
accuracy?

Method - no new training or sampling needed, works purely on existing
predictions:

1. Fragments are placed as RIGID bodies, so every INTRA-fragment bond keeps
   its true length exactly. Bonds whose length changed are therefore exactly
   the cut (inter-fragment) bonds. We use this to RECOVER the fragmentation
   from the predictions themselves: connected components under
   length-preserving bonds = fragments.
2. Per fragment, Kabsch-align predicted vs. true atom positions to get:
     - centroid error  ||c_pred - c_true||        (translation placement error)
     - rotation error  angle of the optimal rotation (orientation error)
     - residual RMSD after optimal rigid alignment (sanity check: should be
       ~0 if the rigid-placement assumption holds)
3. For each cut bond, correlate its length error against the placement
   errors of the two fragments it joins.

If the correlation is strong, the transition bond error is explained by
general placement inaccuracy, and further loss-term surgery is futile - the
question becomes why SigmaFlow places fragments less accurately at all.

Usage: python placement_vs_bond_error.py
Run from SigmaFlow_Variants/posebusters_full_comparison/.
"""

import glob
import os

import numpy as np
from rdkit import Chem

TRUE_DIR = "true_ligands"
PRED_DIRS = {
    "SigmaFlow_12h": "sigmaflow_pred",
    "SigmaDock_12h": "sigmadock_pred",
    "SigmaFlow_anchordist_12h": "sigmaflow_anchordist_pred",
}
TOL = 0.02  # Angstrom; same threshold as bond_length_check.py


from ligand_reference import best_copy, load_copies, load_first as load_mol


def find_fragments(mol_true, mol_pred) -> np.ndarray:
    """Assign each atom a fragment id via connected components over bonds
    whose length is preserved between true and predicted pose."""
    n = mol_true.GetNumAtoms()
    ct, cp = mol_true.GetConformer(), mol_pred.GetConformer()
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for bond in mol_true.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        dt = ct.GetAtomPosition(i).Distance(ct.GetAtomPosition(j))
        dp = cp.GetAtomPosition(i).Distance(cp.GetAtomPosition(j))
        if abs(dp - dt) <= TOL:
            union(i, j)

    roots = {}
    frag_id = np.empty(n, dtype=int)
    for a in range(n):
        r = find(a)
        if r not in roots:
            roots[r] = len(roots)
        frag_id[a] = roots[r]
    return frag_id


def kabsch_errors(P: np.ndarray, Q: np.ndarray) -> tuple[float, float, float]:
    """P=predicted, Q=true, both [n,3]. Returns (centroid_err, rot_err_deg, residual_rmsd)."""
    cP, cQ = P.mean(axis=0), Q.mean(axis=0)
    centroid_err = float(np.linalg.norm(cP - cQ))

    Pc, Qc = P - cP, Q - cQ
    if len(P) < 3:
        return centroid_err, float("nan"), float("nan")

    H = Qc.T @ Pc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T

    cos_angle = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    rot_err_deg = float(np.degrees(np.arccos(cos_angle)))

    residual = Pc - (Qc @ R.T)
    residual_rmsd = float(np.sqrt((residual**2).sum(axis=1).mean()))
    return centroid_err, rot_err_deg, residual_rmsd


def main() -> None:
    true_paths = sorted(glob.glob(os.path.join(TRUE_DIR, "*_ligands.sdf")))

    for label, pred_dir in PRED_DIRS.items():
        if not os.path.isdir(pred_dir):
            print(f"\n=== {label}: SKIPPED (no {pred_dir}/) ===")
            continue

        centroid_errs, rot_errs, residual_rmsds = [], [], []
        bond_err, bond_placement = [], []  # per cut bond
        n_ok = 0

        for true_path in true_paths:
            cid = os.path.basename(true_path).replace("_ligands.sdf", "")
            cands = glob.glob(os.path.join(pred_dir, f"{cid}__*_seed0.sdf"))
            if not cands:
                continue
            mol_pred = load_mol(cands[0])
            if mol_pred is None:
                continue
            # Reference = the crystallographic copy this pose belongs to.
            # Matters a lot here: the centroid error IS the placement metric,
            # so scoring against the wrong copy inflates it by tens of A.
            mol_true, _, _ = best_copy(
                mol_pred.GetConformer().GetPositions(), load_copies(true_path))
            if mol_true is None:
                continue
            if mol_true.GetNumBonds() != mol_pred.GetNumBonds():
                continue

            pos_true = mol_true.GetConformer().GetPositions()
            pos_pred = mol_pred.GetConformer().GetPositions()
            frag_id = find_fragments(mol_true, mol_pred)
            n_ok += 1

            frag_stats = {}
            for f in np.unique(frag_id):
                sel = frag_id == f
                if sel.sum() < 3:
                    continue
                ce, re_, rr = kabsch_errors(pos_pred[sel], pos_true[sel])
                frag_stats[f] = (ce, re_)
                centroid_errs.append(ce)
                rot_errs.append(re_)
                residual_rmsds.append(rr)

            # Cut bonds: endpoints in different fragments
            ct, cp = mol_true.GetConformer(), mol_pred.GetConformer()
            for bond in mol_true.GetBonds():
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                if frag_id[i] == frag_id[j]:
                    continue
                if frag_id[i] not in frag_stats or frag_id[j] not in frag_stats:
                    continue
                dt = ct.GetAtomPosition(i).Distance(ct.GetAtomPosition(j))
                dp = cp.GetAtomPosition(i).Distance(cp.GetAtomPosition(j))
                bond_err.append(abs(dp - dt))
                # combined placement error of the two fragments joined by this bond
                ce_i, _ = frag_stats[frag_id[i]]
                ce_j, _ = frag_stats[frag_id[j]]
                bond_placement.append(np.hypot(ce_i, ce_j))

        centroid_errs = np.array(centroid_errs)
        rot_errs = np.array(rot_errs)
        residual_rmsds = np.array(residual_rmsds)
        bond_err = np.array(bond_err)
        bond_placement = np.array(bond_placement)

        print(f"\n=== {label} ({n_ok} complexes, {len(centroid_errs)} fragments) ===")
        print(
            f"  per-fragment centroid error : mean={centroid_errs.mean():.2f} A  "
            f"median={np.median(centroid_errs):.2f} A"
        )
        print(
            f"  per-fragment rotation error : mean={np.nanmean(rot_errs):.1f} deg  "
            f"median={np.nanmedian(rot_errs):.1f} deg"
        )
        print(
            f"  residual RMSD after rigid fit: mean={np.nanmean(residual_rmsds):.4f} A  "
            f"(sanity: ~0 confirms rigid placement)"
        )
        if len(bond_err) > 2:
            r = np.corrcoef(bond_placement, bond_err)[0, 1]
            print(f"  cut bonds: {len(bond_err)}")
            print(f"  corr(combined fragment placement error, bond length error) = {r:.3f}")


if __name__ == "__main__":
    main()
