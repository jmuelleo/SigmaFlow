"""
Same per-fragment placement analysis as placement_vs_bond_error.py, but for
the 10 DUMMY complexes (overfitting regime) instead of the 209 held-out
PoseBusters ones.

Purpose: separate "undertrained" from "broken". Memorising 10 complexes is a
far easier task than generalising to 209 unseen ones. If the model cannot
drive the per-fragment rotation error well below the 126.5 deg random
baseline even when overfitting 10 complexes, the problem is not compute.

Usage: python placement_dummy.py
Run from SigmaFlow_Variants/posebusters_full_comparison/.
"""

import glob
import os

import numpy as np
from rdkit import Chem

DUMMY_DIR = "../../SigmaFlow_Development/notebooks/dummy_data"
DEV = "../../SigmaFlow_Development"
PRED_DIRS = {
    "SigmaFlow 3h prodhparams": f"{DEV}/sampling_output_prodhparams",
    "SigmaFlow 3h after_edm_fix": f"{DEV}/sampling_output_after_edm_fix",
    "SigmaDock 3h prodhparams": f"{DEV}/sampling_output_sigmadock_prodhparams",
}
TOL = 0.02

RANDOM_BASELINE_DEG = 126.5


def load_mol(path: str):
    for mol in Chem.SDMolSupplier(path, removeHs=False, sanitize=False):
        if mol is not None:
            return mol
    return None


def find_fragments(mol_true, mol_pred) -> np.ndarray:
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


def kabsch_errors(P, Q):
    cP, cQ = P.mean(axis=0), Q.mean(axis=0)
    centroid_err = float(np.linalg.norm(cP - cQ))
    Pc, Qc = P - cP, Q - cQ
    if len(P) < 3:
        return centroid_err, float("nan"), float("nan")
    H = Qc.T @ Pc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    ang = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))
    residual = Pc - (Qc @ R.T)
    return centroid_err, ang, float(np.sqrt((residual**2).sum(axis=1).mean()))


def main() -> None:
    complexes = sorted(
        d for d in os.listdir(DUMMY_DIR) if os.path.isdir(os.path.join(DUMMY_DIR, d))
    )
    print(f"Dummy complexes found: {len(complexes)}")
    print(f"Random-rotation baseline: {RANDOM_BASELINE_DEG} deg\n")

    for label, pred_dir in PRED_DIRS.items():
        if not os.path.isdir(pred_dir):
            print(f"=== {label}: SKIPPED (no {pred_dir}) ===\n")
            continue

        cents, rots, resid, rmsds = [], [], [], []
        n_ok = 0
        for cid in complexes:
            # NOTE: dummy_train.yaml uses sdf_regex ".*_ligand\.sdf$" - SINGULAR,
            # anchored, so the _ligands.sdf (plural) file is NOT what the model
            # was trained/sampled against. They genuinely differ for some
            # complexes (e.g. 1HWI_115), so picking the wrong one silently
            # inflates every error metric. (PoseBusters is the opposite case:
            # its regex is ".*ligands.*\.sdf$" - plural.)
            tp = os.path.join(DUMMY_DIR, cid, f"{cid}_ligand.sdf")
            cands = glob.glob(os.path.join(pred_dir, f"{cid}__*_seed0.sdf"))
            if not cands or not os.path.exists(tp):
                continue
            mt, mp = load_mol(tp), load_mol(cands[0])
            if mt is None or mp is None or mt.GetNumBonds() != mp.GetNumBonds():
                continue

            Q = mt.GetConformer().GetPositions()
            P = mp.GetConformer().GetPositions()
            rmsds.append(np.sqrt(((P - Q) ** 2).sum(1).mean()))
            frag_id = find_fragments(mt, mp)
            n_ok += 1
            for f in np.unique(frag_id):
                sel = frag_id == f
                if sel.sum() < 3:
                    continue
                ce, ang, rr = kabsch_errors(P[sel], Q[sel])
                cents.append(ce)
                rots.append(ang)
                resid.append(rr)

        cents, rots, resid, rmsds = map(np.array, (cents, rots, resid, rmsds))
        print(f"=== {label} ({n_ok} complexes, {len(cents)} fragments) ===")
        print(f"  whole-molecule RMSD          : mean={rmsds.mean():.2f} A  median={np.median(rmsds):.2f} A")
        print(f"  per-fragment centroid error  : mean={cents.mean():.2f} A  median={np.median(cents):.2f} A")
        print(
            f"  per-fragment rotation error  : mean={np.nanmean(rots):.1f} deg  "
            f"median={np.nanmedian(rots):.1f} deg   "
            f"[random = {RANDOM_BASELINE_DEG}]"
        )
        print(f"  residual RMSD after rigid fit: mean={np.nanmean(resid):.4f} A\n")


if __name__ == "__main__":
    main()
