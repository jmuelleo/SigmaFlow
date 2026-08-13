"""
Recreation of the bond-length "transition bond" check used throughout
STATUS.md (Test 1/2/3, 6h/12h/24h full-PoseBusters-set evaluations). The
original script lived in an ephemeral scratchpad and did not survive
between sessions - this reproduces the same methodology:

For every ligand bond, compare its length in the predicted pose against its
length in the true (crystal) pose. Intra-fragment bonds are geometrically
preserved by construction (rigid per-fragment placement), so any bond whose
length differs from the true pose by more than TOL is effectively a
"transition bond" - almost always the torsional bond at a fragment boundary
that both SigmaFlow and SigmaDock place independently.

Usage: python bond_length_check.py
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
    "SigmaFlow_rotdata_12h": "sigmaflow_rotdata_pred",
    "SigmaFlow_anchordist_12h": "sigmaflow_anchordist_pred",
}
TOL = 0.02  # Angstrom


from ligand_reference import best_copy, load_copies, load_first as load_mol


def bond_lengths(mol_pred: Chem.Mol, mol_true: Chem.Mol) -> list[tuple[float, float]]:
    """Returns (abs_diff, predicted_length) per bond, matched by atom index."""
    conf_pred = mol_pred.GetConformer()
    conf_true = mol_true.GetConformer()
    out = []
    for bond in mol_true.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        d_true = conf_true.GetAtomPosition(i).Distance(conf_true.GetAtomPosition(j))
        d_pred = conf_pred.GetAtomPosition(i).Distance(conf_pred.GetAtomPosition(j))
        out.append((abs(d_pred - d_true), d_pred))
    return out


def find_pred_file(pred_dir: str, complex_id: str) -> str | None:
    candidates = glob.glob(os.path.join(pred_dir, f"{complex_id}__*_seed0.sdf"))
    return candidates[0] if candidates else None


def main() -> None:
    true_paths = sorted(glob.glob(os.path.join(TRUE_DIR, "*_ligands.sdf")))
    print(f"Found {len(true_paths)} true-ligand files in {TRUE_DIR}/")

    for label, pred_dir in PRED_DIRS.items():
        if not os.path.isdir(pred_dir):
            print(f"\n=== {label}: SKIPPED (no folder {pred_dir}/) ===")
            continue

        all_pairs: list[tuple[float, float]] = []
        n_ok = 0
        n_parse_fail = 0
        n_bond_mismatch = 0

        for true_path in true_paths:
            complex_id = os.path.basename(true_path).replace("_ligands.sdf", "")
            pred_path = find_pred_file(pred_dir, complex_id)
            if pred_path is None:
                n_parse_fail += 1
                continue

            mol_pred = load_mol(pred_path)
            if mol_pred is None:
                n_parse_fail += 1
                continue
            # Reference = the crystallographic copy this pose belongs to.
            # 84 of 209 files hold several; see ligand_reference.py. For bond
            # LENGTHS the effect is small (copies share covalent geometry),
            # but the reference should be the same one every other metric uses.
            mol_true, _, _ = best_copy(
                mol_pred.GetConformer().GetPositions(), load_copies(true_path))
            if mol_true is None:
                n_parse_fail += 1
                continue
            if mol_true.GetNumBonds() != mol_pred.GetNumBonds():
                n_bond_mismatch += 1
                continue

            all_pairs.extend(bond_lengths(mol_pred, mol_true))
            n_ok += 1

        diffs = np.array([d for d, _ in all_pairs])
        pred_lengths = np.array([p for _, p in all_pairs])
        transition_lengths = pred_lengths[diffs > TOL]

        print(f"\n=== {label} ===")
        print(f"complexes ok: {n_ok}, parse failures: {n_parse_fail}, bond-count mismatches: {n_bond_mismatch}")
        print(f"total bonds: {len(diffs)}")
        if len(transition_lengths) > 0:
            pct = 100.0 * len(transition_lengths) / len(diffs)
            print(
                f"transition bonds (|pred-true| > {TOL} A): {len(transition_lengths)} "
                f"({pct:.1f}%) | mean={transition_lengths.mean():.2f} A | "
                f"median={np.median(transition_lengths):.2f} A | max={transition_lengths.max():.2f} A"
            )
        else:
            print("transition bonds: 0")


if __name__ == "__main__":
    main()
