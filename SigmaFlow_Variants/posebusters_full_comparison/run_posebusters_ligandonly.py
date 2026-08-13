"""
Recreation of the "20 ligand-intrinsic PoseBusters checks" evaluation from
STATUS.md ("Erweiterung (2026-08-08): volle PoseBusters-Chemie-Checks").
The original script lived in an ephemeral scratchpad and did not survive
between sessions - this reproduces it against a new prediction folder,
using the same redock_noprotein.yml config (redock.yml minus the 8
mol_cond/protein-dependent modules, since the 209 receptor PDBs aren't
available locally).

Usage: python run_posebusters_ligandonly.py <pred_dir> <output_csv>
Run from SigmaFlow_Variants/posebusters_full_comparison/.
"""

import glob
import os
import sys
import tempfile

import pandas as pd
import yaml
from posebusters import PoseBusters
from rdkit import Chem

from ligand_reference import best_copy, load_copies

TRUE_DIR = "true_ligands"
CONFIG = "redock_noprotein.yml"


def find_pred_file(pred_dir: str, complex_id: str) -> str | None:
    candidates = glob.glob(os.path.join(pred_dir, f"{complex_id}__*_seed0.sdf"))
    return candidates[0] if candidates else None


def reference_for(true_path: str, pred_path: str, tmp_dir: str) -> tuple[str, int]:
    """Path to a SINGLE-copy reference SDF for this prediction, plus the copy
    index chosen.

    PoseBusters takes the FIRST molecule out of `mol_true`. On the 84 of 209
    files that hold several crystallographic copies that is an arbitrary pick,
    and it silently scores a correct pose against the wrong site - the RMSD
    check then fails for a pose that is fine. We therefore write out the copy
    the prediction actually corresponds to and hand PoseBusters that.

    Falls back to the original path (copy 0) if the prediction cannot be read
    or no copy has a matching atom count.
    """
    copies = load_copies(true_path)
    pred = load_copies(pred_path)
    if not copies or not pred:
        return true_path, 0
    mol_true, _, idx = best_copy(pred[0].GetConformer().GetPositions(), copies)
    if mol_true is None:
        return true_path, 0
    if idx == 0 and len(copies) == 1:
        return true_path, 0          # nothing to disambiguate, avoid a rewrite
    out = os.path.join(tmp_dir, os.path.basename(true_path))
    with Chem.SDWriter(out) as w:
        w.write(mol_true)
    return out, idx


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python run_posebusters_ligandonly.py <pred_dir> <output_csv>")
        sys.exit(1)
    pred_dir, output_csv = sys.argv[1], sys.argv[2]

    true_paths = sorted(glob.glob(os.path.join(TRUE_DIR, "*_ligands.sdf")))
    print(f"Found {len(true_paths)} true-ligand files. Predictions from: {pred_dir}/")

    with open(CONFIG) as f:
        config_dict = yaml.safe_load(f)
    buster = PoseBusters(config=config_dict)

    rows = []
    n_missing = 0
    n_recopied = 0
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, true_path in enumerate(true_paths):
            complex_id = os.path.basename(true_path).replace("_ligands.sdf", "")
            pred_path = find_pred_file(pred_dir, complex_id)
            if pred_path is None:
                n_missing += 1
                print(f"  [WARN] no prediction found for {complex_id}")
                continue

            ref_path, copy_idx = reference_for(true_path, pred_path, tmp_dir)
            if copy_idx != 0:
                n_recopied += 1
            df = buster.bust(mol_pred=pred_path, mol_true=ref_path, mol_cond=None)
            rows.append(df)

            if (i + 1) % 25 == 0:
                print(f"  ...{i + 1}/{len(true_paths)}")

        result = pd.concat(rows)
    result.to_csv(output_csv)
    print(f"\nWrote {len(result)} rows ({n_missing} complexes missing predictions) to {output_csv}")
    print(f"Referenz auf eine andere Kopie als 0 umgestellt bei: {n_recopied} Komplexen")

    bool_cols = result.select_dtypes(include="bool").columns
    print("\nPass rates:")
    print(result[bool_cols].mean().round(3).to_string())


if __name__ == "__main__":
    main()
