"""Reference-ligand loading that respects crystallographic copies.

WHY THIS MODULE EXISTS
A PoseBusters `<complex>_ligands.sdf` file (note the plural) holds EVERY
crystallographic copy of the ligand in the structure, not one. Measured on
this very set: 84 of 209 files hold more than one copy, up to 6, and all
copies within a file share one topology - they are the same molecule at
different sites, not different molecules.

Every analysis script here used to do

    for mol in Chem.SDMolSupplier(path):
        if mol is not None:
            return mol                      # <- silently takes copy 0

which is a CHOICE, not a reference. When the sampled pose corresponds to a
different copy, the RMSD is measured against the wrong site and comes out at
40-80 A while the pose itself is fine.

That is not hypothetical: it produced 42 phantom "catastrophic failures" in
the 2026-08-12 SigmaDock comparison, all 42 of them in multi-copy files, all
42 matching a copy other than 0. Measured against the right copy the median
error was 3.44 A - ordinary performance. Those 42 had already been written up
as a robustness advantage for SigmaFlow before the artefact was found.

THE CONVENTION USED HERE
Score against the copy the prediction is closest to, i.e. take the minimum
RMSD over copies. This is the standard convention for crystallographic
symmetry copies in docking evaluation, and it is applied identically to every
method, so comparisons stay fair.

Its bias is worth stating plainly: min-over-copies can only lower an error,
never raise it, so absolute numbers are mildly optimistic for multi-copy
complexes. Single-copy complexes - 125 of 209 here - are unaffected, and the
same benefit is granted to every run being compared.

NOT handled here: atom-order symmetry within one molecule (e.g. the two
oxygens of a carboxylate, equivalent ring flips). That is a separate
correction; all RMSDs in this project remain index-based. See the CAVEAT in
full_metrics.py.
"""

import numpy as np
from rdkit import Chem


def load_copies(path: str) -> list:
    """Every valid molecule in an SDF file, in file order.

    Returns a list of RDKit Mol objects, empty if none could be read.
    `sanitize=False` keeps RDKit from rejecting unusual crystal ligands;
    `removeHs=False` preserves the atom indexing the predictions use.
    """
    return [m for m in Chem.SDMolSupplier(path, removeHs=False, sanitize=False)
            if m is not None]


def raw_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """Index-based RMSD between two coordinate sets, both [n, 3]."""
    return float(np.sqrt(((P - Q) ** 2).sum(1).mean()))


def best_copy(P: np.ndarray, copies: list):
    """The reference copy that the prediction `P` [n, 3] is closest to.

    Returns (mol, rmsd, index). If no copy has a matching atom count,
    returns (None, inf, -1) - callers should skip such complexes rather than
    compare mismatched molecules.
    """
    best_mol, best_rmsd, best_idx = None, float("inf"), -1
    for i, mol in enumerate(copies):
        Q = mol.GetConformer().GetPositions()
        if Q.shape != P.shape:
            continue
        r = raw_rmsd(P, Q)
        if r < best_rmsd:
            best_mol, best_rmsd, best_idx = mol, r, i
    return best_mol, best_rmsd, best_idx


def load_first(path: str):
    """The old behaviour - first molecule only. Kept ONLY so a script can
    reproduce a historical number on purpose. Do not use for new analysis."""
    copies = load_copies(path)
    return copies[0] if copies else None
