"""EXP-100 Schritt 1: RDKit <-> Graph-Mapping EMPIRISCH rekonstruieren.

Kein Raten. Es wird ein echtes SigmaDataset-Item aus dem Dummy-Satz gebaut und
jeder fuer die Indexabbildung relevante Tensor inspiziert.

    PYTHONPATH=src python tests/inspect_mapping.py
"""

import sys
from pathlib import Path

import torch
from rdkit import RDLogger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RDLogger.DisableLog("rdApp.*")

from sigmadock.data import SigmaDataset      # noqa: E402
from sigmadock.datafronts import DataFront    # noqa: E402
from sigmadock.oracle import HPARAMS          # noqa: E402

DUMMY = ROOT.parents[1] / "SigmaFlow_Minimal" / "notebooks" / "dummy_data"
assert DUMMY.is_dir(), f"Dummy-Daten nicht gefunden: {DUMMY}"

NAMES = {v: k for k, v in HPARAMS.node_entity.entity_indices.items()}
LIG_ATOM = HPARAMS.get_node_idx("ligand_atom")
LIG_ANCHOR = HPARAMS.get_node_idx("ligand_anchor")
LIG_DUMMY = HPARAMS.get_node_idx("ligand_dummy")
LIG_VIRT = HPARAMS.get_node_idx("ligand_virtual")
LIGLIKE = torch.tensor([LIG_ATOM, LIG_ANCHOR, LIG_DUMMY, LIG_VIRT])


def build_dataset(seed=0, sample_conformer=False):
    df = DataFront(str(DUMMY), pdb_regex=r".*_protein\.pdb$", sdf_regex=r".*_ligand\.sdf$")
    return SigmaDataset(
        datafront=df,
        pocket_com_noise=0.0, pocket_distance_cutoff=8.0, pocket_distance_noise=0.0,
        prot_coordinate_distance_noise=0.0, use_esm_embeddings=False,
        ignore_triangulation=False, lig_coordinate_distance_noise=0.0,
        alignment_tries=0,                    # keine Torsions-Augmentierung -> deterministisch
        fragmentation_strategy="canonical",
        pb_check=False, get_mol_info=True, seed=seed,
        random_rotation=False, sample_conformer=sample_conformer,
        skip_bounds_check=True, force_retry=True,
    )


def report(d, idx):
    ne = d.node_entity
    fim = d.frag_idx_map
    N = ne.shape[0]
    liglike = torch.isin(ne, LIGLIKE)
    prot = torch.where(~liglike)[0]
    lig = torch.where(liglike)[0]

    print("=" * 92)
    print(f"ITEM {idx}   Knoten gesamt = {N}")
    print("=" * 92)
    print("node_entity: " + str({NAMES[k]: int((ne == k).sum()) for k in sorted(set(ne.tolist()))}))
    print(f"Reihenfolge: {'PROTEIN zuerst' if len(prot) and prot.min() < lig.min() else 'LIGAND zuerst'}"
          f"   Protein {int(prot.min())}..{int(prot.max())}   Ligand {int(lig.min())}..{int(lig.max())}")
    print(f"Ligandenblock zusammenhaengend: {bool((lig.diff() == 1).all())}")
    print()
    for e in (LIG_ATOM, LIG_ANCHOR, LIG_DUMMY, LIG_VIRT):
        s = torch.where(ne == e)[0]
        if not len(s):
            continue
        print(f"  {NAMES[e]:<15} n={len(s):>4}  Index {int(s.min())}..{int(s.max())}"
              f"  zusammenhaengend={bool((s.diff() == 1).all()) if len(s) > 1 else True}"
              f"  mask={sorted(set(d.mask[s].tolist()))}"
              f"  frag_idx_map={sorted(set(fim[s].tolist()))[:5]}")
    print()
    print(f"frag_idx_map != -1 : {int((fim != -1).sum())}   bei Protein: "
          f"{sorted(set(fim[~liglike].tolist()))}")
    print(f"frag_sizes={d.frag_sizes.tolist()}  dummy_frag_sizes={d.dummy_frag_sizes.tolist()}")
    print(f"frag_atom_idx[:15]={d.frag_atom_idx[:15].tolist()}")
    print(f"frag_counter[:15] ={d.frag_counter[:15].tolist()}")

    mi = getattr(d, "mol_info", None)
    print(f"\nmol_info: {type(mi).__name__}", end="")
    if isinstance(mi, dict):
        print(f"  Schluessel={list(mi.keys())}")
        for k, v in mi.items():
            print(f"   {k}: {type(v).__name__}"
                  + (f", {v.GetNumAtoms()} Atome" if hasattr(v, "GetNumAtoms") else ""))
    elif isinstance(mi, (list, tuple)) and mi:
        print(f"  len={len(mi)}, [0]={type(mi[0]).__name__}")
        if isinstance(mi[0], dict):
            print(f"   Schluessel={list(mi[0].keys())}")
    else:
        print()
    print()


if __name__ == "__main__":
    ds = build_dataset()
    print(f"Dataset: {len(ds)} Komplexe\n")
    for i in range(min(3, len(ds))):
        report(ds[i], i)
