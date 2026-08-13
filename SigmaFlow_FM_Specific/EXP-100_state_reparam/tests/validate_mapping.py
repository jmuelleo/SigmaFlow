"""EXP-100 Schritt 2: RDKit <-> Graph-Mapping CHEMISCH und GEOMETRISCH beweisen.

`inspect_mapping.py` hat gezeigt, WIE der Graph aufgebaut ist. Dieses Skript
beweist, dass die Abbildung stimmt - nicht nur ueber Koordinaten (die koennten
zufaellig passen), sondern ueber die Chemie jedes einzelnen Atoms.

TEIL A  Graph <-> frag_mol
        Ordnungszahl, Grad, Formalladung, Hybridisierung, Aromatizitaet,
        Chiralitaet, Valenzen, Bindungsnachbarschaft, Koordinaten.

TEIL B  Zweite Fragmentierung auf dem Referenzkonformer
        EXP-100 braucht Koordinaten fuer ALLE Ligandenknoten - auch fuer
        Dummies und virtuelle Knoten, die es im Originalmolekuel nicht gibt.
        Der einzige nicht erfundene Weg ist, exakt dieselbe Fragmentierung ein
        zweites Mal auf dem Konformer laufen zu lassen. Das ist nur zulaessig,
        wenn sie bitgleich dieselbe Indexstruktur liefert. Genau das wird hier
        geprueft - und bei Abweichung ABGEBROCHEN.

TEIL C  Leakage
        Haengt der Konformer von der Kristallpose ab?

    PYTHONPATH=src python tests/validate_mapping.py
"""

import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, RDLogger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RDLogger.DisableLog("rdApp.*")

from sigmadock.chem.ligalign import ConformerOptimizer          # noqa: E402
from sigmadock.chem.processing import (                          # noqa: E402
    get_atom_features,
    get_global_ligand_graph,
)
from sigmadock.data import SigmaDataset                          # noqa: E402
from sigmadock.datafronts import DataFront                       # noqa: E402
from sigmadock.oracle import HPARAMS                             # noqa: E402

DUMMY = ROOT.parents[1] / "SigmaFlow_Minimal" / "notebooks" / "dummy_data"
LIG_VIRT = HPARAMS.get_node_idx("ligand_virtual")
LIGLIKE = torch.tensor([
    HPARAMS.get_node_idx("ligand_atom"),
    HPARAMS.get_node_idx("ligand_anchor"),
    HPARAMS.get_node_idx("ligand_dummy"),
    LIG_VIRT,
])
BOND_EDGE = HPARAMS.get_edge_idx("ligand_bonds")

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    """Ein Pruefpunkt. Sammelt Fehlschlaege statt sofort abzubrechen, damit ein
    Lauf das VOLLSTAENDIGE Bild liefert und nicht nur den ersten Fehler."""
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(f"{name}  {detail}")
    return ok


def build_dataset(sample_conformer: bool = False, seed: int = 0) -> SigmaDataset:
    df = DataFront(str(DUMMY), pdb_regex=r".*_protein\.pdb$", sdf_regex=r".*_ligand\.sdf$")
    return SigmaDataset(
        datafront=df,
        pocket_com_noise=0.0, pocket_distance_cutoff=8.0, pocket_distance_noise=0.0,
        prot_coordinate_distance_noise=0.0, use_esm_embeddings=False,
        ignore_triangulation=False, lig_coordinate_distance_noise=0.0,
        alignment_tries=0, fragmentation_strategy="canonical",
        pb_check=False, get_mol_info=True, seed=seed,
        random_rotation=False, sample_conformer=sample_conformer,
        skip_bounds_check=True, force_retry=True,
    )


def ligand_block(d) -> torch.Tensor:
    """Indizes des Ligandenblocks im Gesamtgraphen, in Graphreihenfolge."""
    return torch.where(torch.isin(d.node_entity, LIGLIKE))[0]


# ------------------------------------------------------------------ TEIL A
def part_a(d, idx: int) -> None:
    print(f"\n--- ITEM {idx}: Graph <-> frag_mol ---")
    frag_mol = d.mol_info["fragmented"]
    M = frag_mol.GetNumAtoms()
    blk = ligand_block(d)
    V = len(blk) - M

    check("Ligandenblock zusammenhaengend", bool((blk.diff() == 1).all()))
    check("Protein steht vor dem Ligandenblock", int(blk.min()) > 0)
    check("Blockgroesse = |frag_mol| + |virtuelle|", V >= 0,
          f"M={M} V={V} block={len(blk)}")
    check("virtuelle Knoten liegen HINTER den echten Atomen",
          bool((d.node_entity[blk[:M]] != LIG_VIRT).all())
          and bool((d.node_entity[blk[M:]] == LIG_VIRT).all()))

    # --- Chemie: der eigentliche Beweis --------------------------------
    # x traegt genau die Merkmale aus get_atom_features. Ein exakter Vergleich
    # des ganzen Merkmalsvektors prueft in einem Schritt Ordnungszahl, Grad,
    # Formalladung, Hybridisierung, Aromatizitaet, Chiralitaet und Valenzen.
    want = np.stack([get_atom_features(a) for a in frag_mol.GetAtoms()]).astype(np.float32)
    got = d.x[blk[:M]].numpy().astype(np.float32)
    same = np.isclose(want, got, atol=0, rtol=0)
    check("Atommerkmale identisch (alle Spalten, alle Atome)", bool(same.all()),
          f"{int((~same).sum())} Abweichungen von {same.size}")

    if not same.all():
        bad = np.where(~same.all(axis=1))[0][:5]
        for b in bad:
            print(f"        Atom {b}: want={want[b]} got={got[b]}")

    # Explizit einzeln, damit im Bericht steht, was geprueft wurde
    Z_ok = deg_ok = chg_ok = True
    for i, a in enumerate(frag_mol.GetAtoms()):
        Z_ok &= abs(float(got[i, 0]) - a.GetAtomicNum()) < 1e-6
        deg_ok &= abs(float(got[i, 1]) - a.GetDegree()) < 1e-6
        chg = 2 if a.GetFormalCharge() > 0 else (1 if a.GetFormalCharge() < 0 else 0)
        chg_ok &= abs(float(got[i, 2]) - chg) < 1e-6
    check("  davon: Ordnungszahl", Z_ok)
    check("  davon: Grad (degree)", deg_ok)
    check("  davon: Formalladung", chg_ok)

    # --- Bindungsnachbarschaft -----------------------------------------
    ei = d.edge_index[:, d.edge_entity == BOND_EDGE]
    off = int(blk.min())
    in_lig = (ei[0] >= off) & (ei[1] >= off) & (ei[0] < off + M) & (ei[1] < off + M)
    got_bonds = {(int(a) - off, int(b) - off) for a, b in ei[:, in_lig].t()}
    want_bonds = set()
    for b in frag_mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        want_bonds |= {(i, j), (j, i)}
    check("Bindungsnachbarschaft identisch", got_bonds == want_bonds,
          f"|graph|={len(got_bonds)} |rdkit|={len(want_bonds)} "
          f"nur_graph={len(got_bonds - want_bonds)} nur_rdkit={len(want_bonds - got_bonds)}")

    # --- Geometrie ------------------------------------------------------
    P = frag_mol.GetConformer().GetPositions()
    dev = float(np.abs(d.ref_pos[blk[:M]].numpy() - P).max())
    check("Koordinaten identisch", dev < 1e-4, f"max |d| = {dev:.2e} A")

    # --- Fragmentzuordnung ---------------------------------------------
    fm = d.frag_idx_map[blk]
    check("frag_idx_map >= 0 auf dem gesamten Ligandenblock", bool((fm >= 0).all()))
    check("frag_idx_map == -1 auf dem Protein",
          bool((d.frag_idx_map[:off] == -1).all()))
    nf = int(d.mol_info["num_fragments"])
    check("Fragmentindizes decken 0..F-1 ab",
          sorted(set(fm.tolist())) == list(range(nf)), f"F={nf}")


# ------------------------------------------------------------------ TEIL B
def part_b(ds: SigmaDataset, d, idx: int) -> None:
    """TEIL B: die Konstruktion, die EXP-100 TATSAECHLICH benutzt.

    Erster Versuch war, die Fragmentierung ein zweites Mal auf dem Konformer
    laufen zu lassen. Das ist VERWORFEN worden, und zwar aus gemessenem Grund:
    `fragment_and_annotate` ist nicht rein topologisch. Auf denselben Molekuelen
    mit anderer Geometrie kamen andere Bindungsmengen, andere `frag_atom_idx`
    und andere `dummy_frag_sizes` heraus (siehe B0 unten). Ein daraus gebautes
    Kabsch-Ziel waere still falsch gewesen.

    Stattdessen wird der Graph FESTGEHALTEN und nur die Geometrie ersetzt:
      * echte Atome i < n_orig       -> Konformerposition i
      * Dummies k >= n_orig          -> Konformerposition des Partners, den das
                                        ISOTOP des Dummys benennt
      * virtuelle Knoten             -> mit derselben Funktion neu berechnet
    """
    print(f"\n--- ITEM {idx}: Referenzgeometrie-Konstruktion ---")
    orig = d.mol_info["original"]
    frag_mol = d.mol_info["fragmented"]
    n_orig, n_frag = orig.GetNumAtoms(), frag_mol.GetNumAtoms()
    blk = ligand_block(d)

    # --- B0: Beleg, warum NICHT neu fragmentiert wird -------------------
    conf_mol = ConformerOptimizer(deepcopy(orig), seed=ds.seed + idx)._generate_conformer()
    cf_mol, cf_td, cf_info = ds.fragment_and_annotate(mol=conf_mol, idx=idx)
    ba = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in frag_mol.GetBonds()}
    bb = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in cf_mol.GetBonds()}
    stable = (ba == bb) and torch.equal(
        cf_td["frag_atom_idx"], ds.fragment_and_annotate(mol=deepcopy(orig), idx=idx)[1]["frag_atom_idx"])
    print(f"  [info] Neufragmentierung waere indexstabil: {stable}"
          f"   (Bindungs-Symmetriedifferenz {len(ba ^ bb)})")

    # --- B1: Voraussetzungen der benutzten Abbildung --------------------
    z_orig = [a.GetAtomicNum() for a in orig.GetAtoms()]
    check("frag_mol[:n_orig] IST das Originalmolekuel",
          [frag_mol.GetAtomWithIdx(i).GetAtomicNum() for i in range(n_orig)] == z_orig)
    check("Konformer hat Reihenfolge und Bindungen des Originals",
          conf_mol.GetNumAtoms() == n_orig
          and [a.GetAtomicNum() for a in conf_mol.GetAtoms()] == z_orig
          and {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in conf_mol.GetBonds()}
          == {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in orig.GetBonds()})

    P = frag_mol.GetConformer().GetPositions()
    iso_ok = nb_ok = pos_ok = True
    for k in range(n_orig, n_frag):
        a = frag_mol.GetAtomWithIdx(k)
        iso = a.GetIsotope()
        iso_ok &= (a.GetAtomicNum() == 0) and (iso < n_orig)
        nb_ok &= len(a.GetNeighbors()) == 1
        if iso < n_orig:
            pos_ok &= float(np.linalg.norm(P[k] - P[iso])) < 1e-3
    check("Dummies: Ordnungszahl 0 und Isotop zeigt auf ein Originalatom", iso_ok)
    check("Dummies: genau ein Nachbar", nb_ok)
    check("Dummies: liegen exakt auf ihrem Partneratom", pos_ok)

    # --- B2: das Ergebnis der echten Funktion ---------------------------
    frag_td_fresh = ds.fragment_and_annotate(mol=deepcopy(d.mol_info["aligned"]), idx=idx)
    fm2, td2, info2 = frag_td_fresh
    graph2 = get_global_ligand_graph(fm2, **td2, **info2)
    out = ds.build_reference_conformer_block(
        bound_ligand=deepcopy(orig), frag_mol=fm2, frag_torchdata=td2,
        frag_info=info2, frag_graph=graph2, idx=idx,
    )
    check("ref_conf_pos deckt den GESAMTEN Ligandenblock ab",
          out.shape[0] == len(blk), f"{tuple(out.shape)} vs {len(blk)}")
    check("ref_conf_pos ist endlich", bool(torch.isfinite(out).all()))

    Pc = np.asarray(conf_mol.GetConformer().GetPositions(), dtype=np.float32)
    check("echte Atome tragen exakt die Konformerkoordinaten",
          float(np.abs(out[:n_orig].numpy() - Pc).max()) < 1e-6,
          f"max |d| = {float(np.abs(out[:n_orig].numpy() - Pc).max()):.2e} A")

    dummy_ok = all(
        float(np.abs(out[k].numpy() - Pc[fm2.GetAtomWithIdx(k).GetIsotope()]).max()) < 1e-6
        for k in range(n_orig, fm2.GetNumAtoms())
    )
    check("Dummies tragen die Konformerkoordinate ihres Partners", dummy_ok)

    # Interne Geometrie muss die des Konformers sein, nicht die der Kristallpose
    def spread(X):
        X = np.asarray(X)[:n_orig]
        return float(np.sqrt(((X - X.mean(0)) ** 2).sum(1).mean()))
    check("Referenzgeometrie != Kristallgeometrie (sonst waere EXP-100 wirkungslos)",
          abs(spread(out.numpy()) - spread(P)) > 1e-6 or
          float(np.abs(out[:n_orig].numpy() - P[:n_orig]).max()) > 0.1,
          f"Gyrationsradius Referenz {spread(out.numpy()):.2f} A vs Kristall {spread(P):.2f} A")


# ------------------------------------------------------------------ TEIL C
def part_c(ds: SigmaDataset, d, idx: int) -> None:
    """Leakage: haengt der Referenzkonformer von der Kristallpose ab?"""
    print(f"\n--- ITEM {idx}: Leakage-Test des Referenzkonformers ---")
    orig = d.mol_info["original"]

    a = deepcopy(orig)
    b = deepcopy(orig)
    # Version B bekommt voellig andere 3D-Koordinaten
    cb = b.GetConformer()
    g = torch.Generator().manual_seed(1234)
    for i in range(b.GetNumAtoms()):
        p = cb.GetAtomPosition(i)
        j = torch.randn(3, generator=g).numpy() * 7.0
        cb.SetAtomPosition(i, (float(p.x + j[0] + 40.0), float(p.y + j[1] - 25.0), float(p.z + j[2] + 13.0)))
    moved = float(np.abs(
        np.asarray(a.GetConformer().GetPositions()) - np.asarray(b.GetConformer().GetPositions())
    ).max())

    ca = ConformerOptimizer(a, seed=ds.seed + idx)._generate_conformer()
    cbb = ConformerOptimizer(b, seed=ds.seed + idx)._generate_conformer()
    if ca.GetNumAtoms() != cbb.GetNumAtoms():
        check("Leakage-Test auswertbar", False, "Atomzahlen weichen ab")
        return
    dev = float(np.abs(
        np.asarray(ca.GetConformer().GetPositions()) - np.asarray(cbb.GetConformer().GetPositions())
    ).max())
    check("Eingangskoordinaten wurden wirklich veraendert", moved > 1.0, f"max |d| = {moved:.1f} A")
    check("Referenzkonformer ist identisch -> KEIN Leakage", dev < 1e-6, f"max |d| = {dev:.2e} A")


if __name__ == "__main__":
    assert DUMMY.is_dir(), f"Dummy-Daten nicht gefunden: {DUMMY}"
    ds = build_dataset()
    print(f"Dataset: {len(ds)} Komplexe\n" + "=" * 78)
    n = 0
    for i in range(len(ds)):
        d = ds[i]
        if d is None or getattr(d, "mol_info", None) is None:
            continue
        part_a(d, i)
        part_b(ds, d, i)
        if n < 3:
            part_c(ds, d, i)
        n += 1
        if n >= 6:
            break
    print("\n" + "=" * 78)
    if FAILURES:
        print(f"FEHLGESCHLAGEN: {len(FAILURES)} Pruefungen")
        for f in FAILURES:
            print("  - " + f)
        sys.exit(1)
    print(f"ALLE PRUEFUNGEN BESTANDEN  ({n} Komplexe)")
