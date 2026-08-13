"""
Full side-by-side metric comparison on the 209-complex PoseBusters set.

Covers everything we track, in one place and with one methodology, so that
numbers from different sessions stop being mixed:

  A. Whole-molecule RMSD vs. the crystal pose (raw, and after optimal rigid
     alignment) + the standard docking success rate RMSD < 2 A.
  B. Per-fragment placement: centroid error and absolute rotation error.
  C. Transition ("cut") bonds: how far the anchor-anchor distance is off.
  D. Locality: relative rotation error, bonded vs. non-bonded fragment pairs
     (the criterion the frame fix was judged on; see fragment_locality.py).

WHY TWO EVALUATION SETS
-----------------------
The fragmentation is not stored with the predictions - it is RECOVERED from
them (bonds whose length is preserved = intra-fragment, see find_fragments).
That recovery can differ between runs, and STATUS.md records that it does for
~39% of complexes. Any metric defined PER FRAGMENT or PER CUT BOND is
therefore only comparable on complexes where all compared runs recovered the
SAME partition.

  - RMSD (A) does not depend on fragmentation at all  -> reported on ALL
    complexes that parse.
  - B, C, D depend on it                              -> reported on the
    MATCHED SUBSET only.

This is the fix for the contamination warning in STATUS.md; older
bond-length numbers were computed without this restriction and must not be
compared against the ones printed here.

CAVEAT: RMSD here is index-based, not symmetry-corrected. For symmetric
groups (phenyl flips etc.) that overestimates the error. It is applied
identically to every run, so comparisons between runs stay valid; absolute
values are not directly comparable to published symmetry-corrected numbers.

Usage: python full_metrics.py
Run from SigmaFlow_Variants/posebusters_full_comparison/.
"""

import glob
import os
import sys

import numpy as np
from rdkit import Chem

TRUE_DIR = "true_ligands"
# Keep each group SMALL. The matched subset (see module docstring) requires
# EVERY run in the group to recover the same fragmentation for a complex, so
# each added run shrinks it: 2 runs -> ~120, 3 runs -> 94, 4 runs -> 57.
# Compare ONE question at a time; that is why these are groups and not one
# big list.
GROUPS = {
    # The head-to-head at equal compute. Largest matched subset -> the
    # numbers to quote for "SigmaFlow vs SigmaDock".
    "12h": {
        "SigmaFlow 12h FRAME FIX (8541310)": "sigmaflow_12h_pred",
        "SigmaDock 12h (8541439)": "sigmadock_12h_pred",
    },
    "6h": {
        "SigmaFlow 6h FRAME FIX (8530243)": "sigmaflow_framefix_pred",
        "SigmaDock 6h (8512922)": "sigmadock_lrfix_pred",
    },
    # Does more compute help? One method at a time, so the subset stays big.
    "scaling_sigmaflow": {
        "SigmaFlow 6h FRAME FIX (8530243)": "sigmaflow_framefix_pred",
        "SigmaFlow 12h FRAME FIX (8541310)": "sigmaflow_12h_pred",
    },
    "scaling_sigmadock": {
        "SigmaDock 6h (8512922)": "sigmadock_lrfix_pred",
        "SigmaDock 12h (8541439)": "sigmadock_12h_pred",
    },
    # What the frame fix did, at 6h.
    "framefix": {
        "SigmaFlow 6h no fix (8512798)": "sigmaflow_lrfix_pred",
        "SigmaFlow 6h FRAME FIX (8530243)": "sigmaflow_framefix_pred",
    },
}
RUNS = GROUPS[sys.argv[1] if len(sys.argv) > 1 else "12h"]
TOL = 0.02  # Angstrom: bond counts as "preserved" below this
MIN_FRAG_ATOMS = 3
RANDOM_BASELINE_DEG = 126.5


from ligand_reference import best_copy, load_copies, load_first as load_mol


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


def partition_key(frag_id: np.ndarray):
    """Canonical, label-independent form of a fragmentation, so two runs can
    be compared even if they numbered the fragments differently."""
    groups = {}
    for atom, f in enumerate(frag_id):
        groups.setdefault(int(f), []).append(atom)
    return tuple(sorted(tuple(sorted(g)) for g in groups.values()))


def kabsch(P: np.ndarray, Q: np.ndarray):
    """P=predicted, Q=true, [n,3]. Returns (R, centroid_err, residual_rmsd)."""
    cP, cQ = P.mean(axis=0), Q.mean(axis=0)
    centroid_err = float(np.linalg.norm(cP - cQ))
    Pc, Qc = P - cP, Q - cQ
    H = Qc.T @ Pc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    resid = Pc - (Qc @ R.T)
    return R, centroid_err, float(np.sqrt((resid**2).sum(axis=1).mean()))


def angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


def stat(a) -> str:
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return "  (keine Daten)"
    return f"mean={np.nanmean(a):7.2f}  median={np.nanmedian(a):7.2f}  (n={a.size})"


def main() -> None:
    true_paths = sorted(glob.glob(os.path.join(TRUE_DIR, "*_ligands.sdf")))
    runs = {k: v for k, v in RUNS.items() if os.path.isdir(v)}
    for k in RUNS:
        if k not in runs:
            print(f"SKIPPED (Ordner fehlt): {k} -> {RUNS[k]}")

    # ---- Pass 1: load everything, decide the matched subset -----------------
    per_complex = {}  # cid -> {run: (pos_pred, pos_true, mol_true, frag_id)}
    for tp in true_paths:
        cid = os.path.basename(tp).replace("_ligands.sdf", "")
        true_copies = load_copies(tp)
        if not true_copies:
            continue
        entry = {}
        for label, d in runs.items():
            cands = glob.glob(os.path.join(d, f"{cid}__*_seed0.sdf"))
            if not cands:
                continue
            mp = load_mol(cands[0])
            if mp is None:
                continue
            # Each run is scored against the crystallographic copy ITS pose is
            # closest to. 84 of 209 files hold several copies of the same
            # ligand; copy 0 is an arbitrary pick, not the reference. Runs may
            # legitimately land on different copies - the topology is identical
            # either way, so the fragmentation comparison below stays valid.
            mol_true, _, _ = best_copy(mp.GetConformer().GetPositions(), true_copies)
            if mol_true is None or mp.GetNumBonds() != mol_true.GetNumBonds():
                continue
            entry[label] = (
                mp.GetConformer().GetPositions(),
                mol_true.GetConformer().GetPositions(),
                mol_true,
                find_fragments(mol_true, mp),
            )
        if len(entry) == len(runs):
            per_complex[cid] = entry

    matched = [
        cid for cid, e in per_complex.items()
        if len({partition_key(e[l][3]) for l in runs}) == 1
    ]

    print(f"\nKomplexe mit Vorhersage in ALLEN {len(runs)} Läufen: {len(per_complex)} / {len(true_paths)}")
    print(f"davon mit IDENTISCH rekonstruierter Fragmentierung: {len(matched)} "
          f"({100*len(matched)/max(len(per_complex),1):.0f}%)")
    print("-> A (RMSD) auf allen, B/C/D nur auf dieser Teilmenge.\n")

    # ---- A: RMSD, all complexes -------------------------------------------
    print("=" * 78)
    print("A. GESAMT-RMSD gegen die Kristallpose (alle Komplexe, index-basiert)")
    print("=" * 78)
    for label in runs:
        raw, aligned = [], []
        for cid, e in per_complex.items():
            P, Q = e[label][0], e[label][1]
            raw.append(np.sqrt(((P - Q) ** 2).sum(1).mean()))
            R, _, _ = kabsch(P, Q)
            Pa = (Q - Q.mean(0)) @ R.T + P.mean(0)
            aligned.append(np.sqrt(((P - Pa) ** 2).sum(1).mean()))
        raw, aligned = np.array(raw), np.array(aligned)
        print(f"\n{label}")
        print(f"  RMSD roh              : {stat(raw)} A")
        print(f"  RMSD nach Ausrichtung : {stat(aligned)} A   <- innere Geometrie")
        print(f"  Anteil RMSD < 2 A     : {100*(raw<2).mean():.1f} %")
        print(f"  Anteil RMSD < 5 A     : {100*(raw<5).mean():.1f} %")
        print(f"  Ausreisser  >10 / >20 / >50 A : "
              f"{int((raw>10).sum())} / {int((raw>20).sum())} / {int((raw>50).sum())}"
              f"   max={raw.max():.1f} A  p90={np.percentile(raw,90):.1f} A")

    # ---- B/C/D on the matched subset ---------------------------------------
    print("\n" + "=" * 78)
    print(f"B-D. FRAGMENT-METRIKEN (nur die {len(matched)} Komplexe mit gleicher Fragmentierung)")
    print("=" * 78)
    for label in runs:
        cents, rots, resids = [], [], []
        bond_err, bond_pred, bond_true = [], [], []
        bonded, nonbonded, gaps = [], [], []
        nfrags = []

        for cid in matched:
            P, Q, mol_true, frag_id = per_complex[cid][label]
            nfrags.append(len(np.unique(frag_id)))

            Rs = {}
            for f in np.unique(frag_id):
                sel = frag_id == f
                if sel.sum() < MIN_FRAG_ATOMS:
                    continue
                R, ce, rr = kabsch(P[sel], Q[sel])
                Rs[int(f)] = R
                cents.append(ce)
                rots.append(angle_deg(R))
                resids.append(rr)

            # C: cut bonds
            bonded_pairs = set()
            for b in mol_true.GetBonds():
                i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                fi, fj = int(frag_id[i]), int(frag_id[j])
                if fi == fj:
                    continue
                dt = float(np.linalg.norm(Q[i] - Q[j]))
                dp = float(np.linalg.norm(P[i] - P[j]))
                bond_true.append(dt)
                bond_pred.append(dp)
                bond_err.append(abs(dp - dt))
                if fi in Rs and fj in Rs:
                    bonded_pairs.add((min(fi, fj), max(fi, fj)))

            # D: locality
            ids = sorted(Rs)
            hb, hn = [], []
            for a in range(len(ids)):
                for b_ in range(a + 1, len(ids)):
                    f, g = ids[a], ids[b_]
                    ang = angle_deg(Rs[f].T @ Rs[g])
                    (hb if (f, g) in bonded_pairs else hn).append(ang)
            bonded += hb
            nonbonded += hn
            if hb and hn:
                gaps.append(np.mean(hb) - np.mean(hn))

        print(f"\n{label}")
        print(f"  Fragmente pro Komplex      : {stat(nfrags)}")
        print(f"  B  Schwerpunktfehler       : {stat(cents)} A")
        print(f"  B  Rotationsfehler absolut : {stat(rots)} Grad   [Zufall {RANDOM_BASELINE_DEG}]")
        print(f"  B  Restfehler nach Starrfit: {stat(resids)} A   (Sanity: ~0)")
        print(f"  C  Übergangsbindung Länge  : {stat(bond_pred)} A   (wahr: {np.mean(bond_true):.2f})")
        print(f"  C  Übergangsbindung Fehler : {stat(bond_err)} A")
        print(f"  D  rel. Rotation gebunden  : {stat(bonded)} Grad")
        print(f"  D  rel. Rotation ungebunden: {stat(nonbonded)} Grad")
        if gaps:
            g = np.array(gaps)
            rng = np.random.default_rng(0)
            boot = rng.choice(g, size=(10000, len(g)), replace=True).mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            print(f"  D  LÜCKE (gebunden-ungeb.) : {g.mean():+7.2f} Grad  "
                  f"CI [{lo:+.1f}, {hi:+.1f}]  (n={len(g)})")


if __name__ == "__main__":
    main()
