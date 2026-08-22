"""Exakte Zerlegung des Lagefehlers in Translation und Rotation.

WARUM NICHT KORRELATION
    Ein Vergleich "Korrelation von Winkel [Grad] mit RMSD" gegen "Korrelation
    von Versatz [Angstrom] mit RMSD" ist keine Zerlegung, sondern ein
    Vergleich zweier Einheiten. Der Versatz eines Fragments IST fast schon
    dessen RMSD-Beitrag, waehrend ein Winkel erst ueber den Traegheitsradius
    wirkt. Das verzerrt zugunsten der Translation.

DIE EXAKTE IDENTITAET
    Fuer ein Fragment mit Atomen P_i (vorhergesagt) und Q_i (wahr) und
    t = mean(P) - mean(Q) gilt

        (1/n) sum_i |P_i - Q_i|^2  =  |t|^2  +  (1/n) sum_i |P_i^c - Q_i^c|^2

    Der Kreuzterm verschwindet, weil die zentrierten Abweichungen im Mittel
    null sind. Der erste Summand ist der TRANSLATORISCHE Anteil, der zweite
    der ROTATORISCHE (plus etwaige innere Verformung). BEIDE in Angstrom^2,
    also direkt vergleichbar.

    Ueber alle Fragmente gewichtet mit deren Atomzahl ergibt das den
    Lagefehler des ganzen Liganden.

ABGRENZUNG ZUM BERICHTETEN RMSD
    Der in der Auswertung berichtete RMSD ist symmetriekorrigiert (spyrmsd)
    und kann daher kleiner sein als der hier naiv je Atomindex gerechnete.
    Die Zerlegung bezieht sich auf den naiven Wert; beide werden ausgegeben.
"""
import glob, os, sys
import numpy as np
from fragment_locality import find_fragments
from ligand_reference import best_copy, load_copies, load_first as load_mol
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

arm, root = sys.argv[1], sys.argv[2]
true_paths = sorted(glob.glob("true_ligands/*_ligands.sdf"))
out = open(f"budget_{arm}.csv", "w", encoding="utf-8", newline="")
out.write("complex,seed,n_atoms,n_frag,msd_trans,msd_rot,rmsd_naive\n")

for seed in range(10):
    sd = next((p for p in glob.glob(f"{root}/**/seed_{seed}", recursive=True)
               if os.path.isdir(p)), None)
    if sd is None:
        continue
    for tp in true_paths:
        cid = os.path.basename(tp).replace("_ligands.sdf", "")
        c = glob.glob(os.path.join(sd, f"{cid}__*.sdf"))
        if not c:
            continue
        mp = load_mol(c[0])
        if mp is None:
            continue
        mt, _, _ = best_copy(mp.GetConformer().GetPositions(), load_copies(tp))
        if mt is None or mt.GetNumBonds() != mp.GetNumBonds():
            continue
        pt = mt.GetConformer().GetPositions()
        pp = mp.GetConformer().GetPositions()
        fid = find_fragments(mt, mp)
        N = len(pt)
        st = sr = 0.0
        for f in np.unique(fid):
            sel = fid == f
            P, Q = pp[sel], pt[sel]
            t = P.mean(0) - Q.mean(0)
            st += sel.sum() * float(t @ t)
            Pc, Qc = P - P.mean(0), Q - Q.mean(0)
            sr += float(((Pc - Qc) ** 2).sum())
        msd_t, msd_r = st / N, sr / N
        naive = float(np.sqrt(((pp - pt) ** 2).sum(1).mean()))
        # Gegenprobe der Identitaet: msd_t + msd_r muss dem naiven MSD gleichen
        assert abs((msd_t + msd_r) - naive ** 2) < 1e-6 * max(1.0, naive ** 2), \
            f"Zerlegung stimmt nicht bei {cid} seed {seed}"
        out.write(f"{cid},{seed},{N},{len(np.unique(fid))},"
                  f"{msd_t:.5f},{msd_r:.5f},{naive:.4f}\n")
    print(f"  {arm} seed {seed}", flush=True)
out.close()
print(f"geschrieben: budget_{arm}.csv")
