"""Je Pose: Rotations- und Translationsfehler der Fragmente, Fragmentzahl,
Atomzahl. Grundlage fuer den Fehlerhaushalt und fuer die Frage, ob die
Leistung von der Molekuelgroesse abhaengt.

Fragmentierung wie in fragment_locality.py aus den Posen selbst
zurueckgewonnen: Bindungen, deren Laenge zwischen wahrer und vorhergesagter
Pose erhalten bleibt, liegen innerhalb eines starren Fragments.
"""
import glob, os, sys
import numpy as np
from fragment_locality import (find_fragments, kabsch_rotation,
                               rotation_angle_deg, MIN_FRAG_ATOMS)
from ligand_reference import best_copy, load_copies, load_first as load_mol
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

arm, root = sys.argv[1], sys.argv[2]
true_paths = sorted(glob.glob("true_ligands/*_ligands.sdf"))
out = open(f"geom_{arm}.csv", "w", encoding="utf-8", newline="")
out.write("complex,seed,n_atoms,n_frag,rot_mean,rot_max,trans_mean,trans_com\n")

seed_dirs = sorted(
    (int(os.path.basename(p).split("_")[1]), p)
    for p in glob.glob(f"{root}/**/seed_*", recursive=True) if os.path.isdir(p))
if not seed_dirs:
    raise SystemExit(f"ABBRUCH: keine seed_*-Verzeichnisse unter {root}")
print(f"  {arm}: {len(seed_dirs)} Seeds gefunden", flush=True)

for seed, sd in seed_dirs:
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
        rot, tra = [], []
        for f in np.unique(fid):
            sel = fid == f
            if sel.sum() < MIN_FRAG_ATOMS:
                continue
            rot.append(rotation_angle_deg(kabsch_rotation(pp[sel], pt[sel])))
            tra.append(float(np.linalg.norm(pp[sel].mean(0) - pt[sel].mean(0))))
        if not rot:
            continue
        # Schwerpunktversatz des GANZEN Liganden: der Anteil des Fehlers, der
        # reine Fehlplatzierung ist und nichts mit der inneren Geometrie zu tun hat.
        com = float(np.linalg.norm(pp.mean(0) - pt.mean(0)))
        out.write(f"{cid},{seed},{mt.GetNumAtoms()},{len(np.unique(fid))},"
                  f"{np.mean(rot):.3f},{max(rot):.3f},{np.mean(tra):.4f},{com:.4f}\n")
    print(f"  {arm} seed {seed} fertig", flush=True)
out.close()
print(f"geschrieben: geom_{arm}.csv")
