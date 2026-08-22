"""
Rotationsqualitaet der drei Arme, ueber alle 10 Seeds.

ZWEI GETRENNTE GROESSEN, weil sie verschiedene Fragen beantworten:

(1) ABSOLUTER Fragmentfehler
    Kabsch je Fragment gegen das wahre Fragment, Winkel der Rotation.
    Enthaelt den globalen Orientierungsfehler des ganzen Molekuels mit.
    Frage: "wie richtig zeigt dieses Fragment im Laborsystem?"

(2) RELATIVER Fehler zwischen Fragmentpaaren
    R_rel = R_f^T R_g. Ein gemeinsamer Fehler beider Fragmente kuerzt sich
    exakt heraus. Aufgeteilt nach gebunden / ungebunden.
    Frage: "sind die Fragmente untereinander konsistent orientiert?"
    Der Abstand zwischen beiden ist das Lokalitaetssignal.

NULLHYPOTHESE: bei Haar-gleichverteilter Rotation hat der Winkel die Dichte
(1 - cos t)/pi auf [0, pi]. Mittel 126.5 Grad, Median 132.3 Grad. Wird unten
numerisch nachgerechnet statt erinnert.

VORBEHALT SYMMETRIE: fuer ein symmetrisches Fragment (Phenyl, tert-Butyl) ist
die Rotation nur bis auf die Symmetriegruppe bestimmt. Kabsch auf der
gegebenen Atomnummerierung liefert einen bestimmten Vertreter, nicht den
naechstliegenden. Das BLAEHT den gemessenen Fehler AUF, und zwar fuer alle
drei Arme gleichermassen. Der Vergleich zwischen den Armen bleibt gueltig,
die absolute Hoehe ist eine Obergrenze.


Dieses Skript rechnet BEIDE Groessen sowie die Translation, gepaart je
(Komplex, Seed), mit Bootstrap ueber Komplexe.
"""
import glob, os
import numpy as np
from fragment_locality import (find_fragments, kabsch_rotation,
                               rotation_angle_deg, MIN_FRAG_ATOMS)
from ligand_reference import best_copy, load_copies, load_first as load_mol
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

ARMS = [("SigmaFlow-Minimal", "eval_min_root/results"),
        ("SigmaDock",         "eval_sd_root/results"),
        ("EXP-110",           "exp110_10seeds/results")]
true_paths = sorted(glob.glob("true_ligands/*_ligands.sdf"))

# key = (cid, seed) -> (mittlerer Rotationsfehler, mittlerer Translationsfehler)
per_arm = {}
for label, root in ARMS:
    d = {}
    for seed in range(10):
        sd = next((p for p in glob.glob(f"{root}/**/seed_{seed}", recursive=True)
                   if os.path.isdir(p)), None)
        if sd is None: continue
        for tp in true_paths:
            cid = os.path.basename(tp).replace("_ligands.sdf", "")
            c = glob.glob(os.path.join(sd, f"{cid}__*.sdf"))
            if not c: continue
            mp = load_mol(c[0])
            if mp is None: continue
            mt, _, _ = best_copy(mp.GetConformer().GetPositions(), load_copies(tp))
            if mt is None or mt.GetNumBonds() != mp.GetNumBonds(): continue
            pt = mt.GetConformer().GetPositions()
            pp = mp.GetConformer().GetPositions()
            fid = find_fragments(mt, mp)
            rot, tra = [], []
            for f in np.unique(fid):
                sel = fid == f
                if sel.sum() < MIN_FRAG_ATOMS: continue
                rot.append(rotation_angle_deg(kabsch_rotation(pp[sel], pt[sel])))
                tra.append(float(np.linalg.norm(pp[sel].mean(0) - pt[sel].mean(0))))
            if rot: d[(cid, seed)] = (float(np.mean(rot)), float(np.mean(tra)))
    per_arm[label] = d
    print(f"  {label:20} {len(d)} (Komplex, Seed)-Paare", flush=True)

keys = sorted(set.intersection(*[set(d) for d in per_arm.values()]))
cids = sorted({k[0] for k in keys})
print(f"\n  gepaart ueber {len(keys)} Posen aus {len(cids)} Komplexen\n")

def by_complex(d, idx):
    """je Komplex ueber seine Seeds mitteln -> unabhaengige Einheit"""
    acc = {}
    for (c, s) in keys: acc.setdefault(c, []).append(d[(c, s)][idx])
    return np.array([np.mean(acc[c]) for c in cids])

R = {l: by_complex(per_arm[l], 0) for l, _ in ARMS}
T = {l: by_complex(per_arm[l], 1) for l, _ in ARMS}

print("=" * 76)
print("  Rotation und Translation je Fragment, gemittelt je Komplex")
print("=" * 76)
print(f"\n  {'Arm':20}{'Rotation':>12}{'Translation':>14}")
print("  " + "-" * 48)
for l, _ in ARMS:
    print(f"  {l:20}{R[l].mean():>10.1f} Gr{T[l].mean():>11.2f} A")
print(f"  {'Haar / Zufall':20}{126.5:>10.1f} Gr{'':>13}")

rng = np.random.default_rng(0)
def paired(a, b, name):
    d = a - b
    boot = rng.choice(d, size=(20000, len(d)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return f"  {name:34}{d.mean():>+9.2f}   [{lo:+.2f}, {hi:+.2f}]   p={max(p,1/20000):.4f}"

print()
print("  Gepaarte Differenzen, Bootstrap ueber Komplexe (n=%d)" % len(cids))
print(f"\n  {'Vergleich':34}{'Diff':>9}{'95%-KI':>19}{'':>10}")
print("  " + "-" * 72)
print("  ROTATION (Grad, negativ = erster Arm besser)")
print(paired(R["EXP-110"], R["SigmaFlow-Minimal"], "EXP-110 minus Minimal"))
print(paired(R["EXP-110"], R["SigmaDock"], "EXP-110 minus SigmaDock"))
print(paired(R["SigmaFlow-Minimal"], R["SigmaDock"], "Minimal minus SigmaDock"))
print("\n  TRANSLATION (Angstrom, negativ = erster Arm besser)")
print(paired(T["EXP-110"], T["SigmaFlow-Minimal"], "EXP-110 minus Minimal"))
print(paired(T["EXP-110"], T["SigmaDock"], "EXP-110 minus SigmaDock"))
print(paired(T["SigmaFlow-Minimal"], T["SigmaDock"], "Minimal minus SigmaDock"))
np.save("/tmp/rt.npy", {"R": R, "T": T, "cids": cids}, allow_pickle=True)
