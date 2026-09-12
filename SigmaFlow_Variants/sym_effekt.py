"""Wie stark aendert die Symmetriekorrektur die Clusterung?

Die Auswertung clustert auf dem paarweisen RMSD ueber die Atomreihenfolge
(posencache `d`). RDKits CalcRMS probiert zusaetzlich die Automorphismen des
Molekuelgraphen durch, sodass ein um 180 Grad gedrehter Phenylring nicht mehr
als Unterschied zaehlt. Ohne Korrektur ist der Abstand also eine OBERGRENZE
und die Modenzahl eher zu hoch. Dieses Skript misst, um wie viel.
"""
import argparse, glob, os, re, sys
import numpy as np
from rdkit.Chem import rdMolAlign
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

import posencache
from zellen import SAETZE, lade_zelle, posenordner

p = argparse.ArgumentParser()
p.add_argument("--satz", default="pb308")
p.add_argument("--nfe", type=int, default=25)
p.add_argument("--n", type=int, default=40, help="Zahl der Stichprobenkomplexe")
p.add_argument("--schwelle", type=float, default=2.0)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
rng = np.random.default_rng(a.seed)

ARME = [("SigmaDock", "SigmaDock"), ("Minimal", "SigmaFlow-NE"),
        ("Separate", "SigmaFlow-TR")]


def moden(D, t):
    if len(D) <= 1:
        return 1, 1.0
    lab = fcluster(linkage(squareform(D, checks=False), method="complete"),
                   t=t, criterion="distance")
    g = np.bincount(lab)[1:]
    return len(g), float(g.max() / g.sum())


erste = True
for arm, lab in ARME:
    z = next(z for z in SAETZE[a.satz]["zellen"]
             if z["arm"] == arm and z["nfe"] == a.nfe)
    tab = lade_zelle(z, leise=True)
    w = posenordner(a.satz, arm, a.nfe)
    codes = sorted(tab["complex"].unique())
    if erste:
        stich = list(rng.choice(codes, size=min(a.n, len(codes)), replace=False))
        erste = False
    roh, sym, gr_roh, gr_sym, dmax = [], [], [], [], []
    for code in stich:
        g = tab[tab["complex"] == code]
        heur_von = dict(zip(g["seed"], g["heur"]))
        dateien = sorted(glob.glob(os.path.join(w, "seed_*", f"{code}__*.sdf")),
                         key=lambda f: int(re.search(r"seed_(\d+)", f).group(1)))
        x, mole, gesehen = [], [], set()
        for f in dateien:
            s = int(re.search(r"seed_(\d+)", f).group(1))
            if s not in heur_von or s in gesehen:
                continue
            gesehen.add(s)
            k, el, mol = posencache._lade(f, sanitize=True)
            if k is None or mol is None:
                continue
            x.append(k); mole.append(mol)
        if len(x) < 3:
            continue
        x = np.asarray(x, float); n = len(x)
        D = np.sqrt(((x[:, None] - x[None, :]) ** 2).sum(-1).mean(-1))
        np.fill_diagonal(D, 0.0)
        S = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                try:
                    v = float(rdMolAlign.CalcRMS(mole[i], mole[j]))
                except Exception:
                    v = D[i, j]
                S[i, j] = S[j, i] = v
        mr, pr = moden(D, a.schwelle); ms, ps = moden(S, a.schwelle)
        roh.append(mr); sym.append(ms); gr_roh.append(pr); gr_sym.append(ps)
        dmax.append(float((D - S)[np.triu_indices(n, 1)].max()))
    print(f"{lab:<14} n={len(roh):3d}  Moden {np.mean(roh):6.2f} -> "
          f"{np.mean(sym):6.2f} ({np.mean(sym)-np.mean(roh):+.2f})   "
          f"groesster {100*np.mean(gr_roh):5.1f} % -> {100*np.mean(gr_sym):5.1f} %"
          f"   max. Abstandsreduktion {np.mean(dmax):.3f} A", flush=True)
