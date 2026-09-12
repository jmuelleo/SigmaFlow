"""Waehlt der Ranker aus dem GROESSTEN Modus oder aus einem Nebenmodus?

DIE ENTSCHEIDENDE MESSUNG FUER DIE ERZAEHLUNG
    Die Behauptung lautet: SigmaFlow legt ein paar Posen in unwahrscheinlichere
    Moden, und der Ranker holt von dort die richtige Antwort. Waere das so,
    muesste die vom Ranker gewaehlte Pose bei SigmaFlow oefter AUSSERHALB des
    groessten Modus liegen -- und besonders oft in den Faellen, wo SigmaFlow
    trifft und SigmaDock nicht.

    Liegt die Wahl dagegen meistens im groessten Modus, dann folgt der Ranker
    der Dichte, und der Vorteil entsteht woanders.

    Geclustert wird wie in struktur.py: paarweiser RMSD ohne Ueberlagerung,
    mittlere Bindung, Schwelle 2 A.
"""
import glob, os, re, sys
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zellen import SAETZE, lade_zelle, posenordner

SCHWELLE = 2.0
ARME = ["SigmaDock", "Minimal", "Separate"]

tab = {}
for z in SAETZE["pb308"]["zellen"]:
    if z["nfe"] == 25:
        tab[z["arm"]] = lade_zelle(z, leise=True)

wurzel = {arm: posenordner("pb308", arm, 25) for arm in ARME}
codes = sorted(set.intersection(*[
    {os.path.basename(f).split("__")[0]
     for f in glob.glob(os.path.join(w, "seed_*", "*.sdf"))}
    for w in wurzel.values()]))
print(f"{len(codes)} Komplexe mit Posen\n")

zeilen = []
for code in codes:
    for arm in ARME:
        g = tab[arm]; g = g[g["complex"] == code]
        if g.empty: continue
        acc_von = dict(zip(g["seed"], g["acc"]))
        heur_von = dict(zip(g["seed"], g["heur"]))
        dateien = sorted(glob.glob(os.path.join(wurzel[arm], "seed_*", f"{code}__*.sdf")),
                         key=lambda f: int(re.search(r"seed_(\d+)", f).group(1)))
        x, seeds = [], []
        for f in dateien:
            m = Chem.MolFromMolFile(f, sanitize=False, removeHs=False)
            if m is None or m.GetNumConformers() == 0: continue
            x.append(m.GetConformer().GetPositions())
            seeds.append(int(re.search(r"seed_(\d+)", f).group(1)))
        if len(x) < 3: continue
        x = np.asarray(x, float)
        d = np.sqrt(((x[:, None] - x[None, :]) ** 2).sum(-1).mean(-1))
        np.fill_diagonal(d, 0.0)
        lab = fcluster(linkage(squareform(d, checks=False), method="average"),
                       t=SCHWELLE, criterion="distance")
        groessen = np.bincount(lab)[1:]
        gross = int(np.argmax(groessen)) + 1
        # Welche Pose waehlt der Ranker, und in welchem Modus liegt sie?
        h = np.array([heur_von.get(s, -np.inf) for s in seeds])
        i = int(np.argmax(h))
        zeilen.append({
            "complex": code, "arm": arm,
            "wahl_im_groessten": bool(lab[i] == gross),
            "wahl_modusgroesse": float(groessen[lab[i] - 1] / len(x)),
            "groesster_anteil": float(groessen.max() / len(x)),
            "wahl_trifft": bool(acc_von.get(seeds[i], False)),
            "n_moden": int(len(groessen)),
        })
t = pd.DataFrame(zeilen)
t.to_csv("nebenmodus.csv", index=False)

w = t.pivot_table(index="complex", columns="arm", values="wahl_trifft")
gr = pd.Series("beide daneben", index=w.index)
gr[w.SigmaDock.astype(bool) & w.Minimal.astype(bool)] = "beide treffen"
gr[~w.SigmaDock.astype(bool) & w.Minimal.astype(bool)] = "nur Minimal"
gr[w.SigmaDock.astype(bool) & ~w.Minimal.astype(bool)] = "nur SigmaDock"

print("=== Waehlt der Ranker aus dem groessten Modus? ===")
print(f"  {'Arm':<12}{'ja':>8}{'n':>6}")
for arm in ARME:
    d = t[t.arm == arm]
    print(f"  {arm:<12}{100*d.wahl_im_groessten.mean():7.1f}%{len(d):6d}")

print(f"\n  {'Gruppe':<16}{'n':>4}" + "".join(f"{a[:9]:>11}" for a in ARME))
for g in ("beide treffen", "nur Minimal", "nur SigmaDock", "beide daneben"):
    k = gr.index[gr == g]
    if not len(k): continue
    zs = []
    for arm in ARME:
        d = t[(t.arm == arm) & (t.complex.isin(k))]
        zs.append(f"{100*d.wahl_im_groessten.mean():10.1f}%")
    print(f"  {g:<16}{len(k):4d}" + "".join(zs))

print("\n=== Wenn die Wahl NICHT im groessten Modus lag: trifft sie dann? ===")
print(f"  {'Arm':<12}{'im groessten':>14}{'daneben':>10}{'n daneben':>11}")
for arm in ARME:
    d = t[t.arm == arm]
    a1 = d[d.wahl_im_groessten].wahl_trifft.mean() * 100
    a2 = d[~d.wahl_im_groessten].wahl_trifft.mean() * 100
    print(f"  {arm:<12}{a1:13.1f}%{a2:9.1f}%{(~d.wahl_im_groessten).sum():11d}")
