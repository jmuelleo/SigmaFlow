"""Validitaetszahlen mit Protein (redock) gegen ohne (regen), alle
vorhandenen Seeds. Dieselben Posen, nur ein anderes PoseBusters-Preset.

'valide' = ALLE Checks bestanden ausser rmsd und ausser den drei
Ladepruefungen. mol_cond_loaded wird geprueft und muss True sein, sonst
waere das Protein gar nicht angekommen.
"""
import csv, glob, re, sys
import numpy as np

def tf(v):
    s=str(v).strip().lower(); return True if s in ("true","1","1.0","yes") else False

def load(p):
    rows=list(csv.DictReader(open(p,encoding="utf-8",errors="replace")))
    cols=list(rows[0]); rm=next(c for c in cols if c.startswith("rmsd"))
    lc={"mol_pred_loaded","mol_true_loaded","mol_cond_loaded"}
    ch=[c for c in cols if c not in ({"file","molecule","position",rm,""}|lc)]
    prot=[c for c in ch if any(t in c for t in
          ("protein","cofactor","water","volume","distance"))]
    reg=[c for c in ch if c not in prot]
    out={}
    for r in rows:
        cid=r["file"].replace("\\","/").split("/")[-1].split("__")[0]
        out[cid]=dict(redock=all(tf(r[c]) for c in ch),
                      regen=all(tf(r[c]) for c in reg),
                      acc=tf(r[rm]), cond=tf(r["mol_cond_loaded"]),
                      checks={c:tf(r[c]) for c in prot})
    return out, prot

ARMS=[("SigmaFlow-Minimal","minimal"),("SigmaFlow-Separate","exp110"),("SigmaDock","sigmadock")]
D={}
for lab,k in ARMS:
    per={}
    for f in glob.glob(f"rd_{k}_seed*.csv"):
        s=int(re.search(r"seed(\d+)\.csv$",f).group(1)); per[s],prot=load(f)
    if per: D[lab]=(per,prot)
seeds=sorted(set.intersection(*[set(p) for p,_ in D.values()]))
cids=sorted(set.intersection(*[set(p[s]) for p,_ in D.values() for s in seeds]))
bad=[(l,s) for l,(p,_) in D.items() for s in seeds if not all(p[s][c]["cond"] for c in cids)]
print(f"\n  Seeds: {seeds}   Komplexe: {len(cids)}   Protein ueberall geladen: {not bad}")

def rate(lab, key, seedset):
    p,_=D[lab]
    return np.array([100*sum(p[s][c][key] for c in cids)/len(cids) for s in seedset])
def both(lab, seedset):
    p,_=D[lab]
    return np.array([100*sum(1 for c in cids if p[s][c]["redock"] and p[s][c]["acc"])/len(cids)
                     for s in seedset])

print("\n"+"="*76)
print(f"  PB-VALIDITAET MIT PROTEIN, Mittel ueber {len(seeds)} Seeds x {len(cids)} Komplexe")
print("="*76)
print(f"\n  {'Arm':22}{'regen':>9}{'redock':>10}{'Verlust':>11}{'genau':>9}{'beides':>9}")
print("  "+"-"*70)
for lab,_ in ARMS:
    rg,rd,ac,bo=rate(lab,"regen",seeds),rate(lab,"redock",seeds),rate(lab,"acc",seeds),both(lab,seeds)
    print(f"  {lab:22}{rg.mean():>8.1f}%{rd.mean():>9.1f}%{rd.mean()-rg.mean():>10.1f}pp"
          f"{ac.mean():>8.1f}%{bo.mean():>8.1f}%")

print(f"\n  redock-Validitaet je Seed")
print(f"\n  {'Seed':>6}" + "".join(f"{l.replace('SigmaFlow-','')[:11]:>15}" for l,_ in ARMS))
print("  "+"-"*52)
for i,s in enumerate(seeds):
    print(f"  {s:>6}" + "".join(f"{rate(l,'redock',[s])[0]:>14.1f}%" for l,_ in ARMS))
print("  "+"-"*52)
print(f"  {'Mittel':>6}" + "".join(f"{rate(l,'redock',seeds).mean():>14.1f}%" for l,_ in ARMS))

print("\n  Proteinbezogene Einzelchecks, Mittel ueber die Seeds")
print(f"\n  {'Check':44}" + "".join(f"{l.replace('SigmaFlow-','')[:9]:>11}" for l,_ in ARMS))
print("  "+"-"*77)
for c in D[ARMS[0][0]][1]:
    row=f"  {c:44}"
    for lab,_ in ARMS:
        p,_=D[lab]
        v=np.mean([100*sum(p[s][x]["checks"][c] for x in cids)/len(cids) for s in seeds])
        row+=f"{v:>10.0f}%"
    print(row)

print("\n  P(valide | genau), also wie sauber die TREFFER sind")
print(f"\n  {'Arm':22}{'unter regen':>14}{'unter redock':>15}")
print("  "+"-"*54)
for lab,_ in ARMS:
    p,_=D[lab]
    na=sum(1 for s in seeds for c in cids if p[s][c]["acc"])
    vg=sum(1 for s in seeds for c in cids if p[s][c]["acc"] and p[s][c]["regen"])
    vd=sum(1 for s in seeds for c in cids if p[s][c]["acc"] and p[s][c]["redock"])
    print(f"  {lab:22}{100*vg/na:>13.1f}%{100*vd/na:>14.1f}%   (n={na})")
