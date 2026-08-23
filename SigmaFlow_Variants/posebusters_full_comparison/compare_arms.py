"""Gepaarter Vergleich zweier Arme auf identischen Komplexen.

Einheit ist der KOMPLEX, nicht die Pose: die 40 Ziehungen eines Komplexes
sind nicht unabhaengig voneinander, ein Test ueber 8360 Posen wuerde die
Freiheitsgrade um den Faktor 40 ueberschaetzen.

Verglichen werden je Komplex
  (a) die Trefferzahl unter 2 A aus 40 Ziehungen  -> Erfolge je Komplex
  (b) der Median-RMSD der 40 Ziehungen            -> stetige Skala

Beides gepaart (dieselben Komplexe), Konfidenzintervall per Bootstrap
ueber Komplexe. Kein t-Test: die Verteilung der Trefferzahlen ist stark
nulllastig und schief.

Aufruf:  python compare_arms.py <arm_a> <arm_b> [schwelle]
         wobei pose_<arm>.csv vorliegen muss.
"""
import csv, sys
from collections import defaultdict
import numpy as np

a, b = sys.argv[1], sys.argv[2]
THR = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
RNG = np.random.default_rng(0)


def load(arm):
    d = defaultdict(dict)
    with open(f"pose_{arm}.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            d[r["complex"]][int(r["seed"])] = float(r["rmsd"])
    return d


A, B = load(a), load(b)
cids = sorted(set(A) & set(B))
seeds = sorted(set.intersection(*[set(A[c]) for c in cids], *[set(B[c]) for c in cids]))
print(f"{len(cids)} gemeinsame Komplexe x {len(seeds)} gemeinsame Seeds")

luecke = [c for c in cids if len(A[c]) != len(seeds) or len(B[c]) != len(seeds)]
if luecke:
    raise SystemExit(f"ABBRUCH: {len(luecke)} Komplexe unvollstaendig, z.B. {luecke[:3]}")

hits_a = np.array([sum(A[c][s] < THR for s in seeds) for c in cids], float)
hits_b = np.array([sum(B[c][s] < THR for s in seeds) for c in cids], float)
med_a = np.array([np.median([A[c][s] for s in seeds]) for c in cids])
med_b = np.array([np.median([B[c][s] for s in seeds]) for c in cids])


def boot(d, n=20000):
    idx = RNG.integers(0, len(d), size=(n, len(d)))
    reps = d[idx].mean(axis=1)
    p = 2 * min((reps <= 0).mean(), (reps >= 0).mean())   # zweiseitig, Nullwert 0
    return d.mean(), np.percentile(reps, [2.5, 97.5]), min(p, 1.0)


for name, da, db, einheit in [
    (f"Treffer < {THR} A je Komplex (von {len(seeds)})", hits_a, hits_b, ""),
    ("Median-RMSD je Komplex", med_a, med_b, " A"),
]:
    d = db - da
    m, ci, p = boot(d)
    print(f"\n  {name}")
    print(f"    {a:>14}: {da.mean():.3f}{einheit}")
    print(f"    {b:>14}: {db.mean():.3f}{einheit}")
    print(f"    Differenz   : {m:+.3f}{einheit}  95%-KI [{ci[0]:+.3f}, {ci[1]:+.3f}]  p = {p:.3f}")

print(f"\n  Mindestens ein Treffer aus {len(seeds)}:")
sa, sb = (hits_a > 0), (hits_b > 0)
print(f"    {a:>14}: {sa.sum():>4}  ({100*sa.mean():.1f} %)")
print(f"    {b:>14}: {sb.sum():>4}  ({100*sb.mean():.1f} %)")
print(f"    nur {a}: {int((sa & ~sb).sum())}   nur {b}: {int((sb & ~sa).sum())}   "
      f"beide: {int((sa & sb).sum())}   keiner: {int((~sa & ~sb).sum())}")
# McNemar exakt auf den diskordanten Paaren
n01, n10 = int((sa & ~sb).sum()), int((sb & ~sa).sum())
if n01 + n10:
    from scipy.stats import binomtest
    print(f"    McNemar (exakt): p = {binomtest(n10, n01 + n10, 0.5).pvalue:.4f}")
