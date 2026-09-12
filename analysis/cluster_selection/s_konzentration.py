"""Sagt die Konzentration der Ziehungen allein voraus, ob die gewaehlte Pose
stimmt? Kein Score, keine Kombination -- nur der Anteil der Posen im
groessten Cluster.

Endpunkt: ranker_richtig (die ausgelieferte Pose erfuellt RMSD<2 UND PB-valide).
Mass:     g_relgr des groessten Clusters des Komplexes.
KI:       Bootstrap ueber Komplexe, 10.000 Ziehungen.
"""
import numpy as np
import pandas as pd

ZELLEN = [("SigmaDock", 25), ("Minimal", 25), ("Separate", 25),
          ("Minimal", 5), ("Separate", 5)]
rng = np.random.default_rng(20260909)


def auc(x, y):
    """Mann-Whitney-AUC: P(x hoeher bei y=1 als bei y=0), Bindungen halb."""
    if y.sum() == 0 or (~y).sum() == 0:
        return np.nan
    r = pd.Series(x).rank().to_numpy()
    n1, n0 = int(y.sum()), int((~y).sum())
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


for satz in ("pb308", "astex"):
    df = pd.read_csv(f"cluster_{satz}.csv",
                     usecols=["arm", "nfe", "S", "complex", "cluster",
                              "g_relgr", "ranker_richtig"])
    for S in (2.0, 1.0):
        dS = df[np.isclose(df.S, S)]
        print(f"\n=== {satz.upper()}, Clusterschwelle {S:.1f} A ===")
        print(f"  {'Modell':<16}{'n':>5}{'AUC':>8}  {'95%-KI':<18}"
              f"{'oberes Terzil':>15}{'unteres':>9}")
        for arm, nfe in ZELLEN:
            z = dS[(dS.arm == arm) & (dS.nfe == nfe)]
            if z.empty:
                continue
            g = z.groupby("complex")
            k = pd.DataFrame({"konz": g.g_relgr.max(),
                              "y": g.ranker_richtig.first()})
            x, y = k.konz.to_numpy(float), k.y.to_numpy(bool)
            a = auc(x, y)
            n = len(k)
            bs = []
            for _ in range(10000):
                i = rng.integers(0, n, n)
                v = auc(x[i], y[i])
                if not np.isnan(v):
                    bs.append(v)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            q1, q3 = np.quantile(x, [1 / 3, 2 / 3])
            ob = 100 * y[x >= q3].mean()
            un = 100 * y[x <= q1].mean()
            print(f"  {arm+' '+str(nfe):<16}{n:>5}{a:>8.3f}  "
                  f"[{lo:.3f}, {hi:.3f}]   {ob:>10.1f} %{un:>8.1f} %")
