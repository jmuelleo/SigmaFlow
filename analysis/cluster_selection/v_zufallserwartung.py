"""Ist "richtige Pose im groessten Cluster" haeufiger als der Zufall erlaubt?

NULLHYPOTHESE. Die c richtigen Posen eines Komplexes liegen zufaellig unter
den K Kandidaten, unabhaengig von der Clusterstruktur. Dann ist

    P(groesster Cluster enthaelt mindestens eine richtige Pose)
        = 1 - C(K-L, c) / C(K, c)

mit L der Groesse des groessten Clusters. Das ist hypergeometrisch und
beruecksichtigt beides: dass der groesste Cluster viel Masse traegt, und dass
es oft mehrere richtige Posen gibt.

Beobachtet wird eine Null oder Eins je Komplex. Der Vergleich ist deshalb
eine Poisson-Binomialverteilung; getestet wird mit einem Bootstrap ueber die
Komplexe auf die Differenz Beobachtung minus Erwartung.
"""
import numpy as np
import pandas as pd
from scipy.special import gammaln

ZELLEN = [("SigmaDock", 25), ("Minimal", 25), ("Separate", 25),
          ("Minimal", 5), ("Separate", 5)]
LAB = {"SigmaDock": "SigmaDock", "Minimal": "SigmaFlow-NE",
       "Separate": "SigmaFlow-TR"}
B = 20000
rng = np.random.default_rng(11)


def logC(n, k):
    if k < 0 or k > n:
        return -np.inf
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


df = pd.read_csv("cluster_pb308.csv",
                 usecols=["arm", "nfe", "S", "complex", "cluster", "g_gr", "K",
                          "correct_fraction", "contains_correct",
                          "complex_hat_correct", "ranker_richtig"])
df = df[np.isclose(df.S, 2.0)]
print("PB308, Clusterschwelle 2,0 A, nur rettbare Fehlgriffe\n")
print(f"  {'Modell':<14}{'K':>5}{'n':>5}{'beobachtet':>12}{'Zufall':>9}"
      f"{'Differenz':>11}{'95%-KI':>20}{'p':>9}")
for arm, nfe in ZELLEN:
    z = df[(df.arm == arm) & (df.nfe == nfe)]
    if z.empty:
        continue
    beob, null = [], []
    for code, g in z.groupby("complex"):
        if g.ranker_richtig.iloc[0] or not g.complex_hat_correct.iloc[0]:
            continue
        K = int(g.K.iloc[0])
        c = int(round((g.correct_fraction * g.g_gr).sum()))
        L = int(g.g_gr.max())
        if c < 1 or c > K:
            continue
        gross = g.loc[g.g_gr.idxmax()]
        beob.append(1.0 if bool(gross.contains_correct) else 0.0)
        null.append(1.0 - np.exp(logC(K - L, c) - logC(K, c)))
    beob, null = np.asarray(beob), np.asarray(null)
    d = beob - null
    bs = d[rng.integers(0, len(d), (B, len(d)))].mean(axis=1)
    p = min(1.0, max(2 * min((bs <= 0).mean(), (bs >= 0).mean()), 1.0 / B))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    kk = 40 if (nfe == 25 or arm == "SigmaDock") else 200
    print(f"  {LAB[arm]:<14}{kk:>5}{len(d):>5}{100*beob.mean():>11.1f}%"
          f"{100*null.mean():>8.1f}%{100*d.mean():>+11.1f}"
          f"   [{100*lo:+.1f}, {100*hi:+.1f}]{p:>9.4f}")
