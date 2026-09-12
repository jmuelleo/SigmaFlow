"""Form der erzeugten Verteilung je Modell -- die Kerntabelle von Kapitel 5.

Alles auf den VOLLEN Saetzen (307 / 85), Clusterschwelle 2,0 A, vollstaendige
Verkettung auf dem paarweisen symmetriekorrigierten RMSD ohne Ueberlagerung.

  Cluster        Zahl der Cluster je Komplex  = effektiv unterscheidbare Posen
  groesster      Anteil der Posen im groessten Cluster
  Entropie       normierte Entropie der Clustermassen
  innen          mittlerer Paarabstand INNERHALB eines Clusters, nach
                 Clustergroesse gewichtet, nur Cluster mit >= 2 Posen
Gepaarte Tests gegen SigmaDock: Bootstrap ueber Komplexe, 10.000 Ziehungen.
"""
import numpy as np
import pandas as pd

ZELLEN = [("SigmaDock", 25), ("Minimal", 25), ("Separate", 25),
          ("Minimal", 5), ("Separate", 5)]
SP = ["arm", "nfe", "S", "complex", "K", "cluster", "g_gr", "g_relgr",
      "g_intra_mean", "a_n_cluster", "a_masse_ent_norm", "ranker_richtig"]
rng = np.random.default_rng(20260909)


def je_komplex(z):
    g = z.groupby("complex")
    innen = (z[z.g_gr >= 2].groupby("complex")
             .apply(lambda d: np.average(d.g_intra_mean, weights=d.g_gr)))
    return pd.DataFrame({
        "n_cluster": g.a_n_cluster.first(),
        "groesster": g.g_relgr.max(),
        "entropie": g.a_masse_ent_norm.first(),
        "innen": innen,
        "K": g.K.first(),
    })


for satz in ("pb308", "astex"):
    df = pd.read_csv(f"cluster_{satz}.csv", usecols=SP)
    df = df[np.isclose(df.S, 2.0)]
    tab = {f"{a} {n}": je_komplex(df[(df.arm == a) & (df.nfe == n)])
           for a, n in ZELLEN if not df[(df.arm == a) & (df.nfe == n)].empty}
    ix = sorted(set.intersection(*[set(v.index) for v in tab.values()]))
    basis = tab["SigmaDock 25"].loc[ix]
    print(f"\n{'='*94}\n{satz.upper()}  ({len(ix)} Komplexe)\n{'='*94}")
    print(f"  {'Modell':<15}{'K':>5}{'Cluster':>9}{'je Zug':>8}"
          f"{'groesster':>11}{'Entropie':>10}{'innen [A]':>11}")
    for name, t in tab.items():
        t = t.loc[ix]
        print(f"  {name:<15}{int(t.K.median()):>5}{t.n_cluster.mean():>9.2f}"
              f"{t.n_cluster.mean()/t.K.median():>8.2f}"
              f"{100*t.groesster.mean():>10.1f} %{t.entropie.mean():>10.3f}"
              f"{t.innen.mean():>11.3f}")

    print(f"\n  gepaart gegen SigmaDock 25 (nur die 25-Schritt-Zellen sind")
    print(f"  bei gleicher Ziehungszahl vergleichbar):")
    for name in ("Minimal 25", "Separate 25"):
        t = tab[name].loc[ix]
        for sp, tag in [("n_cluster", "Cluster"), ("groesster", "groesster"),
                        ("innen", "innen")]:
            a = basis[sp].to_numpy(float)
            b = t[sp].to_numpy(float)
            m = ~(np.isnan(a) | np.isnan(b))
            d = b[m] - a[m]
            bs = d[rng.integers(0, len(d), (10000, len(d)))].mean(axis=1)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
            e = " %" if sp == "groesster" else ""
            f = 100 if sp == "groesster" else 1
            print(f"    {name:<13}{tag:<11}{f*d.mean():+8.3f}{e}  "
                  f"[{f*lo:+.3f}, {f*hi:+.3f}]  p = {max(p,1e-4):.4f}")
