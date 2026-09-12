"""Vier Auswahlverfahren auf Basis der Clusterstruktur, gegen die Basis.

BASIS      hoechster Mixed Score ueber ALLE Posen.
GROESSTER  immer den groessten Cluster nehmen, darin Mixed Score.
TERZIL     Anteil im groessten Cluster je Komplex. Liegt er in den oberen
           zwei Terzilen, normal ranken; im unteren Terzil den groessten
           Cluster nehmen und darin ranken. Die Terzilgrenzen werden je
           Modell aus der Verteilung ueber die Komplexe bestimmt.
Beide noch einmal, nachdem alle nicht PB-validen Posen ENTFERNT und die
verbleibenden neu geclustert wurden. Bleibt keine valide Pose uebrig, faellt
das Verfahren auf die Basis zurueck.

Geclustert wird wie ueberall: paarweiser RMSD ohne Ueberlagerung,
vollstaendige Verkettung, Schnitt bei 2,0 A.
"""
import argparse
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

import posencache

p = argparse.ArgumentParser()
p.add_argument("--satz", default="pb308")
p.add_argument("--schwelle", type=float, default=2.0)
a = p.parse_args()

ZELLEN = [("SigmaDock", 25), ("Minimal", 25), ("Separate", 25),
          ("Minimal", 5), ("Separate", 5)]
LAB = {"SigmaDock": "SigmaDock", "Minimal": "SigmaFlow-NE",
       "Separate": "SigmaFlow-TR"}


def cluster(D, t):
    n = len(D)
    if n <= 1:
        return np.zeros(n, int)
    return fcluster(linkage(squareform(D, checks=False), method="complete"),
                    t=t, criterion="distance")


def groesster(lab):
    w = np.bincount(lab)
    return int(w.argmax()), float(w.max() / w.sum())


for ziel in ("beides", "acc"):
    print(f"\n{'='*100}")
    print(f"{a.satz.upper()}, Schnitt {a.schwelle} A, Ziel "
          f"{'RMSD<2 UND PB-valide' if ziel=='beides' else 'RMSD<2'}")
    print(f"{'='*100}")
    print(f"  {'Modell':<14}{'K':>5}{'Basis':>9}{'groesster':>11}{'Terzil':>9}"
          f"{'groesster*':>12}{'Terzil*':>10}   (* = vorher PB-gefiltert)")
    for arm, nfe in ZELLEN:
        daten = posencache.hole(a.satz, arm, nfe, leise=True)
        basis, gr, grf, anteil, anteilf, y_gr, y_grf = [], [], [], [], [], [], []
        for code, v in daten.items():
            D = v["d"].astype(float)
            h = v["heur"].astype(float)
            y = np.asarray(v[ziel], bool)
            ok = np.asarray(v["valid"], bool)
            basis.append(bool(y[int(np.argmax(h))]))

            lab = cluster(D, a.schwelle)
            c, s = groesster(lab)
            m = np.where(lab == c)[0]
            y_gr.append(bool(y[m[int(np.argmax(h[m]))]]))
            anteil.append(s)

            if ok.sum() >= 1:
                idx = np.where(ok)[0]
                labf = cluster(D[np.ix_(idx, idx)], a.schwelle)
                cf, sf = groesster(labf)
                mf = idx[np.where(labf == cf)[0]]
                y_grf.append(bool(y[mf[int(np.argmax(h[mf]))]]))
                anteilf.append(sf)
            else:
                y_grf.append(basis[-1])
                anteilf.append(1.0)

        basis = np.array(basis); y_gr = np.array(y_gr); y_grf = np.array(y_grf)
        anteil = np.array(anteil); anteilf = np.array(anteilf)
        q = np.quantile(anteil, 1 / 3); qf = np.quantile(anteilf, 1 / 3)
        terz = np.where(anteil <= q, y_gr, basis)
        terzf = np.where(anteilf <= qf, y_grf, basis)
        kk = 40 if (nfe == 25 or arm == "SigmaDock") else 200
        print(f"  {LAB[arm]:<14}{kk:>5}{100*basis.mean():>8.2f}%"
              f"{100*y_gr.mean():>10.2f}%{100*terz.mean():>8.2f}%"
              f"{100*y_grf.mean():>11.2f}%{100*terzf.mean():>9.2f}%")
