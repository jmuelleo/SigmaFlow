"""RMSD-Tabelle der vier Snapshots, ohne PoseBusters.

Alles aus per_pose.csv (symmetriekorrigierte RMSD aus evaluate_run/spyrmsd),
also genau die Konvention, die die Arbeit durchgehend berichtet. Kostet
Sekunden, weil nur CSV gelesen wird -- kein RDKit, kein UFF.

Je Zelle:
    Posen / Kompl.   307 Komplexe, 8-11 Posen je Komplex (Sampler-Macke)
    Median           Median der RMSD ueber alle Posen
    <2 A, <5 A       Trefferquote je einzelner Ziehung
    Oracle <2 A      mindestens eine Pose des Komplexes unter 2 A
    Med. beste       Median der je Komplex besten RMSD
"""
import math
import pathlib

import pandas as pd

HIER = pathlib.Path(__file__).resolve().parent
POSITION = {"sched255ep_at_006h": (6.0, 20), "sched255ep_at_012h": (12.0, 41),
            "sched255ep_at_018h": (18.0, 63), "sched255ep_final": (22.6, 78)}


def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    m = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (m - h), 100 * (m + h)


zeilen = []
for zelle in sorted((HIER / "poses").iterdir()):
    if not zelle.is_dir():
        continue
    tag, rest = zelle.name.split("__nfe", 1)
    nfe = int(rest.split("__")[0])
    d = pd.read_csv(zelle / "per_pose.csv")
    d["u2"] = d["rmsd"] < 2.0
    je_kompl = d.groupby("complex")["rmsd"].min()
    h, ep = POSITION[tag]
    lo, hi = wilson(int(d["u2"].sum()), len(d))
    zeilen.append({"h": h, "ep": ep, "nfe": nfe, "n": len(d),
                   "nk": d["complex"].nunique(),
                   "med": d["rmsd"].median(), "u2": 100 * d["u2"].mean(),
                   "lo": lo, "hi": hi,
                   "u5": 100 * (d["rmsd"] < 5.0).mean(),
                   "orak": 100 * (je_kompl < 2.0).mean(),
                   "medbest": je_kompl.median()})

kopf = (f"{'h':>6}{'Ep':>5}{'Posen':>7}{'Kompl':>7}{'Median':>9}"
        f"{'<2A':>8}{'95%-KI':>16}{'<5A':>8}{'Oracle<2A':>11}{'Med.beste':>11}")
for nfe in (25, 5):
    print(f"\n=== {nfe} Integrationsschritte ===")
    print(kopf)
    for z in [z for z in zeilen if z["nfe"] == nfe]:
        print(f"{z['h']:6.1f}{z['ep']:5d}{z['n']:7d}{z['nk']:7d}"
              f"{z['med']:8.2f}A{z['u2']:7.2f}%  [{z['lo']:5.2f},{z['hi']:5.2f}]"
              f"{z['u5']:7.2f}%{z['orak']:10.2f}%{z['medbest']:10.2f}A")

pd.DataFrame(zeilen).to_csv(HIER / "tabelle_rmsd.csv", index=False)
print(f"\ngeschrieben: {HIER / 'tabelle_rmsd.csv'}")
