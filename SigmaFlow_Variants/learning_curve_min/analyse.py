"""Lernkurve SigmaFlow-Minimal ueber die vier Snapshots des Laufs 8648492.

Liest die per_pose.csv, die evaluate_run je Zelle geschrieben hat, und bildet
daraus die Groessen, die in der Arbeit berichtet werden. Die Trainingsposition
je Snapshot kommt aus den .meta.txt auf ARC und steht hier als Tabelle, weil
die Metadateien nicht mit heruntergeladen wurden.
"""
import collections
import csv
import math
import pathlib
import random
import statistics

HIER = pathlib.Path(__file__).resolve().parent
# walltime_h und Epoche aus snap_*.meta.txt (Lauf 8648492)
POSITION = {
    "sched255ep_at_006h": (6.003, 20),
    "sched255ep_at_012h": (12.003, 41),
    "sched255ep_at_018h": (18.005, 63),
    "sched255ep_final": (22.607, 78),
}
random.seed(0)


def wilson(k, n, z=1.96):
    """95-%-Wilson-Intervall fuer einen Anteil. Bei k=0 oder k=n bleibt es
    korrekt, anders als das Normalapproximationsintervall."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    m = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (m - h), 100 * (m + h))


def oracle_at(per_cx, K, schwelle, ziehungen=200):
    """Anteil Komplexe, bei denen unter K zufaellig gezogenen Posen mindestens
    eine unter der Schwelle liegt. Ueber zufaellige K-Teilmengen gemittelt,
    nicht ueber die ersten K -- sonst haengt das Ergebnis an der Seed-Nummer."""
    if K == 1:
        return 100 * statistics.fmean(
            statistics.fmean(1.0 if v < schwelle else 0.0 for v in vs)
            for vs in per_cx.values())
    treffer = []
    for vs in per_cx.values():
        if len(vs) <= K:
            treffer.append(1.0 if min(vs) < schwelle else 0.0)
            continue
        c = sum(1 for _ in range(ziehungen)
                if min(random.sample(vs, K)) < schwelle)
        treffer.append(c / ziehungen)
    return 100 * statistics.fmean(treffer)


zeilen = []
for d in sorted((HIER / "poses").iterdir()):
    if not d.is_dir():
        continue
    tag, rest = d.name.split("__nfe", 1)
    nfe = int(rest.split("__")[0])
    rows = list(csv.DictReader((d / "per_pose.csv").open(encoding="utf-8")))
    per_cx = collections.defaultdict(list)
    for r in rows:
        per_cx[r["complex"]].append(float(r["rmsd"]))
    alle = [v for vs in per_cx.values() for v in vs]
    k2 = sum(1 for v in alle if v < 2.0)
    k5 = sum(1 for v in alle if v < 5.0)
    h, ep = POSITION[tag]
    zeilen.append(dict(
        tag=tag, nfe=nfe, h=h, epoch=ep,
        n_cx=len(per_cx), n_pose=len(alle),
        med_all=statistics.median(alle),
        med_best=statistics.median(min(vs) for vs in per_cx.values()),
        rate2=100 * k2 / len(alle), ci2=wilson(k2, len(alle)),
        rate5=100 * k5 / len(alle),
        o1=oracle_at(per_cx, 1, 2.0),
        o5=oracle_at(per_cx, 5, 2.0),
        o10=oracle_at(per_cx, 10, 2.0),
    ))

for nfe in (25, 5):
    print(f"\n=== SigmaFlow-Minimal, {nfe} Integrationsschritte, "
          f"sampled, PoseBusters-v2-308 ===")
    print(f"{'h':>6} {'Ep.':>4} {'Kx':>4} {'Posen':>6} "
          f"{'medRMSD':>8} {'medBest':>8} "
          f"{'<2A/Zug':>9} {'95%-KI':>14} {'<5A/Zug':>8} "
          f"{'Or@1':>6} {'Or@5':>6} {'Or@10':>6}")
    for z in [z for z in zeilen if z["nfe"] == nfe]:
        lo, hi = z["ci2"]
        print(f"{z['h']:6.1f} {z['epoch']:4d} {z['n_cx']:4d} {z['n_pose']:6d} "
              f"{z['med_all']:8.2f} {z['med_best']:8.2f} "
              f"{z['rate2']:8.2f}% [{lo:5.2f},{hi:5.2f}] {z['rate5']:7.1f}% "
              f"{z['o1']:5.1f}% {z['o5']:5.1f}% {z['o10']:5.1f}%")

with (HIER / "curve_minimal_8648492.csv").open("w", newline="",
                                               encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["snapshot", "nfe", "walltime_h", "epoch", "n_complexes",
                "n_poses", "rmsd_median_all", "rmsd_best_median",
                "success_2A_per_draw", "ci_lo", "ci_hi",
                "success_5A_per_draw", "oracle_2A_K1", "oracle_2A_K5",
                "oracle_2A_K10"])
    for z in zeilen:
        w.writerow([z["tag"], z["nfe"], z["h"], z["epoch"], z["n_cx"],
                    z["n_pose"], round(z["med_all"], 4),
                    round(z["med_best"], 4), round(z["rate2"], 3),
                    round(z["ci2"][0], 3), round(z["ci2"][1], 3),
                    round(z["rate5"], 3), round(z["o1"], 3),
                    round(z["o5"], 3), round(z["o10"], 3)])
print(f"\ngeschrieben: {HIER / 'curve_minimal_8648492.csv'}")
