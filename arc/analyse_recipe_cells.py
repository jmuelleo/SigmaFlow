"""Rezept-Endpunkte: Minimal gegen Separate, 25 und 5 Integrationsschritte.

WAS IMPORTIERT WIRD UND WARUM
    `als_bool` aus pb_bool und `PB_CHECKS`/`BETA` aus dem Repository. Das sind
    die beiden Stellen, an denen eine zweite Fassung teuer waere: die
    Bool-Konvention (siehe Kopf von pb_bool.py) und die Paperformel des Mixed
    Score. Der Rest -- Tabellen zusammensetzen, gruppieren -- steht hier
    ausgeschrieben, weil `SigmaFlow_Variants/vergleich_rezept.py` eine lokale
    Auswertungsdatei ist und auf ARC nicht liegt.

WARUM ALLE 40 ZIEHUNGEN STATT EINES TEILMENGEN-SCHAETZERS
    Alle vier Zellen haben 40 Seeds. Weil alle vier identisch gerechnet
    werden, ist der Vergleich in sich stimmig, auch wenn er um Zehntel von
    Tabellenwerten abweicht, die den 1000-Teilmengen-Schaetzer benutzen. Die
    K-Kurve unten benutzt den Schaetzer.

Aufruf:
    $ARC_SF_ENV/bin/python arc/analyse_recipe_cells.py
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

_HIER = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HIER)
for _p in (_HIER, os.path.join(_REPO, "SigmaFlow_Variants", "learning_curve_min")):
    if os.path.isfile(os.path.join(_p, "pb_bool.py")):
        sys.path.insert(0, _p)
        break
else:
    sys.exit("pb_bool.py nicht gefunden -- keine zweite Fassung schreiben.")
from pb_bool import als_bool  # noqa: E402

sys.path.insert(0, _REPO)
try:
    from SigmaFlow_Evaluation.ranking.heuristic_score import BETA, PB_CHECKS  # noqa: E402
except Exception as e:  # noqa: BLE001
    sys.exit(f"heuristic_score nicht importierbar ({e}). Ohne die Paperformel "
             f"wird hier nichts geraten.")

WURZEL = sys.argv[1] if len(sys.argv) > 1 else "/data/stat-cadd/shug8458/arc_runs"
LADE = {"mol_pred_loaded", "mol_true_loaded", "mol_cond_loaded"}

ZELLEN = {
    ("Minimal", 25): "sigmaflow_minimal__recipe__confsampled",
    ("Minimal", 5): "sigmaflow_minimal__nfe5__recipe__confsampled",
    ("Separate", 25): "exp110__recipe__confsampled",
    ("Separate", 5): "exp110__nfe5__recipe__confsampled",
}


def lade(tag: str) -> pd.DataFrame:
    rg = os.path.join(WURZEL, "posebusters_redock", tag, "rd_*seed*.csv")
    teile = []
    for f in sorted(glob.glob(rg)):
        d = pd.read_csv(f)
        d["seed"] = int(re.search(r"seed(\d+)\.csv$", f).group(1))
        teile.append(d)
    if not teile:
        sys.exit(f"keine Redock-Tabellen unter {rg}")
    d = pd.concat(teile, ignore_index=True)
    d["complex"] = d["file"].map(lambda p: os.path.basename(str(p)).split("__")[0])

    rmsd_sp = next(c for c in d.columns if c.startswith("rmsd"))
    pruef = [c for c in d.columns
             if c not in LADE | {"file", "molecule", "position", "seed", "complex", rmsd_sp}]
    for c in pruef + [rmsd_sp]:
        d[c] = als_bool(d[c])
    d["valid"] = d[pruef].all(axis=1)
    d["p_pb"] = d[list(PB_CHECKS)].mean(axis=1)
    d["acc"] = d[rmsd_sp]

    gn = glob.glob(os.path.join(WURZEL, f"GNINA-SCORE-{tag}_*", f"gnina_scores_{tag}.csv"))
    if not gn:
        sys.exit(f"keine gnina-Tabelle fuer {tag}")
    g = pd.read_csv(gn[0])
    m = d.merge(g[["complex", "seed", "affinity"]], on=["complex", "seed"], how="inner")
    if len(m) != len(d):
        print(f"  HINWEIS {tag}: {len(d)} Redock-Zeilen, {len(m)} nach dem Merge")
    m["beides"] = m["acc"] & m["valid"]
    m["heur"] = -m["affinity"] * (m["p_pb"] ** BETA)
    return m


def wahl(m: pd.DataFrame, ziel: str, wie: str) -> pd.Series:
    g = m.groupby("complex")
    if wie == "orakel":
        return g[ziel].any()
    if wie == "gnina":
        return g.apply(lambda t: t.loc[t["affinity"].idxmin(), ziel],
                       include_groups=False).astype(bool)
    return g.apply(lambda t: t.loc[t["heur"].idxmax(), ziel],
                   include_groups=False).astype(bool)


def boot(a: pd.Series, b: pd.Series, n: int = 8000, seed: int = 20260905):
    """Gepaarter Bootstrap ueber Komplexe auf der Differenz der Mittelwerte."""
    idx = b.index.intersection(a.index)
    x, y = a.loc[idx].to_numpy(float), b.loc[idx].to_numpy(float)
    d = y - x
    rng = np.random.default_rng(seed)
    v = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    lo, hi = np.percentile(v, [2.5, 97.5])
    return 100 * d.mean(), 100 * lo, 100 * hi, min(2 * min((v <= 0).mean(), (v >= 0).mean()), 1.0)


def top1_bei_k(m: pd.DataFrame, ziel: str, ks, B: int = 1000, seed: int = 20260905):
    """Top-1 nach Mixed Score bei K Ziehungen, Erwartung ueber K-Teilmengen."""
    rng = np.random.default_rng(seed)
    aus = {k: [] for k in ks}
    for _, t in m.groupby("complex"):
        h, y = t["heur"].to_numpy(float), t[ziel].to_numpy(bool)
        perm = np.argsort(rng.random((B, len(h))), axis=1)
        for k in ks:
            idx = perm[:, :min(k, len(h))]
            aus[k].append(y[idx[np.arange(B), np.argmax(h[idx], axis=1)]].mean())
    return {k: 100 * float(np.mean(v)) for k, v in aus.items()}


daten = {}
print("=== Laden ===")
for k, tag in ZELLEN.items():
    m = lade(tag)
    daten[k] = m
    print(f"{k[0] + ', ' + str(k[1]) + ' Schr.':<20} {len(m):6d} Posen, "
          f"{m['complex'].nunique():3d} Komplexe, {m['seed'].nunique():3d} Seeds")

for ziel, titel in (("acc", "RMSD < 2 A"), ("valid", "PB-valid"),
                    ("beides", "RMSD<2 UND PB-valid")):
    print(f"\n=== {titel},  K = 40 ===")
    print(f"{'Zelle':<20} {'je Zug':>8} {'gnina':>8} {'Mixed':>8} {'Orakel':>8}")
    for k, m in daten.items():
        z = [100 * m[ziel].mean()] + [100 * wahl(m, ziel, w).mean()
                                      for w in ("gnina", "heuristik", "orakel")]
        print(f"{k[0] + ', ' + str(k[1]) + ' Schr.':<20} " + " ".join(f"{v:8.2f}" for v in z))

print("\n=== Separate minus Minimal, gepaart, 8000 Bootstrap-Ziehungen ===")
print(f"{'Schritte':<9} {'Kriterium':<12} {'Ranker':<11} {'Differenz':>11} "
      f"{'95%-Intervall':>20} {'p':>8}")
for nfe in (25, 5):
    for ziel, t in (("acc", "RMSD<2"), ("beides", "kombiniert")):
        for wie in ("heuristik", "orakel"):
            a = wahl(daten[("Minimal", nfe)], ziel, wie)
            b = wahl(daten[("Separate", nfe)], ziel, wie)
            d, lo, hi, p = boot(a, b)
            print(f"{nfe:<9} {t:<12} {wie:<11} {d:>+10.2f} pp  "
                  f"[{lo:+6.2f}, {hi:+6.2f}]  {p:>8.4f}")

print("\n=== Abfall von 25 auf 5 Schritte (Mixed Score) ===")
print(f"{'Arm':<10} {'RMSD<2':>24} {'kombiniert':>24}")
for arm in ("Minimal", "Separate"):
    aus = []
    for ziel in ("acc", "beides"):
        x = 100 * wahl(daten[(arm, 25)], ziel, "heuristik").mean()
        y = 100 * wahl(daten[(arm, 5)], ziel, "heuristik").mean()
        aus.append(f"{x:6.2f} -> {y:6.2f} ({y - x:+5.2f})")
    print(f"{arm:<10} {aus[0]:>24} {aus[1]:>24}")

KS = [1, 5, 10, 20, 40]
for ziel, titel in (("acc", "RMSD < 2"), ("beides", "RMSD<2 UND PB-valid")):
    print(f"\n=== Verlauf ueber K, Mixed Score, {titel} ===")
    print(f"{'Zelle':<20} " + " ".join(f"{k:>7}" for k in KS))
    for k, m in daten.items():
        c = top1_bei_k(m, ziel, KS)
        print(f"{k[0] + ', ' + str(k[1]) + ' Schr.':<20} " + " ".join(f"{c[x]:7.2f}" for x in KS))
