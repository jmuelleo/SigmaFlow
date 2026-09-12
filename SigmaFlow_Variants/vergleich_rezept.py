"""Rezept-Endpunkt gegen 72-h-Endpunkt, PB308, 25 Schritte, 40 Ziehungen.

Beide Zellen laufen durch DENSELBEN Code: Redock-Tabellen fuer die
PoseBusters-Pruefungen und das RMSD-Kriterium, gnina fuer Vinardo. Die
Heuristik wird aus dem Repository importiert (PB_CHECKS, BETA), damit es
keine zweite Fassung der Paperformel gibt.

WARUM KEIN per_pose.csv
    Die Redock-Tabelle traegt die Spalte `rmsd_<=_2A` und damit das
    Erfolgskriterium selbst. Der stetige RMSD-Wert kaeme aus evaluate_run und
    wird hier nicht gebraucht; er ist nur fuer Mediane noetig.

WARUM ALLE VERFUEGBAREN ZIEHUNGEN STATT EINES TEILMENGEN-SCHAETZERS
    Beide Zellen haben 40 Seeds, und Top-1 ueber alle vorhandenen Posen ist
    genau die Groesse, die die Arbeit bei K = 40 berichtet. Weil BEIDE Seiten
    identisch gerechnet werden, ist der Vergleich in sich stimmig, auch wenn
    er um Zehntel von den Tabellenwerten abweicht, die den 1000-Teilmengen-
    Schaetzer benutzen.
"""
import glob
import pathlib
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

REPO = pathlib.Path(".").resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "SigmaFlow_Variants" / "learning_curve_min"))
from pb_bool import als_bool  # noqa: E402
from SigmaFlow_Evaluation.ranking.heuristic_score import BETA, PB_CHECKS  # noqa: E402

LADE = {"mol_pred_loaded", "mol_true_loaded", "mol_cond_loaded"}


def zelle(redock_glob: str, gnina_csv: str) -> pd.DataFrame:
    teile = []
    for f in sorted(glob.glob(redock_glob)):
        d = pd.read_csv(f)
        d["seed"] = int(re.search(r"seed(\d+)\.csv$", f).group(1))
        teile.append(d)
    if not teile:
        sys.exit(f"keine Redock-Tabellen unter {redock_glob}")
    d = pd.concat(teile, ignore_index=True)
    d["complex"] = d["file"].map(lambda p: pathlib.Path(p).name.split("__")[0])

    rmsd_sp = next(c for c in d.columns if c.startswith("rmsd"))
    pruef = [c for c in d.columns
             if c not in LADE | {"file", "molecule", "position", "seed", "complex", rmsd_sp}]
    for c in pruef + [rmsd_sp]:
        d[c] = als_bool(d[c])

    d["valid"] = d[pruef].all(axis=1)
    d["p_pb"] = d[list(PB_CHECKS)].mean(axis=1)
    d["acc"] = d[rmsd_sp]

    gn = pd.read_csv(gnina_csv)
    dup = int(d.duplicated(["complex", "seed"]).sum())
    if dup:
        print(f"  HINWEIS: {dup} doppelte (complex, seed) in den Redock-Tabellen")
    m = d.merge(gn[["complex", "seed", "affinity"]], on=["complex", "seed"], how="inner")
    if len(m) != len(d):
        print(f"  HINWEIS: {len(d)} Redock-Zeilen, {len(m)} nach dem Merge mit gnina")

    m["beides"] = m["acc"] & m["valid"]
    m["heur"] = -m["affinity"] * (m["p_pb"] ** BETA)
    return m


def wahl(m: pd.DataFrame, ziel: str, wie: str) -> pd.Series:
    g = m.groupby("complex")
    if wie == "orakel":
        return g[ziel].any()
    if wie == "gnina":
        return g.apply(lambda t: t.loc[t["affinity"].idxmin(), ziel], include_groups=False).astype(bool)
    return g.apply(lambda t: t.loc[t["heur"].idxmax(), ziel], include_groups=False).astype(bool)


def boot(a: pd.Series, b: pd.Series, n: int = 8000, seed: int = 20260905):
    """Gepaarter Bootstrap ueber Komplexe auf der Differenz der Mittelwerte."""
    idx = b.index.intersection(a.index)
    x, y = a.loc[idx].to_numpy(float), b.loc[idx].to_numpy(float)
    d = y - x
    rng = np.random.default_rng(seed)
    zieh = rng.integers(0, len(d), size=(n, len(d)))
    verteilung = d[zieh].mean(axis=1)
    lo, hi = np.percentile(verteilung, [2.5, 97.5])
    p = 2 * min((verteilung <= 0).mean(), (verteilung >= 0).mean())
    return 100 * d.mean(), 100 * lo, 100 * hi, min(p, 1.0)


ZELLEN = {
    "72h Minimal 25/40": (
        "_cmp/endpunkt_min_nfe25/*/posebusters_redock_curve/*/rd_*_seed*.csv",
        glob.glob("_cmp/endpunkt_min_nfe25/*/learning_curve_cpu/*/gnina_scores.csv")[0],
    ),
    "Rezept Minimal 25/40": (
        "_recipe/posebusters_redock/sigmaflow_minimal__recipe__confsampled/rd_*_seed*.csv",
        glob.glob("_recipe/GNINA-SCORE-*/gnina_scores_*.csv")[0],
    ),
}

daten = {}
print("=== Laden ===")
for name, (rg, gc) in ZELLEN.items():
    m = zelle(rg, gc)
    daten[name] = m
    print(f"{name:<24} {len(m):6d} Posen, {m['complex'].nunique():3d} Komplexe, "
          f"{m['seed'].nunique():3d} Seeds")

for ziel, titel in (("acc", "RMSD < 2 A"), ("valid", "PB-valid"), ("beides", "RMSD<2 UND PB-valid")):
    print(f"\n=== {titel} ===")
    print(f"{'Zelle':<24} {'je Zug':>8} {'gnina':>8} {'Heuristik':>10} {'Orakel':>8}")
    for name, m in daten.items():
        zeile = [100 * m[ziel].mean()]
        for wie in ("gnina", "heuristik", "orakel"):
            zeile.append(100 * wahl(m, ziel, wie).mean())
        print(f"{name:<24} " + " ".join(f"{v:8.2f}" if i != 2 else f"{v:10.2f}"
                                        for i, v in enumerate(zeile)))

print("\n=== Rezept minus 72 h, gepaart je Komplex, 8000 Bootstrap-Ziehungen ===")
print(f"{'Kriterium':<22} {'Ranker':<12} {'Differenz':>10} {'95%-Intervall':>22} {'p':>8}")
for ziel, titel in (("acc", "RMSD<2"), ("beides", "kombiniert")):
    for wie in ("gnina", "heuristik", "orakel"):
        a = wahl(daten["72h Minimal 25/40"], ziel, wie)
        b = wahl(daten["Rezept Minimal 25/40"], ziel, wie)
        diff, lo, hi, p = boot(a, b)
        print(f"{titel:<22} {wie:<12} {diff:>+9.2f} pp   [{lo:+6.2f}, {hi:+6.2f}]   {p:>8.4f}")


# ---------------------------------------------------------------- Verlauf ueber K
def top1_bei_k(m: pd.DataFrame, ziel: str, ks, B: int = 1000, seed: int = 20260905):
    """Top-1 nach dem Mixed Score bei K Ziehungen, Erwartung ueber K-Teilmengen.

    Eine Permutationsmatrix je Komplex bedient alle K: die ersten k Spalten
    einer gleichverteilten Permutation sind eine gleichverteilte k-Teilmenge.
    """
    rng = np.random.default_rng(seed)
    aus = {k: [] for k in ks}
    for _, t in m.groupby("complex"):
        h = t["heur"].to_numpy(float)
        y = t[ziel].to_numpy(bool)
        n = len(h)
        perm = np.argsort(rng.random((B, n)), axis=1)
        for k in ks:
            kk = min(k, n)
            idx = perm[:, :kk]
            bester = idx[np.arange(B), np.argmax(h[idx], axis=1)]
            aus[k].append(y[bester].mean())
    return {k: 100 * float(np.mean(v)) for k, v in aus.items()}


KS = [1, 5, 10, 20, 40]
for ziel, titel in (("acc", "RMSD < 2"), ("beides", "RMSD<2 UND PB-valid")):
    print(f"\n=== Verlauf ueber K, Mixed Score, {titel} ===")
    print(f"{'Zelle':<24} " + " ".join(f"{k:>7}" for k in KS))
    kurven = {}
    for name, m in daten.items():
        c = top1_bei_k(m, ziel, KS)
        kurven[name] = c
        print(f"{name:<24} " + " ".join(f"{c[k]:7.2f}" for k in KS))
    a, b = list(kurven.values())
    print(f"{'Differenz':<24} " + " ".join(f"{b[k]-a[k]:+7.2f}" for k in KS))
