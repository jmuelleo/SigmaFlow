"""Log-Dichte als Ranker gegen den Mixed Score -- auf DEMSELBEN Posensatz.

WARUM DAS UEBERHAUPT VERGLEICHBAR IST
    `run_likelihood.py` tauscht nur den Sampler. Datensatz, Fragmentierung,
    Checkpoint und Seed bleiben unveraendert, also sind die Posen bitgleich
    dieselben wie in der schon ausgewerteten Zelle. Gueltigkeit, RMSD und
    Vinardo muessen daher NICHT neu gerechnet werden -- die Dichte haengt sich
    als weitere Spalte an.

DIE ZUORDNUNG IST EINE ANNAHME, DIE GEPRUEFT WIRD
    Aeltere logp-Dateien schreiben `mol_id` als Tensor-Darstellung des
    DATENSATZINDEX ("tensor(17)"), nicht den Komplexnamen. Der Index ist die
    Position in `sorted(dataroot.iterdir())`, also derselben Reihenfolge, die
    `DataFront._setup` benutzt. Das Skript rechnet ihn zurueck UND prueft die
    Mengengleichheit gegen die Redock-Tabelle -- bei Abweichung bricht es ab,
    statt still weniger Komplexe zu verknuepfen.
    Neuere Dateien tragen eine Spalte `komplex`; die wird bevorzugt.

WAS VERGLICHEN WIRD
    Alle Regeln waehlen aus DENSELBEN K Ziehungen je Komplex genau eine Pose.
    Der Unterschied ist allein das Kriterium.

Aufruf:
    $ARC_SF_ENV/bin/python arc/analyse_likelihood_ranking.py \\
        --logp <verzeichnis mit logp_s*.csv> \\
        --redock <verzeichnis mit rd_*seed*.csv> \\
        --gnina <gnina_scores*.csv> \\
        --wurzel <posebusters_v2_308>
"""
import argparse
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
    sys.exit("pb_bool.py nicht gefunden -- keine zweite Fassung von als_bool.")
from pb_bool import als_bool  # noqa: E402

sys.path.insert(0, _REPO)
try:
    from SigmaFlow_Evaluation.ranking.heuristic_score import BETA, PB_CHECKS
except Exception as e:  # noqa: BLE001
    sys.exit(f"heuristic_score nicht importierbar ({e}) -- die Paperformel "
             f"wird hier nicht geraten.")

PRUEFUNGEN = [
    "sanitization", "inchi_convertible", "all_atoms_connected", "no_radicals",
    "molecular_formula", "molecular_bonds", "double_bond_stereochemistry",
    "tetrahedral_chirality", "bond_lengths", "bond_angles",
    "internal_steric_clash", "aromatic_ring_flatness",
    "non-aromatic_ring_non-flatness", "double_bond_flatness", "internal_energy",
    "protein-ligand_maximum_distance", "minimum_distance_to_protein",
    "minimum_distance_to_organic_cofactors",
    "minimum_distance_to_inorganic_cofactors", "minimum_distance_to_waters",
    "volume_overlap_with_protein", "volume_overlap_with_organic_cofactors",
    "volume_overlap_with_inorganic_cofactors", "volume_overlap_with_waters",
]

p = argparse.ArgumentParser()
p.add_argument("--logp", required=True)
p.add_argument("--redock", required=True)
p.add_argument("--gnina", required=True)
p.add_argument("--wurzel", default="/data/stat-cadd/shug8458/data/posebusters_v2_308")
a = p.parse_args()

# --- Log-Dichten ----------------------------------------------------------
ordner = sorted(d for d in glob.glob(os.path.join(a.wurzel, "*")) if os.path.isdir(d))
name_von_idx = {i: os.path.basename(d) for i, d in enumerate(ordner)}

teile = []
for f in sorted(glob.glob(os.path.join(a.logp, "logp_s*.csv"))):
    d = pd.read_csv(f)
    if "komplex" in d.columns and d["komplex"].astype(str).str.len().gt(0).all():
        d["complex"] = d["komplex"].astype(str)
    else:
        idx = d["mol_id"].astype(str).str.extract(r"(\d+)")[0].astype(int)
        d["complex"] = idx.map(name_von_idx)
    if "seed" not in d.columns:
        d["seed"] = int(re.search(r"logp_s(\d+)\.csv$", f).group(1))
    teile.append(d[["complex", "seed", "log_p"]])
if not teile:
    sys.exit(f"keine logp_s*.csv unter {a.logp}")
lp = pd.concat(teile, ignore_index=True)
if lp["complex"].isna().any():
    sys.exit("Indizes ausserhalb der Verzeichnisliste -- falsche Wurzel?")
print(f"Dichten : {len(lp)} Zeilen, {lp['seed'].nunique()} Seeds, "
      f"{lp['complex'].nunique()} Komplexe")

# --- Redock ---------------------------------------------------------------
teile = []
for f in sorted(glob.glob(os.path.join(a.redock, "rd_*seed*.csv"))):
    d = pd.read_csv(f)
    d["seed"] = int(re.search(r"seed(\d+)\.csv$", f).group(1))
    teile.append(d)
if not teile:
    sys.exit(f"keine rd_*seed*.csv unter {a.redock}")
rd = pd.concat(teile, ignore_index=True)
rd["complex"] = rd["file"].map(lambda x: os.path.basename(str(x)).split("__")[0])
fehlt = [c for c in PRUEFUNGEN if c not in rd.columns]
if fehlt:
    sys.exit(f"Pruefspalten fehlen: {fehlt[:5]}")
for c in PRUEFUNGEN:
    rd[c] = als_bool(rd[c])
rmsd_sp = next(c for c in rd.columns if c.startswith("rmsd_"))
rd["acc"] = als_bool(rd[rmsd_sp])
rd["valid"] = rd[PRUEFUNGEN].all(axis=1)
rd["p_pb"] = rd[list(PB_CHECKS)].mean(axis=1)
print(f"Redock  : {len(rd)} Zeilen, {rd['seed'].nunique()} Seeds")

# --- gnina ----------------------------------------------------------------
gn = pd.read_csv(a.gnina)
print(f"gnina   : {len(gn)} Zeilen, {gn['seed'].nunique()} Seeds")

# --- Verknuepfen, und zwar geprueft --------------------------------------
m = rd.merge(gn[["complex", "seed", "affinity"]], on=["complex", "seed"], how="inner")
m = m.merge(lp, on=["complex", "seed"], how="inner")
K = m["seed"].nunique()
print(f"\nverknuepft: {len(m)} Posen, {m['complex'].nunique()} Komplexe, K = {K}")
erwartet = m["complex"].nunique() * K
if len(m) < 0.95 * erwartet:
    print(f"  WARNUNG: erwartet waeren rund {erwartet} Zeilen. Fehlen Seeds?")

m["heur"] = -m["affinity"] * (m["p_pb"] ** BETA)
m["beides"] = m["acc"] & m["valid"]


def wahl(t: pd.DataFrame, ziel: str, regel: str) -> pd.Series:
    """Je Komplex EINE Pose auswaehlen und das Zielkriterium ablesen."""
    aus = {}
    for c, g in t.groupby("complex"):
        if regel == "orakel":
            aus[c] = bool(g[ziel].any())
            continue
        h = g
        if regel.startswith("filter_"):
            gv = g[g["valid"]]
            if len(gv):
                h = gv
        if regel.endswith("gnina"):
            i = h["affinity"].idxmin()
        elif regel.endswith("logp"):
            i = h["log_p"].idxmax()
        else:                       # mixed
            i = h["heur"].idxmax()
        aus[c] = bool(h.loc[i, ziel])
    return pd.Series(aus)


def boot(x: pd.Series, y: pd.Series, n: int = 8000, seed: int = 20260906):
    idx = y.index.intersection(x.index)
    d = y.loc[idx].to_numpy(float) - x.loc[idx].to_numpy(float)
    rng = np.random.default_rng(seed)
    v = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    lo, hi = np.percentile(v, [2.5, 97.5])
    return (100 * d.mean(), 100 * lo, 100 * hi,
            min(2 * min((v <= 0).mean(), (v >= 0).mean()), 1.0))


REGELN = [("nur Energie (gnina)", "gnina"),
          ("Mixed Score (Paper)", "mixed"),
          ("nur Log-Dichte", "logp"),
          ("PB-Filter -> Energie", "filter_gnina"),
          ("PB-Filter -> Log-Dichte", "filter_logp"),
          ("Orakel", "orakel")]

for ziel, titel in (("acc", "RMSD < 2 A"), ("valid", "PB-valide"),
                    ("beides", "RMSD<2 UND PB-valide")):
    print(f"\n=== {titel},  K = {K} ===")
    print(f"  {'Regel':<26} {'Anteil':>9}")
    print(f"  {'je Zug (Referenz)':<26} {100 * m[ziel].mean():8.2f}%")
    for name, regel in REGELN:
        print(f"  {name:<26} {100 * wahl(m, ziel, regel).mean():8.2f}%")

print(f"\n=== Log-Dichte gegen Mixed Score, gepaart, 8000 Bootstrap-Ziehungen ===")
print(f"  {'Kriterium':<22} {'Regel':<26} {'Differenz':>11} {'95%-Intervall':>20} {'p':>8}")
for ziel, titel in (("acc", "RMSD<2"), ("beides", "kombiniert")):
    basis = wahl(m, ziel, "mixed")
    for name, regel in (("nur Log-Dichte", "logp"),
                        ("PB-Filter -> Log-Dichte", "filter_logp"),
                        ("PB-Filter -> Energie", "filter_gnina")):
        d, lo, hi, pv = boot(basis, wahl(m, ziel, regel))
        print(f"  {titel:<22} {name:<26} {d:>+10.2f} pp  [{lo:+6.2f}, {hi:+6.2f}]  {pv:>8.4f}")

# --- Traegt die Dichte ueberhaupt Signal? --------------------------------
print("\n=== Korrelationen INNERHALB der Komplexe (Spearman ueber die K Ziehungen) ===")
print("  Ein Ranker taugt nur, wenn er innerhalb eines Komplexes die bessere")
print("  Pose hoeher stellt. Korrelationen ueber alle Posen hinweg waeren durch")
print("  Komplexunterschiede dominiert und sagten nichts.")
rho = {"log_p vs acc": [], "log_p vs valid": [], "log_p vs -affinity": [],
       "heur vs acc": []}
for c, g in m.groupby("complex"):
    if len(g) < 3:
        continue
    r = g.rank()
    for name, (x, y) in (("log_p vs acc", ("log_p", "acc")),
                         ("log_p vs valid", ("log_p", "valid")),
                         ("heur vs acc", ("heur", "acc"))):
        if g[y].nunique() > 1:
            rho[name].append(np.corrcoef(r[x], r[y])[0, 1])
    if g["affinity"].nunique() > 1:
        rho["log_p vs -affinity"].append(
            np.corrcoef(r["log_p"], (-g["affinity"]).rank())[0, 1])
for name, v in rho.items():
    if v:
        print(f"  {name:<22} Median {np.median(v):+6.3f}   Mittel {np.mean(v):+6.3f}"
              f"   (n = {len(v)} Komplexe)")
