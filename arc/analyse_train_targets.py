"""Sind die tatsaechlichen Trainingsziele PB-valide -- und was bewirkt pb_check?

Liest die Tabellen aus `arc/train_targets_pb.slurm` fuer beide Arme
(pb_check false und true) und vergleicht sie GEPAART je Komplex.

WARUM DIE PRUEFLISTE IMPORTIERT WIRD
    `arc/analyse_crystal_pb.py` definiert PRUEFUNGEN, LAGE und INTERN. Eine
    zweite Fassung waere genau der Fehler, der am 2026-09-05 zu "0 % bestanden"
    gefuehrt hat (Diagnosespalten mit umgekehrter Bedeutung).

DIE ENTSCHEIDENDE AUFTEILUNG
    pb_check wirkt NUR auf neu ausgerichtete Ziele. Wo `ausgerichtet` falsch
    ist, steht die rohe Kristallpose, und beide Arme muessen dort identisch
    sein. Weichen sie doch ab, stimmt die Paarung nicht -- das wird geprueft,
    nicht angenommen.

Aufruf:
    $ARC_SF_ENV/bin/python arc/analyse_train_targets.py
    $ARC_SF_ENV/bin/python arc/analyse_train_targets.py /pfad/zu/train_targets_pb
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

_HIER = os.path.dirname(os.path.abspath(__file__))
quelle = open(os.path.join(_HIER, "analyse_crystal_pb.py"), encoding="utf-8").read()
ns = {"__name__": "_defs", "__file__": os.path.join(_HIER, "analyse_crystal_pb.py")}
sys.argv = [sys.argv[0]] + sys.argv[1:2]
exec(compile(quelle[:quelle.index("saetze = sorted")], "analyse_crystal_pb.py", "exec"), ns)  # noqa: S102
als_bool, PRUEFUNGEN, LAGE, INTERN = ns["als_bool"], ns["PRUEFUNGEN"], ns["LAGE"], ns["INTERN"]

WURZEL = sys.argv[1] if len(sys.argv) > 1 else \
    "/data/stat-cadd/shug8458/arc_runs/train_targets_pb"

# Arme nicht fest verdrahten: es koennen mehr werden (pb_check aus, pb_check
# an, volles Rezept, ...). Was da ist, wird ausgewertet.
ARME = {}
for _d in sorted(glob.glob(os.path.join(WURZEL, "*"))):
    if not os.path.isdir(_d) or not glob.glob(os.path.join(_d, "ziele_*.csv")):
        continue
    _n = os.path.basename(_d)
    _label = {"pdbbind-general_pbfalse": "pb_check aus",
              "pdbbind-general_pbtrue": "pb_check an",
              "pdbbind-general_recipe": "volles Rezept"}.get(_n, _n)
    ARME[_label] = _n
if not ARME:
    sys.exit(f"keine Arme unter {WURZEL}")


def lade(tag: str) -> pd.DataFrame:
    teile = sorted(glob.glob(os.path.join(WURZEL, tag, "ziele_*.csv")))
    if not teile:
        sys.exit(f"keine Tabellen unter {os.path.join(WURZEL, tag)}")
    t = pd.concat([pd.read_csv(f) for f in teile], ignore_index=True)
    fehlt = [c for c in PRUEFUNGEN if c not in t.columns]
    if fehlt:
        sys.exit(f"{tag}: Pruefspalten fehlen: {fehlt[:5]}")
    w = pd.DataFrame({c: als_bool(t[c]) for c in PRUEFUNGEN})
    w["ALLE"] = w.all(axis=1)
    w["LAGE"] = w[LAGE].all(axis=1)
    w["INTERN"] = w[INTERN].all(axis=1)
    w["ausgerichtet"] = t["ausgerichtet"].astype(str).str.lower().eq("true")
    w["rmsd"] = t["alignment_rmsd"].astype(float)
    w.index = t["komplex"]
    w.attrs["n_teile"] = len(teile)
    return w


daten = {k: lade(v) for k, v in ARME.items()}

print(f"{'Arm':<16} {'Ziele':>7} {'Bloecke':>8} {'ausgerichtet':>13}")
for k, w in daten.items():
    print(f"{k:<16} {len(w):7d} {w.attrs['n_teile']:8d} "
          f"{100 * w['ausgerichtet'].mean():12.2f}%")

print(f"\n{'Arm':<16} {'ALLE 24':>9} {'nur LAGE':>10} {'nur INTERN':>11} {'beides':>8}")
for k, w in daten.items():
    lg, it = w["LAGE"], w["INTERN"]
    print(f"{k:<16} {100 * w['ALLE'].mean():8.2f}% {100 * (~lg & it).mean():9.2f}% "
          f"{100 * (lg & ~it).mean():10.2f}% {100 * (~lg & ~it).mean():7.2f}%")

print("\n=== Aufgeteilt nach ausgerichtet / roh (pb_check wirkt NUR auf ausgerichtete) ===")
print(f"{'Arm':<16} {'ausgerichtet':>13} {'roh (Kristall)':>16}")
for k, w in daten.items():
    a, r = w[w["ausgerichtet"]], w[~w["ausgerichtet"]]
    print(f"{k:<16} {100 * a['ALLE'].mean():12.2f}% ({len(a):5d}) "
          f"{100 * r['ALLE'].mean():11.2f}% ({len(r):5d})")

# --- Paarung pruefen, nicht annehmen -------------------------------------
if "pb_check aus" not in daten or "pb_check an" not in daten:
    sys.exit("Fuer die Paarungspruefung fehlen die Arme "
             "'pb_check aus'/'pb_check an'.")
a, b = daten["pb_check aus"], daten["pb_check an"]
gem = a.index.intersection(b.index)
a, b = a.loc[gem], b.loc[gem]
roh = ~a["ausgerichtet"] & ~b["ausgerichtet"]
abw = int((a.loc[roh, "ALLE"] != b.loc[roh, "ALLE"]).sum())
print(f"\ngemeinsame Komplexe: {len(gem)}   beide roh: {int(roh.sum())}")
print(f"davon mit ABWEICHENDEM Ergebnis: {abw}"
      f"{'  <- Paarung stimmt' if abw == 0 else '  <- WARNUNG: Paarung stimmt NICHT'}")

print("\n=== Wirkung von pb_check, gepaart ueber dieselben Komplexe ===")
for name, sp in (("alle 24 Pruefungen", "ALLE"), ("nur ligandintern", "INTERN"),
                 ("nur die Lage", "LAGE")):
    x, y = a[sp].to_numpy(), b[sp].to_numpy()
    d = y.astype(float) - x.astype(float)
    rng = np.random.default_rng(20260905)
    v = d[rng.integers(0, len(d), size=(8000, len(d)))].mean(axis=1)
    lo, hi = np.percentile(v, [2.5, 97.5])
    p = min(2 * min((v <= 0).mean(), (v >= 0).mean()), 1.0)
    print(f"{name:<20} {100 * x.mean():7.2f}% -> {100 * y.mean():7.2f}%  "
          f"{100 * d.mean():+6.2f} pp  [{100 * lo:+5.2f}, {100 * hi:+5.2f}]  p={p:.4f}")
    print(f"{'':<20}   an aus: {int(((~x.astype(bool)) & y.astype(bool)).sum()):5d} "
          f"repariert, {int((x.astype(bool) & (~y.astype(bool))).sum()):5d} verschlechtert")
