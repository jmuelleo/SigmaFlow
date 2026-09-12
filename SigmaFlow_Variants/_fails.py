"""Woran scheitern genaue, aber ungueltige Posen?

Betrachtet die nach dem Mixed Score gewaehlte Pose je Komplex, schraenkt auf
die Faelle mit RMSD < 2 aber PB-ungueltig ein und zaehlt, welche der
Einzelpruefungen dort durchfallen.
"""
import glob
import pathlib

import pandas as pd

quelle = pathlib.Path("vergleich_rezept.py").read_text(encoding="utf-8")
ns = {"__name__": "_defs", "__file__": "vergleich_rezept.py"}
exec(compile(quelle[:quelle.index("ZELLEN = {")], "vergleich_rezept.py", "exec"), ns)  # noqa: S102
zelle, LADE = ns["zelle"], ns["LADE"]


def g1(m):
    t = glob.glob(m)
    if not t:
        raise SystemExit(f"nicht gefunden: {m}")
    return t[0]


C = "_cmp"
ZELLEN = {
    "SigmaDock 72h": (f"{C}/sd_endpunkt_40seeds/*/posebusters_redock_curve/*nfe25*/rd_*_seed*.csv",
                      g1(f"{C}/sd_endpunkt_40seeds/*/learning_curve_cpu/*nfe25*/gnina_scores.csv")),
    "Minimal 72h":   (f"{C}/endpunkt_min_nfe25/*/posebusters_redock_curve/*/rd_*_seed*.csv",
                      g1(f"{C}/endpunkt_min_nfe25/*/learning_curve_cpu/*/gnina_scores.csv")),
    "Minimal Rezept": ("_recipe/posebusters_redock/sigmaflow_minimal__recipe__confsampled/rd_*_seed*.csv",
                       g1("_recipe/GNINA-SCORE-*/gnina_scores_*.csv")),
}

for name, (rg, gc) in ZELLEN.items():
    m = zelle(rg, gc)
    rmsd_sp = next(c for c in m.columns if c.startswith("rmsd"))
    pruef = [c for c in m.columns
             if c not in LADE | {"file", "molecule", "position", "seed", "complex",
                                 rmsd_sp, "valid", "p_pb", "acc", "beides",
                                 "affinity", "heur"}]

    # die vom Mixed Score gewaehlte Pose je Komplex
    sel = m.loc[m.groupby("complex")["heur"].idxmax()]
    schlecht = sel[sel["acc"] & ~sel["valid"]]

    print(f"\n########## {name}")
    print(f"  gewaehlte Posen: {len(sel)}   davon genau: {int(sel['acc'].sum())}   "
          f"davon genau ABER ungueltig: {len(schlecht)}")
    if len(schlecht) == 0:
        continue
    quote = (~schlecht[pruef]).mean().sort_values(ascending=False)
    quote = quote[quote > 0]
    print(f"  {'Pruefung':<42} {'faellt durch':>12}")
    for k, v in quote.items():
        print(f"  {k:<42} {100*v:11.1f}%")

    # zum Vergleich: Durchfallquote ueber ALLE Posen
    alle = (~m[pruef]).mean().sort_values(ascending=False)
    alle = alle[alle > 0.005]
    print(f"  --- ueber alle {len(m)} Posen, Durchfallquote > 0,5 % ---")
    for k, v in alle.items():
        print(f"  {k:<42} {100*v:11.1f}%")
