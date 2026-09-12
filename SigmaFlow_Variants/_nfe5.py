"""Rezept gegen 72 h, bei 25 UND 5 Integrationsschritten, PB308.

Alle Zellen werden auf die Seeds 0-39 beschraenkt. Die 72-h-Zelle bei fuenf
Schritten hat 140 Seeds; ohne die Beschraenkung waere ihr Top-1 bei einer
groesseren Ziehungszahl gemessen und der Vergleich schief.
"""
import glob
import pathlib

import numpy as np

quelle = pathlib.Path("vergleich_rezept.py").read_text(encoding="utf-8")
ns = {"__name__": "_defs", "__file__": "vergleich_rezept.py"}
exec(compile(quelle[:quelle.index("ZELLEN = {")], "vergleich_rezept.py", "exec"), ns)  # noqa: S102
zelle, wahl, boot = ns["zelle"], ns["wahl"], ns["boot"]


def g1(m):
    t = glob.glob(m)
    if not t:
        raise SystemExit(f"nicht gefunden: {m}")
    return t[0]


ZELLEN = {
    ("72 h", 25): ("_cmp/endpunkt_min_nfe25/*/posebusters_redock_curve/*/rd_*_seed*.csv",
                   g1("_cmp/endpunkt_min_nfe25/*/learning_curve_cpu/*/gnina_scores.csv")),
    ("Rezept", 25): ("_recipe/posebusters_redock/sigmaflow_minimal__recipe__confsampled/rd_*_seed*.csv",
                     g1("_recipe/GNINA-SCORE-*/gnina_scores_*.csv")),
    ("72 h", 5): ("_cmp/endpunkt_min_nfe5/*/*/*/*/*/posebusters_redock_curve/*/rd_*_seed*.csv",
                  g1("_cmp/endpunkt_min_nfe5/*/learning_curve_cpu/*/gnina_scores.csv")),
    ("Rezept", 5): ("_recipe5/posebusters_redock/sigmaflow_minimal__nfe5__recipe__confsampled/rd_*_seed*.csv",
                    g1("_recipe5/GNINA-SCORE-*/gnina_scores_*.csv")),
}

daten = {}
for k, (rg, gc) in ZELLEN.items():
    m = zelle(rg, gc)
    m = m[m["seed"] < 40]
    daten[k] = m

print(f"{'Zelle':<18} {'Posen':>7} {'Kompl.':>7} {'Seeds':>6}")
for k, m in daten.items():
    print(f"{k[0] + ', ' + str(k[1]) + ' Schr.':<18} {len(m):7d} "
          f"{m['complex'].nunique():7d} {m['seed'].nunique():6d}")

for ziel, titel in (("acc", "RMSD < 2"), ("beides", "RMSD<2 UND PB-valid")):
    print(f"\n=== {titel},  K = 40 ===")
    print(f"{'Zelle':<18} {'je Zug':>8} {'gnina':>8} {'Mixed':>8} {'Orakel':>8}")
    for k, m in daten.items():
        z = [100 * m[ziel].mean()] + [100 * wahl(m, ziel, w).mean()
                                      for w in ("gnina", "heuristik", "orakel")]
        print(f"{k[0] + ', ' + str(k[1]) + ' Schr.':<18} " + " ".join(f"{v:8.2f}" for v in z))

print("\n=== Abfall von 25 auf 5 Schritte (Mixed Score) ===")
print(f"{'Modell':<10} {'RMSD<2':>22} {'kombiniert':>22}")
for mod in ("72 h", "Rezept"):
    aus = []
    for ziel in ("acc", "beides"):
        a = 100 * wahl(daten[(mod, 25)], ziel, "heuristik").mean()
        b = 100 * wahl(daten[(mod, 5)], ziel, "heuristik").mean()
        aus.append(f"{a:6.2f} -> {b:6.2f} ({b-a:+5.2f})")
    print(f"{mod:<10} {aus[0]:>22} {aus[1]:>22}")

print("\n=== Rezept minus 72 h, gepaart, 8000 Bootstrap-Ziehungen ===")
print(f"{'Schritte':<10} {'Kriterium':<12} {'Differenz':>11} {'95%-Intervall':>20} {'p':>8}")
for nfe in (25, 5):
    for ziel, t in (("acc", "RMSD<2"), ("beides", "kombiniert")):
        a = wahl(daten[("72 h", nfe)], ziel, "heuristik")
        b = wahl(daten[("Rezept", nfe)], ziel, "heuristik")
        d, lo, hi, p = boot(a, b)
        print(f"{nfe:<10} {t:<12} {d:>+10.2f} pp  [{lo:+6.2f}, {hi:+6.2f}]  {p:>8.4f}")
