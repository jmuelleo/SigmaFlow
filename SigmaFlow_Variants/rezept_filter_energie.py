"""Erst auf PB-Validitaet filtern, dann nach Energie ranken -- Rezept-Endpunkte.

WARUM DIE LADEFUNKTION IMPORTIERT WIRD
    `arc/analyse_recipe_cells.py` baut die Zellen. Diese Datei fuehrt dessen
    Kopf bis `daten = {}` aus und benutzt `lade`, `wahl` und `boot`. So gibt es
    keine zweite Fassung der Bool-Konvention, der Paperformel oder des
    Bootstrap -- derselbe Kniff wie in vergleich_rezept.py.

DIE ENTSCHEIDENDE DESIGNFRAGE
    Manche Komplexe haben unter 40 Ziehungen KEINE einzige gueltige Pose. Eine
    Filterregel muss sagen, was dann geschieht. Beide Lesarten stehen unten
    nebeneinander, weil die Wahl das Ergebnis verschiebt:
        mit Rueckfall  -- ist keine gueltig, nimm die energetisch beste von
                          allen. Das ist, was ein Anwender taete.
        ohne Rueckfall -- ist keine gueltig, gilt der Komplex als Fehlschlag.
                          Das ist die strengere, ehrlichere Zahl.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
quelle = (REPO / "arc" / "analyse_recipe_cells.py").read_text(encoding="utf-8")
kopf = quelle[:quelle.index("daten = {}")]
ns = {"__name__": "_defs", "__file__": str(REPO / "arc" / "analyse_recipe_cells.py")}
sys.argv = ["x", str(REPO / "SigmaFlow_Variants" / "_recipe_all")]
exec(compile(kopf, "analyse_recipe_cells.py", "exec"), ns)  # noqa: S102
lade, wahl, boot, ZELLEN = ns["lade"], ns["wahl"], ns["boot"], ns["ZELLEN"]

daten = {k: lade(t) for k, t in ZELLEN.items()}


def regel(m: pd.DataFrame, ziel: str, wie: str, rueckfall: bool = True) -> pd.Series:
    """Erst filtern, dann ranken. `wie` = 'gnina' (Energie) oder 'mixed'."""
    aus = {}
    for c, t in m.groupby("complex"):
        g = t[t["valid"]]
        if len(g) == 0:
            if not rueckfall:
                aus[c] = False
                continue
            g = t
        i = g["affinity"].idxmin() if wie == "gnina" else g["heur"].idxmax()
        aus[c] = bool(g.loc[i, ziel])
    return pd.Series(aus)


print(f"{'Zelle':<20} {'Komplexe':>9} {'mit >=1 gueltigen':>18}")
for k, m in daten.items():
    hat = m.groupby("complex")["valid"].any()
    print(f"{k[0] + ', ' + str(k[1]) + ' Schr.':<20} {len(hat):9d} "
          f"{100 * hat.mean():17.2f}%")

REGELN = [
    ("nur Energie (gnina)", lambda m, z: wahl(m, z, "gnina")),
    ("Mixed Score (Paper)", lambda m, z: wahl(m, z, "heuristik")),
    ("PB-Filter -> Energie", lambda m, z: regel(m, z, "gnina", True)),
    ("PB-Filter -> Energie, ohne Rueckfall", lambda m, z: regel(m, z, "gnina", False)),
    ("PB-Filter -> Mixed Score", lambda m, z: regel(m, z, "mixed", True)),
    ("Orakel", lambda m, z: wahl(m, z, "orakel")),
]

for ziel, titel in (("acc", "RMSD < 2 A"), ("beides", "RMSD<2 UND PB-valid")):
    print(f"\n=== {titel},  K = 40 ===")
    kopfz = f"{'Regel':<38}" + "".join(f"{k[0][:4] + ' ' + str(k[1]):>11}" for k in daten)
    print(kopfz)
    for name, f in REGELN:
        print(f"{name:<38}" + "".join(f"{100 * f(m, ziel).mean():11.2f}" for m in daten.values()))

print("\n=== Separate, 25 Schritte: Filterregel gegen Mixed Score ===")
print(f"{'Kriterium':<12} {'Regel':<38} {'Differenz':>11} {'95%-Intervall':>20} {'p':>8}")
m = daten[("Separate", 25)]
for ziel, t in (("acc", "RMSD<2"), ("beides", "kombiniert")):
    basis = wahl(m, ziel, "heuristik")
    for name, f in REGELN[2:5]:
        d, lo, hi, p = boot(basis, f(m, ziel))
        print(f"{t:<12} {name:<38} {d:>+10.2f} pp  [{lo:+6.2f}, {hi:+6.2f}]  {p:>8.4f}")

print("\n=== Separate gegen Minimal UNTER DER FILTERREGEL (mit Rueckfall) ===")
print(f"{'Schritte':<9} {'Kriterium':<12} {'Differenz':>11} {'95%-Intervall':>20} {'p':>8}")
for nfe in (25, 5):
    for ziel, t in (("acc", "RMSD<2"), ("beides", "kombiniert")):
        a = regel(daten[("Minimal", nfe)], ziel, "gnina", True)
        b = regel(daten[("Separate", nfe)], ziel, "gnina", True)
        d, lo, hi, p = boot(a, b)
        print(f"{nfe:<9} {t:<12} {d:>+10.2f} pp  [{lo:+6.2f}, {hi:+6.2f}]  {p:>8.4f}")
