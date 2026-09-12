"""Erst filtern, dann mit der Paper-Heuristik ranken.

Der Mixed Score gewichtet p_pb weich. Ein harter Filter wirft Posen ganz
hinaus. Getestet werden drei Filterstufen, danach wird IMMER mit der
unveraenderten Paperformel s = -b * p^4 ueber die fuenf Paper-Checks geranked.

RUECKFALL
    Bleibt fuer einen Komplex keine Pose uebrig, wird ungefiltert geranked --
    sonst wuerde der Komplex stillschweigend verschwinden und die Quote
    kuenstlich steigen. Die Zahl dieser Faelle wird ausgewiesen.
"""
import glob
import pathlib

import pandas as pd

quelle = pathlib.Path("vergleich_rezept.py").read_text(encoding="utf-8")
ns = {"__name__": "_defs", "__file__": "vergleich_rezept.py"}
exec(compile(quelle[:quelle.index("ZELLEN = {")], "vergleich_rezept.py", "exec"), ns)  # noqa: S102
zelle = ns["zelle"]

PROT = ["minimum_distance_to_protein"]
DIST = PROT + ["minimum_distance_to_organic_cofactors",
               "minimum_distance_to_inorganic_cofactors"]


def g1(m):
    t = glob.glob(m)
    if not t:
        raise SystemExit(m)
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

FILTER = {
    "ohne Filter": None,
    "Protein-Abstand": PROT,
    "alle 3 Abstaende": DIST,
    "voll PB-valid": "valid",
}

for ziel, titel in (("beides", "RMSD<2 UND PB-valid"), ("acc", "RMSD<2")):
    print(f"\n########## {titel}")
    print(f"{'Modell':<16} " + " ".join(f"{k:>18}" for k in FILTER) + f" {'Orakel':>8}")
    for name, (rg, gc) in ZELLEN.items():
        m = zelle(rg, gc)
        zeile, leere = [], []
        for lbl, spalten in FILTER.items():
            if spalten is None:
                maske = pd.Series(True, index=m.index)
            elif spalten == "valid":
                maske = m["valid"]
            else:
                maske = m[spalten].all(axis=1)
            # Rueckfall je Komplex, wenn der Filter alles wegnimmt
            hat = maske.groupby(m["complex"]).transform("any")
            benutzt = maske | ~hat
            leere.append(int((~hat).groupby(m["complex"]).first().sum()))
            sub = m[benutzt]
            gew = sub.loc[sub["heur"].groupby(sub["complex"]).idxmax()]
            zeile.append(100 * gew[ziel].mean())
        orakel = 100 * m.groupby("complex")[ziel].any().mean()
        print(f"{name:<16} " + " ".join(f"{v:18.2f}" for v in zeile) + f" {orakel:8.2f}")
        print(f"{'   Komplexe ohne Ueberlebende:':<16} "
              + " ".join(f"{v:18d}" for v in leere))
