"""PB308-Leistung der 72-h-Endpunkte, aufgeschluesselt nach FRAGMENTZAHL.

WAS HIER ANDERS IST ALS IN DEN BISHERIGEN TABELLEN
    Bisher wurde ueber alle 307 Komplexe gemittelt. Hier wird zusaetzlich nach
    der Zahl der Fragmente stratifiziert, in die SigmaDocks Fragmentierung den
    Liganden zerlegt -- die Groesse, die den Zustandsraum des generativen
    Prozesses bestimmt (je Fragment eine Rotation und eine Translation).

    Ausserdem kommt eine zweite Auswahlregel dazu: erst VOLLSTAENDIG auf
    PB-Validitaet filtern, dann unter den Ueberlebenden nach gnina-Energie
    waehlen -- statt des Mixed Score, der Validitaet nur als weichen Faktor
    p^4 einrechnet.

DIE ZELLEN STEHEN FEST VERDRAHTET, UND ZWAR MIT ABSICHT
    In `final200/.../nfe5__sampled/` liegen ZWEI gnina-Tabellen nebeneinander:
    `gnina_scores.csv` mit 140 Seeds und `gnina_scores_200.csv` mit 200. Ein
    Glob wuerde je nach Sortierung die falsche nehmen, und der Fehler waere an
    den Zahlen nicht zu sehen. Genau das ist hier schon einmal passiert
    (SigmaDock las 59,61 statt 69,71). Deshalb: jede Zelle mit explizitem
    Pfad, und beim Start wird geprueft, dass die Seedzahl zur Erwartung passt.

ZWEI DATENMACKEN, DIE BEHANDELT WERDEN MUESSEN
    1. Je Seed liegen 308 Redock-Zeilen, aber nur 307 Komplexe vor: einer ist
       doppelt (seedabhaengig, z. B. 7X5N_5M5 bei seed 0), einer fehlt
       (7XPO_UPG ueberall). gnina hat 307. Ein Merge auf (complex, seed) wuerde
       den doppelten still verdoppeln. Es wird deduplizert -- erster Eintrag
       nach `file`-Sortierung -- und die Zahl protokolliert.
    2. PB-Pruefspalten duerfen nie mit == "true" gelesen werden; faellt ein
       Modul aus, steht dort NaN oder 1.0. Dafuer ist `pb_bool.als_bool` da.

DIE FRAGMENTZAHL IST EINE NAEHERUNG
    `fragmentzahl.py` rechnet sie nach, weil sie nirgends protokolliert wurde.
    Die Fragmentierung ist stochastisch ("random"); rund 5 % der Komplexe
    liefern ueber Ziehungen nicht immer dieselbe Zahl. Diese Komplexe sind in
    der Spalte `stabil` markiert und werden per Schalter aus- oder
    eingeschlossen. Das gehoert in jede Bildunterschrift.

Aufruf:
    python SigmaFlow_Variants/nach_fragmenten.py
    python SigmaFlow_Variants/nach_fragmenten.py --nur-stabil
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

# Zellendefinitionen, Pruefspalten und der Lader stehen in zellen.py --
# EINMAL, damit die fest verdrahteten gnina-Pfade nicht in zwei
# Skripten auseinanderlaufen koennen.
sys.path.insert(0, _HIER)
from zellen import SAETZE, lade_zelle  # noqa: E402


REGELN = [
    ("je Zug", "zug"),
    ("nur gnina", "gnina"),
    ("Mixed Score (Paper)", "mixed"),
    ("PB-Filter -> gnina", "filter_gnina"),
    ("Orakel", "orakel"),
]
ZIELE = [("acc", "RMSD<2"), ("valid", "PB-valide"), ("beides", "RMSD<2 & PB")]


def waehle(g: pd.DataFrame, ziel: str, regel: str) -> float:
    """Aus den K Posen EINES Komplexes eine waehlen und das Ziel ablesen."""
    if regel == "zug":
        return float(g[ziel].mean())
    if regel == "orakel":
        return float(g[ziel].any())
    h = g
    if regel == "filter_gnina":
        gv = g[g["valid"]]
        # Faellt alles durch den Filter, waere "keine Pose" das ehrliche
        # Ergebnis -- aber die Regel muss EINE Pose liefern, sonst ist sie
        # mit den anderen nicht vergleichbar. Ohne Ueberlebende wird also
        # aus allen gewaehlt; das kann nur schaden, nie helfen.
        if len(gv):
            h = gv
    if regel.endswith("gnina"):
        i = h["affinity"].idxmin()
    elif regel == "mixed":
        i = h["heur"].idxmax()
    else:
        raise ValueError(regel)
    return float(h.loc[i, ziel])


def auswerten(m: pd.DataFrame, ziel: str, regel: str) -> pd.Series:
    return m.groupby("complex").apply(lambda g: waehle(g, ziel, regel),
                                      include_groups=False)


p = argparse.ArgumentParser()
p.add_argument("--satz", choices=sorted(SAETZE), default="pb308",
               help="pb308 oder astex")
p.add_argument("--frag", default=None,
               help="Vorgabe: fragmentzahl_<satz>.csv")
p.add_argument("--nur-stabil", action="store_true",
               help="Komplexe mit schwankender Fragmentzahl ausschliessen")
p.add_argument("--out", default=os.path.join(_HIER, "nach_fragmenten.csv"))
a = p.parse_args()

a.frag = a.frag or os.path.join(_HIER, f"fragmentzahl_{a.satz}.csv")
if not os.path.isfile(a.frag):
    sys.exit(f"ABBRUCH: {a.frag} fehlt. Erst python SigmaFlow_Variants/"
             f"fragmentzahl.py laufen lassen.")
frag = pd.read_csv(a.frag)
frag = frag[frag["n_frag"].notna()].copy()
frag["n_frag"] = frag["n_frag"].astype(int)
if a.nur_stabil:
    vorher = len(frag)
    frag = frag[frag["stabil"]]
    print(f"nur stabile Fragmentzahlen: {len(frag)} von {vorher}")
print(f"Fragmentzahlen: {len(frag)} Komplexe, "
      f"{frag['n_frag'].min()} bis {frag['n_frag'].max()}, "
      f"{(~frag['stabil']).sum()} instabil\n")

print("Zellen:")
daten = {}
for z in SAETZE[a.satz]["zellen"]:
    daten[(z["arm"], z["nfe"])] = lade_zelle(z)

# Die Auswahl geschieht INNERHALB eines Komplexes und haengt deshalb nicht von
# der Fragmentgruppe ab. Sie wird einmal je (Zelle, Ziel, Regel) gerechnet und
# danach nur noch anders gemittelt -- statt 1350 groupby-Laeufen sind es 90.
zeilen = []
for (arm, nfe), m in daten.items():
    k = m["seed"].nunique()
    m = m.merge(frag[["complex", "n_frag", "stabil"]], on="complex", how="inner")
    nfrag_von = m.groupby("complex")["n_frag"].first()
    for ziel, ztitel in ZIELE:
        for rname, regel in REGELN:
            v = auswerten(m, ziel, regel)              # Index = complex
            nf = nfrag_von.reindex(v.index)
            for gname, maske in [("alle", pd.Series(True, index=v.index))] + [
                    (f"frag={x}", nf == x) for x in sorted(nf.unique())]:
                vg = v[maske]
                zeilen.append({
                    "arm": arm, "nfe": nfe, "K": k, "gruppe": gname,
                    "n_komplexe": int(len(vg)), "ziel": ztitel, "regel": rname,
                    "anteil_prozent": round(100 * float(vg.mean()), 2),
                })

t = pd.DataFrame(zeilen)
t.to_csv(a.out, index=False)
print(f"\n{len(t)} Zeilen nach {a.out}")


def tabelle(ziel: str, gruppe: str = "alle") -> None:
    d = t[(t["ziel"] == ziel) & (t["gruppe"] == gruppe)]
    if d.empty:
        return
    print(f"\n=== {ziel}, Gruppe {gruppe} "
          f"(n = {d['n_komplexe'].iloc[0]} Komplexe) ===")
    piv = d.pivot_table(index=["arm", "nfe", "K"], columns="regel",
                        values="anteil_prozent")
    piv = piv[[r for r, _ in REGELN if r in piv.columns]]
    print(piv.to_string(float_format=lambda x: f"{x:6.2f}"))


for _, ztitel in ZIELE:
    tabelle(ztitel)

# --- Ist der Unterschied der beiden Regeln ueberhaupt echt? ---------------
# Die Differenzen liegen bei ein bis zwei Prozentpunkten auf 307 Komplexen.
# Ohne Intervall ist das nicht interpretierbar. Gepaart, weil BEIDE Regeln
# aus DENSELBEN K Ziehungen desselben Komplexes waehlen -- die Streuung
# zwischen Komplexen faellt dadurch heraus.
def boot(x: pd.Series, y: pd.Series, n: int = 10000, seed: int = 20260907):
    idx = x.index.intersection(y.index)
    d = y.loc[idx].to_numpy(float) - x.loc[idx].to_numpy(float)
    rng = np.random.default_rng(seed)
    v = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    lo, hi = np.percentile(v, [2.5, 97.5])
    p = min(2 * min((v <= 0).mean(), (v >= 0).mean()), 1.0)
    return 100 * d.mean(), 100 * lo, 100 * hi, p


print("\n\n=== PB-Filter -> gnina GEGEN Mixed Score, gepaart, "
      "10.000 Bootstrap-Ziehungen ===")
print("  Positive Differenz heisst: der Filter ist besser.\n")
print(f"  {'Zelle':<20} {'Ziel':<14} {'Differenz':>12} {'95%-Intervall':>20} {'p':>8}")
for (arm, nfe), m in daten.items():
    m2 = m.merge(frag[["complex", "n_frag"]], on="complex", how="inner")
    for ziel, ztitel in ZIELE:
        x = auswerten(m2, ziel, "mixed")
        y = auswerten(m2, ziel, "filter_gnina")
        d, lo, hi, pv = boot(x, y)
        stern = " *" if pv < 0.05 else ""
        print(f"  {arm + ' NFE ' + str(nfe):<20} {ztitel:<14} "
              f"{d:>+9.2f} pp  [{lo:+6.2f}, {hi:+6.2f}]  {pv:>8.4f}{stern}")

print("\n\n########## NACH FRAGMENTZAHL ##########")
for _, ztitel in ZIELE:
    for rname in ("Mixed Score (Paper)", "PB-Filter -> gnina"):
        d = t[(t["ziel"] == ztitel) & (t["regel"] == rname) & (t["gruppe"] != "alle")]
        if d.empty:
            continue
        print(f"\n=== {ztitel},  Regel: {rname} ===")
        piv = d.pivot_table(index="gruppe", columns=["arm", "nfe"],
                            values="anteil_prozent")
        n = d.groupby("gruppe")["n_komplexe"].first()
        piv.insert(0, "n", n)
        piv = piv.reindex(sorted(piv.index, key=lambda s: int(s.split("=")[1])))
        print(piv.to_string(float_format=lambda x: f"{x:6.2f}"))
