"""Wie viel vom Oracle-Abstand holt ein echter Ranker?

DREI AUSWAHLREGELN
    Zufall   : eine der K Posen zufaellig -- die Untergrenze, identisch zur
               Rate je Ziehung.
    Vinardo  : die Pose mit der NIEDRIGSTEN Affinitaet (negativer = besser
               gebunden). Das ist SigmaDocks Energieterm allein.
    Oracle   : trifft, sobald IRGENDEINE der K Posen das Kriterium erfuellt.
               Das ist die richtige Obergrenze. Eine erste Fassung nahm statt
               dessen die Pose mit der kleinsten RMSD -- das ist ein ANDERES,
               schwaecheres Oracle: bei "<2 A UND PB-valide" kann die
               RMSD-beste Pose an PoseBusters scheitern, waehrend eine etwas
               schlechtere besteht. Vinardo schlug dieses Pseudo-Oracle dann
               um bis zu 2 Prozentpunkte, was die Definition verriet.

VERFAHREN
    Fuer jeden Komplex werden B Teilmengen der Groesse K ohne Zuruecklegen
    gezogen und je Regel eine Pose gewaehlt. Gemittelt wird ueber Komplexe.
    Monte Carlo statt geschlossener Formel, weil die Vinardo-Regel von der
    konkreten Teilmenge abhaengt.

ZWOELF UNBEWERTETE POSEN
    score_gnina.py bildet (complex, seed) -> EIN SDF ab. Wo zwei Dateien
    denselben Schluessel tragen, faellt eine heraus. Diese Posen werden hier
    verworfen, 12 von 3080 je Zelle.
"""
import re

import numpy as np
import pandas as pd

from arme import zellen
from kurve_daten import als_bool, zelle_laden

# DIE HEURISTIK DES PAPERS (Prat et al. 2026, Anhang F.2)
#     s_i = -b_i * p_i^beta,  beta = 4
#     b_i  Vinardo-Energie (niedriger = besser), p_i der Anteil bestandener
#          Checks aus GENAU FUENF -- nicht aus allen 24.
# Hoehere s_i = hoehere Konfidenz. Der Faktor p^4 lenkt die Auswahl von der
# energetisch besten auf die chemisch sauberste Pose; das ist nicht dieselbe.
FUENF = ["bond_lengths", "bond_angles", "tetrahedral_chirality",
         "internal_steric_clash", "minimum_distance_to_protein"]
BETA = 4.0


def p5_spalte(z: dict) -> pd.DataFrame:
    """Anteil der fuenf Paper-Checks je Pose, Schluessel (complex, seed, rep).

    Die Schluesselbildung ist Zeile fuer Zeile dieselbe wie in
    kurve_daten.zelle_laden -- weicht sie ab, ordnet der Merge Scores den
    falschen Posen zu, und das faellt in keiner Kennzahl auf.
    """
    teile = []
    for f in sorted(z["redock_dir"].glob("rd_*_seed*.csv")):
        d = pd.read_csv(f)
        d["seed"] = int(re.search(r"seed(\d+)\.csv$", f.name).group(1))
        teile.append(d)
    d = pd.concat(teile, ignore_index=True)
    pfad = d["file"].str.replace("\\", "/", regex=False)
    d["complex"] = pfad.str.rsplit("/", n=1).str[-1].str.split("__").str[0]
    d["datei"] = pfad.str.rsplit("/", n=1).str[-1]
    d = d.sort_values(["complex", "seed", "datei"], kind="stable")
    d["rep"] = d.groupby(["complex", "seed"]).cumcount()
    for c in FUENF:
        d[c] = als_bool(d[c])
    d["p5"] = d[FUENF].mean(axis=1)
    return d[["complex", "seed", "rep", "p5"]]

B = 400
SEED = 20260828
ALLE = zellen()


def lade(arm, nfe, ep):
    # Nicht jede (Arm, NFE, Epoche) existiert -- eine Zelle taucht in
    # zellen() erst auf, wenn per_pose UND Redock vorliegen. Fehlt sie,
    # ist das kein Fehler, sondern "noch nicht gerechnet".
    treffer = [z for z in ALLE if z["arm"] == arm and z["nfe"] == nfe
               and z["epoch"] == ep]
    if not treffer:
        return None
    t = treffer[0]
    p = t["per_pose"].parent / "gnina_scores.csv"
    if not p.exists():
        return None
    m = zelle_laden(t)
    g = pd.read_csv(p)
    g["seed"] = g["seed"].astype(int)
    d = m.merge(p5_spalte(t), on=["complex", "seed", "rep"], how="inner")
    # DOPPELTE (complex, seed) GANZ VERWERFEN, nicht rep==0 behalten.
    #   In rund 0,37 % der Faelle liegen zwei Dateien desselben Komplexes in
    #   einem seed_<k>. Fuer diese Paare stammt die RMSD aus per_pose.csv und
    #   die PoseBusters-Fahne aus dem Redock, und dass beide dieselbe physische
    #   Pose meinen, laesst sich nicht beweisen -- `rep` wird auf beiden Seiten
    #   nur aus einer Sortierreihenfolge gebildet. Eine unabhaengige
    #   Gegenrechnung am 2026-08-29 zeigte Unterschiede bis 0,65 pp; die
    #   sichere Variante ist, solche Paare nicht zu verwenden.
    anzahl = d.groupby(["complex", "seed"])["rep"].transform("size")
    d = d[anzahl == 1].merge(g[["complex", "seed", "affinity"]],
                             on=["complex", "seed"], how="inner")
    d["s_heur"] = -d["affinity"] * (d["p5"] ** BETA)
    return d


def auswahl(d: pd.DataFrame, spalte: str, K: int, B=B, seed=SEED):
    """Trefferquote je Regel, gemittelt ueber Komplexe."""
    rng = np.random.default_rng(seed)
    out = {"Zufall": [], "Vinardo": [], "Heuristik": [], "Oracle": []}
    for _, g in d.groupby("complex"):
        aff = g["affinity"].to_numpy()
        rms = g["rmsd"].to_numpy()
        tref = g[spalte].to_numpy(dtype=bool)
        n = len(g)
        k = min(K, n)
        idx = np.argsort(rng.random((B, n)), axis=1)[:, :k]   # B Teilmengen
        out["Zufall"].append(tref[idx[:, 0]].mean())
        out["Vinardo"].append(tref[np.take_along_axis(
            idx, np.argmin(aff[idx], axis=1, keepdims=True), axis=1)].mean())
        heu = g["s_heur"].to_numpy()
        out["Heuristik"].append(tref[np.take_along_axis(
            idx, np.argmax(heu[idx], axis=1, keepdims=True), axis=1)].mean())
        out["Oracle"].append(tref[idx].any(axis=1).mean())
    return {k: 100 * np.mean(v) for k, v in out.items()}


# Nur diese Zellen tragen gnina-Scores. Die Epochen kommen aus dem
# CHECKPOINT (arme.py), nicht aus der meta.txt -- eine fruehere Fassung suchte
# 137/140 und brach mit IndexError ab, seit die Epochen korrigiert wurden.
# Zellen mit gnina-Scores. Ep. 232 ist SigmaDocks Endpunkt und der einzige
# Punkt mit 40 statt 10 Seeds -- dort reicht die Auswahlkurve bis K = 40.
KANDIDATEN = [("SigmaDock", 132), ("SigmaFlow-Minimal", 136),
              ("SigmaFlow-Separate", 136), ("SigmaDock", 232),
              ("SigmaFlow-Minimal", 232), ("SigmaFlow-Separate", 222)]
ZELLEN = {}
for _arm, _ep in KANDIDATEN:
    for _nfe in (25, 5):
        _d = lade(_arm, _nfe, _ep)
        if _d is not None:
            ZELLEN[f"{_arm} {_nfe} Schritte Ep{_ep}"] = _d

ZIELE = (("u2", "RMSD < 2 A"),
         ("pb_prot", "PB-valid mit Protein"),
         ("u2_prot", "< 2 A UND PB-valide mit Protein"))
zeilen = []

for spalte, titel in ZIELE:
    print(f"\n================= {titel} =================")
    print(f"{'Zelle':<38}{'K':>5}{'Zufall':>9}{'gnina':>9}{'Heurist.':>10}{'Oracle':>9}")
    for name, d in ZELLEN.items():
        # auswahl() nimmt min(K, n); ein K weit ueber der Posenzahl stuende
        # sonst als eigene Zeile da, obwohl es dasselbe rechnet. Die Grenze
        # ueber das Minimum aller Komplexe waere aber zu streng: bei 40 Seeds
        # hat GENAU EIN Komplex 35 statt 40 Posen (die bekannte Sampler-Macke,
        # 8-11 statt 10 je Seed-Lauf), und der haette K=40 fuer alle 307
        # gesperrt. Zugelassen wird K, solange hoechstens 1 % der Komplexe
        # darunter liegen; die Abweichung wird genannt, nicht verschwiegen.
        groesse = d.groupby("complex").size()
        Ks = []
        for k in (2, 5, 8, 10, 20, 40, 70, 140):
            knapp = int((groesse < k).sum())
            if knapp <= max(1, int(0.01 * len(groesse))):
                Ks.append(k)
                if knapp:
                    print(f"  [hinweis] K={k}: {knapp} von {len(groesse)} Komplexen "
                          f"haben weniger Posen, dort wird alles gezogen")
        for K in Ks + ["alle"]:
            if K == "alle":
                # Vollpool: deterministisch, exakt, ohne Monte Carlo.
                z = h = v = o = []
                z, v, h, o = [], [], [], []
                for _, g in d.groupby("complex"):
                    t = g[spalte].to_numpy(dtype=bool)
                    z.append(t.mean())
                    v.append(t[np.argmin(g["affinity"].to_numpy())])
                    h.append(t[np.argmax(g["s_heur"].to_numpy())])
                    o.append(t.any())
                r = {"Zufall": 100*np.mean(z), "Vinardo": 100*np.mean(v),
                     "Heuristik": 100*np.mean(h), "Oracle": 100*np.mean(o)}
            else:
                r = auswahl(d, spalte, K)
            spanne = r["Oracle"] - r["Zufall"]
            anteil = (r["Vinardo"] - r["Zufall"]) / spanne if spanne > 0.1 else float("nan")
            print(f"{name:<38}{str(K):>5}{r['Zufall']:8.2f}%{r['Vinardo']:8.2f}%"
                  f"{r['Heuristik']:9.2f}%{r['Oracle']:8.2f}%")
            zeilen.append(dict(zelle=name, ziel=spalte, k=K,
                               zufall_pct=round(r["Zufall"], 4),
                               gnina_pct=round(r["Vinardo"], 4),
                               heuristik_pct=round(r["Heuristik"], 4),
                               oracle_pct=round(r["Oracle"], 4),
                               anteil_geholt=round(anteil, 4)))
        print()

print("=== Gematchter Aufwand: Separate 8 Seeds @ 5 Schritten gegen "
      "SigmaDock 2 Seeds @ 25 Schritten ===")
for spalte, titel in (("u2", "RMSD < 2 A"), ("u2_prot", "< 2 A & PB+Protein")):
    a = auswahl(ZELLEN["SigmaFlow-Separate 5 Schritte Ep136"], spalte, 8)
    b = auswahl(ZELLEN["SigmaDock 25 Schritte Ep132"], spalte, 2)
    print(f"\n{titel}")
    for regel in ("Zufall", "Vinardo", "Oracle"):
        print(f"  {regel:<8} Separate {a[regel]:6.2f}%   "
              f"SigmaDock {b[regel]:6.2f}%   Differenz {a[regel]-b[regel]:+6.2f} pp")


pd.DataFrame(zeilen).to_csv("ranker_kurve.csv", index=False)
print(str(len(zeilen)) + " Zeilen nach ranker_kurve.csv geschrieben.")
