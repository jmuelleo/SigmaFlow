"""Die Endpunktzellen der 72-h-Laeufe fuer PB308 UND Astex -- an EINER Stelle.

WARUM EIN EIGENES MODUL
    Die Pfade standen vorher in `nach_fragmenten.py` und drohten in jedem
    weiteren Auswertungsskript ein zweites Mal zu stehen. Genau daran ist hier
    schon einmal die falsche gnina-Tabelle gezogen worden: in
    `final200/.../nfe5__sampled/` liegen `gnina_scores.csv` (140 Seeds) und
    `gnina_scores_200.csv` (200 Seeds) nebeneinander, und ein Glob nahm die
    erstbeste. SigmaDock las damals 59,61 statt 69,71.

    Deshalb: fest verdrahtete Pfade, eine erwartete Seedzahl je Zelle, und
    `lade_zelle` bricht ab, wenn sie nicht stimmt.

WAS `lade_zelle` LIEFERT
    Eine Zeile je Pose mit den Spalten:
        complex, seed, <24 Pruefspalten als bool>, acc, valid, p_pb,
        affinity, heur, beides
    acc     RMSD < 2 A
    valid   alle 24 PoseBusters-Pruefungen bestanden
    p_pb    Anteil bestandener Pruefungen aus den FUENF des Papers
    heur    Mixed Score -affinity * p_pb^BETA, hoeher ist besser
"""
import glob
import os
import re
import sys

import pandas as pd

_HIER = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HIER)

for _p in (os.path.join(_HIER, "learning_curve_min"), os.path.join(_REPO, "arc")):
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

# Die 24 Pruefungen der redock-Konfiguration, ohne RMSD. Fest aufgezaehlt,
# weil eine Heuristik ueber Spaltentypen die Diagnosespalten mit UMGEKEHRTER
# Bedeutung einsammelt (most_extreme_clash_protein: False = keine Kollision).
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

_E = os.path.join(_HIER, "pb308_endpoints")
_F = os.path.join(_HIER, "final200")
_SD = os.path.join(_E, "sd_endpunkt_40seeds", "SD_BASE_72H_s0_8648493")
_MIN25 = os.path.join(_E, "endpunkt_min_nfe25", "SF_MIN_72H_s0_8653824")
_SEP25 = os.path.join(_E, "endpunkt_sep_nfe25", "SF_2H_72H_s0_8668713")
_MIN5 = os.path.join(_F, "SF_MIN_72H_s0_8653824")
_SEP5 = os.path.join(_F, "SF_2H_72H_s0_8668713")


def _z(arm, nfe, k, wurzel, tag, gnina):
    return dict(arm=arm, nfe=nfe, k=k,
                redock=os.path.join(wurzel, "posebusters_redock_curve", tag),
                gnina=os.path.join(wurzel, "learning_curve_cpu", tag, gnina))


ZELLEN = [
    _z("SigmaDock", 25, 40, _SD, "sched239ep_emergency__nfe25__sampled", "gnina_scores.csv"),
    _z("SigmaDock",  5, 40, _SD, "sched239ep_emergency__nfe5__sampled", "gnina_scores.csv"),
    _z("Minimal",   25, 40, _MIN25, "sched255ep_emergency__nfe25__sampled", "gnina_scores.csv"),
    _z("Minimal",    5, 200, _MIN5, "sched255ep_emergency__nfe5__sampled", "gnina_scores_200.csv"),
    _z("Separate",  25, 40, _SEP25, "sched255ep_emergency__nfe25__sampled", "gnina_scores.csv"),
    _z("Separate",   5, 200, _SEP5, "sched255ep_emergency__nfe5__sampled", "gnina_scores_200.csv"),
]


def lade_zelle(z: dict, leise: bool = False) -> pd.DataFrame:
    dateien = sorted(glob.glob(os.path.join(z["redock"], "rd_*seed*.csv")))
    if not dateien:
        sys.exit(f"ABBRUCH: keine rd_*seed*.csv unter {z['redock']}")
    teile = []
    for f in dateien:
        d = pd.read_csv(f)
        d["seed"] = int(re.search(r"seed(\d+)\.csv$", f).group(1))
        teile.append(d)
    rd = pd.concat(teile, ignore_index=True)
    rd["complex"] = rd["file"].map(lambda x: os.path.basename(str(x)).split("__")[0])

    fehlt = [c for c in PRUEFUNGEN if c not in rd.columns]
    if fehlt:
        sys.exit(f"ABBRUCH: Pruefspalten fehlen in {z['redock']}: {fehlt[:5]}")

    # Je Seed liegen 308 Zeilen, aber nur 307 Komplexe vor: einer ist doppelt
    # (seedabhaengig), einer fehlt ueberall. gnina hat 307. Ohne diese
    # Bereinigung verdoppelt der Merge den doppelten still.
    vorher = len(rd)
    rd = rd.sort_values(["seed", "file"]).drop_duplicates(["complex", "seed"], keep="first")
    doppelt = vorher - len(rd)

    for c in PRUEFUNGEN:
        rd[c] = als_bool(rd[c])
    rmsd_sp = next(c for c in rd.columns if c.startswith("rmsd"))
    rd["acc"] = als_bool(rd[rmsd_sp])
    rd["valid"] = rd[PRUEFUNGEN].all(axis=1)
    rd["p_pb"] = rd[list(PB_CHECKS)].mean(axis=1)

    gn = pd.read_csv(z["gnina"])
    if gn["seed"].nunique() != z["k"]:
        sys.exit(f"ABBRUCH: {z['gnina']} hat {gn['seed'].nunique()} Seeds, "
                 f"erwartet {z['k']}. Falsche gnina-Tabelle -- nicht raten.")
    if rd["seed"].nunique() != z["k"]:
        sys.exit(f"ABBRUCH: {z['redock']} hat {rd['seed'].nunique()} "
                 f"Seed-Tabellen, erwartet {z['k']}.")

    m = rd.merge(gn[["complex", "seed", "affinity"]], on=["complex", "seed"], how="inner")
    m["heur"] = -m["affinity"] * (m["p_pb"] ** BETA)
    m["beides"] = m["acc"] & m["valid"]

    if not leise:
        print(f"  {z['arm']:9s} NFE {z['nfe']:2d}  K={z['k']:3d}  "
              f"{len(m):6d} Posen, {m['complex'].nunique():3d} Komplexe  "
              f"(dedupliziert {doppelt}, beim Merge verloren {len(rd) - len(m)})")
    return m

# ==========================================================================
# ASTEX DIVERSE SET -- DIESELBEN DREI 72-h-LAEUFE, ANDERER TESTSATZ
# ==========================================================================
# Nachgewiesen an der `file`-Spalte: alle sechs Zellen stammen aus
# SD_BASE_72H_s0_8648493, SF_MIN_72H_s0_8653824 und SF_2H_72H_s0_8668713,
# jeweils vom Snapshot `*_emergency`, also dem Endpunkt.
#
# DREI UNTERSCHIEDE ZU PB308, DIE MAN KENNEN MUSS
#   1. Das Sampling liegt unter `learning_curve_cpu_astex`, nicht
#      `learning_curve_cpu`, und die Ergebnisse unter `results/astex/`.
#      Ein Glob ueber `learning_curve_cpu*` traefe beides und zaehlte die
#      Saetze zusammen -- ohne Fehlermeldung.
#   2. Die gnina-Tabellen liegen in eigenen GNINA-SCORE-*-Verzeichnissen mit
#      Jobnummer im Namen. Deshalb hier fest verdrahtet statt geglobt.
#   3. Die Referenzstrukturen unter astex/astex_diverse_set/ enthalten NUR
#      `_ligand.sdf` und `_ligands.sdf`, KEIN Protein. Fuer die Zahlen
#      unerheblich; fuer PyMOL muss das Protein von ARC nachgeladen werden
#      (dort unter data/posebusters_paper/astex_diverse_set/).

_A = os.path.join(_HIER, "astex")
_A2 = os.path.join(_HIER, "final200")

ZELLEN_ASTEX = [
    dict(arm="SigmaDock", nfe=25, k=40,
         redock=os.path.join(_A, "astex_redock", "sigmadock"),
         gnina=os.path.join(_A, "GNINA-SCORE-sigmadock__astex_8680368",
                            "gnina_scores_sigmadock__astex.csv")),
    dict(arm="SigmaDock", nfe=5, k=40,
         redock=os.path.join(_A, "astex_redock", "sigmadock__nfe5"),
         gnina=os.path.join(_A, "GNINA-SCORE-sigmadock__nfe5__astex_8681056",
                            "gnina_scores_sigmadock__nfe5__astex.csv")),
    dict(arm="Minimal", nfe=25, k=40,
         redock=os.path.join(_A, "astex_redock", "sigmaflow_minimal"),
         gnina=os.path.join(_A, "GNINA-SCORE-sigmaflow_minimal__astex_8680369",
                            "gnina_scores_sigmaflow_minimal__astex.csv")),
    dict(arm="Minimal", nfe=5, k=200,
         redock=os.path.join(_A2, "astex_redock", "sigmaflow_minimal__nfe5"),
         gnina=os.path.join(_A2, "GNINA-SCORE-sigmaflow_minimal__nfe5__astex_8686311",
                            "gnina_scores_sigmaflow_minimal__nfe5__astex.csv")),
    dict(arm="Separate", nfe=25, k=40,
         redock=os.path.join(_A, "astex_redock", "exp110"),
         gnina=os.path.join(_A, "GNINA-SCORE-exp110__astex_8680370",
                            "gnina_scores_exp110__astex.csv")),
    dict(arm="Separate", nfe=5, k=200,
         redock=os.path.join(_A2, "astex_redock", "exp110__nfe5"),
         gnina=os.path.join(_A2, "GNINA-SCORE-exp110__nfe5__astex_8686313",
                            "gnina_scores_exp110__nfe5__astex.csv")),
]

# Beschreibung beider Saetze an EINER Stelle, damit ein Auswertungsskript
# ueber `--satz` umschalten kann, ohne dass irgendwo ein Pfad zweimal steht.
SAETZE = {
    "pb308": dict(
        zellen=ZELLEN,
        posen_unterordner="learning_curve_cpu",
        ergebnis_unterordner="posebusters308",
        referenz=os.path.join(_HIER, "learning_curve_min", "true308"),
        referenz_ligand="{code}_ligands.sdf",
        referenz_protein="{code}_protein.pdb",
    ),
    "astex": dict(
        zellen=ZELLEN_ASTEX,
        posen_unterordner="learning_curve_cpu_astex",
        ergebnis_unterordner="astex",
        referenz=os.path.join(_A, "astex_diverse_set"),
        referenz_ligand="{code}_ligands.sdf",
        referenz_protein=None,          # liegt lokal nicht vor, s. o.
    ),
}

LAEUFE = {
    ("SigmaDock", 25): ("SD_BASE_72H_s0_8648493", "sched239ep_emergency__nfe25__sampled"),
    ("SigmaDock", 5):  ("SD_BASE_72H_s0_8648493", "sched239ep_emergency__nfe5__sampled"),
    ("Minimal", 25):   ("SF_MIN_72H_s0_8653824", "sched255ep_emergency__nfe25__sampled"),
    ("Minimal", 5):    ("SF_MIN_72H_s0_8653824", "sched255ep_emergency__nfe5__sampled"),
    ("Separate", 25):  ("SF_2H_72H_s0_8668713", "sched255ep_emergency__nfe25__sampled"),
    ("Separate", 5):   ("SF_2H_72H_s0_8668713", "sched255ep_emergency__nfe5__sampled"),
}


def posenordner(satz: str, arm: str, nfe: int) -> str:
    """Verzeichnis mit den seed_*/ Unterordnern fuer eine Zelle."""
    lauf, tag = LAEUFE[(arm, nfe)]
    s = SAETZE[satz]
    return os.path.join(_HIER, lauf, s["posen_unterordner"], tag, "results",
                        s["ergebnis_unterordner"], f"snap_{tag.split('__')[0]}")


def alle_zellen(leise: bool = False, satz: str = "pb308") -> dict:
    return {(z["arm"], z["nfe"]): lade_zelle(z, leise)
            for z in SAETZE[satz]["zellen"]}


# ==========================================================================
# TOP-1 BEI K ZIEHUNGEN -- 1000 ZUFALLSTEILMENGEN
# ==========================================================================
# DIE FESTGELEGTE METHODIK
#   Je Komplex werden N_ZIEHUNGEN Teilmengen der Groesse K aus den
#   vorhandenen Posen gezogen, ohne Zuruecklegen. In jeder Teilmenge waehlt
#   die Regel eine Pose; ob sie das Ziel erfuellt, ist eine Null oder Eins.
#   Der Mittelwert dieser Ziehungen ist der Wert des Komplexes, und erst
#   darueber wird ueber die Komplexe gemittelt.
#
#   Diese Reihenfolge ist wichtig: erst innerhalb des Komplexes mitteln, dann
#   ueber Komplexe. Wuerde man alle Ziehungen aller Komplexe in einen Topf
#   werfen, bekaemen Komplexe mit mehr Posen mehr Gewicht.
#
# WARUM K = n KEIN SONDERFALL IST
#   Bei K gleich der Zahl der vorhandenen Posen gibt es genau eine Teilmenge.
#   Die Funktion erkennt das und rechnet direkt, statt tausendmal dasselbe zu
#   ziehen.
#
# WIE DIE ZIEHUNG UMGESETZT IST
#   Die Posen werden EINMAL nach der Regel sortiert, beste zuerst. Danach ist
#   die Gewinnerin einer Teilmenge die mit der kleinsten Position in dieser
#   Sortierung -- man muss also nicht je Ziehung neu vergleichen, sondern nur
#   das Minimum der gezogenen Positionen nehmen. `argpartition` reicht dafuer,
#   `argsort` waere unnoetig teuer.
#
# WELCHE REGELN ES GIBT
#   "mixed"   -a * p_pb^BETA, hoeher ist besser -- der Score des Papers
#   "gnina"   nur die Bindungsenergie, kleiner ist besser
#   "filter"  erst PB-Validitaet hart, darunter die Bindungsenergie. Das ist
#             eine lexikographische und damit immer noch TOTALE Ordnung, also
#             genauso behandelbar wie ein Score.
#   "orakel"  trifft irgendeine Pose der Teilmenge?
#   "zug"     Mittel ueber alle Posen; von K unabhaengig, dient als Bezug.

N_ZIEHUNGEN = 1000
ZIEH_SEED = 20260908


def _ordnung(g: pd.DataFrame, regel: str):
    """Posen eines Komplexes nach der Regel sortieren, beste zuerst."""
    if regel == "mixed":
        return g.sort_values("heur", ascending=False)
    if regel == "gnina":
        return g.sort_values("affinity", ascending=True)
    if regel == "filter":
        return g.sort_values(["valid", "affinity"], ascending=[False, True])
    raise ValueError(f"unbekannte Regel: {regel}")


def top1(m: pd.DataFrame, ziel: str, K: int, regel: str = "mixed",
         n_ziehungen: int = N_ZIEHUNGEN, seed: int = ZIEH_SEED) -> float:
    """Anteil der Komplexe, bei denen die gewaehlte Pose das Ziel erfuellt.

    m       Posen einer Zelle (Ausgabe von lade_zelle)
    ziel    "acc", "valid" oder "beides"
    K       Ziehungen je Komplex
    regel   "mixed", "gnina", "filter", "orakel" oder "zug"
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    je_komplex = []
    for _, g in m.groupby("complex"):
        n = len(g)
        k = min(K, n)
        if regel == "zug":
            je_komplex.append(float(g[ziel].mean()))
            continue
        if regel == "orakel":
            y = g[ziel].to_numpy(bool)
            if k >= n:
                je_komplex.append(float(y.any()))
            else:
                pick = rng.random((n_ziehungen, n)).argpartition(
                    k - 1, axis=1)[:, :k]
                je_komplex.append(float(y[pick].any(axis=1).mean()))
            continue
        y = _ordnung(g, regel)[ziel].to_numpy(float)   # sortiert, beste zuerst
        if k >= n:
            je_komplex.append(float(y[0]))
            continue
        # Gewinnerin = kleinste gezogene Position in der Sortierung.
        pos = rng.random((n_ziehungen, n)).argpartition(k - 1, axis=1)[:, :k]
        je_komplex.append(float(y[pos.min(axis=1)].mean()))
    return 100.0 * float(np.mean(je_komplex))
