"""Schadensprofil der AUSGEWAEHLTEN Posen, plus Gueltigkeit als Schwellenkurve.

ZWEI FRAGEN
    1. Die Profile in `analyse_posen_schaden.py` gelten JE ZUG. Interessant
       ist aber die Pose, die der Mixed Score aus K Ziehungen auswaehlt --
       nur die sieht ein Anwender je. Wie kaputt ist die?
    2. Die Schwelle 0,75 ist eine Konvention, kein Naturgesetz. Wie stark
       haengt die berichtete Validitaet daran?

DIE WICHTIGE GROESSE: UNTERSCHREITUNG DER SCHWELLE
    `most_extreme_sum_radii_scaled_protein` minus `most_extreme_distance` ist
    der Abstand zur BERUEHRUNG. PoseBusters verlangt aber nur 0,75 davon. Die
    tatsaechliche Verletzung ist
        (0,75 - relativer_Abstand) * Radiensumme
    und damit erheblich kleiner. Wer die erste Zahl berichtet, stellt den
    Fehler um ein Mehrfaches zu gross dar.

Aufruf:
    $ARC_SF_ENV/bin/python arc/analyse_nach_ranking.py <verzeichnis> [...]
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

ARC_RUNS = "/data/stat-cadd/shug8458/arc_runs"
KRISTALL = os.path.join(ARC_RUNS, "crystal_pb", "pdbbind_pocket")
SCHWELLE = 0.75
WAHR = {"true", "1", "1.0"}

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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from SigmaFlow_Evaluation.ranking.heuristic_score import BETA, PB_CHECKS
except Exception as e:  # noqa: BLE001
    sys.exit(f"heuristic_score nicht importierbar ({e}) -- die Paperformel "
             f"wird hier nicht geraten.")


def b(s):
    return s.astype(str).str.strip().str.lower().isin(WAHR)


def z(t, c):
    if c not in t.columns:
        return pd.Series(np.nan, index=t.index, dtype=float)
    return pd.to_numeric(t[c], errors="coerce")


def q(v, name, einheit=""):
    v = pd.Series(v).dropna()
    if not len(v):
        print(f"  {name:<42}   (keine Werte)")
        return
    p = np.percentile(v, [50, 75, 90, 99])
    print(f"  {name:<42} {p[0]:8.3f} {p[1]:8.3f} {p[2]:8.3f} {p[3]:8.3f} "
          f"{v.max():9.3f} {einheit}")


def kopf():
    print(f"  {'':<42} {'Median':>8} {'75 %':>8} {'90 %':>8} {'99 %':>8} {'max':>9}")


def unterschreitung(t):
    """Wieviele Angstroem liegt der extremste Kontakt UNTER der Schwelle?"""
    rel = z(t, "most_extreme_relative_distance_protein")
    summe = z(t, "most_extreme_sum_radii_scaled_protein")
    return (SCHWELLE - rel) * summe


def lade_zelle(d):
    """Volltabellen einlesen, Seed und Komplex ableiten."""
    teile = []
    for f in sorted(glob.glob(os.path.join(d, "rd_*seed*.csv"))):
        x = pd.read_csv(f)
        x["seed"] = int(re.search(r"seed(\d+)\.csv$", f).group(1))
        teile.append(x)
    if not teile:
        return None
    t = pd.concat(teile, ignore_index=True)
    t["complex"] = t["file"].map(lambda p: os.path.basename(str(p)).split("__")[0])
    for c in PRUEFUNGEN:
        if c not in t.columns:
            sys.exit(f"{d}: Pruefspalte fehlt: {c}")
        t[c] = b(t[c])
    rmsd_sp = next((c for c in t.columns if c.startswith("rmsd_")), None)
    t["acc"] = b(t[rmsd_sp]) if rmsd_sp else np.nan
    t["valid"] = t[PRUEFUNGEN].all(axis=1)
    t["p_pb"] = t[list(PB_CHECKS)].mean(axis=1)
    return t


def mit_gnina(t, tag, pfad=None):
    """gnina-Tabelle NICHT raten.

    Die beiden Konventionen liegen an verschiedenen Orten (Rezeptzellen:
    `GNINA-SCORE-<tag>_<jobid>/`; 72-h-Zellen: `gnina_scores.csv` IM
    Sampling-Verzeichnis). Ein weiter Glob ueber alle Zellen trifft die
    falsche, und weil die Verknuepfung ueber (Komplex, Seed) trotzdem
    gelingt, faellt es NICHT auf -- es entstehen nur fremde Affinitaeten und
    damit eine falsche Auswahl. Genau das ist am 2026-09-05 passiert.

    Deshalb: entweder ein eindeutiger Treffer unter `GNINA-SCORE-<tag>_*`,
    oder der Pfad wird ausdruecklich uebergeben (Argument `verzeichnis:csv`).
    Alles andere bricht ab.
    """
    if pfad:
        if not os.path.isfile(pfad):
            sys.exit(f"{tag}: angegebene gnina-Tabelle fehlt: {pfad}")
        g = [pfad]
    else:
        g = sorted(glob.glob(os.path.join(ARC_RUNS, f"GNINA-SCORE-{tag}_*",
                                          f"gnina_scores_{tag}.csv")))
        if len(g) > 1:
            sys.exit(f"{tag}: {len(g)} gnina-Tabellen passen. "
                     f"Ausdruecklich uebergeben als <verzeichnis>:<csv>.\n  "
                     + "\n  ".join(g))
        if not g:
            print(f"  KEIN Ranking fuer {tag}: keine Tabelle unter "
                  f"GNINA-SCORE-{tag}_*. Bei 72-h-Zellen liegt sie im "
                  f"Sampling-Verzeichnis; so uebergeben: "
                  f"<verzeichnis>:<pfad/gnina_scores.csv>")
            return None
    gn = pd.read_csv(g[0])
    print(f"  gnina: {os.path.basename(g[0])}, {len(gn)} Zeilen, "
          f"{gn['seed'].nunique()} Seeds")
    m = t.merge(gn[["complex", "seed", "affinity"]], on=["complex", "seed"],
                how="inner")
    if len(m) < 0.5 * len(t):
        sys.exit(f"{tag}: nur {len(m)} von {len(t)} Zeilen nach dem Merge -- "
                 f"passt diese gnina-Tabelle ueberhaupt zur Zelle?")
    m["heur"] = -m["affinity"] * (m["p_pb"] ** BETA)
    return m

def schwellenkurve(t, name):
    """Zwei Lesarten der Aufweichung.

    VERHAELTNIS -- die Schwelle 0,75 durch eine andere ersetzen. Physikalisch
        heikel: 0,60 erlaubt bei C-O 1,93 A, weniger als ein
        Nichtbindungskontakt je misst. Nur zur Empfindlichkeitsanalyse.
    TOLERANZ IN ANGSTROEM -- die Schwelle beibehalten und eine
        Unterschreitung von hoechstens tau zulassen. Als
        Koordinatenunsicherheit lesbar (mittlere Aufloesung 0,2 bis 0,5 A)
        und damit die verteidigbare weiche Variante.
    """
    rel = z(t, "most_extreme_relative_distance_protein")
    da = rel.notna()
    andere = t.loc[da, [c for c in PRUEFUNGEN
                        if c != "minimum_distance_to_protein"]].all(axis=1)
    acc = t.loc[da, "acc"].astype(bool) if "acc" in t.columns else None
    u = unterschreitung(t)[da]

    print(f"\n  A) Gueltigkeit bei anderer VERHAELTNIS-Schwelle -- {name}")
    sp = f"  {'Schwelle':>9} {'Abstand ok':>12} {'alle 24 ok':>12}"
    if acc is not None:
        sp += f" {'+ RMSD<2':>10}"
    print(sp)
    for x in (0.80, 0.78, 0.75, 0.72, 0.70, 0.68, 0.65, 0.60):
        ok = rel[da] >= x
        mark = "   <- PoseBusters" if abs(x - SCHWELLE) < 1e-9 else ""
        r = f"  {x:9.2f} {100 * ok.mean():11.2f}% {100 * (ok & andere).mean():11.2f}%"
        if acc is not None:
            r += f" {100 * (ok & andere & acc).mean():9.2f}%"
        print(r + mark)

    print(f"\n  B) Gueltigkeit mit TOLERANZ tau unter der Schwelle -- {name}")
    sp = f"  {'tau [A]':>9} {'Abstand ok':>12} {'alle 24 ok':>12}"
    if acc is not None:
        sp += f" {'+ RMSD<2':>10}"
    print(sp)
    for tau in (0.00, 0.05, 0.10, 0.15, 0.25, 0.40, 0.60):
        ok = u <= tau
        mark = "   <- PoseBusters" if tau == 0.0 else ""
        if abs(tau - 0.25) < 1e-9:
            mark = "   <- Koordinatenunsicherheit"
        r = f"  {tau:9.2f} {100 * ok.mean():11.2f}% {100 * (ok & andere).mean():11.2f}%"
        if acc is not None:
            r += f" {100 * (ok & andere & acc).mean():9.2f}%"
        print(r + mark)


def wie_knapp(t, name):
    """Wie knapp scheitern die ungueltigen Posen -- Tiefe x Ausdehnung.

    Eine Pose mit zwei Atompaaren, die die Schwelle um 0,2 A unterschreiten,
    sollte nicht wie eine mit zwanzig Paaren zu je 1,5 A bewertet werden. Die
    Tabelle trennt beides: Zeilen = wie TIEF der schlimmste Kontakt unter der
    Schwelle liegt, Spalten = an WIEVIELEN Paaren es klemmt.

    `most_extreme` ist der SCHLIMMSTE Kontakt. Liegt er <= tau, liegen ALLE
    darunter -- die Zeile ist also eine harte Schranke, keine Mittelung.
    """
    ok = t["minimum_distance_to_protein"]
    andere = t[[c for c in PRUEFUNGEN
                if c != "minimum_distance_to_protein"]].all(axis=1)
    u = unterschreitung(t)
    npr = z(t, "num_pairwise_clashes_protein")
    s = ~ok & andere            # scheitert NUR am Abstand
    n_s = int(s.sum())
    print(f"\n  Von {len(t)} Posen scheitern {n_s} NUR an der "
          f"Kollisionspruefung ({100 * n_s / len(t):.2f} %)")
    if not n_s:
        return
    print(f"  Tiefe x Ausdehnung, Anteil dieser {n_s} Posen:")
    print(f"  {'schlimmster Kontakt':<26} {'1 Paar':>8} {'2-3':>8} "
          f"{'4-10':>8} {'>10':>8} {'Summe':>8}")
    kum = 0.0
    for lo, hi, txt in [(0.0, 0.10, "bis 0,10 A zu nah"),
                        (0.10, 0.25, "0,10 - 0,25 A"),
                        (0.25, 0.50, "0,25 - 0,50 A"),
                        (0.50, 1.00, "0,50 - 1,00 A"),
                        (1.00, 99.0, "ueber 1,00 A")]:
        m = s & (u > lo) & (u <= hi)
        r = [int((m & (npr == 1)).sum()), int((m & (npr >= 2) & (npr <= 3)).sum()),
             int((m & (npr >= 4) & (npr <= 10)).sum()), int((m & (npr > 10)).sum())]
        anteil = 100 * sum(r) / n_s
        kum += anteil
        print(f"  {txt:<26} {r[0]:8d} {r[1]:8d} {r[2]:8d} {r[3]:8d} "
              f"{anteil:7.1f}%   kumuliert {kum:5.1f}%")


def weiche_zielmetrik(t, name):
    """Kombiniert (RMSD<2 UND gueltig) bei Toleranz tau und hoechstens n Paaren.

    Die Toleranz gilt fuer den SCHLIMMSTEN Kontakt; liegt er unter tau, liegen
    alle darunter. Die Paarbegrenzung ist zusaetzliche Strenge: sie fragt,
    ob der Fehler auch RAEUMLICH eng begrenzt ist.
    """
    ok = t["minimum_distance_to_protein"]
    andere = t[[c for c in PRUEFUNGEN
                if c != "minimum_distance_to_protein"]].all(axis=1)
    acc = t["acc"].astype(bool)
    u = unterschreitung(t)
    npr = z(t, "num_pairwise_clashes_protein").fillna(0)
    print(f"\n  Weiche Zielmetrik: RMSD<2 UND gueltig -- {name}")
    print(f"  {'':<12}" + "".join(f"{c:>16}" for c in
                                   ("hoechstens 1 Paar", "bis 3 Paare", "beliebig viele")))
    for tau in (0.00, 0.10, 0.25, 0.40):
        r = []
        for kap in (1, 3, 10**9):
            g = ok | ((u <= tau) & (npr <= kap))
            r.append(100 * (g & andere & acc).mean())
        mark = "  <- PoseBusters" if tau == 0 else ""
        print(f"  tau {tau:5.2f} A " + "".join(f"{x:15.2f}%" for x in r) + mark)


def profil(name, t, suffix=""):
    print()
    print("=" * 92)
    print(f"  {name}{suffix}   ({len(t)} Posen)")
    print("=" * 92)
    ok = t["minimum_distance_to_protein"]
    npr = z(t, "num_pairwise_clashes_protein")
    u = unterschreitung(t)
    print(f"  PB-valide (alle 24): {100 * t['valid'].mean():.2f} %"
          f"     Kollision mit Protein: {100 * (~ok).mean():.2f} %")
    kopf()
    q(npr[~ok], "kollidierende Atompaare (Anzahl)")
    q(u[~ok], "UNTERSCHREITUNG der Schwelle", "A")
    q(z(t, "most_extreme_relative_distance_protein")[~ok], "extremster Kontakt, relativ")
    q(z(t, "volume_overlap_protein")[~ok], "Volumenueberlapp (Schwelle 0,075)")

    aus_b = (z(t, "number_short_outlier_bonds").fillna(0)
             + z(t, "number_long_outlier_bonds").fillna(0))
    aus_w = z(t, "number_outlier_angles").fillna(0)
    kl = z(t, "number_clashes").fillna(0)
    print(f"  ligandintern mit Fehler:  Bindung {100 * (aus_b > 0).mean():5.2f} %"
          f"   Winkel {100 * (aus_w > 0).mean():5.2f} %"
          f"   innerer Clash {100 * (kl > 0).mean():5.2f} %")
    q(z(t, "longest_bond_relative_length")[aus_b > 0], "laengste Bindung, relativ")


k = None
if os.path.isdir(KRISTALL):
    f = [x for x in sorted(glob.glob(os.path.join(KRISTALL, "crystal_*.csv")))
         if not x.endswith("crystal_alle.csv")]
    if f:
        k = pd.concat([pd.read_csv(x) for x in f], ignore_index=True)
        k["komplex"] = k["file"].map(lambda p: os.path.basename(os.path.dirname(str(p))))
        k = k.groupby("komplex", sort=False).first()
        for c in PRUEFUNGEN:
            k[c] = b(k[c])
        k["valid"] = k[PRUEFUNGEN].all(axis=1)
        profil("BEZUGSGROESSE: PDBBind-Kristallposen", k)
        schwellenkurve(k, "Kristallposen")

for arg in sys.argv[1:]:
    # Ein Argument darf "<verzeichnis>:<gnina.csv>" lauten. Unter Windows
    # gaebe es ein Laufwerksproblem, auf ARC nicht -- hier reicht rsplit.
    d, _, gpfad = arg.partition("::")
    gpfad = gpfad or None
    tag = os.path.basename(d.rstrip("/")).replace("__full", "")
    t = lade_zelle(d)
    if t is None:
        print(f"\n{d}: keine rd_*seed*.csv")
        continue
    K = t["seed"].nunique()
    profil(tag, t, "  -- JE ZUG")
    schwellenkurve(t, f"{tag}, je Zug")

    m = mit_gnina(t, tag, pfad=gpfad)
    if m is None:
        print(f"\n  (keine gnina-Tabelle fuer {tag} -- kein Ranking moeglich)")
        continue
    idx = m.groupby("complex")["heur"].idxmax()
    aus = m.loc[idx]
    profil(tag, aus, f"  -- TOP-1 NACH MIXED SCORE, K = {K}")
    print(f"  RMSD < 2 A: {100 * aus['acc'].mean():.2f} %"
          f"     kombiniert: {100 * (aus['acc'] & aus['valid']).mean():.2f} %")
    wie_knapp(aus, tag)
    weiche_zielmetrik(aus, f"{tag}, Top-1 bei K={K}")
    schwellenkurve(aus, f"{tag}, Top-1 bei K={K}")
