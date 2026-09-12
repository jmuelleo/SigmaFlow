"""Bestehen die KRISTALLPOSEN selbst die PoseBusters-Pruefungen?

Liest die Teiltabellen aus `arc/crystal_pb.slurm` und berichtet je Datensatz
die Bestehensquote -- je Pruefung und fuer alle zusammen.

WARUM EINE EXPLIZITE PRUEFLISTE UND KEINE dtype-HEURISTIK
    `full_report=True` liefert NEBEN den Pruefungen viele Diagnosespalten,
    und bei zweien ist die Bedeutung UMGEKEHRT:
        most_extreme_clash_protein   False heisst "kein Clash", also GUT
        not_too_far_away_waters      False, wenn es gar keine Wasser gibt
    Eine Heuristik "alle bool-Spalten sind Pruefungen" sammelt beide ein und
    meldet 0 % bestanden -- genau das ist am 2026-09-05 passiert. Beachte
    ausserdem das Paar `volume_overlap_with_protein` (Pruefung) gegen
    `volume_overlap_protein` (Zahlenwert): ein Unterstrich Unterschied.

    Die Liste unten ist woertlich der Spaltensatz der Redock-Tabellen der
    Auswertungskette (`rd_*_seed*.csv`, Spalten 7-30). Fehlt eine Spalte,
    bricht das Skript ab, statt still weniger zu pruefen.

WARUM rmsd AUSGESCHLOSSEN IST
    Das Skript setzt mol_pred = mol_true = Kristallligand. Der RMSD ist
    identisch null und `rmsd_<=_2A` trivial wahr -- die Spalte traegt keine
    Information und wuerde die Gesamtquote schoenen.

WARUM DREI LESARTEN
    `bust_table` liefert eine Zeile je MOLEKUEL. Die Dateien `*_ligands.sdf`
    von posebusters und astex enthalten mehrere Ligandkopien (astex: 148
    Molekuele auf 85 Komplexe). Berichtet wird deshalb je Komplex:
        Kopie 0  -- die, an der die gesamte Auswertungskette haengt
        beste    -- besteht IRGENDEINE Kopie
        alle     -- bestehen ALLE Kopien
    Die Arbeit nennt Kopie 0; die anderen zeigen die Empfindlichkeit gegen
    diese Wahl. Bei PDBBind (`*ligand*.sdf`, Singular) fallen alle drei
    zusammen.

Aufruf:
    $ARC_SF_ENV/bin/python arc/analyse_crystal_pb.py
    $ARC_SF_ENV/bin/python arc/analyse_crystal_pb.py /pfad/zu/crystal_pb
"""
import glob
import os
import sys

import pandas as pd

# als_bool wird IMPORTIERT, nicht neu geschrieben. Eine zweite Fassung war
# genau der Fehler vom 2026-08-27 (siehe Kopfkommentar von pb_bool.py).
_HIER = os.path.dirname(os.path.abspath(__file__))
_KANDIDATEN = [
    os.path.join(_HIER, "..", "SigmaFlow_Variants", "learning_curve_min"),
    os.path.join(_HIER, "..", "SigmaFlow_Evaluation"),
    _HIER,
]
for _p in _KANDIDATEN:
    if os.path.isfile(os.path.join(_p, "pb_bool.py")):
        sys.path.insert(0, _p)
        break
else:
    sys.exit("pb_bool.py nicht gefunden. Gesucht in:\n  " +
             "\n  ".join(os.path.normpath(p) for p in _KANDIDATEN) +
             "\nEine zweite Fassung von als_bool wird hier NICHT geschrieben.")
from pb_bool import als_bool  # noqa: E402

WURZEL = sys.argv[1] if len(sys.argv) > 1 else \
    "/data/stat-cadd/shug8458/arc_runs/crystal_pb"

# Woertlich die Spalten 7-30 einer Redock-Tabelle der Auswertungskette.
PRUEFUNGEN = [
    "sanitization",
    "inchi_convertible",
    "all_atoms_connected",
    "no_radicals",
    "molecular_formula",
    "molecular_bonds",
    "double_bond_stereochemistry",
    "tetrahedral_chirality",
    "bond_lengths",
    "bond_angles",
    "internal_steric_clash",
    "aromatic_ring_flatness",
    "non-aromatic_ring_non-flatness",
    "double_bond_flatness",
    "internal_energy",
    "protein-ligand_maximum_distance",
    "minimum_distance_to_protein",
    "minimum_distance_to_organic_cofactors",
    "minimum_distance_to_inorganic_cofactors",
    "minimum_distance_to_waters",
    "volume_overlap_with_protein",
    "volume_overlap_with_organic_cofactors",
    "volume_overlap_with_inorganic_cofactors",
    "volume_overlap_with_waters",
]

# Aufteilung nach dem, was der Datenpfad ueberhaupt beeinflussen kann.
#   LAGE      -- Abstaende und Ueberlappungen zum Rezeptor.
#   ACHTUNG: Frueher stand hier, `pb_check` koenne die Lage nicht
#   reparieren, weil die neue Konformation auf die Kristallpose ausgerichtet
#   wird. DAS IST WIDERLEGT (2026-09-05, arc/analyse_train_targets.py):
#   gemessen wirkt pb_check FAST AUSSCHLIESSLICH auf die Lage (+5,68 pp),
#   nicht auf die Ligandchemie (+0,14 pp, n. s.). Die Ausrichtung ist nicht
#   starr -- die Torsionsoptimierung bewegt Atome, und
#   `alignment_rmsd_tolerance = 1,0 A` reicht, einen grenzwertigen Kontakt zu
#   entschaerfen. Diese Aufteilung beschreibt also die Kristallposen, sie
#   sagt NICHTS ueber die Wirkung von pb_check.
#   INTERN    -- Chemie, Bindungen, Planaritaeten, Energie. Genau das,
#                was die Torsionsoptimierung aendert und `pb_check` prueft.
LAGE = [c for c in PRUEFUNGEN
        if c.startswith(('minimum_distance_to', 'volume_overlap_with'))
        or c == 'protein-ligand_maximum_distance']
INTERN = [c for c in PRUEFUNGEN if c not in LAGE]
assert len(LAGE) + len(INTERN) == len(PRUEFUNGEN)

# Die fuenf, aus denen das Paper seinen Mixed Score bildet -- nur zum
# Vergleich mitberichtet, nicht Teil der Gesamtquote.
try:
    sys.path.insert(0, os.path.join(_HIER, "..", "SigmaFlow_Evaluation", "ranking"))
    from heuristic_score import PB_CHECKS  # noqa: E402
except Exception:
    PB_CHECKS = None


def lade(d: str) -> pd.DataFrame | None:
    teile = [t for t in sorted(glob.glob(os.path.join(d, "crystal_*.csv")))
             if not t.endswith("crystal_alle.csv")]
    if not teile:
        return None
    t = pd.concat([pd.read_csv(f) for f in teile], ignore_index=True)
    t.attrs["n_teile"] = len(teile)
    return t


def bericht(name: str, t: pd.DataFrame) -> None:
    fehlt = [c for c in PRUEFUNGEN if c not in t.columns]
    if fehlt:
        sys.exit(f"{name}: {len(fehlt)} Pruefspalte(n) fehlen: {fehlt[:5]}\n"
                 f"  Nicht stillschweigend weniger pruefen -- nachsehen.")

    sp = next((c for c in ("file", "mol_pred") if c in t.columns), t.columns[0])
    if "komplex" not in t.columns:
        t["komplex"] = t[sp].map(lambda p: os.path.basename(os.path.dirname(str(p))))

    # NaN heisst "Modul konnte nicht laufen" und zaehlt laut pb_bool als
    # NICHT bestanden. Bei den Benchmarks kommt das nicht vor, bei PDBBind
    # schon -- dann ist die Zahl selbst ein Befund und gehoert berichtet,
    # statt stillschweigend ins Durchfallen zu wandern.
    nan_je = {c: int(t[c].isna().sum()) for c in PRUEFUNGEN}
    nan_ges = int(t[PRUEFUNGEN].isna().any(axis=1).sum())

    w = pd.DataFrame({c: als_bool(t[c]) for c in PRUEFUNGEN})
    w["ALLE"] = w.all(axis=1)
    w["komplex"] = t["komplex"].to_numpy()

    n_k = w["komplex"].nunique()
    print(f"\n{'=' * 70}")
    print(f"{name}: {len(w)} Molekuele, {n_k} Komplexe "
          f"({len(w) / n_k:.2f} Kopien je Komplex), "
          f"{len(PRUEFUNGEN)} Pruefungen, {t.attrs.get('n_teile', '?')} Teiltabellen")
    print("=" * 70)

    g = w.groupby("komplex", sort=False)
    kopie0 = g.first()

    if nan_ges:
        print(f"\nHINWEIS: bei {nan_ges} von {len(t)} Molekuelen konnte "
              f"mindestens eine Pruefung nicht laufen (NaN). Sie zaehlen "
              f"laut pb_bool als NICHT bestanden.")
        for c, n in sorted(nan_je.items(), key=lambda kv: -kv[1]):
            if n:
                print(f"         {c:<44} {n:6d}")
    print(f"\n{'Pruefung':<42} {'Kopie 0':>9} {'beste':>8} {'alle':>8} {'NaN':>6}")
    for c in sorted(PRUEFUNGEN, key=lambda k: kopie0[k].mean()):
        stern = " *" if PB_CHECKS and c in PB_CHECKS else ""
        print(f"{c + stern:<42} {100 * kopie0[c].mean():8.2f}% "
              f"{100 * g[c].any().mean():7.2f}% {100 * g[c].all().mean():7.2f}% {nan_je[c]:6d}")
    print("-" * 70)
    a0 = kopie0["ALLE"]
    print(f"{'ALLE 24 PRUEFUNGEN':<42} {100 * a0.mean():8.2f}% "
          f"{100 * g['ALLE'].any().mean():7.2f}% {100 * g['ALLE'].all().mean():7.2f}%")
    print(f"   Kopie 0 bestanden: {int(a0.sum())} von {len(a0)}")
    if PB_CHECKS:
        f5 = kopie0[list(PB_CHECKS)].all(axis=1)
        print(f"   nur die {len(PB_CHECKS)} Mixed-Score-Pruefungen (*): "
              f"{100 * f5.mean():.2f}%")

    lg, it = w[LAGE].all(axis=1), w[INTERN].all(axis=1)
    lg.index = it.index = w['komplex']
    lg, it = lg.groupby(level=0, sort=False).first(), it.groupby(level=0, sort=False).first()
    print(f"\n{'Aufteilung (Kopie 0)':<42} {'Anteil':>9}")
    print(f"{'  gueltig':<42} {100 * (lg & it).mean():8.2f}%")
    print(f"{'  nur die LAGE schlecht':<42} {100 * (~lg & it).mean():8.2f}%"
          f"   <- Abstaende/Ueberlappungen zum Rezeptor")
    print(f"{'  nur ligandintern schlecht':<42} {100 * (lg & ~it).mean():8.2f}%"
          f"   <- Chemie, Bindungen, Planaritaeten, Energie")
    print(f"{'  beides schlecht':<42} {100 * (~lg & ~it).mean():8.2f}%")

    durch = sorted(a0[~a0].index)
    if durch:
        print(f"   durchgefallen ({len(durch)}): {', '.join(map(str, durch[:20]))}"
              f"{' ...' if len(durch) > 20 else ''}")


saetze = sorted(d for d in glob.glob(os.path.join(WURZEL, "*")) if os.path.isdir(d))
if not saetze:
    sys.exit(f"keine Unterverzeichnisse unter {WURZEL}")

print(f"Wurzel: {WURZEL}")
for d in saetze:
    t = lade(d)
    if t is None:
        print(f"\n{os.path.basename(d)}: noch keine Teiltabellen")
        continue
    bericht(os.path.basename(d), t)
