"""WIE schwer scheitern die ERZEUGTEN Posen -- ein Kontakt oder zwanzig?

Braucht Redock-Tabellen mit `full_report=True`
(`FULL_REPORT=1 ... sbatch arc/posebusters_redock.slurm`), weil die
Ja/Nein-Tabellen die Zahlenwerte nicht enthalten.

DIE FRAGE
    Kristallposen aus PDBBind scheitern, wenn sie scheitern, meist an EINEM
    Atompaar (Median 1 von 311 geprueften). Wenn die Posen des Modells
    genauso scheitern, ist es knapp daneben und eine Abstossungskorrektur
    koennte die Validitaet heben. Scheitern sie an zwanzig Paaren, sitzt der
    Ligand falsch und keine Nachbearbeitung hilft.

Aufruf:
    $ARC_SF_ENV/bin/python arc/analyse_clash_schwere.py <verzeichnis> [<verzeichnis> ...]
    (jedes Verzeichnis enthaelt rd_*_seed*.csv mit Vollbericht)
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

KRISTALL = "/data/stat-cadd/shug8458/arc_runs/crystal_pb/pdbbind_pocket"


def lade(d: str, muster: str) -> pd.DataFrame:
    f = sorted(glob.glob(os.path.join(d, muster)))
    f = [x for x in f if not x.endswith("crystal_alle.csv")]
    if not f:
        return None
    return pd.concat([pd.read_csv(x) for x in f], ignore_index=True)


def bericht(name: str, t: pd.DataFrame) -> None:
    noetig = ["minimum_distance_to_protein", "num_pairwise_clashes_protein",
              "number_noncov_pairs", "most_extreme_distance_protein"]
    fehlt = [c for c in noetig if c not in t.columns]
    if fehlt:
        print(f"\n{name}: Spalten fehlen {fehlt} -- wurde ohne full_report "
              f"geschrieben?")
        return

    ok = t["minimum_distance_to_protein"].astype(str).str.strip().str.lower().isin(
        {"true", "1", "1.0"})
    s = t[~ok].copy()
    n = pd.to_numeric(s["num_pairwise_clashes_protein"], errors="coerce")
    paare = pd.to_numeric(t["number_noncov_pairs"], errors="coerce")
    d = pd.to_numeric(s["most_extreme_distance_protein"], errors="coerce")

    print(f"\n{'=' * 70}")
    print(f"{name}")
    print(f"  {len(t)} Posen, davon {len(s)} mit Proteinkollision "
          f"({100 * len(s) / len(t):.1f} %)")
    print(f"  geprueffte Kontaktpaare je Pose: Median {paare.median():.0f}")
    print("=" * 70)

    print("  kollidierende Atompaare je betroffener Pose:")
    for lo, hi, txt in [(1, 1, "genau 1"), (2, 2, "2"), (3, 5, "3-5"),
                        (6, 10, "6-10"), (11, 20, "11-20"), (21, 10**9, "ueber 20")]:
        m = (n >= lo) & (n <= hi)
        print(f"    {txt:<10} {int(m.sum()):6d}  ({100 * m.mean():5.1f} %)")
    print(f"    Median {n.median():.0f},  75 % {np.nanpercentile(n, 75):.0f},  "
          f"95 % {np.nanpercentile(n, 95):.0f},  max {n.max():.0f}")

    print("  Abstand des extremsten Kontakts (A):")
    for lo, hi, txt in [(0, 0.001, "exakt 0"), (0.001, 1.9, "unter 1,9  (kovalentnah)"),
                        (1.9, 2.4, "1,9 - 2,4  (kurz polar)"), (2.4, 99, "ab 2,4")]:
        m = (d >= lo) & (d < hi)
        print(f"    {txt:<26} {int(m.sum()):6d}  ({100 * m.mean():5.1f} %)")
    print(f"    Median {d.median():.3f}")

    # WELCHE Atome -- die Elementpaare des extremsten Kontakts. Bei den
    # Kristallposen trennen sie kovalent (C-S, B-O, C-O) sauber von
    # polar (O-O, O-N). Beim Modell gibt es keine kovalente Chemie, dort
    # ist jedes Paar im kovalenten Fenster ein echter Fehler.
    sp_l = 'most_extreme_ligand_element_protein'
    sp_p = 'most_extreme_protein_element_protein'
    if sp_l in s.columns and sp_p in s.columns:
        paar = s[sp_l].astype(str) + '-' + s[sp_p].astype(str)
        print('  haeufigste Elementpaare des extremsten Kontakts:')
        for name, anz in paar.value_counts().head(8).items():
            md = d[paar == name].median()
            print(f'    {name:<8} {anz:6d}  ({100 * anz / len(s):5.1f} %)   Median {md:.3f} A')
        eng = paar[(d >= 1.0) & (d < 1.9)]
        print(f'  im kovalenten Fenster 1,0-1,9 A: {len(eng)} ({100 * len(eng) / len(s):.1f} %)')
        if len(eng):
            print('    ' + ', '.join(f'{k}:{v}' for k, v in
                                     eng.value_counts().head(6).items()))


print("Bezugsgroesse: PDBBind-Kristallposen")
k = lade(KRISTALL, "crystal_*.csv")
if k is not None:
    sp = "file" if "file" in k.columns else k.columns[0]
    if "komplex" not in k.columns:
        k["komplex"] = k[sp].map(lambda p: os.path.basename(os.path.dirname(str(p))))
    bericht("PDBBind, Kristallposen (Kopie 0)", k.groupby("komplex", sort=False).first())

for d in sys.argv[1:]:
    t = lade(d, "rd_*seed*.csv")
    if t is None:
        print(f"\n{d}: keine rd_*seed*.csv gefunden")
        continue
    bericht(os.path.basename(d.rstrip("/")), t)
