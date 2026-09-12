"""Waehlt Komplexe fuer die PyMOL-Betrachtung aus und erzeugt die ARC-Befehle.

WARUM AUSWAEHLEN STATT ALLES HOLEN
    307 Komplexe x (40+40+40+200+40+200) Seeds sind rund 172.000 SDF-Dateien.
    Auf dem geteilten Dateisystem ist das Anlegen und Uebertragen vieler
    kleiner Dateien der Flaschenhals; ein einzelnes Archiv mit einer Auswahl
    kostet Sekunden statt Stunden.

WONACH AUSGEWAEHLT WIRD
    Die Frage lautet: erzeugt SigmaFlow eine BREITERE Verteilung als SigmaDock?
    Dafuer taugt keine Zufallsauswahl. Gebraucht werden Faelle, in denen sich
    die Arme unterscheiden, ueber die ganze Spanne der Fragmentzahl:

      leicht   wenige Fragmente, alle Arme treffen -> wie eng ist "eng"?
      mittel   mittlere Fragmentzahl, Arme treffen unterschiedlich oft
      schwer   viele Fragmente, alle Arme scheitern -> woran genau?
      kontrast SigmaDock trifft, Flow nicht (und umgekehrt)

    Als Streumass dient die STANDARDABWEICHUNG des RMSD zur Kristallpose ueber
    die Seeds. Das ist NICHT dasselbe wie die Posen-zu-Posen-Streuung -- zwei
    Posen koennen denselben RMSD zur Wahrheit haben und trotzdem weit
    auseinanderliegen. Es ist ein Vorabmass, um Kandidaten zu finden; die
    eigentliche Aussage kommt aus der Anschauung in PyMOL.

Aufruf:
    python SigmaFlow_Variants/pymol_auswahl.py
    python SigmaFlow_Variants/pymol_auswahl.py --je-gruppe 3
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

_HIER = os.path.dirname(os.path.abspath(__file__))

# Dieselben Laeufe und Etiketten wie in nach_fragmenten.py. Bewusst
# wiederholt statt importiert: dieses Skript soll auch dann laufen, wenn
# dort etwas umgebaut wird.
LAEUFE = {
    "SigmaDock": ("SD_BASE_72H_s0_8648493", "sched239ep_emergency"),
    "Minimal":   ("SF_MIN_72H_s0_8653824", "sched255ep_emergency"),
    "Separate":  ("SF_2H_72H_s0_8668713", "sched255ep_emergency"),
}
# per_pose.csv je Zelle -- complex, seed, rmsd. Reicht fuer die Vorauswahl;
# die PB-Pruefungen braucht es dafuer nicht.
POSEN = {
    ("SigmaDock", 25): "pb308_endpoints/sd_endpunkt_40seeds/SD_BASE_72H_s0_8648493/learning_curve_cpu/sched239ep_emergency__nfe25__sampled/per_pose.csv",
    ("SigmaDock", 5):  "pb308_endpoints/sd_endpunkt_40seeds/SD_BASE_72H_s0_8648493/learning_curve_cpu/sched239ep_emergency__nfe5__sampled/per_pose.csv",
    ("Minimal", 25):   "pb308_endpoints/endpunkt_min_nfe25/SF_MIN_72H_s0_8653824/learning_curve_cpu/sched255ep_emergency__nfe25__sampled/per_pose.csv",
    ("Minimal", 5):    "final200/SF_MIN_72H_s0_8653824/learning_curve_cpu/sched255ep_emergency__nfe5__sampled/per_pose.csv",
    ("Separate", 25):  "pb308_endpoints/endpunkt_sep_nfe25/SF_2H_72H_s0_8668713/learning_curve_cpu/sched255ep_emergency__nfe25__sampled/per_pose.csv",
    ("Separate", 5):   "final200/SF_2H_72H_s0_8668713/learning_curve_cpu/sched255ep_emergency__nfe5__sampled/per_pose.csv",
}

p = argparse.ArgumentParser()
p.add_argument("--je-gruppe", type=int, default=2)
p.add_argument("--frag", default=os.path.join(_HIER, "fragmentzahl_pb308.csv"))
a = p.parse_args()

frag = pd.read_csv(a.frag)
frag = frag[frag["n_frag"].notna() & frag["stabil"]].copy()
frag["n_frag"] = frag["n_frag"].astype(int)

teile = []
for (arm, nfe), rel in POSEN.items():
    f = os.path.join(_HIER, rel)
    if not os.path.isfile(f):
        sys.exit(f"ABBRUCH: {f} fehlt.")
    d = pd.read_csv(f)
    fehlt = {"complex", "seed", "rmsd"} - set(d.columns)
    if fehlt:
        sys.exit(f"ABBRUCH: {f} fehlen Spalten {fehlt}")
    d["arm"], d["nfe"] = arm, nfe
    teile.append(d)
pp = pd.concat(teile, ignore_index=True)
print(f"{len(pp)} Posen aus {len(POSEN)} Zellen, "
      f"{pp['complex'].nunique()} Komplexe\n")

s = (pp.groupby(["arm", "nfe", "complex"])
       .agg(treffer=("rmsd", lambda r: float((r < 2).mean())),
            rmsd_median=("rmsd", "median"),
            rmsd_std=("rmsd", "std"),
            n=("rmsd", "size"))
       .reset_index())
s = s.merge(frag[["complex", "n_frag"]], on="complex", how="inner")

# Breite Form: je Komplex eine Zeile, je Zelle eine Spalte.
tr = s.pivot_table(index="complex", columns=["arm", "nfe"], values="treffer")
sd = s.pivot_table(index="complex", columns=["arm", "nfe"], values="rmsd_std")
nf = s.groupby("complex")["n_frag"].first()
tr, sd = tr.dropna(), sd.dropna()
gemeinsam = tr.index.intersection(sd.index)
tr, sd = tr.loc[gemeinsam], sd.loc[gemeinsam]
print(f"{len(tr)} Komplexe mit Daten in allen sechs Zellen\n")

SD25, MIN25, MIN5 = ("SigmaDock", 25), ("Minimal", 25), ("Minimal", 5)


def waehle(maske: pd.Series, sortier: pd.Series, name: str, absteigend=True):
    k = tr.index[maske.reindex(tr.index, fill_value=False)]
    if len(k) == 0:
        print(f"  [{name}] keine Kandidaten")
        return []
    o = sortier.reindex(k).sort_values(ascending=not absteigend).index[:a.je_gruppe]
    for c in o:
        print(f"  [{name:9s}] {c:12s} frag {nf[c]:2d}  "
              f"Treffer SD25 {tr.loc[c, SD25]:.2f} MIN25 {tr.loc[c, MIN25]:.2f} "
              f"MIN5 {tr.loc[c, MIN5]:.2f}   "
              f"RMSD-Streuung SD25 {sd.loc[c, SD25]:5.2f} MIN25 {sd.loc[c, MIN25]:5.2f}")
    return list(o)


print("Auswahl:")
codes = []
codes += waehle((nf.reindex(tr.index) <= 3) & (tr[MIN25] > 0.8) & (tr[SD25] > 0.8),
                sd[MIN25], "leicht")
codes += waehle((nf.reindex(tr.index).between(4, 6)) & (tr[MIN25].between(0.3, 0.8)),
                sd[MIN25], "mittel")
codes += waehle((nf.reindex(tr.index) >= 7) & (tr[MIN25] < 0.3) & (tr[SD25] < 0.3),
                nf.reindex(tr.index), "schwer")
codes += waehle((tr[SD25] - tr[MIN25]) > 0.3, tr[SD25] - tr[MIN25], "SD>Flow")
codes += waehle((tr[MIN25] - tr[SD25]) > 0.3, tr[MIN25] - tr[SD25], "Flow>SD")

codes = list(dict.fromkeys(codes))     # Reihenfolge erhalten, doppelte raus
print(f"\n{len(codes)} Komplexe ausgewaehlt:")
print("  " + " ".join(codes))

n_dateien = len(codes) * (40 + 40 + 40 + 200 + 40 + 200)
print(f"\nGeschaetzt {n_dateien} SDF-Dateien, grob "
      f"{n_dateien * 3 / 1024:.0f} MB unkomprimiert.")

print(f"""
==================================================================
BEFEHLE FUER ARC
==================================================================

CODES="{' '.join(codes)}"

cd /data/stat-cadd/shug8458/arc_runs
PAT=$(echo $CODES | tr ' ' '|')

find {' '.join(r for r, _ in LAEUFE.values())} \\
     -path "*/learning_curve_cpu/*__sampled/results/posebusters308/*/seed_*/*.sdf" \\
  | grep -E "/($PAT)__" > /tmp/auswahl.txt

wc -l /tmp/auswahl.txt          # Erwartung: rund {n_dateien}

# Erst pruefen, dann packen. Sind es Zehntausende, stimmt der Filter nicht.
tar -czf /data/stat-cadd/shug8458/poses_pymol.tgz -T /tmp/auswahl.txt
ls -lh /data/stat-cadd/shug8458/poses_pymol.tgz

==================================================================
DANN LOKAL (PowerShell)
==================================================================

cd "C:\\Users\\julia\\Documents\\SigmaFlow\\SigmaFlow_Variants"
scp shug8458@htc-login.arc.ox.ac.uk:/data/stat-cadd/shug8458/poses_pymol.tgz .
tar -xzf poses_pymol.tgz -C .

Die Kristallposen und Proteine liegen schon lokal unter
learning_curve_min/true308/<code>/ -- die muessen NICHT geholt werden.
==================================================================""")
