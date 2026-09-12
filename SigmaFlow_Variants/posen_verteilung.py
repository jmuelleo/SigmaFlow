"""Wie breit streuen die vorgeschlagenen Posen eines Komplexes -- STRUKTURELL?

WARUM DIE BISHERIGE ZAHL DIE FRAGE NICHT BEANTWORTET
    Frueher wurde die Standardabweichung des RMSD ZUR KRISTALLPOSE ueber die
    Seeds gemessen. Zwei Posen koennen aber denselben RMSD zur Wahrheit haben
    und trotzdem auf gegenueberliegenden Seiten der Tasche liegen. Gesucht ist
    die Streuung der Posen UNTEREINANDER.

DREI MASSE, ALLE OHNE UEBERLAGERUNG
    Die Posen liegen bereits im selben Koordinatensystem (derselbe Rezeptor).
    Sie vor dem Vergleich aufeinander zu legen wuerde genau das wegrechnen,
    was uns interessiert -- deshalb KEIN Kabsch, keine Superposition.

      zentroid_std   Streuung der Ligandschwerpunkte. Rein translatorisch:
                     "landet der Ligand ueberhaupt an derselben Stelle?"
      rmsf           Wurzel aus der mittleren Positionsvarianz ueber alle
                     Atome. Enthaelt Translation, Rotation und Konformation.
      paar_rmsd      Mittlerer paarweiser RMSD, das klassische Diversitaetsmass.
                     Bei 200 Posen waeren das 19.900 Paare je Zelle; gerechnet
                     wird auf einer Zufallsstichprobe von PAARE Paaren.

    Symmetrie wird NICHT beruecksichtigt (kein GetBestRMS). Bei symmetrischen
    Gruppen -- ein gedrehter Phenylring etwa -- ueberschaetzt das die
    Diversitaet leicht. Fuer den VERGLEICH zwischen Armen ist das unschaedlich,
    weil es alle gleich trifft; als Absolutwert ist es eine Obergrenze.

DIE ATOMREIHENFOLGE WIRD GEPRUEFT, NICHT ANGENOMMEN
    Alle Posen eines Komplexes stammen aus derselben Pipeline und sollten
    dieselbe Reihenfolge haben. Sollten. Das Skript vergleicht Atomzahl und
    Elementfolge gegen die erste Pose und bricht bei Abweichung ab, statt
    stillschweigend Unsinn zu rechnen.

Aufruf:
    python SigmaFlow_Variants/posen_verteilung.py
    python SigmaFlow_Variants/posen_verteilung.py --pymol --max-posen 40
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

_HIER = os.path.dirname(os.path.abspath(__file__))

from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

# Endpunkt-Zellen. Der Etikettenteil "emergency" ist der Endpunkt; die
# "at_XXXh" sind Snapshots und gehoeren NICHT hierher -- ein Glob ueber
# "*__sampled" wuerde sie mitnehmen und einen Zwischenstand als Ergebnis
# ausgeben. Das ist hier schon einmal passiert.
ZELLEN = [
    ("SigmaDock", 25, "SD_BASE_72H_s0_8648493", "sched239ep_emergency__nfe25__sampled"),
    ("SigmaDock",  5, "SD_BASE_72H_s0_8648493", "sched239ep_emergency__nfe5__sampled"),
    ("Minimal",   25, "SF_MIN_72H_s0_8653824", "sched255ep_emergency__nfe25__sampled"),
    ("Minimal",    5, "SF_MIN_72H_s0_8653824", "sched255ep_emergency__nfe5__sampled"),
    ("Separate",  25, "SF_2H_72H_s0_8668713", "sched255ep_emergency__nfe25__sampled"),
    ("Separate",   5, "SF_2H_72H_s0_8668713", "sched255ep_emergency__nfe5__sampled"),
]
FARBEN = {"SigmaDock": "salmon", "Minimal": "skyblue", "Separate": "palegreen"}


def zellpfad(lauf: str, tag: str) -> str:
    schnipsel = tag.split("__")[0]          # z. B. sched255ep_emergency
    return os.path.join(_HIER, lauf, "learning_curve_cpu", tag, "results",
                        "posebusters308", f"snap_{schnipsel}")


def lade_posen(wurzel: str, code: str):
    """Alle Posen eines Komplexes -> (Koordinaten [N, A, 3], Dateiliste)."""
    dateien = sorted(glob.glob(os.path.join(wurzel, "seed_*", f"{code}__*.sdf")),
                     key=lambda f: (int(re.search(r"seed_(\d+)", f).group(1)), f))
    koord, benutzt, referenz = [], [], None
    for f in dateien:
        m = Chem.MolFromMolFile(f, sanitize=False, removeHs=False)
        if m is None or m.GetNumConformers() == 0:
            continue
        elemente = tuple(a.GetSymbol() for a in m.GetAtoms())
        if referenz is None:
            referenz = elemente
        elif elemente != referenz:
            sys.exit(f"ABBRUCH: {f} hat eine andere Atomfolge als die erste "
                     f"Pose ({len(elemente)} gegen {len(referenz)} Atome). "
                     f"Ein Vergleich ohne Zuordnung waere sinnlos.")
        koord.append(m.GetConformer().GetPositions())
        benutzt.append(f)
    if not koord:
        return None, []
    return np.asarray(koord, dtype=float), benutzt


def masse(x: np.ndarray, paare: int, rng) -> dict:
    """x: [N, A, 3]. Alle Masse OHNE Ueberlagerung."""
    n = len(x)
    zent = x.mean(axis=1)                                   # [N, 3]
    zentroid_std = float(np.sqrt(((zent - zent.mean(0)) ** 2).sum(1).mean()))
    varianz = x.var(axis=0)                                 # [A, 3]
    rmsf = float(np.sqrt(varianz.sum(1).mean()))
    if n < 2:
        return {"n": n, "zentroid_std": zentroid_std, "rmsf": rmsf,
                "paar_rmsd": float("nan")}
    alle = n * (n - 1) // 2
    if alle <= paare:
        i, j = np.triu_indices(n, k=1)
    else:
        i = rng.integers(0, n, size=paare)
        j = rng.integers(0, n, size=paare)
        gut = i != j
        i, j = i[gut], j[gut]
    d = np.sqrt(((x[i] - x[j]) ** 2).sum(-1).mean(-1))      # [P]
    return {"n": n, "zentroid_std": zentroid_std, "rmsf": rmsf,
            "paar_rmsd": float(d.mean())}


p = argparse.ArgumentParser()
p.add_argument("--codes", nargs="*", default=None,
               help="Standard: alle Komplexe, die in allen Zellen vorliegen")
p.add_argument("--paare", type=int, default=4000)
p.add_argument("--pymol", action="store_true", help=".pml je Komplex schreiben")
p.add_argument("--max-posen", type=int, default=40,
               help="je Zelle hoechstens so viele Posen in die .pml")
p.add_argument("--true", default=os.path.join(_HIER, "learning_curve_min", "true308"))
p.add_argument("--out", default=os.path.join(_HIER, "posen_verteilung.csv"))
a = p.parse_args()

vorhanden = {}
for arm, nfe, lauf, tag in ZELLEN:
    w = zellpfad(lauf, tag)
    if not os.path.isdir(w):
        sys.exit(f"ABBRUCH: Zelle fehlt: {w}")
    codes = {os.path.basename(f).split("__")[0]
             for f in glob.glob(os.path.join(w, "seed_*", "*.sdf"))}
    vorhanden[(arm, nfe)] = (w, codes)

gemeinsam = set.intersection(*(c for _, c in vorhanden.values()))
codes = sorted(a.codes) if a.codes else sorted(gemeinsam)
fehlend = [c for c in codes if c not in gemeinsam]
if fehlend:
    sys.exit(f"ABBRUCH: nicht in allen Zellen vorhanden: {fehlend}")
print(f"{len(codes)} Komplexe: {' '.join(codes)}\n")

frag = None
fp = os.path.join(_HIER, "fragmentzahl_pb308.csv")
if os.path.isfile(fp):
    f = pd.read_csv(fp)
    frag = f.set_index("complex")["n_frag"].to_dict()

rng = np.random.default_rng(20260907)
zeilen = []
for code in codes:
    for (arm, nfe), (w, _) in vorhanden.items():
        x, dateien = lade_posen(w, code)
        if x is None:
            print(f"  {code} {arm} NFE {nfe}: keine Posen")
            continue
        m = masse(x, a.paare, rng)
        m.update(complex=code, arm=arm, nfe=nfe,
                 n_frag=(frag or {}).get(code, np.nan))
        zeilen.append(m)

t = pd.DataFrame(zeilen)[["complex", "n_frag", "arm", "nfe", "n",
                          "zentroid_std", "rmsf", "paar_rmsd"]]
t.to_csv(a.out, index=False)
print(f"{len(t)} Zeilen nach {a.out}\n")

print("=== Streuung der Posen UNTEREINANDER, Angstroem, ohne Ueberlagerung ===")
for code, g in t.groupby("complex"):
    nf = g["n_frag"].iloc[0]
    print(f"\n{code}  ({nf:.0f} Fragmente)" if pd.notna(nf) else f"\n{code}")
    print(f"  {'Zelle':<16}{'n':>5}{'zentroid_std':>14}{'rmsf':>9}{'paar_rmsd':>11}")
    for _, r in g.sort_values(["arm", "nfe"]).iterrows():
        print(f"  {r['arm'] + ' ' + str(r['nfe']):<16}{r['n']:5d}"
              f"{r['zentroid_std']:14.2f}{r['rmsf']:9.2f}{r['paar_rmsd']:11.2f}")

if len(codes) >= 5:
    print("\n=== Mittel ueber die Komplexe ===")
    print(t.groupby(["arm", "nfe"])[["zentroid_std", "rmsf", "paar_rmsd"]]
           .mean().to_string(float_format=lambda x: f"{x:7.2f}"))

# --- PyMOL ---------------------------------------------------------------
if a.pymol:
    for code in codes:
        prot = os.path.join(a.true, code, f"{code}_protein.pdb")
        lig = os.path.join(a.true, code, f"{code}_ligands.sdf")
        if not os.path.isfile(prot) or not os.path.isfile(lig):
            print(f"  {code}: Referenz fehlt unter {a.true}, .pml uebersprungen")
            continue
        z = [f"# {code} -- Verteilung der Seeds, erzeugt von posen_verteilung.py",
             "# Aufruf:  pymol " + f"{code}_verteilung.pml", "",
             "bg_color white", "set ray_opaque_background, 0", "",
             f'load {prot.replace(os.sep, "/")}, rezeptor',
             "hide everything, rezeptor",
             "show cartoon, rezeptor", "color grey80, rezeptor",
             "set cartoon_transparency, 0.6, rezeptor", "",
             f'load {lig.replace(os.sep, "/")}, kristall',
             "show sticks, kristall", "color yellow, kristall",
             "set stick_radius, 0.25, kristall", ""]
        for (arm, nfe), (w, _) in vorhanden.items():
            _, dateien = lade_posen(w, code)
            if not dateien:
                continue
            schritt = max(1, len(dateien) // a.max_posen)
            nimm = dateien[::schritt][:a.max_posen]
            gruppe = f"{arm}_nfe{nfe}".lower()
            z.append(f"# --- {arm} NFE {nfe}: {len(nimm)} von {len(dateien)} Posen")
            for k, f in enumerate(nimm):
                z.append(f'load {f.replace(os.sep, "/")}, {gruppe}_{k:03d}')
            z += [f"group {gruppe}, {gruppe}_*",
                  f"show lines, {gruppe}",
                  f"color {FARBEN[arm]}, {gruppe}",
                  f"set line_width, 1.5, {gruppe}",
                  f"disable {gruppe}", ""]
        z += ["orient kristall", "zoom kristall, 8", "",
              "# Zum Anschauen jeweils EINE Gruppe einschalten, z. B.:",
              "#   enable sigmadock_nfe25", "#   enable minimal_nfe25"]
        ziel = os.path.join(_HIER, f"{code}_verteilung.pml")
        with open(ziel, "w", encoding="utf-8") as fh:
            fh.write("\n".join(z) + "\n")
        print(f"  geschrieben: {os.path.relpath(ziel, _HIER)}")
