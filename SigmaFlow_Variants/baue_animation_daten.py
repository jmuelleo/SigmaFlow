"""Bereitet die 7KRU-Trajektorien fuer eine Web-Animation auf.

WAS DARAUS WIRD
    Eine JSON-Datei mit allen sechs Trajektorien, der Kristallpose und der
    Bindetasche. Sie wird in eine eigenstaendige HTML-Seite eingebettet, die
    ohne Server und ohne Nachladen funktioniert -- also auch aus einem
    GitHub-Pages-Verzeichnis heraus.

DIE DURCHTRENNTEN BINDUNGEN
    SigmaDocks Fragmentierung schneidet den Liganden an vier Bindungen
    (`SCHNITT` in <cid>_info.py). Waehrend der Integration bewegt sich jedes
    Fragment als eigener starrer Koerper; die vier Bindungen existieren in
    diesen Zwischenzustaenden noch nicht.

    In den gespeicherten SDF stehen sie trotzdem drin -- die Datei traegt
    immer die volle Konnektivitaet. Wuerde man sie so zeichnen, saehe man
    Gummibaender zwischen auseinanderfliegenden Fragmenten, und der ganze
    Punkt der Abbildung ginge verloren. Deshalb werden sie hier fuer alle
    Schritte AUSSER dem letzten entfernt.

    Der letzte Schritt behaelt sie: dort fuegen sich die Fragmente zum
    fertigen Liganden zusammen, und genau das soll die Animation zeigen.

WARUM SDF-TEXT UND NICHT KOORDINATEN
    3Dmol.js liest SDF direkt und kann mehrere Records als Einzelbilder einer
    Animation laden. Ein eigenes Format waere kleiner, aber wir muessten die
    Molekuelaufbau-Logik im Browser nachbauen -- mehr Code, mehr Fehler.

Aufruf:
    python SigmaFlow_Variants/baue_animation_daten.py
"""
import argparse
import json
import os
import re
import sys

_HIER = os.path.dirname(os.path.abspath(__file__))
QUELLE = os.path.join(_HIER, "posebusters_full_comparison", "pymol_steps")

from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

ZELLEN = [
    ("SigmaDock", 25, "sigmadock_nfe25"),
    ("SigmaDock", 5, "sigmadock_nfe5"),
    ("SigmaFlow-Minimal", 25, "minimal_nfe25"),
    ("SigmaFlow-Minimal", 5, "minimal_nfe5"),
    ("SigmaFlow-Separate", 25, "separate_nfe25"),
    ("SigmaFlow-Separate", 5, "separate_nfe5"),
]

p = argparse.ArgumentParser()
p.add_argument("--cid", default="7KRU_ATP")
p.add_argument("--out", default=os.path.join(_HIER, "animation_daten.json"))
a = p.parse_args()

info_datei = os.path.join(QUELLE, f"{a.cid}_info.py")
if not os.path.isfile(info_datei):
    sys.exit(f"ABBRUCH: {info_datei} fehlt.")
info = {}
exec(open(info_datei).read(), info)          # noqa: S102 -- eigene Datei
SCHNITT = {tuple(sorted(b)) for b in info["SCHNITT"]}
print(f"{a.cid}: {info['N_ATOME']} Atome, {info['N_FRAG']} Fragmente, "
      f"{len(SCHNITT)} Schnitte {sorted(SCHNITT)}")


def ohne_schnitte(mol: Chem.Mol) -> Chem.Mol:
    """Kopie ohne die durchtrennten Bindungen."""
    rw = Chem.RWMol(mol)
    for i, j in sorted(SCHNITT, reverse=True):
        if rw.GetBondBetweenAtoms(i, j) is not None:
            rw.RemoveBond(i, j)
    return rw.GetMol()


def sdf_text(mols: list) -> str:
    from io import StringIO
    puffer = StringIO()
    w = Chem.SDWriter(puffer)
    w.SetKekulize(False)
    for m in mols:
        w.write(m)
    w.close()
    return puffer.getvalue()


daten = {"cid": a.cid, "n_atome": info["N_ATOME"], "n_frag": info["N_FRAG"],
         "fragment_von_atom": info["FRAGMENT_VON_ATOM"], "zellen": []}

for arm, nfe, schluessel in ZELLEN:
    f = os.path.join(QUELLE, f"{a.cid}_{schluessel}_steps.sdf")
    if not os.path.isfile(f):
        sys.exit(f"ABBRUCH: {f} fehlt.")
    mols = [m for m in Chem.SDMolSupplier(f, sanitize=False, removeHs=False)
            if m is not None]
    if not mols:
        sys.exit(f"ABBRUCH: {f} enthaelt keine lesbaren Records.")
    # Alle ausser dem letzten ohne die Schnittbindungen.
    frames = [ohne_schnitte(m) for m in mols[:-1]] + [mols[-1]]
    n_bind = [m.GetNumBonds() for m in frames]
    daten["zellen"].append({
        "arm": arm, "nfe": nfe, "schluessel": schluessel,
        "n_schritte": len(frames),
        "sdf": sdf_text(frames),
    })
    print(f"  {arm:20s} NFE {nfe:2d}: {len(frames):2d} Schritte, "
          f"Bindungen {n_bind[0]} waehrend -> {n_bind[-1]} am Ende")

for name, datei in (("kristall", f"{a.cid}_crystal.sdf"),
                    ("tasche", f"{a.cid}_pocket.pdb")):
    f = os.path.join(QUELLE, datei)
    if not os.path.isfile(f):
        sys.exit(f"ABBRUCH: {f} fehlt.")
    daten[name] = open(f, encoding="utf-8").read()
    print(f"  {name:20s}: {len(daten[name]) / 1024:.1f} KB")

with open(a.out, "w", encoding="utf-8") as fh:
    json.dump(daten, fh)
groesse = os.path.getsize(a.out) / 1024
print(f"\n{a.out}\n{groesse:.0f} KB")
if groesse > 4000:
    print("WARNUNG: ueber 4 MB -- fuer eine eingebettete Seite viel.")
