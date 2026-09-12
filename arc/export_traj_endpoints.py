"""Trajektorien der 72-h-Endpunkte exportieren, fuer mehrere Kandidaten.

LAEUFT AUF ARC, weil die predictions.pt dort liegen und gross sind. Je Komplex
entstehen neun kleine Dateien:

    <CID>_crystal.sdf   <CID>_pocket.pdb   <CID>_info.py
    <CID>_{sigmadock,minimal,separate}_nfe{25,5}_steps.sdf

Aufruf:
    python arc/export_traj_endpoints.py
    KOMPLEXE=7RC3_SAH,7SCW_GSP python arc/export_traj_endpoints.py

EINHEITEN -- die Falle
    In predictions.pt ist NUR `trajectory` um den Ursprung skaliert:
        physikalisch = trajectory * DIMENSIONAL_SCALE + com
    `x0` und `x0_hat` stehen bereits physikalisch da. Wer die Umrechnung auf
    alle anwendet, landet rund 60 A daneben. Die Gegenprobe unten faengt das
    ab: der letzte Schritt MUSS x0_hat sein.

WAS GEDRUCKT WIRD
    Je Komplex die Wanderstrecke der Fragmente im Lauf SigmaFlow-Minimal bei
    25 Schritten -- also genau das, was ein schoenes Bild ausmacht:

        Start   mittlerer Abstand der Fragmentschwerpunkte zur Kristallpose
                im ERSTEN Schritt. Gross heisst: die Fragmente kommen von weit.
        Weg     mittlere zurueckgelegte Bahnlaenge je Fragment.
        Ende    mittlerer Restabstand im LETZTEN Schritt. Klein heisst: sie
                kommen an.

    Gesucht ist gross/gross/klein: weit her, sichtbar unterwegs, angekommen.
"""
import os
import sys

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Geometry import Point3D

RDLogger.DisableLog("rdApp.*")

RUNS = "/data/stat-cadd/shug8458/arc_runs"
HELFER = ("/data/stat-cadd/shug8458/SigmaFlow_Development_JulianMueller"
          "/SigmaFlow/SigmaFlow_Variants/posebusters_full_comparison")
sys.path.insert(0, HELFER)
from trajectory_geometry import (lade_pt, fragment_ids,          # noqa: E402
                                 DIMENSIONAL_SCALE)

STANDARD = ("7RC3_SAH,7SCW_GSP,7KRU_ATP,7QHL_D5P,8SLG_G5A,7Q5I_I0F,"
            "7K0V_VQP,7NLV_UJE")
KOMPLEXE = [k.strip() for k in
            os.environ.get("KOMPLEXE", STANDARD).split(",") if k.strip()]
AUS = os.environ.get("AUS", "traj_endpunkte")

ARME = [
    ("sigmadock", "SD_BASE_72H_s0_8648493", "sched239ep_emergency"),
    ("minimal",   "SF_MIN_72H_s0_8653824",  "sched255ep_emergency"),
    ("separate",  "SF_2H_72H_s0_8668713",   "sched255ep_emergency"),
]
NFES = [25, 5]

os.makedirs(AUS, exist_ok=True)
_cache = {}


def lade(lauf, tag, nfe):
    p = (f"{RUNS}/{lauf}/learning_curve_cpu/{tag}__nfe{nfe}__sampled"
         f"/results/posebusters308/snap_{tag}/seed_0/predictions.pt")
    if p not in _cache:
        if not os.path.exists(p):
            raise SystemExit(f"FEHLT: {p}")
        _cache[p] = lade_pt(p)["results"]
    return _cache[p]


def bahn(e):
    """Physikalische Trajektorie, mit Gegenprobe gegen x0_hat."""
    tr = np.asarray(e["trajectory"]) * DIMENSIONAL_SCALE + np.asarray(e["com"])
    abw = float(np.abs(tr[-1] - np.asarray(e["x0_hat"])).max())
    if abw >= 1e-4:
        raise SystemExit(f"Umrechnung falsch, max|Diff| = {abw}")
    return tr


def schreib_sdf(mol, rahmen, datei, name):
    w = Chem.SDWriter(datei)
    for t, pos in enumerate(rahmen):
        m = Chem.Mol(mol)
        c = m.GetConformer()
        for i in range(m.GetNumAtoms()):
            c.SetAtomPosition(i, Point3D(*(float(v) for v in pos[i])))
        m.SetProp("_Name", f"{name}_step{t:02d}")
        w.write(m)
    w.close()


print(f"{'Komplex':<12}{'Frag':>5}{'Atome':>7}{'Start':>8}{'Weg':>8}"
      f"{'Ende':>7}   Minimal, 25 Schritte")
gut = []
for cid_kurz in KOMPLEXE:
    cid = f"{cid_kurz}::{cid_kurz}_ligand"
    try:
        e0 = lade(ARME[1][1], ARME[1][2], 25)[cid][0]
    except KeyError:
        print(f"{cid_kurz:<12}   nicht im Lauf enthalten")
        continue

    mol = e0["lig_ref"]
    na = mol.GetNumAtoms()
    kri_mol = Chem.MolFromMolFile(str(e0["ligand_path"]), sanitize=False,
                                  removeHs=True)
    if kri_mol is None or kri_mol.GetNumAtoms() != na:
        print(f"{cid_kurz:<12}   Kristallpose passt nicht, uebersprungen")
        continue
    kristall = kri_mol.GetConformer().GetPositions()

    tr = bahn(e0)
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    fid = fragment_ids(tr, bonds, na)
    schnitt = [(i, j) for i, j in bonds if fid[i] != fid[j]]
    ids = sorted(set(fid.tolist()))

    start, weg, ende = [], [], []
    for f in ids:
        idx = np.where(fid == f)[0]
        z = tr[:, idx, :].mean(axis=1)
        ziel = kristall[idx].mean(0)
        start.append(float(np.linalg.norm(z[0] - ziel)))
        weg.append(float(np.linalg.norm(np.diff(z, axis=0), axis=1).sum()))
        ende.append(float(np.linalg.norm(z[-1] - ziel)))
    print(f"{cid_kurz:<12}{len(ids):>5}{na:>7}{np.mean(start):>8.1f}"
          f"{np.mean(weg):>8.1f}{np.mean(ende):>7.1f}")

    # alle sechs Zellen schreiben
    schreib_sdf(mol, [kristall], f"{AUS}/{cid_kurz}_crystal.sdf", cid_kurz)
    if e0.get("prot_ref") is not None:
        Chem.MolToPDBFile(e0["prot_ref"], f"{AUS}/{cid_kurz}_pocket.pdb")
    with open(f"{AUS}/{cid_kurz}_info.py", "w", encoding="utf-8") as fh:
        fh.write(f"CID = {cid_kurz!r}\nN_ATOME = {na}\nN_FRAG = {len(ids)}\n")
        fh.write(f"N_STEPS = {tr.shape[0]}\nSCHNITT = {schnitt!r}\n")
        fh.write(f"FRAGMENT_VON_ATOM = {fid.tolist()!r}\n")
    for arm, lauf, tag in ARME:
        for nfe in NFES:
            e = lade(lauf, tag, nfe)[cid][0]
            schreib_sdf(mol, bahn(e),
                        f"{AUS}/{cid_kurz}_{arm}_nfe{nfe}_steps.sdf",
                        f"{cid_kurz}_{arm}_nfe{nfe}")
    gut.append(cid_kurz)

print()
print(f"{len(gut)} Komplexe exportiert nach {AUS}/")
print("Gesucht ist ein grosser Start, ein grosser Weg und ein kleines Ende: "
      "die Fragmente kommen von weit, sind sichtbar unterwegs und kommen an.")
