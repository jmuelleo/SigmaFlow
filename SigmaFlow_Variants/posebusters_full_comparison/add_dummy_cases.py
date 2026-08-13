"""Fuegt die 10 Dummy-Komplexe zur PyMOL-Fallsammlung hinzu.

Schreibt nach pymol_cases/ und aktualisiert pymol_cases/cases.json. Das
PyMOL-Skript im Hauptbericht liest diese Datei, neue Faelle erscheinen dort
also automatisch - der Bericht muss nicht neu erzeugt werden.

[!]️ VORBEHALT ZU DEN DUMMY-KOMPLEXEN
Die 10 (1G9V_RQ3, 1HWI_115, 1MZC_BNE, 1OWE_675, 1R1H_BIR, 1S3V_TQD,
1U1C_BAU, 1V4S_MRK, 1YQY_915, 2BSM_BSM) sind klassische PDBBind-Eintraege und
mit hoher Wahrscheinlichkeit Trainingsdaten. Ausserdem wurden sie fuer die
Overfit-Laeufe benutzt. KEIN Testset - nur zum Ansehen.

Usage:
  # Voreinstellung: die lokal vorhandenen 3h-Laeufe von Juli 2026 (VOR dem Frame-Fix)
  python add_dummy_cases.py

  # spaeter, mit den von ARC geholten aktuellen Laeufen:
  python add_dummy_cases.py <sigmaflow_dir> <sigmadock_dir> "<Label>" ["<Warnung>"]
"""

import glob
import json
import os
import sys

import numpy as np
from rdkit import Chem

from ligand_reference import load_copies

DUMMY_DIR = "../../SigmaFlow_Minimal/notebooks/dummy_data"
CASE_DIR = "pymol_cases"

DEFAULT_SF = "../../archive/SigmaFlow_Development_pre_framefix/sampling_output_prodhparams"
DEFAULT_SD = "../../archive/SigmaFlow_Development_pre_framefix/sampling_output_sigmadock_prodhparams"
DEFAULT_LABEL = "Dummy (3h, VOR Frame-Fix)"
DEFAULT_WARN = ("Vorhersagen von Juli 2026, also VOR dem Frame-Fix vom 2026-08-09. "
                "Zeigt den alten, kaputten Rotationskanal - nicht das aktuelle Modell. "
                "Ausserdem Trainingsdaten-Ueberlappung: kein Benchmark.")


def kabsch(P, Q):
    cP, cQ = P.mean(0), Q.mean(0)
    Pc, Qc = P - cP, Q - cQ
    U, _, Vt = np.linalg.svd(Qc.T @ Pc)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    ali = float(np.sqrt(((Pc - Qc @ R.T) ** 2).sum(1).mean()))
    ang = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    return ali, ang, float(np.linalg.norm(cP - cQ))


def one(path):
    m = load_copies(path)
    return m[0] if m else None


def main():
    sf_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SF
    sd_dir = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_SD
    label = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_LABEL
    warn = sys.argv[4] if len(sys.argv) > 4 else (DEFAULT_WARN if len(sys.argv) <= 1 else "")

    for d in (sf_dir, sd_dir):
        if not os.path.isdir(d):
            print(f"FEHLT: {d}")
            sys.exit(1)
    os.makedirs(CASE_DIR, exist_ok=True)

    cids = sorted(d for d in os.listdir(DUMMY_DIR)
                  if os.path.isdir(os.path.join(DUMMY_DIR, d)))
    print(f"Dummy-Komplexe gefunden: {len(cids)}")
    print(f"SigmaFlow : {sf_dir}")
    print(f"SigmaDock : {sd_dir}")
    print(f"Label     : {label}\n")

    cj = os.path.join(CASE_DIR, "cases.json")
    reg = {}
    if os.path.exists(cj):
        try:
            reg = json.load(open(cj, encoding="utf-8"))
        except Exception:
            reg = {}
    reg = {k: v for k, v in reg.items() if v.get("set") != label}   # idempotent

    print(f"{'Komplex':<12}{'SF RMSD':>9}{'SD RMSD':>9}{'SF orient':>11}{'SD orient':>11}"
          f"{'SF ausger.':>12}{'SF Schwerp.':>13}")
    print("-" * 78)
    rows, n_ok = [], 0
    for cid in cids:
        tp = os.path.join(DUMMY_DIR, cid, f"{cid}_ligand.sdf")   # SINGULAR = kanonisch
        sfp = glob.glob(os.path.join(sf_dir, f"{cid}__*_seed0.sdf"))
        sdp = glob.glob(os.path.join(sd_dir, f"{cid}__*_seed0.sdf"))
        if not (os.path.exists(tp) and sfp and sdp):
            print(f"{cid:<12}  (unvollstaendig - uebersprungen)")
            continue
        mt, mf, md = one(tp), one(sfp[0]), one(sdp[0])
        if mt is None or mf is None or md is None:
            continue
        Q = mt.GetConformer().GetPositions()
        Pf = mf.GetConformer().GetPositions()
        Pd = md.GetConformer().GetPositions()
        if Pf.shape != Q.shape or Pd.shape != Q.shape:
            print(f"{cid:<12}  (Atomzahl passt nicht - uebersprungen)")
            continue
        rf = float(np.sqrt(((Pf - Q) ** 2).sum(1).mean()))
        rd = float(np.sqrt(((Pd - Q) ** 2).sum(1).mean()))
        af, gf, cf = kabsch(Pf, Q)
        _, gd, _ = kabsch(Pd, Q)
        for tag, mol in (("TRUE", mt), ("SIGMAFLOW", mf), ("SIGMADOCK", md)):
            with Chem.SDWriter(os.path.join(CASE_DIR, f"{cid}_{tag}.sdf")) as wr:
                wr.write(mol)
        reg[cid] = {"set": label, "rolle": "Dummy-Komplex",
                    "rmsd_sf": round(rf, 2), "rmsd_sd": round(rd, 2),
                    "orient_sf": round(gf, 0), "aligned_sf": round(af, 2)}
        if warn:
            reg[cid]["warnung"] = warn
        rows.append((cid, rf, rd, gf, gd, af, cf))
        n_ok += 1
        print(f"{cid:<12}{rf:>9.2f}{rd:>9.2f}{gf:>9.0f}Grad{gd:>9.0f}Grad{af:>12.2f}{cf:>13.2f}")

    if rows:
        a = np.array([[r[1], r[2], r[3], r[4], r[5], r[6]] for r in rows])
        print("-" * 78)
        print(f"{'MEDIAN':<12}{np.median(a[:,0]):>9.2f}{np.median(a[:,1]):>9.2f}"
              f"{np.median(a[:,2]):>9.0f}Grad{np.median(a[:,3]):>9.0f}Grad"
              f"{np.median(a[:,4]):>12.2f}{np.median(a[:,5]):>13.2f}")
        print(f"\nSigmaFlow besser bei {int((a[:,0] < a[:,1]).sum())} von {len(rows)} Komplexen")
        print("Zufallsbaseline Orientierung: Median 132.3 Grad")

    with open(cj, "w", encoding="utf-8") as f:
        json.dump(reg, f, indent=2)
    print(f"\n{n_ok} Faelle nach {CASE_DIR}/ geschrieben, cases.json aktualisiert.")
    print("In PyMOL erscheinen sie sofort - einfach  liste  eingeben.")
    if warn:
        print(f"\n[!] {warn}")


if __name__ == "__main__":
    main()
