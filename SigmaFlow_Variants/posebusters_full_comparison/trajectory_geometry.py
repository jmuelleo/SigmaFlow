"""Wie bewegen sich die Fragmente waehrend der Integration?

EINGABE
    <root>/results/posebusters/*/seed_*/predictions.pt
    erzeugt von scripts/sample.py; je Komplex ein Eintrag mit

        trajectory  [T, n_atoms, 3]   Zustand nach jedem Integrationsschritt
        x0          [n_atoms, 3]      wahre Pose
        x0_hat      [n_atoms, 3]      Endvorhersage
        com         [3]               Taschenschwerpunkt
        lig_ref     RDKit-Mol         Referenzligand, liefert die Bindungen

KOORDINATENRAHMEN
    trajectory liegt im internen Modellrahmen. Die Ruecktransformation ist

        physikalisch = trajectory * dimensional_scale + com

    mit dimensional_scale = 2.7 (HPARAMS.general.dimensional_scale). Der Wert
    ist numerisch gegen x0_hat geprueft: trajectory[-1] * 2.7 + com stimmt mit
    x0_hat auf drei Nachkommastellen ueberein.

FRAGMENTZERLEGUNG
    Der Zustand hat sechs Freiheitsgrade je Fragment, Fragmente bewegen sich
    also als starre Koerper. Eine Bindung liegt genau dann INNERHALB eines
    Fragments, wenn ihre Laenge ueber die ganze Trajektorie konstant bleibt;
    die Schnittbindungen sind die veraenderlichen. Das ist robuster als der
    Vergleich mit der wahren Pose, weil es nur die Trajektorie selbst braucht.

    Gegenprobe an einem Beispiel: 13 von 14 Bindungen konstant auf 0.01 A, die
    eine veraenderliche mit Spanne 7.9 A. Die Trennung ist eindeutig.

    Fragmente unter MIN_FRAG_ATOMS Atomen werden bei der Rotation ausgelassen
    -- fuer weniger als drei Atome ist eine Drehung nicht definiert.

REFERENZ -- WICHTIG
    Als Wahrheit dient der Referenzligand aus true_ligands/<cid>_ligands.sdf,
    naechstgelegene kristallographische Kopie ueber best_copy (84 der 209
    Dateien enthalten mehrere).

    NICHT x0 aus predictions.pt. Das war die erste Fassung und ist FALSCH,
    sobald graph.sample_conformer=true gesetzt ist: x0 traegt dann den
    generierten Konformer in seinem eigenen Bezugssystem, im Mittel 50.4 A
    von der wahren Pose entfernt. Im bound-Modus stimmt x0 exakt mit der
    wahren Pose ueberein, weshalb der Fehler dort nicht auffiel.

    Gegenprobe, die den Bruch lokalisiert hat: x0_hat stimmt in BEIDEN Modi
    auf 0.0000 A mit der geschriebenen SDF ueberein, und
    trajectory[-1] * 2.7 + com ebenfalls -- nur die Referenz war falsch.

AUSGABE
    traj_agg_<arm>.csv    je Schritt ueber alle Posen gemittelt, immer
    traj_pose_<arm>.csv   je Pose und Schritt, nur fuer die ersten
                          --detail_seeds Seeds (sonst wird die Datei riesig)

AUFRUF
    python trajectory_geometry.py <arm> <root> [--detail_seeds 5] [--max_seeds N]
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import pathlib
import re

import numpy as np
import torch

from ligand_reference import best_copy, load_copies

DIMENSIONAL_SCALE = 2.7   # HPARAMS.general.dimensional_scale
MIN_FRAG_ATOMS = 3        # weniger -> Drehung nicht definiert
BOND_TOL = 0.01           # Angstrom; Spanne, ab der eine Bindung als
                          # veraenderlich gilt (Schnittbindung)


def lade_pt(pfad: str):
    """torch.load mit Umweg um die PosixPath-Objekte aus dem ARC-Pickle."""
    orig = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath if os.name == "nt" else orig
    try:
        return torch.load(pfad, map_location="cpu", weights_only=False)
    finally:
        pathlib.PosixPath = orig


def fragment_ids(traj: np.ndarray, bonds: list[tuple[int, int]], n_atoms: int) -> np.ndarray:
    """Zusammenhangskomponenten ueber die Bindungen konstanter Laenge."""
    eltern = list(range(n_atoms))

    def wurzel(a: int) -> int:
        while eltern[a] != a:
            eltern[a] = eltern[eltern[a]]
            a = eltern[a]
        return a

    if bonds:
        i_idx = np.array([i for i, _ in bonds])
        j_idx = np.array([j for _, j in bonds])
        laengen = np.linalg.norm(traj[:, i_idx] - traj[:, j_idx], axis=2)  # [T, n_bonds]
        spanne = laengen.max(axis=0) - laengen.min(axis=0)
        for (i, j), sp in zip(bonds, spanne):
            if sp < BOND_TOL:
                ra, rb = wurzel(i), wurzel(j)
                if ra != rb:
                    eltern[rb] = ra

    roots, fid = {}, np.empty(n_atoms, dtype=int)
    for a in range(n_atoms):
        r = wurzel(a)
        if r not in roots:
            roots[r] = len(roots)
        fid[a] = roots[r]
    return fid


def kabsch_winkel(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Drehwinkel in Grad zwischen P[t] und Q, gestapelt ueber t.

    P: [T, n, 3]. Q ist entweder [n, 3] (feste Referenz, etwa die wahre Pose)
    oder [T, n, 3] (Schritt-gegen-Schritt). Beide werden zentriert; die Drehung
    ist die orthogonale Prokrustes-Loesung mit korrigierter Determinante, damit
    keine Spiegelung entsteht.

    Der Schritt-gegen-Schritt-Fall ist der wichtigere: er misst, WIEVIEL sich
    ein Fragment bewegt, unabhaengig davon, ob die Bewegung hilft. Weil dabei
    dieselben Atomindizes verglichen werden, ist er vom Symmetrievorbehalt der
    absoluten Winkel nicht betroffen.
    """
    Pc = P - P.mean(axis=1, keepdims=True)
    if Q.ndim == 2:
        Qc = Q - Q.mean(axis=0, keepdims=True)
        H = np.einsum("tni,nj->tij", Pc, Qc)
    else:
        Qc = Q - Q.mean(axis=1, keepdims=True)
        H = np.einsum("tni,tnj->tij", Pc, Qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(np.einsum("tij,tjk->tik", U, Vt)))
    D = np.zeros((len(P), 3, 3))
    D[:, 0, 0] = D[:, 1, 1] = 1.0
    D[:, 2, 2] = d
    R = np.einsum("tij,tjk,tkl->til", U, D, Vt)
    spur = np.trace(R, axis1=1, axis2=2)
    return np.degrees(np.arccos(np.clip((spur - 1.0) / 2.0, -1.0, 1.0)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arm")
    ap.add_argument("root")
    ap.add_argument("--true_dir", default="true_ligands",
                    help="Verzeichnis mit <cid>_ligands.sdf")
    ap.add_argument("--detail_seeds", type=int, default=5,
                    help="fuer wie viele Seeds die Datei je Pose geschrieben wird")
    ap.add_argument("--max_seeds", type=int, default=None,
                    help="nur Seeds kleiner als dieser Wert auswerten")
    a = ap.parse_args()

    pts = sorted(
        ((int(re.search(r"seed_(\d+)", p).group(1)), p)
         for p in glob.glob(os.path.join(a.root, "**", "seed_*", "predictions.pt"),
                            recursive=True)),
        key=lambda x: x[0])
    if a.max_seeds is not None:
        pts = [(s, p) for s, p in pts if s < a.max_seeds]
    if not pts:
        raise SystemExit(f"ABBRUCH: keine predictions.pt unter {a.root}")
    print(f"{a.arm}: {len(pts)} Seeds gefunden ({pts[0][0]}..{pts[-1][0]})", flush=True)

    detail = open(f"traj_pose_{a.arm}.csv", "w", encoding="utf-8", newline="")
    dw = csv.writer(detail)
    dw.writerow(["complex", "seed", "step", "n_frag", "n_atoms",
                 "rmsd", "rot_mean_deg", "rot_max_deg", "trans_mean_A", "com_offset_A",
                 "step_rot_deg", "step_trans_A"])

    # Aggregation je Schritt. Die Schrittzahl steht erst beim ersten Komplex fest.
    summe: dict[int, np.ndarray] = {}
    anzahl: dict[int, int] = {}
    T_gesehen = set()
    fehlend = 0

    for seed, pfad in pts:
        d = lade_pt(pfad)
        for key, lst in d["results"].items():
            for e in lst:
                cid = key.split("::")[0]
                traj = e["trajectory"].numpy() * DIMENSIONAL_SCALE + e["com"].numpy()
                T, n_atoms, _ = traj.shape
                T_gesehen.add(T)
                mol = e["lig_ref"]
                bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
                if mol.GetNumAtoms() != n_atoms:
                    continue          # Atomzahl passt nicht -> nicht auswertbar
                tp = os.path.join(a.true_dir, f"{cid}_ligands.sdf")
                if not os.path.isfile(tp):
                    fehlend += 1
                    continue
                mt, _, _ = best_copy(traj[-1], load_copies(tp))
                if mt is None or mt.GetNumAtoms() != n_atoms:
                    fehlend += 1
                    continue
                wahr = mt.GetConformer().GetPositions()
                fid = fragment_ids(traj, bonds, n_atoms)

                rot_sum = np.zeros(T)
                rot_max = np.zeros(T)
                tra_sum = np.zeros(T)
                # Bewegung von t nach t+1; Index t traegt den Schritt t->t+1,
                # der letzte Eintrag bleibt 0.
                dstep_rot = np.zeros(T)
                dstep_tra = np.zeros(T)
                n_rot = 0
                for f in np.unique(fid):
                    sel = fid == f
                    P, Q = traj[:, sel], wahr[sel]
                    versatz = np.linalg.norm(P.mean(axis=1) - Q.mean(axis=0), axis=1)
                    tra_sum += versatz
                    dstep_tra[:-1] += np.linalg.norm(
                        P[1:].mean(axis=1) - P[:-1].mean(axis=1), axis=1)
                    if sel.sum() >= MIN_FRAG_ATOMS:
                        w = kabsch_winkel(P, Q)
                        rot_sum += w
                        rot_max = np.maximum(rot_max, w)
                        dstep_rot[:-1] += kabsch_winkel(P[1:], P[:-1])
                        n_rot += 1
                n_frag = len(np.unique(fid))
                rot_mean = rot_sum / n_rot if n_rot else np.full(T, np.nan)
                tra_mean = tra_sum / n_frag
                dstep_rot = dstep_rot / n_rot if n_rot else np.full(T, np.nan)
                dstep_tra = dstep_tra / n_frag
                rmsd = np.sqrt(((traj - wahr) ** 2).sum(axis=2).mean(axis=1))
                com = np.linalg.norm(traj.mean(axis=1) - wahr.mean(axis=0), axis=1)

                if seed < a.detail_seeds:
                    for t in range(T):
                        dw.writerow([cid, seed, t, n_frag, n_atoms,
                                     f"{rmsd[t]:.4f}", f"{rot_mean[t]:.3f}",
                                     f"{rot_max[t]:.3f}", f"{tra_mean[t]:.4f}",
                                     f"{com[t]:.4f}", f"{dstep_rot[t]:.3f}",
                                     f"{dstep_tra[t]:.4f}"])

                if T not in summe:
                    summe[T] = np.zeros((T, 6))
                    anzahl[T] = 0
                gueltig = np.nan_to_num(rot_mean, nan=0.0)
                summe[T] += np.stack([rmsd, gueltig, tra_mean, com,
                                      np.nan_to_num(dstep_rot), dstep_tra], axis=1)
                anzahl[T] += 1
        print(f"  seed {seed} fertig", flush=True)
    detail.close()

    if len(T_gesehen) > 1:
        print(f"WARNUNG: unterschiedliche Schrittzahlen im selben Lauf: {sorted(T_gesehen)}")

    with open(f"traj_agg_{a.arm}.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "n_steps", "step", "n_poses",
                    "rmsd_mean", "rot_mean_deg", "trans_mean_A", "com_offset_A",
                    "step_rot_deg", "step_trans_A"])
        for T, S in sorted(summe.items()):
            M = S / anzahl[T]
            for t in range(T):
                w.writerow([a.arm, T, t, anzahl[T]] + [f"{v:.4f}" for v in M[t]])
    if fehlend:
        print(f"WARNUNG: {fehlend} Posen ohne passende Referenz uebersprungen")
    print(f"geschrieben: traj_agg_{a.arm}.csv und traj_pose_{a.arm}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
