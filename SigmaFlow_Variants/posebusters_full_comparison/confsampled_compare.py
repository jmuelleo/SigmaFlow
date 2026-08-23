"""Der Prior-Test: alle drei Arme bei gleicher Startbedingung.

FRAGE
    SigmaDock trifft in der bisherigen Auswertung rund doppelt so oft unter
    2 A wie die SigmaFlow-Arme. Kommt das vom Modell oder vom Startpunkt?

WARUM DIE FRAGE UEBERHAUPT ENTSTEHT
    SigmaDock zieht die Startrotation aus IGSO(3) beim maximalen Rauschpegel
    sigma_max = 1.5 (so3_diffuser.sample_ref, mit dem Kommentar der
    Originalautoren "NOTE: replace with sample uniform?"). Diese Verteilung
    ist NICHT das Haar-Mass: mittlerer Winkel 114.97 statt 126.48 Grad.
    SigmaFlow zieht exakt gleichverteilt (so3_utils.sample_uniform).

    Mit graph.sample_conformer=false stammt die Fragmentgeometrie aus der
    GEBUNDENEN Pose, die Identitaetsrotation ist dann die WAHRE Orientierung
    -- und IGSO(3) hat um sie herum mehr Masse. SigmaDock startet damit
    systematisch naeher an der Antwort.

    Mit sample_conformer=true stammt sie aus einem generierten Konformer,
    dessen Orientierung willkuerlich ist. Der Vorteil sollte verschwinden.

REFERENZ
    Wahre Pose aus true_ligands/<cid>_ligands.sdf, naechstgelegene
    kristallographische Kopie ueber best_copy. NICHT x0 aus predictions.pt --
    das traegt im sampled-Modus den generierten Konformer in seinem eigenen
    Bezugssystem, im Mittel 50.4 A daneben.

AUFRUF   python confsampled_compare.py
"""
from __future__ import annotations

import glob
import os
import pathlib
import re

import numpy as np
import torch
from rdkit import RDLogger

from ligand_reference import best_copy, load_copies

RDLogger.DisableLog("rdApp.*")

SCALE = 2.7
MIN_FRAG_ATOMS = 3
BOND_TOL = 0.01
LAEUFE = [("Minimal", "min80", "mincs"),
          ("Separate", "exp80", "expcs"),
          ("SigmaDock", "sd80", "sdcs")]


def kabsch_winkel(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    Pc = P - P.mean(axis=1, keepdims=True)
    Qc = Q - Q.mean(axis=0, keepdims=True)
    H = np.einsum("tni,nj->tij", Pc, Qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(np.einsum("tij,tjk->tik", U, Vt)))
    D = np.zeros((len(P), 3, 3))
    D[:, 0, 0] = D[:, 1, 1] = 1.0
    D[:, 2, 2] = d
    R = np.einsum("tij,tjk,tkl->til", U, D, Vt)
    return np.degrees(np.arccos(np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1, 1)))


def fragmente(traj: np.ndarray, bonds, n: int) -> np.ndarray:
    eltern = list(range(n))

    def wurzel(a):
        while eltern[a] != a:
            eltern[a] = eltern[eltern[a]]
            a = eltern[a]
        return a

    ii = np.array([i for i, _ in bonds])
    jj = np.array([j for _, j in bonds])
    L = np.linalg.norm(traj[:, ii] - traj[:, jj], axis=2)
    for (i, j), sp in zip(bonds, L.max(axis=0) - L.min(axis=0)):
        if sp < BOND_TOL:
            ra, rb = wurzel(i), wurzel(j)
            if ra != rb:
                eltern[rb] = ra
    return np.array([wurzel(a) for a in range(n)])


def messe(root: str, n_seeds: int):
    """-> dict complex -> Liste von (rot_start, rot_ende, rmsd_ende) je Seed."""
    orig = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath if os.name == "nt" else orig
    aus = {}
    try:
        pts = sorted(((int(re.search(r"seed_(\d+)", p).group(1)), p)
                      for p in glob.glob(f"{root}/**/seed_*/predictions.pt", recursive=True)),
                     key=lambda x: x[0])[:n_seeds]
        for seed, pfad in pts:
            d = torch.load(pfad, map_location="cpu", weights_only=False)
            for key, lst in d["results"].items():
                e = lst[0]
                cid = key.split("::")[0]
                tp = f"true_ligands/{cid}_ligands.sdf"
                if not os.path.isfile(tp):
                    continue
                traj = e["trajectory"].numpy() * SCALE + e["com"].numpy()
                na = traj.shape[1]
                mol = e["lig_ref"]
                bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
                if mol.GetNumAtoms() != na or not bonds:
                    continue
                mt, _, _ = best_copy(traj[-1], load_copies(tp))
                if mt is None or mt.GetNumAtoms() != na:
                    continue
                W = mt.GetConformer().GetPositions()
                fid = fragmente(traj, bonds, na)
                r0 = rE = 0.0
                k = 0
                for f in np.unique(fid):
                    s = fid == f
                    if s.sum() >= MIN_FRAG_ATOMS:
                        r0 += kabsch_winkel(traj[:1, s], W[s])[0]
                        rE += kabsch_winkel(traj[-1:, s], W[s])[0]
                        k += 1
                if not k:
                    continue
                rmsd = float(np.sqrt(((traj[-1] - W) ** 2).sum(1).mean()))
                aus.setdefault(cid, []).append((r0 / k, rE / k, rmsd))
            print(f"    seed {seed}", flush=True)
    finally:
        pathlib.PosixPath = orig
    return aus


def main() -> int:
    N = 40
    daten = {}
    for arm, wb, wc in LAEUFE:
        print(f"{arm} bound:")
        daten[(arm, "bound")] = messe(wb, N)
        print(f"{arm} sampled:")
        daten[(arm, "sampled")] = messe(wc, N)

    print()
    print("  " + "=" * 78)
    print(f"  {'Arm':<11}{'Modus':<10}{'n Posen':>9}{'Rot Start':>12}"
          f"{'Rot Ende':>11}{'RMSD Ende':>12}{'<2A':>9}")
    print("  " + "-" * 78)
    for arm, _, _ in LAEUFE:
        for modus in ("bound", "sampled"):
            v = [x for lst in daten[(arm, modus)].values() for x in lst]
            r0 = np.mean([a for a, _, _ in v])
            rE = np.mean([b for _, b, _ in v])
            dE = np.array([c for _, _, c in v])
            print(f"  {arm:<11}{modus:<10}{len(v):>9}{r0:>11.2f}°{rE:>10.2f}°"
                  f"{dE.mean():>12.3f}{100*np.mean(dE < 2):>8.2f}%")
    print("  " + "=" * 78)
    print("  Referenz: Haar 126.48°   IGSO(3) bei sigma=1.5: 114.97°")

    # Gepaart ueber Komplexe: Trefferquote je Komplex, dann Bootstrap.
    rng = np.random.default_rng(0)
    print()
    print("  GEPAARTER VERGLEICH im Modus sampled (Bootstrap ueber Komplexe)")
    cids = sorted(set.intersection(*[set(daten[(a, "sampled")]) for a, _, _ in LAEUFE]))
    quote = {a: np.array([np.mean([c < 2 for _, _, c in daten[(a, "sampled")][x]])
                          for x in cids]) for a, _, _ in LAEUFE}
    for a, b in (("Minimal", "Separate"), ("Minimal", "SigmaDock"), ("Separate", "SigmaDock")):
        d = quote[b] - quote[a]
        idx = rng.integers(0, len(d), size=(20000, len(d)))
        reps = d[idx].mean(axis=1)
        p = min(1.0, 2 * min((reps <= 0).mean(), (reps >= 0).mean()))
        print(f"    {b} minus {a:<10}{100*d.mean():+7.2f} pp  "
              f"[{100*np.percentile(reps,2.5):+.2f},{100*np.percentile(reps,97.5):+.2f}]  p={p:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
