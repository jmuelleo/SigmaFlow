"""Oracle@K-Audit — trennt Generatorqualitaet, Diversitaet und Rankingqualitaet.

DIE FRAGE
  Ist Oracle@K deutlich besser als ein Einzelsample?
    ja   -> Ranking ist ein grosser Hebel, ein Confidence-Kopf lohnt sich
    nein -> Ranking loest das Kernproblem nicht, der Generator ist der Engpass

WARNUNG ZUR LESART (steht so auch in SIGMAFLOW_RESEARCH_ROADMAP.md 2.6)
  Ein hoher Oracle@K-Wert ist NICHT automatisch ein Beleg fuer einen guten
  Generator. Wenn der Rotationskanal auf Zufallsniveau arbeitet, ist Best-of-K
  im Kern "K-mal wuerfeln und das Beste behalten". Deshalb gibt dieses Skript
  zusaetzlich den Rotationsfehler aus - erst beides zusammen ist interpretierbar.

KRISTALLKOPIEN
  84 von 209 PoseBusters-Dateien `<id>_ligands.sdf` (PLURAL) enthalten mehrere
  kristallographische Kopien desselben Liganden. Naiv die erste zu nehmen
  erzeugte in einer frueheren Auswertung 42 Phantom-Ausreisser. Hier wird
  immer die naechstgelegene Kopie gewertet.

    python oracle_k_audit.py --true_dir <...> --pred_glob "<...>/seed_{seed}/*.sdf" --n_seeds 10
"""

import argparse
import glob
import os
import re
from collections import defaultdict

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

K_VALUES = (1, 5, 10, 20, 40)


def load_mols(path):
    try:
        return [m for m in Chem.SDMolSupplier(path, sanitize=False, removeHs=True) if m is not None]
    except Exception:
        return []


def raw_rmsd(P, Q):
    return float(np.sqrt(((P - Q) ** 2).sum(1).mean()))


def best_copy(P, copies):
    """Naechstgelegene Kristallkopie. Siehe Kopfkommentar."""
    best, best_r = None, float("inf")
    for m in copies:
        Q = m.GetConformer().GetPositions()
        if Q.shape != P.shape:
            continue
        r = raw_rmsd(P, Q)
        if r < best_r:
            best, best_r = m, r
    return best, best_r


def kabsch_angle(A, B):
    A = A - A.mean(0)
    B = B - B.mean(0)
    U, _, Vt = np.linalg.svd(A.T @ B)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--true_dir", required=True, help="Ordner mit <id>_ligands.sdf")
    ap.add_argument("--pred_glob", required=True,
                    help="Muster mit {seed}, z.B. '/pfad/seed_{seed}/*.sdf'")
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--label", default="model")
    args = ap.parse_args()

    truth = {}
    for tp in sorted(glob.glob(os.path.join(args.true_dir, "*_ligands.sdf"))):
        cid = os.path.basename(tp).replace("_ligands.sdf", "")
        mols = load_mols(tp)
        if mols:
            truth[cid] = mols
    print(f"Referenzliganden: {len(truth)}  "
          f"(mit mehreren Kristallkopien: {sum(1 for v in truth.values() if len(v) > 1)})")

    per_complex = defaultdict(dict)   # cid -> seed -> (rmsd, rot_err)
    for s in range(args.n_seeds):
        files = glob.glob(args.pred_glob.format(seed=s))
        for f in files:
            base = os.path.basename(f)
            cid = None
            for k in truth:
                if base.startswith(k):
                    cid = k
                    break
            if cid is None:
                continue
            pm = load_mols(f)
            if not pm:
                continue
            P = pm[0].GetConformer().GetPositions()
            mt, r = best_copy(P, truth[cid])
            if mt is None:
                continue
            rot = kabsch_angle(mt.GetConformer().GetPositions(), P)
            per_complex[cid][s] = (r, rot)

    n_full = sum(1 for v in per_complex.values() if len(v) >= args.n_seeds)
    print(f"Komplexe mit Vorhersagen: {len(per_complex)}, "
          f"davon vollstaendig ueber {args.n_seeds} Seeds: {n_full}\n")
    if not per_complex:
        print("Keine Vorhersagen gefunden - pred_glob pruefen.")
        return

    print("=" * 88)
    print(f"ORACLE@K  —  {args.label}")
    print("=" * 88)
    print(f"{'K':>4} {'<2A':>8} {'<5A':>8} {'median':>9} {'mean':>8} {'rot med':>9}  Bedeutung")
    print("-" * 88)

    rng = np.random.default_rng(0)
    for K in K_VALUES:
        if K > args.n_seeds:
            print(f"{K:>4} {'--':>8} {'--':>8} {'--':>9} {'--':>8} {'--':>9}  "
                  f"(nur {args.n_seeds} Seeds gesampelt)")
            continue
        # Jede gezogene Teilmenge ist eine eigene Beobachtung. Erst das Minimum
        # der Teilmenge bilden, DANN schwellen - nicht die Minima ueber die
        # Teilmengen mitteln und den Mittelwert schwellen. Der Unterschied ist
        # nicht kosmetisch: bei K=1 macht er aus den korrekten ~4 % unter 2 A
        # scheinbare 0 %, weil ein Mittelwert ueber zehn Ziehungen die seltenen
        # guten Treffer wegglaettet.
        best_r, best_rot = [], []
        for cid, seeds in per_complex.items():
            vals = [seeds[s] for s in sorted(seeds) if s < args.n_seeds]
            if not vals:
                continue
            if K >= len(vals):
                sub = [vals]
            else:
                idx = [rng.choice(len(vals), K, replace=False) for _ in range(20)]
                sub = [[vals[j] for j in ii] for ii in idx]
            best_r.extend(min(v[0] for v in ss) for ss in sub)
            best_rot.extend(min(v[1] for v in ss) for ss in sub)
        a, ar = np.array(best_r), np.array(best_rot)
        meaning = "Einzelsample (Generatorqualitaet)" if K == 1 else \
                  f"Obergrenze bei perfektem Ranking aus {K}"
        print(f"{K:>4} {100*(a<2).mean():7.1f}% {100*(a<5).mean():7.1f}% "
              f"{np.median(a):8.2f}A {a.mean():7.2f}A {np.median(ar):8.1f}d  {meaning}")

    print()
    print("DIVERSITAET (streut derselbe Komplex ueber die Seeds ueberhaupt?)")
    sds = [np.std([v[0] for v in s.values()]) for s in per_complex.values() if len(s) > 2]
    if sds:
        print(f"  RMSD-Streuung je Komplex: median {np.median(sds):.2f} A, mean {np.mean(sds):.2f} A")
        print("  Nahe null hiesse: die Quelle ist wirkungslos, Multi-Sampling bringt nichts.")

    print()
    print("EINORDNUNG")
    print("  Haar-Referenz fuer den Rotationsfehler: median 132.3 Grad.")
    print("  Liegt 'rot med' bei K=1 dort, traegt der Rotationskanal keine Information")
    print("  und ein hoher Oracle@K-Wert ist teilweise nur Wuerfelglueck - siehe")
    print("  SIGMAFLOW_RESEARCH_ROADMAP.md Abschnitt 2.6.")


if __name__ == "__main__":
    main()
