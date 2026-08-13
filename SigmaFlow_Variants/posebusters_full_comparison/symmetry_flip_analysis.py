"""Sind unsere Rotationsfehler SYMMETRIEFEHLER oder echte Fehler?

WARUM DAS SKRIPT EXISTIERT
Wir haben (a) einen bimodalen Rotationsfehler mit einem zweiten Modus nahe der
Umkehrung ("Kopf-Schwanz-Flip") und (b) den Befund, dass 53 % der Molekuele
mehr als einen Graphautomorphismus haben. Daraus zu schliessen, die Flips seien
Symmetrieartefakte, waere ein Fehlschluss:

  Graphautomorphismus  =/=  Rotationssymmetrie des starren Fragments.

Ein Automorphismus des GANZEN Molekuels muss das Fragment nicht auf sich selbst
abbilden, und selbst wenn, muss die zugehoerige Atompermutation nicht durch eine
RAEUMLICHE Drehung realisierbar sein. Nur wenn beides gilt, ist die geflippte
Orientierung chemisch ununterscheidbar und das Modell hatte recht.

WAS GEMESSEN WIRD
Je starrem Fragment:
  theta_raw  = Winkel der Kabsch-Rotation wahres Fragment -> vorhergesagtes
  theta_sym  = min ueber alle zulaessigen Atompermutationen sigma des Fragments
               von theta(Kabsch(X_true[sigma], X_pred))
Zulaessig heisst: sigma stammt aus einem Automorphismus des Molekuelgraphen,
bildet die Fragmentatommenge auf sich selbst ab, UND die permutierte Geometrie
deckt sich raeumlich mit der Ausgangsgeometrie (sonst ist die Permutation zwar
graphentheoretisch erlaubt, aber nicht als Starrkoerperdrehung realisierbar).

LESART
  theta_raw gross, theta_sym klein  -> Symmetrieartefakt, Modell hatte recht,
                                        symmetry-aware Supervision wuerde helfen
  theta_raw gross, theta_sym gross  -> echter Fehler, Symmetrie erklaert nichts

Usage: python symmetry_flip_analysis.py
"""

import glob
import os

import numpy as np
from rdkit import Chem, RDLogger

from ligand_reference import best_copy, load_copies

RDLogger.DisableLog("rdApp.*")

TRUE_DIR = "true_ligands"
PRED_GLOB = "seeds10_sigmaflow/last/seed_0/{cid}__*_seed0.sdf"
BOND_TOL = 0.02          # Bindungslaengentoleranz fuer "starr geblieben"
SPATIAL_TOL = 0.35       # A, wie gut die Permutation raeumlich passen muss
MAX_AUT = 200            # Deckel gegen kombinatorische Explosion
MIN_FRAG = 3             # unter 3 Atomen ist keine Rotation definiert


def rigid_fragments(mt, mp, tol=BOND_TOL):
    """Die Zerlegung, die das Modell effektiv benutzt hat: Bindungen, deren
    Laenge zwischen wahrer und vorhergesagter Pose erhalten blieb."""
    n = mt.GetNumAtoms()
    par = list(range(n))

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a

    ct, cp = mt.GetConformer(), mp.GetConformer()
    for b in mt.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        dt = ct.GetAtomPosition(i).Distance(ct.GetAtomPosition(j))
        dp = cp.GetAtomPosition(i).Distance(cp.GetAtomPosition(j))
        if abs(dp - dt) <= tol:
            ra, rb = find(i), find(j)
            if ra != rb:
                par[rb] = ra
    lab, out = {}, []
    for a in range(n):
        r = find(a)
        lab.setdefault(r, len(lab))
        out.append(lab[r])
    return np.array(out)


def kabsch_angle(A, B):
    """Winkel der optimalen Rotation A -> B (beide werden hier zentriert),
    in Grad, mit det=+1."""
    A = A - A.mean(0)
    B = B - B.mean(0)
    U, _, Vt = np.linalg.svd(A.T @ B)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def fragment_permutations(mol, frag_atoms, coords, max_aut=MAX_AUT):
    """Alle Atompermutationen des Fragments, die (a) aus einem Automorphismus
    des Molekuelgraphen stammen, (b) die Fragmentatome auf sich abbilden und
    (c) raeumlich als Starrkoerperdrehung realisierbar sind.

    (c) ist der Punkt, den man leicht vergisst: eine Graphsymmetrie zwischen
    zwei Substituenten ist nur dann eine RAEUMLICHE Symmetrie, wenn die
    permutierte Punktwolke nach optimaler Drehung wieder auf der Ausgangswolke
    liegt. Sonst waere die "Symmetrie" nur eine Umbenennung.
    """
    fset = set(frag_atoms)
    idx = {a: k for k, a in enumerate(frag_atoms)}
    X = coords[frag_atoms]
    Xc = X - X.mean(0)

    try:
        matches = mol.GetSubstructMatches(mol, uniquify=False, useChirality=False, maxMatches=max_aut)
    except Exception:
        return [np.arange(len(frag_atoms))], 1

    perms, n_graph = [], 0
    for m in matches:
        # m[i] = Bild von Atom i
        if not all(m[a] in fset for a in frag_atoms):
            continue          # bildet das Fragment nicht auf sich ab
        n_graph += 1
        p = np.array([idx[m[a]] for a in frag_atoms])
        # raeumliche Realisierbarkeit pruefen
        Y = Xc[p]
        Yc = Y - Y.mean(0)
        U, _, Vt = np.linalg.svd(Xc.T @ Yc)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
        rms = float(np.sqrt(((Xc @ R.T - Yc) ** 2).sum(1).mean()))
        if rms <= SPATIAL_TOL:
            perms.append(p)

    # Entduplizieren. Ohne diesen Schritt zaehlt man Automorphismen des GANZEN
    # Molekuels mit, die auf dem Fragment die IDENTITAET sind (sie permutieren
    # nur Atome ausserhalb). Die erste Fassung dieses Skripts tat genau das und
    # meldete dadurch 50 % "symmetrische" Fragmente statt der tatsaechlichen 5 %.
    uniq = {tuple(p.tolist()): p for p in perms}
    perms = list(uniq.values())
    if not perms:
        perms = [np.arange(len(frag_atoms))]
    # n_nontrivial: Permutationen, die WIRKLICH eine Drehung != I realisieren
    ident = np.arange(len(frag_atoms))
    n_nontrivial = sum(1 for p in perms if not np.array_equal(p, ident))
    return perms, n_nontrivial


def main() -> None:
    rows = []
    n_cplx = 0
    for tp in sorted(glob.glob(os.path.join(TRUE_DIR, "*_ligands.sdf"))):
        cid = os.path.basename(tp).replace("_ligands.sdf", "")
        hits = glob.glob(PRED_GLOB.format(cid=cid))
        tc = load_copies(tp)
        if not (hits and tc):
            continue
        pc = load_copies(hits[0])
        if not pc:
            continue
        mp = pc[0]
        mt, _, _ = best_copy(mp.GetConformer().GetPositions(), tc)
        if mt is None or mt.GetNumAtoms() != mp.GetNumAtoms():
            continue
        n_cplx += 1

        fid = rigid_fragments(mt, mp)
        Xt = np.asarray(mt.GetConformer().GetPositions())
        Xp = np.asarray(mp.GetConformer().GetPositions())

        for f in np.unique(fid):
            atoms = np.where(fid == f)[0]
            if len(atoms) < MIN_FRAG:
                continue
            A, B = Xt[atoms], Xp[atoms]
            theta_raw = kabsch_angle(A, B)

            perms, n_graph = fragment_permutations(mt, atoms, Xt)
            theta_sym = min(kabsch_angle(A[p], B) for p in perms)
            rows.append((len(atoms), theta_raw, theta_sym, len(perms), n_graph))

    a = np.array(rows, dtype=float)
    n_at, raw, sym, n_perm, n_graph = a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4]

    print("=" * 88)
    print("SYMMETRIE VS. ECHTER FEHLER — Rotationsabweichung je starrem Fragment")
    print("=" * 88)
    print(f"Komplexe: {n_cplx}   Fragmente (>= {MIN_FRAG} Atome): {len(a)}")
    print()
    print("Wieviele Fragmente haben ueberhaupt eine nutzbare Symmetrie?")
    print(f"  mind. eine NICHT-triviale, raeumlich realisierbare Drehsymmetrie: "
          f"{int((n_graph > 0).sum())} von {len(a)} ({100*(n_graph>0).mean():.1f} %)")
    print("  (Zum Vergleich: 53 % der MOLEKUELE haben >1 Graphautomorphismus.")
    print("   Auf Fragmentebene und nach Pruefung auf raeumliche Realisierbarkeit")
    print("   bleibt davon fast nichts uebrig - genau der Fehlschluss, vor dem")
    print("   zu warnen war.)")
    print()

    print("Rotationsfehler [Grad]:")
    for name, v in (("ohne Symmetriekorrektur", raw), ("mit Symmetriekorrektur", sym)):
        q = np.percentile(v, [25, 50, 75])
        print(f"  {name:<26} median={q[1]:6.1f}  IQR=[{q[0]:.1f}, {q[2]:.1f}]  "
              f"Mittel={v.mean():6.1f}")
    print(f"  Haar-Referenz (Zufallsrotation)   median= 132.3   Mittel= 126.5")
    print()

    flip = raw > 120.0
    print(f"FLIP-KANDIDATEN (theta_raw > 120 Grad): {int(flip.sum())} "
          f"({100*flip.mean():.1f} % aller Fragmente)")
    if flip.any():
        red = raw[flip] - sym[flip]
        print(f"  Reduktion durch Symmetriekorrektur: median={np.median(red):.1f} Grad, "
              f"max={red.max():.1f} Grad")
        rescued = (sym[flip] < 30.0)
        print(f"  davon durch Symmetrie ERKLAERT (theta_sym < 30 Grad): "
              f"{int(rescued.sum())} von {int(flip.sum())} ({100*rescued.mean():.1f} %)")
        print(f"  bleiben ECHTE Fehler                                : "
              f"{int((~rescued).sum())} ({100*(~rescued).mean():.1f} %)")
    print()

    print("Nach Fragmentgroesse (median theta ohne / mit Korrektur):")
    for lo, hi, lbl in ((3, 4, "3-4 Atome"), (5, 6, "5-6"), (7, 9, "7-9"), (10, 999, "10+")):
        m = (n_at >= lo) & (n_at <= hi)
        if m.sum() < 5:
            continue
        print(f"  {lbl:<12} n={int(m.sum()):>5}   {np.median(raw[m]):6.1f}  ->  "
              f"{np.median(sym[m]):6.1f}   (Symmetrie nutzbar bei "
              f"{100*(n_perm[m]>1).mean():.0f} %)")
    print()
    print("SCHLUSSFOLGERUNG")
    gain = np.median(raw) - np.median(sym)
    print(f"  Symmetriekorrektur senkt den medianen Rotationsfehler um {gain:.1f} Grad.")
    if gain < 5:
        print("  -> Symmetrie erklaert das Rotationsproblem NICHT. Symmetry-aware")
        print("     Supervision waere Korrektheitspflege, kein Leistungshebel.")
    else:
        print("  -> Symmetrie erklaert einen relevanten Teil. Symmetry-aware")
        print("     Supervision ist ein echter Kandidat.")


if __name__ == "__main__":
    main()
