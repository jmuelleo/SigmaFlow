"""EXP-101 — Source Distance Audit.

FRAGE
Bevor irgendeine konditionierte Quelle gebaut wird: wie weit ist die
uninformierte Quelle vom Ziel entfernt, und schafft es eine EINFACHE,
inferenzverfuegbare Heuristik ueberhaupt, diese Distanz zu verkleinern?

Wenn nein, ist der ganze Conditional-Source-Strang schwach motiviert und
EXP-102 wird nicht gebaut. Das ist das Abbruchkriterium aus FM_SOURCE_ROADMAP.md.

VORAUSSETZUNG
Laeuft in der EXP-100-Codebasis. In SigmaFlow-Minimal ist `R_1 = I` per
Konstruktion, dort waere jede Distanzmessung tautologisch.

EIN KONZEPTIONELLER PUNKT, DER BEIM SCHREIBEN AUFFIEL
Eine Quelle wird bei t=0 gezogen. Zu diesem Zeitpunkt haben die Fragmente noch
KEINE Position - die wird ja gerade erst erzeugt. Eine "fragmentweise
konditionierte Quelle" kann sich also nicht auf die Umgebung des Fragments
beziehen, denn die ist noch unbekannt. Konditioniert werden kann nur auf
  (Fragmentidentitaet, Ligandtopologie, Tasche als Ganzes).
Das ist deutlich schwaecher als es auf den ersten Blick klingt und schraenkt
die realistisch erreichbare Verbesserung ein. Deshalb misst H1 unten eine
GLOBALE Ausrichtung (eine Rotation fuer alle Fragmente eines Liganden) - das
ist das Staerkste, was ohne Positionswissen sauber definierbar ist.

    PYTHONPATH=src python exp101_distance_audit.py --data_dir ... --experiment posebusters
"""

import argparse
import sys

import numpy as np
import torch
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


def so3_angle_deg(R: torch.Tensor) -> torch.Tensor:
    """Rotationswinkel in Grad, [..., 3, 3] -> [...]."""
    tr = R.diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.rad2deg(torch.arccos(torch.clamp((tr - 1) / 2, -1.0, 1.0)))


def principal_axes(X: np.ndarray) -> np.ndarray:
    """Rechtshaendiges Hauptachsensystem einer Punktwolke, Spalten = Achsen.

    Die Vorzeichen der Eigenvektoren sind nicht eindeutig; wir fixieren sie
    ueber das dritte Moment. Ohne diese Fixierung waere die "Heuristik" in
    Wahrheit zufaellig zwischen vier gleichwertigen Rahmen - ein Fehler, der
    sich als plausibles Ergebnis tarnen wuerde.
    """
    Xc = X - X.mean(0)
    cov = Xc.T @ Xc / max(len(Xc) - 1, 1)
    w, V = np.linalg.eigh(cov)
    V = V[:, ::-1]                                   # groesste Varianz zuerst
    proj = Xc @ V
    for k in range(3):
        if (proj[:, k] ** 3).sum() < 0:              # Schiefe positiv machen
            V[:, k] = -V[:, k]
    if np.linalg.det(V) < 0:                         # Rechtshaendigkeit erzwingen
        V[:, 2] = -V[:, 2]
    return V


def report(name: str, ang: np.ndarray) -> dict:
    q = np.percentile(ang, [25, 50, 75, 90])
    print(f"  {name:<34} n={len(ang):>5}  mean={ang.mean():6.1f}  median={q[1]:6.1f}  "
          f"p25={q[0]:6.1f}  p75={q[2]:6.1f}  p90={q[3]:6.1f}")
    return {"name": name, "n": int(len(ang)), "mean": float(ang.mean()),
            "median": float(q[1]), "p25": float(q[0]), "p75": float(q[2]), "p90": float(q[3])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--experiment", default="posebusters")
    ap.add_argument("--max_complexes", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from sigmadock.data import SigmaDataset
    from sigmadock.datafronts import DataFront
    from sigmadock.diff.sigma_flow_generator import SigmaFlowGenerator
    from sigmadock.oracle import HPARAMS
    from torch_geometric.data import Batch

    assert hasattr(SigmaFlowGenerator, "get_fragment_com_and_rot_reparam"), (
        "EXP-101 braucht die EXP-100-Codebasis. In SigmaFlow-Minimal ist R_1 = I "
        "und jede Distanzmessung waere tautologisch."
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = DataFront(f"{args.data_dir}/{args.experiment}",
                   pdb_regex=r".*_protein\.pdb$", sdf_regex=r".*_ligand\.sdf$")
    ds = SigmaDataset(
        datafront=df, pocket_com_noise=0.0, pocket_distance_cutoff=8.0,
        pocket_distance_noise=0.0, prot_coordinate_distance_noise=0.0,
        use_esm_embeddings=False, ignore_triangulation=False,
        lig_coordinate_distance_noise=0.0, alignment_tries=0,
        fragmentation_strategy="canonical", pb_check=False, get_mol_info=True,
        seed=args.seed, random_rotation=False, sample_conformer=False,
        skip_bounds_check=True, force_retry=False,
    )
    print(f"Datafront: {len(df)} Komplexe, ausgewertet werden bis zu {args.max_complexes}")

    gen = SigmaFlowGenerator(model=torch.nn.Identity(), sigma_min=0.0)
    LIG_VIRT = HPARAMS.get_node_idx("ligand_virtual")

    d_haar, d_ident, d_pax, dt_zero, dt_haar = [], [], [], [], []
    n_ok = n_skip = 0

    for i in range(min(args.max_complexes, len(ds))):
        try:
            d = ds[i]
        except Exception:
            n_skip += 1
            continue
        if d is None:
            n_skip += 1
            continue
        b = Batch.from_data_list([d])
        try:
            pos_0, trans_1, R_1, _ = gen._get_initial_states(b)
        except Exception:
            n_skip += 1
            continue
        n_ok += 1
        F = R_1.shape[0]

        # --- H0: Haar. Die Referenz, gegen die alles antreten muss. ---------
        R_haar = gen.flow_matcher.sample_init(F, R_1.device)["R_0"].to(R_1)
        d_haar.append(so3_angle_deg(R_haar.transpose(-1, -2) @ R_1).numpy())

        # --- H_id: R_0 = I. Kontrolle, KEINE zulaessige Quelle. -------------
        # Zeigt, wie weit das Konformer-Eigenframe schon vom Ziel entfernt ist.
        d_ident.append(so3_angle_deg(R_1).numpy())

        # --- H1: globale Hauptachsenausrichtung Ligand -> Tasche ------------
        # Eine Rotation fuer ALLE Fragmente. Alles Feinere braucht Wissen
        # ueber die Fragmentposition, das bei t=0 nicht existiert (siehe Kopf).
        try:
            lig = (b.frag_idx_map >= 0) & (b.node_entity != LIG_VIRT) & b.mask
            prot = b.frag_idx_map < 0
            V_lig = principal_axes(pos_0[lig].numpy())
            V_pkt = principal_axes(b.ref_pos[prot].numpy())
            Q = torch.tensor(V_pkt @ V_lig.T, dtype=R_1.dtype)
            if torch.linalg.det(Q) < 0:
                Q[:, 2] = -Q[:, 2]
            R_pax = Q.unsqueeze(0).expand_as(R_1)
            d_pax.append(so3_angle_deg(R_pax.transpose(-1, -2) @ R_1).numpy())
        except Exception:
            pass

        # --- Translation zum Vergleich -------------------------------------
        dt_zero.append(np.linalg.norm(trans_1.numpy(), axis=-1))
        t0 = gen.flow_matcher.sample_init(F, R_1.device)["trans_0"].to(trans_1)
        dt_haar.append(np.linalg.norm((trans_1 - t0).numpy(), axis=-1))

    print(f"\nausgewertet: {n_ok} Komplexe, uebersprungen: {n_skip}\n")
    print("=" * 96)
    print("ROTATIONSDISTANZ  d_SO(3)(R_0, R_1)  [Grad]")
    print("=" * 96)
    rows = [
        report("H0  Haar (uninformiert)", np.concatenate(d_haar)),
        report("H1  Hauptachsen Ligand->Tasche", np.concatenate(d_pax)) if d_pax else None,
        report("Hid R_0 = I  (KONTROLLE)", np.concatenate(d_ident)),
    ]
    print()
    print("  Erwartung fuer H0: median ~132.3, mean ~126.5 (Haar auf SO(3)).")
    print("  Weicht H0 davon ab, stimmt etwas an der Messung nicht - nicht an der Heuristik.")
    print()
    h0 = rows[0]["median"]
    for r in rows[1:]:
        if r is None:
            continue
        gain = h0 - r["median"]
        verdict = "TRAEGT" if gain > 10 else ("marginal" if gain > 3 else "TRAEGT NICHT")
        print(f"  {r['name']:<34} Gewinn gegen Haar: {gain:+6.1f} Grad  -> {verdict}")

    print()
    print("=" * 96)
    print("TRANSLATIONSDISTANZ  |p_0 - p_1|  [normalisierte Einheiten]")
    print("=" * 96)
    report("Quelle N(0,I)", np.concatenate(dt_haar))
    report("Quelle = Taschenmitte (p_0=0)", np.concatenate(dt_zero))
    print()
    print("  Die zweite Zeile ist der informative Prior, den die Translation")
    print("  BEREITS hat: pocket_com liegt im Ursprung. Fuer die Rotation gibt")
    print("  es kein Gegenstueck - Haar hat keinen Mittelwert. Genau diese")
    print("  Asymmetrie ist die Motivation des ganzen Strangs.")

    print()
    print("=" * 96)
    print("ABBRUCHKRITERIUM")
    print("=" * 96)
    # Nur die zulaessigen Heuristiken zaehlen - die Kontrolle Hid nicht.
    candidates = [r["median"] for r in rows[1:-1] if r is not None]
    best = min(candidates) if candidates else h0
    if h0 - best > 10:
        print(f"  Beste Heuristik senkt den Median um {h0 - best:.1f} Grad (>10).")
        print("  -> EXP-102 (heuristische konditionierte Quelle) ist motiviert.")
    else:
        print(f"  Beste Heuristik senkt den Median nur um {h0 - best:.1f} Grad (<=10).")
        print("  -> EXP-102 NICHT bauen. Der Engpass ist dann nicht die Quelle,")
        print("     sondern der Rotationskanal selbst (siehe SIGMAFLOW_RESEARCH_ROADMAP.md,")
        print("     Abschnitt 2.3 und die Diagnose G8).")


if __name__ == "__main__":
    sys.exit(main())
