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
import pathlib
import sys

import numpy as np
import torch
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


def so3_angle_deg(R: torch.Tensor) -> torch.Tensor:
    """Rotationswinkel in Grad, [..., 3, 3] -> [...].

    KONDITIONIERUNG - warum nicht einfach arccos((tr-1)/2)
      Naheliegend ist w = arccos((tr - 1)/2), denn tr(R) = 1 + 2 cos w. arccos
      hat bei Argument +-1 aber unendliche Ableitung: ein Rundungsfehler eps im
      Kosinus wird zu einem Winkelfehler sqrt(2 eps). In float32 sind das rund
      0.03 Grad bei w = 0.

      Fuer kleine Winkel ist die Sehnenform stabil. Aus
      ||A - B||_F^2 = 6 - 2 tr(A^T B) folgt ||R - I||_F = 2 sqrt(2) sin(w/2),
      also w = 2 arcsin(||R - I||_F / (2 sqrt(2))); arcsin ist bei 0 flach.

      Die Sehne wird DIREKT aus R - I genommen, nicht ueber sqrt(6 - 2 tr).
      Die beiden Formen sind algebraisch gleich, numerisch nicht: fuer kleine w
      ist tr ~ 3, und 6 - 2*3 subtrahiert zwei fast gleiche grosse Zahlen. Der
      absolute Rundungsfehler bleibt stehen, waehrend das Ergebnis wie w^2
      gegen null geht. Gemessen kostete das bei w = 1e-6 rad vier Stellen; die
      direkte Form ist um Faktor ~4700 genauer und LINEAR statt sqrt in eps.

      Fuer dieses Gate (Effekte ab 10 Grad) ist das irrelevant. Es steht hier,
      weil der Unterschied in einem Test zunaechst als "Grenze der Darstellung"
      fehlgedeutet wurde - er passte fast exakt zu sqrt(2*eps32).
    """
    R64 = R.to(torch.float64)
    tr = R64.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos_w = ((tr - 1.0) / 2.0).clamp(-1.0, 1.0)
    eye = torch.eye(3, dtype=R64.dtype, device=R64.device).expand_as(R64)
    chord = torch.linalg.matrix_norm(R64 - eye, ord="fro")
    w_small = 2.0 * torch.arcsin((chord / (2.0 * np.sqrt(2.0))).clamp(-1.0, 1.0))
    w = torch.where(cos_w >= 0.0, w_small, torch.arccos(cos_w))
    return torch.rad2deg(w).to(torch.float32)


def karcher_mean(R: torch.Tensor, iters: int = 50, tol: float = 1e-7) -> torch.Tensor:
    """Riemannsches Mittel: argmin_M sum_i d(M, R_i)^2.

    Liefert die BESTE KONSTANTE Quelle. Sie ist selbst keine zulaessige
    Heuristik - eine feste Rotation traegt keine Information ueber den
    konkreten Komplex -, aber sie ist die scharfe obere Schranke fuer alles,
    was eine konstante Quelle je erreichen koennte. Liegt H1 nicht klar besser
    als diese Schranke, nutzt die Heuristik die Tasche nicht wirklich aus.
    """
    from sigmadock.diff import so3_utils
    M = R[0].clone()
    for _ in range(iters):
        xi = so3_utils.Log(M.transpose(-1, -2) @ R)      # [n,3] im Tangentialraum
        step = xi.mean(dim=0)
        M = M @ so3_utils.Exp(step[None])[0]
        if float(step.norm()) < tol:
            break
    return M


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
    ap.add_argument("--out_json", default=None,
                    help="Ergebnis zusaetzlich maschinenlesbar ablegen.")
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
    all_trans_1, all_R_1 = [], []
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
        all_trans_1.append(trans_1.numpy())
        all_R_1.append(R_1)
        dt_zero.append(np.linalg.norm(trans_1.numpy(), axis=-1))
        t0 = gen.flow_matcher.sample_init(F, R_1.device)["trans_0"].to(trans_1)
        dt_haar.append(np.linalg.norm((trans_1 - t0).numpy(), axis=-1))

    print(f"\nausgewertet: {n_ok} Komplexe, uebersprungen: {n_skip}\n")
    if n_ok == 0:
        raise SystemExit("Keine Komplexe auswertbar - Datenpfad pruefen.")
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

    # --- Beste konstante Quelle: die scharfe Schranke fuer H1 ---------------
    # Eine feste Rotation traegt KEINE Information ueber den konkreten Komplex
    # und ist deshalb selbst keine zulaessige Heuristik. Sie ist aber die obere
    # Schranke fuer alles, was ohne Konditionierung erreichbar waere. Liegt H1
    # nicht klar darunter, nutzt die Heuristik die Tasche nicht wirklich aus,
    # sondern findet nur eine global haeufige Orientierung wieder.
    R_all = torch.cat(all_R_1, dim=0)
    R_star = karcher_mean(R_all)
    d_const = so3_angle_deg(R_star.transpose(-1, -2) @ R_all).numpy()
    print()
    print("  Zusaetzliche Referenz (KEINE zulaessige Heuristik):")
    r_const = report("Hc  beste KONSTANTE Rotation", d_const)

    # --- Ist DIMENSIONAL_SCALE empirisch richtig gewaehlt? -------------------
    # chem/__init__.py nennt 2.7 A die "STD scale ... on ligand fragment
    # centroids". trans_1 ist bereits dadurch geteilt; hat es Einheitsvarianz,
    # war der Wert korrekt und N(0,I) ist per Konstruktion massstabsangepasst.
    # Diese Zahl ist im Projekt bisher nirgends nachgerechnet worden.
    from sigmadock.chem import DIMENSIONAL_SCALE
    all_t1 = np.concatenate(all_trans_1, axis=0)
    std_iso = float(all_t1.std())
    implied = DIMENSIONAL_SCALE * std_iso
    print()
    print("=" * 96)
    print("MASSSTAB DER TRANSLATIONSQUELLE")
    print("=" * 96)
    print(f"  DIMENSIONAL_SCALE im Code          : {DIMENSIONAL_SCALE:.3f} A")
    print(f"  Std von trans_1 (normalisiert)     : {std_iso:.3f}   (1.0 waere exakt passend)")
    print(f"  empirisch implizierter Massstab    : {implied:.3f} A")
    print(f"  Mittelwert je Achse (normalisiert) : {np.round(all_t1.mean(0), 3).tolist()}")
    if abs(std_iso - 1.0) > 0.25:
        print("  -> ABWEICHUNG > 25 %: N(0,I) ist NICHT massstabsangepasst. Das ist ein")
        print("     billiger, FM-spezifischer Ansatzpunkt, unabhaengig vom Rotations-")
        print("     ergebnis unten.")
    else:
        print("  -> passend. Ein Skalen-Fix an der Translationsquelle braechte wenig.")

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

    if args.out_json:
        import json
        payload = {
            "n_complexes_evaluated": int(n_ok),
            "n_skipped": int(n_skip),
            "rotation_deg": {r["name"]: r for r in rows if r is not None},
            "best_constant_rotation_deg": r_const,
            "gate": {
                "median_haar_deg": float(h0),
                "median_best_admissible_deg": float(best),
                "gain_deg": float(h0 - best),
                "threshold_deg": 10.0,
                "build_exp102": bool(h0 - best > 10),
            },
            "translation": {
                "dimensional_scale_in_code": float(DIMENSIONAL_SCALE),
                "std_trans_1_normalised": std_iso,
                "implied_optimal_scale_A": float(implied),
                "mean_per_axis_normalised": [float(v) for v in all_t1.mean(0)],
            },
        }
        pathlib.Path(args.out_json).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[exp101] JSON geschrieben: {args.out_json}")


if __name__ == "__main__":
    sys.exit(main())
