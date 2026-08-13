"""EXP-100: ein ECHTER Trainingsschritt durch den vollstaendigen Generator.

Die anderen Tests pruefen Mathematik und Transformationskette. Dieser hier
schickt einen echten Batch durch `SigmaFlowGenerator.forward` mitsamt
EquiformerV2 und `compute_losses` - also durch genau den Pfad, den das Training
auf ARC benutzt. Das Netz ist bewusst winzig; getestet wird die Verdrahtung,
nicht die Qualitaet.

Was hier schiefgehen koennte und sonst nirgends auffiele:
  * `R_ref`-Umbenennung an einer Aufrufstelle vergessen  -> TypeError
  * `ref_conf_pos` batcht nicht mit                      -> Shape-Fehler
  * Ziele tragen einen Autograd-Graphen                  -> Gradient leckt
  * Verlust wird NaN, weil R_1 nicht mehr die Identitaet ist

    PYTHONPATH=src python tests/test_exp100_forward.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
from rdkit import RDLogger
from torch_geometric.data import Batch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
RDLogger.DisableLog("rdApp.*")

from sigmadock.config import StructuralConfig                      # noqa: E402
from sigmadock.diff.sigma_flow_generator import SigmaFlowGenerator  # noqa: E402
from sigmadock.net.model import EquiformerV2                        # noqa: E402
from sigmadock.oracle import HPARAMS                                # noqa: E402
from validate_mapping import build_dataset                          # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(f"{name}  {detail}")


def tiny_denoiser() -> SigmaFlowGenerator:
    model = EquiformerV2(
        atom_feature_dims=list(StructuralConfig.atom_feature_dims),
        edge_feature_dims=list(StructuralConfig.edge_feature_dims),
        average_degrees=HPARAMS.all_degrees,
        use_esm_embeddings=False,
        num_layers=1, num_heads=1,
        sphere_channels=16, edge_channels=8,
        attn_hidden_channels=8, attn_alpha_channels=8, attn_value_channels=8,
        ffn_hidden_channels=16,
        lmax_list=[1], mmax_list=[1],
        distance_expansion_dim=8, t_emb_dim=8,
        alpha_drop=0.0, drop_path_rate=0.0,
        protein_ligand_interactions=True, ligand_ligand_interactions=True,
        # Default ist True und nullt die letzte Schicht -> das Netz gaebe exakt 0
        # aus und der Sampler bewegte nichts. Fuer einen Verdrahtungstest waere
        # das ein blinder Fleck.
        zero_init_last=False,
    )
    return SigmaFlowGenerator(
        model, sigma_min=0.0,
        cutoff_complex_interactions=HPARAMS.get_edge_spec("inter_complex").r_max,
        cutoff_fragment_interactions=HPARAMS.get_edge_spec("inter_fragments").r_max,
        cutoff_complex_virtual=HPARAMS.get_edge_spec("complex_lv2pv").r_max,
        verbose=False,
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    ds = build_dataset()
    items = [d for d in (ds[i] for i in range(len(ds))) if d is not None][:4]
    print(f"Batch aus {len(items)} Komplexen, "
          f"Graphgroessen {[int(d.x.shape[0]) for d in items]}")
    batch = Batch.from_data_list(items)

    den = tiny_denoiser()
    n_par = sum(p.numel() for p in den.parameters())
    print(f"Netz: {n_par/1e3:.0f}k Parameter (bewusst winzig)\n")

    out = den(batch)
    check("forward() laeuft durch", True)

    for k in ("pred_u_t_trans", "pred_u_t_R", "u_t_trans", "u_t_R", "R_1", "trans_1", "pos_0", "pos_t"):
        v = out[k]
        check(f"{k}: endlich", bool(torch.isfinite(v).all()), f"shape {tuple(v.shape)}")

    ang = torch.rad2deg(torch.arccos(torch.clamp(
        (out["R_1"].diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2, -1, 1)))
    check("R_1 ist NICHT die Identitaet (EXP-100 wirkt wirklich)",
          float(ang.median()) > 20.0,
          f"Winkel zu I: median={float(ang.median()):.1f} max={float(ang.max()):.1f} Grad")
    check("Ziele tragen keinen Autograd-Graphen",
          (not out["u_t_R"].requires_grad) and (not out["R_1"].requires_grad))
    check("Vorhersagen HABEN einen Autograd-Graphen", out["pred_u_t_R"].requires_grad)

    # compute_losses liefert den Verlust PRO FRAGMENT ([BxF]); die Reduktion
    # macht sonst das LightningModule. Fuer den Verdrahtungstest genuegt der
    # Mittelwert - entscheidend ist, dass Gradienten fliessen und endlich sind.
    losses = den.compute_losses(out)
    for k, v in losses.items():
        check(f"Verlust {k} endlich", bool(torch.isfinite(v).all()),
              f"shape {tuple(v.shape)}, Mittel {float(v.mean()):.4f}")
    check("Verluste sind pro Fragment definiert",
          all(v.shape[0] == out["R_1"].shape[0] for v in losses.values()))

    loss = sum(v.mean() for v in losses.values())
    loss.backward()
    grads = [p.grad for p in den.parameters() if p.grad is not None]
    check("Rueckwaertsschritt erzeugt Gradienten", len(grads) > 0, f"{len(grads)} Tensoren")
    check("alle Gradienten endlich", all(bool(torch.isfinite(g).all()) for g in grads))
    gn = torch.sqrt(sum((g ** 2).sum() for g in grads))
    check("Gradientennorm > 0", float(gn) > 0, f"||g|| = {float(gn):.3e}")

    # ---------------------------------------------------------------
    # Der Sampler: sechs Aufrufstellen wurden dort umbenannt bzw. auf die
    # Identitaet umgestellt. Ein echter ODE-Lauf trifft alle gleichzeitig.
    print("\nSampler mit echtem Modell:")
    from copy import deepcopy

    from sigmadock.diff.sampling import sample_notebook, sampler

    # Beide Sampler-Varianten - je drei gepatchte Aufrufstellen.
    for name, fn, extra in (("sampler", sampler, {}),
                            ("sample_notebook", sample_notebook, {"seed": 0})):
        with torch.no_grad():
            b2 = Batch.from_data_list(deepcopy(items))
            # sample_notebook gibt zusaetzlich Kanten zurueck - zweites Element
            # ist in beiden Faellen die Positionstrajektorie.
            res = fn(denoiser=den, batch=b2, num_steps=5, **extra)
        traj = np.asarray(res[1])
        check(f"{name}() laeuft durch", True, f"{len(traj)} Trajektorienschritte")
        check(f"{name}: Endpositionen endlich", bool(np.isfinite(traj[-1]).all()))
        check(f"{name}: Positionen haben sich bewegt",
              float(np.abs(traj[-1] - traj[0]).max()) > 1e-3,
              f"max |d| = {float(np.abs(traj[-1] - traj[0]).max()):.3f}")

    # Die INFERENZ-Pipeline (sample_conformer=True) muss ein exakter No-Op sein:
    # dort ist ref_conf_pos identisch zu ref_pos, also R_1 = I wie in Minimal.
    print("\nInferenzpfad (sample_conformer=True) muss R_1 = I liefern:")
    ds_inf = build_dataset(sample_conformer=True)
    d_inf = next(d for d in (ds_inf[i] for i in range(len(ds_inf))) if d is not None)
    b_inf = Batch.from_data_list([d_inf])
    check("ref_conf_pos == ref_pos an der Inferenz",
          bool(torch.allclose(b_inf.ref_conf_pos, b_inf.ref_pos, atol=0, rtol=0)))
    _, _, R1_inf, _ = den._get_initial_states(b_inf)
    ang_inf = torch.rad2deg(torch.arccos(torch.clamp(
        (R1_inf.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2, -1, 1)))
    check("R_1 == I an der Inferenz (EXP-100 ist dort ein No-Op)",
          float(ang_inf.max()) < 1e-4, f"max Winkel = {float(ang_inf.max()):.2e} Grad")

    print("\n" + "=" * 78)
    if FAILURES:
        print(f"FEHLGESCHLAGEN: {len(FAILURES)}")
        for f in FAILURES:
            print("  - " + f)
        sys.exit(1)
    print("ECHTER TRAININGSSCHRITT BESTANDEN")
