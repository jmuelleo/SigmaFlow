"""FOLLOW-UP to TEST 2: is MAIN's residual 1e-2 numerical, or a real frame error?

The main equivariance test put MAIN at ~6e-3 (translation) / ~1.4e-2 (rotation)
relative error, two orders of magnitude below the deliberately broken controls
(~1.0-2.0) but above the 1e-3 tolerance declared in advance.

Two candidate explanations, which this script separates:

  H1 DISCRETE GRAPH REBUILD. _update_batch reconstructs interaction edges from
     distances with a cutoff. A global rotation preserves distances exactly in
     exact arithmetic, but not bit-exactly in float32, so a pair sitting at the
     cutoff can enter or leave the neighbour list. One edge changing alters the
     output discontinuously. Prediction: the error is concentrated on samples
     whose EDGE COUNT changed; samples with an identical graph should be near
     machine precision.

  H2 REAL FRAME ERROR. Prediction: the error is present regardless of whether
     the graph changed, and does not shrink with better precision.

Decisive because H1 and H2 make opposite predictions on the SAME data.

Usage: identical Hydra overrides as equivariance.py.
"""

import json
import sys
from pathlib import Path

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch_geometric.loader import DataLoader as GeometricDataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from equivariance import field_at_state, haar_rotations, pin  # noqa: E402,F401

from sigmadock.data import SigmaDataset  # noqa: E402
from sigmadock.diff.sigma_flow_generator import SigmaFlowGenerator  # noqa: E402
from sigmadock.sampling_setup import (  # noqa: E402
    build_sampling_datafront, prepare_sampling_cfg, resolve_sampling_data_dir,
)
from sigmadock.trainer import SigmaLightningModule  # noqa: E402
from sigmadock.utils import load_from_scratch  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
T_PROBE = 0.5


@torch.no_grad()
def edge_signature(denoiser, batch, t_value, base_seed, Q=None, a=None):
    """Number of edges of the graph the model actually sees at this state."""
    from copy import deepcopy
    b = denoiser._prepare_batch(deepcopy(batch))
    pos_0, trans_1, R_1, num_fragments = denoiser._get_initial_states(b)
    B = int(num_fragments.shape[0])
    t = torch.full((B,), float(t_value), device=b.x.device, dtype=pos_0.dtype)
    t_batch = t.repeat_interleave(num_fragments)
    pin(base_seed)
    sf = denoiser._sample_flow(trans_1=trans_1, R_1=R_1, t_batch=t_batch)
    pos_t = denoiser._apply_transformations(
        pos_0=pos_0, batch=b, trans_1=trans_1, R_1=R_1,
        R_t=sf["R_t"], trans_t=sf["trans_t"])
    if Q is not None:
        pos_0 = pos_0 @ Q.T + a
        pos_t = pos_t @ Q.T + a
    b = denoiser._update_batch(batch=b, pos_0=pos_0, pos_t=pos_t)
    return int(b.edge_index.shape[1])


@hydra.main(version_base=None, config_path="../../conf/", config_name="sampling/base")
def main(global_cfg: DictConfig) -> None:
    cfg = prepare_sampling_cfg(global_cfg)
    diag = cfg.get("diag", {})
    out_dir = Path(diag.get("out_dir", PROJECT_ROOT / "diagnostics/rotation_completion/out"))
    out_dir.mkdir(parents=True, exist_ok=True)
    n_complexes = int(diag.get("n_complexes", 30))
    n_rot = int(diag.get("n_rot", 5))
    batch_size = int(diag.get("batch_size", 4))

    pl.seed_everything(int(cfg.seed), workers=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_DIR = resolve_sampling_data_dir(cfg)
    dataset = SigmaDataset(
        datafront=build_sampling_datafront(cfg, DATA_DIR),
        pocket_com_noise=cfg.graph.pocket_com_noise,
        pocket_distance_cutoff=cfg.graph.pocket_distance_cutoff,
        pocket_distance_noise=cfg.graph.pocket_distance_noise,
        prot_coordinate_distance_noise=cfg.graph.pocket_coordinate_jitter,
        use_esm_embeddings=cfg.graph.use_esm_embeddings,
        ignore_triangulation=cfg.graph.ignore_triangulation,
        lig_coordinate_distance_noise=0.0,
        alignment_tries=cfg.graph.alignment_tries,
        fragmentation_strategy=cfg.graph.fragmentation_strategy,
        ignore_conjugated_torsions=cfg.graph.ignore_conjugated_torsion,
        pb_check=False, get_mol_info=True, seed=int(cfg.seed),
        random_rotation=cfg.graph.random_rotation,
        sample_conformer=cfg.graph.sample_conformer,
        skip_bounds_check=True, force_retry=True)
    model: SigmaLightningModule = load_from_scratch(
        Path(cfg.model.ckpt_dir), load_ema=cfg.model.use_ema,
        enforced_cfg={"sigma_min": 0.0, "cache_path": PROJECT_ROOT / cfg.model.cached_s03_dir}
        if cfg.model.cached_s03_dir is not None else None, strict=True)
    denoiser: SigmaFlowGenerator = (model.ema_model if cfg.model.use_ema else model).model
    denoiser.eval().to(device)
    dt = next(denoiser.parameters()).dtype

    subset = torch.utils.data.Subset(dataset, list(range(min(n_complexes, len(dataset)))))
    loader = GeometricDataLoader(subset, batch_size=batch_size, shuffle=False)

    rows = []
    for bi, batch in enumerate(loader):
        batch = batch.to(device)
        Qs = haar_rotations(n_rot, seed=4242 + bi, device=device, dtype=dt)
        g = torch.Generator().manual_seed(99 + bi)
        As = (torch.randn(n_rot, 3, generator=g) * 2.0).to(device=device, dtype=dt)

        e0 = edge_signature(denoiser, batch, T_PROBE, 7000 + bi)
        v_t0, v_r0 = field_at_state(denoiser, batch, T_PROBE, 7000 + bi)
        for r in range(n_rot):
            Q, a = Qs[r], As[r]
            e1 = edge_signature(denoiser, batch, T_PROBE, 7000 + bi, Q=Q, a=a)
            v_t1, v_r1 = field_at_state(denoiser, batch, T_PROBE, 7000 + bi,
                                        Q=Q, a=a, rotate_state=True, apply_frame_fix=True)
            et = ((v_t1 @ Q - v_t0).norm(dim=-1) / v_t0.norm(dim=-1).clamp_min(1e-12))
            er = ((v_r1 - v_r0).norm(dim=-1) / v_r0.norm(dim=-1).clamp_min(1e-12))
            rows.append({"batch": bi, "rot": r, "edges_before": e0, "edges_after": e1,
                         "edges_changed": bool(e0 != e1), "d_edges": abs(e1 - e0),
                         "trans_err_med": float(et.median()), "rot_err_med": float(er.median()),
                         "trans_err_max": float(et.max()), "rot_err_max": float(er.max())})
        print(f"  batch {bi+1}/{len(loader)} done")

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "equivariance_followup_raw.csv", index=False)

    L = ["", "=" * 94,
         "FOLLOW-UP: ist der Restfehler numerisch (Graph-Neuaufbau) oder ein Rahmenfehler?",
         "=" * 94,
         f"Paare gesamt: {len(df)}   davon mit VERAENDERTER Kantenzahl: "
         f"{int(df.edges_changed.sum())} ({100*df.edges_changed.mean():.1f}%)",
         f"Kantenzahl-Differenz wenn veraendert: median={df[df.edges_changed].d_edges.median() if df.edges_changed.any() else 0:.0f}",
         "", f"{'Gruppe':<28}{'n':>6}{'trans err med':>16}{'rot err med':>14}{'rot err max':>14}"]
    L.append("-" * 94)
    for lbl, m in (("Graph IDENTISCH", ~df.edges_changed), ("Graph VERAENDERT", df.edges_changed)):
        d = df[m]
        if not len(d):
            L.append(f"{lbl:<28}{0:>6}   (keine Faelle)")
            continue
        L.append(f"{lbl:<28}{len(d):>6}{d.trans_err_med.median():>16.2e}"
                 f"{d.rot_err_med.median():>14.2e}{d.rot_err_max.max():>14.2e}")
    L += ["", "DEUTUNG:",
          "  H1 bestaetigt (numerisch): 'Graph IDENTISCH' liegt nahe Maschinengenauigkeit",
          "     (<1e-4) und 'Graph VERAENDERT' traegt den Fehler.",
          "  H2 bestaetigt (Rahmenfehler): beide Gruppen liegen gleich hoch bei ~1e-2.",
          "  Zur Erinnerung: ein ECHTER Konventionsfehler liegt laut Kontrollen bei 1.0-2.0."]
    txt = "\n".join(L)
    print(txt)
    (out_dir / "equivariance_followup_summary.txt").write_text(txt, encoding="utf-8")
    (out_dir / "equivariance_followup.json").write_text(json.dumps({
        "n_pairs": len(df), "frac_edges_changed": float(df.edges_changed.mean()),
        "err_graph_identical": None if (~df.edges_changed).sum() == 0 else {
            "trans_med": float(df[~df.edges_changed].trans_err_med.median()),
            "rot_med": float(df[~df.edges_changed].rot_err_med.median())},
        "err_graph_changed": None if df.edges_changed.sum() == 0 else {
            "trans_med": float(df[df.edges_changed].trans_err_med.median()),
            "rot_med": float(df[df.edges_changed].rot_err_med.median())},
    }, indent=2))


if __name__ == "__main__":
    main()
