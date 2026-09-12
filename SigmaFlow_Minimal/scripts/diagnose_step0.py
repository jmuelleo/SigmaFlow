"""
Pinpoints exactly why sampler()'s step 0 shows inf/nan loss (see diagnose_vector_field.py).
Manually replicates step 0 of sampler(), but checks isnan/isinf SEPARATELY on:
  - the raw network output (lig_pseudoforces) - is the network's own prediction already broken?
  - the predicted vector field (pred_u_t_trans/pred_u_t_R, after force/torque aggregation)
  - the TRUE vector field (u_t_trans/u_t_R) used only for the diagnostic loss, not the real trajectory
For whichever one is broken, prints per-fragment detail (rotation angle of R_0 vs R_1, distance
trans_0 vs trans_1) to identify the responsible fragment and likely numerical cause.

Usage: same Hydra overrides as diagnose_vector_field.py, e.g.
    python scripts/diagnose_step0.py \
        ckpt=<path to .ckpt> data_dir=notebooks experiment=dummy_train graph.sample_conformer=false
"""

from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch_geometric.loader import DataLoader as GeometricDataLoader

from sigmadock.data import SigmaDataset
from sigmadock.diff.sigma_flow_generator import SigmaFlowGenerator
from sigmadock.sampling_setup import (
    build_sampling_datafront,
    prepare_sampling_cfg,
    resolve_sampling_data_dir,
)
from sigmadock.trainer import SigmaLightningModule
from sigmadock.utils import load_from_scratch

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def rotation_angle_deg(R: torch.Tensor) -> torch.Tensor:
    """Angle (degrees) of rotation matrices [..., 3, 3], via trace: angle = arccos((tr(R)-1)/2)."""
    tr = R.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos_angle = ((tr - 1) / 2).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.arccos(cos_angle))


def describe(name: str, x: torch.Tensor) -> None:
    isnan = torch.isnan(x)
    isinf = torch.isinf(x)
    print(
        f"  {name}: shape={tuple(x.shape)} nan_count={int(isnan.sum())} inf_count={int(isinf.sum())} "
        f"min={x[torch.isfinite(x)].min().item() if torch.isfinite(x).any() else 'n/a'} "
        f"max={x[torch.isfinite(x)].max().item() if torch.isfinite(x).any() else 'n/a'}"
    )


@hydra.main(version_base=None, config_path="../conf/", config_name="sampling/base")
def diagnose(global_cfg: DictConfig) -> None:
    cfg = prepare_sampling_cfg(global_cfg)
    pl.seed_everything(int(cfg.seed), workers=True)

    CKPT_DIR = Path(cfg.model.ckpt_dir)
    DATA_DIR = resolve_sampling_data_dir(cfg)
    datafront = build_sampling_datafront(cfg, DATA_DIR)

    dataset = SigmaDataset(
        datafront=datafront,
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
        pb_check=False,
        get_mol_info=True,
        seed=int(cfg.seed),
        random_rotation=cfg.graph.random_rotation,
        sample_conformer=cfg.graph.sample_conformer,
        skip_bounds_check=True,
        force_retry=True,
    )
    loader = GeometricDataLoader(dataset, batch_size=len(dataset), shuffle=False)
    batch = next(iter(loader))

    model: SigmaLightningModule = load_from_scratch(
        CKPT_DIR,
        load_ema=cfg.model.use_ema,
        enforced_cfg={
            "sigma_min": 0.0,
            "cache_path": PROJECT_ROOT / cfg.model.cached_s03_dir,
        }
        if cfg.model.cached_s03_dir is not None
        else None,
        strict=True,
    )
    lightning_model = model.ema_model if cfg.model.use_ema else model
    denoiser: SigmaFlowGenerator = lightning_model.model
    denoiser.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    denoiser.to(device)
    batch = batch.to(device)

    t_min = float(cfg.ode.t_min)
    print(f"=== Replicating sampler() step 0 manually, t_min={t_min} ===\n")

    # --- Same setup as sampler()/sample_notebook() before the loop ---
    pos_0, trans_1, R_1, num_fragments = denoiser._get_initial_states(batch)
    sampled_init = denoiser.flow_matcher.sample_init(torch.sum(num_fragments), batch.x.device)
    trans_0, R_0 = sampled_init["trans_0"], sampled_init["R_0"]

    print("--- Initial random state (trans_0, R_0) vs reference (trans_1, R_1) ---")
    dist_0_1 = (trans_0 - trans_1).norm(dim=-1)
    angle_0_1 = rotation_angle_deg(R_0.transpose(-1, -2) @ R_1)
    for f in range(trans_0.shape[0]):
        flag = "  <== SUSPECT (near 180 deg)" if angle_0_1[f] > 170 else ""
        print(
            f"  fragment {f}: |trans_0 - trans_1| = {dist_0_1[f].item():.3f}, "
            f"angle(R_0, R_1) = {angle_0_1[f].item():.2f} deg{flag}"
        )
    print()

    pos_T = denoiser._apply_transformations(
        pos_0=pos_0, batch=batch, trans_1=trans_1, R_1=R_1, trans_t=trans_0, R_t=R_0
    )
    describe("pos_T (initial atom positions)", pos_T)
    print()

    batch = denoiser._update_batch(batch=batch, pos_0=pos_0, pos_t=pos_T)

    t_batch = torch.tensor(t_min, device=device).repeat_interleave(sum(num_fragments))

    # --- 1) Raw network output, BEFORE any aggregation ---
    lig_pseudoforces, forces_idxs = denoiser._compute_forces(
        batch=batch, t=torch.tensor([t_min] * batch.num_graphs, device=device)
    )
    print("--- 1) Raw network output (lig_pseudoforces, per-atom) ---")
    describe("lig_pseudoforces", lig_pseudoforces)
    print()

    # --- 2) Aggregated force/torque -> predicted vector field (Newton-Maruyama) ---
    force_per_fragment, torque_per_fragment, frag_mass, frag_inertia_t = denoiser._compute_fragment_dynamics(
        batch=batch, R_t=R_0, trans_t=trans_0, R_1=R_0, lig_forces=lig_pseudoforces, forces_idxs=forces_idxs
    )
    print("--- 2) Per-fragment force/torque (pre Newton-Maruyama) ---")
    describe("force_per_fragment", force_per_fragment)
    describe("torque_per_fragment", torque_per_fragment)
    describe("frag_mass", frag_mass)
    describe("frag_inertia_t", frag_inertia_t)
    print()

    fragment_updates = denoiser._predict_fragment_updates(
        force_per_fragment=force_per_fragment,
        torque_per_fragment=torque_per_fragment,
        frag_mass=frag_mass,
        frag_inertia_t=frag_inertia_t,
    )
    pred_vector_field = denoiser._compute_vector_field({"R_t": R_0, "trans_t": trans_0}, fragment_updates, t_batch)
    print("--- 3) PREDICTED vector field (network-derived, used in the real trajectory) ---")
    describe("pred_u_t_trans", pred_vector_field["pred_u_t_trans"])
    describe("pred_u_t_R", pred_vector_field["pred_u_t_R"])
    print()

    # --- 4) TRUE vector field (used only for the diagnostic loss, NOT the real trajectory) ---
    true_vector_field = denoiser._compute_true_vector_field(
        trans_1=trans_1, R_1=R_1, Tt=trans_0, Rt=R_0, t_batch=t_batch
    )
    print("--- 4) TRUE vector field (diagnostic-only, compares against known bound pose) ---")
    describe("u_t_trans (true)", true_vector_field["u_t_trans"])
    describe("u_t_R (true)", true_vector_field["u_t_R"])
    print()

    print("=== Verdict ===")
    pred_broken = (not torch.isfinite(pred_vector_field["pred_u_t_trans"]).all()) or (
        not torch.isfinite(pred_vector_field["pred_u_t_R"]).all()
    )
    true_broken = (not torch.isfinite(true_vector_field["u_t_trans"]).all()) or (
        not torch.isfinite(true_vector_field["u_t_R"]).all()
    )
    raw_broken = not torch.isfinite(lig_pseudoforces).all()
    print(f"Raw network output finite: {not raw_broken}")
    print(f"Predicted vector field finite (this DOES affect the real sampling trajectory): {not pred_broken}")
    print(f"True vector field finite (this is ONLY a diagnostic/logging value): {not true_broken}")


if __name__ == "__main__":
    diagnose()
