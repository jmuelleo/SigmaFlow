"""
Diagnostic script: prints how far the network's predicted vector field deviates
from the TRUE vector field, at every integration step of a real sampling run.

scripts/sample.py computes this same `all_losses` list inside sampler() (sampling.py)
but discards it (`_, pos, _ = self.sampler(batch=batch)` in SamplingModule.predict_step).
This script reuses the same checkpoint/data loading path as scripts/sample.py, but calls
sampler() directly on one batch and prints the per-step loss_trans/loss_R instead of
throwing them away.

Usage: same Hydra overrides as slurm/sample.sh, e.g.
    python scripts/diagnose_vector_field.py \
        ckpt=<path to .ckpt> \
        data_dir=notebooks \
        experiment=dummy_train \
        graph.sample_conformer=false
"""

from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch_geometric.loader import DataLoader as GeometricDataLoader

from sigmadock.data import SigmaDataset
from sigmadock.datafronts import MetaFront
from sigmadock.diff.sampling import sampler
from sigmadock.diff.sigma_flow_generator import SigmaFlowGenerator
from sigmadock.sampling_setup import (
    build_sampling_datafront,
    experiment_name_is_set,
    prepare_sampling_cfg,
    resolve_sampling_data_dir,
)
from sigmadock.trainer import SigmaLightningModule
from sigmadock.utils import load_from_scratch

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
    print(f"Dataset size: {len(dataset)}")

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

    print(f"Running sampler() with use_true_vector_field=False, {cfg.ode.num_steps} steps...")
    final_batch, all_pos, all_losses = sampler(
        denoiser=denoiser,
        batch=batch,
        t_min=cfg.ode.t_min,
        rho=cfg.ode.rho,
        num_steps=cfg.ode.num_steps,
        solver=cfg.ode.solver,
        discretization=cfg.ode.discretization,
        use_true_vector_field=False,
        verbose=False,
        debug_first_step=True,
    )

    print(f"\n{'step':>5} {'loss_trans (mean)':>20} {'loss_R (mean)':>16}")
    for i, losses in enumerate(all_losses):
        print(f"{i:>5} {losses['loss_trans'].mean().item():>20.4f} {losses['loss_R'].mean().item():>16.4f}")

    # Real-coordinate final deviation (Angstrom), NOT the (1-t)-singular vector-field loss above.
    from sigmadock.oracle import HPARAMS as _HPARAMS

    is_lig = torch.where(final_batch.frag_idx_map != -1)[0]
    ref_lig_pos = final_batch.ref_pos
    pred_lig_pos = final_batch.pos_t * _HPARAMS.general.dimensional_scale + final_batch.pocket_com.repeat_interleave(
        torch.bincount(final_batch.batch), dim=0
    )
    dev = (ref_lig_pos[is_lig] - pred_lig_pos[is_lig]).norm(dim=-1)
    print(f"\nFinal per-atom deviation from true bound pose (Angstrom-scale): mean={dev.mean().item():.3f}, "
          f"median={dev.median().item():.3f}, max={dev.max().item():.3f}")



if __name__ == "__main__":
    diagnose()
