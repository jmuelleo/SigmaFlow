"""TEST 1 - Cosine-by-t diagnostic for the learned SigmaFlow vector field.

WHAT IT DOES
For states x_t drawn from the ACTUAL training probability path, compare the
trained network's predicted vector field against the exact Flow Matching
target, separately for translation and rotation, as a function of t.

WHY THIS EXACT CONSTRUCTION
The script mirrors SigmaFlowGenerator.forward() call for call, so that every
convention (frame, sign, trivialisation, fragment indexing, inertia
normalisation) is the one training actually used. The only deviations:
  1. t is FIXED to a grid value instead of sampled, and
  2. the base draw (x_0, R_0) is pinned per complex by re-seeding, so that all
     20 t-values for one complex lie on ONE coherent path rather than on 20
     unrelated ones. Without this the cosine-vs-t curve mixes paths and the
     trend is diluted.

TARGET DEFINITION - important
The training loss (compute_losses) consumes out["u_t_R"], which comes from
_sample_flow -> conditional_probability_path, i.e. the CONSTANT form
    u_t_R = log(R_0^T R_1),      u_t_trans = x_1 - x_0
NOT the (1-t)-divided form from _compute_true_vector_field (that one is only
used for logging inside sampler()). This script uses the former, because the
question is "does the network predict what it was trained on".

ROTATION IS COMPARED IN R^3, not as matrix entries: both predicted and target
so(3) elements are mapped through vee(), which is the faithful representation
of the Frobenius loss (||hat(a)-hat(b)||_F^2 = 2|a-b|^2).

Usage (same Hydra overrides as slurm/sample.sh):
    python diagnostics/rotation_completion/cosine_by_t.py \
        ckpt=<abs path to last.ckpt> \
        data_dir=<abs data dir> \
        experiment=posebusters \
        graph.sample_conformer=false \
        +diag.out_dir=<abs out dir> \
        +diag.max_complexes=200 \
        +diag.n_t=20
"""

import json
import os
import platform
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from torch_geometric.loader import DataLoader as GeometricDataLoader

from sigmadock.data import SigmaDataset
from sigmadock.diff import so3_utils
from sigmadock.diff.sigma_flow_generator import SigmaFlowGenerator
from sigmadock.sampling_setup import (
    build_sampling_datafront,
    prepare_sampling_cfg,
    resolve_sampling_data_dir,
)
from sigmadock.trainer import SigmaLightningModule
from sigmadock.utils import load_from_scratch

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unavailable"


def pin_base_draw(seed: int) -> None:
    """Pin BOTH RNGs the probability path consumes.

    R3_FlowMatcher.sample_init uses torch.randn; SO3_FlowMatcher.sample_init
    goes through so3_utils.sample_uniform, which is numpy. Seeding only torch
    would silently leave the rotational base draw floating between t-values.
    """
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))


@torch.no_grad()
def vector_fields_at_t(denoiser: SigmaFlowGenerator, batch, t_value: float, base_seed: int):
    """One faithful replay of forward() at a fixed t. Returns a dict of tensors."""
    b = denoiser._prepare_batch(deepcopy(batch))

    pos_0, trans_1, R_1, num_fragments = denoiser._get_initial_states(b)
    B = int(num_fragments.shape[0])
    t = torch.full((B,), float(t_value), device=b.x.device, dtype=pos_0.dtype)
    t_batch = t.repeat_interleave(num_fragments)

    pin_base_draw(base_seed)
    sampled_flow = denoiser._sample_flow(trans_1=trans_1, R_1=R_1, t_batch=t_batch)

    pos_t = denoiser._apply_transformations(
        pos_0=pos_0, batch=b, trans_1=trans_1, R_1=R_1,
        R_t=sampled_flow["R_t"], trans_t=sampled_flow["trans_t"],
    )
    b = denoiser._update_batch(batch=b, pos_0=pos_0, pos_t=pos_t)

    lig_forces, forces_idxs = denoiser._compute_forces(batch=b, t=t)
    f_frag, tau_frag, mass, inertia_t = denoiser._compute_fragment_dynamics(
        batch=b, R_t=sampled_flow["R_t"], trans_t=sampled_flow["trans_t"],
        R_1=R_1, lig_forces=lig_forces, forces_idxs=forces_idxs,
    )
    updates = denoiser._predict_fragment_updates(
        force_per_fragment=f_frag, torque_per_fragment=tau_frag,
        frag_mass=mass, frag_inertia_t=inertia_t,
    )
    vf = denoiser._compute_vector_field(sampled_flow, updates, t_batch)

    return {
        "pred_trans": vf["pred_u_t_trans"],          # [F,3]
        "targ_trans": sampled_flow["u_t_trans"],     # [F,3]
        "pred_rot": so3_utils.vee(vf["pred_u_t_R"]),      # [F,3]
        "targ_rot": so3_utils.vee(sampled_flow["u_t_R"]),  # [F,3]
        "num_fragments": num_fragments,
        "batch_obj": b,
        "torque": tau_frag,
        "inertia": inertia_t,
    }


def cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    na, nb = a.norm(dim=-1), b.norm(dim=-1)
    ok = (na > 1e-12) & (nb > 1e-12)
    out = torch.full_like(na, float("nan"))
    out[ok] = (a[ok] * b[ok]).sum(-1) / (na[ok] * nb[ok])
    return out


@hydra.main(version_base=None, config_path="../../conf/", config_name="sampling/base")
def main(global_cfg: DictConfig) -> None:
    cfg = prepare_sampling_cfg(global_cfg)
    diag = cfg.get("diag", {})
    out_dir = Path(diag.get("out_dir", PROJECT_ROOT / "diagnostics/rotation_completion/out"))
    out_dir.mkdir(parents=True, exist_ok=True)
    max_complexes = int(diag.get("max_complexes", 200))
    n_t = int(diag.get("n_t", 20))
    batch_size = int(diag.get("batch_size", 8))

    pl.seed_everything(int(cfg.seed), workers=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        pb_check=False, get_mol_info=True, seed=int(cfg.seed),
        random_rotation=cfg.graph.random_rotation,
        sample_conformer=cfg.graph.sample_conformer,
        skip_bounds_check=True, force_retry=True,
    )
    print(f"Dataset size: {len(dataset)}; using first {min(max_complexes, len(dataset))}")

    model: SigmaLightningModule = load_from_scratch(
        Path(cfg.model.ckpt_dir), load_ema=cfg.model.use_ema,
        enforced_cfg={"sigma_min": 0.0, "cache_path": PROJECT_ROOT / cfg.model.cached_s03_dir}
        if cfg.model.cached_s03_dir is not None else None,
        strict=True,
    )
    denoiser: SigmaFlowGenerator = (model.ema_model if cfg.model.use_ema else model).model
    denoiser.eval().to(device)

    idx = list(range(min(max_complexes, len(dataset))))
    subset = torch.utils.data.Subset(dataset, idx)
    loader = GeometricDataLoader(subset, batch_size=batch_size, shuffle=False)
    t_grid = torch.linspace(0.01, 0.99, n_t).tolist()

    records = []
    for bi, batch in enumerate(loader):
        batch = batch.to(device)
        B = int(batch.num_graphs)
        # ligand atom count per complex
        lig_mask = batch.frag_idx_map != -1
        lig_per_complex = torch.zeros(B, dtype=torch.long, device=device)
        lig_per_complex.index_add_(0, batch.batch[lig_mask],
                                   torch.ones(int(lig_mask.sum()), dtype=torch.long, device=device))
        for ti, tv in enumerate(t_grid):
            try:
                r = vector_fields_at_t(denoiser, batch, tv, base_seed=1000 + bi)
            except Exception as e:  # never let one batch kill the sweep
                print(f"[WARN] batch {bi} t={tv:.3f} failed: {type(e).__name__}: {e}")
                continue
            nf = r["num_fragments"]
            frag2cplx = torch.arange(B, device=device).repeat_interleave(nf)
            # fragment atom counts
            cont = denoiser.get_flat_fragment_index(r["batch_obj"])
            valid = cont >= 0
            frag_sizes = torch.bincount(cont[valid], minlength=int(nf.sum()))

            c_tr = cos(r["pred_trans"], r["targ_trans"]).cpu().numpy()
            c_ro = cos(r["pred_rot"], r["targ_rot"]).cpu().numpy()
            npt = r["pred_trans"].norm(dim=-1).cpu().numpy()
            ntt = r["targ_trans"].norm(dim=-1).cpu().numpy()
            npr = r["pred_rot"].norm(dim=-1).cpu().numpy()
            ntr = r["targ_rot"].norm(dim=-1).cpu().numpy()
            tq = r["torque"].norm(dim=-1).cpu().numpy()
            fc = frag2cplx.cpu().numpy()
            fs = frag_sizes.cpu().numpy()[: len(fc)]
            lc = lig_per_complex.cpu().numpy()

            for k in range(len(c_tr)):
                records.append({
                    "batch": bi, "complex_in_batch": int(fc[k]),
                    "global_complex": int(bi * batch_size + fc[k]),
                    "fragment": k, "t": float(tv),
                    "ligand_size": int(lc[fc[k]]),
                    "fragment_size": int(fs[k]) if k < len(fs) else -1,
                    "cos_trans": float(c_tr[k]), "cos_rot": float(c_ro[k]),
                    "norm_pred_trans": float(npt[k]), "norm_targ_trans": float(ntt[k]),
                    "ratio_trans": float(npt[k] / ntt[k]) if ntt[k] > 1e-12 else float("nan"),
                    "norm_pred_rot": float(npr[k]), "norm_targ_rot": float(ntr[k]),
                    "ratio_rot": float(npr[k] / ntr[k]) if ntr[k] > 1e-12 else float("nan"),
                    "torque_norm": float(tq[k]),
                    "neg_trans": bool(c_tr[k] < 0), "neg_rot": bool(c_ro[k] < 0),
                })
        print(f"  batch {bi+1}/{len(loader)} done ({len(records)} observations)")

    import pandas as pd
    df = pd.DataFrame.from_records(records)
    raw = out_dir / "cosine_by_t_raw.csv"
    df.to_csv(raw, index=False)

    meta = {
        "test": "cosine_by_t",
        "checkpoint": str(cfg.model.ckpt_dir),
        "git_hash": git_hash(),
        "config": str(getattr(cfg, "experiments", {}).get("name", "unknown")),
        "model_variant": "SigmaFlow d_frame_fix",
        "seed": int(cfg.seed),
        "datetime": datetime.now().isoformat(timespec="seconds"),
        "n_complexes": len(idx), "n_t": n_t, "n_observations": len(df),
        "device": str(device),
        "torch": torch.__version__, "numpy": np.__version__,
        "python": platform.python_version(),
        "target_definition": "conditional_probability_path (constant form), vee-mapped",
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    (out_dir / "cosine_by_t_meta.json").write_text(json.dumps(meta, indent=2))

    # ---- human-readable summary ----
    lines = ["", "=" * 104,
             "COSINE-BY-t   (positiv = Vorhersage zeigt in Zielrichtung)", "=" * 104,
             f"{'t-bin':<12}{'n':>7}{'trans mean':>12}{'trans med':>11}{'rot mean':>11}"
             f"{'rot med':>10}{'rot q25':>9}{'rot q75':>9}{'rot<0 %':>9}{'rot>0.3 %':>11}{'rot ratio':>11}"]
    lines.append("-" * 104)
    df["bin"] = np.clip((df["t"] * 10).astype(int), 0, 9)
    for b in range(10):
        d = df[df["bin"] == b]
        if len(d) == 0:
            continue
        lines.append(
            f"{f'[{b/10:.1f},{(b+1)/10:.1f})':<12}{len(d):>7}"
            f"{d.cos_trans.mean():>12.3f}{d.cos_trans.median():>11.3f}"
            f"{d.cos_rot.mean():>11.3f}{d.cos_rot.median():>10.3f}"
            f"{d.cos_rot.quantile(.25):>9.3f}{d.cos_rot.quantile(.75):>9.3f}"
            f"{100*(d.cos_rot < 0).mean():>8.1f}%{100*(d.cos_rot > 0.3).mean():>10.1f}%"
            f"{d.ratio_rot.median():>11.3f}")
    lines.append("")
    lines.append(f"GESAMT  trans: mean={df.cos_trans.mean():.3f} median={df.cos_trans.median():.3f} "
                 f"neg={100*(df.cos_trans<0).mean():.1f}%  |  "
                 f"rot: mean={df.cos_rot.mean():.3f} median={df.cos_rot.median():.3f} "
                 f"neg={100*(df.cos_rot<0).mean():.1f}%")
    lines.append(f"Normverhaeltnis |pred|/|target|  trans median={df.ratio_trans.median():.3f}  "
                 f"rot median={df.ratio_rot.median():.3f}")

    for name, col in (("Fragmentgroesse", "fragment_size"),
                      ("Ligandgroesse", "ligand_size"),
                      ("Zielnorm Rotation", "norm_targ_rot")):
        q = df[col].quantile([1/3, 2/3]).values
        lines += ["", f"Rotation nach {name} (Tertile):"]
        for lo, hi, lbl in ((-np.inf, q[0], "klein"), (q[0], q[1], "mittel"), (q[1], np.inf, "gross")):
            d = df[(df[col] > lo) & (df[col] <= hi)]
            if len(d):
                lines.append(f"   {lbl:<8} n={len(d):>6}  cos mean={d.cos_rot.mean():>7.3f}  "
                             f"median={d.cos_rot.median():>7.3f}  neg={100*(d.cos_rot<0).mean():>5.1f}%")

    txt = "\n".join(lines)
    print(txt)
    (out_dir / "cosine_by_t_summary.txt").write_text(txt, encoding="utf-8")
    print(f"\nRohdaten : {raw}\nMetadaten: {out_dir/'cosine_by_t_meta.json'}")


if __name__ == "__main__":
    main()
