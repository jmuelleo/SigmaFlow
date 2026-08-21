#!/usr/bin/env python3
"""
EXP-110 -- ein echter Trainingsschritt auf echten Daten.

Die anderen Audits pruefen Verdrahtung und Geometrie, aber alle mit
Stellvertreterrumpf oder ohne echten Graphen. Dieser hier ruft den echten
EquiformerV2 auf einem echten Komplexbatch auf und geht die volle Kette:
Vorwaertslauf, Loss, Rueckwaertslauf, Optimiererschritt.

Genau diese Luecke hat am 2026-08-21 zwei Fehler durchgelassen, die den
12h-Lauf in den ersten Minuten getoetet haetten:
  - forward() gab die Restschluessel force_per_fragment / torque_per_fragment
    zurueck, die es im Zwei-Kopf-Pfad nicht mehr gibt (NameError)
  - trainer.py entpackte score_terms["pseudoforces"], ebenfalls nicht mehr da

Aufruf (aus dem Variantenordner):
    PYTHONPATH=src python audits/test_real_training_step.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
VARIANT = HERE.parent
sys.path.insert(0, str(VARIANT / "src"))

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(cond), detail))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))


def sec(t: str) -> None:
    print()
    print("=" * 74)
    print(t)
    print("=" * 74)


from sigmadock.config import RunConfig, StructuralConfig, get_experiment_config  # noqa: E402
from sigmadock.data import SigmaDataModule  # noqa: E402
from sigmadock.datafronts import MetaFront  # noqa: E402
from sigmadock.diff.sigma_flow_generator import SigmaFlowGenerator  # noqa: E402
from sigmadock.net.model import EquiformerV2  # noqa: E402
from sigmadock.oracle import HPARAMS  # noqa: E402

# =========================================================================
sec("1. ECHTE DATEN LADEN (notebooks/dummy_data)")
a = RunConfig()
torch.manual_seed(0)

ec = get_experiment_config("dummy_train", root_dir=VARIANT / "notebooks")
front = MetaFront([ec])
check("Datafront gebaut", len(front) > 0, f"{len(front)} Paare")

structural = StructuralConfig()
dm = SigmaDataModule(
    train_datafront=front,
    val_datafront=front,
    test_datafront=front,
    batch_size=2,
    num_workers=0,
    persistent_workers=False,
    cache_factor=a.cache_factor,
    cache_cycles=a.cache_cycles,
    dataset_augmentation_factor=a.dataset_augmentation_factor,
    val_cycles=a.val_cycles,
    **structural.__dict__,
)
dm.setup("fit")
batch = next(iter(dm.train_dataloader()))
check("Batch erhalten", batch is not None, f"{int(getattr(batch, 'num_graphs', 0))} Graphen")

# =========================================================================
sec("2. ECHTES MODELL, ECHTER VORWAERTSLAUF")
model = EquiformerV2(
    use_esm_embeddings=a.use_esm_embeddings,
    num_layers=a.num_layers,
    num_heads=a.num_heads,
    atom_feature_dims=a.atom_feature_dims,
    average_degrees=a.average_degrees,
    edge_feature_dims=a.edge_feature_dims,
    lmax_list=a.l_max_list,
    mmax_list=a.m_max_list,
    protein_ligand_interactions=a.include_protein_ligand_interactions,
    ligand_ligand_interactions=a.include_fragment_fragment_interactions,
    sphere_channels=a.sphere_channels,
    edge_channels=a.edge_channels,
    attn_hidden_channels=a.attn_hidden_channels,
    attn_alpha_channels=a.attn_alpha_channels,
    attn_value_channels=a.attn_value_channels,
    ffn_hidden_channels=a.ffn_hidden_channels,
    t_emb_dim=a.t_emb_dim,
    t_emb_type=a.t_emb_type,
    t_emb_scale=a.t_emb_scale,
    distance_expansion_dim=a.distance_expansion_dim,
    smearing_type=a.smearing_type,
    radial_cutoff_function=a.radial_type,
    rel_distance=a.rel_distance,
    alpha_drop=a.attention_dropout,
    drop_path_rate=a.edge_dropout,
    zero_init_last=a.zero_init_last,
    share_edge_mlp=False,
)
denoiser = SigmaFlowGenerator(
    model,
    cache_path=VARIANT / "experiments" / "cache",
    cutoff_complex_interactions=HPARAMS.get_edge_spec("inter_complex").r_max
    if a.include_protein_ligand_interactions
    else -1,
    cutoff_fragment_interactions=HPARAMS.get_edge_spec("inter_fragments").r_max
    if a.include_fragment_fragment_interactions
    else -1,
    cutoff_complex_virtual=HPARAMS.get_edge_spec("complex_lv2pv").r_max,
    **a.__dict__,
)
denoiser.train()

out = denoiser(batch)
check("forward() laeuft durch", isinstance(out, dict), f"{len(out)} Schluessel")
for k in ("pred_u_t_trans", "pred_u_t_R", "u_t_trans", "u_t_R", "atom_trans_field", "atom_rot_field"):
    check(f"Schluessel '{k}' vorhanden", k in out)
for dead in ("force_per_fragment", "torque_per_fragment", "pseudoforces"):
    check(f"toter Schluessel '{dead}' ist WEG", dead not in out)

M = out["pred_u_t_trans"].shape[0]
print(f"    Fragmente im Batch: {M}")
check("pred_u_t_trans [M,3]", tuple(out["pred_u_t_trans"].shape) == (M, 3))
check("pred_u_t_R [M,3,3]", tuple(out["pred_u_t_R"].shape) == (M, 3, 3))
sk = (out["pred_u_t_R"] + out["pred_u_t_R"].transpose(-1, -2)).abs().max().item()
check("pred_u_t_R schiefsymmetrisch", sk < 1e-5, f"max|X+X^T| = {sk:.2e}")
check("alle Ausgaben endlich", all(torch.isfinite(v).all() for v in out.values() if torch.is_tensor(v)))

# =========================================================================
sec("3. LOSS")
losses = denoiser.compute_losses(out)
lt, lr = losses["loss_trans"], losses["loss_R"]
print(f"    loss_trans je Fragment: mean {lt.mean():.4f}  min {lt.min():.4f}  max {lt.max():.4f}")
print(f"    loss_R     je Fragment: mean {lr.mean():.4f}  min {lr.min():.4f}  max {lr.max():.4f}")
check("loss_trans endlich und > 0", torch.isfinite(lt).all() and lt.mean() > 0)
check("loss_R endlich und > 0", torch.isfinite(lr).all() and lr.mean() > 0)

# Groessenordnung: bei zero_init_last ist die Vorhersage anfangs ~0, also
# sollte der Loss etwa der quadrierten Norm des Ziels entsprechen.
tgt_t = out["u_t_trans"].pow(2).sum(-1)
tgt_r = out["u_t_R"].pow(2).sum(dim=(-1, -2))
print(f"    ||u_t_trans||^2 : mean {tgt_t.mean():.4f}")
print(f"    ||u_t_R||_F^2   : mean {tgt_r.mean():.4f}   (Haar-Erwartung ~ 2*(2.0 rad)^2 = 8)")
check("loss_trans in der Groessenordnung des Ziels",
      0.2 < (lt.mean() / tgt_t.mean()).item() < 5.0,
      f"Verhaeltnis {(lt.mean()/tgt_t.mean()).item():.3f}")
check("loss_R in der Groessenordnung des Ziels",
      0.2 < (lr.mean() / tgt_r.mean()).item() < 5.0,
      f"Verhaeltnis {(lr.mean()/tgt_r.mean()).item():.3f}")

total = 2.0 * lt.mean() + 0.5 * lr.mean()
print(f"    gewichteter Gesamtloss (2.0/0.5): {total.item():.4f}")

# =========================================================================
sec("4. RUECKWAERTSLAUF UND OPTIMIERERSCHRITT")
opt = torch.optim.AdamW(denoiser.parameters(), lr=1e-4)
opt.zero_grad(set_to_none=True)
total.backward()

g_tr = [p.grad for p in model.trans_block.parameters() if p.grad is not None]
g_ro = [p.grad for p in model.rot_block.parameters() if p.grad is not None]
n_tr = sum(g.abs().sum().item() for g in g_tr)
n_ro = sum(g.abs().sum().item() for g in g_ro)
n_all = sum(p.grad.abs().sum().item() for p in denoiser.parameters() if p.grad is not None)
print(f"    |grad| trans_block {n_tr:.4e} | rot_block {n_ro:.4e} | gesamt {n_all:.4e}")
check("trans_block bekommt Gradient", n_tr > 0)
check("rot_block bekommt Gradient", n_ro > 0)
check("alle Gradienten endlich",
      all(torch.isfinite(p.grad).all() for p in denoiser.parameters() if p.grad is not None))

frac = sum(1 for p in denoiser.parameters() if p.grad is not None) / sum(1 for _ in denoiser.parameters())
check("Anteil Parameter mit Gradient > 50 %", frac > 0.5, f"{100*frac:.1f} %")

before = model.rot_block.parameters().__next__().detach().clone()
opt.step()
after = next(model.rot_block.parameters()).detach()
check("Optimiererschritt aendert rot_block", not torch.equal(before, after),
      f"max|delta| = {(after-before).abs().max().item():.3e}")

print()
print("=" * 74)
ok = sum(1 for _, c, _ in CHECKS if c)
print(f"ERGEBNIS: {ok}/{len(CHECKS)} Checks bestanden")
print("=" * 74)
raise SystemExit(0 if ok == len(CHECKS) else 1)
