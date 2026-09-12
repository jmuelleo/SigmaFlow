#!/usr/bin/env python3
"""
EXP-110 -- trainiert der 12h-Lauf wirklich die gemeinte Groesse?

Der teuerste denkbare Fehler ist nicht ein Absturz, sondern ein Lauf, der
zwoelf Stunden lang sauber gegen ein FALSCHES Ziel optimiert. Dieser Audit
prueft die Zielgroessen auf ECHTEN Protein-Ligand-Batches, nicht auf
konstruierten Tensoren.

Die Kernpruefung ist konventionsunabhaengig. Statt Formeln nachzurechnen und
dabei dieselbe Konvention noch einmal anzunehmen, wird die DEFINIERENDE
Eigenschaft geprueft:

    Wer bei (x_t, t) startet und (1-t) lang mit u_t faehrt, muss exakt bei
    x_1 landen.

    Translation:  x_t + (1-t) * u_t_trans        == x_1
    Rotation:     R_t @ exp((1-t) * u_t_R)       == R_1

Das prueft in einem Zug: Log-Konvention, Rahmenkonvention (Koerper gegen
Welt), Multiplikationsseite und Zeitparametrisierung. Wenn irgendeine davon
falsch waere, landet man nicht bei R_1.

Aufruf (aus dem Variantenordner):
    PYTHONPATH=src python audits/test_targets_on_real_batch.py
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
from sigmadock.diff import so3_utils  # noqa: E402
from sigmadock.diff.sigma_flow_generator import SigmaFlowGenerator  # noqa: E402
from sigmadock.net.model import EquiformerV2  # noqa: E402
from sigmadock.oracle import HPARAMS  # noqa: E402

a = RunConfig()
torch.manual_seed(3)

# =========================================================================
sec("1. ECHTER BATCH")
front = MetaFront([get_experiment_config("dummy_train", root_dir=VARIANT / "notebooks")])
dm = SigmaDataModule(
    train_datafront=front, val_datafront=front, test_datafront=front,
    batch_size=3, num_workers=0, persistent_workers=False,
    cache_factor=a.cache_factor, cache_cycles=a.cache_cycles,
    dataset_augmentation_factor=a.dataset_augmentation_factor,
    val_cycles=a.val_cycles, **StructuralConfig().__dict__,
)
dm.setup("fit")
batch = next(iter(dm.train_dataloader()))
check("Batch geladen", batch is not None, f"{int(batch.num_graphs)} Komplexe")

model = EquiformerV2(
    use_esm_embeddings=a.use_esm_embeddings, num_layers=a.num_layers,
    num_heads=a.num_heads, atom_feature_dims=a.atom_feature_dims,
    average_degrees=a.average_degrees, edge_feature_dims=a.edge_feature_dims,
    lmax_list=a.l_max_list, mmax_list=a.m_max_list,
    protein_ligand_interactions=a.include_protein_ligand_interactions,
    ligand_ligand_interactions=a.include_fragment_fragment_interactions,
    sphere_channels=a.sphere_channels, edge_channels=a.edge_channels,
    attn_hidden_channels=a.attn_hidden_channels,
    attn_alpha_channels=a.attn_alpha_channels,
    attn_value_channels=a.attn_value_channels,
    ffn_hidden_channels=a.ffn_hidden_channels, t_emb_dim=a.t_emb_dim,
    t_emb_type=a.t_emb_type, t_emb_scale=a.t_emb_scale,
    distance_expansion_dim=a.distance_expansion_dim,
    smearing_type=a.smearing_type, radial_cutoff_function=a.radial_type,
    rel_distance=a.rel_distance, alpha_drop=a.attention_dropout,
    drop_path_rate=a.edge_dropout, zero_init_last=a.zero_init_last,
    share_edge_mlp=False,
)
den = SigmaFlowGenerator(
    model, cache_path=VARIANT / "experiments" / "cache",
    cutoff_complex_interactions=-1, cutoff_fragment_interactions=-1,
    cutoff_complex_virtual=HPARAMS.get_edge_spec("complex_lv2pv").r_max,
    **a.__dict__,
)

# =========================================================================
sec("2. ZIELGROESSEN AUF DEM ECHTEN BATCH")
b = den._prepare_batch(batch)
t = den._sample_time(b)
pos_0, trans_1, R_1, num_fragments = den._get_initial_states(b)
t_batch = t.repeat_interleave(num_fragments)
sf = den._sample_flow(trans_1=trans_1, R_1=R_1, t_batch=t_batch)

M = R_1.shape[0]
print(f"    Komplexe {int(b.num_graphs)}, Fragmente {M}, "
      f"Fragmente je Komplex {num_fragments.tolist()}")
print(f"    t je Komplex: {[round(x, 3) for x in t.tolist()]}")
check("R_1 ist [M,3,3]", tuple(R_1.shape) == (M, 3, 3))
check("R_t ist [M,3,3]", tuple(sf["R_t"].shape) == (M, 3, 3))
check("u_t_R ist [M,3,3]", tuple(sf["u_t_R"].shape) == (M, 3, 3))
check("u_t_trans ist [M,3]", tuple(sf["u_t_trans"].shape) == (M, 3))
check("t_batch ist [M]", tuple(t_batch.shape) == (M,))

# R_1 ist bei dieser Konstruktion per Definition die Identitaet, weil die
# Fragmentrotation relativ zum Konformer definiert wird.
I3 = torch.eye(3).expand_as(R_1)
dev_I = (R_1 - I3).abs().max().item()
print(f"    max|R_1 - I| = {dev_I:.2e}   (R_1 = I ist die Konstruktion in _get_initial_states)")

check("R_t ist eine echte Rotation",
      (sf["R_t"].transpose(-1, -2) @ sf["R_t"] - I3).abs().max().item() < 1e-5,
      f"max|R^T R - I| = {(sf['R_t'].transpose(-1,-2) @ sf['R_t'] - I3).abs().max().item():.2e}")
check("u_t_R ist schiefsymmetrisch",
      (sf["u_t_R"] + sf["u_t_R"].transpose(-1, -2)).abs().max().item() < 1e-5)

# =========================================================================
sec("3. DIE DEFINIERENDE EIGENSCHAFT")
print("    Wer bei (x_t, t) startet und (1-t) lang mit u_t faehrt, landet bei x_1.")
one_m_t = (1.0 - t_batch)

x_end = sf["trans_t"] + one_m_t[:, None] * sf["u_t_trans"]
err_x = (x_end - trans_1).abs().max().item()
check("Translation: x_t + (1-t) u_t == x_1", err_x < 1e-4, f"max|Fehler| = {err_x:.2e}")

R_end = sf["R_t"] @ so3_utils.exp(one_m_t[:, None, None] * sf["u_t_R"])
err_R = (R_end - R_1).abs().amax(dim=(-1, -2))
# Massgeblich ist der VOLLE Winkel R_0 -> R_1, dessen Logarithmus beim
# Pfadbau genommen wird, NICHT der verbleibende Winkel R_t -> R_1. Nachgemessen:
# die groessten Fehler haben volle Winkel von 178-180 Grad bei verbleibenden
# Winkeln ab 36 Grad. Der volle Winkel ist die Norm des Rotationsvektors von
# u_t_R, weil u_t_R = log(R_0^T R_1) die Endpunktform ist.
ang = so3_utils.vee(sf["u_t_R"]).norm(dim=-1) * 180.0 / 3.141592653589793
far = ang < 170.0
print(f"    voller Winkel R_0->R_1: min {ang.min():.1f}, median {ang.median():.1f}, max {ang.max():.1f} Grad")
# Abseits des Cut Locus muss es scharf stimmen.
check("Rotation abseits des Cut Locus (<170 Grad) exakt",
      bool(far.any()) and err_R[far].max().item() < 1e-4,
      f"n={int(far.sum())}, max|Fehler| = {err_R[far].max().item() if far.any() else float('nan'):.2e}")
# Nahe pi ist der Log schlecht konditioniert. Das ist eine dokumentierte
# Eigenschaft von so3_utils, identisch in Minimal, und wird hier nur
# quantifiziert statt als Fehler gewertet. Gemessen: bis ~1.6e-2 in
# Matrixeintraegen, also unter einem Grad, auf gut einem Prozent der Ziehungen.
check("Rotation insgesamt innerhalb der Cut-Locus-Toleranz",
      err_R.max().item() < 3e-2, f"max|Fehler| = {err_R.max().item():.2e}")

# Kann der Test ueberhaupt unterscheiden? Auf den ECHTEN Daten nicht, weil
# _get_initial_states R_1 = I setzt und damit alles auf EINER
# Ein-Parameter-Untergruppe liegt, wo links und rechts kommutieren. Die
# Unterscheidungskraft wird deshalb an synthetischem R_1 != I gezeigt.
g = torch.Generator().manual_seed(11)
R1s = so3_utils.exp(so3_utils.hat(torch.randn(400, 3, generator=g) * 0.9))
R0s = so3_utils.exp(so3_utils.hat(torch.randn(400, 3, generator=g) * 0.9))
ts = torch.rand(400, generator=g) * 0.8
us = so3_utils.log(R0s.transpose(-1, -2) @ R1s)
Rts = R0s @ so3_utils.exp(ts[:, None, None] * us)
Es = so3_utils.exp((1.0 - ts)[:, None, None] * us)
ang_s = so3_utils.vee(us).norm(dim=-1) * 180.0 / 3.141592653589793
far_s = ang_s < 170.0
e_right = (Rts @ Es - R1s).abs().amax(dim=(-1, -2))
e_left = (Es @ Rts - R1s).abs().max().item()
print(f"    synthetisch, R_1 != I:  rechts (Winkel<170) {e_right[far_s].max().item():.2e}   "
      f"links {e_left:.2e}")
check("synthetisch: RECHTSmultiplikation trifft R_1", e_right[far_s].max().item() < 1e-4,
      f"n={int(far_s.sum())}, max {e_right[far_s].max().item():.2e}")
check("synthetisch: LINKSmultiplikation trifft R_1 NICHT", e_left > 1e-2, f"{e_left:.2e}")

sec("4. ZIEL IST KONSTANT LAENGS DER TRAJEKTORIE")
print("    Die Endpunktform log(R_0^T R_1) haengt nicht von t ab; die")
print("    Zustandsform log(R_t^T R_1)/(1-t) muss denselben Wert geben.")
u_state = so3_utils.log(sf["R_t"].transpose(-1, -2) @ R_1) / one_m_t[:, None, None]
d = (u_state - sf["u_t_R"]).abs().max().item()
check("log(R_t^T R_1)/(1-t) == u_t_R (Cut-Locus-Toleranz)", d < 5e-2,
      f"max|Differenz| = {d:.2e}")

u_state_x = (trans_1 - sf["trans_t"]) / one_m_t[:, None]
dx = (u_state_x - sf["u_t_trans"]).abs().max().item()
check("(x_1 - x_t)/(1-t) == u_t_trans", dx < 1e-4, f"max|Differenz| = {dx:.2e}")

# =========================================================================
sec("5. FRAGMENTZUORDNUNG BEIM POOLING")
pos_t = den._apply_transformations(pos_0=pos_0, batch=b, trans_1=trans_1, R_1=R_1,
                                   R_t=sf["R_t"], trans_t=sf["trans_t"])
b = den._update_batch(batch=b, pos_0=pos_0, pos_t=pos_t)
tf, rf, idxs = den._compute_forces(batch=b, t=t)
flat = den.get_flat_fragment_index(b)
frag_idx = flat.index_select(0, idxs)

check("kein Nicht-Ligandatom im Index", (frag_idx >= 0).all().item())
check("Fragmentindex im gueltigen Bereich", int(frag_idx.max()) < M,
      f"max = {int(frag_idx.max())}, M = {M}")
counts = torch.bincount(frag_idx, minlength=M)
check("jedes Fragment hat mindestens ein Atom", int(counts.min()) >= 1,
      f"min {int(counts.min())}, max {int(counts.max())} Atome je Fragment")
check("Atomzahl stimmt", int(counts.sum()) == idxs.numel(),
      f"{int(counts.sum())} == {idxs.numel()}")

# Pooling gegen eine unabhaengige Referenzimplementierung
v, w = den.pool_fragment_fields(tf, rf, frag_idx, M)
v_ref = torch.stack([tf[frag_idx == k].mean(0) for k in range(M)])
w_ref = torch.stack([rf[frag_idx == k].mean(0) for k in range(M)])
check("v_frag == unabhaengig gerechnetes Mittel",
      (v - v_ref).abs().max().item() < 1e-5, f"max|d| = {(v-v_ref).abs().max().item():.2e}")
check("omega_frag == unabhaengig gerechnetes Mittel",
      (w - w_ref).abs().max().item() < 1e-5, f"max|d| = {(w-w_ref).abs().max().item():.2e}")

# =========================================================================
sec("6. VORHERSAGE IM RICHTIGEN RAHMEN")
vf = den._compute_vector_field(sf, {"v_frag": v, "omega_frag": w}, t_batch)
manual = sf["R_t"].transpose(-1, -2) @ so3_utils.hat(w) @ sf["R_t"]
check("pred_u_t_R == R_t^T hat(omega) R_t",
      (vf["pred_u_t_R"] - manual).abs().max().item() < 1e-6)
check("pred_u_t_trans == v_frag ohne Skalierung",
      (vf["pred_u_t_trans"] - v).abs().max().item() == 0.0)
check("pred_u_t_R schiefsymmetrisch",
      (vf["pred_u_t_R"] + vf["pred_u_t_R"].transpose(-1, -2)).abs().max().item() < 1e-6)

# Gegenprobe. Mit zero_init_last gibt das untrainierte Netz exakt 0 aus,
# dann ist die Konjugation trivial ein No-Op. Deshalb wird sie hier mit einem
# kuenstlich gesetzten, von Null verschiedenen Feld geprueft.
check("Netz gibt bei zero_init_last exakt 0 aus", w.abs().max().item() == 0.0,
      f"max|omega_frag| = {w.abs().max().item():.2e}")
g2 = torch.Generator().manual_seed(5)
w_fake = torch.randn(w.shape, generator=g2)
vf_fake = den._compute_vector_field(sf, {"v_frag": v, "omega_frag": w_fake}, t_batch)
d_conj = (vf_fake["pred_u_t_R"] - so3_utils.hat(w_fake)).abs().max().item()
check("Gegenprobe: Konjugation aendert ein NICHT-Null-Feld wirklich", d_conj > 1e-3,
      f"max|mit - ohne| = {d_conj:.2e}")
check("konjugiertes Nicht-Null-Feld bleibt schiefsymmetrisch",
      (vf_fake["pred_u_t_R"] + vf_fake["pred_u_t_R"].transpose(-1, -2)).abs().max().item() < 1e-5)

# =========================================================================
sec("7. LOSS-AGGREGATION")
out = den(batch)
losses = den.compute_losses(out)
check("loss_trans je Fragment [M']", losses["loss_trans"].dim() == 1)
check("loss_R je Fragment [M']", losses["loss_R"].dim() == 1)
nf = out["num_fragments"]
check("Loss-Laenge == Summe der Fragmente",
      losses["loss_trans"].shape[0] == int(nf.sum()),
      f"{losses['loss_trans'].shape[0]} == {int(nf.sum())}")

scaled = den.scaled_fragmented_loss(losses, nf, a.fragment_scaling)
check("nach Aggregation eine Zahl je Komplex",
      scaled["loss_trans"].shape[0] == int(nf.shape[0]),
      f"{scaled['loss_trans'].shape[0]} == {int(nf.shape[0])}")
manual_agg = torch.stack([
    losses["loss_trans"][int(nf[:i].sum()):int(nf[:i + 1].sum())].sum() / (nf[i] ** a.fragment_scaling)
    for i in range(nf.shape[0])
])
check("Aggregation == unabhaengig gerechnete Summe/F^s",
      (scaled["loss_trans"] - manual_agg).abs().max().item() < 1e-4,
      f"max|d| = {(scaled['loss_trans'] - manual_agg).abs().max().item():.2e}")

print()
print("=" * 74)
ok = sum(1 for _, c, _ in CHECKS if c)
print(f"ERGEBNIS: {ok}/{len(CHECKS)} Checks bestanden")
print("=" * 74)
raise SystemExit(0 if ok == len(CHECKS) else 1)
